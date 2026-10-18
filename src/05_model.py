"""Fit a Random Forest on hexagon surface temperature and test it two ways.

Random folds mix neighbouring hexagons across train and test, which flatters the score
because a hexagon's neighbours are nearly a copy of it. Spatial folds keep whole H3
resolution 6 blocks together, so the test areas are genuinely unseen. Both are reported
and the gap between them is the finding.

Writes outputs/metrics.json and outputs/predictions.csv.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[1]

# Columns that are identifiers, the target itself, or would leak it. Population is held
# back for the vulnerability index so the model cannot lean on it.
EXCLUDED = {
    "h3_id",
    "h3_parent",
    "centroid_lat",
    "centroid_lon",
    "area_km2",
    "population",
    "pop_density",
}
EXCLUDED_PREFIXES = ("lst_",)

# Coverage columns say how much of a hexagon was observed, not what it is like.
EXCLUDED_SUFFIXES = ("_valid_share",)


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def feature_columns(features: pd.DataFrame, target: str) -> list[str]:
    columns = []
    for column in features.columns:
        if column in EXCLUDED or column == target:
            continue
        if column.startswith(EXCLUDED_PREFIXES) or column.endswith(EXCLUDED_SUFFIXES):
            continue
        if column.startswith("valid_share"):
            continue
        if not pd.api.types.is_numeric_dtype(features[column]):
            continue
        columns.append(column)
    return columns


def make_model(config: dict, seed: int) -> RandomForestRegressor:
    settings = config["model"]
    return RandomForestRegressor(
        n_estimators=settings["n_estimators"],
        min_samples_leaf=settings["min_samples_leaf"],
        max_features=settings["max_features"],
        random_state=seed,
        n_jobs=-1,
    )


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    config: dict,
    scheme: str,
    seed: int,
) -> tuple[list[dict], np.ndarray]:
    """Run one cross-validation scheme and return per-fold scores and predictions."""
    n_splits = config["model"]["n_splits"]
    if scheme == "spatial":
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X, y, groups=groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(X, y)

    folds = []
    predictions = np.full(len(y), np.nan)
    for fold, (train_index, test_index) in enumerate(splits):
        model = make_model(config, seed)
        model.fit(X.iloc[train_index], y.iloc[train_index])
        predicted = model.predict(X.iloc[test_index])
        predictions[test_index] = predicted

        folds.append(
            {
                "fold": fold,
                "n_train": int(len(train_index)),
                "n_test": int(len(test_index)),
                "r2": float(r2_score(y.iloc[test_index], predicted)),
                "rmse": float(np.sqrt(mean_squared_error(y.iloc[test_index], predicted))),
            }
        )
    return folds, predictions


def summarise(folds: list[dict]) -> dict:
    r2 = np.array([f["r2"] for f in folds])
    rmse = np.array([f["rmse"] for f in folds])
    return {
        "r2_mean": float(r2.mean()),
        "r2_sd": float(r2.std(ddof=0)),
        "rmse_mean": float(rmse.mean()),
        "rmse_sd": float(rmse.std(ddof=0)),
    }


def shap_summary(model: RandomForestRegressor, X: pd.DataFrame, outputs: Path) -> dict:
    """Write a SHAP summary plot and return the mean absolute value per feature.

    SHAP says which features the forest leaned on and in which direction. It describes
    the model, not the city: a feature can stand in for something it is correlated with,
    and none of this shows that a feature causes heat.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X, check_additivity=False)

    plt.figure()
    shap.summary_plot(values, X, show=False, plot_size=(9, 6))
    plt.title("What the forest used to predict hexagon temperature")
    plt.tight_layout()
    plot_path = outputs / "shap_summary.png"
    plt.savefig(plot_path, dpi=150)
    plt.close("all")
    print(f"written: {plot_path.relative_to(ROOT)}")

    mean_abs = np.abs(values).mean(axis=0)
    return {
        column: float(value)
        for column, value in sorted(zip(X.columns, mean_abs), key=lambda item: -item[1])
    }


def main() -> None:
    config = load_config()
    target = config["model"]["target"]

    features = pd.read_csv(ROOT / config["paths"]["features"])
    columns = feature_columns(features, target)
    X = features[columns]
    y = features[target]
    groups = features["h3_parent"]

    print(f"rows: {len(features)}, features: {len(columns)}, blocks: {groups.nunique()}")
    print("features used:", ", ".join(columns))

    results = {"target": target, "n_rows": int(len(features)), "features": columns, "schemes": {}}
    predictions = features[["h3_id", "h3_parent", target]].copy()

    for scheme in ("random", "spatial"):
        per_seed = []
        for seed in config["model"]["seeds"]:
            folds, predicted = cross_validate(X, y, groups, config, scheme, seed)
            per_seed.append(summarise(folds))
            if seed == config["model"]["seeds"][0]:
                predictions[f"predicted_{scheme}"] = predicted

        results["schemes"][scheme] = {
            "per_seed": per_seed,
            "r2_mean": float(np.mean([s["r2_mean"] for s in per_seed])),
            "r2_sd_over_seeds": float(np.std([s["r2_mean"] for s in per_seed], ddof=0)),
            "rmse_mean": float(np.mean([s["rmse_mean"] for s in per_seed])),
            "rmse_sd_over_seeds": float(np.std([s["rmse_mean"] for s in per_seed], ddof=0)),
        }
        summary = results["schemes"][scheme]
        print(f"{scheme:8s} R2 {summary['r2_mean']:.3f} (sd {summary['r2_sd_over_seeds']:.3f}), "
              f"RMSE {summary['rmse_mean']:.3f} C (sd {summary['rmse_sd_over_seeds']:.3f})")

    gap = results["schemes"]["random"]["r2_mean"] - results["schemes"]["spatial"]["r2_mean"]
    results["r2_gap_random_minus_spatial"] = float(gap)
    print(f"gap (random - spatial) in R2: {gap:.3f}")

    # A model on all the data, kept for the feature explanation and for stage 6.
    final = make_model(config, config["model"]["seeds"][0]).fit(X, y)
    predictions["predicted_full"] = final.predict(X)
    results["importance"] = {
        column: float(value)
        for column, value in sorted(
            zip(columns, final.feature_importances_), key=lambda item: -item[1]
        )
    }

    outputs = ROOT / config["paths"]["outputs_dir"]
    outputs.mkdir(parents=True, exist_ok=True)

    results["shap_mean_abs"] = shap_summary(final, X, outputs)

    (outputs / "metrics.json").write_text(json.dumps(results, indent=2))
    predictions.to_csv(outputs / "predictions.csv", index=False)

    print("top features:", ", ".join(list(results["importance"])[:5]))
    print(f"written: {(outputs / 'metrics.json').relative_to(ROOT)}, "
          f"{(outputs / 'predictions.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
