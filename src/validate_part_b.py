"""Validate Part B artifacts, reference values, leakage controls, and model reload."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PART_B = OUT / "part_b"


def close(actual: float, expected: float, tolerance: float) -> bool:
    return abs(float(actual) - expected) <= tolerance


def main() -> None:
    metrics = json.loads((PART_B / "metrics.json").read_text(encoding="utf-8"))
    rfm = metrics["rfm"]
    lapse = metrics["lapse"]
    checks: list[bool] = []

    required_files = [
        OUT / "customer_segments.csv", OUT / "lapse_model.joblib", OUT / "retention_list.csv",
        PART_B / "metrics.json", PART_B / "kmeans_metrics.csv", PART_B / "cluster_summary.csv",
        PART_B / "lapse_features.csv", PART_B / "split_membership.csv",
        PART_B / "balanced_training_membership.csv", PART_B / "model_reload_check.csv",
        PART_B / "top_forest_feature_importance.csv", PART_B / "top_logistic_coefficients.csv",
    ]
    checks.append(all(path.is_file() and path.stat().st_size > 0 for path in required_files))

    segments = pd.read_csv(OUT / "customer_segments.csv")
    expected_segment_columns = ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
    checks.append(segments.columns.tolist() == expected_segment_columns)
    checks.append(len(segments) == 1175 and segments["customer_id"].is_unique)
    checks.append(rfm["customers"] == 1175)
    checks.append(rfm["recency_median"] == 19 and rfm["recency_min"] == 0 and rfm["recency_max"] == 180)
    checks.append(close(rfm["frequency_mean"], 9.46, 0.01) and rfm["frequency_max"] == 33)
    checks.append(close(rfm["monetary_mean"], 2552.06, 0.01) and close(rfm["monetary_max"], 14766.55, 0.01))
    checks.append(metrics["selected_k"] in {3, 4, 5})
    kmeans_metrics = pd.read_csv(PART_B / "kmeans_metrics.csv").set_index("k")
    expected_k = {
        2: (1820.8, 0.465), 3: (1268.2, 0.445), 4: (743.1, 0.526),
        5: (617.4, 0.481), 6: (503.2, 0.482), 7: (408.5, 0.416), 8: (366.5, 0.412),
    }
    checks.append(list(kmeans_metrics.index) == list(range(2, 9)))
    checks.append(all(close(kmeans_metrics.loc[k, "inertia"], vals[0], 0.5) and close(kmeans_metrics.loc[k, "silhouette"], vals[1], 0.005) for k, vals in expected_k.items()))
    cluster_summary = pd.read_csv(PART_B / "cluster_summary.csv")
    premium_summary = cluster_summary.loc[cluster_summary["segment_name"].eq("Premium")]
    checks.append(len(premium_summary) == 1)
    checks.append(bool(premium_summary["mean_monetary"].iloc[0] == cluster_summary["mean_monetary"].max()))
    checks.append(int(cluster_summary["customer_count"].sum()) == 1175)
    pca = json.loads((PART_B / "pca_metrics.json").read_text(encoding="utf-8"))
    checks.append(close(pca["explained_variance_ratio"][0], 0.634, 0.005))
    checks.append(close(pca["explained_variance_ratio"][1], 0.218, 0.005))
    checks.append(close(pca["total_explained_variance_ratio"], 0.852, 0.005))

    checks.append(lapse["cohort"] == 1156 and lapse["lapsed"] == 443)
    checks.append(close(lapse["lapsed_rate"], 0.3832, 0.001))
    checks.append(lapse["train_rows"] == 924 and lapse["test_rows"] == 232)
    split_membership = pd.read_csv(PART_B / "split_membership.csv")
    checks.append(split_membership["split"].value_counts().to_dict() == {"train": 924, "test": 232})
    features = pd.read_csv(PART_B / "lapse_features.csv", parse_dates=["feature_max_order_ts"])
    checks.append(not features[lapse["feature_columns"]].isna().any().any())
    checks.append(all(value == 0 for value in lapse["feature_missing_counts"].values()))
    cutoff = pd.Timestamp(lapse["cutoff"])
    checks.append(features["feature_max_order_ts"].lt(cutoff).all())
    checks.append(pd.Timestamp(lapse["max_feature_order_timestamp"]) < cutoff)
    checks.append(lapse["post_cutoff_features_used"] is False)
    checks.append(not ({"lapsed", "segment_id", "is_premium", "segment_name"} & set(lapse["feature_columns"])))
    checks.append(lapse["preprocessing_fit_rows_chosen_model"] == lapse["train_rows"])
    checks.append(lapse["train_only_preprocessing"] is True)

    baseline = metrics["baseline_all_negative"]
    checks.append(close(baseline["accuracy"], 0.6164, 0.002) and baseline["f1_lapsed"] == 0)
    checks.append(metrics["models"]["random_forest"]["cv_fits"] == 45)
    checks.append(metrics["models"]["random_forest"]["best_params"] == {
        "classifier__max_depth": 8, "classifier__min_samples_leaf": 10,
    })
    checks.append(metrics["models"]["logistic_regression"]["best_params"]["classifier__C"] in [0.01, 0.1, 1, 10, 100])
    checks.append(metrics["models"][metrics["chosen_model"]]["test"]["f1_lapsed"] >= 0.55)

    oversampling = metrics["oversampling"]
    checks.append(oversampling["original_training_rows_retained"] == 924)
    checks.append(oversampling["balanced_training_counts"] == {"0": 570, "1": 570})
    checks.append(oversampling["added_lapsed_duplicates"] == 216)
    checks.append(oversampling["preprocessing_fit_rows"] == 1140)
    balanced_membership = pd.read_csv(PART_B / "balanced_training_membership.csv")
    checks.append((balanced_membership["source"] == "original_training_row").sum() == 924)
    checks.append((balanced_membership["source"] == "duplicated_lapsed_training_row").sum() == 216)
    duplicate_rows = balanced_membership["source"].eq("duplicated_lapsed_training_row")
    checks.append(balanced_membership.loc[duplicate_rows, "lapsed"].eq(1).all())

    model = joblib.load(OUT / "lapse_model.joblib")
    reload_check = pd.read_csv(PART_B / "model_reload_check.csv")
    test_ids = reload_check["customer_id"]
    expected_test_ids = set(split_membership.loc[split_membership["split"].eq("test"), "customer_id"])
    checks.append(set(test_ids) == expected_test_ids)
    test_features = features.set_index("customer_id").loc[test_ids, lapse["feature_columns"]]
    scaler_rows = int(model.named_steps["preprocessor"].named_transformers_["numeric"].n_samples_seen_)
    checks.append(scaler_rows == 924)
    checks.append(len(reload_check) == 232 and (reload_check["prediction_before_reload"] == reload_check["prediction_after_reload"]).all())
    checks.append(np.array_equal(model.predict(test_features), reload_check["prediction_after_reload"].to_numpy()))
    checks.append(metrics["model_reload_predictions_identical"] is True)

    retention = pd.read_csv(OUT / "retention_list.csv")
    checks.append(retention.columns.tolist() == ["customer_id", "lapse_probability", "actually_lapsed"])
    checks.append(len(retention) <= 10 and retention["customer_id"].is_unique)
    premium_ids = set(segments.loc[segments["is_premium"], "customer_id"])
    checks.append(set(retention["customer_id"]).issubset(premium_ids))
    checks.append(retention["lapse_probability"].is_monotonic_decreasing)
    checks.append(int(retention["actually_lapsed"].sum()) == metrics["retention_list"]["actually_lapsed_top_10"])

    if not all(checks):
        failed = [i + 1 for i, passed in enumerate(checks) if not passed]
        raise AssertionError(f"Part B validation failed at checkpoint(s): {failed} of {len(checks)}")
    print("ALL PART B CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
