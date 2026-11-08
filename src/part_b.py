"""Part B: reproducible customer segments and lapse prediction."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PART_B_OUT = OUT / "part_b"
CLEAN = ROOT / "data" / "clean"
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")
RFM_COLUMNS = ["recency", "frequency", "monetary"]
NUMERIC_FEATURES = [
    "recency_days", "frequency", "monetary", "average_discount_pct",
    "cancellation_rate", "average_delivery_minutes", "age", "tenure_days",
]
CATEGORICAL_FEATURES = ["city", "gender", "most_frequent_category"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def as_number(value):
    return value.item() if isinstance(value, np.generic) else value


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if pd.isna(value):
        return None
    return value


def load_inputs():
    customers = pd.read_csv(CLEAN / "customers.csv", parse_dates=["signup_date"])
    orders = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
    products = pd.read_csv(CLEAN / "products.csv")
    required = {
        "customers": {"customer_id", "city", "signup_date", "age", "gender"},
        "orders": {"order_id", "customer_id", "product_id", "order_ts", "status", "discount_pct", "order_amount", "delivery_minutes"},
        "products": {"product_id", "category", "price_inr"},
    }
    for name, frame in (("customers", customers), ("orders", orders), ("products", products)):
        absent = required[name] - set(frame.columns)
        if absent:
            raise ValueError(f"{name}.csv is missing required columns: {sorted(absent)}")
    return customers, orders, products


def build_rfm(orders: pd.DataFrame) -> pd.DataFrame:
    completed = orders.loc[orders["status"].eq("completed")].copy()
    rfm = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.total_seconds().floordiv(86400).astype(int)
    return rfm.reset_index()[["customer_id", *RFM_COLUMNS]]


def segment_names(cluster_summary: pd.DataFrame) -> tuple[dict[int, str], dict[str, str]]:
    """Name clusters by their measured profiles; numeric IDs carry no ranking."""
    premium_id = int(cluster_summary["mean_monetary"].idxmax())
    other_ids = [int(i) for i in cluster_summary.index if int(i) != premium_id]
    risk_id = max(other_ids, key=lambda i: (cluster_summary.loc[i, "mean_recency"], -cluster_summary.loc[i, "mean_frequency"]))
    remaining = [i for i in other_ids if i != risk_id]
    frequent_id = max(remaining, key=lambda i: (cluster_summary.loc[i, "mean_frequency"], -cluster_summary.loc[i, "mean_recency"]))
    names = {premium_id: "Premium", risk_id: "At-Risk Customers", frequent_id: "Frequent Loyalists"}
    for cluster_id in other_ids:
        if cluster_id not in names:
            names[cluster_id] = "Core Customers"
    actions = {
        "Premium": "Offer tailored recognition and early access; protect the high-value relationship.",
        "At-Risk Customers": "Prioritize a timely, relevant win-back offer and monitor for renewed activity.",
        "Frequent Loyalists": "Reinforce the habit with convenience benefits and complementary recommendations.",
        "Core Customers": "Use low-cost personalized nudges to encourage a second or more frequent purchase.",
    }
    return names, actions


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def score_model(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_lapsed": float(precision_score(y_test, pred, pos_label=1, zero_division=0)),
        "recall_lapsed": float(recall_score(y_test, pred, pos_label=1, zero_division=0)),
        "f1_lapsed": float(f1_score(y_test, pred, pos_label=1, zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
    }


def describe_feature(feature: str, coefficient: float) -> str:
    raw = feature.split("__", 1)[-1]
    numeric_meanings = {
        "recency_days": "More days since the last order",
        "frequency": "More pre-cutoff completed orders",
        "monetary": "Higher pre-cutoff spending",
        "average_discount_pct": "Higher average discount use",
        "cancellation_rate": "Higher pre-cutoff cancellation share",
        "average_delivery_minutes": "Longer average completed delivery time",
        "age": "Higher customer age",
        "tenure_days": "Longer customer tenure",
    }
    if raw in numeric_meanings:
        meaning = numeric_meanings[raw]
        comparator = "positive association with" if coefficient > 0 else "negative association with"
        return f"{meaning} has a {comparator} predicted lapse odds (standardized coefficient)."
    for column in CATEGORICAL_FEATURES:
        prefix = f"{column}_"
        if raw.startswith(prefix):
            category = raw[len(prefix):]
            return f"{column}={category} has a {'positive' if coefficient > 0 else 'negative'} association with predicted lapse odds relative to the dropped reference level."
    return "Association with predicted lapse odds; see the encoded feature name and coefficient sign."


def main() -> None:
    PART_B_OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    customers, orders, products = load_inputs()
    orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left", validate="many_to_one")

    # Whole-window RFM segmentation.
    rfm = build_rfm(orders)
    rfm.to_csv(PART_B_OUT / "rfm.csv", index=False)
    rfm_scaler = StandardScaler()
    scaled_rfm = rfm_scaler.fit_transform(rfm[RFM_COLUMNS])
    k_rows = []
    k_models = {}
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(scaled_rfm)
        k_models[k] = (model, labels)
        k_rows.append({"k": k, "inertia": float(model.inertia_), "silhouette": float(silhouette_score(scaled_rfm, labels))})
    k_metrics = pd.DataFrame(k_rows)
    k_metrics.to_csv(PART_B_OUT / "kmeans_metrics.csv", index=False)
    business_usable_k = k_metrics.loc[k_metrics["k"].isin([3, 4, 5])]
    selected_k = int(business_usable_k.loc[business_usable_k["silhouette"].idxmax(), "k"])
    final_kmeans, cluster_ids = k_models[selected_k]
    rfm["segment_id"] = cluster_ids.astype(int)
    cluster_summary = rfm.groupby("segment_id").agg(
        customer_count=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
        median_recency=("recency", "median"),
        median_frequency=("frequency", "median"),
        median_monetary=("monetary", "median"),
    )
    names, actions = segment_names(cluster_summary)
    cluster_summary["segment_name"] = [names[int(i)] for i in cluster_summary.index]
    cluster_summary["suggested_action"] = cluster_summary["segment_name"].map(actions)
    cluster_summary.reset_index().to_csv(PART_B_OUT / "cluster_summary.csv", index=False)
    segments = rfm.copy()
    segments["segment_name"] = segments["segment_id"].map(names)
    premium_id = int(cluster_summary["mean_monetary"].idxmax())
    segments["is_premium"] = segments["segment_id"].eq(premium_id).astype(bool)
    segments[["customer_id", *RFM_COLUMNS, "segment_id", "segment_name", "is_premium"]].to_csv(OUT / "customer_segments.csv", index=False)
    pca = PCA(n_components=2)
    pca.fit(scaled_rfm)
    pca_ratios = pca.explained_variance_ratio_.tolist()
    (PART_B_OUT / "pca_metrics.json").write_text(json.dumps({
        "explained_variance_ratio": pca_ratios,
        "total_explained_variance_ratio": float(sum(pca_ratios)),
    }, indent=2), encoding="utf-8")

    # Cutoff cohort: features use only completed/terminal order history strictly before Sep 1.
    completed_before = orders.loc[(orders["status"].eq("completed")) & (orders["order_ts"] < CUTOFF)].copy()
    cohort_ids = completed_before["customer_id"].drop_duplicates()
    completed_features = completed_before.groupby("customer_id").agg(
        last_completed_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        average_discount_pct=("discount_pct", "mean"),
        average_delivery_minutes=("delivery_minutes", "mean"),
        most_frequent_category=("category", lambda values: values.value_counts().sort_index().idxmax()),
        feature_max_order_ts=("order_ts", "max"),
    )
    completed_features["recency_days"] = (CUTOFF - completed_features["last_completed_ts"]).dt.total_seconds().floordiv(86400).astype(int)

    terminal_before = orders.loc[(orders["order_ts"] < CUTOFF) & orders["status"].isin(["completed", "cancelled"])].copy()
    status_counts = terminal_before.groupby(["customer_id", "status"]).size().unstack(fill_value=0)
    for status in ("completed", "cancelled"):
        if status not in status_counts:
            status_counts[status] = 0
    status_counts["cancellation_rate"] = status_counts["cancelled"] / (status_counts["cancelled"] + status_counts["completed"])

    customer_features = customers.set_index("customer_id")[["city", "age", "gender", "signup_date"]].copy()
    customer_features["tenure_days"] = (CUTOFF - customer_features["signup_date"]).dt.days
    customer_features = customer_features.drop(columns="signup_date")
    feature_table = completed_features.join(status_counts[["cancellation_rate"]], how="left").join(customer_features, how="left")
    feature_table["lapsed"] = 1
    september_completed = orders.loc[
        orders["status"].eq("completed") & orders["order_ts"].ge(CUTOFF) & orders["order_ts"].lt(LABEL_END), "customer_id"
    ].unique()
    feature_table.loc[feature_table.index.isin(september_completed), "lapsed"] = 0
    feature_table = feature_table.reset_index()
    feature_table["age"] = pd.to_numeric(feature_table["age"], errors="coerce")
    feature_table["frequency"] = feature_table["frequency"].astype(int)
    feature_table["monetary"] = feature_table["monetary"].astype(float)
    feature_table["most_frequent_category"] = feature_table["most_frequent_category"].astype(str)
    X = feature_table[FEATURE_COLUMNS].copy()
    y = feature_table["lapsed"].astype(int)
    missing_counts = X.isna().sum()
    if int(missing_counts.sum()) != 0:
        raise ValueError(f"Predictive feature matrix has missing values: {missing_counts[missing_counts.gt(0)].to_dict()}")
    feature_table[["customer_id", *FEATURE_COLUMNS, "lapsed", "feature_max_order_ts"]].to_csv(PART_B_OUT / "lapse_features.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    split = pd.DataFrame({"customer_id": feature_table.loc[X.index, "customer_id"], "split": "train", "lapsed": y})
    split.loc[X_test.index, "split"] = "test"
    split.to_csv(PART_B_OUT / "split_membership.csv", index=False)

    logit_pipeline = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    logit_search = GridSearchCV(
        logit_pipeline, {"classifier__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1", n_jobs=1, refit=True,
    )
    logit_search.fit(X_train, y_train)

    forest_pipeline = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
    ])
    forest_search = GridSearchCV(
        forest_pipeline,
        {"classifier__max_depth": [3, 5, 8], "classifier__min_samples_leaf": [5, 10, 20]},
        cv=5, scoring="f1", n_jobs=1, refit=True,
    )
    forest_search.fit(X_train, y_train)

    model_results = {
        "logistic_regression": {
            "best_params": logit_search.best_params_,
            "best_cv_f1": float(logit_search.best_score_),
            "test": score_model(logit_search.best_estimator_, X_test, y_test),
        },
        "random_forest": {
            "best_params": forest_search.best_params_,
            "best_cv_f1": float(forest_search.best_score_),
            "test": score_model(forest_search.best_estimator_, X_test, y_test),
            "cv_fits": int(len(forest_search.cv_results_["params"]) * forest_search.n_splits_),
        },
    }
    # Choose from training-only cross-validation, never from the held-out test scores.
    chosen_name = "random_forest" if forest_search.best_score_ > logit_search.best_score_ else "logistic_regression"
    chosen_search = forest_search if chosen_name == "random_forest" else logit_search
    chosen_model = chosen_search.best_estimator_
    baseline_prediction = np.zeros(len(y_test), dtype=int)
    baseline = {
        "accuracy": float(accuracy_score(y_test, baseline_prediction)),
        "precision_lapsed": float(precision_score(y_test, baseline_prediction, pos_label=1, zero_division=0)),
        "recall_lapsed": float(recall_score(y_test, baseline_prediction, pos_label=1, zero_division=0)),
        "f1_lapsed": float(f1_score(y_test, baseline_prediction, pos_label=1, zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, baseline_prediction, labels=[0, 1]).tolist(),
    }

    # Oversample only original training-fold rows; preserve every original row.
    lapsed_train = X_train.loc[y_train.eq(1)]
    non_lapsed_count = int(y_train.eq(0).sum())
    lapsed_count = int(y_train.eq(1).sum())
    oversampled = resample(lapsed_train, replace=True, n_samples=non_lapsed_count - lapsed_count, random_state=42)
    X_balanced = pd.concat([X_train, oversampled], axis=0)
    y_balanced = pd.concat([y_train, pd.Series(1, index=oversampled.index, name=y_train.name)], axis=0)
    balanced_logit = clone(logit_search.best_estimator_)
    balanced_logit.fit(X_balanced, y_balanced)
    oversampling_result = {
        "original_training_counts": {str(int(k)): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "balanced_training_counts": {"0": int(y_balanced.eq(0).sum()), "1": int(y_balanced.eq(1).sum())},
        "original_training_rows_retained": len(X_train),
        "added_lapsed_duplicates": len(oversampled),
        "preprocessing_fit_rows": int(balanced_logit.named_steps["preprocessor"].named_transformers_["numeric"].n_samples_seen_),
        "before": model_results["logistic_regression"]["test"],
        "after": score_model(balanced_logit, X_test, y_test),
    }
    original_train_members = pd.DataFrame({
        "customer_id": feature_table.loc[X_train.index, "customer_id"].to_numpy(),
        "lapsed": y_train.to_numpy(),
        "source": "original_training_row",
    })
    duplicate_members = pd.DataFrame({
        "customer_id": feature_table.loc[oversampled.index, "customer_id"].to_numpy(),
        "lapsed": 1,
        "source": "duplicated_lapsed_training_row",
    })
    pd.concat([original_train_members, duplicate_members], ignore_index=True).to_csv(
        PART_B_OUT / "balanced_training_membership.csv", index=False,
    )

    # Feature interpretation from the fitted training pipelines.
    forest_estimator = forest_search.best_estimator_
    transformed_names = forest_estimator.named_steps["preprocessor"].get_feature_names_out()
    importances = forest_estimator.named_steps["classifier"].feature_importances_
    importance_frame = pd.DataFrame({"feature": transformed_names, "importance": importances}).sort_values(
        ["importance", "feature"], ascending=[False, True],
    )
    importance_frame.head(5).to_csv(PART_B_OUT / "top_forest_feature_importance.csv", index=False)
    logit_model = logit_search.best_estimator_
    logit_names = logit_model.named_steps["preprocessor"].get_feature_names_out()
    logit_coefficients = logit_model.named_steps["classifier"].coef_[0]
    coefficient_frame = pd.DataFrame({"feature": logit_names, "coefficient": logit_coefficients})
    coefficient_frame["absolute_coefficient"] = coefficient_frame["coefficient"].abs()
    coefficient_frame = coefficient_frame.sort_values(["absolute_coefficient", "feature"], ascending=[False, True]).head(3).copy()
    coefficient_frame["business_interpretation"] = [
        describe_feature(row.feature, row.coefficient) for row in coefficient_frame.itertuples()
    ]
    coefficient_frame.to_csv(PART_B_OUT / "top_logistic_coefficients.csv", index=False)

    model_path = OUT / "lapse_model.joblib"
    joblib.dump(chosen_model, model_path)
    reloaded_model = joblib.load(model_path)
    test_predictions_before_reload = chosen_model.predict(X_test)
    test_predictions_after_reload = reloaded_model.predict(X_test)
    reload_identical = bool(np.array_equal(test_predictions_before_reload, test_predictions_after_reload))
    pd.DataFrame({
        "customer_id": feature_table.loc[X_test.index, "customer_id"].to_numpy(),
        "prediction_before_reload": test_predictions_before_reload,
        "prediction_after_reload": test_predictions_after_reload,
    }).to_csv(PART_B_OUT / "model_reload_check.csv", index=False)

    # Score Premium segment customers that are in the model's pre-cutoff eligible cohort.
    premium_ids = set(segments.loc[segments["is_premium"], "customer_id"])
    eligible_features = feature_table.loc[feature_table["customer_id"].isin(premium_ids)].copy()
    eligible_X = eligible_features[FEATURE_COLUMNS]
    probabilities = reloaded_model.predict_proba(eligible_X)[:, 1]
    retention = pd.DataFrame({
        "customer_id": eligible_features["customer_id"].to_numpy(),
        "lapse_probability": probabilities,
        "actually_lapsed": eligible_features["lapsed"].astype(bool).to_numpy(),
    }).sort_values(["lapse_probability", "customer_id"], ascending=[False, True]).head(10)
    retention.to_csv(OUT / "retention_list.csv", index=False)

    split_counts = pd.crosstab(split["split"], split["lapsed"]).reindex(index=["train", "test"], columns=[0, 1], fill_value=0)
    preprocessing_fit_rows = int(chosen_model.named_steps["preprocessor"].named_transformers_["numeric"].n_samples_seen_)
    latest_feature_timestamp = completed_before["order_ts"].max().strftime("%Y-%m-%d %H:%M:%S")
    metrics = {
        "snapshot_date": SNAPSHOT.strftime("%Y-%m-%d %H:%M:%S"),
        "rfm": {
            "customers": len(rfm),
            "recency_median": float(rfm["recency"].median()),
            "recency_min": int(rfm["recency"].min()),
            "recency_max": int(rfm["recency"].max()),
            "frequency_mean": float(rfm["frequency"].mean()),
            "frequency_max": int(rfm["frequency"].max()),
            "monetary_mean": float(rfm["monetary"].mean()),
            "monetary_max": float(rfm["monetary"].max()),
        },
        "kmeans": k_rows,
        "selected_k": selected_k,
        "cluster_summary": cluster_summary.reset_index().to_dict(orient="records"),
        "premium_segment_id": premium_id,
        "pca_explained_variance_ratio": pca_ratios,
        "pca_total_explained_variance_ratio": float(sum(pca_ratios)),
        "lapse": {
            "cutoff": CUTOFF.strftime("%Y-%m-%d %H:%M:%S"),
            "label_window_end_exclusive": LABEL_END.strftime("%Y-%m-%d %H:%M:%S"),
            "cohort": len(feature_table),
            "lapsed": int(y.sum()),
            "lapsed_rate": float(y.mean()),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "split_counts": {str(split_name): {str(int(label)): int(count) for label, count in row.items()} for split_name, row in split_counts.iterrows()},
            "feature_columns": FEATURE_COLUMNS,
            "feature_missing_counts": {str(column): int(value) for column, value in missing_counts.items()},
            "max_feature_order_timestamp": latest_feature_timestamp,
            "post_cutoff_features_used": False,
            "preprocessing_fit_rows_chosen_model": preprocessing_fit_rows,
            "train_only_preprocessing": preprocessing_fit_rows == len(X_train),
        },
        "baseline_all_negative": baseline,
        "models": model_results,
        "chosen_model": chosen_name,
        "oversampling": oversampling_result,
        "forest_top_five": importance_frame.head(5).to_dict(orient="records"),
        "logistic_top_three_coefficients": coefficient_frame.to_dict(orient="records"),
        "model_reload_predictions_identical": reload_identical,
        "retention_list": {
            "premium_customers_total": len(premium_ids),
            "premium_customers_in_lapse_cohort": len(eligible_features),
            "premium_customers_without_pre_cutoff_eligibility": len(premium_ids) - len(eligible_features),
            "rows": len(retention),
            "actually_lapsed_top_10": int(retention["actually_lapsed"].sum()),
        },
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "joblib": joblib.__version__,
        },
    }
    (PART_B_OUT / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2), encoding="utf-8")
    print(
        f"Part B complete: {len(rfm):,} RFM customers, K={selected_k}, "
        f"lapse cohort={len(feature_table):,}, chosen={chosen_name}, "
        f"test F1={model_results[chosen_name]['test']['f1_lapsed']:.3f}"
    )


if __name__ == "__main__":
    main()
