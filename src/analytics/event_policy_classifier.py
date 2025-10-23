"""Multiclass classifier for best event-level policy prediction (PROMPT E5)."""
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)

POLICY_NAMES = [
    "fifo", "lifo", "same_period", "largest", "smallest",
    "eomfirst", "regmonth", "quarter", "seqblock", "typeprio"
]

OUTPUT_PATH = Path("data/curated/event_policy_predictions.parquet")


def prepare_event_features(project_root: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Derive event-level features used as classifier predictors."""

    events_path = project_root / "data" / "curated" / "events.parquet"
    events = pd.read_parquet(events_path)
    events["cliente"] = events["cliente"].astype("string")
    events["data_pareggio"] = pd.to_datetime(events["data_pareggio"])

    engineered = events.copy()
    engineered["cliente_code"] = engineered["cliente"].astype("category").cat.codes
    engineered["event_day"] = engineered["data_pareggio"].dt.day
    engineered["event_month"] = engineered["data_pareggio"].dt.month
    engineered["event_quarter"] = engineered["data_pareggio"].dt.quarter
    engineered["is_month_end"] = engineered["data_pareggio"].dt.is_month_end.astype(int)
    engineered["is_quarter_end"] = engineered["data_pareggio"].dt.is_quarter_end.astype(int)

    feature_cols = [
        "cliente_code",
        "n_righe_evento",
        "inter_event_gap_days",
        "importo_evento",
        "event_day",
        "event_month",
        "event_quarter",
        "is_month_end",
        "is_quarter_end",
    ]

    selected = engineered[["event_key", *feature_cols]].copy()
    return selected, feature_cols


def build_training_table(project_root: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Combine per-event policy recalls with engineered features."""

    features, feature_cols = prepare_event_features(project_root)
    scores_path = project_root / "data" / "curated" / "per_event_policy_scores.parquet"
    scores = pd.read_parquet(scores_path)

    recall_cols = [policy for policy in POLICY_NAMES if policy in scores.columns]
    if not recall_cols:
        raise ValueError("Per-event policy score table does not contain expected policy columns.")

    scores = scores[["event_key", *recall_cols]].copy()
    merged = scores.merge(features, on="event_key", how="inner")
    if merged.empty:
        raise ValueError("No overlapping events between per-event scores and features.")

    merged["event_key"] = merged["event_key"].astype("string")
    merged["policy_true"] = merged[recall_cols].idxmax(axis=1)
    merged = merged.dropna(subset=["policy_true"])

    mask = merged[recall_cols].sum(axis=1) > 0
    merged = merged.loc[mask].reset_index(drop=True)
    return merged, recall_cols, feature_cols


def train_classifier(X: pd.DataFrame, y: pd.Series, class_names: List[str]) -> CatBoostClassifier:
    """Train CatBoost multiclass classifier."""
    model = CatBoostClassifier(
        iterations=300,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=0
    )
    model.fit(X, y)
    return model


def main() -> None:
    setup_logging(logging.INFO)
    project_root = Path(__file__).resolve().parents[2]
    try:
        merged, recall_cols, feature_cols = build_training_table(project_root)
    except ValueError as exc:
        LOGGER.error(str(exc))
        return

    if merged.empty:
        LOGGER.error("Training table is empty. Aborting classifier training.")
        return

    X = merged[feature_cols].fillna(0)
    y = merged["policy_true"]
    event_keys = pd.Index(merged["event_key"].astype("string"))
    X.index = event_keys
    y.index = event_keys
    test_size = max(1, int(max(len(X) * 0.25, 1)))
    if len(X) <= test_size:
        LOGGER.warning("Not enough samples for hold-out split; falling back to leave-one-out style evaluation.")
        test_size = 1

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        shuffle=True,
    )

    model = train_classifier(X_train, y_train, POLICY_NAMES)
    train_labels = sorted(str(label) for label in y_train.unique())

    y_pred = pd.Series(model.predict(X_test).flatten(), index=X_test.index, name="policy_pred")
    proba = model.predict_proba(X_test)
    class_labels_raw = getattr(model, "classes_", None)
    if class_labels_raw is None:
        class_labels = train_labels
    else:
        class_labels = [str(label) for label in class_labels_raw]
    proba_df = pd.DataFrame(
        proba,
        columns=[f"proba_{label}" for label in class_labels],
        index=X_test.index,
    )
    for policy in POLICY_NAMES:
        col = f"proba_{policy}"
        if col not in proba_df.columns:
            proba_df[col] = 0.0
    proba_df = proba_df[[f"proba_{p}" for p in POLICY_NAMES]]

    acc = float((y_pred == y_test).mean())
    LOGGER.info("Top-1 accuracy: %.3f", acc)

    merged_indexed = merged.set_index("event_key")
    recall_at_k = []
    for idx, pred_policy in y_pred.items():
        event_key = str(idx)
        event_row = merged_indexed.loc[event_key]
        recall = event_row.get(pred_policy, np.nan)
        if not np.isnan(recall):
            recall_at_k.append(float(recall))
    mean_recall = float(np.mean(recall_at_k)) if recall_at_k else 0.0
    LOGGER.info("Mean recall@k (predicted policy): %.3f", mean_recall)

    out_df = pd.DataFrame(
        {
            "event_key": y_pred.index,
            "policy_pred": y_pred.values,
            "policy_true": y_test.loc[y_pred.index].values,
            "accuracy": acc,
            "mean_recall": mean_recall,
        }
    ).reset_index(drop=True)
    out_df = pd.concat([out_df, proba_df.reset_index(drop=True)], axis=1)

    output_path = project_root / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    LOGGER.info("Saved predictions to %s", output_path)

if __name__ == "__main__":
    main()
