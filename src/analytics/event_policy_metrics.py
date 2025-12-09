"""Event-level policy classifier metrics: confusion matrix, accuracy, and recall@k uplift analysis."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
CURATED_DIR = BASE_DIR / "data" / "curated"
POLICY_METRICS_PATH = CURATED_DIR / "event_policy_metrics.parquet"
POLICY_CONFUSION_PATH = CURATED_DIR / "event_policy_confusion.parquet"

POLICY_NAMES = [
    "fifo",
    "lifo",
    "same_period",
    "largest",
    "smallest",
    "eomfirst",
    "regmonth",
    "quarter",
    "seqblock",
    "typeprio",
]
BASELINE_POLICY = "fifo"
CARDINALITY_BUCKETS = {
    "k=1": (1, 1),
    "k=2-3": (2, 3),
    "k>=4": (4, float("inf")),
}

@dataclass
class PolicyMetrics:
    """Container for policy classifier performance metrics."""

    segment_type: str
    segment_value: str
    n_events: int
    top1_accuracy: float
    top2_hit_rate: float
    baseline_recall: float
    predicted_recall: float
    uplift_absolute: float
    uplift_relative: float

def configure_logging(level: int | str | None = logging.INFO) -> Path:
    """Configure shared logging handlers for CLI execution."""
    return setup_logging(level)

def load_policy_data(project_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load per-event policy scores and predictions."""
    curated_dir = project_root / "data" / "curated"
    scores_path = curated_dir / "per_event_policy_scores.parquet"
    predictions_path = curated_dir / "event_policy_predictions.parquet"

    if not scores_path.exists():
        raise FileNotFoundError(f"Missing per_event_policy_scores.parquet: {scores_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing event_policy_predictions.parquet: {predictions_path}")

    scores = pd.read_parquet(scores_path)
    predictions = pd.read_parquet(predictions_path)

    LOGGER.info("Loaded %d policy score events and %d predictions", len(scores), len(predictions))
    return scores, predictions

def build_confusion_matrix(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build confusion matrix with row normalization (true vs predicted policy)."""
    if predictions.empty or "policy_true" not in predictions.columns or "policy_pred" not in predictions.columns:
        LOGGER.warning("Insufficient data for confusion matrix; returning empty.")
        return pd.DataFrame()

    all_policies = sorted(set(predictions["policy_true"].unique()) | set(predictions["policy_pred"].unique()))

    confusion = pd.crosstab(
        predictions["policy_true"], predictions["policy_pred"], rownames=["true"], colnames=["pred"], dropna=False
    )

    for policy in all_policies:
        if policy not in confusion.index:
            confusion.loc[policy] = 0
        if policy not in confusion.columns:
            confusion[policy] = 0

    confusion = confusion.loc[sorted(confusion.index), sorted(confusion.columns)]

    row_sums = confusion.sum(axis=1)
    confusion_norm = confusion.div(row_sums, axis=0).fillna(0.0)

    confusion_norm = confusion_norm.reset_index()
    confusion_norm = confusion_norm.rename(columns={"true": "policy_true"})

    return confusion_norm

def compute_top1_accuracy(predictions: pd.DataFrame) -> float:
    """Compute top-1 accuracy: fraction of events where predicted policy matches true policy."""
    if predictions.empty or "policy_true" not in predictions.columns or "policy_pred" not in predictions.columns:
        return 0.0

    matches = (predictions["policy_true"] == predictions["policy_pred"]).sum()
    return float(matches) / len(predictions)

def compute_top2_hit_rate(predictions: pd.DataFrame) -> float:
    """Compute top-2 hit rate: fraction of events where true policy is in top-2 predicted policies."""
    if predictions.empty:
        return 0.0

    proba_cols = [col for col in predictions.columns if col.startswith("proba_")]
    if not proba_cols:
        LOGGER.warning("No probability columns found; top-2 hit rate defaulting to 0.")
        return 0.0

    hits = 0
    for _, row in predictions.iterrows():
        true_policy = row.get("policy_true")
        if pd.isna(true_policy):
            continue

        proba_dict = {col.replace("proba_", ""): row[col] for col in proba_cols if col in row.index}
        top2_policies = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)[:2]
        top2_names = [p[0] for p in top2_policies]

        if true_policy in top2_names:
            hits += 1

    return float(hits) / len(predictions)

def compute_recall_for_policy(scores: pd.DataFrame, policy_name: str) -> float:
    """Compute average recall@k for a given policy across events."""
    if scores.empty or policy_name not in scores.columns:
        return 0.0

    policy_recalls = scores[policy_name].fillna(0.0)
    return float(policy_recalls.mean())

def compute_uplift(
    scores: pd.DataFrame, predictions: pd.DataFrame, baseline_policy: str = BASELINE_POLICY
) -> Tuple[float, float, float]:
    """Compute recall uplift: predicted policy vs baseline policy."""
    merged = scores.merge(predictions[["event_key", "policy_pred"]], on="event_key", how="inner")

    if merged.empty:
        return 0.0, 0.0, 0.0

    baseline_recall = compute_recall_for_policy(merged, baseline_policy)

    predicted_recalls = []
    for _, row in merged.iterrows():
        pred_policy = row.get("policy_pred")
        if pd.notna(pred_policy) and pred_policy in merged.columns:
            predicted_recalls.append(row[pred_policy])
        else:
            predicted_recalls.append(0.0)

    predicted_recall = float(np.mean(predicted_recalls)) if predicted_recalls else 0.0

    uplift_absolute = predicted_recall - baseline_recall
    uplift_relative = (uplift_absolute / baseline_recall * 100.0) if baseline_recall > 0 else 0.0

    return baseline_recall, predicted_recall, uplift_relative

def segment_by_cardinality(scores: pd.DataFrame, events: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Segment scores by event cardinality buckets."""
    if "cardinality" not in scores.columns and events is not None:
        try:
            event_items_path = CURATED_DIR / "event_items.parquet"
            if event_items_path.exists():
                event_items = pd.read_parquet(event_items_path)
                event_card = event_items.groupby("event_key").size().reset_index(name="cardinality")
                scores = scores.merge(event_card, on="event_key", how="left")
            else:
                LOGGER.warning("event_items.parquet not found; cannot compute cardinality segmentation.")
                return {"all": scores}
        except Exception as exc:
            LOGGER.warning("Failed to load cardinality: %s", exc)
            return {"all": scores}

    if "cardinality" not in scores.columns:
        return {"all": scores}

    segments = {}
    for bucket_name, (min_k, max_k) in CARDINALITY_BUCKETS.items():
        bucket_df = scores[(scores["cardinality"] >= min_k) & (scores["cardinality"] <= max_k)]
        if not bucket_df.empty:
            segments[bucket_name] = bucket_df

    return segments if segments else {"all": scores}

def get_top_volume_clients(scores: pd.DataFrame, top_n: int = 10) -> List[str]:
    """Identify top-N clients by event volume."""
    if "cliente" not in scores.columns:
        return []
    client_counts = scores["cliente"].value_counts().head(top_n)
    return client_counts.index.tolist()

def evaluate_policy_metrics(
    scores: pd.DataFrame, predictions: pd.DataFrame, events: Optional[pd.DataFrame] = None
) -> List[PolicyMetrics]:
    """Compute policy classifier metrics for all segments."""
    metrics_list: List[PolicyMetrics] = []

    baseline_recall, predicted_recall, uplift_relative = compute_uplift(scores, predictions, BASELINE_POLICY)
    top1_acc = compute_top1_accuracy(predictions)
    top2_hit = compute_top2_hit_rate(predictions)

    metrics_list.append(
        PolicyMetrics(
            segment_type="global",
            segment_value="all",
            n_events=len(predictions),
            top1_accuracy=top1_acc,
            top2_hit_rate=top2_hit,
            baseline_recall=baseline_recall,
            predicted_recall=predicted_recall,
            uplift_absolute=predicted_recall - baseline_recall,
            uplift_relative=uplift_relative,
        )
    )

    card_segments = segment_by_cardinality(scores, events)
    for bucket_name, bucket_scores in card_segments.items():
        bucket_preds = predictions[predictions["event_key"].isin(bucket_scores["event_key"])]
        if bucket_preds.empty:
            continue

        baseline_recall, predicted_recall, uplift_relative = compute_uplift(
            bucket_scores, bucket_preds, BASELINE_POLICY
        )
        top1_acc = compute_top1_accuracy(bucket_preds)
        top2_hit = compute_top2_hit_rate(bucket_preds)

        metrics_list.append(
            PolicyMetrics(
                segment_type="cardinality_bucket",
                segment_value=bucket_name,
                n_events=len(bucket_preds),
                top1_accuracy=top1_acc,
                top2_hit_rate=top2_hit,
                baseline_recall=baseline_recall,
                predicted_recall=predicted_recall,
                uplift_absolute=predicted_recall - baseline_recall,
                uplift_relative=uplift_relative,
            )
        )

    top_clients = get_top_volume_clients(scores)
    for cliente in top_clients:
        client_scores = scores[scores["cliente"] == cliente]
        client_preds = predictions[predictions["event_key"].isin(client_scores["event_key"])]

        if client_preds.empty or len(client_preds) < 2:
            continue

        baseline_recall, predicted_recall, uplift_relative = compute_uplift(
            client_scores, client_preds, BASELINE_POLICY
        )
        top1_acc = compute_top1_accuracy(client_preds)
        top2_hit = compute_top2_hit_rate(client_preds)

        metrics_list.append(
            PolicyMetrics(
                segment_type="cliente",
                segment_value=cliente,
                n_events=len(client_preds),
                top1_accuracy=top1_acc,
                top2_hit_rate=top2_hit,
                baseline_recall=baseline_recall,
                predicted_recall=predicted_recall,
                uplift_absolute=predicted_recall - baseline_recall,
                uplift_relative=uplift_relative,
            )
        )

    return metrics_list

def print_summary(metrics_list: List[PolicyMetrics], confusion: pd.DataFrame) -> None:
    """Print console summary of policy classifier metrics."""
    LOGGER.info("=== Policy Classifier Metrics Summary ===")

    global_metrics = [m for m in metrics_list if m.segment_type == "global"]
    for m in global_metrics:
        LOGGER.info(
            "Global (n=%d) → Top-1 Accuracy: %.2f%%, Top-2 Hit: %.2f%%",
            m.n_events,
            m.top1_accuracy * 100,
            m.top2_hit_rate * 100,
        )
        LOGGER.info(
            "  Baseline (%s) recall: %.3f | Predicted recall: %.3f | Uplift: %+.2f%%",
            BASELINE_POLICY.upper(),
            m.baseline_recall,
            m.predicted_recall,
            m.uplift_relative,
        )

    bucket_metrics = [m for m in metrics_list if m.segment_type == "cardinality_bucket"]
    if bucket_metrics:
        LOGGER.info("=== Cardinality Bucket Metrics ===")
        for m in bucket_metrics:
            LOGGER.info(
                "%s (n=%d) → Accuracy: %.2f%%, Uplift: %+.2f%%",
                m.segment_value,
                m.n_events,
                m.top1_accuracy * 100,
                m.uplift_relative,
            )

    client_metrics = [m for m in metrics_list if m.segment_type == "cliente"]
    if client_metrics:
        LOGGER.info("=== Top Client Metrics ===")
        sorted_clients = sorted(client_metrics, key=lambda m: m.uplift_relative, reverse=True)[:5]
        for m in sorted_clients:
            LOGGER.info(
                "Cliente %s (n=%d) → Accuracy: %.2f%%, Uplift: %+.2f%%",
                m.segment_value,
                m.n_events,
                m.top1_accuracy * 100,
                m.uplift_relative,
            )

    if not confusion.empty:
        LOGGER.info("=== Confusion Matrix (normalized by row) ===")
        LOGGER.info("Confusion matrix has %d rows (true policies) × %d columns (predicted policies)", len(confusion), len(confusion.columns) - 1)

def save_outputs(metrics_list: List[PolicyMetrics], confusion: pd.DataFrame, project_root: Path) -> Tuple[Path, Path]:
    """Save policy metrics and confusion matrix to Parquet files."""
    curated_dir = project_root / "data" / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)

    metrics_records = [
        {
            "segment_type": m.segment_type,
            "segment_value": m.segment_value,
            "n_events": m.n_events,
            "top1_accuracy": m.top1_accuracy,
            "top2_hit_rate": m.top2_hit_rate,
            "baseline_recall": m.baseline_recall,
            "predicted_recall": m.predicted_recall,
            "uplift_absolute": m.uplift_absolute,
            "uplift_relative": m.uplift_relative,
        }
        for m in metrics_list
    ]
    metrics_df = pd.DataFrame(metrics_records)

    metrics_path = curated_dir / "event_policy_metrics.parquet"
    confusion_path = curated_dir / "event_policy_confusion.parquet"

    metrics_df.to_parquet(metrics_path, compression="snappy", index=False)
    confusion.to_parquet(confusion_path, compression="snappy", index=False)

    LOGGER.info("Saved policy metrics to %s", metrics_path)
    LOGGER.info("Saved confusion matrix to %s", confusion_path)
    return metrics_path, confusion_path

def main() -> None:
    """Main entry point for policy classifier metrics evaluation."""
    configure_logging()
    project_root = Path(__file__).resolve().parents[2]

    LOGGER.info("Loading policy data...")
    scores, predictions = load_policy_data(project_root)

    events_path = project_root / "data" / "curated" / "events.parquet"
    events = pd.read_parquet(events_path) if events_path.exists() else None

    LOGGER.info("Building confusion matrix...")
    confusion = build_confusion_matrix(predictions)

    LOGGER.info("Computing policy classifier metrics...")
    metrics_list = evaluate_policy_metrics(scores, predictions, events)

    print_summary(metrics_list, confusion)
    save_outputs(metrics_list, confusion, project_root)

    LOGGER.info("Policy metrics evaluation complete.")

if __name__ == "__main__":
    main()

