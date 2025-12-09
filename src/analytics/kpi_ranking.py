"""Compute extended ranking KPIs (precision@k, coverage@k, MRR, nDCG) for invoice recommendation evaluation."""
from __future__ import annotations

import argparse
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
KPI_PARQUET_PATH = CURATED_DIR / "kpi_ranking.parquet"
KPI_CSV_PATH = CURATED_DIR / "kpi_ranking.csv"

K_VALUES = [1, 2, 3, 5]
CARDINALITY_BUCKETS = {
    "k=1": (1, 1),
    "k=2-3": (2, 3),
    "k>=4": (4, float("inf")),
}

@dataclass
class EventRankingMetrics:
    """Container for per-event ranking KPIs."""

    event_key: str
    cliente: str
    cardinality: int
    precision_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]

def configure_logging(level: int | str | None = logging.INFO) -> Path:
    """Configure shared logging handlers for CLI execution."""
    return setup_logging(level)

def load_test_data(project_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load test split features and events for evaluation."""
    curated_dir = project_root / "data" / "curated"
    features_path = curated_dir / "train_features.parquet"
    events_path = curated_dir / "events.parquet"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing train_features.parquet: {features_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events.parquet: {events_path}")

    features = pd.read_parquet(features_path)
    events = pd.read_parquet(events_path)

    event_dates = events[["event_key", "data_pareggio"]].dropna().copy()
    event_dates["data_pareggio"] = pd.to_datetime(event_dates["data_pareggio"], errors="coerce")
    event_dates = event_dates.dropna(subset=["data_pareggio"])

    features_with_dates = features.merge(event_dates, on="event_key", how="inner")
    features_with_dates = features_with_dates.sort_values("data_pareggio")

    cutoff_index = int(len(features_with_dates) * 0.7)
    cutoff_index = max(cutoff_index, 1)
    test_df = features_with_dates.iloc[cutoff_index:]

    if test_df.empty:
        LOGGER.warning("Test split is empty; using last 30%% of data.")
        test_df = features_with_dates.tail(max(1, int(len(features_with_dates) * 0.3)))

    LOGGER.info("Loaded %d test samples from %d total features", len(test_df), len(features))
    return test_df, events

def load_model_predictions(test_df: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    """Load or generate calibrated model predictions for test set."""
    models_dir = project_root / "models"
    catboost_path = models_dir / "catboost.cbm"

    if catboost_path.exists():
        try:
            from catboost import CatBoostClassifier, Pool

            model = CatBoostClassifier()
            model.load_model(str(catboost_path))

            drop_cols = ["label", "data_pareggio", "cliente", "event_key", "numero_documento"]
            feature_df = test_df.drop(columns=[col for col in drop_cols if col in test_df.columns])

            categorical_cols = [
                col
                for col in ["tipo_documento", "chiave_contabile", "divisa", "periodo_contabile", "macro_tipo"]
                if col in feature_df.columns
            ]
            cat_indices = [feature_df.columns.get_loc(col) for col in categorical_cols if col in feature_df.columns]

            pool = Pool(data=feature_df, cat_features=cat_indices if cat_indices else None)
            probs = model.predict_proba(pool)[:, 1]

            calibrator_path = models_dir / "calibrator.pkl"
            if calibrator_path.exists():
                import pickle

                with open(calibrator_path, "rb") as f:
                    calibrator = pickle.load(f)
                    try:
                        probs = calibrator.predict(probs)
                    except Exception:
                        LOGGER.warning("Calibrator failed; using raw probabilities.")

            result = test_df[["cliente", "event_key", "numero_documento", "label"]].copy()
            result["prob"] = probs
            LOGGER.info("Loaded CatBoost predictions for %d test samples", len(result))
            return result

        except Exception as exc:
            LOGGER.warning("Failed to load CatBoost model: %s; falling back to heuristic", exc)

    result = test_df[["cliente", "event_key", "numero_documento", "label"]].copy()
    if "age_days" in test_df.columns:
        result["prob"] = test_df["age_days"].fillna(0).clip(0, 365) / 365.0
    else:
        result["prob"] = np.random.rand(len(result))
    LOGGER.info("Using fallback heuristic predictions for %d test samples", len(result))
    return result

def compute_precision_at_k(ranked_labels: np.ndarray, k: int) -> float:
    """Compute precision@k: fraction of top-k items that are relevant."""
    if k <= 0 or len(ranked_labels) == 0:
        return 0.0
    top_k = ranked_labels[:k]
    return float(np.sum(top_k)) / k

def compute_hit_rate_at_k(ranked_labels: np.ndarray, k: int, n_relevant: int) -> float:
    """Compute hit rate (recall)@k: fraction of relevant items found in top-k."""
    if n_relevant <= 0 or k <= 0 or len(ranked_labels) == 0:
        return 0.0
    top_k = ranked_labels[:k]
    return float(np.sum(top_k)) / n_relevant

def compute_mrr(ranked_labels: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank: 1 / rank of first relevant item."""
    if len(ranked_labels) == 0:
        return 0.0
    relevant_indices = np.where(ranked_labels == 1)[0]
    if len(relevant_indices) == 0:
        return 0.0
    first_relevant_rank = relevant_indices[0] + 1
    return 1.0 / first_relevant_rank

def compute_ndcg_at_k(ranked_labels: np.ndarray, k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain@k."""
    if k <= 0 or len(ranked_labels) == 0:
        return 0.0

    top_k = ranked_labels[:k]
    if len(top_k) == 0:
        return 0.0

    dcg = 0.0
    for i, rel in enumerate(top_k):
        dcg += float(rel) / np.log2(i + 2)

    n_relevant = int(np.sum(ranked_labels))
    ideal_labels = np.sort(ranked_labels)[::-1][:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_labels):
        idcg += float(rel) / np.log2(i + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def evaluate_event_ranking(event_group: pd.DataFrame, k_values: List[int]) -> EventRankingMetrics:
    """Compute ranking KPIs for a single clearing event."""
    event_key = str(event_group.iloc[0]["event_key"])
    cliente = str(event_group.iloc[0]["cliente"])

    sorted_group = event_group.sort_values("prob", ascending=False).reset_index(drop=True)
    ranked_labels = sorted_group["label"].to_numpy(dtype=int)
    n_relevant = int(np.sum(ranked_labels))

    precision_at_k = {}
    hit_rate_at_k = {}
    ndcg_at_k = {}

    for k in k_values:
        precision_at_k[k] = compute_precision_at_k(ranked_labels, k)
        hit_rate_at_k[k] = compute_hit_rate_at_k(ranked_labels, k, n_relevant)
        ndcg_at_k[k] = compute_ndcg_at_k(ranked_labels, k)

    mrr = compute_mrr(ranked_labels)

    return EventRankingMetrics(
        event_key=event_key,
        cliente=cliente,
        cardinality=n_relevant,
        precision_at_k=precision_at_k,
        hit_rate_at_k=hit_rate_at_k,
        mrr=mrr,
        ndcg_at_k=ndcg_at_k,
    )

def compute_coverage_at_k(event_metrics: List[EventRankingMetrics], k: int) -> float:
    """Compute coverage@k: fraction of events with at least one hit in top-k."""
    if not event_metrics:
        return 0.0
    events_with_hit = sum(1 for m in event_metrics if m.hit_rate_at_k.get(k, 0) > 0)
    return events_with_hit / len(event_metrics)

def aggregate_kpis(
    event_metrics: List[EventRankingMetrics], aggregation_key: str, aggregation_value: str
) -> Dict[str, object]:
    """Aggregate event-level KPIs by client, bucket, or globally."""
    if not event_metrics:
        return {
            "aggregation_key": aggregation_key,
            "aggregation_value": aggregation_value,
            "n_events": 0,
        }

    n_events = len(event_metrics)
    result: Dict[str, object] = {
        "aggregation_key": aggregation_key,
        "aggregation_value": aggregation_value,
        "n_events": n_events,
    }

    for k in K_VALUES:
        precisions = [m.precision_at_k.get(k, 0.0) for m in event_metrics]
        hit_rates = [m.hit_rate_at_k.get(k, 0.0) for m in event_metrics]
        ndcgs = [m.ndcg_at_k.get(k, 0.0) for m in event_metrics]

        result[f"precision@{k}"] = float(np.mean(precisions)) if precisions else 0.0
        result[f"hit_rate@{k}"] = float(np.mean(hit_rates)) if hit_rates else 0.0
        result[f"ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
        result[f"coverage@{k}"] = compute_coverage_at_k(event_metrics, k)

    mrrs = [m.mrr for m in event_metrics]
    result["mrr"] = float(np.mean(mrrs)) if mrrs else 0.0

    return result

def build_kpi_table(event_metrics: List[EventRankingMetrics]) -> pd.DataFrame:
    """Build comprehensive KPI table with per-client, global, and cardinality-bucket aggregations."""
    rows: List[Dict[str, object]] = []

    rows.append(aggregate_kpis(event_metrics, "global", "all"))

    client_groups: Dict[str, List[EventRankingMetrics]] = {}
    for metric in event_metrics:
        client_groups.setdefault(metric.cliente, []).append(metric)

    for cliente, client_metrics in client_groups.items():
        rows.append(aggregate_kpis(client_metrics, "cliente", cliente))

    for bucket_name, (min_k, max_k) in CARDINALITY_BUCKETS.items():
        bucket_metrics = [m for m in event_metrics if min_k <= m.cardinality <= max_k]
        if bucket_metrics:
            rows.append(aggregate_kpis(bucket_metrics, "cardinality_bucket", bucket_name))

    return pd.DataFrame(rows)

def print_summary_table(kpi_df: pd.DataFrame) -> None:
    """Print console summary table with key KPIs."""
    global_row = kpi_df[kpi_df["aggregation_key"] == "global"]
    if not global_row.empty:
        LOGGER.info("=== Global Ranking KPIs ===")
        for k in K_VALUES:
            LOGGER.info(
                "k=%d → precision: %.3f, hit_rate: %.3f, coverage: %.3f, nDCG: %.3f",
                k,
                global_row.iloc[0].get(f"precision@{k}", 0.0),
                global_row.iloc[0].get(f"hit_rate@{k}", 0.0),
                global_row.iloc[0].get(f"coverage@{k}", 0.0),
                global_row.iloc[0].get(f"ndcg@{k}", 0.0),
            )
        LOGGER.info("MRR: %.3f", global_row.iloc[0].get("mrr", 0.0))

    bucket_rows = kpi_df[kpi_df["aggregation_key"] == "cardinality_bucket"]
    if not bucket_rows.empty:
        LOGGER.info("=== Cardinality Bucket KPIs ===")
        for _, row in bucket_rows.iterrows():
            bucket = row["aggregation_value"]
            n_events = int(row["n_events"])
            LOGGER.info(
                "%s (n=%d) → precision@3: %.3f, hit_rate@3: %.3f, MRR: %.3f",
                bucket,
                n_events,
                row.get("precision@3", 0.0),
                row.get("hit_rate@3", 0.0),
                row.get("mrr", 0.0),
            )

    client_rows = kpi_df[kpi_df["aggregation_key"] == "cliente"].copy()
    if not client_rows.empty:
        client_rows = client_rows.sort_values("hit_rate@3", ascending=False).head(5)
        LOGGER.info("=== Top 5 Clients by hit_rate@3 ===")
        for _, row in client_rows.iterrows():
            LOGGER.info(
                "Cliente %s (n=%d) → hit_rate@3: %.3f, MRR: %.3f",
                row["aggregation_value"],
                int(row["n_events"]),
                row.get("hit_rate@3", 0.0),
                row.get("mrr", 0.0),
            )

def save_kpi_table(kpi_df: pd.DataFrame, project_root: Path) -> Tuple[Path, Path]:
    """Save KPI table to both Parquet and CSV formats."""
    curated_dir = project_root / "data" / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = curated_dir / "kpi_ranking.parquet"
    csv_path = curated_dir / "kpi_ranking.csv"

    kpi_df.to_parquet(parquet_path, compression="snappy", index=False)
    kpi_df.to_csv(csv_path, index=False, float_format="%.4f")

    LOGGER.info("Saved ranking KPIs to %s and %s", parquet_path, csv_path)
    return parquet_path, csv_path

def main() -> None:
    """Main entry point for KPI ranking computation."""
    configure_logging()
    project_root = Path(__file__).resolve().parents[2]

    LOGGER.info("Loading test data and model predictions...")
    test_df, events = load_test_data(project_root)
    predictions = load_model_predictions(test_df, project_root)

    LOGGER.info("Computing per-event ranking metrics...")
    event_metrics: List[EventRankingMetrics] = []
    for event_key, group in predictions.groupby("event_key"):
        if group["label"].sum() == 0:
            continue
        metrics = evaluate_event_ranking(group, K_VALUES)
        event_metrics.append(metrics)

    LOGGER.info("Evaluated %d events with ranking KPIs", len(event_metrics))

    LOGGER.info("Aggregating KPIs by client, cardinality bucket, and globally...")
    kpi_df = build_kpi_table(event_metrics)

    print_summary_table(kpi_df)
    save_kpi_table(kpi_df, project_root)

    LOGGER.info("KPI ranking computation complete.")

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compute extended ranking KPIs for invoice recommendation.")
    parser.add_argument("--k-values", nargs="+", type=int, help="List of k values for ranking metrics.")
    return parser.parse_args()

if __name__ == "__main__":
    main()

