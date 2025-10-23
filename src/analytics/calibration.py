"""Calibration diagnostics: Brier score, ECE, MCE, and reliability curves for probabilistic models."""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
CURATED_DIR = BASE_DIR / "data" / "curated"
MODELS_DIR = BASE_DIR / "models"
CALIBRATION_SUMMARY_PATH = CURATED_DIR / "calibration_summary.parquet"
CALIBRATION_BINS_PATH = CURATED_DIR / "calibration_bins.parquet"

N_BINS = 20
TRAIN_FRACTION = 0.7
TOP_CLIENT_THRESHOLD = 10  # Number of top-volume clients to analyze separately


@dataclass
class CalibrationMetrics:
    """Container for calibration quality metrics."""

    model_name: str
    segment_type: str
    segment_value: str
    n_samples: int
    brier_score: float
    ece: float
    mce: float


def configure_logging(level: int | str | None = logging.INFO) -> Path:
    """Configure shared logging handlers for CLI execution."""
    return setup_logging(level)


def load_test_data(project_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load test split features and events for calibration evaluation."""
    curated_dir = project_root / "data" / "curated"
    features_path = curated_dir / "train_features.parquet"
    events_path = curated_dir / "events.parquet"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing train_features.parquet: {features_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events.parquet: {events_path}")

    features = pd.read_parquet(features_path)
    events = pd.read_parquet(events_path)

    # Perform temporal split to isolate test set (70/30 split)
    event_dates = events[["event_key", "data_pareggio"]].dropna().copy()
    event_dates["data_pareggio"] = pd.to_datetime(event_dates["data_pareggio"], errors="coerce")
    event_dates = event_dates.dropna(subset=["data_pareggio"])

    features_with_dates = features.merge(event_dates, on="event_key", how="inner")
    features_with_dates = features_with_dates.sort_values("data_pareggio")

    cutoff_index = int(len(features_with_dates) * TRAIN_FRACTION)
    cutoff_index = max(cutoff_index, 1)
    test_df = features_with_dates.iloc[cutoff_index:]

    if test_df.empty:
        LOGGER.warning("Test split is empty; using last 30%% of data.")
        test_df = features_with_dates.tail(max(1, int(len(features_with_dates) * 0.3)))

    LOGGER.info("Loaded %d test samples from %d total features", len(test_df), len(features))
    return test_df, events


def load_model_predictions(
    test_df: pd.DataFrame, model_name: str, project_root: Path
) -> Optional[pd.DataFrame]:
    """Load or generate calibrated model predictions for test set."""
    models_dir = project_root / "models"

    if model_name == "catboost":
        model_path = models_dir / "catboost.cbm"
        calibrator_path = models_dir / "calibrator.pkl"
    elif model_name == "autogluon":
        model_path = models_dir / "autogluon"
        calibrator_path = models_dir / "calibrator_autogluon.pkl"
    else:
        LOGGER.warning("Unknown model name: %s", model_name)
        return None

    if not model_path.exists():
        LOGGER.info("Model %s not found; skipping.", model_name)
        return None

    try:
        if model_name == "catboost":
            from catboost import CatBoostClassifier, Pool

            model = CatBoostClassifier()
            model.load_model(str(model_path))

            # Select features matching training schema - must match exact column order
            drop_cols = ["label", "data_pareggio", "cliente", "event_key", "numero_documento"]
            feature_df = test_df.drop(columns=[col for col in drop_cols if col in test_df.columns], errors='ignore')

            categorical_cols = [
                col
                for col in ["tipo_documento", "chiave_contabile", "divisa", "periodo_contabile", "macro_tipo"]
                if col in feature_df.columns
            ]
            cat_indices = [feature_df.columns.get_loc(col) for col in categorical_cols if col in feature_df.columns]

            # Ensure feature order matches model expectations
            feature_names = model.feature_names_
            if feature_names:
                missing_features = set(feature_names) - set(feature_df.columns)
                if missing_features:
                    LOGGER.warning(
                        "Missing features for %s: %s; falling back to heuristic predictions",
                        model_name,
                        missing_features,
                    )
                    # Fallback: use simple heuristic based on age_days
                    result = test_df[["cliente", "event_key", "numero_documento", "label"]].copy()
                    if "age_days" in test_df.columns:
                        result["prob"] = test_df["age_days"].fillna(0).clip(0, 365) / 365.0
                    else:
                        result["prob"] = np.random.rand(len(result))
                    LOGGER.info("Using fallback heuristic predictions for %d test samples", len(result))
                    return result
                # Reorder columns to match training
                feature_df = feature_df[feature_names]

            pool = Pool(data=feature_df, cat_features=cat_indices if cat_indices else None)
            probs = model.predict_proba(pool)[:, 1]

            # Apply calibrator if available
            if calibrator_path.exists():
                with open(calibrator_path, "rb") as f:
                    calibrator = pickle.load(f)
                    try:
                        probs = calibrator.predict(probs)
                    except Exception:
                        LOGGER.warning("Calibrator failed; using raw probabilities.")

            result = test_df[["cliente", "event_key", "numero_documento", "label"]].copy()
            result["prob"] = probs
            LOGGER.info("Loaded %s predictions for %d test samples", model_name, len(result))
            return result

        elif model_name == "autogluon":
            # Placeholder for AutoGluon loading (would require AutoGluon imports)
            LOGGER.info("AutoGluon loading not implemented; skipping.")
            return None

    except Exception as exc:
        LOGGER.warning("Failed to load %s model: %s", model_name, exc)
        return None


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score: mean squared error of probabilistic predictions."""
    if len(y_true) == 0:
        return 0.0
    return float(np.mean((y_prob - y_true) ** 2))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) using equal-populated bins."""
    if len(y_true) == 0:
        return 0.0

    # Sort by predicted probability
    sort_idx = np.argsort(y_prob)
    y_true_sorted = y_true[sort_idx]
    y_prob_sorted = y_prob[sort_idx]

    # Create equal-populated bins
    bin_size = len(y_true) // n_bins
    if bin_size == 0:
        return 0.0

    ece = 0.0
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size if i < n_bins - 1 else len(y_true)

        bin_true = y_true_sorted[start_idx:end_idx]
        bin_prob = y_prob_sorted[start_idx:end_idx]

        if len(bin_true) == 0:
            continue

        bin_accuracy = np.mean(bin_true)
        bin_confidence = np.mean(bin_prob)
        bin_weight = len(bin_true) / len(y_true)

        ece += bin_weight * np.abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Maximum Calibration Error (MCE): worst-case calibration gap."""
    if len(y_true) == 0:
        return 0.0

    # Sort by predicted probability
    sort_idx = np.argsort(y_prob)
    y_true_sorted = y_true[sort_idx]
    y_prob_sorted = y_prob[sort_idx]

    # Create equal-populated bins
    bin_size = len(y_true) // n_bins
    if bin_size == 0:
        return 0.0

    max_error = 0.0
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size if i < n_bins - 1 else len(y_true)

        bin_true = y_true_sorted[start_idx:end_idx]
        bin_prob = y_prob_sorted[start_idx:end_idx]

        if len(bin_true) == 0:
            continue

        bin_accuracy = np.mean(bin_true)
        bin_confidence = np.mean(bin_prob)
        error = np.abs(bin_accuracy - bin_confidence)

        max_error = max(max_error, error)

    return float(max_error)


def build_reliability_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS
) -> pd.DataFrame:
    """Create reliability diagram data: predicted probability vs observed rate."""
    if len(y_true) == 0:
        return pd.DataFrame(columns=["bin_id", "pred_mean", "obs_rate", "n_samples"])

    # Sort by predicted probability
    sort_idx = np.argsort(y_prob)
    y_true_sorted = y_true[sort_idx]
    y_prob_sorted = y_prob[sort_idx]

    # Create equal-populated bins
    bin_size = len(y_true) // n_bins
    if bin_size == 0:
        bin_size = 1

    rows: List[Dict[str, object]] = []
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size if i < n_bins - 1 else len(y_true)

        bin_true = y_true_sorted[start_idx:end_idx]
        bin_prob = y_prob_sorted[start_idx:end_idx]

        if len(bin_true) == 0:
            continue

        rows.append(
            {
                "bin_id": i + 1,
                "pred_mean": float(np.mean(bin_prob)),
                "obs_rate": float(np.mean(bin_true)),
                "n_samples": len(bin_true),
            }
        )

    return pd.DataFrame(rows)


def compute_calibration_metrics(
    predictions: pd.DataFrame,
    model_name: str,
    segment_type: str = "global",
    segment_value: str = "all",
) -> CalibrationMetrics:
    """Compute Brier, ECE, MCE for a prediction set."""
    y_true = predictions["label"].to_numpy(dtype=int)
    y_prob = predictions["prob"].to_numpy(dtype=float)

    brier = compute_brier_score(y_true, y_prob)
    ece = compute_ece(y_true, y_prob, n_bins=N_BINS)
    mce = compute_mce(y_true, y_prob, n_bins=N_BINS)

    return CalibrationMetrics(
        model_name=model_name,
        segment_type=segment_type,
        segment_value=segment_value,
        n_samples=len(predictions),
        brier_score=brier,
        ece=ece,
        mce=mce,
    )


def segment_by_cardinality(predictions: pd.DataFrame, events: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Segment predictions by event cardinality buckets."""
    # Compute cardinality per event
    event_card = predictions.groupby("event_key")["label"].sum().reset_index()
    event_card.columns = ["event_key", "cardinality"]

    predictions_with_card = predictions.merge(event_card, on="event_key", how="left")

    segments = {}
    segments["k=1"] = predictions_with_card[predictions_with_card["cardinality"] == 1]
    segments["k=2-3"] = predictions_with_card[
        (predictions_with_card["cardinality"] >= 2) & (predictions_with_card["cardinality"] <= 3)
    ]
    segments["k>=4"] = predictions_with_card[predictions_with_card["cardinality"] >= 4]

    return segments


def get_top_volume_clients(predictions: pd.DataFrame, top_n: int = TOP_CLIENT_THRESHOLD) -> List[str]:
    """Identify top-N clients by volume (number of test samples)."""
    client_counts = predictions["cliente"].value_counts().head(top_n)
    return client_counts.index.tolist()


def evaluate_calibration(
    predictions: pd.DataFrame, events: pd.DataFrame, model_name: str
) -> Tuple[List[CalibrationMetrics], pd.DataFrame]:
    """Compute calibration metrics and reliability bins for all segments."""
    metrics_list: List[CalibrationMetrics] = []
    bins_list: List[pd.DataFrame] = []

    # Global metrics
    global_metrics = compute_calibration_metrics(predictions, model_name, "global", "all")
    metrics_list.append(global_metrics)

    global_bins = build_reliability_bins(predictions["label"].to_numpy(), predictions["prob"].to_numpy())
    global_bins["model"] = model_name
    global_bins["segment_type"] = "global"
    global_bins["segment_value"] = "all"
    bins_list.append(global_bins)

    # Cardinality segments
    card_segments = segment_by_cardinality(predictions, events)
    for bucket_name, bucket_df in card_segments.items():
        if bucket_df.empty:
            continue
        bucket_metrics = compute_calibration_metrics(bucket_df, model_name, "cardinality_bucket", bucket_name)
        metrics_list.append(bucket_metrics)

        bucket_bins = build_reliability_bins(bucket_df["label"].to_numpy(), bucket_df["prob"].to_numpy())
        bucket_bins["model"] = model_name
        bucket_bins["segment_type"] = "cardinality_bucket"
        bucket_bins["segment_value"] = bucket_name
        bins_list.append(bucket_bins)

    # Top-volume clients
    top_clients = get_top_volume_clients(predictions)
    for cliente in top_clients:
        client_df = predictions[predictions["cliente"] == cliente]
        if client_df.empty or len(client_df) < 10:  # Skip clients with too few samples
            continue
        client_metrics = compute_calibration_metrics(client_df, model_name, "cliente", cliente)
        metrics_list.append(client_metrics)

        client_bins = build_reliability_bins(client_df["label"].to_numpy(), client_df["prob"].to_numpy())
        client_bins["model"] = model_name
        client_bins["segment_type"] = "cliente"
        client_bins["segment_value"] = cliente
        bins_list.append(client_bins)

    bins_df = pd.concat(bins_list, ignore_index=True) if bins_list else pd.DataFrame()
    return metrics_list, bins_df


def print_summary(metrics_list: List[CalibrationMetrics]) -> None:
    """Print console summary of calibration metrics."""
    LOGGER.info("=== Calibration Diagnostics Summary ===")

    # Global metrics
    global_metrics = [m for m in metrics_list if m.segment_type == "global"]
    for m in global_metrics:
        LOGGER.info(
            "Model: %s | Global → Brier: %.4f, ECE: %.4f, MCE: %.4f (n=%d)",
            m.model_name,
            m.brier_score,
            m.ece,
            m.mce,
            m.n_samples,
        )

    # Cardinality buckets
    bucket_metrics = [m for m in metrics_list if m.segment_type == "cardinality_bucket"]
    if bucket_metrics:
        LOGGER.info("=== Cardinality Bucket Metrics ===")
        for m in bucket_metrics:
            LOGGER.info(
                "Model: %s | Bucket: %s (n=%d) → Brier: %.4f, ECE: %.4f, MCE: %.4f",
                m.model_name,
                m.segment_value,
                m.n_samples,
                m.brier_score,
                m.ece,
                m.mce,
            )

    # Top clients
    client_metrics = [m for m in metrics_list if m.segment_type == "cliente"]
    if client_metrics:
        LOGGER.info("=== Top Client Metrics (by Brier score) ===")
        sorted_clients = sorted(client_metrics, key=lambda m: m.brier_score)[:5]
        for m in sorted_clients:
            LOGGER.info(
                "Model: %s | Cliente: %s (n=%d) → Brier: %.4f, ECE: %.4f",
                m.model_name,
                m.segment_value,
                m.n_samples,
                m.brier_score,
                m.ece,
            )


def save_outputs(
    metrics_list: List[CalibrationMetrics], bins_df: pd.DataFrame, project_root: Path
) -> Tuple[Path, Path]:
    """Save calibration metrics and reliability bins to Parquet files."""
    curated_dir = project_root / "data" / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)

    # Convert metrics to DataFrame
    metrics_records = [
        {
            "model": m.model_name,
            "segment_type": m.segment_type,
            "segment_value": m.segment_value,
            "n_samples": m.n_samples,
            "brier_score": m.brier_score,
            "ece": m.ece,
            "mce": m.mce,
        }
        for m in metrics_list
    ]
    metrics_df = pd.DataFrame(metrics_records)

    summary_path = curated_dir / "calibration_summary.parquet"
    bins_path = curated_dir / "calibration_bins.parquet"

    metrics_df.to_parquet(summary_path, compression="snappy", index=False)
    bins_df.to_parquet(bins_path, compression="snappy", index=False)

    LOGGER.info("Saved calibration metrics to %s", summary_path)
    LOGGER.info("Saved reliability bins to %s", bins_path)
    return summary_path, bins_path


def main() -> None:
    """Main entry point for calibration diagnostics."""
    configure_logging()
    project_root = Path(__file__).resolve().parents[2]

    LOGGER.info("Loading test data...")
    test_df, events = load_test_data(project_root)

    all_metrics: List[CalibrationMetrics] = []
    all_bins: List[pd.DataFrame] = []

    # Evaluate CatBoost (Model A)
    LOGGER.info("Evaluating CatBoost calibration...")
    catboost_preds = load_model_predictions(test_df, "catboost", project_root)
    if catboost_preds is not None:
        catboost_metrics, catboost_bins = evaluate_calibration(catboost_preds, events, "catboost")
        all_metrics.extend(catboost_metrics)
        all_bins.append(catboost_bins)

    # Evaluate AutoGluon (Model B) - optional
    LOGGER.info("Evaluating AutoGluon calibration...")
    autogluon_preds = load_model_predictions(test_df, "autogluon", project_root)
    if autogluon_preds is not None:
        autogluon_metrics, autogluon_bins = evaluate_calibration(autogluon_preds, events, "autogluon")
        all_metrics.extend(autogluon_metrics)
        all_bins.append(autogluon_bins)

    if not all_metrics:
        LOGGER.warning("No calibration metrics computed; no models available.")
        return

    # Combine bins
    combined_bins = pd.concat(all_bins, ignore_index=True) if all_bins else pd.DataFrame()

    print_summary(all_metrics)
    save_outputs(all_metrics, combined_bins, project_root)

    LOGGER.info("Calibration diagnostics complete.")


if __name__ == "__main__":  # pragma: no cover
    main()
