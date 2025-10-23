"""Train AutoGluon Tabular model and compute client-level metrics."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, cast

try:  # AutoGluon>=1.0 exposes helper, but guard older versions
    from autogluon.common.utils.resource_utils import get_gpu_count_torch
except ImportError:  # pragma: no cover - fallback for older releases
    def get_gpu_count_torch() -> int:  # type: ignore
        return 0

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import average_precision_score, log_loss

from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)
CURATED_DIR = Path("data/curated")
MODELS_DIR = Path("models")
PREDICTOR_DIR = MODELS_DIR / "autogluon"
CLIENT_METRICS_PATH = CURATED_DIR / "client_model_metrics_autogluon.parquet"
TRAIN_FRACTION = 0.7


def configure_logging(level: int | str | None = logging.INFO) -> Path:
    return setup_logging(level)
def detect_compute_environment() -> Dict[str, object]:
    """Inspect GPU availability across supported backends and log results."""
    info: Dict[str, object] = {
        "torch_cuda_available": False,
        "torch_device_name": None,
        "torch_gpu_count": 0,
    }

    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_gpu_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            info["torch_device_name"] = torch.cuda.get_device_name(0)
        else:
            info["torch_device_name"] = "cpu"
    except Exception as exc:  # pragma: no cover - diagnostic only
        LOGGER.warning("Unable to inspect torch CUDA status: %s", exc)
        info["torch_device_name"] = "unavailable"

    # AutoGluon helper (falls back to torch internally)
    try:
        from autogluon.common.utils.resource_utils import get_gpu_count_torch  # type: ignore

        ag_gpu_count = int(get_gpu_count_torch())
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("AutoGluon GPU probe failed: %s", exc)
        ag_gpu_count = 0
    info["autogluon_gpu_count"] = ag_gpu_count

    try:
        from catboost.utils import get_gpu_device_count

        info["catboost_gpu_count"] = int(get_gpu_device_count())
    except Exception:  # pragma: no cover - catboost optional
        info["catboost_gpu_count"] = 0

    LOGGER.info(
        "GPU availability -> torch_cuda=%s (count=%s, device=%s), autogluon=%s, catboost=%s",
        info["torch_cuda_available"],
        info["torch_gpu_count"],
        info["torch_device_name"],
        info["autogluon_gpu_count"],
        info["catboost_gpu_count"],
    )
    return info




def load_curated_tables(project_root: Path) -> Dict[str, pd.DataFrame]:
    curated_dir = project_root / CURATED_DIR
    paths = {
        "train_features": curated_dir / "train_features.parquet",
        "events": curated_dir / "events.parquet",
        "event_items": curated_dir / "event_items.parquet",
        "catboost_metrics": curated_dir / "client_model_metrics_catboost.parquet",
    }
    tables: Dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.exists() and name in {"event_items", "catboost_metrics"}:
            LOGGER.warning("Optional file missing: %s", path)
            tables[name] = pd.DataFrame()
            continue
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        LOGGER.info("Loading %s from %s", name, path)
        tables[name] = pd.read_parquet(path)
    return tables


def enrich_features(features: pd.DataFrame, event_items: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    if not event_items.empty:
        join_cols = ["cliente", "event_key", "numero_documento"]
        available = [col for col in join_cols if col in event_items.columns]
        if len(available) == len(join_cols):
            enrich = event_items[join_cols + ["periodo_contabile"]].drop_duplicates()
            df = df.merge(enrich, on=join_cols, how="left")
    if "periodo_contabile" not in df.columns:
        df["periodo_contabile"] = pd.NA
    df["macro_tipo"] = df["tipo_documento"].str[:1]

    categorical_targets = ["periodo_contabile", "tipo_documento", "chiave_contabile", "divisa", "macro_tipo"]
    for col in categorical_targets:
        if col not in df.columns:
            df[col] = "UNKNOWN"
        df[col] = df[col].fillna("UNKNOWN").astype(str)
    return df


def temporal_split(df: pd.DataFrame, events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "data_pareggio" not in events.columns:
        raise ValueError("events table must include data_pareggio")
    event_dates = events[["event_key", "data_pareggio"]].dropna().copy()
    event_dates["data_pareggio"] = pd.to_datetime(event_dates["data_pareggio"], errors="coerce")
    event_dates = event_dates.dropna(subset=["data_pareggio"])
    df = df.merge(event_dates, on="event_key", how="inner")
    df = df.sort_values("data_pareggio")
    cutoff_index = int(len(df) * TRAIN_FRACTION)
    cutoff_index = max(cutoff_index, 1)
    train_df = df.iloc[:cutoff_index]
    test_df = df.iloc[cutoff_index:]
    if test_df.empty:
        test_df = train_df.tail(max(1, int(len(train_df) * (1 - TRAIN_FRACTION))))
        train_df = train_df.iloc[:-len(test_df)]
    return train_df, test_df


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    target = df["label"].astype(int)
    drop_cols = ["label", "data_pareggio", "cliente", "event_key", "numero_documento"]
    feature_df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    return feature_df, target


def evaluate_invoice_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    unique_labels = np.unique(labels)
    clipped_probs = np.clip(probs, 1e-6, 1 - 1e-6)
    if unique_labels.size < 2:
        LOGGER.warning("Test labels contain a single class; metrics may be unstable.")
        metrics["log_loss"] = float(log_loss(labels, clipped_probs, labels=[0, 1]))
        metrics["auc_pr"] = float(average_precision_score(labels, clipped_probs)) if unique_labels[0] == 1 else 0.0
    else:
        metrics["log_loss"] = float(log_loss(labels, clipped_probs, labels=[0, 1]))
        metrics["auc_pr"] = float(average_precision_score(labels, probs))
    return metrics


def evaluate_event_metrics(df: pd.DataFrame, probs: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["prob"] = probs
    event_records: List[Dict[str, object]] = []
    client_records: Dict[str, List[float]] = {}
    match_records: Dict[str, List[float]] = {}

    for event_key, group in df.groupby("event_key"):
        positives = group[group["label"] == 1]
        k = len(positives)
        if k == 0:
            continue
        group_sorted = group.sort_values("prob", ascending=False)
        top_docs = group_sorted.head(k)
        hits = int((top_docs["label"] == 1).sum())
        recall = hits / k
        match_rate = 1.0 if hits == k else 0.0
        cliente = str(group.iloc[0]["cliente"])
        event_records.append(
            {
                "cliente": cliente,
                "event_key": event_key,
                "recall_at_k": recall,
                "match_rate": match_rate,
            }
        )
        client_records.setdefault(cliente, []).append(recall)
        match_records.setdefault(cliente, []).append(match_rate)

    client_rows: List[Dict[str, object]] = []
    for cliente, recalls in client_records.items():
        matches = match_records.get(cliente, [])
        client_rows.append(
            {
                "cliente": cliente,
                "model_recall_at_k": float(np.mean(recalls)),
                "model_match_rate": float(np.mean(matches)) if matches else 0.0,
                "events_tested": len(recalls),
            }
        )

    return pd.DataFrame(event_records), pd.DataFrame(client_rows)


def save_client_metrics(df: pd.DataFrame, project_root: Path) -> Path:
    curated_dir = project_root / CURATED_DIR
    curated_dir.mkdir(parents=True, exist_ok=True)
    output_path = curated_dir / CLIENT_METRICS_PATH.name
    df.to_parquet(output_path, compression="snappy", index=False)
    LOGGER.info("Saved client metrics to %s", output_path)
    return output_path


def print_comparison(autogluon_metrics: pd.DataFrame, catboost_metrics: pd.DataFrame) -> None:
    if autogluon_metrics.empty:
        LOGGER.warning("No AutoGluon client metrics to compare.")
        return
    print("AutoGluon client metrics (sample):")
    print(autogluon_metrics.head(5).to_string(index=False))

    if catboost_metrics.empty:
        LOGGER.info("CatBoost metrics not available for comparison.")
        return

    merged = autogluon_metrics.merge(catboost_metrics, on="cliente", how="left", suffixes=("_autogluon", "_catboost"))
    print("\nConfronto AutoGluon vs CatBoost (prime righe):")
    columns = [
        "cliente",
        "model_recall_at_k_autogluon",
        "model_recall_at_k_catboost",
        "model_match_rate_autogluon",
        "model_match_rate_catboost",
    ]
    existing_cols = [col for col in columns if col in merged.columns]
    print(merged[existing_cols].head(5).to_string(index=False))


def train_autogluon(project_root: Path, tables: Dict[str, pd.DataFrame]) -> Tuple[TabularPredictor, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    features = enrich_features(tables["train_features"], tables["event_items"])
    train_df, test_df = temporal_split(features, tables["events"])

    train_features, train_target = select_features(train_df)
    test_features, test_target = select_features(test_df)

    predictor_path = project_root / PREDICTOR_DIR
    predictor_path.mkdir(parents=True, exist_ok=True)

    train_data = train_features.copy()
    train_data["label"] = train_target.to_numpy()

    env_info = detect_compute_environment()
    gpu_count = int(cast(int, env_info.get("autogluon_gpu_count", 0)))
    if gpu_count == 0 and bool(env_info.get("torch_cuda_available")):
        gpu_count = int(cast(int, env_info.get("torch_gpu_count", 0)))
    if gpu_count > 0:
        LOGGER.info("Configuring AutoGluon to use %s GPU(s).", gpu_count)
    else:
        LOGGER.warning(
            "No GPUs detected by AutoGluon/Torch. Training will proceed on CPU. "
            "Verify that your PyTorch installation includes CUDA support (current: CPU-only)."
        )

    predictor = TabularPredictor(
        label="label",
        problem_type="binary",
        eval_metric="log_loss",
        path=str(predictor_path),
        verbosity=3,
    )
    predictor.fit(
        train_data=train_data,
        presets="medium_quality_faster_train",
        excluded_model_types=["NN_TORCH", "FASTAI"],
        ag_args_fit={
            "num_bag_folds": 0,
            "num_gpus": gpu_count if gpu_count > 0 else None,
        },
        auto_stack=False,
        time_limit=900,
        verbosity=3,
    )

    test_data = test_features.copy()
    test_data["label"] = test_target.to_numpy()
    leaderboard_df = predictor.leaderboard(test_data, silent=True)

    proba_df = predictor.predict_proba(test_features)
    if isinstance(proba_df, pd.DataFrame):
        probs = proba_df.iloc[:, 1].to_numpy()
    else:
        probs = np.asarray(proba_df)
    probs = np.asarray(probs)

    invoice_metrics = evaluate_invoice_metrics(probs, test_target.to_numpy())
    _, client_metrics = evaluate_event_metrics(test_df, probs)

    return predictor, client_metrics, leaderboard_df, invoice_metrics


def main() -> None:
    configure_logging()
    project_root = Path(__file__).resolve().parents[2]
    tables = load_curated_tables(project_root)

    predictor, client_metrics_df, leaderboard_df, invoice_metrics = train_autogluon(project_root, tables)
    save_client_metrics(client_metrics_df, project_root)

    LOGGER.info("Invoice-level metrics: %s", json.dumps(invoice_metrics, indent=2))
    print("\nAutoGluon leaderboard (test data):")
    print(leaderboard_df.head(10).to_string(index=False))

    print_comparison(client_metrics_df, tables.get("catboost_metrics", pd.DataFrame()))

    predictor_path = project_root / PREDICTOR_DIR
    LOGGER.info("AutoGluon predictor saved under %s", predictor_path)


if __name__ == "__main__":  # pragma: no cover
    main()
