"""Train CatBoost pointwise classifier with isotonic calibration and client metrics."""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import GroupKFold

from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)
CURATED_DIR = Path("data/curated")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "catboost.cbm"
CALIBRATOR_PATH = MODELS_DIR / "calibrator.pkl"
CLIENT_METRICS_PATH = CURATED_DIR / "client_model_metrics_catboost.parquet"
TRAIN_FRACTION = 0.7
N_FOLDS = 3

class IdentityCalibrator:
    """Fallback calibrator that returns raw probabilities."""

    def fit(self, *_: object, **__: object) -> "IdentityCalibrator":
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)




@dataclass
class DatasetSplit:
    train_idx: np.ndarray
    valid_idx: np.ndarray


def configure_logging(level: int | str | None = logging.INFO) -> Path:
    return setup_logging(level)


def load_curated_tables(project_root: Path) -> Dict[str, pd.DataFrame]:
    curated_dir = project_root / CURATED_DIR
    paths = {
        "train_features": curated_dir / "train_features.parquet",
        "events": curated_dir / "events.parquet",
        "event_items": curated_dir / "event_items.parquet",
    }
    tables: Dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.exists() and name == "event_items":
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


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[int]]:
    target = df["label"].astype(int)
    drop_cols = ["label", "data_pareggio", "cliente", "event_key", "numero_documento"]
    feature_df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    categorical_cols = [
        col
        for col in ["tipo_documento", "chiave_contabile", "divisa", "periodo_contabile", "macro_tipo"]
        if col in feature_df.columns
    ]
    cat_indices = [cast(int, feature_df.columns.get_loc(col)) for col in categorical_cols]
    return feature_df, target, categorical_cols, cat_indices


def build_train_validation_split(train_df: pd.DataFrame) -> DatasetSplit:
    n_samples = len(train_df)
    groups = train_df["cliente"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if n_samples < 3 or unique_groups.size < 2:
        # Fallback: simple chronological split 80/20
        split_point = max(1, int(n_samples * 0.8))
        if split_point >= n_samples:
            split_point = n_samples - 1
        split_point = max(split_point, 1)
        indices = np.arange(n_samples)
        train_idx = indices[:split_point]
        valid_idx = indices[split_point:]
        if valid_idx.size == 0:
            valid_idx = indices[-1:]
            train_idx = indices
        return DatasetSplit(train_idx=train_idx, valid_idx=valid_idx)

    n_splits = min(N_FOLDS, unique_groups.size)
    if n_splits < 2:
        n_splits = 2
    kfold = GroupKFold(n_splits=n_splits)
    indices = np.arange(n_samples)
    split_iter = kfold.split(indices, groups=groups)
    train_idx, valid_idx = next(split_iter)
    return DatasetSplit(train_idx=train_idx, valid_idx=valid_idx)


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_indices: List[int],
    split: DatasetSplit,
) -> CatBoostClassifier:
    train_pool = Pool(data=features.iloc[split.train_idx], label=target.iloc[split.train_idx], cat_features=categorical_indices)
    valid_pool = Pool(data=features.iloc[split.valid_idx], label=target.iloc[split.valid_idx], cat_features=categorical_indices)

    params = {
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
        "iterations": 1500,
        "loss_function": "Logloss",
        "early_stopping_rounds": 200,
        "verbose": 100,
        "eval_metric": "AUC",
        "random_seed": 42,
    }

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model


def calibrate_probabilities(
    model: CatBoostClassifier,
    features: pd.DataFrame,
    target: pd.Series,
    categorical_indices: List[int],
    split: DatasetSplit,
) -> IsotonicRegression | IdentityCalibrator:
    valid_pool = Pool(data=features.iloc[split.valid_idx], label=target.iloc[split.valid_idx], cat_features=categorical_indices)
    raw_probs = model.predict_proba(valid_pool)[:, 1]
    y_valid = target.iloc[split.valid_idx]
    if len(np.unique(y_valid)) < 2 or raw_probs.size < 2:
        LOGGER.warning("Insufficient variation in validation labels; using identity calibrator.")
        return IdentityCalibrator()
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, y_valid)
    return calibrator


def evaluate_invoice_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        LOGGER.warning("Test labels contain a single class; metrics may be undefined.")
        metrics["log_loss"] = float(log_loss(labels, np.clip(probs, 1e-6, 1 - 1e-6), labels=[0, 1]))
        metrics["auc_pr"] = float(average_precision_score(labels, probs)) if unique_labels[0] == 1 else 0.0
    else:
        metrics["log_loss"] = float(log_loss(labels, probs, labels=[0, 1]))
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
        event_records.append({
            "cliente": cliente,
            "event_key": event_key,
            "recall_at_k": recall,
            "match_rate": match_rate,
        })
        client_records.setdefault(cliente, []).append(recall)
        match_records.setdefault(cliente, []).append(match_rate)

    client_rows = []
    for cliente, recalls in client_records.items():
        matches = match_records.get(cliente, [])
        client_rows.append({
            "cliente": cliente,
            "model_recall_at_k": float(np.mean(recalls)),
            "model_match_rate": float(np.mean(matches)) if matches else 0.0,
            "events_tested": len(recalls),
        })

    return pd.DataFrame(event_records), pd.DataFrame(client_rows)


def save_artifacts(model: CatBoostClassifier, calibrator: IsotonicRegression, project_root: Path) -> None:
    models_dir = project_root / MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(models_dir / MODEL_PATH.name))
    with open(models_dir / CALIBRATOR_PATH.name, "wb") as f:
        pickle.dump(calibrator, f)
    LOGGER.info("Saved model to %s and calibrator to %s", models_dir / MODEL_PATH.name, models_dir / CALIBRATOR_PATH.name)


def save_client_metrics(df: pd.DataFrame, project_root: Path) -> Path:
    curated_dir = project_root / CURATED_DIR
    curated_dir.mkdir(parents=True, exist_ok=True)
    output_path = curated_dir / CLIENT_METRICS_PATH.name
    df.to_parquet(output_path, compression="snappy", index=False)
    LOGGER.info("Saved client metrics to %s", output_path)
    return output_path


def main() -> None:
    configure_logging()
    project_root = Path(__file__).resolve().parents[2]
    tables = load_curated_tables(project_root)

    features = enrich_features(tables["train_features"], tables["event_items"])
    train_df, test_df = temporal_split(features, tables["events"])
    train_features, train_target, cat_cols, cat_indices = select_features(train_df)

    if len(cat_indices) == 0:
        LOGGER.warning("No categorical features detected; CatBoost will treat all features as numerical.")

    split = build_train_validation_split(train_df)
    model = train_model(train_features, train_target, cat_indices, split)

    calibrator = calibrate_probabilities(model, train_features, train_target, cat_indices, split)

    test_features, test_target, _, _ = select_features(test_df)
    test_pool = Pool(data=test_features, cat_features=cat_indices)
    raw_probs = model.predict_proba(test_pool)[:, 1]
    calibrated_probs = calibrator.predict(raw_probs)
    invoice_metrics = evaluate_invoice_metrics(calibrated_probs, test_target.to_numpy())

    event_metrics_df, client_metrics_df = evaluate_event_metrics(test_df, calibrated_probs)
    save_client_metrics(client_metrics_df, project_root)
    save_artifacts(model, calibrator, project_root)

    LOGGER.info("Invoice-level metrics: %s", json.dumps(invoice_metrics, indent=2))
    if not client_metrics_df.empty:
        print(client_metrics_df.head(5).to_string(index=False))
    else:
        LOGGER.warning("No client metrics computed.")


if __name__ == "__main__":  # pragma: no cover
    main()
