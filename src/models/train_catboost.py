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
    optional_paths = {
        "client_clusters": curated_dir / "client_clusters.parquet",
        "client_patterns": curated_dir / "client_patterns.parquet",
        "cluster_priors": curated_dir / "cluster_priors.parquet",
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
    for name, path in optional_paths.items():
        if path.exists():
            LOGGER.info("Loading optional %s from %s", name, path)
            tables[name] = pd.read_parquet(path)
        else:
            LOGGER.warning("Optional file %s not found; skipping", path)
            tables[name] = pd.DataFrame()
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

def enrich_with_cluster_priors(
    df: pd.DataFrame,
    client_clusters: pd.DataFrame,
    client_patterns: pd.DataFrame,
    cluster_priors: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich features with cluster priors and pattern information.
    
    Merges cluster_label, membership_prob, dominant_policy from clusters/patterns,
    and P(policy|cluster) from cluster_priors. Noise clients (cluster_label=-1)
    receive global priors as fallback.
    """
    out = df.copy()
    
    if not client_clusters.empty and "cliente" in client_clusters.columns:
        cluster_cols = [c for c in ["cliente", "cluster_label", "membership_prob"] if c in client_clusters.columns]
        out = out.merge(client_clusters[cluster_cols], on="cliente", how="left")
        out["cluster_label"] = out["cluster_label"].fillna(-1).astype(int)
        out["membership_prob"] = out["membership_prob"].fillna(0.0)
        LOGGER.info("Merged cluster features from client_clusters")
    else:
        out["cluster_label"] = -1
        out["membership_prob"] = 0.0
    
    if not client_patterns.empty and "cliente" in client_patterns.columns:
        pattern_cols = ["cliente", "dominant_policy", "pattern_confidence"]
        pattern_cols = [c for c in pattern_cols if c in client_patterns.columns]
        if len(pattern_cols) > 1:
            out = out.merge(client_patterns[pattern_cols], on="cliente", how="left")
            if "dominant_policy" in out.columns:
                out["dominant_policy"] = out["dominant_policy"].fillna("UNKNOWN").astype(str)
            if "pattern_confidence" in out.columns:
                out["pattern_confidence"] = out["pattern_confidence"].fillna(0.0)
            LOGGER.info("Merged pattern features from client_patterns")
    
    if not cluster_priors.empty and "cluster_label" in cluster_priors.columns:
        prior_cols = [c for c in cluster_priors.columns if c.startswith("P_") or c == "cluster_label"]
        global_priors = cluster_priors[cluster_priors["cluster_label"] == -1]
        if global_priors.empty:
            global_priors = cluster_priors.drop(columns=["cluster_label"]).mean().to_frame().T
            global_priors["cluster_label"] = -1
        out = out.merge(cluster_priors[prior_cols], on="cluster_label", how="left")
        p_cols = [c for c in prior_cols if c.startswith("P_")]
        for p_col in p_cols:
            fallback_val = global_priors[p_col].iloc[0] if p_col in global_priors.columns else 0.0
            out[p_col] = out[p_col].fillna(fallback_val)
        LOGGER.info("Merged cluster priors: %s", p_cols)
    else:
        LOGGER.warning("No cluster_priors available; skipping priors enrichment")
    
    if "dominant_policy" in out.columns:
        out["dominant_policy"] = out["dominant_policy"].fillna("UNKNOWN").astype(str)
    
    return out

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
        for col in ["tipo_documento", "chiave_contabile", "divisa", "periodo_contabile", "macro_tipo", "dominant_policy"]
        if col in feature_df.columns
    ]
    cat_indices = [cast(int, feature_df.columns.get_loc(col)) for col in categorical_cols]
    return feature_df, target, categorical_cols, cat_indices

def build_train_validation_split(train_df: pd.DataFrame) -> DatasetSplit:
    n_samples = len(train_df)
    groups = train_df["cliente"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if n_samples < 3 or unique_groups.size < 2:
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
    features = enrich_with_cluster_priors(
        features,
        tables.get("client_clusters", pd.DataFrame()),
        tables.get("client_patterns", pd.DataFrame()),
        tables.get("cluster_priors", pd.DataFrame()),
    )
    train_df, test_df = temporal_split(features, tables["events"])
    train_features, train_target, cat_cols, cat_indices = select_features(train_df)

    if len(cat_indices) == 0:
        LOGGER.warning("No categorical features detected; CatBoost will treat all features as numerical.")

    split = build_train_validation_split(train_df)
    model = train_model(train_features, train_target, cat_indices, split)

    feature_importance = model.get_feature_importance()
    feature_names = train_features.columns.tolist()
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importance,
    }).sort_values("importance", ascending=False)
    LOGGER.info("Top 15 feature importances:\n%s", importance_df.head(15).to_string(index=False))
    
    importance_path = project_root / CURATED_DIR / "feature_importance.parquet"
    importance_df.to_parquet(importance_path, compression="snappy", index=False)
    LOGGER.info("Saved feature importance to %s", importance_path)

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

if __name__ == "__main__":
    main()

