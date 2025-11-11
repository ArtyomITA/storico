"""Unified pointwise trainer for CatBoost (primary) and AutoGluon (optional).

Prompt 6: Train models on invoice-level features with cluster/pattern features,
temporal split, class-imbalance handling, isotonic calibration.
Saves models/catboost.cbm, models/calibrator.pkl, models/champion.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for pointwise trainer."""
    
    model_type: str = "catboost"  # "catboost" or "autogluon"
    temporal_split_fraction: float = 0.7
    n_folds: int = 3
    catboost_iterations: int = 1500
    catboost_depth: int = 6
    catboost_learning_rate: float = 0.05
    catboost_l2_leaf_reg: float = 5.0
    early_stopping_rounds: int = 200
    autogluon_time_limit: int = 3600
    autogluon_presets: str = "best_quality"
    apply_calibration: bool = True
    random_seed: int = 42


class IdentityCalibrator:
    """Fallback calibrator that returns raw probabilities."""

    def fit(self, *_: object, **__: object) -> "IdentityCalibrator":
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)


def configure_logging(level: int | str | None = logging.INFO) -> Path:
    """Configure logging handlers."""
    return setup_logging(level)


def load_curated_tables(project_root: Path) -> Dict[str, pd.DataFrame]:
    """Load curated parquet tables required for training.
    
    Prompt 6: Load train_features + cluster/pattern features (no leakage).
    """
    curated_dir = project_root / "data" / "curated"
    
    tables: Dict[str, pd.DataFrame] = {}
    required = {
        "train_features": "train_features.parquet",
        "events": "events.parquet",
    }
    
    optional = {
        "event_items": "event_items.parquet",
        "client_patterns": "client_patterns.parquet",  # Prompt 5 output
        "client_clusters": "client_clusters.parquet",  # Prompt 4 output
    }
    
    for name, filename in required.items():
        path = curated_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        LOGGER.info("Loading %s from %s", name, path)
        tables[name] = pd.read_parquet(path)
    
    for name, filename in optional.items():
        path = curated_dir / filename
        if path.exists():
            LOGGER.info("Loading %s from %s", name, path)
            tables[name] = pd.read_parquet(path)
        else:
            LOGGER.warning("Optional file %s not found; skipping", path)
            tables[name] = pd.DataFrame()
    
    return tables


def enrich_with_cluster_features(
    features: pd.DataFrame,
    client_patterns: pd.DataFrame,
    client_clusters: pd.DataFrame
) -> pd.DataFrame:
    """Enrich invoice-level features with client cluster/pattern features (no leakage).
    
    Prompt 6: Include cluster/pattern features of the cliente (historical, no post-event info).
    """
    df = features.copy()
    
    # Merge client_clusters (cluster_label, membership_prob)
    if not client_clusters.empty and "cliente" in client_clusters.columns:
        cluster_cols = ["cliente", "cluster_label", "membership_prob"]
        cluster_cols = [col for col in cluster_cols if col in client_clusters.columns]
        df = df.merge(client_clusters[cluster_cols], on="cliente", how="left")
        df["cluster_label"] = df["cluster_label"].fillna(-1).astype(int)
        df["membership_prob"] = df["membership_prob"].fillna(0.0)
        LOGGER.info("Merged cluster features: %s", cluster_cols)
    
    # Merge client_patterns (dominant_policy, pattern_confidence, periodicity, k_mode, etc.)
    if not client_patterns.empty and "cliente" in client_patterns.columns:
        # Select pattern features (exclude cliente)
        pattern_features = [
            col for col in client_patterns.columns
            if col != "cliente" and not col.endswith("_score")  # Avoid leaking policy_scores
        ]
        
        # Include key pattern features
        pattern_to_include = [
            "dominant_policy", "pattern_confidence",
            "periodicity_weekly_flag", "periodicity_monthly_flag",
            "k_mode", "k_entropy",
            "month_coverage_mean", "month_coverage_p95", "month_coverage_full_rate"
        ]
        pattern_features = [col for col in pattern_to_include if col in client_patterns.columns]
        
        if pattern_features:
            df = df.merge(
                client_patterns[["cliente"] + pattern_features],
                on="cliente",
                how="left"
            )
            LOGGER.info("Merged pattern features: %s", pattern_features)
    
    return df


def enrich_event_items_features(
    features: pd.DataFrame,
    event_items: pd.DataFrame
) -> pd.DataFrame:
    """Enrich with event_items metadata (periodo_contabile, etc.)."""
    df = features.copy()
    
    if not event_items.empty:
        join_cols = ["cliente", "event_key", "numero_documento"]
        available = [col for col in join_cols if col in event_items.columns]
        
        if len(available) == len(join_cols):
            enrich = event_items[join_cols + ["periodo_contabile"]].drop_duplicates()
            df = df.merge(enrich, on=join_cols, how="left")
    
    # Add categorical transformations
    if "periodo_contabile" not in df.columns:
        df["periodo_contabile"] = pd.NA
    
    df["macro_tipo"] = df["tipo_documento"].str[:1] if "tipo_documento" in df.columns else "UNKNOWN"
    
    # Convert categoricals to string
    categorical_targets = [
        "periodo_contabile", "tipo_documento", "chiave_contabile",
        "divisa", "macro_tipo", "dominant_policy"
    ]
    
    for col in categorical_targets:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").astype(str)
        else:
            df[col] = "UNKNOWN"
    
    return df


def temporal_split(
    df: pd.DataFrame,
    events: pd.DataFrame,
    split_fraction: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal train/test split based on data_pareggio.
    
    Prompt 6: No future leakage - split by event date.
    """
    if "data_pareggio" not in events.columns:
        raise ValueError("events table must include data_pareggio for temporal split")
    
    event_dates = events[["event_key", "data_pareggio"]].dropna().copy()
    event_dates["data_pareggio"] = pd.to_datetime(event_dates["data_pareggio"], errors="coerce")
    event_dates = event_dates.dropna(subset=["data_pareggio"])
    
    df = df.merge(event_dates, on="event_key", how="inner")
    df = df.sort_values("data_pareggio")
    
    cutoff_index = int(len(df) * split_fraction)
    cutoff_index = max(cutoff_index, 1)
    
    train_df = df.iloc[:cutoff_index]
    test_df = df.iloc[cutoff_index:]
    
    if test_df.empty:
        # Fallback: use last 30% as test
        test_df = train_df.tail(max(1, int(len(train_df) * (1 - split_fraction))))
        train_df = train_df.iloc[:-len(test_df)]
    
    LOGGER.info(
        "Temporal split: train=%d (%.1f%%), test=%d (%.1f%%)",
        len(train_df), len(train_df) / len(df) * 100,
        len(test_df), len(test_df) / len(df) * 100
    )
    
    return train_df, test_df


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[int]]:
    """Select feature columns and target, identify categorical features."""
    target = df["label"].astype(int)
    
    drop_cols = ["label", "data_pareggio", "cliente", "event_key", "numero_documento"]
    feature_df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Identify categorical columns
    categorical_cols = [
        col for col in [
            "tipo_documento", "chiave_contabile", "divisa",
            "periodo_contabile", "macro_tipo",
            "dominant_policy",  # Prompt 6: pattern feature
            "cluster_label"  # Prompt 6: cluster feature (will be treated as categorical)
        ]
        if col in feature_df.columns
    ]
    
    # Also treat boolean flags as categorical
    bool_cols = [
        col for col in feature_df.columns
        if col.endswith("_flag") or col in ["periodicity_weekly_flag", "periodicity_monthly_flag"]
    ]
    categorical_cols.extend(bool_cols)
    categorical_cols = list(set(categorical_cols))
    
    # Get categorical indices for CatBoost
    cat_indices = [int(feature_df.columns.get_loc(col)) for col in categorical_cols if col in feature_df.columns]
    
    LOGGER.info("Selected %d features (%d categorical)", len(feature_df.columns), len(cat_indices))
    
    return feature_df, target, categorical_cols, cat_indices


def handle_class_imbalance(
    train_target: pd.Series,
    config: TrainingConfig
) -> Optional[Dict]:
    """Compute class weights for imbalance handling.
    
    Prompt 6: Class-imbalance handling for pointwise ranking.
    """
    class_counts = train_target.value_counts().to_dict()
    total = train_target.count()
    
    pos_ratio = class_counts.get(1, 0) / total if total > 0 else 0.5
    neg_ratio = class_counts.get(0, 0) / total if total > 0 else 0.5
    
    LOGGER.info("Class distribution: positive=%.2f%%, negative=%.2f%%", pos_ratio * 100, neg_ratio * 100)
    
    if pos_ratio < 0.1 or pos_ratio > 0.9:
        # Severe imbalance
        LOGGER.warning("Severe class imbalance detected (pos=%.2f%%)", pos_ratio * 100)
        
        # CatBoost uses scale_pos_weight or class_weights
        if config.model_type == "catboost":
            scale_pos_weight = neg_ratio / pos_ratio if pos_ratio > 0 else 1.0
            return {"scale_pos_weight": scale_pos_weight}
    
    return None


def train_catboost(
    features: pd.DataFrame,
    target: pd.Series,
    cat_indices: List[int],
    config: TrainingConfig,
    class_weights: Optional[Dict] = None
) -> object:
    """Train CatBoost classifier with validation split.
    
    Prompt 6: CatBoost primario con early stopping e validation.
    """
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        raise ImportError("CatBoost not installed. Install: pip install catboost")
    
    # Group-based validation split (by cliente)
    groups = features.index.to_series().map(
        lambda idx: "unknown"  # Will use row-based split instead
    )
    
    # Simple 80/20 split within train
    n_samples = len(features)
    split_point = int(n_samples * 0.8)
    split_point = max(1, min(split_point, n_samples - 1))
    
    train_idx = np.arange(split_point)
    valid_idx = np.arange(split_point, n_samples)
    
    train_pool = Pool(
        data=features.iloc[train_idx],
        label=target.iloc[train_idx],
        cat_features=cat_indices
    )
    
    valid_pool = Pool(
        data=features.iloc[valid_idx],
        label=target.iloc[valid_idx],
        cat_features=cat_indices
    )
    
    params = {
        "depth": config.catboost_depth,
        "learning_rate": config.catboost_learning_rate,
        "l2_leaf_reg": config.catboost_l2_leaf_reg,
        "iterations": config.catboost_iterations,
        "loss_function": "Logloss",
        "early_stopping_rounds": config.early_stopping_rounds,
        "verbose": 100,
        "eval_metric": "AUC",
        "random_seed": config.random_seed,
    }
    
    if class_weights and "scale_pos_weight" in class_weights:
        params["scale_pos_weight"] = class_weights["scale_pos_weight"]
        LOGGER.info("Applied scale_pos_weight=%.3f", class_weights["scale_pos_weight"])
    
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    
    LOGGER.info("CatBoost training complete: %d iterations", model.tree_count_)
    
    return model


def train_autogluon(
    features: pd.DataFrame,
    target: pd.Series,
    config: TrainingConfig
) -> object:
    """Train AutoGluon TabularPredictor (optional).
    
    Prompt 6: AutoGluon opzionale.
    """
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        raise ImportError("AutoGluon not installed. Install: pip install autogluon")
    
    train_data = features.copy()
    train_data["label"] = target
    
    predictor = TabularPredictor(
        label="label",
        problem_type="binary",
        eval_metric="roc_auc",
        path="models/autogluon",
        verbosity=2
    )
    
    predictor.fit(
        train_data,
        time_limit=config.autogluon_time_limit,
        presets=config.autogluon_presets,
        ag_args_fit={"num_gpus": 0}  # CPU only
    )
    
    LOGGER.info("AutoGluon training complete")
    
    return predictor


def calibrate_probabilities(
    model: object,
    features: pd.DataFrame,
    target: pd.Series,
    cat_indices: List[int],
    model_type: str
) -> IsotonicRegression | IdentityCalibrator:
    """Apply isotonic calibration on validation set.
    
    Prompt 6: Calibrazione isotonic.
    """
    # Use last 20% of training data for calibration
    n_samples = len(features)
    calib_start = int(n_samples * 0.8)
    calib_idx = np.arange(calib_start, n_samples)
    
    if len(calib_idx) < 10:
        LOGGER.warning("Insufficient calibration samples; using identity calibrator")
        return IdentityCalibrator()
    
    # Get raw probabilities
    if model_type == "catboost":
        from catboost import Pool
        calib_pool = Pool(data=features.iloc[calib_idx], cat_features=cat_indices)
        raw_probs = model.predict_proba(calib_pool)[:, 1]
    else:
        # AutoGluon
        raw_probs = model.predict_proba(features.iloc[calib_idx], as_pandas=False)[:, 1]
    
    y_calib = target.iloc[calib_idx].values
    
    if len(np.unique(y_calib)) < 2:
        LOGGER.warning("Calibration set has single class; using identity calibrator")
        return IdentityCalibrator()
    
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, y_calib)
    
    LOGGER.info("Isotonic calibration fitted on %d samples", len(calib_idx))
    
    return calibrator


def evaluate_metrics(
    probs: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """Evaluate invoice-level metrics.
    
    Prompt 6: AUC, log-loss, average precision.
    """
    unique_labels = np.unique(labels)
    
    metrics: Dict[str, float] = {}
    
    if len(unique_labels) < 2:
        LOGGER.warning("Test labels contain single class; metrics may be undefined")
        metrics["log_loss"] = float(log_loss(labels, np.clip(probs, 1e-6, 1 - 1e-6), labels=[0, 1]))
        metrics["auc_roc"] = 0.5
        metrics["auc_pr"] = float(average_precision_score(labels, probs)) if unique_labels[0] == 1 else 0.0
    else:
        metrics["log_loss"] = float(log_loss(labels, probs, labels=[0, 1]))
        metrics["auc_roc"] = float(roc_auc_score(labels, probs))
        metrics["auc_pr"] = float(average_precision_score(labels, probs))
    
    LOGGER.info("Test metrics: AUC-ROC=%.4f, AUC-PR=%.4f, Log-loss=%.4f",
                metrics["auc_roc"], metrics["auc_pr"], metrics["log_loss"])
    
    return metrics


def save_champion_metadata(
    model_type: str,
    metrics: Dict[str, float],
    project_root: Path
) -> Path:
    """Save champion.json with model metadata.
    
    Prompt 6: models/champion.json con modello scelto.
    """
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    champion_path = models_dir / "champion.json"
    
    metadata = {
        "model": model_type,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "config": {
            "temporal_split": True,
            "calibration": "isotonic"
        }
    }
    
    champion_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    
    LOGGER.info("Saved champion metadata to %s", champion_path)
    
    return champion_path


def save_catboost_artifacts(
    model: object,
    calibrator: IsotonicRegression | IdentityCalibrator,
    project_root: Path
) -> None:
    """Save CatBoost model and calibrator.
    
    Prompt 6: models/catboost.cbm, models/calibrator.pkl
    """
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "catboost.cbm"
    calibrator_path = models_dir / "calibrator.pkl"
    
    model.save_model(str(model_path))
    
    with open(calibrator_path, "wb") as f:
        pickle.dump(calibrator, f)
    
    LOGGER.info("Saved CatBoost model to %s", model_path)
    LOGGER.info("Saved calibrator to %s", calibrator_path)


def save_autogluon_artifacts(predictor: object, project_root: Path) -> None:
    """Save AutoGluon predictor (already saved during fit)."""
    LOGGER.info("AutoGluon model saved to models/autogluon/")


def main(argv: Optional[List[str]] = None) -> None:
    """Prompt 6: Unified pointwise trainer entry point."""
    parser = argparse.ArgumentParser(
        description="Train pointwise model (CatBoost or AutoGluon) with cluster/pattern features"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="catboost",
        choices=["catboost", "autogluon"],
        help="Model type to train"
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip isotonic calibration"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args(argv)
    configure_logging(level=args.log_level)
    
    config = TrainingConfig(
        model_type=args.model,
        apply_calibration=not args.no_calibration
    )
    
    LOGGER.info("=== Prompt 6: Pointwise Trainer ===")
    LOGGER.info("Model: %s", config.model_type)
    LOGGER.info("Calibration: %s", config.apply_calibration)
    
    project_root = Path(__file__).resolve().parents[2]
    
    # Load tables
    tables = load_curated_tables(project_root)
    
    # Enrich features with cluster/pattern features (Prompt 6: no leakage)
    features = enrich_with_cluster_features(
        tables["train_features"],
        tables.get("client_patterns", pd.DataFrame()),
        tables.get("client_clusters", pd.DataFrame())
    )
    
    features = enrich_event_items_features(features, tables.get("event_items", pd.DataFrame()))
    
    # Temporal split
    train_df, test_df = temporal_split(features, tables["events"], config.temporal_split_fraction)
    
    # Select features
    train_features, train_target, cat_cols, cat_indices = select_features(train_df)
    test_features, test_target, _, _ = select_features(test_df)
    
    # Handle class imbalance
    class_weights = handle_class_imbalance(train_target, config)
    
    # Train model
    if config.model_type == "catboost":
        model = train_catboost(train_features, train_target, cat_indices, config, class_weights)
    else:
        model = train_autogluon(train_features, train_target, config)
    
    # Calibrate
    if config.apply_calibration:
        calibrator = calibrate_probabilities(
            model, train_features, train_target, cat_indices, config.model_type
        )
    else:
        calibrator = IdentityCalibrator()
    
    # Evaluate on test
    if config.model_type == "catboost":
        from catboost import Pool
        test_pool = Pool(data=test_features, cat_features=cat_indices)
        raw_probs = model.predict_proba(test_pool)[:, 1]
    else:
        raw_probs = model.predict_proba(test_features, as_pandas=False)[:, 1]
    
    calibrated_probs = calibrator.predict(raw_probs)
    metrics = evaluate_metrics(calibrated_probs, test_target.values)
    
    # Save artifacts
    if config.model_type == "catboost":
        save_catboost_artifacts(model, calibrator, project_root)
    else:
        save_autogluon_artifacts(model, project_root)
    
    save_champion_metadata(config.model_type, metrics, project_root)
    
    LOGGER.info("=== Training complete ===")


if __name__ == "__main__":
    main()
