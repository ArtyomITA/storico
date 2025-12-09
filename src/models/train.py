"""Unified training entrypoint with champion metadata selection.

This CLI orchestrates the CatBoost and AutoGluon training pipelines and
persists a ``champion.json`` manifest describing the latest trained model.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.models import train_autogluon as auto
from src.models import train_catboost as cat
from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)
DEFAULT_TRAIN_FEATURES = Path("data/curated/train_features.parquet")
DEFAULT_OUTPUT_DIR = Path("models")

@dataclass
class ChampionMetrics:
    """Container for the metrics persisted to ``champion.json``."""

    auc_pr: float
    match_rate: float

def configure_logging(level: int | str | None = logging.INFO) -> Path:
    """Initialise module logging using the shared configuration."""

    return setup_logging(level)

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the training orchestrator."""
    parser = argparse.ArgumentParser(
        description=(
            "Train either the CatBoost or AutoGluon classifier and persist a "
            "champion.json manifest with key metrics."
        )
    )
    parser.add_argument(
        "--model",
        choices=("catboost", "autogluon"),
        required=True,
        help="Model family to train.",
    )
    parser.add_argument(
        "--train-features",
        type=Path,
        default=None,
        help=(
            "Path to the train_features parquet file. Defaults to "
            "data/curated/train_features.parquet inside the project root."
        ),
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where model artefacts and champion.json will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed applied to NumPy/Python for reproducibility.",
    )
    return parser.parse_args(argv)

def set_random_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def resolve_project_root() -> Path:
    """Return the repository root based on the file location."""
    return Path(__file__).resolve().parents[2]

def resolve_path(path: Optional[Path], project_root: Path) -> Optional[Path]:
    """Resolve a possibly-relative path against the project root."""
    if path is None:
        return None
    return path if path.is_absolute() else project_root / path

def ensure_output_dir(path: Path) -> Path:
    """Ensure the output directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def override_train_features(
    tables: Dict[str, pd.DataFrame],
    train_features_path: Optional[Path],
) -> None:
    """Optionally replace the train features table with a custom parquet."""
    if train_features_path is None:
        return
    if not train_features_path.exists():
        raise FileNotFoundError(f"Train features file not found: {train_features_path}")
    LOGGER.info("Overriding train features with %s", train_features_path)
    tables["train_features"] = pd.read_parquet(train_features_path)

def compute_match_rate(event_metrics: pd.DataFrame) -> float:
    """Aggregate match rate from event-level metrics."""
    if event_metrics.empty or "match_rate" not in event_metrics.columns:
        return 0.0
    return float(event_metrics["match_rate"].mean())

def run_catboost(
    project_root: Path,
    train_features_path: Optional[Path],
    output_dir: Path,
) -> ChampionMetrics:
    """Execute the CatBoost training pipeline and return champion metrics."""
    cat.MODELS_DIR = output_dir
    tables = cat.load_curated_tables(project_root)
    override_train_features(tables, train_features_path)

    features = cat.enrich_features(tables["train_features"], tables["event_items"])
    train_df, test_df = cat.temporal_split(features, tables["events"])
    train_features, train_target, _, cat_indices = cat.select_features(train_df)

    split = cat.build_train_validation_split(train_df)
    model = cat.train_model(train_features, train_target, cat_indices, split)
    calibrator = cat.calibrate_probabilities(model, train_features, train_target, cat_indices, split)

    test_features, test_target, _, _ = cat.select_features(test_df)
    test_pool = cat.Pool(data=test_features, cat_features=cat_indices)
    raw_probs = model.predict_proba(test_pool)[:, 1]
    calibrated_probs = calibrator.predict(raw_probs)

    invoice_metrics = cat.evaluate_invoice_metrics(calibrated_probs, test_target.to_numpy())
    event_metrics_df, client_metrics_df = cat.evaluate_event_metrics(test_df, calibrated_probs)

    cat.save_client_metrics(client_metrics_df, project_root)
    cat.save_artifacts(model, calibrator, project_root)

    match_rate = compute_match_rate(event_metrics_df)
    auc_pr = float(invoice_metrics.get("auc_pr", 0.0))
    LOGGER.info("CatBoost AUC-PR: %.4f | Match rate: %.4f", auc_pr, match_rate)
    return ChampionMetrics(auc_pr=auc_pr, match_rate=match_rate)

def run_autogluon(
    project_root: Path,
    train_features_path: Optional[Path],
    output_dir: Path,
    seed: int,
) -> ChampionMetrics:
    """Execute the AutoGluon training pipeline and return champion metrics."""
    auto.MODELS_DIR = output_dir
    auto.PREDICTOR_DIR = output_dir / "autogluon"

    tables = auto.load_curated_tables(project_root)
    override_train_features(tables, train_features_path)

    features = auto.enrich_features(tables["train_features"], tables["event_items"])
    train_df, test_df = auto.temporal_split(features, tables["events"])
    train_features, train_target = auto.select_features(train_df)
    test_features, test_target = auto.select_features(test_df)

    predictor_path = project_root / auto.PREDICTOR_DIR
    predictor_path.mkdir(parents=True, exist_ok=True)

    train_data = train_features.copy()
    train_data["label"] = train_target.to_numpy()

    env_info: Dict[str, Any] = auto.detect_compute_environment()
    gpu_count = int(env_info.get("autogluon_gpu_count") or 0)
    if gpu_count == 0 and bool(env_info.get("torch_cuda_available", False)):
        gpu_count = int(env_info.get("torch_gpu_count") or 0)

    predictor = auto.TabularPredictor(
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
        ag_args_fit={"num_bag_folds": 0, "num_gpus": gpu_count if gpu_count > 0 else None},
        auto_stack=False,
        time_limit=900,
        verbosity=3,
    )

    test_probs = predictor.predict_proba(test_features)
    if isinstance(test_probs, pd.DataFrame):
        probs = test_probs.iloc[:, 1].to_numpy()
    else:
        probs = np.asarray(test_probs)

    invoice_metrics = auto.evaluate_invoice_metrics(probs, test_target.to_numpy())
    event_metrics_df, client_metrics_df = auto.evaluate_event_metrics(test_df, probs)

    auto.save_client_metrics(client_metrics_df, project_root)

    match_rate = compute_match_rate(event_metrics_df)
    auc_pr = float(invoice_metrics.get("auc_pr", 0.0))
    LOGGER.info("AutoGluon AUC-PR: %.4f | Match rate: %.4f", auc_pr, match_rate)
    return ChampionMetrics(auc_pr=auc_pr, match_rate=match_rate)

def write_champion_manifest(output_dir: Path, model_name: str, metrics: ChampionMetrics) -> Path:
    """Persist the champion.json manifest and return its path."""
    payload = {
        "model": model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "auc_pr": metrics.auc_pr,
            "match_rate": metrics.match_rate,
        },
    }
    manifest_path = output_dir / "champion.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    LOGGER.info("Champion manifest written to %s", manifest_path)
    return manifest_path

def run(argv: Sequence[str] | None = None) -> None:
    """Entry point for CLI execution."""
    configure_logging()
    args = parse_args(argv)
    set_random_seed(args.seed)

    project_root = resolve_project_root()
    output_dir = resolve_path(args.output_dir, project_root)
    if output_dir is None:
        output_dir = project_root / DEFAULT_OUTPUT_DIR
    output_dir = ensure_output_dir(output_dir)

    train_features_path = resolve_path(args.train_features, project_root)
    if train_features_path is None:
        train_features_path = project_root / DEFAULT_TRAIN_FEATURES

    LOGGER.info("Starting training for model=%s", args.model)

    if args.model == "catboost":
        metrics = run_catboost(project_root, train_features_path, output_dir)
    else:
        metrics = run_autogluon(project_root, train_features_path, output_dir, seed=args.seed)

    write_champion_manifest(output_dir, args.model, metrics)

def main() -> None:
    run()

if __name__ == "__main__":
    main()

