"""Unified command to execute the full payment-pattern analytics pipeline.

Prompt 8: Updated DAG order:
normalize → features → embeddings → HDBSCAN → pattern profiles → training → scoring → analytics
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.utils.logging_setup import (
    configure_logging as setup_logging,
    get_current_log_file,
    log_timed,
)

LOGGER = logging.getLogger(__name__)

def _get_data_root() -> Path:
    """Get data root: data_mock/ if USE_MOCK=true, else data/."""
    use_mock = os.getenv("USE_MOCK", "false").lower() == "true"
    return Path("data_mock" if use_mock else "data")

CURATED_DIR = _get_data_root() / "curated"
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
LOGS_DIR = Path("logs")

PIPELINE_STEPS = [
    "normalize",
    "features",
    "embeddings",
    "hdbscan",
    "patterns",
    "train",
    "score",
    "analytics",
]

@dataclass(frozen=True)
class PipelineConfig:
    """Configuration derived from CLI options.
    
    Prompt 8: Idempotent pipeline with step flags.
    """

    steps: List[str]
    skip_normalize: bool
    skip_features: bool
    skip_embeddings: bool
    skip_hdbscan: bool
    skip_patterns: bool
    skip_train: bool
    skip_score: bool
    skip_analytics: bool
    model_type: str
    log_level: str

def parse_args(argv: Sequence[str] | None = None) -> PipelineConfig:
    """Parse CLI options for the unified pipeline runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the end-to-end analytics workflow: ingest → features → training → "
            "analytics artefacts, optionally including embeddings and clustering."
        )
    )
    parser.add_argument(
        "--input",
        dest="input_patterns",
        action="append",
        help=(
            "Glob pattern(s) for raw CSV/XLSX files. Repeat to provide multiple patterns. "
            "Defaults to the ingestion module fallback when omitted."
        ),
    )
    parser.add_argument(
        "--docmap",
        dest="docmap_path",
        type=Path,
        help="Optional parquet path for the document map generated during ingestion.",
    )
    parser.add_argument(
        "--model",
        choices=("catboost", "autogluon"),
        default="catboost",
        help="Model family to train for champion selection.",
    )
    parser.add_argument(
        "--make-embeddings",
        action="store_true",
        help="Enable SHAP-based client embedding generation.",
    )
    parser.add_argument(
        "--run-hdbscan",
        action="store_true",
        help="Cluster embeddings with HDBSCAN and produce cluster summaries.",
    )
    parser.add_argument(
        "--neighbors",
        dest="embedding_neighbors",
        type=int,
        default=10,
        help="Number of neighbours per client when building the similarity index.",
    )
    parser.add_argument(
        "--umap",
        dest="use_umap",
        action="store_true",
        help="Use UMAP instead of PCA for the 2D embedding projection when available.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=30,
        help="HDBSCAN min_cluster_size parameter (only if --run-hdbscan).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=15,
        help="HDBSCAN min_samples parameter (only if --run-hdbscan).",
    )
    parser.add_argument(
        "--make-kpi",
        action="store_true",
        help="Compute extended ranking KPIs (precision@k, MRR, nDCG@k) on test set.",
    )
    parser.add_argument(
        "--make-calibration",
        action="store_true",
        help="Compute calibration diagnostics (Brier, ECE, MCE, reliability curves) on test set.",
    )
    parser.add_argument(
        "--make-policy-metrics",
        action="store_true",
        help="Compute policy classifier metrics (confusion matrix, accuracy, uplift) on events.",
    )

    parser.add_argument(
        "--step",
        choices=[
            "ingest", "events", "labels", "features", "validate",
            "train", "patterns", "rules", "table", "reports",
            "embeddings", "projection", "hdbscan", "clusters",
            "kpi", "calibration", "policy"
        ],
        help="Run only a specific pipeline step (skip others).",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip pre-training validation checks (use with caution).",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity for debugging.",
    )

    args = parser.parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log_file = get_current_log_file()
    if log_file is not None:
        LOGGER.info("JSON log file: %s", log_file)

    patterns = tuple(args.input_patterns) if args.input_patterns else tuple()
    return PipelineConfig(
        input_patterns=patterns,
        docmap_path=args.docmap_path,
        model=args.model,
        make_embeddings=args.make_embeddings,
        run_hdbscan=args.run_hdbscan,
        embedding_neighbors=args.embedding_neighbors,
        use_umap=args.use_umap,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        make_kpi=args.make_kpi,
        make_calibration=args.make_calibration,
        make_policy_metrics=args.make_policy_metrics,
        step=args.step,
        skip_validation=args.skip_validation,
    )

class PipelineRunner:
    """Execute pipeline steps sequentially with structured logging."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.project_root = Path(__file__).resolve().parents[2]

    def run(self) -> None:
        step_registry: Dict[str, Tuple[str, Callable[[], object]]] = {
            "normalize": ("Normalize SAP data (Prompt 2)", self.run_normalize_sap),
            "ingest": ("Ingest raw datasets", self.run_ingestion),
            "events": ("Build clearing events", self.run_build_events),
            "labels": ("Generate training labels", self.run_make_labels),
            "features": ("Engineer features & client profiles", self.run_make_features),
            "validate": ("Validate curated datasets", self.run_pretraining_validations),
            "embeddings": ("Create client embeddings", self.run_make_embeddings),
            "projection": ("Compute embedding projection", self.run_embedding_projection),
            "hdbscan": ("Cluster embeddings with HDBSCAN", self.run_hdbscan),
            "patterns": ("Compute pattern detectors (Prompt 5)", self.run_pattern_detectors),
            "train": ("Train pointwise model (Prompt 6)", self.run_pointwise_trainer),
            "score": ("Score open invoices (Prompt 7)", self.run_score_open_invoices),
            "clusters": ("Summarise clusters", self.run_cluster_profiles),
            "rules": ("Backtest policy rules", self.run_rules_backtest),
            "table": ("Assemble pattern table", self.run_pattern_table),
            "reports": ("Render pattern card reports", self.run_pattern_reports),
            "kpi": ("Compute ranking KPIs", self.run_kpi_ranking),
            "calibration": ("Compute calibration diagnostics", self.run_calibration),
            "policy": ("Compute policy classifier metrics", self.run_policy_metrics),
        }

        if self.config.step:
            if self.config.step not in step_registry:
                raise ValueError(f"Unknown step: {self.config.step}")
            name, callback = step_registry[self.config.step]
            LOGGER.info(f"Running single step: {self.config.step}")
            self._run_step(name, callback)
            return

        steps: List[Tuple[str, Callable[[], object]]] = []

        steps.append(step_registry["normalize"])
        steps.append(step_registry["ingest"])
        steps.append(step_registry["events"])
        steps.append(step_registry["labels"])

        steps.append(step_registry["features"])

        if not self.config.skip_validation:
            steps.append(step_registry["validate"])

        if self.config.make_embeddings:
            steps.append(step_registry["embeddings"])

        steps.append(step_registry["projection"])

        if self.config.run_hdbscan:
            steps.append(step_registry["hdbscan"])

        steps.append(step_registry["clusters"])

        steps.append(step_registry["patterns"])

        steps.append(step_registry["train"])

        steps.append(step_registry["score"])

        steps.append(step_registry["rules"])
        steps.append(step_registry["table"])
        steps.append(step_registry["reports"])

        if self.config.make_kpi:
            steps.append(step_registry["kpi"])

        if self.config.make_calibration:
            steps.append(step_registry["calibration"])

        if self.config.make_policy_metrics:
            steps.append(step_registry["policy"])

        for name, callback in steps:
            self._run_step(name, callback)

    def _run_step(self, name: str, callback: Callable[[], object]) -> None:
        try:
            callback()
        except Exception as exc:
            raise RuntimeError(f"Pipeline step '{name}' failed") from exc

    @log_timed("Normalize SAP data (Prompt 2)")
    def run_normalize_sap(self) -> None:
        """Execute src/data/normalize_sap.py to produce normalized/events.parquet."""
        output_path = self.project_root / "data" / "normalized" / "events.parquet"
        if output_path.exists():
            LOGGER.info(f"Skipping normalize: {output_path} already exists")
            return

        LOGGER.info("Running SAP normalization (Prompt 2)...")
        from src.data import normalize_sap

        normalize_sap.main([])

    @log_timed("Ingest raw datasets")
    def run_ingestion(self) -> None:
        output_path = self.project_root / CURATED_DIR / "line_items.parquet"
        if output_path.exists():
            LOGGER.info(f"Skipping ingest: {output_path} already exists")
            return

        argv: List[str] = []
        for pattern in self.config.input_patterns:
            argv.extend(["--input", pattern])
        if self.config.docmap_path is not None:
            argv.extend(["--docmap", str(self.config.docmap_path)])
        ingest_multi.run(argv if argv else None)

    @log_timed("Build clearing events")
    def run_build_events(self) -> None:
        line_items_path = self.project_root / CURATED_DIR / "line_items.parquet"
        if not line_items_path.exists():
            LOGGER.warning(f"Prerequisite missing: {line_items_path}. Run 'ingest' step first.")
            return

        events_path = self.project_root / CURATED_DIR / "events.parquet"
        if events_path.exists():
            LOGGER.info(f"Skipping events build: {events_path} already exists")
            return

        build_events.configure_logging()
        build_events.build_events_tables(self.project_root)

    @log_timed("Generate training labels")
    def run_make_labels(self) -> dict[str, int] | None:
        events_path = self.project_root / CURATED_DIR / "events.parquet"
        if not events_path.exists():
            LOGGER.warning(f"Prerequisite missing: {events_path}. Run 'events' step first.")
            return None

        output_path = self.project_root / CURATED_DIR / "train_pointwise.parquet"
        if output_path.exists():
            LOGGER.info(f"Skipping labels: {output_path} already exists")
            df = pd.read_parquet(output_path)
            return {"rows": len(df)}

        make_labels.configure_logging()
        labelled = make_labels.build_pointwise_labels(self.project_root)
        if labelled.empty:
            LOGGER.warning("No labelled samples generated; skipping parquet export.")
        else:
            make_labels.save_labels(labelled, self.project_root)
        make_labels.summarise_labels(labelled)
        return {"rows": len(labelled)}

    @log_timed("Engineer features & client profiles")
    def run_make_features(self) -> dict[str, int] | None:
        labels_path = self.project_root / CURATED_DIR / "train_pointwise.parquet"
        if not labels_path.exists():
            LOGGER.warning(f"Prerequisite missing: {labels_path}. Run 'labels' step first.")
            return None

        output_path = self.project_root / CURATED_DIR / "train_features.parquet"
        if output_path.exists():
            LOGGER.info(f"Skipping features: {output_path} already exists")
            df = pd.read_parquet(output_path)
            return {"rows": len(df)}

        make_features.configure_logging()
        datasets = make_features.load_curated_datasets(self.project_root)
        client_profiles = make_features.compute_client_profiles(
            datasets["events"], datasets["event_items"]
        ).fillna(0.0)
        feature_matrix = make_features.build_feature_matrix(
            datasets["train_pointwise"],
            datasets["events"],
            client_profiles,
        )
        if feature_matrix.empty:
            LOGGER.warning("Feature matrix is empty; outputs not refreshed.")
            return None
        make_features.save_feature_outputs(feature_matrix, client_profiles, self.project_root)
        make_features.run_random_forest_smoke_test(feature_matrix)
        return {"rows": len(feature_matrix)}

    @log_timed("Validate curated datasets")
    def run_pretraining_validations(self) -> None:
        curated_dir = self.project_root / CURATED_DIR
        events_path = curated_dir / "events.parquet"
        event_items_path = curated_dir / "event_items.parquet"
        features_path = curated_dir / "train_features.parquet"

        required_paths: Dict[str, Path] = {
            "events": events_path,
            "event_items": event_items_path,
            "train_features": features_path,
        }
        for label, path in required_paths.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Required curated artefact '{label}' not found at {path}. Run prior stages before training."
                )

        events_df = pd.read_parquet(events_path)
        if "n_righe_evento" not in events_df.columns:
            raise RuntimeError("Column 'n_righe_evento' missing from events.parquet")

        summary_rows: List[Dict[str, object]] = []
        failures: List[str] = []

        total_events = int(len(events_df))
        righe_series = pd.to_numeric(events_df["n_righe_evento"], errors="coerce").fillna(0)
        zero_rows = int((righe_series <= 0).sum())
        summary_rows.append(
            {
                "check": "events_n_righe_evento_zero",
                "total": total_events,
                "issues": zero_rows,
                "pct_issues": _pct(zero_rows, total_events),
                "note": "-",
            }
        )
        if zero_rows > 0:
            failures.append(f"{zero_rows} eventi con n_righe_evento <= 0")

        if "importo_evento" in events_df.columns:
            importo_series = pd.to_numeric(events_df["importo_evento"], errors="coerce")
            amount_total = int(importo_series.notna().sum())
            non_positive = int((importo_series.dropna() <= 0).sum())
            summary_rows.append(
                {
                    "check": "events_importo_evento_non_positive",
                    "total": amount_total,
                    "issues": non_positive,
                    "pct_issues": _pct(non_positive, amount_total),
                    "note": "-" if amount_total else "nessun importo disponibile",
                }
            )
            if non_positive > 0:
                failures.append(f"{non_positive} eventi con importo_evento <= 0")
        else:
            summary_rows.append(
                {
                    "check": "events_importo_evento_non_positive",
                    "total": 0,
                    "issues": 0,
                    "pct_issues": 0.0,
                    "note": "colonna importo_evento assente",
                }
            )

        event_items_df = pd.read_parquet(event_items_path)
        for column in ("data_documento", "data_pareggio"):
            if column not in event_items_df.columns:
                raise RuntimeError(f"Column '{column}' missing from event_items.parquet")

        doc_dates = pd.to_datetime(event_items_df["data_documento"], errors="coerce")
        pareggio_dates = pd.to_datetime(event_items_df["data_pareggio"], errors="coerce")
        comparable_mask = doc_dates.notna() & pareggio_dates.notna()
        comparable_total = int(comparable_mask.sum())
        violation_count = int((doc_dates.gt(pareggio_dates) & comparable_mask).sum())
        violation_pct = (violation_count / comparable_total) if comparable_total else 0.0
        summary_rows.append(
            {
                "check": "event_items_data_documento_after_pareggio",
                "total": comparable_total,
                "issues": violation_count,
                "pct_issues": _pct(violation_count, comparable_total),
                "note": f"soglia {_DATE_MISMATCH_THRESHOLD * 100:.1f}%",
            }
        )
        if comparable_total and violation_pct > _DATE_MISMATCH_THRESHOLD:
            failures.append(
                f"{violation_pct * 100:.2f}% di righe hanno data_documento > data_pareggio (soglia { _DATE_MISMATCH_THRESHOLD * 100:.1f}%)"
            )

        features_df = pd.read_parquet(features_path)
        missing_cols = [col for col in CRITICAL_FEATURE_COLUMNS if col not in features_df.columns]
        summary_rows.append(
            {
                "check": "train_features_missing_critical_columns",
                "total": len(CRITICAL_FEATURE_COLUMNS),
                "issues": len(missing_cols),
                "pct_issues": _pct(len(missing_cols), len(CRITICAL_FEATURE_COLUMNS)),
                "note": ", ".join(missing_cols) if missing_cols else "-",
            }
        )
        if missing_cols:
            failures.append("colonne critiche mancanti: " + ", ".join(missing_cols))

        summary_df = pd.DataFrame(summary_rows)
        LOGGER.info("Pre-training validation summary:\n%s", summary_df.to_string(index=False))

        if failures:
            LOGGER.error("Pre-training validation failed: %s", "; ".join(failures))
            LOGGER.error("Risolvi le incoerenze sopra prima di proseguire con il training.")
            raise SystemExit(1)

        LOGGER.info("Pre-training validation passed; proceeding to training stage.")

    @log_timed("Train model & update champion")
    def run_training(self) -> None:
        features_path = self.project_root / CURATED_DIR / "train_features.parquet"
        if not features_path.exists():
            LOGGER.warning(f"Prerequisite missing: {features_path}. Run 'features' step first.")
            return

        champion_path = self.project_root / MODELS_DIR / "champion.json"
        if champion_path.exists():
            LOGGER.info(f"Model already trained: {champion_path} exists. Use --force to retrain.")
            return

        argv = ["--model", self.config.model]
        train.run(argv)

    @log_timed("Compute pattern detectors (Prompt 5)")
    def run_pattern_detectors(self) -> None:
        """Execute src/analytics/pattern_detectors.py to generate client_patterns.parquet."""
        profiles_path = self.project_root / CURATED_DIR / "client_profiles.parquet"
        clusters_path = self.project_root / CURATED_DIR / "client_clusters.parquet"

        if not profiles_path.exists():
            LOGGER.warning(f"Prerequisite missing: {profiles_path}. Run 'features' step first.")
            return

        output_path = self.project_root / CURATED_DIR / "client_patterns.parquet"
        if output_path.exists():
            LOGGER.info(f"Pattern detectors already built: {output_path} exists")
            return

        LOGGER.info("Running pattern detectors (Prompt 5)...")
        from src.analytics import pattern_detectors

        pattern_detectors.main([])

    @log_timed("Train pointwise model (Prompt 6)")
    def run_pointwise_trainer(self) -> None:
        """Execute src/training/pointwise_trainer.py to train CatBoost/AutoGluon."""
        features_path = self.project_root / CURATED_DIR / "train_features.parquet"
        patterns_path = self.project_root / CURATED_DIR / "client_patterns.parquet"

        if not features_path.exists():
            LOGGER.warning(f"Prerequisite missing: {features_path}. Run 'features' step first.")
            return
        if not patterns_path.exists():
            LOGGER.warning(f"Prerequisite missing: {patterns_path}. Run 'patterns' step first.")
            return

        champion_path = self.project_root / MODELS_DIR / "champion.json"
        if champion_path.exists():
            LOGGER.info(f"Model already trained: {champion_path} exists")
            return

        LOGGER.info("Running pointwise trainer (Prompt 6)...")
        from src.training import pointwise_trainer

        model_arg = getattr(self.config, "model_type", "catboost")
        pointwise_trainer.main(["--model", model_arg])

    @log_timed("Score open invoices (Prompt 7)")
    def run_score_open_invoices(self) -> None:
        """Execute src/inference/score_open_invoices.py to generate predictions."""
        champion_path = self.project_root / MODELS_DIR / "champion.json"
        patterns_path = self.project_root / CURATED_DIR / "client_patterns.parquet"
        line_items_path = self.project_root / CURATED_DIR / "line_items.parquet"

        if not champion_path.exists():
            LOGGER.warning(f"Prerequisite missing: {champion_path}. Run 'train' step first.")
            return
        if not patterns_path.exists():
            LOGGER.warning(f"Prerequisite missing: {patterns_path}. Run 'patterns' step first.")
            return
        if not line_items_path.exists():
            LOGGER.warning(f"Prerequisite missing: {line_items_path}. Run 'ingest' step first.")
            return

        output_path = self.project_root / CURATED_DIR / "event_policy_predictions.parquet"
        if output_path.exists():
            LOGGER.info(f"Invoice scoring already completed: {output_path} exists")
            return

        LOGGER.info("Running invoice scoring (Prompt 7)...")
        from src.inference import score_open_invoices

        score_open_invoices.main([])

    @log_timed("Backtest policy rules")
    def run_rules_backtest(self) -> None:
        patterns_path = self.project_root / CURATED_DIR / "client_pattern_scores.parquet"
        if not patterns_path.exists():
            LOGGER.warning(f"Prerequisite missing: {patterns_path}. Run 'patterns' step first.")
            return

        output_path = self.project_root / CURATED_DIR / "client_rules_accuracy.parquet"
        if output_path.exists():
            LOGGER.info(f"Rules backtest already completed: {output_path} exists")
            return

        rules_backtest.configure_logging()
        rules_backtest.main()

    @log_timed("Assemble pattern table")
    def run_pattern_table(self) -> dict[str, int]:
        curated_dir = self.project_root / CURATED_DIR
        pattern_scores_path = curated_dir / "client_pattern_scores.parquet"
        rules_path = curated_dir / "client_rules_accuracy.parquet"
        cat_metrics_path = curated_dir / "client_model_metrics_catboost.parquet"
        auto_metrics_path = curated_dir / "client_model_metrics_autogluon.parquet"

        output_path = self.project_root / PATTERN_OUTPUT
        if output_path.exists():
            LOGGER.info(f"Pattern table already created: {output_path} exists")
            df = pd.read_parquet(output_path)
            return {"rows": len(df)}

        if not pattern_scores_path.exists():
            raise FileNotFoundError(
                f"Missing pattern scores parquet at {pattern_scores_path}. Run pattern_detectors first."
            )
        pattern_scores = pd.read_parquet(pattern_scores_path)

        if rules_path.exists():
            rules_metrics = pd.read_parquet(rules_path)
        else:
            LOGGER.warning("Rules accuracy parquet missing at %s; using empty placeholder.", rules_path)
            rules_metrics = pd.DataFrame(columns=["cliente", "rules_match_rate"])

        if cat_metrics_path.exists():
            cat_metrics = pd.read_parquet(cat_metrics_path)
        else:
            LOGGER.warning(
                "CatBoost client metrics not found at %s; pattern table will include NaNs for cat model columns.",
                cat_metrics_path,
            )
            cat_metrics = pd.DataFrame(columns=["cliente", "model_match_rate"])

        auto_metrics: Optional[pd.DataFrame]
        if auto_metrics_path.exists():
            auto_metrics = pd.read_parquet(auto_metrics_path)
        else:
            auto_metrics = None

        pattern_table_df = build_pattern_table(pattern_scores, rules_metrics, cat_metrics, auto_metrics)
        output_path = self.project_root / PATTERN_OUTPUT
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pattern_table_df.to_parquet(output_path, index=False)
        LOGGER.info("Saved pattern table to %s", output_path)
        return {"rows": len(pattern_table_df)}

    @log_timed("Render pattern card reports")
    def run_pattern_reports(self) -> dict[str, int]:
        table_path = self.project_root / PATTERN_OUTPUT
        if not table_path.exists():
            LOGGER.warning(f"Prerequisite missing: {table_path}. Run 'table' step first.")
            return {"rows": 0}

        reports_dir = self.project_root / REPORTS_DIR
        existing_reports = list(reports_dir.glob("pattern_*.png"))
        if existing_reports:
            LOGGER.info(f"Pattern reports already generated: {len(existing_reports)} files in {reports_dir}")
            return {"rows": len(existing_reports)}

        make_report.configure_logging()
        generated = make_report.generate_reports()
        LOGGER.info("Generated %s pattern card report(s).", len(generated))
        return {"rows": len(generated)}

    @log_timed("Create client embeddings")
    def run_make_embeddings(self) -> None:
        features_path = self.project_root / CURATED_DIR / "train_features.parquet"
        if not features_path.exists():
            LOGGER.warning(f"Prerequisite missing: {features_path}. Run 'features' step first.")
            return

        output_path = self.project_root / CURATED_DIR / "client_embeddings.parquet"
        if output_path.exists():
            LOGGER.info(f"Client embeddings already created: {output_path} exists")
            return

        argv = []
        make_client_embeddings.configure_logging()
        make_client_embeddings.main(argv)

    @log_timed("Compute embedding projection")
    def run_embedding_projection(self) -> None:
        embeddings_path = self.project_root / CURATED_DIR / "client_embeddings.parquet"
        if not embeddings_path.exists():
            LOGGER.info("Skipping embedding projection; %s not found.", embeddings_path)
            return

        output_path = self.project_root / CURATED_DIR / "client_embedding_2d.parquet"
        if output_path.exists():
            LOGGER.info(f"Embedding projection already computed: {output_path} exists")
            return

        argv = [
            "--embeddings",
            str(embeddings_path),
            "--out-dir",
            str(self.project_root / CURATED_DIR),
            "--neighbors",
            str(self.config.embedding_neighbors),
        ]
        if self.config.use_umap:
            argv.append("--umap")
        embedding_viz.configure_logging()
        embedding_viz.main(argv)

    @log_timed("Cluster embeddings with HDBSCAN")
    def run_hdbscan(self) -> None:
        embeddings_path = self.project_root / CURATED_DIR / "client_embeddings.parquet"
        if not embeddings_path.exists():
            LOGGER.info("Skipping HDBSCAN: embeddings parquet not found at %s", embeddings_path)
            return

        output_path = self.project_root / CURATED_DIR / "client_clusters.parquet"
        if output_path.exists():
            LOGGER.info(f"HDBSCAN clustering already computed: {output_path} exists")
            return

        argv = [
            "--embeddings",
            str(embeddings_path),
            "--out-dir",
            str(self.project_root / CURATED_DIR),
            "--min-cluster-size",
            str(self.config.min_cluster_size),
            "--min-samples",
            str(self.config.min_samples),
        ]
        hdbscan_module.configure_logging()
        hdbscan_module.main(argv)

    @log_timed("Summarise clusters")
    def run_cluster_profiles(self) -> None:
        clusters_path = self.project_root / CURATED_DIR / "client_clusters.parquet"
        if not clusters_path.exists():
            LOGGER.info("Skipping cluster summary; %s not available.", clusters_path)
            return

        output_path = self.project_root / CURATED_DIR / "cluster_summary.parquet"
        if output_path.exists():
            LOGGER.info(f"Cluster profiles already computed: {output_path} exists")
            return

        cluster_profile.configure_logging()
        cluster_profile.main()

    @log_timed("Compute ranking KPIs")
    def run_kpi_ranking(self) -> None:
        champion_path = self.project_root / MODELS_DIR / "champion.json"
        if not champion_path.exists():
            LOGGER.warning(f"Prerequisite missing: {champion_path}. Run 'train' step first.")
            return

        output_path = self.project_root / CURATED_DIR / "kpi_ranking.parquet"
        if output_path.exists():
            LOGGER.info(f"KPI ranking already computed: {output_path} exists")
            return

        from src.analytics import kpi_ranking

        kpi_ranking.main()

    @log_timed("Compute calibration diagnostics")
    def run_calibration(self) -> None:
        champion_path = self.project_root / MODELS_DIR / "champion.json"
        if not champion_path.exists():
            LOGGER.warning(f"Prerequisite missing: {champion_path}. Run 'train' step first.")
            return

        output_path = self.project_root / CURATED_DIR / "calibration_summary.parquet"
        if output_path.exists():
            LOGGER.info(f"Calibration diagnostics already computed: {output_path} exists")
            return

        from src.analytics import calibration

        calibration.main()

    @log_timed("Compute policy classifier metrics")
    def run_policy_metrics(self) -> None:
        events_path = self.project_root / CURATED_DIR / "events.parquet"
        if not events_path.exists():
            LOGGER.warning(f"Prerequisite missing: {events_path}. Run 'events' step first.")
            return

        output_path = self.project_root / CURATED_DIR / "event_policy_metrics.parquet"
        if output_path.exists():
            LOGGER.info(f"Policy metrics already computed: {output_path} exists")
            return

        from src.analytics import event_policy_metrics

        event_policy_metrics.main()

def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    runner = PipelineRunner(config)
    runner.run()

if __name__ == "__main__":
    main()

