"""Unified command to execute the full payment-pattern analytics pipeline."""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.analytics import (
    cluster_profile,
    embedding_viz,
    make_client_embeddings,
    pattern_detectors,
    rules_backtest,
)
from src.analytics import run_hdbscan as hdbscan_module
from src.analytics.make_pattern_table import (
    OUTPUT_PATH as PATTERN_OUTPUT,
    build_pattern_table,
)
from src.data import ingest_multi
from src.features import build_events, make_features, make_labels
from src.models import train
from src.report import make_report
from src.utils.logging_setup import (
    configure_logging as setup_logging,
    get_current_log_file,
    log_timed,
)

LOGGER = logging.getLogger(__name__)
CURATED_DIR = Path("data/curated")
MODELS_DIR = Path("models")
CRITICAL_FEATURE_COLUMNS: Tuple[str, ...] = (
    "cliente",
    "event_key",
    "numero_documento",
    "label",
    "age_days",
    "lag_reg_vs_doc",
    "same_period",
    "amt_ratio_B",
    "amt_share_rank_event",
    "docnum_delta_prev",
    "weekly_cadence_client",
    "eom_cadence_client",
)
_DATE_MISMATCH_THRESHOLD = 0.05


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 2)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration derived from CLI options."""

    input_patterns: Tuple[str, ...]
    docmap_path: Optional[Path]
    model: str
    make_embeddings: bool
    run_hdbscan: bool
    embedding_neighbors: int
    use_umap: bool
    min_cluster_size: int
    min_samples: int


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
    )


class PipelineRunner:
    """Execute pipeline steps sequentially with structured logging."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.project_root = Path(__file__).resolve().parents[2]

    def run(self) -> None:
        steps: List[Tuple[str, Callable[[], object]]] = [
            ("Ingest raw datasets", self.run_ingestion),
            ("Build clearing events", self.run_build_events),
            ("Generate training labels", self.run_make_labels),
            ("Engineer features & client profiles", self.run_make_features),
            ("Validate curated datasets", self.run_pretraining_validations),
            ("Train model & update champion", self.run_training),
            ("Compute pattern detectors", self.run_pattern_detectors),
            ("Backtest policy rules", self.run_rules_backtest),
            ("Assemble pattern table", self.run_pattern_table),
            ("Render pattern card reports", self.run_pattern_reports),
        ]

        if self.config.make_embeddings:
            steps.append(("Create client embeddings", self.run_make_embeddings))

        steps.append(("Compute embedding projection", self.run_embedding_projection))

        if self.config.run_hdbscan:
            steps.append(("Cluster embeddings with HDBSCAN", self.run_hdbscan))

        steps.append(("Summarise clusters", self.run_cluster_profiles))

        for name, callback in steps:
            self._run_step(name, callback)

    def _run_step(self, name: str, callback: Callable[[], object]) -> None:
        try:
            callback()
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Pipeline step '{name}' failed") from exc

    # --- core steps -----------------------------------------------------------------

    @log_timed("Ingest raw datasets")
    def run_ingestion(self) -> None:
        argv: List[str] = []
        for pattern in self.config.input_patterns:
            argv.extend(["--input", pattern])
        if self.config.docmap_path is not None:
            argv.extend(["--docmap", str(self.config.docmap_path)])
        ingest_multi.run(argv if argv else None)

    @log_timed("Build clearing events")
    def run_build_events(self) -> None:
        build_events.configure_logging()
        build_events.build_events_tables(self.project_root)

    @log_timed("Generate training labels")
    def run_make_labels(self) -> dict[str, int] | None:
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
            return
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
        argv = ["--model", self.config.model]
        train.run(argv)

    @log_timed("Compute pattern detectors")
    def run_pattern_detectors(self) -> None:
        pattern_detectors.configure_logging()
        pattern_detectors.main()

    @log_timed("Backtest policy rules")
    def run_rules_backtest(self) -> None:
        rules_backtest.configure_logging()
        rules_backtest.main()

    @log_timed("Assemble pattern table")
    def run_pattern_table(self) -> dict[str, int]:
        curated_dir = self.project_root / CURATED_DIR
        pattern_scores_path = curated_dir / "client_pattern_scores.parquet"
        rules_path = curated_dir / "client_rules_accuracy.parquet"
        cat_metrics_path = curated_dir / "client_model_metrics_catboost.parquet"
        auto_metrics_path = curated_dir / "client_model_metrics_autogluon.parquet"

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
        make_report.configure_logging()
        generated = make_report.generate_reports()
        LOGGER.info("Generated %s pattern card report(s).", len(generated))
        return {"rows": len(generated)}

    # --- optional analytics ---------------------------------------------------------

    @log_timed("Create client embeddings")
    def run_make_embeddings(self) -> None:
        argv = []
        make_client_embeddings.configure_logging()
        make_client_embeddings.main(argv)

    @log_timed("Compute embedding projection")
    def run_embedding_projection(self) -> None:
        embeddings_path = self.project_root / CURATED_DIR / "client_embeddings.parquet"
        if not embeddings_path.exists():
            LOGGER.info("Skipping embedding projection; %s not found.", embeddings_path)
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
        cluster_profile.configure_logging()
        cluster_profile.main()


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    runner = PipelineRunner(config)
    runner.run()


if __name__ == "__main__":  # pragma: no cover
    main()
