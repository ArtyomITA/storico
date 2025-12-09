"""CLI utility to ingest multiple SAP export files into curated parquet datasets.

This script extends the single-source ingestion pipeline with support for mixing
multiple CSV/XLSX files and optionally producing a document-level mapping table.
It reuses the normalisation helpers from ``src.data.ingest`` to guarantee
consistency across ingestion workflows.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

from src.data.ingest import (
    collect_raw_files,
    configure_logging,
    log_dataset_summary,
    log_quality_report,
    normalise_dataframe,
    read_raw_file,
    COL_MAP,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/curated/line_items.parquet")
DEFAULT_DEDUP_KEYS: Sequence[str] = (
    "cliente",
    "numero_documento",
    "doc_pareggio",
    "data_documento",
)

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the ingestion command."""
    parser = argparse.ArgumentParser(
        description=(
            "Ingest multiple SAP estratti contabili (CSV/XLSX) into the curated line "
            "items parquet and optionally derive a document mapping parquet."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        dest="input_patterns",
        help=(
            "File path or glob pattern for raw CSV/XLSX exports. Repeat the flag "
            "to supply multiple patterns. Defaults to data/raw/* if omitted."
        ),
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Destination parquet for the curated line items dataset. Defaults to "
            "data/curated/line_items.parquet."
        ),
    )
    parser.add_argument(
        "--docmap",
        dest="docmap_path",
        type=Path,
        help="Optional destination parquet for the document map summary.",
    )
    parser.add_argument(
        "--dedupe-on",
        dest="dedupe_keys",
        nargs="+",
        default=list(DEFAULT_DEDUP_KEYS),
        help=(
            "Column subset to enforce deduplication on after concatenating all "
            "inputs. Defaults to cliente numero_documento doc_pareggio "
            "data_documento."
        ),
    )
    parser.add_argument(
        "--keep-first",
        action="store_true",
        default=False,
        help=(
            "If provided, retains the first occurrence when dropping duplicates. "
            "Otherwise the last occurrence is preserved."
        ),
    )
    parser.add_argument(
        "--use-extended-map",
        action="store_true",
        default=False,
        help=(
            "Use extended COL_MAP (19 columns) instead of legacy "
            "RENAME_MAP (13 columns). Enables support for Società, Posizione, "
            "Importo, Divisa funzionale, Data base pagamento, Cond. pagamento."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help=(
            "Enforce presence of all RAW_COLUMNS (backward compatible mode). "
            "If omitted, accepts any columns present in the selected mapping."
        ),
    )

    return parser.parse_args(argv)

def _expand_pattern(pattern: str, project_root: Path) -> List[Path]:
    """Expand a single filesystem pattern into matching files."""
    candidate = Path(pattern)
    if not candidate.is_absolute():
        candidate = project_root / candidate

    if candidate.is_dir():
        files = list(candidate.glob("**/*"))
    else:
        files = list(candidate.parent.glob(candidate.name)) if candidate.parent.exists() else []

    paths = [path for path in files if path.suffix.lower() in {".csv", ".xlsx"} and path.is_file()]
    return sorted(set(paths))

def resolve_input_files(
    input_patterns: Iterable[str] | None,
    project_root: Path,
) -> List[Path]:
    """Resolve CLI-supplied patterns to concrete files, defaulting to data/raw."""
    if input_patterns:
        files: List[Path] = []
        for pattern in input_patterns:
            matches = _expand_pattern(pattern, project_root)
            if not matches:
                LOGGER.warning("Pattern %s did not match any CSV/XLSX files", pattern)
            files.extend(matches)
        unique_files = sorted(set(files))
        if not unique_files:
            raise FileNotFoundError(
                "No files matched the provided --input patterns."
            )
        return unique_files

    raw_dir = project_root / "data" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}. Provide --input patterns instead."
        )

    files = collect_raw_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No raw files found in {raw_dir}. Provide --input patterns instead."
        )
    return files

def load_and_normalise(
    files: Sequence[Path],
    use_extended_map: bool = False,
    strict: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Load, concatenate, and normalise raw SAP exports.
    
    Args:
        files: List of CSV/XLSX files to process
        use_extended_map: If True, use COL_MAP (PROMPT 1). If False, use RENAME_MAP (legacy).
        strict: If True, enforce presence of RAW_COLUMNS. If False, accept any COL_MAP columns.
    
    Returns:
        Tuple of (normalized_dataframe, original_row_count)
    """
    frames: List[pd.DataFrame] = []
    original_row_count = 0

    for path in files:
        frame = read_raw_file(path, strict=strict)
        original_row_count += len(frame)
        normalised = normalise_dataframe(frame, use_extended_map=use_extended_map)
        frames.append(normalised)
        LOGGER.info("  %s: %d rows", path.name, len(normalised))

    dataset = pd.concat(frames, ignore_index=True)

    if use_extended_map:
        expected_columns = list(COL_MAP.values())
        available_columns = [col for col in expected_columns if col in dataset.columns]
        dataset = dataset.loc[:, available_columns]

    return dataset, original_row_count

def deduplicate_dataset(
    dataset: pd.DataFrame,
    subset: Sequence[str],
    keep_first: bool = False,
) -> pd.DataFrame:
    """Drop duplicate rows based on the provided subset of columns."""
    missing = [column for column in subset if column not in dataset.columns]
    if missing:
        raise KeyError(
            f"Cannot deduplicate: dataset missing columns {', '.join(missing)}"
        )

    keep = "first" if keep_first else "last"
    deduped = dataset.drop_duplicates(subset=list(subset), keep=keep)
    if len(deduped) != len(dataset):
        LOGGER.info(
            "Deduplicated dataset: removed %s duplicate rows (keeping %s occurrence)",
            len(dataset) - len(deduped),
            keep,
        )
    return deduped

def build_doc_map(dataset: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a document-level view from line items for quick lookup."""
    if dataset.empty:
        return pd.DataFrame(
            columns=[
                "cliente",
                "numero_documento",
                "record_count",
                "importo_totale",
                "data_documento_min",
                "data_documento_max",
                "data_pareggio_min",
                "data_pareggio_max",
                "divisa",
                "tipo_documento",
                "doc_pareggio_primo",
            ]
        )

    sorted_dataset = dataset.sort_values(
        ["cliente", "numero_documento", "data_documento"],
        kind="mergesort",
    )

    aggregations = {
        "imp_div_int": "sum",
        "data_documento": ["min", "max"],
        "data_pareggio": ["min", "max"],
        "divisa": "first",
        "tipo_documento": "first",
        "doc_pareggio": "first",
    }

    grouped = sorted_dataset.groupby(
        ["cliente", "numero_documento"],
        dropna=False,
        observed=True,
    )

    doc_map = grouped.agg(aggregations)
    doc_map.columns = [
        "importo_totale",
        "data_documento_min",
        "data_documento_max",
        "data_pareggio_min",
        "data_pareggio_max",
        "divisa",
        "tipo_documento",
        "doc_pareggio_primo",
    ]
    doc_map = doc_map.reset_index()
    doc_map["record_count"] = grouped.size().values

    ordered_columns = [
        "cliente",
        "numero_documento",
        "record_count",
        "importo_totale",
        "data_documento_min",
        "data_documento_max",
        "data_pareggio_min",
        "data_pareggio_max",
        "divisa",
        "tipo_documento",
        "doc_pareggio_primo",
    ]
    doc_map = doc_map.loc[:, ordered_columns]
    doc_map["importo_totale"] = doc_map["importo_totale"].fillna(0.0)

    return doc_map

def ensure_parent(path: Path) -> None:
    """Ensure the parent directory of a path exists."""
    path.parent.mkdir(parents=True, exist_ok=True)

def run(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for programmatic execution (testable main)."""
    configure_logging()
    args = parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    files = resolve_input_files(args.input_patterns, project_root)
    LOGGER.info("Resolved %s input files", len(files))

    dataset, original_row_count = load_and_normalise(
        files,
        use_extended_map=args.use_extended_map,
        strict=args.strict,
    )
    dataset = deduplicate_dataset(dataset, args.dedupe_keys, keep_first=args.keep_first)

    if args.use_extended_map:
        log_quality_report(dataset, original_row_count)
    else:
        log_dataset_summary(dataset)

    output_path = args.output_path
    if not output_path.is_absolute():
        output_path = project_root / output_path
    ensure_parent(output_path)
    dataset.to_parquet(output_path, compression="snappy", index=False)
    LOGGER.info("Wrote curated line items to %s", output_path)
    LOGGER.info("Output schema: %d columns, %d rows", len(dataset.columns), len(dataset))

    if args.docmap_path:
        docmap_path = args.docmap_path
        if not docmap_path.is_absolute():
            docmap_path = project_root / docmap_path
        ensure_parent(docmap_path)

        doc_map = build_doc_map(dataset)
        doc_map.to_parquet(docmap_path, compression="snappy", index=False)
        LOGGER.info("Wrote document map to %s", docmap_path)

if __name__ == "__main__":
    run()

