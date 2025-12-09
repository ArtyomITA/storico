"""Ingestion script for customer payment line items.

This module loads raw CSV and Excel files from ``data/raw`` that follow the
expected SAP schema, normalises the data types and column names, logs basic
statistics, and writes the curated dataset to ``data/curated/line_items.parquet``
with snappy compression.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)

COL_MAP = {
    "Società": "societa",
    "Cliente": "cliente",
    "Data di pareggio": "data_pareggio",
    "Doc. pareggio": "doc_pareggio",
    "Esercizio": "esercizio",
    "Numero documento": "numero_documento",
    "Posizione": "posizione",
    "Data registrazione": "data_registrazione",
    "Data documento": "data_documento",
    "Data di acquisizione": "data_acquisizione",
    "Divisa": "divisa",
    "Tipo di documento": "tipo_documento",
    "Periodo contabile": "periodo_contabile",
    "Chiave contabile": "chiave_contabile",
    "Imp. div. int.": "imp_div_int",
    "Importo": "importo_raw",
    "Divisa funzionale": "divisa_fun",
    "Data base pagamento": "data_base_pagamento",
    "Cond. pagamento": "cond_pagamento",
}

RAW_COLUMNS: List[str] = [
    "Cliente",
    "Data di pareggio",
    "Doc. pareggio",
    "Esercizio",
    "Numero documento",
    "Imp. div. int.",
    "Data registrazione",
    "Data documento",
    "Data di acquisizione",
    "Divisa",
    "Tipo di documento",
    "Periodo contabile",
    "Chiave contabile",
]

RENAME_MAP = {
    "Cliente": "cliente",
    "Data di pareggio": "data_pareggio",
    "Doc. pareggio": "doc_pareggio",
    "Esercizio": "esercizio",
    "Numero documento": "numero_documento",
    "Imp. div. int.": "imp_div_int",
    "Data registrazione": "data_registrazione",
    "Data documento": "data_documento",
    "Data di acquisizione": "data_acquisizione",
    "Divisa": "divisa",
    "Tipo di documento": "tipo_documento",
    "Periodo contabile": "periodo_contabile",
    "Chiave contabile": "chiave_contabile",
}

DATE_COLUMNS = [
    "data_pareggio",
    "data_registrazione",
    "data_documento",
    "data_acquisizione",
    "data_base_pagamento",
]

AMOUNT_COLUMNS = ["imp_div_int", "importo_raw"]

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}

_NULL_TOKENS = {"", "nan", "NaN", "None", "<NA>", "NaT"}

def _clean_string_series(series: pd.Series) -> pd.Series:
    """Normalise textual columns by trimming and dropping placeholder tokens.
    
    - Strips leading/trailing whitespace
    - Replaces null tokens (empty, 'nan', etc.) with pd.NA
    - Removes trailing .0 from document IDs (e.g., 'INV123.0' → 'INV123', '12345.0' → '12345')
    """
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace({token: pd.NA for token in _NULL_TOKENS})
    cleaned = cleaned.str.replace(r"(\.0+)$", "", regex=True)
    return cleaned

def _parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse heterogeneous date representations into timezone-naive datetimes.
    
    Supports:
    - DD/MM/YYYY format (day-first, Italian convention)
    - ISO 8601 formats (YYYY-MM-DD)
    - Excel serial dates (numeric)
    
    PROMPT 1 requirement: parse dates day-first (dayfirst=True).
    """
    cleaned = _clean_string_series(series)
    parsed = pd.to_datetime(cleaned, format="%d/%m/%Y", errors="coerce", dayfirst=True)

    remaining_mask = parsed.isna() & cleaned.notna()
    if remaining_mask.any():
        fallback = pd.to_datetime(cleaned[remaining_mask], errors="coerce", dayfirst=True)
        parsed.loc[remaining_mask] = fallback

    numeric_values = pd.to_numeric(cleaned, errors="coerce")
    if numeric_values.notna().any():
        excel_dates = pd.to_datetime(
            numeric_values,
            unit="d",
            origin="1899-12-30",
            errors="coerce",
        )
        parsed = parsed.combine_first(excel_dates)

    return parsed

def _parse_amount_series(series: pd.Series) -> pd.Series:
    """Normalise amount column to floating point values.
    
    Handles:
    - European decimal comma format (1.234,56 → 1234.56)
    - Non-breaking spaces and regular spaces as thousand separators
    - Negative amounts with minus sign
    
    PROMPT 1 requirement: uniforma virgole/decimali.
    """
    cleaned = (
        series.astype("string")
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    cleaned = cleaned.replace({token: pd.NA for token in _NULL_TOKENS})
    return pd.to_numeric(cleaned, errors="coerce")

def configure_logging(level: int | str | None = logging.INFO) -> Path:
    """Configure shared logging handlers for ingestion steps."""

    return setup_logging(level)

def collect_raw_files(raw_dir: Path) -> List[Path]:
    """Return the list of raw data files to ingest."""
    files = [
        path
        for path in sorted(raw_dir.iterdir())
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file()
    ]
    return files

def read_raw_file(path: Path, strict: bool = True) -> pd.DataFrame:
    """Load a single raw CSV or Excel file as a DataFrame of strings.
    
    Args:
        path: Path to CSV or XLSX file
        strict: If True (default), enforce presence of RAW_COLUMNS (backward compat).
                If False, accept any columns present in COL_MAP.
    
    Returns:
        DataFrame with string dtype for all columns.
    """
    LOGGER.info("Reading file: %s", path.name)

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )
    else:
        frame = pd.read_excel(
            path,
            dtype=str,
            engine="openpyxl",
        )
        frame = frame.fillna("")

    if strict:
        missing_columns = [column for column in RAW_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"File {path} is missing required columns: {', '.join(missing_columns)}"
            )
        frame = frame.loc[:, RAW_COLUMNS].copy()
    else:

        available_mapped = [col for col in frame.columns if col in COL_MAP]
        extra_columns = [col for col in frame.columns if col not in COL_MAP]

        if extra_columns:
            LOGGER.info(
                "File %s contains %d extra columns not in COL_MAP (will be dropped): %s",
                path.name,
                len(extra_columns),
                ", ".join(extra_columns[:5]) + ("..." if len(extra_columns) > 5 else ""),
            )

        if not available_mapped:
            raise ValueError(
                f"File {path} contains no recognized columns from COL_MAP. "
                f"Available: {', '.join(frame.columns[:10])}"
            )

        frame = frame.loc[:, available_mapped].copy()

    return frame

def normalise_dataframe(frame: pd.DataFrame, use_extended_map: bool = False) -> pd.DataFrame:
    """Apply schema normalisation and type conversions.
    
    Args:
        frame: Raw DataFrame with Italian column names
        use_extended_map: If True, use COL_MAP .
                          If False, use RENAME_MAP (backward compatible).
    
    Returns:
        Normalized DataFrame with snake_case columns, typed correctly.
    """
    col_map = COL_MAP if use_extended_map else RENAME_MAP

    columns_to_rename = {k: v for k, v in col_map.items() if k in frame.columns}
    normalised = frame.rename(columns=columns_to_rename)

    if use_extended_map:
        expected_columns = set(col_map.values())
        for col in expected_columns:
            if col not in normalised.columns:
                normalised[col] = pd.NA
                LOGGER.debug("Added missing column '%s' as NaN", col)

    string_columns = [
        column
        for column in normalised.columns
        if column not in DATE_COLUMNS + AMOUNT_COLUMNS
    ]
    for column in string_columns:
        normalised[column] = _clean_string_series(normalised[column])

    for column in DATE_COLUMNS:
        if column in normalised.columns:
            normalised[column] = _parse_datetime_series(normalised[column])

    for column in AMOUNT_COLUMNS:
        if column in normalised.columns:
            normalised[column] = _parse_amount_series(normalised[column])

    return normalised

def log_dataset_summary(dataset: pd.DataFrame) -> None:
    """Emit dataset summary statistics to the console."""
    row_count = len(dataset)
    LOGGER.info("Total rows: %s", row_count)

    if row_count == 0:
        LOGGER.warning("Dataset is empty after ingestion.")
        return

    date_series = dataset["data_documento"].dropna()
    if not date_series.empty:
        LOGGER.info(
            "Data documento range: %s → %s",
            date_series.min().date(),
            date_series.max().date(),
        )
    else:
        LOGGER.info("Data documento range: no valid dates")

    tipo_counts = dataset["tipo_documento"].fillna("<NA>").value_counts()
    LOGGER.info("Records per tipo_documento: %s", tipo_counts.to_dict())

    chiave_counts = dataset["chiave_contabile"].fillna("<NA>").value_counts()
    LOGGER.info("Records per chiave_contabile: %s", chiave_counts.to_dict())

def log_quality_report(dataset: pd.DataFrame, original_row_count: int) -> None:
    """Emit comprehensive quality report .
    
    Reports:
    - Unique cliente count
    - Date range (data_documento, data_pareggio)
    - Percentage of rows discarded (if any filtering applied)
    - Missing value percentages for key columns
    """
    row_count = len(dataset)
    LOGGER.info("=" * 60)
    LOGGER.info("QUALITY REPORT")
    LOGGER.info("=" * 60)

    LOGGER.info("Rows processed: %d", original_row_count)
    LOGGER.info("Rows retained: %d", row_count)
    if original_row_count > row_count:
        pct_discarded = 100.0 * (original_row_count - row_count) / original_row_count
        LOGGER.info("Rows discarded: %d (%.2f%%)", original_row_count - row_count, pct_discarded)
    else:
        LOGGER.info("Rows discarded: 0 (0.00%%)")

    if "cliente" in dataset.columns:
        unique_clients = dataset["cliente"].dropna().nunique()
        LOGGER.info("Unique clients: %d", unique_clients)

    if "data_documento" in dataset.columns:
        date_series = dataset["data_documento"].dropna()
        if not date_series.empty:
            LOGGER.info(
                "Data documento range: %s → %s (%d days)",
                date_series.min().date(),
                date_series.max().date(),
                (date_series.max() - date_series.min()).days,
            )
        else:
            LOGGER.warning("Data documento: no valid dates")

    if "data_pareggio" in dataset.columns:
        pareggio_series = dataset["data_pareggio"].dropna()
        if not pareggio_series.empty:
            LOGGER.info(
                "Data pareggio range: %s → %s",
                pareggio_series.min().date(),
                pareggio_series.max().date(),
            )
        else:
            LOGGER.info("Data pareggio: no valid dates (open invoices only)")

    key_columns = ["cliente", "numero_documento", "data_documento", "imp_div_int"]
    LOGGER.info("Missing value percentages:")
    for col in key_columns:
        if col in dataset.columns:
            missing_count = dataset[col].isna().sum()
            pct_missing = 100.0 * missing_count / row_count if row_count > 0 else 0.0
            LOGGER.info("  %s: %d (%.2f%%)", col, missing_count, pct_missing)

    if "tipo_documento" in dataset.columns:
        tipo_counts = dataset["tipo_documento"].fillna("<NA>").value_counts()
        LOGGER.info("Top 5 tipo_documento: %s", dict(tipo_counts.head(5)))

    if "divisa" in dataset.columns:
        divisa_counts = dataset["divisa"].fillna("<NA>").value_counts()
        LOGGER.info("Currency distribution: %s", dict(divisa_counts.head(5)))

    LOGGER.info("=" * 60)

def ingest() -> None:
    """Main entry point for the ingestion pipeline (backward compatible)."""
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    curated_dir = project_root / "data" / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}. Expected CSV or XLSX files."
        )

    files = collect_raw_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No raw files found in {raw_dir}. Expected *.csv or *.xlsx files."
        )

    frames: List[pd.DataFrame] = []
    for path in files:
        frame = read_raw_file(path, strict=True)
        normalised = normalise_dataframe(frame, use_extended_map=False)
        frames.append(normalised)

    dataset = pd.concat(frames, ignore_index=True)
    log_dataset_summary(dataset)

    output_path = curated_dir / "line_items.parquet"
    dataset.to_parquet(output_path, compression="snappy", index=False)
    LOGGER.info("Wrote curated dataset to %s", output_path)

def ingest_normalize(
    raw_dir: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """PROMPT 1 ingestion with extended COL_MAP and quality reporting.
    
    This is the new recommended entry point that uses the full COL_MAP mapping
    and generates comprehensive quality reports.
    
    Args:
        raw_dir: Directory containing CSV/XLSX files (defaults to data/raw)
        output_path: Destination parquet file (defaults to data/curated/line_items.parquet)
    
    Returns:
        Normalized DataFrame with snake_case columns from COL_MAP.
    """
    project_root = Path(__file__).resolve().parents[2]

    if raw_dir is None:
        raw_dir = project_root / "data" / "raw"

    if output_path is None:
        output_path = project_root / "data" / "curated" / "line_items.parquet"

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}. Expected CSV or XLSX files."
        )

    files = collect_raw_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No raw files found in {raw_dir}. Expected *.csv or *.xlsx files."
        )

    LOGGER.info("Processing %d files from %s", len(files), raw_dir)

    frames: List[pd.DataFrame] = []
    original_row_count = 0

    for path in files:
        frame = read_raw_file(path, strict=False)
        original_row_count += len(frame)

        normalised = normalise_dataframe(frame, use_extended_map=True)
        frames.append(normalised)

        LOGGER.info("  %s: %d rows", path.name, len(normalised))

    dataset = pd.concat(frames, ignore_index=True)

    expected_columns = list(COL_MAP.values())
    available_columns = [col for col in expected_columns if col in dataset.columns]
    dataset = dataset.loc[:, available_columns]

    log_quality_report(dataset, original_row_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, compression="snappy", index=False)
    LOGGER.info("Wrote normalized dataset to %s", output_path)
    LOGGER.info("Output schema: %d columns, %d rows", len(dataset.columns), len(dataset))

    return dataset

def main() -> None:
    """Run the ingestion pipeline with logging configured."""
    configure_logging()
    ingest()

if __name__ == "__main__":
    main()

