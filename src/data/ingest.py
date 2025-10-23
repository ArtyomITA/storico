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
]

AMOUNT_COLUMN = "imp_div_int"

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}

_NULL_TOKENS = {"", "nan", "NaN", "None", "<NA>", "NaT"}


def _clean_string_series(series: pd.Series) -> pd.Series:
    """Normalise textual columns by trimming and dropping placeholder tokens."""
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace({token: pd.NA for token in _NULL_TOKENS})
    cleaned = cleaned.str.replace(r"^(-?\d+)(?:\.0+)$", r"\1", regex=True)
    return cleaned


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse heterogeneous date representations into timezone-naive datetimes."""
    cleaned = _clean_string_series(series)
    parsed = pd.to_datetime(cleaned, format="%d/%m/%Y", errors="coerce")

    remaining_mask = parsed.isna() & cleaned.notna()
    if remaining_mask.any():
        fallback = pd.to_datetime(cleaned[remaining_mask], errors="coerce")
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
    """Normalise amount column to floating point values."""
    cleaned = (
        series.astype("string")
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
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


def read_raw_file(path: Path) -> pd.DataFrame:
    """Load a single raw CSV or Excel file as a DataFrame of strings."""
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

    missing_columns = [column for column in RAW_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"File {path} is missing required columns: {', '.join(missing_columns)}"
        )

    # Keep only the expected columns in the desired order.
    frame = frame.loc[:, RAW_COLUMNS].copy()
    return frame


def normalise_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply schema normalisation and type conversions."""
    normalised = frame.rename(columns=RENAME_MAP)

    # Trim string columns and replace empty strings with NA for all non-date columns.
    string_columns = [
        column
        for column in normalised.columns
        if column not in DATE_COLUMNS + [AMOUNT_COLUMN]
    ]
    for column in string_columns:
        normalised[column] = _clean_string_series(normalised[column])

    for column in DATE_COLUMNS:
        normalised[column] = _parse_datetime_series(normalised[column])

    normalised[AMOUNT_COLUMN] = _parse_amount_series(normalised[AMOUNT_COLUMN])

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


def ingest() -> None:
    """Main entry point for the ingestion pipeline."""
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
        frame = read_raw_file(path)
        normalised = normalise_dataframe(frame)
        frames.append(normalised)

    dataset = pd.concat(frames, ignore_index=True)
    log_dataset_summary(dataset)

    output_path = curated_dir / "line_items.parquet"
    dataset.to_parquet(output_path, compression="snappy", index=False)
    LOGGER.info("Wrote curated dataset to %s", output_path)


def main() -> None:
    """Run the ingestion pipeline with logging configured."""
    configure_logging()
    ingest()


if __name__ == "__main__":
    main()
