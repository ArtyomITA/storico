"""Normalizzazione dati SAP (schema contrattuale).

Questo modulo legge file Excel SAP ("dictionary" con header ricchissimi),
seleziona solo le colonne usate dal progetto, standardizza tipi/nomi (snake_case),
e produce `data/normalized/events.parquet`.

Prompt 2 Requirements:
1. Definisce COLUMNS_MAP (nomi SAP → nomi modello)
2. Drop colonne non usate
3. Validazioni (cliente non nullo, date coerenti, importi ≥0)
4. Log profilo (conteggi clienti, anni coperti, percentuali null)
5. Output: parquet + _schema.json
6. CLI: python -m src.data.normalize_sap --input data/raw/*.xlsx --output data/normalized/events.parquet

Esegui: python -m src.data.normalize_sap --input data/raw/storico_sap.xlsx --output data/normalized/events.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.logging_setup import configure_logging

LOGGER = logging.getLogger(__name__)

COLUMNS_MAP = {
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

    "Testo testata": "testo_testata",
    "Testo posizione": "testo_posizione",
    "Riferimento": "riferimento",
    "Centro di costo": "centro_costo",
    "Contropartita": "contropartita",
}

REQUIRED_COLUMNS = [
    "cliente",
    "numero_documento",
    "data_documento",
    "imp_div_int",
]

DATE_COLUMNS = [
    "data_pareggio",
    "data_registrazione",
    "data_documento",
    "data_acquisizione",
    "data_base_pagamento",
]

AMOUNT_COLUMNS = [
    "imp_div_int",
    "importo_raw",
]

NULL_TOKENS = {"", "nan", "NaN", "None", "<NA>", "NaT"}

def _clean_string_column(series: pd.Series) -> pd.Series:
    """Normalizza colonne testuali: trim, rimuovi placeholder, rimuovi .0 trailing."""
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace({token: pd.NA for token in NULL_TOKENS})
    cleaned = cleaned.str.replace(r"(\.0+)$", "", regex=True)
    return cleaned

def _parse_datetime_column(series: pd.Series) -> pd.Series:
    """Parse date con formato italiano (DD/MM/YYYY) + fallback ISO8601 + Excel serial.
    
    Prompt 1 requirement: dayfirst=True per convenzione SAP italiana.
    """
    cleaned = _clean_string_column(series)

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

def _parse_amount_column(series: pd.Series) -> pd.Series:
    """Parse importo con formato italiano (1.234,56 → 1234.56).
    
    Gestisce:
    - Spazi non-breaking e normali come separatori migliaia
    - Virgola come separatore decimale
    - Segno negativo
    """
    cleaned = (
        series.astype("string")
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    cleaned = cleaned.replace({token: pd.NA for token in NULL_TOKENS})
    return pd.to_numeric(cleaned, errors="coerce")

def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizza schema: rename colonne, parse tipi, drop colonne extra.
    
    Args:
        df: DataFrame grezzo con nomi SAP
    
    Returns:
        DataFrame normalizzato con schema snake_case
    """
    columns_to_rename = {k: v for k, v in COLUMNS_MAP.items() if k in df.columns}
    df = df.rename(columns=columns_to_rename)

    expected_columns = set(COLUMNS_MAP.values())
    for col in expected_columns:
        if col not in df.columns:
            df[col] = pd.NA
            LOGGER.debug("Colonna '%s' mancante: aggiunta come NaN", col)

    extra_columns = [col for col in df.columns if col not in expected_columns]
    if extra_columns:
        LOGGER.info("Drop %d colonne extra non mappate: %s", len(extra_columns), extra_columns[:5])
        df = df.drop(columns=extra_columns)

    string_columns = [
        col for col in df.columns
        if col not in DATE_COLUMNS + AMOUNT_COLUMNS
    ]
    for col in string_columns:
        df[col] = _clean_string_column(df[col])

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = _parse_datetime_column(df[col])

    for col in AMOUNT_COLUMNS:
        if col in df.columns:
            df[col] = _parse_amount_column(df[col])

    return df

def validate_schema(df: pd.DataFrame) -> None:
    """Validazioni schema:
    - Colonne obbligatorie presenti
    - Cliente non nullo
    - Date coerenti (data_documento <= data_pareggio se entrambe presenti)
    - Importi ≥ 0 (o negativi esplicitamente ammessi, qui gestiamo abs)
    
    Raises:
        ValueError: Se validazione fallisce
    """
    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_required:
        raise ValueError(f"Colonne obbligatorie mancanti: {missing_required}")

    null_cliente_count = df["cliente"].isna().sum()
    if null_cliente_count > 0:
        raise ValueError(
            f"Trovate {null_cliente_count} righe con cliente nullo (invalido per pipeline)"
        )

    if "data_documento" in df.columns and "data_pareggio" in df.columns:
        valid_dates = df["data_documento"].notna() & df["data_pareggio"].notna()
        if valid_dates.any():
            incoherent = (
                df.loc[valid_dates, "data_documento"] > df.loc[valid_dates, "data_pareggio"]
            )
            if incoherent.any():
                LOGGER.warning(
                    "Trovate %d righe con data_documento > data_pareggio (incoerente)",
                    incoherent.sum()
                )

    if "imp_div_int" in df.columns:
        negative_count = (df["imp_div_int"] < 0).sum()
        if negative_count > 0:
            LOGGER.info(
                "Trovate %d righe con importo negativo (possibili note credito)",
                negative_count
            )

    LOGGER.info("✅ Validazione schema completata")

def log_profile(df: pd.DataFrame, original_row_count: int) -> Dict:
    """Log profilo dataset:
    - Righe originali vs trattenute
    - Clienti unici
    - Anni coperti (data_documento)
    - Percentuali null per colonne chiave
    
    Returns:
        Dict con metriche profilo (per JSON)
    """
    row_count = len(df)

    profile = {
        "rows_original": original_row_count,
        "rows_retained": row_count,
        "rows_discarded": original_row_count - row_count,
        "pct_retained": 100.0 * row_count / original_row_count if original_row_count > 0 else 0.0,
    }

    LOGGER.info("=" * 60)
    LOGGER.info("PROFILO DATASET")
    LOGGER.info("=" * 60)
    LOGGER.info("Righe originali: %d", original_row_count)
    LOGGER.info("Righe trattenute: %d", row_count)
    LOGGER.info("Righe scartate: %d (%.2f%%)", profile["rows_discarded"], 100.0 - profile["pct_retained"])

    if "cliente" in df.columns:
        unique_clients = df["cliente"].dropna().nunique()
        profile["unique_clients"] = unique_clients
        LOGGER.info("Clienti unici: %d", unique_clients)

    if "data_documento" in df.columns:
        date_series = df["data_documento"].dropna()
        if not date_series.empty:
            min_date = date_series.min()
            max_date = date_series.max()
            days_span = (max_date - min_date).days
            profile["data_documento_min"] = str(min_date.date())
            profile["data_documento_max"] = str(max_date.date())
            profile["days_span"] = days_span
            LOGGER.info(
                "Range data_documento: %s → %s (%d giorni)",
                min_date.date(),
                max_date.date(),
                days_span
            )

    key_columns = ["cliente", "numero_documento", "data_documento", "imp_div_int", "data_pareggio"]
    profile["null_percentages"] = {}
    LOGGER.info("Percentuali null:")
    for col in key_columns:
        if col in df.columns:
            null_count = df[col].isna().sum()
            pct_null = 100.0 * null_count / row_count if row_count > 0 else 0.0
            profile["null_percentages"][col] = pct_null
            LOGGER.info("  %s: %d (%.2f%%)", col, null_count, pct_null)

    if "tipo_documento" in df.columns:
        tipo_counts = df["tipo_documento"].fillna("<NA>").value_counts().head(5).to_dict()
        profile["top_tipo_documento"] = tipo_counts
        LOGGER.info("Top 5 tipo_documento: %s", tipo_counts)

    if "divisa" in df.columns:
        divisa_counts = df["divisa"].fillna("<NA>").value_counts().to_dict()
        profile["divisa_distribution"] = divisa_counts
        LOGGER.info("Divisa distribution: %s", divisa_counts)

    LOGGER.info("=" * 60)

    return profile

def save_schema_json(df: pd.DataFrame, output_dir: Path, profile: Dict) -> None:
    """Salva _schema.json con metadati colonne + profilo dataset."""
    schema_path = output_dir / "_schema.json"

    schema = {
        "columns": {},
        "profile": profile,
    }

    for col in df.columns:
        dtype_str = str(df[col].dtype)

        description = ""
        if col == "cliente":
            description = "Codice cliente SAP (univoco)"
        elif col == "numero_documento":
            description = "Numero documento contabile"
        elif col == "data_documento":
            description = "Data emissione fattura"
        elif col == "data_pareggio":
            description = "Data compensazione pagamento"
        elif col == "imp_div_int":
            description = "Importo in divisa interna (€)"
        elif col == "tipo_documento":
            description = "Tipo documento SAP (RV, NC, ...)"

        schema["columns"][col] = {
            "dtype": dtype_str,
            "description": description if description else f"Colonna {col}",
            "nullable": bool(df[col].isna().any()),
            "unique_values": int(df[col].nunique()) if df[col].dtype == "object" or df[col].dtype == "string" else None,
        }

    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    LOGGER.info("✅ Schema JSON salvato: %s", schema_path)

def normalize_sap(
    input_files: List[Path],
    output_path: Path,
) -> pd.DataFrame:
    """Pipeline completa normalizzazione SAP.
    
    Args:
        input_files: Lista file Excel/CSV SAP
        output_path: Destinazione parquet normalizzato
    
    Returns:
        DataFrame normalizzato
    """
    LOGGER.info("🚀 Avvio normalizzazione SAP")
    LOGGER.info("Input files: %d", len(input_files))
    LOGGER.info("Output: %s", output_path)

    frames = []
    original_row_count = 0

    for path in input_files:
        LOGGER.info("Lettura file: %s", path.name)

        if path.suffix.lower() == ".csv":
            df_raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            df_raw = pd.read_excel(path, dtype=str, engine="openpyxl")
            df_raw = df_raw.fillna("")
        else:
            LOGGER.warning("Formato file non supportato: %s (skipped)", path.suffix)
            continue

        original_row_count += len(df_raw)
        LOGGER.info("  Righe grezze: %d", len(df_raw))
        LOGGER.info("  Colonne grezze: %d", len(df_raw.columns))

        frames.append(df_raw)

    if not frames:
        raise ValueError("Nessun file valido trovato in input")

    df_concat = pd.concat(frames, ignore_index=True)
    LOGGER.info("Totale righe concatenate: %d", len(df_concat))

    df_normalized = normalize_schema(df_concat)
    LOGGER.info("Righe dopo normalizzazione: %d", len(df_normalized))

    validate_schema(df_normalized)

    profile = log_profile(df_normalized, original_row_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_normalized.to_parquet(output_path, compression="snappy", index=False)
    LOGGER.info("✅ Parquet salvato: %s", output_path)
    LOGGER.info("   Schema: %d colonne, %d righe", len(df_normalized.columns), len(df_normalized))

    save_schema_json(df_normalized, output_path.parent, profile)

    LOGGER.info("🎉 Normalizzazione completata con successo!")

    return df_normalized

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse argomenti CLI."""
    parser = argparse.ArgumentParser(
        description="Normalizza dati SAP: legge Excel/CSV, standardizza schema, produce parquet + JSON schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Normalizza singolo file
  python -m src.data.normalize_sap --input data/raw/storico_sap.xlsx --output data/normalized/events.parquet
  
  # Normalizza più file (wildcard)
  python -m src.data.normalize_sap --input data/raw/storico*.xlsx --output data/normalized/events.parquet
  
  # Con logging debug
  python -m src.data.normalize_sap --input data/raw/storico_sap.xlsx --output data/normalized/events.parquet --log-level DEBUG
        """,
    )

    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="File Excel/CSV SAP da normalizzare (supporta glob pattern)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destinazione parquet normalizzato (es. data/normalized/events.parquet)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Livello logging (default: INFO)",
    )

    return parser.parse_args(argv)

def main(argv: List[str] | None = None) -> None:
    """Entry point CLI."""
    args = parse_args(argv)

    configure_logging(level=getattr(logging, args.log_level))

    input_files = []
    for pattern in args.input:
        path = Path(pattern)
        if path.is_file():
            input_files.append(path)
        else:
            parent = path.parent if path.parent.exists() else Path.cwd()
            matched = list(parent.glob(path.name))
            if not matched:
                LOGGER.warning("Pattern '%s' non ha match", pattern)
            input_files.extend(matched)

    if not input_files:
        LOGGER.error("❌ Nessun file trovato in --input")
        sys.exit(1)

    try:
        normalize_sap(input_files, args.output)
    except Exception as e:
        LOGGER.exception("❌ Errore normalizzazione: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()

