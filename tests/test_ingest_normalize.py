"""Unit tests for PROMPT 1 SAP ingestion normalizer.

Tests the extended COL_MAP mapping, date parsing, amount normalization,
and quality reporting features.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.ingest import (
    COL_MAP,
    _clean_string_series,
    _parse_amount_series,
    _parse_datetime_series,
    ingest_normalize,
    log_quality_report,
    normalise_dataframe,
    read_raw_file,
)


class TestColumnMapping:
    """Test COL_MAP coverage and completeness."""

    def test_col_map_contains_all_prompt1_columns(self):
        """Verify COL_MAP includes all 19 columns from PROMPT 1."""
        expected_italian = {
            "Società",
            "Cliente",
            "Data di pareggio",
            "Doc. pareggio",
            "Esercizio",
            "Numero documento",
            "Posizione",
            "Data registrazione",
            "Data documento",
            "Data di acquisizione",
            "Divisa",
            "Tipo di documento",
            "Periodo contabile",
            "Chiave contabile",
            "Imp. div. int.",
            "Importo",
            "Divisa funzionale",
            "Data base pagamento",
            "Cond. pagamento",
        }
        assert set(COL_MAP.keys()) == expected_italian

    def test_col_map_values_are_snake_case(self):
        """Verify all mapped values are valid snake_case identifiers."""
        for italian, snake in COL_MAP.items():
            assert snake.islower(), f"{italian} -> {snake} is not lowercase"
            assert " " not in snake, f"{italian} -> {snake} contains spaces"
            # Allow underscores and alphanumerics only
            assert all(c.isalnum() or c == "_" for c in snake), (
                f"{italian} -> {snake} contains invalid characters"
            )


class TestStringCleaning:
    """Test _clean_string_series for trimming and .0 removal."""

    def test_trim_whitespace(self):
        """Strip leading/trailing spaces."""
        series = pd.Series(["  CUST001  ", "CUST002", " CUST003"])
        cleaned = _clean_string_series(series)
        assert cleaned.tolist() == ["CUST001", "CUST002", "CUST003"]

    def test_remove_trailing_dot_zero(self):
        """Remove .0 suffix from document numbers ."""
        series = pd.Series(["123456.0", "789012.00", "345678"])
        cleaned = _clean_string_series(series)
        assert cleaned.tolist() == ["123456", "789012", "345678"]

    def test_replace_null_tokens_with_na(self):
        """Replace empty strings, 'nan', 'None' with pd.NA."""
        series = pd.Series(["", "nan", "NaN", "None", "<NA>", "VALID"])
        cleaned = _clean_string_series(series)
        assert pd.isna(cleaned.iloc[0])
        assert pd.isna(cleaned.iloc[1])
        assert pd.isna(cleaned.iloc[2])
        assert pd.isna(cleaned.iloc[3])
        assert pd.isna(cleaned.iloc[4])
        assert cleaned.iloc[5] == "VALID"


class TestDateParsing:
    """Test _parse_datetime_series for day-first Italian dates."""

    def test_parse_italian_day_first_format(self):
        """Parse DD/MM/YYYY format (Italian convention)."""
        series = pd.Series(["15/01/2024", "28/02/2024", "31/12/2023"])
        parsed = _parse_datetime_series(series)
        assert parsed.iloc[0] == pd.Timestamp("2024-01-15")
        assert parsed.iloc[1] == pd.Timestamp("2024-02-28")
        assert parsed.iloc[2] == pd.Timestamp("2023-12-31")

    def test_parse_iso_format_fallback(self):
        """Fallback to ISO 8601 format (YYYY-MM-DD)."""
        series = pd.Series(["2024-01-15", "2024-02-28"])
        parsed = _parse_datetime_series(series)
        assert parsed.iloc[0] == pd.Timestamp("2024-01-15")
        assert parsed.iloc[1] == pd.Timestamp("2024-02-28")

    def test_parse_excel_serial_dates(self):
        """Parse Excel numeric serial dates."""
        # Excel serial 44927 = 2023-01-01 (from 1899-12-30 origin)
        series = pd.Series(["44927", "44958"])
        parsed = _parse_datetime_series(series)
        assert parsed.iloc[0] == pd.Timestamp("2023-01-01")
        assert parsed.iloc[1] == pd.Timestamp("2023-02-01")

    def test_invalid_dates_return_nat(self):
        """Invalid date strings return NaT."""
        series = pd.Series(["not-a-date", "99/99/9999", ""])
        parsed = _parse_datetime_series(series)
        assert pd.isna(parsed.iloc[0])
        assert pd.isna(parsed.iloc[1])
        assert pd.isna(parsed.iloc[2])


class TestAmountParsing:
    """Test _parse_amount_series for European decimal format."""

    def test_parse_european_decimal_comma(self):
        """Parse 1.234,56 → 1234.56 (European format)."""
        series = pd.Series(["1.234,56", "2.500,00", "999,99"])
        parsed = _parse_amount_series(series)
        assert parsed.iloc[0] == pytest.approx(1234.56)
        assert parsed.iloc[1] == pytest.approx(2500.00)
        assert parsed.iloc[2] == pytest.approx(999.99)

    def test_parse_negative_amounts(self):
        """Handle negative amounts with minus sign."""
        series = pd.Series(["-1.234,56", "-500,00"])
        parsed = _parse_amount_series(series)
        assert parsed.iloc[0] == pytest.approx(-1234.56)
        assert parsed.iloc[1] == pytest.approx(-500.00)

    def test_remove_non_breaking_spaces(self):
        """Remove non-breaking spaces (thousand separator)."""
        series = pd.Series(["1\xa0234,56", "2\xa0500,00"])
        parsed = _parse_amount_series(series)
        assert parsed.iloc[0] == pytest.approx(1234.56)
        assert parsed.iloc[1] == pytest.approx(2500.00)

    def test_invalid_amounts_return_nan(self):
        """Invalid amount strings return NaN."""
        series = pd.Series(["not-a-number", "", "ABC"])
        parsed = _parse_amount_series(series)
        assert pd.isna(parsed.iloc[0])
        assert pd.isna(parsed.iloc[1])
        assert pd.isna(parsed.iloc[2])


class TestNormalizeDataFrame:
    """Test normalise_dataframe with extended COL_MAP."""

    def test_normalize_minimal_columns(self):
        """Normalize DataFrame with only Cliente + Imp. div. int."""
        raw = pd.DataFrame({
            "Cliente": ["CUST001", "CUST002"],
            "Imp. div. int.": ["1.234,56", "2.500,00"],
        })
        normalized = normalise_dataframe(raw, use_extended_map=True)
        
        assert "cliente" in normalized.columns
        assert "imp_div_int" in normalized.columns
        assert normalized["cliente"].tolist() == ["CUST001", "CUST002"]
        assert normalized["imp_div_int"].iloc[0] == pytest.approx(1234.56)

    def test_add_missing_columns_as_nan(self):
        """Add columns not present in raw DataFrame as NaN ."""
        raw = pd.DataFrame({
            "Cliente": ["CUST001"],
        })
        normalized = normalise_dataframe(raw, use_extended_map=True)
        
        # Should have all COL_MAP columns, missing ones filled with NaN
        assert "societa" in normalized.columns
        assert "posizione" in normalized.columns
        assert "cond_pagamento" in normalized.columns
        assert pd.isna(normalized["societa"].iloc[0])
        assert pd.isna(normalized["posizione"].iloc[0])

    def test_parse_all_date_columns(self):
        """Parse all 5 date columns correctly."""
        raw = pd.DataFrame({
            "Cliente": ["CUST001"],
            "Data documento": ["15/01/2024"],
            "Data di pareggio": ["20/01/2024"],
            "Data registrazione": ["16/01/2024"],
            "Data di acquisizione": ["17/01/2024"],
            "Data base pagamento": ["18/01/2024"],
        })
        normalized = normalise_dataframe(raw, use_extended_map=True)
        
        assert normalized["data_documento"].iloc[0] == pd.Timestamp("2024-01-15")
        assert normalized["data_pareggio"].iloc[0] == pd.Timestamp("2024-01-20")
        assert normalized["data_registrazione"].iloc[0] == pd.Timestamp("2024-01-16")
        assert normalized["data_acquisizione"].iloc[0] == pd.Timestamp("2024-01-17")
        assert normalized["data_base_pagamento"].iloc[0] == pd.Timestamp("2024-01-18")

    def test_backward_compatibility_with_rename_map(self):
        """Test use_extended_map=False maintains backward compatibility."""
        raw = pd.DataFrame({
            "Cliente": ["CUST001"],
            "Data documento": ["15/01/2024"],
            "Imp. div. int.": ["1.234,56"],
        })
        normalized = normalise_dataframe(raw, use_extended_map=False)
        
        # Should only have RENAME_MAP columns (no societa, posizione, etc.)
        assert "cliente" in normalized.columns
        assert "data_documento" in normalized.columns
        assert "imp_div_int" in normalized.columns
        # Extended columns should NOT be added
        assert "societa" not in normalized.columns
        assert "posizione" not in normalized.columns


class TestReadRawFile:
    """Test read_raw_file with strict vs non-strict modes."""

    def test_strict_mode_enforces_raw_columns(self, tmp_path):
        """strict=True requires all RAW_COLUMNS to be present."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "Cliente,Data documento\n"
            "CUST001,15/01/2024\n",
            encoding="utf-8",
        )
        
        with pytest.raises(ValueError, match="missing required columns"):
            read_raw_file(csv_path, strict=True)

    def test_non_strict_mode_accepts_any_col_map_columns(self, tmp_path):
        """strict=False accepts any columns present in COL_MAP."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "Cliente,Società,Posizione\n"
            "CUST001,SOC01,POS1\n",
            encoding="utf-8",
        )
        
        frame = read_raw_file(csv_path, strict=False)
        assert "Cliente" in frame.columns
        assert "Società" in frame.columns
        assert "Posizione" in frame.columns
        assert len(frame) == 1

    def test_non_strict_mode_logs_extra_columns(self, tmp_path, caplog):
        """strict=False logs columns not in COL_MAP."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "Cliente,UnknownColumn,AnotherExtra\n"
            "CUST001,VALUE,VALUE2\n",
            encoding="utf-8",
        )
        
        frame = read_raw_file(csv_path, strict=False)
        # Should only keep Cliente (in COL_MAP), drop others
        assert "Cliente" in frame.columns
        assert "UnknownColumn" not in frame.columns
        assert "extra columns" in caplog.text.lower()


class TestQualityReport:
    """Test log_quality_report output."""

    def test_quality_report_logs_key_metrics(self, caplog):
        """Verify quality report includes clients, date ranges, missing %."""
        dataset = pd.DataFrame({
            "cliente": ["CUST001", "CUST002", "CUST001"],
            "numero_documento": ["INV001", "INV002", "INV003"],
            "data_documento": pd.to_datetime(["2024-01-15", "2024-02-20", "2024-03-10"]),
            "data_pareggio": pd.to_datetime(["2024-01-20", pd.NaT, "2024-03-15"]),
            "imp_div_int": [1234.56, np.nan, 3456.78],
        })
        
        log_quality_report(dataset, original_row_count=5)
        
        log_text = caplog.text.lower()
        assert "quality report" in log_text
        assert "unique clients: 2" in log_text
        assert "data documento range" in log_text
        assert "missing value percentages" in log_text
        assert "rows discarded" in log_text


class TestIngestNormalizeE2E:
    """End-to-end test for ingest_normalize function."""

    def test_ingest_normalize_full_workflow(self, tmp_path):
        """Test complete ingestion workflow with sample CSV."""
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        
        # Create sample CSV with Italian headers
        csv_path = raw_dir / "sample.csv"
        csv_content = """Società,Cliente,Data documento,Data di pareggio,Numero documento,Imp. div. int.,Tipo di documento,Divisa
SOC01,CUST001,15/01/2024,20/01/2024,INV001,"1.234,56",AB,EUR
SOC01,CUST002,20/02/2024,,INV002,"2.500,00",DG,EUR
SOC02,CUST001,10/03/2024,15/03/2024,INV003,"3.456,78",AB,USD
"""
        csv_path.write_text(csv_content, encoding="utf-8")
        
        output_path = tmp_path / "output" / "line_items.parquet"
        
        # Run ingestion
        result = ingest_normalize(raw_dir=raw_dir, output_path=output_path)
        
        # Verify output
        assert output_path.exists()
        assert len(result) == 3
        assert "cliente" in result.columns
        assert "societa" in result.columns
        assert "imp_div_int" in result.columns
        
        # Verify data types
        assert result["data_documento"].dtype == "datetime64[ns]"
        assert result["imp_div_int"].dtype == "float64"
        
        # Verify values
        assert result["cliente"].tolist() == ["CUST001", "CUST002", "CUST001"]
        assert result["societa"].tolist() == ["SOC01", "SOC01", "SOC02"]
        assert result["imp_div_int"].iloc[0] == pytest.approx(1234.56)
        
        # Verify NaN for missing data_pareggio
        assert pd.isna(result["data_pareggio"].iloc[1])

    def test_ingest_normalize_handles_extra_columns(self, tmp_path, caplog):
        """Test that extra columns not in COL_MAP are dropped."""
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        
        csv_path = raw_dir / "sample.csv"
        csv_content = """Cliente,ExtraColumn1,Imp. div. int.,UnknownField
CUST001,VALUE1,1000,VALUE2
"""
        csv_path.write_text(csv_content, encoding="utf-8")
        
        output_path = tmp_path / "output" / "line_items.parquet"
        result = ingest_normalize(raw_dir=raw_dir, output_path=output_path)
        
        # Should only have columns from COL_MAP
        assert "cliente" in result.columns
        assert "imp_div_int" in result.columns
        assert "ExtraColumn1" not in result.columns
        assert "UnknownField" not in result.columns
        
        # Should log extra columns
        assert "extra columns" in caplog.text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
