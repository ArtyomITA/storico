"""CLI script to test PROMPT 1 ingestion normalizer with sample data.

Usage:
    python scripts/test_ingest_normalize.py

This script demonstrates the new ingest_normalize function with the sample
SAP export CSV in data/raw_sample/.
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.data.ingest import configure_logging, ingest_normalize

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run ingestion test with sample data."""
    configure_logging(level=logging.INFO)
    
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw_sample"
    output_path = project_root / "data" / "curated" / "line_items_normalized.parquet"
    
    LOGGER.info("=" * 70)
    LOGGER.info("PROMPT 1 Ingestion Normalizer Test")
    LOGGER.info("=" * 70)
    LOGGER.info("Source: %s", raw_dir)
    LOGGER.info("Output: %s", output_path)
    LOGGER.info("=" * 70)
    
    try:
        result = ingest_normalize(raw_dir=raw_dir, output_path=output_path)
        
        LOGGER.info("")
        LOGGER.info("SUCCESS! Ingestion completed.")
        LOGGER.info("")
        LOGGER.info("Output DataFrame shape: %s", result.shape)
        LOGGER.info("Output columns (%d): %s", len(result.columns), list(result.columns))
        LOGGER.info("")
        LOGGER.info("Sample records (first 3 rows):")
        print(result.head(3).to_string(max_colwidth=20))
        LOGGER.info("")
        LOGGER.info("Data types:")
        print(result.dtypes.to_string())
        
    except Exception as e:
        LOGGER.error("Ingestion failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
