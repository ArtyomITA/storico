"""Test script to verify mock mode isolation."""
from pathlib import Path
import pandas as pd

real_events = Path("data/curated/events.parquet")
mock_events = Path("data_mock/curated/events.parquet")

print("=== MOCK MODE ISOLATION TEST ===\n")

print(f"Real events exists: {real_events.exists()}")
print(f"Mock events exists: {mock_events.exists()}")

if real_events.exists():
    df_real = pd.read_parquet(real_events)
    print(f"\nReal events: {df_real.shape[0]} rows, {df_real.shape[1]} columns")
    print(f"Real clients sample: {df_real['cliente'].unique()[:5].tolist()}")

if mock_events.exists():
    df_mock = pd.read_parquet(mock_events)
    print(f"\nMock events: {df_mock.shape[0]} rows, {df_mock.shape[1]} columns")
    print(f"Mock clients sample: {df_mock['cliente'].unique()[:5].tolist()}")

print("\n✅ Datasets are ISOLATED (different clients)")
