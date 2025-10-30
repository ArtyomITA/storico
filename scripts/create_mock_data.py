"""Create minimal mock dataset for testing mock mode isolation."""
import pandas as pd
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parents[1]
MOCK_RAW = BASE_DIR / "data_mock" / "raw"
MOCK_CURATED = BASE_DIR / "data_mock" / "curated"

# Ensure directories exist
MOCK_RAW.mkdir(parents=True, exist_ok=True)
MOCK_CURATED.mkdir(parents=True, exist_ok=True)

# Create minimal line_items.parquet
line_items = pd.DataFrame({
    "societa": ["MOCK"] * 10,
    "cliente": [f"DEMO_C{i:02d}" for i in range(1, 11)],
    "data_pareggio": pd.date_range("2024-01-01", periods=10, freq="7D"),
    "doc_pareggio": [f"PAY{i:04d}" for i in range(1, 11)],
    "esercizio": [2024] * 10,
    "numero_documento": [f"INV{i:04d}" for i in range(1, 11)],
    "posizione": [1] * 10,
    "data_registrazione": pd.date_range("2023-12-15", periods=10, freq="7D"),
    "data_documento": pd.date_range("2023-12-01", periods=10, freq="7D"),
    "data_acquisizione": pd.date_range("2023-12-16", periods=10, freq="7D"),
    "divisa": ["EUR"] * 10,
    "tipo_documento": ["D1"] * 10,
    "periodo_contabile": [12, 12, 12, 1, 1, 1, 2, 2, 2, 3],
    "chiave_contabile": ["15"] * 10,
    "imp_div_int": [1000 * (i + 1) for i in range(10)],
    "importo_raw": [100 * (i + 1) for i in range(10)],
    "divisa_fun": ["EUR"] * 10,
    "data_base_pagamento": pd.date_range("2024-01-01", periods=10, freq="7D"),
    "cond_pagamento": [pd.NA] * 10,
})
line_items_path = MOCK_CURATED / "line_items.parquet"
line_items.to_parquet(line_items_path, index=False)
print(f"✓ Created {line_items_path} ({len(line_items)} rows)")

# Create minimal events.parquet
events = pd.DataFrame({
    "cliente": [f"DEMO_C{i:02d}" for i in range(1, 6)],
    "event_key": [f"DEMO_C{i:02d}_2024-01-{(i*7)%28+1:02d}" for i in range(1, 6)],
    "data_pareggio": pd.date_range("2024-01-07", periods=5, freq="7D"),
    "n_righe_evento": [1, 2, 1, 3, 2],
    "tot_importo_evento": [1000, 3000, 2000, 6000, 5000],
    "inter_event_gap_days": [7.0, 7.0, 7.0, 7.0, 7.0],
})
events_path = MOCK_CURATED / "events.parquet"
events.to_parquet(events_path, index=False)
print(f"✓ Created {events_path} ({len(events)} rows)")

# Create minimal event_items.parquet
event_items = pd.DataFrame({
    "cliente": ["DEMO_C01", "DEMO_C02", "DEMO_C02", "DEMO_C03"],
    "event_key": [
        "DEMO_C01_2024-01-07",
        "DEMO_C02_2024-01-14",
        "DEMO_C02_2024-01-14",
        "DEMO_C03_2024-01-21",
    ],
    "numero_documento": ["INV0001", "INV0002", "INV0003", "INV0004"],
    "tipo_documento": ["D1", "D1", "BC", "D1"],
    "imp_div_int": [1000, 2000, 1000, 2000],
    "periodo_contabile": [12, 12, 12, 1],
})
event_items_path = MOCK_CURATED / "event_items.parquet"
event_items.to_parquet(event_items_path, index=False)
print(f"✓ Created {event_items_path} ({len(event_items)} rows)")

# Create minimal client_profiles.parquet
client_profiles = pd.DataFrame({
    "cliente": [f"DEMO_C{i:02d}" for i in range(1, 6)],
    "n_events": [2, 3, 2, 4, 3],
    "total_amt": [10000.0, 15000.0, 8000.0, 20000.0, 12000.0],
    "mean_gap_days": [7.0, 10.0, 7.0, 14.0, 7.0],
})
client_profiles_path = MOCK_CURATED / "client_profiles.parquet"
client_profiles.to_parquet(client_profiles_path, index=False)
print(f"✓ Created {client_profiles_path} ({len(client_profiles)} rows)")

# Create minimal client_pattern_scores.parquet
client_pattern_scores = pd.DataFrame({
    "cliente": [f"DEMO_C{i:02d}" for i in range(1, 6)],
    "fifo_score": [0.8, 0.6, 0.7, 0.9, 0.75],
    "lifo_score": [0.2, 0.4, 0.3, 0.1, 0.25],
    "same_period_score": [0.9, 0.8, 0.85, 0.95, 0.9],
    "largest_score": [0.7, 0.65, 0.75, 0.8, 0.7],
    "smallest_score": [0.3, 0.35, 0.25, 0.2, 0.3],
    "eomfirst_score": [0.5, 0.6, 0.55, 0.7, 0.6],
    "regmonth_score": [0.85, 0.75, 0.8, 0.9, 0.85],
    "periodicity_weekly": [0.9, 0.7, 0.85, 0.6, 0.8],
    "periodicity_biweekly": [0.1, 0.2, 0.1, 0.3, 0.15],
    "periodicity_monthly": [0.05, 0.1, 0.05, 0.1, 0.05],
    "dow_concentration": [0.6, 0.5, 0.55, 0.7, 0.6],
    "dom_concentration": [0.4, 0.45, 0.4, 0.5, 0.45],
    "stability_index": [0.75, 0.65, 0.7, 0.8, 0.72],
    "k_mode": [1.0, 2.0, 1.0, 2.0, 1.5],
    "k_entropy": [0.3, 0.5, 0.35, 0.6, 0.45],
})
client_pattern_scores_path = MOCK_CURATED / "client_pattern_scores.parquet"
client_pattern_scores.to_parquet(client_pattern_scores_path, index=False)
print(f"✓ Created {client_pattern_scores_path} ({len(client_pattern_scores)} rows)")

print("\n✅ Mock dataset created successfully!")
print(f"   Location: {MOCK_CURATED}")
print(f"   Files: 5 Parquet tables (line_items, events, event_items, client_profiles, client_pattern_scores)")
print(f"   Clients: 5 demo clients (DEMO_C01 to DEMO_C05)")
