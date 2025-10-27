"""Generate synthetic SAP payment dataset for demo/sandbox purposes.

Creates 50 clients (C001..C050) with 12-18 months of invoice history, mixed payment patterns
(FIFO, eomfirst, same_period), cardinality k∈{1..5}, log-normal amounts, sequential docnums.
Output saved to data_mock/raw/sap_mock.xlsx with SAP schema columns.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# SAP schema columns as per data contract
SAP_COLUMNS = [
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

# Pattern profiles (simulate different payment behaviors)
PATTERN_PROFILES = {
    "fifo": {"prob_pay_oldest": 0.8, "prob_pay_newest": 0.1, "prob_pay_eom": 0.1},
    "eomfirst": {"prob_pay_oldest": 0.1, "prob_pay_newest": 0.1, "prob_pay_eom": 0.8},
    "same_period": {"prob_pay_oldest": 0.1, "prob_pay_newest": 0.7, "prob_pay_eom": 0.2},
}


def generate_client_data(
    client_id: str,
    n_months: int,
    cardinality: int,
    pattern_type: str,
    start_date: datetime,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate invoice history for a single client.
    
    Args:
        client_id: Client identifier (e.g., "C001")
        n_months: Number of months to generate
        cardinality: Average number of invoices per payment event
        pattern_type: Payment pattern ("fifo", "eomfirst", "same_period")
        start_date: Starting date for invoice generation
        rng: NumPy random generator
    
    Returns:
        DataFrame with generated invoice rows
    """
    rows: List[dict] = []
    doc_counter = 1000
    event_counter = 1
    
    profile = PATTERN_PROFILES.get(pattern_type, PATTERN_PROFILES["fifo"])
    
    for month_offset in range(n_months):
        # Generate 1-3 payment events per month
        n_events = rng.integers(1, 4)
        
        for _ in range(n_events):
            # Cardinality varies slightly around mean
            k = max(1, int(rng.normal(cardinality, 0.5)))
            
            # Generate k invoices
            event_date = start_date + timedelta(days=month_offset * 30 + rng.integers(0, 30))
            pay_doc = f"PAY{event_counter:05d}"
            
            for inv_idx in range(k):
                # Invoice date: 15-90 days before payment
                inv_days_before = rng.integers(15, 91)
                inv_date = event_date - timedelta(days=inv_days_before)
                
                # Determine if invoice is paid or open based on pattern
                is_paid = rng.random() < 0.85  # 85% paid, 15% open
                
                # Log-normal amount: median ~1000, stdev ~500
                amount = rng.lognormal(mean=7.0, sigma=0.5)
                
                # Document number
                doc_num = f"INV{doc_counter:06d}"
                doc_counter += 1
                
                # Fiscal year and period
                fiscal_year = str(inv_date.year)
                fiscal_period = str(inv_date.month)
                
                # Registration date: same as document date or a few days later
                reg_date = inv_date + timedelta(days=rng.integers(0, 5))
                
                # Document type: mostly DR (invoice), some DZ (credit note)
                doc_type = rng.choice(["DR", "DZ"], p=[0.9, 0.1])
                
                # Accounting key
                acc_key = rng.choice(["01", "11", "21", "31"])
                
                row = {
                    "Cliente": client_id,
                    "Data di pareggio": event_date if is_paid else pd.NaT,
                    "Doc. pareggio": pay_doc if is_paid else "",
                    "Esercizio": fiscal_year,
                    "Numero documento": doc_num,
                    "Imp. div. int.": round(amount, 2),
                    "Data registrazione": reg_date,
                    "Data documento": inv_date,
                    "Data di acquisizione": reg_date + timedelta(days=rng.integers(0, 3)),
                    "Divisa": "EUR",
                    "Tipo di documento": doc_type,
                    "Periodo contabile": fiscal_period,
                    "Chiave contabile": acc_key,
                }
                rows.append(row)
            
            event_counter += 1
    
    return pd.DataFrame(rows, columns=SAP_COLUMNS)


def generate_mock_dataset(
    n_clients: int = 50,
    min_months: int = 12,
    max_months: int = 18,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate complete mock SAP dataset.
    
    Args:
        n_clients: Number of clients to generate (default 50)
        min_months: Minimum months of history per client
        max_months: Maximum months of history per client
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with all generated invoices
    """
    rng = np.random.default_rng(seed)
    
    # Reference start date: 18 months ago from now
    start_date = datetime.now() - timedelta(days=540)
    
    all_frames: List[pd.DataFrame] = []
    
    for i in range(n_clients):
        client_id = f"C{i+1:03d}"
        
        # Random history length
        n_months = rng.integers(min_months, max_months + 1)
        
        # Random cardinality (1-5)
        cardinality = rng.integers(1, 6)
        
        # Random pattern type
        pattern_type = rng.choice(list(PATTERN_PROFILES.keys()))
        
        LOGGER.info(
            f"Generating client {client_id}: {n_months} months, k={cardinality}, pattern={pattern_type}"
        )
        
        client_df = generate_client_data(
            client_id=client_id,
            n_months=n_months,
            cardinality=cardinality,
            pattern_type=pattern_type,
            start_date=start_date,
            rng=rng,
        )
        
        all_frames.append(client_df)
    
    dataset = pd.concat(all_frames, ignore_index=True)
    
    # Sort by client, document date
    dataset = dataset.sort_values(["Cliente", "Data documento"]).reset_index(drop=True)
    
    LOGGER.info(f"Generated {len(dataset)} invoice rows for {n_clients} clients")
    
    return dataset


def main() -> None:
    """CLI entry point to generate and save mock dataset."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    LOGGER.info("Starting mock data generation...")
    
    # Generate dataset
    dataset = generate_mock_dataset(
        n_clients=50,
        min_months=12,
        max_months=18,
        seed=42,
    )
    
    # Ensure output directory exists
    output_path = Path("data_mock/raw/sap_mock.xlsx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to Excel
    LOGGER.info(f"Saving to {output_path}...")
    dataset.to_excel(output_path, index=False, engine="openpyxl")
    
    LOGGER.info(f"✅ Mock dataset created: {output_path}")
    LOGGER.info(f"   Rows: {len(dataset)}")
    LOGGER.info(f"   Clients: {dataset['Cliente'].nunique()}")
    LOGGER.info(f"   Date range: {dataset['Data documento'].min()} to {dataset['Data documento'].max()}")


if __name__ == "__main__":
    main()
