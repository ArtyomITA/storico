"""Generate deterministic mock SAP payment data for pipeline validation.

Creates 3 types of artificial clients with predictable payment behaviors:
1. FIFO Metodico: Always pays oldest invoices first
2. BigSpender (Largest First): Always pays largest invoices first
3. Caotico (Random): Pays randomly

Expected validation results:
- HDBSCAN: FIFO clients in one cluster, Largest in another, Random as noise
- CatBoost accuracy > 95% on deterministic clients

Output format matches: data/bsad_zmc_zint_2023_2025_bz_bc.xlsx (30 columns)
"""
import logging
from pathlib import Path
from datetime import date, timedelta
from typing import List, Tuple
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

N_CLIENTS_PER_TYPE = 33
YEARS_OF_HISTORY = 2
INVOICES_PER_MONTH_MIN = 3
INVOICES_PER_MONTH_MAX = 7
PAYMENTS_PER_MONTH = 2
RANDOM_SEED = 42

CLIENT_TYPES = ["FIFO", "LARGEST", "RANDOM"]

SAP_COLUMNS = [
    "Società",
    "Cliente",
    "Data di pareggio",
    "Doc. pareggio",
    "Attribuzione",
    "Esercizio",
    "Numero documento",
    "Posizione",
    "Data registrazione",
    "Data documento",
    "Data di acquisizione",
    "Divisa",
    "Riferimento",
    "Tipo di documento",
    "Periodo contabile",
    "Chiave contabile",
    "Indic. dare/avere",
    "Settore contabile",
    "Imp. div. int.",
    "Importo",
    "Importo in divisa funzionale",
    "Divisa funzionale",
    "Conto Co.Ge.",
    "Conto Co.Ge..1",
    "Data base pagamento",
    "Divisa Contabilità generale",
    "Importo Contabilità generale",
    "Area contr. credito",
    "Es. pareggio",
    "Operazione di rif.",
]


def generate_client_ids(n_per_type: int = N_CLIENTS_PER_TYPE) -> pd.DataFrame:
    """Generate client IDs with known type labels."""
    clients = []
    base_id = 21000000
    for i, ctype in enumerate(CLIENT_TYPES):
        for j in range(n_per_type):
            clients.append({
                "cliente": base_id + i * 1000 + j,
                "cliente_name": f"{ctype}_{j+1:03d}",
                "client_type": ctype,
                "type_idx": i,
            })
    df = pd.DataFrame(clients)
    LOGGER.info("Generated %d clients (%d per type)", len(df), n_per_type)
    return df


def generate_invoices_for_client(
    cliente: int,
    start_date: date,
    end_date: date,
    rng: np.random.Generator,
) -> List[dict]:
    """Generate monthly invoices for a single client in SAP format."""
    invoices = []
    doc_counter = 1900000000
    current = date(start_date.year, start_date.month, 1)
    societa_list = ["ZMC", "ZINT", "ZBC", "ZBZ"]
    
    while current < end_date:
        n_invoices = rng.integers(INVOICES_PER_MONTH_MIN, INVOICES_PER_MONTH_MAX + 1)
        for _ in range(n_invoices):
            day = rng.integers(1, 28)
            inv_date = date(current.year, current.month, day)
            amount = round(rng.uniform(100, 10000), 2)
            doc_num = doc_counter + len(invoices)
            invoices.append({
                "Società": rng.choice(societa_list),
                "Cliente": cliente,
                "Data di pareggio": pd.NaT,
                "Doc. pareggio": pd.NA,
                "Attribuzione": cliente,
                "Esercizio": inv_date.year,
                "Numero documento": doc_num,
                "Posizione": 1,
                "Data registrazione": inv_date,
                "Data documento": inv_date,
                "Data di acquisizione": inv_date,
                "Divisa": "EUR",
                "Riferimento": f"REF{doc_num}",
                "Tipo di documento": rng.choice(["RV", "DR", "DG"]),
                "Periodo contabile": inv_date.month,
                "Chiave contabile": rng.choice([1, 11]),
                "Indic. dare/avere": "S",
                "Settore contabile": np.nan,
                "Imp. div. int.": amount,
                "Importo": amount,
                "Importo in divisa funzionale": amount,
                "Divisa funzionale": "EUR",
                "Conto Co.Ge.": 14000000,
                "Conto Co.Ge..1": 14000000,
                "Data base pagamento": inv_date + timedelta(days=30),
                "Divisa Contabilità generale": "EUR",
                "Importo Contabilità generale": amount,
                "Area contr. credito": "Z004",
                "Es. pareggio": pd.NA,
                "Operazione di rif.": "BKPF",
            })
        current = date(current.year + (current.month // 12), (current.month % 12) + 1, 1)
    
    return invoices


def apply_fifo_payments(
    invoices: List[dict],
    end_date: date,
    rng: np.random.Generator,
) -> List[dict]:
    """Apply FIFO payment strategy: oldest invoices paid first."""
    df = pd.DataFrame(invoices)
    df = df.sort_values("Data documento").reset_index(drop=True)
    paid_mask = np.zeros(len(df), dtype=bool)
    
    current = date(df["Data documento"].min().year, df["Data documento"].min().month, 1)
    current = date(current.year + (current.month // 12), (current.month % 12) + 1, 1)
    
    doc_pareggio_counter = 1400000000
    
    while current < end_date:
        payment_date = date(current.year, current.month, rng.integers(15, 28))
        n_to_pay = rng.integers(1, PAYMENTS_PER_MONTH + 2)
        open_mask = (~paid_mask) & (pd.to_datetime(df["Data documento"]).dt.date < payment_date)
        open_indices = df.index[open_mask].tolist()
        
        if len(open_indices) > 0:
            oldest = open_indices[:n_to_pay]
            doc_p = doc_pareggio_counter
            doc_pareggio_counter += 1
            for idx in oldest:
                paid_mask[idx] = True
                invoices[idx]["Data di pareggio"] = payment_date
                invoices[idx]["Doc. pareggio"] = doc_p
                invoices[idx]["Es. pareggio"] = payment_date.year
        
        current = date(current.year + (current.month // 12), (current.month % 12) + 1, 1)
    
    return invoices


def apply_largest_payments(
    invoices: List[dict],
    end_date: date,
    rng: np.random.Generator,
) -> List[dict]:
    """Apply Largest-First payment strategy: biggest invoices paid first."""
    df = pd.DataFrame(invoices)
    df = df.sort_values("Imp. div. int.", ascending=False).reset_index(drop=True)
    original_order = {inv["Numero documento"]: i for i, inv in enumerate(invoices)}
    paid_mask = np.zeros(len(df), dtype=bool)
    
    current = date(df["Data documento"].min().year, df["Data documento"].min().month, 1)
    current = date(current.year + (current.month // 12), (current.month % 12) + 1, 1)
    
    doc_pareggio_counter = 1400000000
    
    while current < end_date:
        payment_date = date(current.year, current.month, rng.integers(15, 28))
        n_to_pay = rng.integers(1, PAYMENTS_PER_MONTH + 2)
        open_mask = (~paid_mask) & (pd.to_datetime(df["Data documento"]).dt.date < payment_date)
        open_indices = df.index[open_mask].tolist()
        
        if len(open_indices) > 0:
            largest = open_indices[:n_to_pay]
            doc_p = doc_pareggio_counter
            doc_pareggio_counter += 1
            for idx in largest:
                paid_mask[idx] = True
                doc_num = df.loc[idx, "Numero documento"]
                orig_idx = original_order[doc_num]
                invoices[orig_idx]["Data di pareggio"] = payment_date
                invoices[orig_idx]["Doc. pareggio"] = doc_p
                invoices[orig_idx]["Es. pareggio"] = payment_date.year
        
        current = date(current.year + (current.month // 12), (current.month % 12) + 1, 1)
    
    return invoices


def apply_random_payments(
    invoices: List[dict],
    end_date: date,
    rng: np.random.Generator,
) -> List[dict]:
    """Apply Random payment strategy: invoices paid in random order."""
    df = pd.DataFrame(invoices).reset_index(drop=True)
    original_order = {inv["Numero documento"]: i for i, inv in enumerate(invoices)}
    paid_mask = np.zeros(len(df), dtype=bool)
    
    current = date(df["Data documento"].min().year, df["Data documento"].min().month, 1)
    current = date(current.year + (current.month // 12), (current.month % 12) + 1, 1)
    
    doc_pareggio_counter = 1400000000
    
    while current < end_date:
        payment_date = date(current.year, current.month, rng.integers(15, 28))
        n_to_pay = rng.integers(1, PAYMENTS_PER_MONTH + 2)
        open_mask = (~paid_mask) & (pd.to_datetime(df["Data documento"]).dt.date < payment_date)
        open_indices = df.index[open_mask].tolist()
        
        if len(open_indices) > 0:
            chosen = rng.choice(open_indices, size=min(n_to_pay, len(open_indices)), replace=False)
            doc_p = doc_pareggio_counter
            doc_pareggio_counter += 1
            for idx in chosen:
                paid_mask[idx] = True
                doc_num = df.loc[idx, "Numero documento"]
                orig_idx = original_order[doc_num]
                invoices[orig_idx]["Data di pareggio"] = payment_date
                invoices[orig_idx]["Doc. pareggio"] = doc_p
                invoices[orig_idx]["Es. pareggio"] = payment_date.year
        
        current = date(current.year + (current.month // 12), (current.month % 12) + 1, 1)
    
    return invoices


def generate_mock_dataset(
    n_clients_per_type: int = N_CLIENTS_PER_TYPE,
    years: int = YEARS_OF_HISTORY,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate complete mock SAP dataset with deterministic payment patterns.
    
    Returns:
        DataFrame in SAP BSAD format (30 columns)
    """
    rng = np.random.default_rng(seed)
    
    end_date = date.today()
    start_date = date(end_date.year - years, end_date.month, 1)
    
    clients = generate_client_ids(n_clients_per_type)
    
    all_invoices = []
    
    for _, row in clients.iterrows():
        cliente = row["cliente"]
        ctype = row["client_type"]
        
        invoices = generate_invoices_for_client(cliente, start_date, end_date, rng)
        
        if ctype == "FIFO":
            invoices = apply_fifo_payments(invoices, end_date, rng)
        elif ctype == "LARGEST":
            invoices = apply_largest_payments(invoices, end_date, rng)
        else:
            invoices = apply_random_payments(invoices, end_date, rng)
        
        all_invoices.extend(invoices)
    
    df = pd.DataFrame(all_invoices)
    df = df[SAP_COLUMNS]
    
    LOGGER.info("Generated %d rows in SAP BSAD format", len(df))
    
    return df


def save_mock_data(project_root: Path) -> Path:
    """Generate and save mock data to data/mock/ directory."""
    mock_dir = project_root / "data" / "mock"
    mock_dir.mkdir(parents=True, exist_ok=True)
    
    df = generate_mock_dataset()
    
    xlsx_path = mock_dir / "sap_mock_bsad.xlsx"
    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    
    csv_path = mock_dir / "sap_mock_bsad.csv"
    df.to_csv(csv_path, index=False)
    
    clients_path = mock_dir / "mock_clients_ground_truth.csv"
    clients = generate_client_ids()
    clients.to_csv(clients_path, index=False)
    
    LOGGER.info("Saved mock SAP BSAD to %s", xlsx_path)
    LOGGER.info("Saved ground truth to %s", clients_path)
    
    paid_count = df["Data di pareggio"].notna().sum()
    open_count = df["Data di pareggio"].isna().sum()
    
    print(f"\n✅ Mock data generated successfully!")
    print(f"   📄 XLSX: {xlsx_path}")
    print(f"   📄 CSV:  {csv_path}")
    print(f"   📄 Ground Truth: {clients_path}")
    print(f"\n   Stats:")
    print(f"   - Clients: {len(clients)} (33 FIFO, 33 LARGEST, 33 RANDOM)")
    print(f"   - Total rows: {len(df)}")
    print(f"   - Paid invoices: {paid_count}")
    print(f"   - Open invoices: {open_count}")
    print(f"   - Columns: {len(df.columns)} (matching SAP BSAD format)")
    
    return xlsx_path


def main() -> None:
    """Main entry point for mock data generation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    project_root = Path(__file__).resolve().parents[2]
    save_mock_data(project_root)


if __name__ == "__main__":
    main()
