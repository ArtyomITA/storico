"""
Month Full Payment Coverage metric.

Separates "same registration month" (regmonth) from "paid all invoices in the month".
Calculates per-event and per-client coverage metrics.
"""
from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)


def compute_month_coverage_metrics(
    events: pd.DataFrame,
    event_items: pd.DataFrame,
    line_items: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate month full payment coverage metrics per client.
    
    For each event E with data_pareggio = d:
      - Target month YM = (cliente, year(d), month(d))
      - N_total = # open invoices with data_documento in YM
      - N_paid = # of those present in event_items for E
      - coverage = N_paid / max(1, N_total)
    
    Client metrics:
      - month_coverage_mean: average coverage across events
      - month_coverage_p95: 95th percentile coverage
      - month_coverage_full_rate: % events with coverage = 1.0 (paid entire month)
    
    Args:
        events: Events dataframe with cliente, event_key, data_pareggio
        event_items: Event items with event_key, numero_documento
        line_items: Line items with cliente, data_documento, numero_documento
    
    Returns:
        DataFrame with columns: cliente, month_coverage_mean, month_coverage_p95, month_coverage_full_rate
    """
    if events.empty or event_items.empty or line_items.empty:
        LOGGER.warning("Empty input dataframes for month coverage calculation")
        return pd.DataFrame(columns=["cliente", "month_coverage_mean", "month_coverage_p95", "month_coverage_full_rate"])
    
    # Ensure data_pareggio is datetime
    if "data_pareggio" not in events.columns:
        LOGGER.warning("data_pareggio column missing in events")
        return pd.DataFrame(columns=["cliente", "month_coverage_mean", "month_coverage_p95", "month_coverage_full_rate"])
    
    events_copy = events.copy()
    events_copy["data_pareggio"] = pd.to_datetime(events_copy["data_pareggio"], errors="coerce")
    events_copy = events_copy.dropna(subset=["data_pareggio"])
    
    if events_copy.empty:
        LOGGER.warning("No valid data_pareggio in events")
        return pd.DataFrame(columns=["cliente", "month_coverage_mean", "month_coverage_p95", "month_coverage_full_rate"])
    
    # Extract year-month from data_pareggio
    events_copy["year_month"] = events_copy["data_pareggio"].dt.to_period("M")
    
    # Ensure data_documento is datetime in line_items
    line_items_copy = line_items.copy()
    if "data_documento" in line_items_copy.columns:
        line_items_copy["data_documento"] = pd.to_datetime(line_items_copy["data_documento"], errors="coerce")
        line_items_copy["year_month_doc"] = line_items_copy["data_documento"].dt.to_period("M")
    else:
        LOGGER.warning("data_documento column missing in line_items")
        return pd.DataFrame(columns=["cliente", "month_coverage_mean", "month_coverage_p95", "month_coverage_full_rate"])
    
    # For each event, calculate coverage
    coverage_records = []
    
    for _, event_row in events_copy.iterrows():
        cliente = event_row["cliente"]
        event_key = event_row["event_key"]
        year_month = event_row["year_month"]
        
        # N_total: all invoices for this client in this year-month
        client_invoices_in_month = line_items_copy[
            (line_items_copy["cliente"] == cliente) &
            (line_items_copy["year_month_doc"] == year_month)
        ]
        n_total = len(client_invoices_in_month["numero_documento"].unique())
        
        if n_total == 0:
            # No invoices in this month (shouldn't happen often)
            continue
        
        # N_paid: invoices from this month that are in the event
        event_docs = event_items[event_items["event_key"] == event_key]["numero_documento"].unique()
        month_docs = client_invoices_in_month["numero_documento"].unique()
        n_paid = len(set(event_docs) & set(month_docs))
        
        coverage = n_paid / max(1, n_total)
        
        coverage_records.append({
            "cliente": cliente,
            "event_key": event_key,
            "year_month": str(year_month),
            "n_total": n_total,
            "n_paid": n_paid,
            "coverage": coverage,
        })
    
    if not coverage_records:
        LOGGER.warning("No coverage records calculated")
        return pd.DataFrame(columns=["cliente", "month_coverage_mean", "month_coverage_p95", "month_coverage_full_rate"])
    
    coverage_df = pd.DataFrame(coverage_records)
    
    # Aggregate per client
    client_metrics = coverage_df.groupby("cliente").agg(
        month_coverage_mean=("coverage", "mean"),
        month_coverage_p95=("coverage", lambda x: x.quantile(0.95) if len(x) > 0 else 0.0),
        month_coverage_full_rate=("coverage", lambda x: (x == 1.0).mean()),
    ).reset_index()
    
    LOGGER.info(f"Computed month coverage for {len(client_metrics)} clients")
    
    return client_metrics


def main(args: list[str] | None = None) -> None:
    """
    CLI entry point for computing month coverage metrics.
    
    Usage:
        python -m src.analytics.month_coverage \\
            --events data/curated/events.parquet \\
            --event-items data/curated/event_items.parquet \\
            --line-items data/curated/line_items.parquet \\
            --output data/curated/month_coverage.parquet
    """
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Compute month full payment coverage metrics")
    parser.add_argument("--events", type=Path, required=True, help="Path to events.parquet")
    parser.add_argument("--event-items", type=Path, required=True, help="Path to event_items.parquet")
    parser.add_argument("--line-items", type=Path, required=True, help="Path to line_items.parquet")
    parser.add_argument("--output", type=Path, required=True, help="Output path for month_coverage.parquet")
    
    parsed = parser.parse_args(args)
    
    LOGGER.info(f"Loading events from {parsed.events}")
    events = pd.read_parquet(parsed.events)
    
    LOGGER.info(f"Loading event_items from {parsed.event_items}")
    event_items = pd.read_parquet(parsed.event_items)
    
    LOGGER.info(f"Loading line_items from {parsed.line_items}")
    line_items = pd.read_parquet(parsed.line_items)
    
    LOGGER.info("Computing month coverage metrics")
    coverage_df = compute_month_coverage_metrics(events, event_items, line_items)
    
    LOGGER.info(f"Writing {len(coverage_df)} rows to {parsed.output}")
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    coverage_df.to_parquet(parsed.output, index=False)
    
    LOGGER.info("Month coverage computation complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
