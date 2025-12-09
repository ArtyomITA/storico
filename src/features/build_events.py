"""Build clearing events and event-item relationships from line items."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)

def configure_logging(level: int | str | None = logging.INFO) -> Path:
    """Configure shared logging handlers for the module."""

    return setup_logging(level)

def load_line_items(project_root: Path) -> pd.DataFrame:
    """Load curated line items."""
    path = project_root / "data" / "curated" / "line_items.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing curated line items at {path}")
    LOGGER.info("Loading line items from %s", path)
    return pd.read_parquet(path)

def build_event_key_column(line_items: pd.DataFrame) -> pd.Series:
    """Construct the event_key string for rows with a clearing date."""
    key = pd.Series(pd.NA, index=line_items.index, dtype="string")
    mask = line_items["data_pareggio"].notna()
    key.loc[mask] = (
        line_items.loc[mask, "cliente"].astype(str)
        + "|"
        + line_items.loc[mask, "data_pareggio"].dt.strftime("%Y-%m-%d")
        + "|"
        + line_items.loc[mask, "doc_pareggio"].astype(str)
    )
    return key

def compute_events(event_items: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level metrics."""
    grouped = (
        event_items.dropna(subset=["event_key"])
        .groupby(["cliente", "data_pareggio", "doc_pareggio", "event_key"], dropna=False)
        .agg(
            importo_evento=("imp_div_int", "sum"),
            n_righe_evento=("imp_div_int", "size"),
        )
        .reset_index()
    )

    grouped = grouped.sort_values(["cliente", "data_pareggio", "doc_pareggio"])
    grouped["prev_event_date"] = grouped.groupby("cliente")["data_pareggio"].shift(1)
    grouped["inter_event_gap_days"] = (
        grouped["data_pareggio"] - grouped["prev_event_date"]
    ).dt.days

    return grouped

def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Persist dataframe to Parquet with snappy compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy", index=False)
    LOGGER.info("Saved %s rows to %s", len(df), path)

def print_event_examples(events: pd.DataFrame, limit: int = 3) -> None:
    """Print sample events to satisfy acceptance criteria."""
    sample = events.loc[:, ["event_key", "n_righe_evento", "inter_event_gap_days"]].head(limit)
    if sample.empty:
        print("No events available for preview.")
    else:
        print("Sample events:")
        print(sample.to_string(index=False))

def build_events_tables(project_root: Path) -> None:
    """Execute pipeline to materialise event tables."""
    line_items = load_line_items(project_root)
    line_items["event_key"] = build_event_key_column(line_items)

    event_items = line_items.copy()
    events = compute_events(event_items)

    curated_dir = project_root / "data" / "curated"
    save_parquet(event_items, curated_dir / "event_items.parquet")
    save_parquet(events, curated_dir / "events.parquet")

    print_event_examples(events)

def main() -> None:
    """Entry point when executing as a module."""
    configure_logging()
    project_root = Path(__file__).resolve().parents[2]
    build_events_tables(project_root)

if __name__ == "__main__":
    main()

