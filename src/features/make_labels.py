"""Generate pointwise training labels for invoice-event pairs."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)

MAX_NEGATIVES_PER_EVENT = 20

def configure_logging(level: int | str | None = logging.INFO) -> Path:
    """Ensure standard pipeline logging handlers are configured."""

    return setup_logging(level)

def load_curated_inputs(project_root: Path) -> Dict[str, pd.DataFrame]:
    """Load curated datasets required for label generation."""
    curated_dir = project_root / "data" / "curated"
    paths = {
        "events": curated_dir / "events.parquet",
        "event_items": curated_dir / "event_items.parquet",
    }
    dataframes: Dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required input file: {path}")
        LOGGER.info("Loading %s from %s", name, path)
        dataframes[name] = pd.read_parquet(path)
    return dataframes

def prepare_event_items(event_items: pd.DataFrame) -> pd.DataFrame:
    """Prepare event items with original event key preserved."""
    items = event_items.copy()
    items["source_event_key"] = items["event_key"].astype("string")
    return items.drop(columns=["event_key"])

def append_labels_for_event(
    event_row: pd.Series,
    customer_items: pd.DataFrame,
    results: List[pd.DataFrame],
) -> None:
    """Append positive and negative samples for a single event."""
    event_key: str = event_row["event_key"]
    event_date: pd.Timestamp = event_row["data_pareggio"]

    positives = customer_items[customer_items["source_event_key"] == event_key].copy()
    if positives.empty:
        LOGGER.warning("Event %s for customer %s has no positive rows.", event_key, event_row["cliente"])
        return

    positives["event_key"] = event_key
    positives["label"] = 1
    results.append(positives)

    mask_open = (
        (customer_items["data_documento"] <= event_date)
        & (
            customer_items["data_pareggio"].isna()
            | (customer_items["data_pareggio"] >= event_date)
        )
    )
    candidates = customer_items.loc[mask_open].copy()
    candidates = candidates[
        candidates["source_event_key"].fillna("") != event_key
    ]

    if candidates.empty:
        LOGGER.debug("No negative candidates for event %s", event_key)
        return

    tipo_set = set(positives["tipo_documento"].dropna())
    periodo_set = set(positives["periodo_contabile"].dropna())

    candidates["_priority"] = (
        candidates["tipo_documento"].isin(tipo_set).astype(int)
        + candidates["periodo_contabile"].isin(periodo_set).astype(int)
    )

    candidates = candidates.sort_values(
        ["_priority", "data_documento", "numero_documento"],
        ascending=[False, False, True],
    )

    negatives = candidates.head(MAX_NEGATIVES_PER_EVENT).copy()
    negatives.drop(columns=["_priority"], inplace=True)
    negatives["event_key"] = event_key
    negatives["label"] = 0
    results.append(negatives)

def build_pointwise_labels(project_root: Path) -> pd.DataFrame:
    """Produce the pointwise training dataset with balanced negatives."""
    dataframes = load_curated_inputs(project_root)
    events = dataframes["events"].copy()
    event_items = prepare_event_items(dataframes["event_items"])

    results: List[pd.DataFrame] = []
    events = events.sort_values(["cliente", "data_pareggio", "doc_pareggio"])

    for cliente, customer_events in events.groupby("cliente"):
        customer_items = event_items[event_items["cliente"] == cliente].copy()
        for _, event_row in customer_events.iterrows():
            append_labels_for_event(event_row, customer_items, results)

    if not results:
        LOGGER.warning("No labelled samples generated.")
        return pd.DataFrame()

    labelled = pd.concat(results, ignore_index=True)
    labelled["label"] = labelled["label"].astype("Int64")
    return labelled

def summarise_labels(labelled: pd.DataFrame) -> None:
    """Print acceptance metrics for generated labels."""
    if labelled.empty:
        print("No samples generated.")
        return

    event_counts = labelled["event_key"].dropna().nunique()
    positives = int((labelled["label"] == 1).sum())
    negatives = int((labelled["label"] == 0).sum())
    ratio = negatives / positives if positives else float("nan")

    print(
        "Label summary: #eventi={:d}, #positivi={:d}, #negativi={:d}, ratio={:.2f}".format(
            event_counts, positives, negatives, ratio
        )
    )

def save_labels(labelled: pd.DataFrame, project_root: Path) -> None:
    """Persist pointwise labels dataset to disk."""
    curated_dir = project_root / "data" / "curated"
    output_path = curated_dir / "train_pointwise.parquet"
    curated_dir.mkdir(parents=True, exist_ok=True)
    labelled.to_parquet(output_path, compression="snappy", index=False)
    LOGGER.info("Saved %s labelled rows to %s", len(labelled), output_path)

def main() -> None:
    """Run label generation end-to-end."""
    configure_logging()
    project_root = Path(__file__).resolve().parents[2]

    labelled = build_pointwise_labels(project_root)
    if labelled.empty:
        print("No labelled samples generated.")
        return

    save_labels(labelled, project_root)
    summarise_labels(labelled)

if __name__ == "__main__":
    main()

