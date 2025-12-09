"""Backtest accuracy of client clearing policies on recent events."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.analytics.pattern_detectors import POLICY_NAMES
from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)
CURATED_DIR = Path("data/curated")
OUTPUT_FILENAME = "client_rules_accuracy.parquet"
TEST_FRACTION = 0.2

@dataclass(frozen=True)
class EventEvaluation:
    cliente: str
    event_key: str
    policy: str
    recall: float
    match_rate: float

def configure_logging(level: int | str | None = logging.INFO) -> Path:
    """Configure shared logging handlers when invoked as CLI."""

    return setup_logging(level)

def load_curated_tables(project_root: Path) -> Dict[str, pd.DataFrame]:
    curated_dir = project_root / CURATED_DIR
    paths = {
        "events": curated_dir / "events.parquet",
        "event_items": curated_dir / "event_items.parquet",
        "pattern_scores": curated_dir / "client_pattern_scores.parquet",
    }
    tables: Dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        LOGGER.info("Loading %s from %s", name, path)
        tables[name] = pd.read_parquet(path)
    return tables

def prepare_document_pool(event_items: pd.DataFrame) -> pd.DataFrame:
    """Prepare document-level dataset with derived attributes."""
    docs = event_items.copy()
    if docs.empty:
        return docs

    docs["cliente"] = docs["cliente"].astype("string")
    docs["numero_documento"] = docs["numero_documento"].astype("string")

    for column in ["data_documento", "data_pareggio", "data_registrazione", "data_acquisizione"]:
        if column in docs.columns:
            docs[column] = pd.to_datetime(docs[column], errors="coerce")

    docs["abs_amount"] = docs["imp_div_int"].abs().fillna(0.0)
    if "periodo_contabile" in docs.columns:
        docs["periodo_num"] = pd.to_numeric(docs["periodo_contabile"], errors="coerce")
    else:
        docs["periodo_num"] = np.nan
    return docs

def dominant_policy_map(pattern_scores: pd.DataFrame) -> Dict[str, str]:
    """Return mapping cliente -> dominant policy based on highest score."""
    if pattern_scores.empty:
        return {}
    policies = {f"{policy}_score": policy for policy in POLICY_NAMES}
    records: Dict[str, str] = {}
    for _, row in pattern_scores.iterrows():
        cliente = str(row.get("cliente"))
        best_policy: Optional[str] = None
        best_score = -np.inf
        for column, policy in policies.items():
            if column not in row or pd.isna(row[column]):
                continue
            score = float(row[column])
            if score > best_score:
                best_score = score
                best_policy = policy
        if best_policy is not None and best_score > -np.inf:
            records[cliente] = best_policy
    return records

def split_holdout_events(events: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Select the most recent TEST_FRACTION events per client as hold-out set."""
    holdout: Dict[str, pd.DataFrame] = {}
    if events.empty:
        return holdout
    events = events.copy()
    events["data_pareggio"] = pd.to_datetime(events["data_pareggio"], errors="coerce")
    events = events.dropna(subset=["data_pareggio"])
    for cliente, group in events.groupby("cliente"):
        group_sorted = group.sort_values("data_pareggio")
        n_events = len(group_sorted)
        if n_events == 0:
            continue
        n_test = max(1, math.ceil(n_events * TEST_FRACTION))
        holdout[str(cliente)] = group_sorted.tail(n_test)
    return holdout

def _rank_candidates_for_policy(policy: str, candidates: pd.DataFrame) -> pd.DataFrame:
    """Return candidate documents ordered according to the specified policy."""
    if candidates.empty:
        return candidates

    if policy == "same_period":
        order_cols = ["same_period_flag", "age_days", "abs_amount"]
        ascending = [False, False, False]
    elif policy == "fifo":
        order_cols = ["age_days", "abs_amount"]
        ascending = [False, False]
    elif policy == "lifo":
        order_cols = ["age_days", "abs_amount"]
        ascending = [True, False]
    elif policy == "largest":
        order_cols = ["abs_amount", "age_days"]
        ascending = [False, False]
    elif policy == "smallest":
        order_cols = ["abs_amount", "age_days"]
        ascending = [True, False]
    else:
        raise ValueError(f"Unknown policy: {policy}")

    return candidates.sort_values(order_cols, ascending=ascending)

def evaluate_event(
    policy: str,
    event_row: pd.Series,
    customer_docs: pd.DataFrame,
    actual_docs: Iterable[str],
) -> Optional[EventEvaluation]:
    """Compute recall@k and match rate for a single event."""
    event_value = event_row.get("data_pareggio")
    if pd.isna(event_value):
        return None
    event_date = pd.Timestamp(event_value)
    if pd.isna(event_date):
        return None

    actual = [str(doc) for doc in actual_docs if pd.notna(doc)]
    if not actual:
        return None
    k = len(actual)

    mask_open = (
        (customer_docs["data_documento"] <= event_date)
        & (
            customer_docs["data_pareggio"].isna()
            | (customer_docs["data_pareggio"] >= event_date)
        )
    )
    candidates = customer_docs.loc[mask_open].copy()
    if candidates.empty:
        return EventEvaluation(
            cliente=str(event_row.get("cliente")),
            event_key=str(event_row.get("event_key")),
            policy=policy,
            recall=0.0,
            match_rate=0.0,
        )

    doc_dates = pd.to_datetime(candidates["data_documento"], errors="coerce")
    candidates["age_days"] = doc_dates.apply(
        lambda dt: int((event_date - dt).days) if pd.notna(dt) else 0
    )
    periodo = candidates["periodo_num"].round().astype("Int64")
    same_period_flag = periodo.notna() & (periodo.astype(float) == event_date.month)
    candidates["same_period_flag"] = same_period_flag.astype(int)

    ranking = _rank_candidates_for_policy(policy, candidates)
    top_docs = ranking.head(k)["numero_documento"].astype("string")
    actual_set = set(actual)
    hits = sum(1 for doc in top_docs if str(doc) in actual_set)
    recall = hits / k if k else 0.0
    match_rate = 1.0 if hits == k else 0.0

    return EventEvaluation(
        cliente=str(event_row.get("cliente")),
        event_key=str(event_row.get("event_key")),
        policy=policy,
        recall=recall,
        match_rate=match_rate,
    )

def compute_rules_accuracy(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    events = tables["events"]
    event_items = tables["event_items"]
    pattern_scores = tables["pattern_scores"]

    doc_pool = prepare_document_pool(event_items)
    policy_lookup = dominant_policy_map(pattern_scores)
    holdout_map = split_holdout_events(events)

    if not policy_lookup:
        LOGGER.warning("No pattern scores available; skipping evaluation.")
        return pd.DataFrame(columns=["cliente", "policy_dom", "rules_recall_at_k", "rules_match_rate", "events_tested"])

    doc_group = doc_pool.groupby("cliente") if not doc_pool.empty else None
    items_by_event = event_items.groupby("event_key") if not event_items.empty else None

    client_records: List[Dict[str, object]] = []

    for cliente, holdout_events in holdout_map.items():
        policy = policy_lookup.get(str(cliente))
        if policy is None:
            continue
        if doc_group is None:
            continue
        if cliente not in doc_group.groups:
            continue
        customer_docs = doc_group.get_group(cliente)
        if customer_docs.empty:
            continue

        evaluations: List[EventEvaluation] = []
        for _, event_row in holdout_events.iterrows():
            event_key = event_row.get("event_key")
            if items_by_event is None:
                continue
            if event_key not in items_by_event.groups:
                continue
            actual_group = items_by_event.get_group(event_key)
            actual_docs = actual_group["numero_documento"].dropna().astype("string").tolist()
            evaluation = evaluate_event(policy, event_row, customer_docs, actual_docs)
            if evaluation:
                evaluations.append(evaluation)

        if not evaluations:
            continue

        recall_values = [ev.recall for ev in evaluations]
        match_values = [ev.match_rate for ev in evaluations]
        client_records.append(
            {
                "cliente": str(cliente),
                "policy_dom": policy,
                "rules_recall_at_k": float(np.mean(recall_values)),
                "rules_match_rate": float(np.mean(match_values)),
                "events_tested": len(evaluations),
            }
        )

    columns = ["cliente", "policy_dom", "rules_recall_at_k", "rules_match_rate", "events_tested"]
    return pd.DataFrame(client_records, columns=columns)

def save_results(df: pd.DataFrame, project_root: Path) -> Path:
    curated_dir = project_root / CURATED_DIR
    curated_dir.mkdir(parents=True, exist_ok=True)
    output_path = curated_dir / OUTPUT_FILENAME
    df.to_parquet(output_path, compression="snappy", index=False)
    LOGGER.info("Saved %s rows to %s", len(df), output_path)
    return output_path

def evaluate_holdout_accuracy(cliente: str) -> Optional[float]:
    """Compute hold-out accuracy for a specific client and return recall@k."""
    project_root = Path(__file__).resolve().parents[2]
    tables = load_curated_tables(project_root)
    df = compute_rules_accuracy(tables)
    if df.empty:
        return None
    save_results(df, project_root)
    row = df[df["cliente"] == str(cliente)]
    if row.empty:
        return None
    return float(row.iloc[0]["rules_recall_at_k"])

def main() -> None:
    configure_logging()
    project_root = Path(__file__).resolve().parents[2]
    tables = load_curated_tables(project_root)
    df = compute_rules_accuracy(tables)
    if df.empty:
        LOGGER.warning("No hold-out evaluations were produced.")
        return
    save_results(df, project_root)
    print(df.head(3).to_string(index=False))

if __name__ == "__main__":
    main()

