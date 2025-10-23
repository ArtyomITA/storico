"""Pattern detection and policy scoring for client clearing behaviour."""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)

BASE_POLICY_NAMES = ["fifo", "lifo", "same_period", "largest", "smallest"]
EXTRA_POLICY_NAMES = ["eomfirst", "regmonth", "quarter", "seqblock", "typeprio"]
POLICY_NAMES = [*BASE_POLICY_NAMES, *EXTRA_POLICY_NAMES]

EOM_THRESHOLD_DAYS = 3
SEQ_BLOCK_MIN_LEN = 3
TYPE_PRIORITY_WINDOW_DAYS = 90
PERIODICITY_TOLERANCE_DAYS = 2
PER_EVENT_OUTPUT = "per_event_policy_scores.parquet"

PERIODICITY_WINDOWS: Dict[str, int] = {
    "weekly": 7,
    "biweekly": 14,
    "monthly": 30,
}


@dataclass
class PolicyRanking:
    """Container for policy recall values computed on a single event."""

    cliente: str
    event_key: str
    data_evento: pd.Timestamp
    recalls: Dict[str, float]


def _annotate_doc_sequences(group: pd.DataFrame) -> pd.DataFrame:
    """Annotate consecutive document sequences within a client-month group."""

    if group.empty:
        return pd.DataFrame(
            {
                "docnum_is_consecutive_block": pd.Series(dtype=int),
                "seq_block_len_client_month": pd.Series(dtype=int),
            }
        )

    working = group.sort_values("numero_documento").copy()
    doc_values = pd.to_numeric(working["numero_documento"], errors="coerce").to_numpy(dtype=float)
    prev_values = np.roll(doc_values, 1)
    prev_values[0] = np.nan

    valid = ~np.isnan(doc_values)
    valid_prev = ~np.isnan(prev_values)
    valid_diff = valid & valid_prev

    diff = np.zeros_like(doc_values, dtype=float)
    diff[valid_diff] = doc_values[valid_diff] - prev_values[valid_diff]
    consecutive = valid_diff & np.isclose(diff, 1.0, atol=1e-6)

    block_break = ~consecutive
    if block_break.size:
        block_break[0] = True

    block_id = np.cumsum(block_break.astype(int)) - 1 if block_break.size else np.array([0])
    if block_id.size == 0:
        block_id = np.array([0])

    block_counts = np.bincount(block_id if block_id.size else np.array([0]))
    seq_lengths = block_counts[block_id] if block_id.size else np.array([1])

    is_numeric = ~np.isnan(doc_values)
    consecutive_flag = np.where(is_numeric & (seq_lengths >= SEQ_BLOCK_MIN_LEN), 1, 0)

    annotated = pd.DataFrame(
        {
            "docnum_is_consecutive_block": consecutive_flag.astype(int),
            "seq_block_len_client_month": seq_lengths.astype(int),
        },
        index=working.index,
    )

    return annotated.loc[group.index]


def prepare_doc_map(line_items: pd.DataFrame) -> pd.DataFrame:
    """Create a document map with derived fields for policy ranking."""
    doc_map = line_items.copy()
    doc_map["cliente"] = doc_map["cliente"].astype("string")
    doc_map["numero_documento"] = doc_map["numero_documento"].astype("string")

    for column in ["data_documento", "data_pareggio", "data_registrazione", "data_acquisizione"]:
        if column in doc_map.columns:
            doc_map[column] = pd.to_datetime(doc_map[column])

    doc_map["periodo_num"] = pd.to_numeric(doc_map["periodo_contabile"], errors="coerce")
    doc_map["abs_amount"] = doc_map["imp_div_int"].abs().fillna(0.0)

    doc_map["docnum_is_consecutive_block"] = 0
    doc_map["seq_block_len_client_month"] = 1
    doc_map["_doc_month"] = doc_map["data_documento"].dt.to_period("M")
    valid_seq_mask = doc_map["_doc_month"].notna()
    if valid_seq_mask.any():
        sequence_annotations = (
            doc_map.loc[valid_seq_mask]
            .groupby(["cliente", "_doc_month"], group_keys=False, sort=False)
            .apply(_annotate_doc_sequences)
        )
        if not sequence_annotations.empty:
            doc_map.loc[sequence_annotations.index, [
                "docnum_is_consecutive_block",
                "seq_block_len_client_month",
            ]] = sequence_annotations
    doc_map.drop(columns=["_doc_month"], inplace=True)
    doc_map["docnum_is_consecutive_block"] = doc_map["docnum_is_consecutive_block"].fillna(0).astype(int)
    doc_map["seq_block_len_client_month"] = doc_map["seq_block_len_client_month"].fillna(1).astype(int)

    # Pre-compute amount deciles per client to reuse when needed.
    def _decile_transform(col: pd.Series) -> pd.Series:
        if col.empty:
            return pd.Series(0, index=col.index, dtype=int)
        ranks = col.rank(method="first", pct=True)
        deciles = np.floor(ranks * 10).clip(0, 9).astype(int)
        return pd.Series(deciles, index=col.index, dtype=int)

    doc_map["amt_decile_client"] = (
        doc_map.groupby("cliente")["abs_amount"].transform(_decile_transform).fillna(0).astype(int)
    )

    return doc_map


def prepare_candidate_table(
    events: pd.DataFrame,
    train_features: pd.DataFrame,
    train_pointwise: pd.DataFrame,
) -> pd.DataFrame:
    """Merge engineered features with pointwise raw data for event candidates."""
    if train_features.empty:
        return pd.DataFrame()

    features = train_features.copy()
    features["cliente"] = features["cliente"].astype("string")
    features["numero_documento"] = features["numero_documento"].astype("string")

    raw_cols = [
        "cliente",
        "event_key",
        "numero_documento",
        "data_documento",
        "data_registrazione",
        "periodo_contabile",
        "tipo_documento",
        "imp_div_int",
    ]
    raw = train_pointwise.loc[:, [col for col in raw_cols if col in train_pointwise.columns]].copy()
    if raw.empty:
        return pd.DataFrame()

    raw["cliente"] = raw["cliente"].astype("string")
    raw["numero_documento"] = raw["numero_documento"].astype("string")
    for column in ["data_documento", "data_registrazione"]:
        if column in raw.columns:
            raw[column] = pd.to_datetime(raw[column])
    if "periodo_contabile" in raw.columns:
        raw["periodo_contabile_num"] = pd.to_numeric(raw["periodo_contabile"], errors="coerce")
    else:
        raw["periodo_contabile_num"] = np.nan

    event_dates = events.loc[:, ["cliente", "event_key", "data_pareggio"]].copy()
    event_dates["cliente"] = event_dates["cliente"].astype("string")
    event_dates["data_pareggio"] = pd.to_datetime(event_dates["data_pareggio"])

    merged = features.merge(
        raw.drop_duplicates(subset=["cliente", "event_key", "numero_documento"]),
        on=["cliente", "event_key", "numero_documento"],
        how="left",
        suffixes=("", "_raw"),
    )
    merged = merged.merge(event_dates, on=["cliente", "event_key"], how="left")
    merged["abs_amount"] = merged["imp_div_int"].abs().fillna(0.0)

    return merged


def compute_periodicity_metrics(events: pd.DataFrame) -> pd.DataFrame:
    """Compute periodicity ratios for each client based on event cadence."""

    columns = ["cliente", *[f"periodicity_{name}" for name in PERIODICITY_WINDOWS]]
    if events.empty:
        return pd.DataFrame(columns=columns)

    working = events.dropna(subset=["cliente", "data_pareggio"]).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)

    working["cliente"] = working["cliente"].astype("string")
    working["data_pareggio"] = pd.to_datetime(working["data_pareggio"])

    records: List[Dict[str, object]] = []
    tolerance = PERIODICITY_TOLERANCE_DAYS
    for cliente, group in working.groupby("cliente"):
        group = group.sort_values("data_pareggio")
        deltas = group["data_pareggio"].diff().dt.days.dropna()
        metrics: Dict[str, float] = {}
        if deltas.empty:
            metrics = {f"periodicity_{name}": 0.0 for name in PERIODICITY_WINDOWS}
        else:
            for name, window in PERIODICITY_WINDOWS.items():
                mask = deltas.between(window - tolerance, window + tolerance)
                metrics[f"periodicity_{name}"] = float(mask.mean()) if not mask.empty else 0.0
        records.append({"cliente": str(cliente), **metrics})

    return pd.DataFrame(records, columns=columns)


def configure_logging(level: int | str | None = logging.INFO) -> Path:
    """Configure shared logging handlers for CLI execution."""

    return setup_logging(level)


def load_curated_tables(project_root: Path) -> Dict[str, pd.DataFrame]:
    """Load curated parquet tables required by the analytics pipeline."""

    curated_dir = project_root / "data" / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "events": "events.parquet",
        "event_items": "event_items.parquet",
        "line_items": "line_items.parquet",
        "train_features": "train_features.parquet",
        "train_pointwise": "train_pointwise.parquet",
    }

    tables: Dict[str, pd.DataFrame] = {}
    for key, filename in sources.items():
        path = curated_dir / filename
        if path.exists():
            tables[key] = pd.read_parquet(path)
        else:
            LOGGER.warning("Missing curated table %s at %s", key, path)
            tables[key] = pd.DataFrame()
    return tables


def compute_type_priority_lookup(
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    window_days: int = TYPE_PRIORITY_WINDOW_DAYS,
) -> Dict[str, Optional[str]]:
    """Determine dominant document type per event using a rolling historical window."""
    lookup: Dict[str, Optional[str]] = {}
    if events.empty or candidates.empty:
        return lookup

    window = np.timedelta64(window_days, "D")
    events = events.copy()
    events["cliente"] = events["cliente"].astype("string")
    events["data_pareggio"] = pd.to_datetime(events["data_pareggio"])

    docs = candidates.loc[
        :, [
            "cliente",
            "numero_documento",
            "data_documento",
            "tipo_documento",
        ]
    ].copy()
    docs["cliente"] = docs["cliente"].astype("string")
    docs["tipo_documento"] = docs["tipo_documento"].astype("string")
    docs["data_documento"] = pd.to_datetime(docs["data_documento"])
    docs = docs.dropna(subset=["cliente", "data_documento", "tipo_documento"])
    if docs.empty:
        for event_key in events["event_key"].astype("string"):
            lookup[str(event_key)] = None
        return lookup

    docs = docs.sort_values("data_documento")
    docs_by_client = docs.groupby("cliente")

    for cliente, event_group in events.groupby("cliente"):
        if cliente not in docs_by_client.groups:
            for _, event_row in event_group.iterrows():
                lookup[str(event_row["event_key"])] = None
            continue

        client_docs = docs_by_client.get_group(cliente)
        dates = client_docs["data_documento"].to_numpy(dtype="datetime64[ns]")
        types = client_docs["tipo_documento"].to_numpy(dtype=object)
        start = 0
        end = 0
        counts: defaultdict[str, int] = defaultdict(int)

        for _, event_row in event_group.sort_values("data_pareggio").iterrows():
            event_key = str(event_row["event_key"])
            event_date = event_row["data_pareggio"]
            if pd.isna(event_date):
                lookup[event_key] = None
                continue

            event_ts = event_date.to_datetime64()
            cutoff = event_ts - window

            while end < len(dates) and dates[end] <= event_ts:
                doc_type = types[end]
                if isinstance(doc_type, str) and doc_type:
                    if dates[end] >= cutoff:
                        counts[doc_type] += 1
                end += 1

            while start < end and dates[start] < cutoff:
                doc_type = types[start]
                if isinstance(doc_type, str) and doc_type in counts:
                    counts[doc_type] -= 1
                    if counts[doc_type] <= 0:
                        counts.pop(doc_type, None)
                start += 1

            if counts:
                top_type = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            else:
                top_type = None
            lookup[event_key] = top_type

    return lookup


def _policy_rankings_for_event(
    event_row: pd.Series,
    candidates: pd.DataFrame,
    top_doc_type: Optional[str],
) -> Optional[PolicyRanking]:
    """Compute policy recall scores for a single event."""
    raw_event_date = event_row.get("data_pareggio")
    event_date = pd.NaT
    if isinstance(raw_event_date, pd.Timestamp):
        event_date = raw_event_date
    elif isinstance(raw_event_date, (datetime, date)):
        event_date = pd.Timestamp(raw_event_date)
    elif isinstance(raw_event_date, np.datetime64):
        event_date = pd.Timestamp(raw_event_date)
    elif isinstance(raw_event_date, str):
        event_date = pd.to_datetime(raw_event_date, errors="coerce")
    if pd.isna(event_date):
        return None

    if candidates.empty or "numero_documento" not in candidates.columns:
        return None

    working = candidates.copy()
    working["numero_documento"] = working["numero_documento"].astype("string")

    label_series = working["label"] if "label" in working.columns else pd.Series(0, index=working.index)
    label_series = pd.to_numeric(label_series, errors="coerce").fillna(0).astype(int)
    actual_docs = working.loc[label_series == 1, "numero_documento"]
    actual_set = set(actual_docs.dropna().astype("string"))
    if not actual_set:
        return None

    doc_dates_source = (
        working["data_documento"]
        if "data_documento" in working.columns
        else pd.Series(pd.NaT, index=working.index)
    )
    doc_dates = pd.to_datetime(doc_dates_source, errors="coerce")
    doc_dates = doc_dates.fillna(event_date)
    working["data_documento"] = doc_dates

    reg_dates_source = (
        working["data_registrazione"]
        if "data_registrazione" in working.columns
        else pd.Series(pd.NaT, index=working.index)
    )
    reg_dates = pd.to_datetime(reg_dates_source, errors="coerce")
    working["data_registrazione"] = reg_dates

    if "abs_amount" in working.columns:
        abs_amount_source = working["abs_amount"]
    else:
        abs_amount_source = working.get(
            "imp_div_int", pd.Series(0.0, index=working.index, dtype=float)
        )
    working["abs_amount"] = (
        pd.to_numeric(abs_amount_source, errors="coerce")
        .fillna(0.0)
        .abs()
    )

    working["same_period_flag"] = np.where(
        doc_dates.dt.to_period("M") == event_date.to_period("M"),
        1,
        0,
    )

    event_dt64 = event_date.to_datetime64()
    doc_dt64 = doc_dates.to_numpy(dtype="datetime64[ns]")
    age_delta = event_dt64 - doc_dt64
    age_values = age_delta.astype("timedelta64[D]").astype(float)
    working["age_days"] = pd.Series(np.nan_to_num(age_values, nan=0.0), index=working.index)

    event_month = int(event_date.month)
    event_year = int(event_date.year)
    event_quarter = event_date.to_period("Q")

    days_in_month = doc_dates.dt.days_in_month.astype(float)
    day_of_month = doc_dates.dt.day.astype(float)
    working["doc_days_to_eom"] = np.nan_to_num(days_in_month - day_of_month, nan=EOM_THRESHOLD_DAYS + 5)

    event_is_eom = ((event_date + MonthEnd(0)) - event_date).days <= EOM_THRESHOLD_DAYS
    working["priority_eom"] = np.where(
        event_is_eom,
        (working["doc_days_to_eom"] <= EOM_THRESHOLD_DAYS).astype(int),
        0,
    )

    reg_ref = reg_dates.where(reg_dates.notna(), doc_dates)
    working["reg_month_match"] = np.where(
        reg_ref.notna()
        & (reg_ref.dt.month == event_month)
        & (reg_ref.dt.year == event_year),
        1,
        0,
    )
    reg_dt64 = reg_ref.to_numpy(dtype="datetime64[ns]")
    reg_gap_delta = reg_dt64 - event_dt64
    reg_gap_values = np.abs(reg_gap_delta.astype("timedelta64[D]").astype(float))
    working["reg_gap_abs"] = pd.Series(
        np.nan_to_num(reg_gap_values, nan=10_000.0), index=working.index
    )

    doc_quarter = doc_dates.dt.to_period("Q")
    working["quarter_match"] = np.where(doc_quarter == event_quarter, 1, 0)
    quarter_gap_delta = doc_dt64 - event_dt64
    quarter_gap_values = np.abs(quarter_gap_delta.astype("timedelta64[D]").astype(float))
    working["quarter_gap"] = pd.Series(
        np.nan_to_num(quarter_gap_values, nan=10_000.0), index=working.index
    )

    seq_len_source = working.get(
        "seq_block_len_client_month", pd.Series(1, index=working.index, dtype=float)
    )
    block_flag_source = working.get(
        "docnum_is_consecutive_block", pd.Series(0, index=working.index, dtype=float)
    )
    working["seq_block_len_client_month"] = (
        pd.to_numeric(seq_len_source, errors="coerce").fillna(1).astype(int)
    )
    working["docnum_is_consecutive_block"] = (
        pd.to_numeric(block_flag_source, errors="coerce").fillna(0).astype(int)
    )
    working["seqblock_priority"] = np.where(
        (working["docnum_is_consecutive_block"] > 0)
        & (working["seq_block_len_client_month"] >= SEQ_BLOCK_MIN_LEN),
        1,
        0,
    )

    types = working.get(
        "tipo_documento", pd.Series(index=working.index, dtype="string")
    ).astype("string")
    if top_doc_type:
        working["type_priority_hit"] = (types == str(top_doc_type)).astype(int)
    else:
        working["type_priority_hit"] = 0

    ranking_specs: Dict[str, Tuple[List[str], List[bool]]] = {
        "fifo": (["data_documento", "abs_amount", "numero_documento"], [True, False, True]),
        "lifo": (["data_documento", "abs_amount", "numero_documento"], [False, False, True]),
        "same_period": (["same_period_flag", "data_documento", "abs_amount"], [False, True, False]),
        "largest": (["abs_amount", "data_documento", "numero_documento"], [False, True, True]),
        "smallest": (["abs_amount", "data_documento", "numero_documento"], [True, True, True]),
        "eomfirst": (["priority_eom", "doc_days_to_eom", "data_documento"], [False, True, True]),
        "regmonth": (["reg_month_match", "reg_gap_abs", "data_documento"], [False, True, True]),
        "quarter": (["quarter_match", "quarter_gap", "data_documento"], [False, True, True]),
        "seqblock": (
            ["seqblock_priority", "seq_block_len_client_month", "data_documento"],
            [False, False, True],
        ),
        "typeprio": (["type_priority_hit", "data_documento", "abs_amount"], [False, True, False]),
    }

    recalls: Dict[str, float] = {}
    k = len(actual_set)
    for policy, (order_cols, asc_flags) in ranking_specs.items():
        filtered = [
            (col, asc)
            for col, asc in zip(order_cols, asc_flags)
            if col in working.columns
        ]
        if not filtered:
            recalls[policy] = 0.0
            continue
        cols, asc_list = zip(*filtered)
        ranking = working.sort_values(list(cols), ascending=list(asc_list), kind="mergesort")
        top_docs = (
            ranking.head(k)["numero_documento"].astype("string")
            if k
            else pd.Series(dtype="string")
        )
        hits = sum(1 for doc in top_docs if doc in actual_set)
        recalls[policy] = hits / k if k else 0.0

    return PolicyRanking(
        cliente=str(event_row.get("cliente")),
        event_key=str(event_row.get("event_key")),
        data_evento=event_date,
        recalls=recalls,
    )


def compute_policy_metrics(
    events: pd.DataFrame,
    candidates: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute aggregated policy scores and per-event recall values."""
    if events.empty or candidates.empty:
        empty_scores = pd.DataFrame(columns=["cliente", *[f"{policy}_score" for policy in POLICY_NAMES]])
        return empty_scores, pd.DataFrame()

    policy_records: List[PolicyRanking] = []
    grouped_candidates = candidates.groupby("event_key")
    type_priority_lookup = compute_type_priority_lookup(events, candidates)

    for _, event_row in events.iterrows():
        event_key = event_row["event_key"]
        if event_key not in grouped_candidates.groups:
            continue
        event_candidates = grouped_candidates.get_group(event_key)
        top_doc_type = type_priority_lookup.get(str(event_key))
        policy_entry = _policy_rankings_for_event(event_row, event_candidates, top_doc_type)
        if policy_entry:
            policy_records.append(policy_entry)

    if not policy_records:
        empty_scores = pd.DataFrame(columns=["cliente", *[f"{policy}_score" for policy in POLICY_NAMES]])
        return empty_scores, pd.DataFrame()

    per_event_rows: List[Dict[str, object]] = []
    for entry in policy_records:
        row = {
            "cliente": entry.cliente,
            "event_key": entry.event_key,
            "data_evento": entry.data_evento,
        }
        for policy in POLICY_NAMES:
            value = entry.recalls.get(policy, 0.0)
            row[policy] = value
            if policy in EXTRA_POLICY_NAMES:
                row[f"policy_{policy}"] = value
        per_event_rows.append(row)

    per_event_df = pd.DataFrame(per_event_rows)

    def iqr(series: pd.Series) -> float:
        valid = series.dropna()
        if valid.empty:
            return 0.0
        q75, q25 = np.percentile(valid, [75, 25])
        return float(q75 - q25)

    aggregates = per_event_df.groupby("cliente")[POLICY_NAMES].agg(["mean", iqr])
    combined_scores = {}
    for policy in POLICY_NAMES:
        mean_series = aggregates[(policy, "mean")]
        iqr_series = aggregates[(policy, "iqr")]
        # Combine mean and dispersion into a single score (higher is better, dispersion penalised).
        combined = mean_series * (1 - np.clip(iqr_series, 0, 1))
        combined_scores[f"{policy}_score"] = combined

    policy_scores = pd.DataFrame(combined_scores).reset_index()
    for policy in EXTRA_POLICY_NAMES:
        alias_col = f"policy_{policy}"
        score_col = f"{policy}_score"
        if score_col in policy_scores.columns:
            policy_scores[alias_col] = policy_scores[score_col]
    return policy_scores, per_event_df


def compute_cardinality_metrics(events: pd.DataFrame) -> pd.DataFrame:
    """Derive client-level statistics over event cardinalities."""
    if events.empty:
        columns = [
            "cliente",
            "P_k1",
            "P_k2",
            "P_kge3",
            "k_mode",
            "k_entropy",
        ]
        return pd.DataFrame(columns=columns)

    records: List[Dict[str, object]] = []
    for cliente, group in events.groupby("cliente"):
        sizes = group["n_righe_evento"].dropna().astype(int)
        total = len(sizes)
        if total == 0:
            continue

        p_k1 = float((sizes == 1).mean())
        p_k2 = float((sizes == 2).mean())
        p_kge3 = float((sizes >= 3).mean())

        value_counts = sizes.value_counts(normalize=True)
        k_mode = int(value_counts.idxmax()) if not value_counts.empty else 0
        probs = value_counts.to_numpy()
        if probs.size <= 1:
            k_entropy = 0.0
        else:
            entropy = float(np.sum(-probs * np.log(probs + 1e-12)))
            k_entropy = float(entropy / np.log(probs.size))

        records.append(
            {
                "cliente": str(cliente),
                "P_k1": p_k1,
                "P_k2": p_k2,
                "P_kge3": p_kge3,
                "k_mode": float(k_mode),
                "k_entropy": k_entropy,
            }
        )

    return pd.DataFrame(records)


def compute_amount_preference(train_features: pd.DataFrame) -> pd.Series:
    """Compute point-biserial correlation between amount decile and label per client."""
    if train_features.empty:
        return pd.Series(dtype=float)

    tf = train_features.dropna(subset=["amt_decile_client"]).copy()
    tf["amt_decile_client"] = pd.to_numeric(tf["amt_decile_client"], errors="coerce")
    tf = tf.dropna(subset=["amt_decile_client"])
    tf["label"] = tf["label"].astype(int)

    slopes = {}
    for cliente, group in tf.groupby("cliente"):
        values = group["amt_decile_client"].to_numpy(dtype=float)
        labels = group["label"].to_numpy(dtype=float)
        if values.size < 2 or labels.sum() == 0 or labels.sum() == values.size:
            slopes[str(cliente)] = 0.0
            continue

        mean_all = float(values.mean())
        std_all = float(values.std(ddof=0))
        if std_all == 0:
            slopes[str(cliente)] = 0.0
            continue

        mask = labels == 1
        mean_pos = float(values[mask].mean())
        mean_neg = float(values[~mask].mean())
        p_pos = float(mask.mean())
        p_neg = 1 - p_pos
        slopes[str(cliente)] = ((mean_pos - mean_neg) / std_all) * math.sqrt(p_pos * p_neg)

    return pd.Series(slopes, name="amount_pref_slope")


def compute_stability_index(per_event_scores: pd.DataFrame, window: int = 3) -> pd.Series:
    """Compute stability index (1 stable, 0 unstable) per client."""
    if per_event_scores.empty:
        return pd.Series(dtype=float)

    results = {}
    policy_cols = [col for col in per_event_scores.columns if col in POLICY_NAMES]
    for cliente, group in per_event_scores.groupby("cliente"):
        group = group.sort_values("data_evento")
        std_values: List[float] = []
        for policy in policy_cols:
            series = group[policy].astype(float)
            if series.count() < 2:
                continue
            rolling_std = series.rolling(window=window, min_periods=2).std()
            valid = rolling_std.dropna()
            if not valid.empty:
                std_values.append(float(valid.mean()))
        if std_values:
            avg_std = float(np.mean(std_values))
            stability = 1.0 / (1.0 + avg_std)
        else:
            stability = 1.0
        results[str(cliente)] = stability
    return pd.Series(results, name="stability_index")


def build_client_pattern_scores(
    events: pd.DataFrame,
    policy_scores: pd.DataFrame,
    cardinality_df: pd.DataFrame,
    train_features: pd.DataFrame,
    per_event_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble complete client-level pattern detection table."""
    periodicity_df = compute_periodicity_metrics(events)
    amount_pref = compute_amount_preference(train_features)
    stability = compute_stability_index(per_event_scores)

    merged = periodicity_df.merge(policy_scores, on="cliente", how="outer")
    merged = merged.merge(cardinality_df, on="cliente", how="outer")
    merged = merged.merge(amount_pref.rename_axis("cliente").reset_index(), on="cliente", how="left")
    merged = merged.merge(stability.rename_axis("cliente").reset_index(), on="cliente", how="left")

    value_cols = [col for col in merged.columns if col != "cliente"]
    merged[value_cols] = merged[value_cols].apply(pd.to_numeric, errors="coerce")
    merged[value_cols] = merged[value_cols].fillna(0.0)

    return merged


def infer_pattern_label(row: pd.Series) -> Tuple[str, float]:
    """Infer a human-readable pattern label and confidence from scores."""
    if pd.isna(row).all():
        return "irregolare", 0.0

    periodicity_monthly = row.get("periodicity_monthly", 0.0)
    periodicity_biweekly = row.get("periodicity_biweekly", 0.0)
    periodicity_weekly = row.get("periodicity_weekly", 0.0)
    same_period = row.get("same_period_score", 0.0)

    if periodicity_monthly > 0.6 and same_period > 0.6:
        conf = float(np.mean([periodicity_monthly, same_period]))
        return "mensile", min(conf, 1.0)
    if periodicity_biweekly > 0.6:
        return "quindicinale", float(min(periodicity_biweekly, 1.0))
    if periodicity_weekly > 0.6:
        return "settimanale", float(min(periodicity_weekly, 1.0))

    policy_scores = {policy: row.get(f"{policy}_score", 0.0) for policy in POLICY_NAMES}
    dominant_policy, dominant_score = max(policy_scores.items(), key=lambda kv: kv[1])
    if dominant_score > 0.5:
        return f"{dominant_policy}_driven", float(min(dominant_score, 1.0))

    stability = float(row.get("stability_index", 0.0))
    confidence = max(0.2, min((dominant_score + stability) / 2, 1.0))
    return "irregolare", confidence


def save_client_pattern_scores(df: pd.DataFrame, project_root: Path) -> Path:
    """Persist client pattern scores to curated directory."""
    curated_dir = project_root / "data" / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)
    output_path = curated_dir / "client_pattern_scores.parquet"
    df.to_parquet(output_path, compression="snappy", index=False)
    LOGGER.info("Saved %s client pattern rows to %s", len(df), output_path)
    return output_path


def save_per_event_scores(df: pd.DataFrame, project_root: Path) -> Path:
    """Save per-event policy recall scores for downstream tasks."""
    curated_dir = project_root / "data" / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)
    output_path = curated_dir / PER_EVENT_OUTPUT
    df.to_parquet(output_path, compression="snappy", index=False)
    LOGGER.info("Saved %s per-event policy rows to %s", len(df), output_path)
    return output_path


def print_pattern_examples(scores: pd.DataFrame, per_event_scores: pd.DataFrame, examples: int = 3) -> None:
    """Print sample clients with inferred labels and top policies."""
    if scores.empty:
        print("No pattern scores to display.")
        return

    merged = scores.copy()
    for idx, row in merged.head(examples).iterrows():
        label, confidence = infer_pattern_label(row)
        policy_values = {policy: row.get(f"{policy}_score", 0.0) for policy in POLICY_NAMES}
        top_policies = sorted(policy_values.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_repr = ", ".join(f"{name}={value:.2f}" for name, value in top_policies)
        print(
            f"Cliente {row['cliente']}: label={label}, confidence={confidence:.2f}, top policies: {top_repr}"
        )


def main() -> None:
    """Entry-point for running the pattern detection workflow."""
    configure_logging()
    project_root = Path(__file__).resolve().parents[2]
    tables = load_curated_tables(project_root)

    candidate_table = prepare_candidate_table(
        tables["events"], tables["train_features"], tables["train_pointwise"]
    )
    policy_scores, per_event_scores = compute_policy_metrics(
        tables["events"], candidate_table
    )
    cardinality_df = compute_cardinality_metrics(tables["events"])
    client_scores = build_client_pattern_scores(
        tables["events"],
        policy_scores,
        cardinality_df,
        tables["train_features"],
        per_event_scores,
    )

    save_client_pattern_scores(client_scores, project_root)
    if not per_event_scores.empty:
        save_per_event_scores(per_event_scores, project_root)
    print_pattern_examples(client_scores, per_event_scores)


if __name__ == "__main__":
    main()
