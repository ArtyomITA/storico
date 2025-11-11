"""Feature engineering for invoice-event training data."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)

EOM_THRESHOLD_DAYS = 3
ROLLING_WINDOW_DAYS = 180
NEW_CALENDAR_FEATURES = [
    "is_eom_event",
    "days_to_eom",
    "is_quarter_end_event",
    "days_to_qe",
    "dow_event",
    "dow_run_is_thu",
    "weekly_cadence_client",
    "eom_cadence_client",
    "same_fiscal_year",
    "same_quarter_as_event",
]
EVENT_CALENDAR_COLUMNS = {
    "is_eom_event",
    "days_to_eom",
    "is_quarter_end_event",
    "days_to_qe",
    "dow_event",
    "dow_run_is_thu",
    "weekly_cadence_client",
    "eom_cadence_client",
}
EVENT_AMOUNT_FEATURES = [
    "amt_ratio_B",
    "amt_share_rank_event",
    "event_gini_amounts_open",
    "event_amount_scale",
]
FEATURES_TO_LOG = NEW_CALENDAR_FEATURES + EVENT_AMOUNT_FEATURES


def configure_logging(level: int | str | None = logging.INFO) -> Path:
    """Set up shared logging handlers for feature engineering."""

    return setup_logging(level)


def load_curated_datasets(project_root: Path) -> Dict[str, pd.DataFrame]:
    """Load curated artifacts required for feature engineering.
    
    Prompt 3 enhancement: also load line_items for month_coverage calculation.
    """
    curated_dir = project_root / "data" / "curated"
    paths = {
        "train_pointwise": curated_dir / "train_pointwise.parquet",
        "events": curated_dir / "events.parquet",
        "event_items": curated_dir / "event_items.parquet",
        "line_items": curated_dir / "line_items.parquet",  # Added for month_coverage
    }

    dataframes: Dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.exists():
            if name == "line_items":
                # Optional: month_coverage will be skipped if not available
                LOGGER.warning("Optional file %s not found; month_coverage will be zeros", path)
                dataframes[name] = pd.DataFrame()
            else:
                raise FileNotFoundError(f"Missing required input file: {path}")
        else:
            LOGGER.info("Loading %s from %s", name, path)
            dataframes[name] = pd.read_parquet(path)
    return dataframes


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric, preserving NA for invalid entries."""
    return pd.to_numeric(series, errors="coerce")


def _cyclical_encoding(values: pd.Series, period: int) -> pd.DataFrame:
    """Return sine and cosine encodings for ordinal values with given period."""
    radians = 2 * math.pi * (values % period) / period
    return pd.DataFrame({
        "sin": np.sin(radians),
        "cos": np.cos(radians),
    })


def _gini_coefficient(values: Sequence[float]) -> float:
    """Compute the Gini coefficient for a sequence of values."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    array = np.abs(array)
    array = array[~np.isnan(array)]
    if array.size == 0:
        return 0.0
    total = array.sum()
    if total <= 0:
        return 0.0
    sorted_vals = np.sort(array)
    cumulative = np.cumsum(sorted_vals)
    n = sorted_vals.size
    gini = (n + 1 - 2 * np.sum(cumulative) / total) / n
    return float(max(0.0, gini))


def _compute_client_cadence_windows(events: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Compute rolling cadence ratios per client for weekly and EOM patterns."""
    if events.empty:
        events["weekly_cadence_client"] = 0.0
        events["eom_cadence_client"] = 0.0
        return events

    enriched = events.copy()
    enriched["weekly_cadence_client"] = 0.0
    enriched["eom_cadence_client"] = 0.0

    window = np.timedelta64(window_days, "D")

    for cliente, group in enriched.groupby("cliente", sort=False):
        group_sorted = group.sort_values("data_pareggio")
        valid_mask = group_sorted["data_pareggio"].notna()
        if not valid_mask.any():
            continue

        valid_group = group_sorted.loc[valid_mask]
        indices = valid_group.index.to_numpy()
        timestamps = valid_group["data_pareggio"].to_numpy(dtype="datetime64[ns]")
        weekday = valid_group["dow_event"].astype(int).to_numpy()
        is_eom = valid_group["is_eom_event"].astype(int).to_numpy()

        weekly_ratio = np.zeros(len(indices), dtype=float)
        eom_ratio = np.zeros(len(indices), dtype=float)

        start = 0
        counts_per_weekday = np.zeros(7, dtype=float)
        total_events = 0.0
        eom_events = 0.0

        for pos in range(len(indices)):
            current_time = timestamps[pos]
            lower_bound = current_time - window

            while start <= pos and timestamps[start] < lower_bound:
                counts_per_weekday[weekday[start]] -= 1
                total_events -= 1
                eom_events -= is_eom[start]
                start += 1

            counts_per_weekday[weekday[pos]] += 1
            total_events += 1
            eom_events += is_eom[pos]

            if total_events > 0:
                weekly_ratio[pos] = counts_per_weekday[weekday[pos]] / total_events
                eom_ratio[pos] = eom_events / total_events

        enriched.loc[indices, "weekly_cadence_client"] = weekly_ratio
        enriched.loc[indices, "eom_cadence_client"] = eom_ratio

    return enriched


def enrich_events_with_calendar_features(
    events: pd.DataFrame,
    eom_threshold_days: int = EOM_THRESHOLD_DAYS,
    window_days: int = ROLLING_WINDOW_DAYS,
) -> pd.DataFrame:
    """Augment events with calendar and cadence attributes."""
    if events.empty:
        return events.assign(
            is_eom_event=0,
            days_to_eom=0,
            is_quarter_end_event=0,
            days_to_qe=0,
            dow_event=-1,
            dow_run_is_thu=0,
            weekly_cadence_client=0.0,
            eom_cadence_client=0.0,
        )

    enriched = events.copy()
    valid_dates = enriched["data_pareggio"].notna()

    enriched["days_to_eom"] = 0
    enriched["is_eom_event"] = 0
    enriched["days_to_qe"] = 0
    enriched["is_quarter_end_event"] = 0
    enriched["dow_event"] = -1
    enriched["dow_run_is_thu"] = 0

    if valid_dates.any():
        event_dates = enriched.loc[valid_dates, "data_pareggio"]

        days_to_eom = (event_dates.dt.days_in_month - event_dates.dt.day).astype(int)
        enriched.loc[valid_dates, "days_to_eom"] = days_to_eom
        enriched.loc[valid_dates, "is_eom_event"] = (days_to_eom < max(eom_threshold_days, 1)).astype(int)

        quarter_end_dates = event_dates.dt.to_period("Q").dt.end_time.dt.normalize()
        days_to_qe = (quarter_end_dates - event_dates).dt.days.astype(int)
        enriched.loc[valid_dates, "days_to_qe"] = days_to_qe
        enriched.loc[valid_dates, "is_quarter_end_event"] = (days_to_qe == 0).astype(int)

        dow_values = event_dates.dt.weekday.astype(int)
        enriched.loc[valid_dates, "dow_event"] = dow_values
        enriched.loc[valid_dates, "dow_run_is_thu"] = (dow_values == 3).astype(int)

    enriched = _compute_client_cadence_windows(enriched, window_days)
    return enriched


def _compute_sequence_features(group: pd.DataFrame) -> pd.DataFrame:
    """Compute document sequence metrics within a client-month group."""
    group = group.copy()
    original_index = group.index

    docnum_str = group["numero_documento"].fillna("").astype("string")
    numeric_component = docnum_str.str.replace(r"\D", "", regex=True)
    docnum_numeric = pd.to_numeric(numeric_component, errors="coerce")
    has_numeric = docnum_numeric.notna()

    working = group.assign(
        _docnum_str=docnum_str,
        _docnum_numeric=docnum_numeric,
        _has_numeric=has_numeric,
    ).sort_values(
        by=["_has_numeric", "_docnum_numeric", "_docnum_str"],
        ascending=[False, True, True],
        kind="mergesort",
    )

    doc_values = working["_docnum_numeric"].to_numpy(dtype=float, na_value=np.nan)
    prev_values = np.roll(doc_values, 1)
    prev_values[0] = np.nan
    valid_diff = ~np.isnan(doc_values) & ~np.isnan(prev_values)

    diff = np.zeros(len(doc_values), dtype=float)
    diff[valid_diff] = doc_values[valid_diff] - prev_values[valid_diff]
    delta_prev = np.zeros(len(doc_values), dtype=int)
    delta_prev[valid_diff] = np.round(diff[valid_diff]).astype(int)

    consecutive = valid_diff & np.isclose(diff, 1.0, atol=1e-6)
    block_break = ~consecutive
    if block_break.size:
        block_break[0] = True
    block_id = np.cumsum(block_break.astype(int)) - 1
    if block_id.size == 0:
        block_id = np.array([0])

    block_counts = np.bincount(block_id if block_id.size else np.array([0]))
    seq_block_len = block_counts[block_id]
    is_numeric = ~np.isnan(doc_values)
    is_consecutive_block = np.where(is_numeric & (seq_block_len >= 2), 1, 0)

    working = working.assign(
        docnum_delta_prev=delta_prev.astype(int, copy=False),
        seq_block_len_client_month=seq_block_len.astype(int, copy=False),
        docnum_is_consecutive_block=is_consecutive_block.astype(int, copy=False),
        _seq_block_id=block_id.astype(int, copy=False),
    )

    block_last_dates = working.groupby("_seq_block_id")["data_documento"].transform("max")
    working["_block_last_doc_date"] = block_last_dates

    block_summary = (
        working.groupby("_seq_block_id")["_block_last_doc_date"]
        .first()
        .sort_values()
    )
    if block_summary.empty:
        block_ids_sorted = np.array([], dtype=int)
        block_dates_sorted = np.array([], dtype="datetime64[ns]")
    else:
        block_ids_sorted = block_summary.index.to_numpy(dtype=int)
        block_dates_sorted = block_summary.to_numpy(dtype="datetime64[ns]")
        valid_block_dates = ~pd.isna(block_summary.values)
        block_ids_sorted = block_ids_sorted[valid_block_dates]
        block_dates_sorted = block_dates_sorted[valid_block_dates]

    recent_flags = np.zeros(len(working), dtype=int)
    event_dates = working["data_evento"].to_numpy(dtype="datetime64[ns]")
    block_ids_array = working["_seq_block_id"].to_numpy(dtype=int)

    if block_dates_sorted.size:
        for idx_row, event_date in enumerate(event_dates):
            if np.isnat(event_date):
                continue
            pos = np.searchsorted(block_dates_sorted, event_date, side="right") - 1
            if pos >= 0 and block_ids_array[idx_row] == block_ids_sorted[pos]:
                recent_flags[idx_row] = 1

    working["close_to_recent_block"] = recent_flags

    result = working[[
        "docnum_delta_prev",
        "docnum_is_consecutive_block",
        "seq_block_len_client_month",
        "close_to_recent_block",
        "_seq_block_id",
    ]]

    return result.reindex(original_index)


def compute_client_profiles(
    events: pd.DataFrame, 
    event_items: pd.DataFrame,
    line_items: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute historical client-level summary statistics (for HDBSCAN).
    
    Prompt 3 requirements:
    - Cardinalità k per evento (k_mode, k_entropy, P_k1, P_k2, P_kge3)
    - Periodicità (inter-event gaps, ACF features, dow/dom concentration)
    - Coverage mensile (month_coverage_* se line_items disponibile)
    - Distribuzioni policy storiche (p_same_period_hist)
    
    Args:
        events: Events DataFrame with cliente, data_pareggio, n_righe_evento
        event_items: Event items with policy scores
        line_items: Optional line items for month_coverage calculation
    
    Returns:
        DataFrame with client-level features (one row per cliente)
    """
    if event_items.empty:
        return pd.DataFrame(
            columns=[
                "cliente",
                "p_same_period_hist",
                "age_med_hist",
                "p_event_k1",
                "p_event_k2",
                "p_event_kge3",
                "gap_med_events",
                "dow_sin_mean_hist",
                "dow_cos_mean_hist",
                "month_coverage_mean",
                "month_coverage_p95",
                "month_coverage_full_rate",
                "acf_lag1",
                "acf_lag7",
                "acf_lag14",
                "acf_lag30",
            ]
        )

    items = event_items.copy()
    items["periodo_num"] = _safe_numeric(items["periodo_contabile"])
    items["event_month"] = items["data_pareggio"].dt.month
    valid_period = items["periodo_num"].between(1, 12)
    items["same_period_flag"] = np.where(
        valid_period & items["event_month"].notna(),
        (items["periodo_num"].round().astype("Int64") == items["event_month"].astype("Int64")).astype(float),
        np.nan,
    )
    items["age_days_hist"] = (
        items["data_pareggio"] - items["data_documento"]
    ).dt.days

    dow = items["data_pareggio"].dt.weekday
    dow_enc = _cyclical_encoding(dow.fillna(0), period=7)
    items["dow_sin_hist"] = dow_enc["sin"].where(dow.notna(), np.nan)
    items["dow_cos_hist"] = dow_enc["cos"].where(dow.notna(), np.nan)

    cliente_index = pd.Index(sorted(items["cliente"].unique()), name="cliente")
    profiles = pd.DataFrame(index=cliente_index)

    profiles["p_same_period_hist"] = items.groupby("cliente")["same_period_flag"].mean()
    profiles["age_med_hist"] = items.groupby("cliente")["age_days_hist"].median()
    profiles["dow_sin_mean_hist"] = items.groupby("cliente")["dow_sin_hist"].mean()
    profiles["dow_cos_mean_hist"] = items.groupby("cliente")["dow_cos_hist"].mean()

    # Event-level statistics
    if not events.empty:
        events = events.copy()
        events["dow"] = events["data_pareggio"].dt.weekday
        dow_enc_events = _cyclical_encoding(events["dow"].fillna(0), period=7)
        events["dow_sin_event"] = dow_enc_events["sin"].where(events["dow"].notna(), np.nan)
        events["dow_cos_event"] = dow_enc_events["cos"].where(events["dow"].notna(), np.nan)

        grouped_events = events.groupby("cliente")
        profiles["p_event_k1"] = grouped_events["n_righe_evento"].apply(
            lambda series: float((series == 1).mean()) if len(series) else 0.0
        )
        profiles["p_event_k2"] = grouped_events["n_righe_evento"].apply(
            lambda series: float((series == 2).mean()) if len(series) else 0.0
        )
        profiles["p_event_kge3"] = grouped_events["n_righe_evento"].apply(
            lambda series: float((series >= 3).mean()) if len(series) else 0.0
        )
        profiles["gap_med_events"] = grouped_events["inter_event_gap_days"].median()
        profiles["dow_sin_event_mean"] = grouped_events["dow_sin_event"].mean()
        profiles["dow_cos_event_mean"] = grouped_events["dow_cos_event"].mean()
    else:
        profiles["p_event_k1"] = 0.0
        profiles["p_event_k2"] = 0.0
        profiles["p_event_kge3"] = 0.0
        profiles["gap_med_events"] = 0.0
        profiles["dow_sin_event_mean"] = 0.0
        profiles["dow_cos_event_mean"] = 0.0

    # ACF (Autocorrelation Function) for periodicity - Prompt 3 requirement
    # Compute ACF on inter-event gaps for each client
    if not events.empty and "inter_event_gap_days" in events.columns:
        acf_profiles = _compute_acf_features(events)
        profiles = profiles.join(acf_profiles, how="left")
    else:
        profiles["acf_lag1"] = 0.0
        profiles["acf_lag7"] = 0.0
        profiles["acf_lag14"] = 0.0
        profiles["acf_lag30"] = 0.0
    
    # Month coverage metrics - Prompt 3 requirement
    # Integrate month_coverage_* if line_items available
    if line_items is not None and not line_items.empty:
        try:
            from src.analytics.month_coverage import compute_month_coverage_metrics
            
            coverage_df = compute_month_coverage_metrics(events, event_items, line_items)
            if not coverage_df.empty:
                profiles = profiles.merge(
                    coverage_df[["cliente", "month_coverage_mean", "month_coverage_p95", "month_coverage_full_rate"]],
                    on="cliente",
                    how="left"
                )
            else:
                profiles["month_coverage_mean"] = 0.0
                profiles["month_coverage_p95"] = 0.0
                profiles["month_coverage_full_rate"] = 0.0
        except ImportError:
            LOGGER.warning("month_coverage module not available; skipping coverage metrics")
            profiles["month_coverage_mean"] = 0.0
            profiles["month_coverage_p95"] = 0.0
            profiles["month_coverage_full_rate"] = 0.0
    else:
        profiles["month_coverage_mean"] = 0.0
        profiles["month_coverage_p95"] = 0.0
        profiles["month_coverage_full_rate"] = 0.0

    # Tipo documento distribution
    tipo_counts = (
        items.groupby(["cliente", "tipo_documento"]).size().unstack(fill_value=0)
    )
    tipo_shares = tipo_counts.div(tipo_counts.sum(axis=1), axis=0).fillna(0.0)
    tipo_shares = tipo_shares.add_prefix("share_tipo_documento_")

    profiles = profiles.join(tipo_shares, how="left")
    profiles = profiles.fillna(0.0).reset_index()
    profiles["cliente"] = profiles["cliente"].astype("string")

    return profiles


def _compute_acf_features(events: pd.DataFrame) -> pd.DataFrame:
    """Compute Autocorrelation Function (ACF) features for inter-event gaps.
    
    Prompt 3 requirement: periodicity detection via ACF on inter-event gaps.
    Computes ACF at lags 1, 7, 14, 30 days to detect:
    - lag1: immediate correlation (smoothness)
    - lag7: weekly patterns
    - lag14: bi-weekly patterns
    - lag30: monthly patterns
    
    Args:
        events: Events DataFrame with cliente, inter_event_gap_days
    
    Returns:
        DataFrame indexed by cliente with acf_lag1, acf_lag7, acf_lag14, acf_lag30
    """
    if "inter_event_gap_days" not in events.columns:
        return pd.DataFrame(columns=["acf_lag1", "acf_lag7", "acf_lag14", "acf_lag30"])
    
    acf_records = []
    lags = [1, 7, 14, 30]
    
    for cliente, group in events.groupby("cliente"):
        gaps = group["inter_event_gap_days"].dropna().to_numpy(dtype=float)
        
        if gaps.size < 3:
            # Not enough data for ACF
            acf_values = [0.0] * len(lags)
        else:
            acf_values = []
            for lag in lags:
                if lag >= gaps.size:
                    acf_values.append(0.0)
                else:
                    # Compute Pearson correlation at given lag
                    x = gaps[:-lag]
                    y = gaps[lag:]
                    if x.size < 2:
                        acf_values.append(0.0)
                    else:
                        try:
                            corr = float(np.corrcoef(x, y)[0, 1])
                            acf_values.append(corr if not np.isnan(corr) else 0.0)
                        except:
                            acf_values.append(0.0)
        
        acf_records.append({
            "cliente": cliente,
            "acf_lag1": acf_values[0],
            "acf_lag7": acf_values[1],
            "acf_lag14": acf_values[2],
            "acf_lag30": acf_values[3],
        })
    
    acf_df = pd.DataFrame(acf_records)
    acf_df = acf_df.set_index("cliente")
    return acf_df


def _percentile_rank_to_decile(rank: pd.Series, count: pd.Series) -> pd.Series:
    """Convert within-group ranks to deciles spanning 0-9."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(count > 1, (rank - 1) / (count - 1), 0.0)
    ratio = np.clip(ratio, 0.0, 1.0)
    return pd.Series(np.floor(ratio * 10).astype(int), index=rank.index)


def build_feature_matrix(
    train_df: pd.DataFrame,
    events: pd.DataFrame,
    client_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Construct the modelling feature matrix from base datasets."""
    if train_df.empty:
        LOGGER.warning("train_pointwise dataset is empty; returning empty feature matrix.")
        return pd.DataFrame()

    if EVENT_CALENDAR_COLUMNS.issubset(events.columns):
        events_with_calendar = events.copy()
    else:
        events_with_calendar = enrich_events_with_calendar_features(events)

    if "cliente" in events_with_calendar.columns and "importo_evento" in events_with_calendar.columns:
        client_event_amount_median = (
            events_with_calendar.groupby("cliente")["importo_evento"].median()
        )
    else:
        client_event_amount_median = pd.Series(dtype=float)

    event_info = events_with_calendar[[
        "event_key",
        "data_pareggio",
        "importo_evento",
        "n_righe_evento",
        "inter_event_gap_days",
        "is_eom_event",
        "days_to_eom",
        "is_quarter_end_event",
        "days_to_qe",
        "dow_event",
        "dow_run_is_thu",
        "weekly_cadence_client",
        "eom_cadence_client",
    ]].rename(columns={"data_pareggio": "data_evento"})

    features = train_df.merge(event_info, on="event_key", how="left")

    features["periodo_num"] = _safe_numeric(features["periodo_contabile"])
    features["event_month"] = features["data_evento"].dt.month
    features["age_days"] = (
        features["data_evento"] - features["data_documento"]
    ).dt.days
    features["same_period"] = np.where(
        features["periodo_num"].between(1, 12) & features["event_month"].notna(),
        (features["periodo_num"].round().astype("Int64") == features["event_month"].astype("Int64")).astype(int),
        0,
    )

    event_year = features["data_evento"].dt.year
    if "esercizio" in features:
        invoice_year = _safe_numeric(features["esercizio"])
    else:
        invoice_year = pd.Series(index=features.index, dtype=float)
    features["same_fiscal_year"] = 0
    valid_years = event_year.notna() & invoice_year.notna()
    if valid_years.any():
        event_year_int = event_year.loc[valid_years].astype(int)
        invoice_year_int = invoice_year.loc[valid_years].round().astype(int)
        comparison = (invoice_year_int.to_numpy() == event_year_int.to_numpy()).astype(int)
        features.loc[valid_years, "same_fiscal_year"] = comparison

    features["same_quarter_as_event"] = 0
    valid_quarter = features["data_evento"].notna() & features["data_documento"].notna()
    if valid_quarter.any():
        doc_dates = features.loc[valid_quarter, "data_documento"]
        evt_dates = features.loc[valid_quarter, "data_evento"]
        doc_quarter = ((doc_dates.dt.month - 1) // 3 + 1).astype(int)
        evt_quarter = ((evt_dates.dt.month - 1) // 3 + 1).astype(int)
        same_quarter = (
            (doc_dates.dt.year.to_numpy() == evt_dates.dt.year.to_numpy())
            & (doc_quarter.to_numpy() == evt_quarter.to_numpy())
        ).astype(int)
        features.loc[valid_quarter, "same_quarter_as_event"] = same_quarter

    dow = features["data_evento"].dt.weekday
    dow_enc = _cyclical_encoding(dow.fillna(0), period=7)
    features["dow_sin"] = dow_enc["sin"].where(dow.notna(), 0.0)
    features["dow_cos"] = dow_enc["cos"].where(dow.notna(), 0.0)

    dom = features["data_evento"].dt.day.fillna(1)
    dom_enc = _cyclical_encoding(dom - 1, period=31)
    features["dom_sin"] = dom_enc["sin"].where(features["data_evento"].notna(), 0.0)
    features["dom_cos"] = dom_enc["cos"].where(features["data_evento"].notna(), 0.0)

    month_vals = features["data_evento"].dt.month.fillna(1)
    month_enc = _cyclical_encoding(month_vals - 1, period=12)
    features["mon_sin"] = month_enc["sin"].where(features["data_evento"].notna(), 0.0)
    features["mon_cos"] = month_enc["cos"].where(features["data_evento"].notna(), 0.0)

    features["lag_reg_vs_doc"] = (
        features["data_registrazione"] - features["data_documento"]
    ).dt.days
    features["lag_acq_vs_doc"] = (
        features["data_acquisizione"] - features["data_documento"]
    ).dt.days

    features["abs_amount"] = features["imp_div_int"].abs().fillna(0.0)

    features["client_event_amount_median"] = features["cliente"].map(
        client_event_amount_median
    )
    features["client_event_amount_median"] = features["client_event_amount_median"].fillna(0.0)
    denom = features["client_event_amount_median"].replace(0.0, np.nan)
    event_amount_scale = np.zeros(len(features), dtype=float)
    valid_scale = denom.notna() & features["importo_evento"].notna()
    if valid_scale.any():
        event_amount_scale[valid_scale.to_numpy()] = (
            features.loc[valid_scale, "importo_evento"].astype(float).to_numpy()
            / denom.loc[valid_scale].to_numpy()
        )
    features["event_amount_scale"] = event_amount_scale
    features.drop(columns=["client_event_amount_median"], inplace=True)

    features["_doc_month"] = features["data_documento"].dt.to_period("M").astype("string")
    features["docnum_delta_prev"] = 0
    features["docnum_is_consecutive_block"] = 0
    features["seq_block_len_client_month"] = 1
    features["close_to_recent_block"] = 0
    features["_seq_block_id"] = -1

    valid_doc_mask = features["data_documento"].notna()
    if valid_doc_mask.any():
        seq_features = (
            features.loc[valid_doc_mask]
            .groupby(["cliente", "_doc_month"], group_keys=False, sort=False)[
                ["numero_documento", "data_documento", "data_evento"]
            ]
            .apply(_compute_sequence_features)
        )

        sequence_cols = [
            "docnum_delta_prev",
            "docnum_is_consecutive_block",
            "seq_block_len_client_month",
            "close_to_recent_block",
            "_seq_block_id",
        ]
        features.loc[seq_features.index, sequence_cols] = seq_features[sequence_cols]

    features["docnum_delta_prev"] = features["docnum_delta_prev"].fillna(0).astype(int)
    features["docnum_is_consecutive_block"] = features["docnum_is_consecutive_block"].fillna(0).astype(int)
    features["seq_block_len_client_month"] = features["seq_block_len_client_month"].fillna(1).astype(int)
    features["close_to_recent_block"] = features["close_to_recent_block"].fillna(0).astype(int)

    features["amt_rank_open"] = (
        features.groupby(["cliente", "event_key"])["abs_amount"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    event_group = features.groupby(["cliente", "event_key"])
    event_candidate_count = event_group["abs_amount"].transform("count").astype(int)
    features["amt_share_rank_event"] = 0.0
    valid_share = event_candidate_count > 1
    if valid_share.any():
        rank_values = features.loc[valid_share, "amt_rank_open"].astype(float) - 1.0
        denom_share = (event_candidate_count.loc[valid_share] - 1).replace(0, np.nan)
        share_values = rank_values / denom_share
        features.loc[valid_share, "amt_share_rank_event"] = share_values.fillna(0.0)
    features["amt_share_rank_event"] = features["amt_share_rank_event"].clip(0.0, 1.0)

    features["event_gini_amounts_open"] = event_group["abs_amount"].transform(
        lambda vals: _gini_coefficient(vals.to_numpy(dtype=float, copy=False))
    )
    features["event_gini_amounts_open"] = features["event_gini_amounts_open"].fillna(0.0)

    group = features.groupby("cliente")["abs_amount"]
    rank_within = group.rank(method="first")
    count_within = group.transform("count")
    features["amt_decile_client"] = _percentile_rank_to_decile(rank_within, count_within)

    stats = group.agg(["mean", "std"])  # type: ignore[arg-type]
    stats = stats.rename(columns={"mean": "_mean_amt", "std": "_std_amt"})
    features = features.merge(stats, left_on="cliente", right_index=True, how="left")
    features["amt_z_client"] = np.where(
        features["_std_amt"].fillna(0) > 0,
        (features["abs_amount"] - features["_mean_amt"].fillna(0)) / features["_std_amt"].fillna(1),
        0.0,
    )

    denominator = features["importo_evento"].abs()
    ratio = np.zeros(len(features), dtype=float)
    valid_ratio = denominator > 0
    if valid_ratio.any():
        ratio[valid_ratio.to_numpy()] = (
            features.loc[valid_ratio, "abs_amount"].to_numpy()
            / denominator.loc[valid_ratio].to_numpy()
        )
    features["amt_ratio_B"] = np.clip(ratio, 0.0, 1.0)

    # Join client profiles
    features = features.merge(client_profiles, on="cliente", how="left")

    # Assemble final feature set
    log_consecutive_block_examples(features)
    features.drop(columns=["_doc_month", "_seq_block_id"], inplace=True, errors="ignore")

    profile_columns = [col for col in client_profiles.columns if col != "cliente"]
    categorical_cols = ["tipo_documento", "chiave_contabile", "divisa"]
    base_columns = [
        "cliente",
        "event_key",
        "numero_documento",
        "imp_div_int",
        "importo_evento",
        "n_righe_evento",
        "inter_event_gap_days",
        "is_eom_event",
        "days_to_eom",
        "is_quarter_end_event",
        "days_to_qe",
        "dow_event",
        "dow_run_is_thu",
        "weekly_cadence_client",
        "eom_cadence_client",
        "age_days",
        "same_period",
        "same_fiscal_year",
        "same_quarter_as_event",
        "docnum_delta_prev",
        "docnum_is_consecutive_block",
        "seq_block_len_client_month",
        "close_to_recent_block",
        "dow_sin",
        "dow_cos",
        "dom_sin",
        "dom_cos",
        "mon_sin",
        "mon_cos",
        "lag_reg_vs_doc",
        "lag_acq_vs_doc",
        "amt_rank_open",
        "amt_share_rank_event",
        "amt_decile_client",
        "amt_z_client",
        "amt_ratio_B",
        "event_gini_amounts_open",
        "event_amount_scale",
    ] + profile_columns + categorical_cols + ["label"]

    available_columns = [col for col in base_columns if col in features.columns]
    feature_matrix = features[available_columns].copy()

    numeric_cols = feature_matrix.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols_no_label = [col for col in numeric_cols if col != "label"]
    if numeric_cols_no_label:
        feature_matrix[numeric_cols_no_label] = feature_matrix[numeric_cols_no_label].fillna(0.0)
    if "label" in feature_matrix.columns:
        feature_matrix["label"] = feature_matrix["label"].fillna(0).astype(int)

    for col in categorical_cols:
        if col in feature_matrix.columns:
            feature_matrix[col] = feature_matrix[col].fillna("missing").astype("string")

    for col in ["cliente", "event_key", "numero_documento"]:
        if col in feature_matrix.columns:
            feature_matrix[col] = feature_matrix[col].fillna("missing").astype("string")

    feature_matrix.drop(columns=["_mean_amt", "_std_amt"], errors="ignore", inplace=True)

    return feature_matrix


def log_feature_statistics(feature_matrix: pd.DataFrame, columns: Sequence[str]) -> None:
    """Log mean and variance for the selected feature columns."""
    for column in columns:
        if column not in feature_matrix.columns:
            LOGGER.warning("Feature %s is not present in the feature matrix.", column)
            continue

        values = pd.to_numeric(feature_matrix[column], errors="coerce").fillna(0.0)
        array_values = values.to_numpy(dtype=float, copy=False)
        if array_values.size == 0:
            mean_val = 0.0
            var_val = 0.0
        else:
            mean_val = float(array_values.mean())
            var_val = float(array_values.var(ddof=0))
        LOGGER.info("Feature %s statistics -> mean: %.6f, variance: %.6f", column, mean_val, var_val)


def log_mean_gini_per_client(feature_matrix: pd.DataFrame) -> None:
    """Log the mean event-level Gini coefficient per client."""
    if "event_gini_amounts_open" not in feature_matrix.columns or "cliente" not in feature_matrix.columns:
        LOGGER.warning("Missing columns for Gini logging; skipping.")
        return

    gini_by_client = (
        feature_matrix.groupby("cliente")["event_gini_amounts_open"].mean().sort_index()
    )
    if gini_by_client.empty:
        LOGGER.warning("No rows available to compute mean Gini per client.")
        return

    LOGGER.info(
        "Mean event_gini_amounts_open per client: %s",
        gini_by_client.round(6).to_dict(),
    )


def log_consecutive_block_examples(features: pd.DataFrame, max_examples: int = 3) -> None:
    """Log sample consecutive blocks with their lengths and date ranges."""
    if "docnum_is_consecutive_block" not in features.columns:
        LOGGER.warning("Consecutive block features not present; skipping examples logging.")
        return

    mask = features["docnum_is_consecutive_block"] == 1
    if not mask.any():
        LOGGER.warning("No consecutive document blocks detected to log examples.")
        return

    required_cols = [
        "cliente",
        "_doc_month",
        "_seq_block_id",
        "seq_block_len_client_month",
        "data_documento",
        "numero_documento",
    ]
    missing = [col for col in required_cols if col not in features.columns]
    if missing:
        LOGGER.warning("Missing columns %s for block logging; aborting.", missing)
        return

    sample_df = features.loc[mask, required_cols].dropna(subset=["data_documento"])
    if sample_df.empty:
        LOGGER.warning("No valid document dates found for consecutive block logging.")
        return

    block_summary = (
        sample_df.groupby(["cliente", "_doc_month", "_seq_block_id"], dropna=False)
        .agg(
            block_len=("seq_block_len_client_month", "max"),
            start_date=("data_documento", "min"),
            end_date=("data_documento", "max"),
        )
        .reset_index()
        .sort_values("end_date", ascending=False)
    )

    for _, row in block_summary.head(max_examples).iterrows():
        cliente = row["cliente"]
        month_key = row["_doc_month"]
        block_id = row["_seq_block_id"]
        block_len = int(row["block_len"])
        start_date = row["start_date"]
        end_date = row["end_date"]
        block_docs = sample_df[
            (sample_df["cliente"] == cliente)
            & (sample_df["_doc_month"] == month_key)
            & (sample_df["_seq_block_id"] == block_id)
        ]["numero_documento"].tolist()

        LOGGER.info(
            "Consecutive block example -> cliente=%s, month=%s, length=%d, date_range=[%s, %s], docs=%s",
            cliente,
            month_key,
            block_len,
            start_date.date() if pd.notna(start_date) else "NA",
            end_date.date() if pd.notna(end_date) else "NA",
            block_docs,
        )


def save_feature_outputs(
    feature_matrix: pd.DataFrame,
    client_profiles: pd.DataFrame,
    events_with_features: pd.DataFrame,
    project_root: Path,
) -> None:
    """Persist engineered datasets to the curated directory.
    
    Prompt 3 outputs:
    - train_features.parquet: invoice-level features for ranking (no leakage)
    - client_profiles.parquet: client-level features for HDBSCAN
    - events_with_features.parquet: events enriched with calendar features
    """
    curated_dir = project_root / "data" / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)

    features_path = curated_dir / "train_features.parquet"
    profiles_path = curated_dir / "client_profiles.parquet"
    events_path = curated_dir / "events_with_features.parquet"

    feature_matrix.to_parquet(features_path, compression="snappy", index=False)
    client_profiles.to_parquet(profiles_path, compression="snappy", index=False)
    events_with_features.to_parquet(events_path, compression="snappy", index=False)

    LOGGER.info(
        "Saved %s rows to %s, %s client profiles to %s, and %s events to %s",
        len(feature_matrix),
        features_path,
        len(client_profiles),
        profiles_path,
        len(events_with_features),
        events_path,
    )


def run_random_forest_smoke_test(train_features: pd.DataFrame) -> None:
    """Fit a small RandomForest to report top feature importances."""
    if train_features.empty:
        LOGGER.warning("Skipped RandomForest smoke test because feature matrix is empty.")
        return

    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn is required for the smoke test. Please install scikit-learn."
        ) from exc

    rf_df = train_features.copy()
    categorical_cols = ["tipo_documento", "chiave_contabile", "divisa"]
    for col in categorical_cols:
        if col in rf_df.columns:
            rf_df[col] = rf_df[col].astype("category").cat.codes

    drop_cols = {"cliente", "event_key", "numero_documento", "label"}
    X = rf_df.drop(columns=list(drop_cols.intersection(rf_df.columns)))
    y = rf_df["label"].astype(int)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    top10 = importances.sort_values(ascending=False).head(10)
    print("RandomForest feature importances (top 10):")
    print(top10.to_string())


def main() -> None:
    """Run the complete feature engineering pipeline."""
    configure_logging()
    project_root = Path(__file__).resolve().parents[2]

    datasets = load_curated_datasets(project_root)
    events_with_calendar = enrich_events_with_calendar_features(datasets["events"])
    
    # Prompt 3: pass line_items to compute_client_profiles for month_coverage
    line_items = datasets.get("line_items", pd.DataFrame())
    client_profiles = compute_client_profiles(
        events_with_calendar, 
        datasets["event_items"],
        line_items if not line_items.empty else None
    )
    client_profiles = client_profiles.fillna(0.0)

    feature_matrix = build_feature_matrix(
        datasets["train_pointwise"],
        events_with_calendar,
        client_profiles,
    )

    if feature_matrix.empty:
        LOGGER.warning("No features generated; skipping save and smoke test.")
        return

    log_feature_statistics(feature_matrix, FEATURES_TO_LOG)
    log_mean_gini_per_client(feature_matrix)
    
    # Prompt 3: export events_with_features.parquet
    save_feature_outputs(feature_matrix, client_profiles, events_with_calendar, project_root)
    run_random_forest_smoke_test(feature_matrix)


if __name__ == "__main__":
    main()
