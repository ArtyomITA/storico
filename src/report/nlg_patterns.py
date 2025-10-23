"""Natural-language generation utilities for pattern summaries."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analytics.pattern_detectors import POLICY_NAMES, infer_pattern_label

DOW_LABELS = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]

LOGGER = logging.getLogger(__name__)


def _safe_float(value: Any) -> float:
    """Return ``value`` as float, defaulting to NaN if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_percent(value: Any, digits: int = 2) -> str:
    """Format a numeric value as percentage with a safe fallback."""
    numeric = _safe_float(value)
    if np.isnan(numeric):
        return "n.d."
    return f"{numeric:.{digits}%}"


def _format_decimal(value: Any, digits: int = 2) -> str:
    """Format a numeric value with a fixed amount of decimal places."""
    numeric = _safe_float(value)
    if np.isnan(numeric):
        return "n.d."
    return f"{numeric:.{digits}f}"


def _pick_first(series: Mapping[str, Any], *candidates: str) -> float:
    """Return the first available value among ``candidates`` converted to float."""
    for name in candidates:
        if name in series:
            numeric = _safe_float(series[name])
            if not np.isnan(numeric):
                return numeric
    return float("nan")


def _dow_label(dow_value: Any) -> str:
    """Return the Italian weekday label for ``dow_value`` if valid."""
    numeric_raw = _safe_float(dow_value)
    if np.isnan(numeric_raw):
        return "n.d."
    try:
        numeric = int(numeric_raw)
    except (TypeError, ValueError):
        return "n.d."
    if 0 <= numeric < len(DOW_LABELS):
        return DOW_LABELS[numeric]
    return "n.d."


def _policy_scores(series: Mapping[str, Any]) -> List[Tuple[str, float]]:
    """Collect and sort policy scores contained in ``series``."""
    scores: List[Tuple[str, float]] = []
    for policy in POLICY_NAMES:
        raw = series.get(f"{policy}_score")
        value = _safe_float(raw)
        if not np.isnan(value):
            scores.append((policy, value))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


def _policy_label(name: str) -> str:
    """Return a human-friendly label for a policy code."""
    return name.replace("_", " ").upper()


def verbalize_cadence(row: Mapping[str, Any]) -> str:
    """Summarise cadence signals with explicit evidence values."""
    eom_ratio = _pick_first(row, "eom_ratio", "eom_cadence_client")
    dow_mode = row.get("dow_mode")
    dow_conc = _pick_first(row, "dow_concentration", "dow_conc")

    if np.isnan(eom_ratio):
        cadence_prefix = "Eventi distribuiti nel mese"
    elif eom_ratio >= 0.5:
        cadence_prefix = "Eventi prevalgono **fine mese**"
    elif eom_ratio <= 0.2:
        cadence_prefix = "Eventi raramente **fine mese**"
    else:
        cadence_prefix = "Eventi con cadenza mista fine mese"

    ratio_text = _format_percent(eom_ratio, digits=1)
    dow_text = _dow_label(dow_mode)
    conc_text = _format_decimal(dow_conc, digits=2)
    return f"{cadence_prefix}: {ratio_text}; giorno tipico **{dow_text}**, concentrazione {conc_text}"


def verbalize_policy(row: Mapping[str, Any]) -> str:
    """Describe the dominant policy preference with alternatives."""
    scores = _policy_scores(row)
    if not scores:
        return "Policy di matching non disponibili"

    top_policy, top_score = scores[0]
    alternatives = scores[1:3]
    alt_chunks = [f"{_policy_label(name)} {score:.2f}" for name, score in alternatives if not np.isnan(score)]
    alt_text = ", ".join(alt_chunks) if alt_chunks else "nessuna alternativa forte"
    return (
        f"Comportamento simile a **{_policy_label(top_policy)}** (recall medio {top_score:.2f}), "
        f"alternative: {alt_text}"
    )


def verbalize_cardinality(row: Mapping[str, Any]) -> str:
    """Explain typical cardinality behaviour including entropy evidence."""
    k_mode = _safe_float(row.get("k_mode"))
    k_entropy = _safe_float(row.get("k_entropy"))
    return (
        "Cardinalità tipica **{mode}** (entropia {entropy})".format(
            mode=_format_decimal(k_mode, digits=1),
            entropy=_format_decimal(k_entropy, digits=2),
        )
    )


def verbalize_amount_pref(row: Mapping[str, Any]) -> str:
    """Summarise amount preference slope with qualitative label."""
    slope = _safe_float(row.get("amount_pref_slope"))
    if np.isnan(slope):
        return "Propensione sugli importi non disponibile"
    if abs(slope) < 0.05:
        descriptor = "bilanciati"
    elif slope > 0:
        descriptor = "alti"
    else:
        descriptor = "bassi"
    return f"Propensione verso importi **{descriptor}** (r={slope:.2f})"


def _policy_coherence(scores: Sequence[Tuple[str, float]]) -> float:
    """Compute a coherence proxy as delta between top and runner-up policy."""
    if not scores:
        return float("nan")
    if len(scores) == 1:
        return float("nan")
    top, runner = scores[0], scores[1]
    return max(0.0, top[1] - runner[1])


def verbalize_confidence(row: Mapping[str, Any], *, scores: Optional[Sequence[Tuple[str, float]]] = None) -> str:
    """Build a confidence statement referencing stability and policy coherence."""
    confidence_candidates = ("confidence", "pattern_confidence", "confidence_score")
    confidence = _pick_first(row, *confidence_candidates)
    if np.isnan(confidence):
        _, inferred_conf = infer_pattern_label(pd.Series(row))
        confidence = inferred_conf

    stability = _pick_first(row, "stability_index", "stability")
    if np.isnan(stability):
        stability = float("nan")

    score_pairs = list(scores or _policy_scores(row))
    coherence = _policy_coherence(score_pairs)
    coherence_text = _format_decimal(coherence, digits=2)
    return (
        "Confidenza complessiva {conf} su base stabilità {stability} e coerenza policy {coherence}".format(
            conf=_format_decimal(confidence, digits=2),
            stability=_format_decimal(stability, digits=2),
            coherence=coherence_text,
        )
    )


@dataclass
class SummaryContext:
    cliente: str
    score_row: Optional[pd.Series]
    pattern_row: Optional[pd.Series]


def _resolve_context(
    cliente: str,
    score_row: Optional[pd.Series],
    pattern_row: Optional[pd.Series],
    tables: Optional[Mapping[str, pd.DataFrame]],
) -> SummaryContext:
    """Resolve score and pattern rows from explicit inputs or tables."""
    resolved_score: Optional[pd.Series] = score_row
    resolved_pattern: Optional[pd.Series] = pattern_row

    if resolved_score is None and tables is not None:
        score_table = tables.get("client_pattern_scores")
        if isinstance(score_table, pd.DataFrame) and not score_table.empty:
            subset = score_table[score_table["cliente"].astype(str) == str(cliente)]
            if not subset.empty:
                resolved_score = subset.iloc[0]

    if resolved_pattern is None and tables is not None:
        pattern_table = tables.get("pattern_table")
        if isinstance(pattern_table, pd.DataFrame) and not pattern_table.empty:
            subset = pattern_table[pattern_table["cliente"].astype(str) == str(cliente)]
            if not subset.empty:
                resolved_pattern = subset.iloc[0]

    return SummaryContext(cliente=cliente, score_row=resolved_score, pattern_row=resolved_pattern)


def render_pattern_summary(
    cliente: str,
    *,
    score_row: Optional[pd.Series] = None,
    pattern_row: Optional[pd.Series] = None,
    tables: Optional[Mapping[str, pd.DataFrame]] = None,
) -> List[str]:
    """Return claim+evidence sentences for the selected client.

    The function accepts either pre-filtered pandas Series (``score_row`` and
    ``pattern_row``) or a mapping of curated tables from which the appropriate
    rows will be resolved lazily. When no score row is available an empty list is
    returned and the condition is logged for observability.
    """
    context = _resolve_context(cliente, score_row, pattern_row, tables)
    if context.score_row is None:
        LOGGER.warning("Pattern scores non disponibili per il cliente %s", cliente)
        return []

    merged_data: Dict[str, Any] = {"cliente": context.cliente}
    score_dict = {str(key): value for key, value in context.score_row.to_dict().items()}
    merged_data.update(score_dict)
    if context.pattern_row is not None:
        for key, value in context.pattern_row.items():
            str_key = str(key)
            merged_data.setdefault(str_key, value)

    scores = _policy_scores(merged_data)
    return [
        verbalize_cadence(merged_data),
        verbalize_policy(merged_data),
        verbalize_cardinality(merged_data),
        verbalize_amount_pref(merged_data),
        verbalize_confidence(merged_data, scores=scores),
    ]
