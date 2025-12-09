"""Build unified pattern table combining rules and model metrics.

This module consolidates client-level pattern scores, rule accuracy, and model
metrics into a single parquet artifact. It also exposes helpers to identify
clear versus noisy clients for auditing workflows.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)

DATA_DIR = Path("data/curated")
OUTPUT_PATH = DATA_DIR / "pattern_table.parquet"

PATTERN_PROB_COLUMNS = ("P_k1", "P_k2", "P_kge3")
REQUIRED_PATTERN_COLUMNS = {
    "cliente",
    "fifo_score",
    "lifo_score",
    "same_period_score",
    "largest_score",
    "smallest_score",
    "k_mode",
    "k_entropy",
    "amount_pref_slope",
}

FINAL_COLUMNS = [
    "cliente",
    "pattern_label",
    "confidence",
    "rules_match_rate",
    "model_match_rate_cat",
    "model_match_rate_auto",
    "fifo_score",
    "lifo_score",
    "same_period_score",
    "largest_score",
    "smallest_score",
    "k_mode",
    "k_entropy",
    "amount_pref_slope",
    "stability_index",
]

def _configure_logging(level: int | str | None = logging.INFO) -> Path:
    return setup_logging(level)

def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file with logging.

    Parameters
    ----------
    path: Path
        Location of the parquet dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    LOGGER.info("Loading data from %s", path)
    return pd.read_parquet(path)

def derive_pattern_attributes(pattern_scores: pd.DataFrame) -> pd.DataFrame:
    """Augment pattern scores with label, confidence, and stability index.

    The pattern label is inferred from the probability columns (P_k1, P_k2,
    P_kge3). Confidence corresponds to the maximum probability observed across
    these buckets. Stability index is computed as ``1 - k_entropy`` when not
    already provided.
    """
    df = pattern_scores.copy()

    missing = REQUIRED_PATTERN_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Pattern scores missing required columns: {sorted(missing)}")

    unnamed_cols = [col for col in df.columns if col.startswith("Unnamed") or col == "0"]
    if unnamed_cols:
        LOGGER.warning("Dropping auxiliary columns from pattern scores: %s", unnamed_cols)
        df = df.drop(columns=unnamed_cols)

    available_prob_cols = [col for col in PATTERN_PROB_COLUMNS if col in df.columns]
    if not available_prob_cols:
        raise ValueError("Pattern scores missing probability columns needed for label/confidence inference.")

    prob_matrix = df[available_prob_cols]
    label_idx = prob_matrix.idxmax(axis=1)
    confidence = prob_matrix.max(axis=1)

    label_mapping = {
        "P_k1": "pattern_k1",
        "P_k2": "pattern_k2",
        "P_kge3": "pattern_k_ge_3",
    }

    df["pattern_label"] = label_idx.map(label_mapping).fillna(label_idx)
    df["confidence"] = confidence

    if "stability_index" not in df.columns:
        df["stability_index"] = 1.0 - df["k_entropy"]

    return df

def _safe_rename_match_rate(df: pd.DataFrame, column: str, new_name: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Expected column '{column}' not found in dataframe: {list(df.columns)}")
    return df.rename(columns={column: new_name})

def build_pattern_table(
    pattern_scores: pd.DataFrame,
    rules_metrics: pd.DataFrame,
    cat_metrics: pd.DataFrame,
    auto_metrics: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge pattern, rules, and model metrics into a unified dataframe."""
    df = derive_pattern_attributes(pattern_scores)

    rules_df = rules_metrics.copy()
    rules_df = _safe_rename_match_rate(rules_df, "rules_match_rate", "rules_match_rate")
    rules_df = rules_df[["cliente", "rules_match_rate"]]

    cat_df = _safe_rename_match_rate(cat_metrics.copy(), "model_match_rate", "model_match_rate_cat")[[
        "cliente",
        "model_match_rate_cat",
    ]]

    merged = df.merge(rules_df, on="cliente", how="left")
    merged = merged.merge(cat_df, on="cliente", how="left")

    if auto_metrics is not None:
        auto_df = _safe_rename_match_rate(
            auto_metrics.copy(),
            "model_match_rate",
            "model_match_rate_auto",
        )[["cliente", "model_match_rate_auto"]]
        merged = merged.merge(auto_df, on="cliente", how="left")
    else:
        merged["model_match_rate_auto"] = pd.NA

    for column in FINAL_COLUMNS:
        if column not in merged.columns:
            merged[column] = pd.NA

    return merged[FINAL_COLUMNS].sort_values("cliente").reset_index(drop=True)

def filter_top_clients(pattern_table: pd.DataFrame, n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identify top-N clear and noisy clients.

    Parameters
    ----------
    pattern_table: pd.DataFrame
        Unified dataset produced by :func:`build_pattern_table`.
    n: int
        Number of clients to return for each category.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Two dataframes: (clear_clients, noisy_clients).
    """
    if n <= 0:
        raise ValueError("Parameter 'n' must be positive.")

    required = {"confidence", "k_entropy"}
    if not required.issubset(pattern_table.columns):
        raise ValueError(f"Pattern table missing required columns: {sorted(required - set(pattern_table.columns))}")

    clear_clients = pattern_table.sort_values([
        "confidence",
        "k_entropy",
    ], ascending=[False, True]).head(n)
    noisy_clients = pattern_table.sort_values([
        "confidence",
        "k_entropy",
    ], ascending=[True, False]).head(n)

    return clear_clients.reset_index(drop=True), noisy_clients.reset_index(drop=True)

def main() -> None:
    _configure_logging()

    pattern_scores = load_parquet(DATA_DIR / "client_pattern_scores.parquet")
    rules_metrics = load_parquet(DATA_DIR / "client_rules_accuracy.parquet")
    cat_metrics = load_parquet(DATA_DIR / "client_model_metrics_catboost.parquet")

    auto_path = DATA_DIR / "client_model_metrics_autogluon.parquet"
    auto_metrics: Optional[pd.DataFrame]
    if auto_path.exists():
        auto_metrics = load_parquet(auto_path)
    else:
        LOGGER.warning("AutoGluon metrics not found at %s; column will be empty.", auto_path)
        auto_metrics = None

    pattern_table = build_pattern_table(pattern_scores, rules_metrics, cat_metrics, auto_metrics)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pattern_table.to_parquet(OUTPUT_PATH, index=False)
    LOGGER.info("Saved unified pattern table to %s", OUTPUT_PATH)

    LOGGER.info("Top 5 clients by confidence:\n%s", pattern_table.sort_values("confidence", ascending=False).head(5).to_string(index=False))
    LOGGER.info("Top 5 clients by k_entropy:\n%s", pattern_table.sort_values("k_entropy", ascending=False).head(5).to_string(index=False))

    clear_clients, noisy_clients = filter_top_clients(pattern_table, n=5)
    LOGGER.info("Top clear clients (confidence high, entropy low):\n%s", clear_clients.to_string(index=False))
    LOGGER.info("Top noisy clients (confidence low, entropy high):\n%s", noisy_clients.to_string(index=False))

if __name__ == "__main__":
    main()

