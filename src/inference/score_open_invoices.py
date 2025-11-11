"""Score open invoices with pattern coherence analysis.

Prompt 7: Given a client and their open invoices, compute p_i (selection probability)
using the trained model (CatBoost/AutoGluon), and calculate agreement_score showing
coherence with the client's dominant pattern/cluster.

Outputs:
- event_policy_predictions.parquet: p_i scores per invoice
- Coherence metrics: agreement_score, pattern alignment indicators
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Container for scoring results with coherence metrics."""
    
    cliente: str
    predictions: pd.DataFrame  # Invoice-level predictions
    agreement_score: float  # Coherence with dominant pattern [0, 1]
    alignment_status: str  # "aligned", "partial", "divergent"
    dominant_policy: str
    cluster_label: int
    pattern_confidence: float


def configure_logging(level: int | str | None = logging.INFO) -> Path:
    """Configure logging handlers."""
    return setup_logging(level)


def load_champion_model(models_dir: Path) -> Tuple[object, object, str]:
    """Load champion model (CatBoost or AutoGluon) with calibrator.
    
    Returns:
        (model, calibrator, model_type)
    """
    champion_path = models_dir / "champion.json"
    
    if not champion_path.exists():
        raise FileNotFoundError(f"Champion metadata not found: {champion_path}")
    
    with open(champion_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    model_type = metadata.get("model", "catboost")
    LOGGER.info("Loading champion model: %s", model_type)
    
    if model_type == "catboost":
        from catboost import CatBoostClassifier
        
        model_path = models_dir / "catboost.cbm"
        if not model_path.exists():
            raise FileNotFoundError(f"CatBoost model not found: {model_path}")
        
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        
        calibrator_path = models_dir / "calibrator.pkl"
        if calibrator_path.exists():
            with open(calibrator_path, "rb") as f:
                calibrator = pickle.load(f)
        else:
            LOGGER.warning("Calibrator not found; using identity")
            calibrator = IdentityCalibrator()
    
    elif model_type == "autogluon":
        from autogluon.tabular import TabularPredictor
        
        predictor_path = models_dir / "autogluon"
        if not predictor_path.exists():
            raise FileNotFoundError(f"AutoGluon predictor not found: {predictor_path}")
        
        model = TabularPredictor.load(str(predictor_path))
        
        calibrator_path = models_dir / "calibrator.pkl"
        if calibrator_path.exists():
            with open(calibrator_path, "rb") as f:
                calibrator = pickle.load(f)
        else:
            calibrator = IdentityCalibrator()
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, calibrator, model_type


class IdentityCalibrator:
    """Fallback calibrator that returns raw probabilities."""

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)


def load_client_context(
    cliente: str,
    project_root: Path
) -> Dict[str, pd.DataFrame]:
    """Load client-specific context: patterns, clusters, open invoices.
    
    Args:
        cliente: Client identifier
        project_root: Project root directory
        
    Returns:
        Dict with client_pattern, client_cluster, open_invoices DataFrames
    """
    curated_dir = project_root / "data" / "curated"
    
    context: Dict[str, pd.DataFrame] = {}
    
    # Load client_patterns
    patterns_path = curated_dir / "client_patterns.parquet"
    if patterns_path.exists():
        patterns = pd.read_parquet(patterns_path)
        client_pattern = patterns[patterns["cliente"] == cliente]
        context["client_pattern"] = client_pattern
        LOGGER.info("Loaded client pattern for %s", cliente)
    else:
        context["client_pattern"] = pd.DataFrame()
        LOGGER.warning("client_patterns.parquet not found")
    
    # Load client_clusters
    clusters_path = curated_dir / "client_clusters.parquet"
    if clusters_path.exists():
        clusters = pd.read_parquet(clusters_path)
        client_cluster = clusters[clusters["cliente"] == cliente]
        context["client_cluster"] = client_cluster
    else:
        context["client_cluster"] = pd.DataFrame()
    
    # Load line_items for open invoices (saldo_div_int != 0)
    line_items_path = curated_dir / "line_items.parquet"
    if line_items_path.exists():
        line_items = pd.read_parquet(line_items_path)
        client_items = line_items[line_items["cliente"] == cliente].copy()
        
        # Filter open invoices (non-zero balance)
        if "saldo_div_int" in client_items.columns:
            client_items["saldo_div_int"] = pd.to_numeric(
                client_items["saldo_div_int"], errors="coerce"
            ).fillna(0.0)
            open_invoices = client_items[client_items["saldo_div_int"].abs() > 0.01]
        else:
            # Fallback: assume all are open
            open_invoices = client_items
        
        context["open_invoices"] = open_invoices
        LOGGER.info("Found %d open invoices for %s", len(open_invoices), cliente)
    else:
        context["open_invoices"] = pd.DataFrame()
        LOGGER.warning("line_items.parquet not found")
    
    return context


def prepare_invoice_features(
    open_invoices: pd.DataFrame,
    client_pattern: pd.DataFrame,
    client_cluster: pd.DataFrame,
    project_root: Path
) -> pd.DataFrame:
    """Prepare feature matrix for open invoices.
    
    Prompt 7: Build features for scoring, including client pattern/cluster features.
    """
    if open_invoices.empty:
        return pd.DataFrame()
    
    # Start with invoice-level features
    features = open_invoices.copy()
    
    # Add client pattern features (if available)
    if not client_pattern.empty:
        pattern_cols = [
            "dominant_policy", "pattern_confidence",
            "k_mode", "month_coverage_mean", "month_coverage_p95"
        ]
        pattern_cols = [col for col in pattern_cols if col in client_pattern.columns]
        
        for col in pattern_cols:
            features[col] = client_pattern.iloc[0][col]
    
    # Add cluster features (if available)
    if not client_cluster.empty:
        cluster_cols = ["cluster_label", "membership_prob"]
        cluster_cols = [col for col in cluster_cols if col in client_cluster.columns]
        
        for col in cluster_cols:
            features[col] = client_cluster.iloc[0][col]
    
    # Ensure categorical columns are strings
    categorical_cols = ["tipo_documento", "chiave_contabile", "divisa", "dominant_policy"]
    for col in categorical_cols:
        if col in features.columns:
            features[col] = features[col].fillna("UNKNOWN").astype(str)
    
    return features


def score_invoices(
    features: pd.DataFrame,
    model: object,
    calibrator: object,
    model_type: str
) -> np.ndarray:
    """Score invoices using trained model and calibrator.
    
    Args:
        features: Invoice feature matrix
        model: Trained model (CatBoost or AutoGluon)
        calibrator: Isotonic calibrator
        model_type: "catboost" or "autogluon"
        
    Returns:
        Array of calibrated p_i scores
    """
    if features.empty:
        return np.array([])
    
    # Drop non-feature columns
    drop_cols = [
        "cliente", "numero_documento", "data_documento",
        "saldo_div_int", "imp_div_int"
    ]
    feature_matrix = features.drop(
        columns=[col for col in drop_cols if col in features.columns],
        errors="ignore"
    )
    
    # Get raw probabilities
    if model_type == "catboost":
        from catboost import Pool
        
        # Identify categorical features
        cat_cols = [
            col for col in feature_matrix.columns
            if col in ["tipo_documento", "chiave_contabile", "divisa", "dominant_policy", "macro_tipo"]
        ]
        cat_indices = [feature_matrix.columns.get_loc(col) for col in cat_cols]
        
        pool = Pool(data=feature_matrix, cat_features=cat_indices)
        raw_probs = model.predict_proba(pool)[:, 1]
    
    else:  # autogluon
        raw_probs = model.predict_proba(feature_matrix, as_pandas=False)[:, 1]
    
    # Apply calibration
    calibrated_probs = calibrator.predict(raw_probs)
    
    return calibrated_probs


def compute_agreement_score(
    invoices: pd.DataFrame,
    probabilities: np.ndarray,
    dominant_policy: str
) -> Tuple[float, str]:
    """Compute agreement score between model ranking and dominant pattern.
    
    Prompt 7: Coherence metric comparing p_i with pattern policy.
    
    Args:
        invoices: Open invoices DataFrame
        probabilities: Model p_i scores
        dominant_policy: Client's dominant policy (from pattern detector)
        
    Returns:
        (agreement_score [0,1], alignment_status)
    """
    if len(invoices) == 0 or len(probabilities) == 0:
        return 0.0, "unknown"
    
    # Add probabilities to invoices
    scored = invoices.copy()
    scored["p_i"] = probabilities
    
    # Rank by model (highest p_i first)
    scored = scored.sort_values("p_i", ascending=False).reset_index(drop=True)
    
    # Rank by dominant policy
    policy_ranking = _rank_by_policy(scored, dominant_policy)
    
    if policy_ranking.empty:
        return 0.5, "partial"
    
    # Compute overlap in top-k
    k = min(5, len(scored))  # Top-5 or fewer
    
    model_top_k = set(scored.head(k)["numero_documento"].values)
    policy_top_k = set(policy_ranking.head(k)["numero_documento"].values)
    
    overlap = len(model_top_k & policy_top_k)
    agreement_score = overlap / k if k > 0 else 0.0
    
    # Determine alignment status
    if agreement_score >= 0.8:
        alignment_status = "aligned"
    elif agreement_score >= 0.4:
        alignment_status = "partial"
    else:
        alignment_status = "divergent"
    
    LOGGER.info(
        "Agreement score: %.3f (%s) - %d/%d invoices match %s policy",
        agreement_score, alignment_status, overlap, k, dominant_policy
    )
    
    return agreement_score, alignment_status


def _rank_by_policy(invoices: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Rank invoices according to a specific policy.
    
    Implements policy ranking logic (FIFO, LIFO, largest, etc.).
    """
    if invoices.empty:
        return invoices
    
    df = invoices.copy()
    
    # Ensure columns exist
    if "data_documento" in df.columns:
        df["data_documento"] = pd.to_datetime(df["data_documento"], errors="coerce")
    else:
        df["data_documento"] = pd.NaT
    
    if "imp_div_int" in df.columns:
        df["abs_amount"] = df["imp_div_int"].abs()
    else:
        df["abs_amount"] = 0.0
    
    # Apply policy-specific ranking
    if policy == "fifo":
        df = df.sort_values(["data_documento", "abs_amount"], ascending=[True, False])
    elif policy == "lifo":
        df = df.sort_values(["data_documento", "abs_amount"], ascending=[False, False])
    elif policy == "largest":
        df = df.sort_values(["abs_amount", "data_documento"], ascending=[False, True])
    elif policy == "smallest":
        df = df.sort_values(["abs_amount", "data_documento"], ascending=[True, True])
    elif policy == "eomfirst":
        # End-of-month documents first
        if "data_documento" in df.columns and df["data_documento"].notna().any():
            df["days_to_eom"] = (
                df["data_documento"] + pd.offsets.MonthEnd(0) - df["data_documento"]
            ).dt.days
            df = df.sort_values(["days_to_eom", "data_documento"], ascending=[True, True])
        else:
            df = df.sort_values("abs_amount", ascending=False)
    else:
        # Default: FIFO
        df = df.sort_values("data_documento", ascending=True)
    
    return df


def save_predictions(
    results: List[ScoringResult],
    project_root: Path
) -> Path:
    """Save event_policy_predictions.parquet with p_i and coherence metrics.
    
    Prompt 7: Export predictions + agreement_score.
    """
    curated_dir = project_root / "data" / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)
    
    # Flatten results into DataFrame
    records = []
    for result in results:
        for _, row in result.predictions.iterrows():
            records.append({
                "cliente": result.cliente,
                "numero_documento": row.get("numero_documento"),
                "p_i": row.get("p_i", 0.0),
                "rank": row.get("rank", 0),
                "agreement_score": result.agreement_score,
                "alignment_status": result.alignment_status,
                "dominant_policy": result.dominant_policy,
                "cluster_label": result.cluster_label,
                "pattern_confidence": result.pattern_confidence,
            })
    
    predictions_df = pd.DataFrame(records)
    
    output_path = curated_dir / "event_policy_predictions.parquet"
    predictions_df.to_parquet(output_path, compression="snappy", index=False)
    
    LOGGER.info("Saved %d predictions to %s", len(predictions_df), output_path)
    
    return output_path


def score_client(
    cliente: str,
    model: object,
    calibrator: object,
    model_type: str,
    project_root: Path
) -> ScoringResult:
    """Score all open invoices for a single client.
    
    Prompt 7: Main scoring logic with coherence analysis.
    """
    # Load client context
    context = load_client_context(cliente, project_root)
    
    # Extract pattern/cluster info
    if not context["client_pattern"].empty:
        pattern_row = context["client_pattern"].iloc[0]
        dominant_policy = str(pattern_row.get("dominant_policy", "irregolare"))
        pattern_confidence = float(pattern_row.get("pattern_confidence", 0.0))
    else:
        dominant_policy = "irregolare"
        pattern_confidence = 0.0
    
    if not context["client_cluster"].empty:
        cluster_label = int(context["client_cluster"].iloc[0].get("cluster_label", -1))
    else:
        cluster_label = -1
    
    # Prepare features
    features = prepare_invoice_features(
        context["open_invoices"],
        context["client_pattern"],
        context["client_cluster"],
        project_root
    )
    
    if features.empty:
        LOGGER.warning("No open invoices for client %s", cliente)
        return ScoringResult(
            cliente=cliente,
            predictions=pd.DataFrame(),
            agreement_score=0.0,
            alignment_status="unknown",
            dominant_policy=dominant_policy,
            cluster_label=cluster_label,
            pattern_confidence=pattern_confidence
        )
    
    # Score invoices
    probabilities = score_invoices(features, model, calibrator, model_type)
    
    # Build predictions DataFrame
    predictions = context["open_invoices"][["numero_documento", "data_documento", "imp_div_int"]].copy()
    predictions["p_i"] = probabilities
    predictions = predictions.sort_values("p_i", ascending=False).reset_index(drop=True)
    predictions["rank"] = range(1, len(predictions) + 1)
    
    # Compute agreement score
    agreement_score, alignment_status = compute_agreement_score(
        context["open_invoices"],
        probabilities,
        dominant_policy
    )
    
    return ScoringResult(
        cliente=cliente,
        predictions=predictions,
        agreement_score=agreement_score,
        alignment_status=alignment_status,
        dominant_policy=dominant_policy,
        cluster_label=cluster_label,
        pattern_confidence=pattern_confidence
    )


def main(argv: Optional[List[str]] = None) -> None:
    """Prompt 7: Score open invoices with pattern coherence analysis."""
    parser = argparse.ArgumentParser(
        description="Score open invoices and compute pattern coherence"
    )
    parser.add_argument(
        "--cliente",
        type=str,
        help="Client identifier to score (if omitted, scores all clients with open invoices)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args(argv)
    configure_logging(level=args.log_level)
    
    LOGGER.info("=== Prompt 7: Invoice Scoring with Pattern Coherence ===")
    
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "models"
    
    # Load champion model
    model, calibrator, model_type = load_champion_model(models_dir)
    
    # Get clients to score
    if args.cliente:
        clientes = [args.cliente]
    else:
        # Score all clients with open invoices
        line_items_path = project_root / "data" / "curated" / "line_items.parquet"
        if line_items_path.exists():
            line_items = pd.read_parquet(line_items_path)
            if "saldo_div_int" in line_items.columns:
                open_items = line_items[line_items["saldo_div_int"].abs() > 0.01]
                clientes = open_items["cliente"].unique().tolist()
                LOGGER.info("Found %d clients with open invoices", len(clientes))
            else:
                clientes = line_items["cliente"].unique().tolist()
        else:
            LOGGER.error("line_items.parquet not found")
            return
    
    # Score each client
    results: List[ScoringResult] = []
    for cliente in clientes:
        LOGGER.info("Scoring client: %s", cliente)
        result = score_client(cliente, model, calibrator, model_type, project_root)
        results.append(result)
        
        # Print summary
        if not result.predictions.empty:
            print(f"\nClient: {cliente}")
            print(f"  Dominant policy: {result.dominant_policy}")
            print(f"  Cluster: {result.cluster_label}")
            print(f"  Agreement score: {result.agreement_score:.3f} ({result.alignment_status})")
            print(f"  Top 3 invoices:")
            print(result.predictions.head(3)[["numero_documento", "p_i", "rank"]].to_string(index=False))
    
    # Save predictions
    if results:
        save_predictions(results, project_root)
    
    LOGGER.info("=== Scoring complete ===")


if __name__ == "__main__":
    main()
