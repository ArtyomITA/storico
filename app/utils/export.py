"""Export and audit utilities for Streamlit pages."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import pandas as pd
import streamlit as st


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first matching column name present in the DataFrame."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def generate_event_explain_json(
    event_row: pd.Series,
    policy_scores: Optional[pd.DataFrame],
    predictions: Optional[pd.DataFrame],
    clusters: Optional[pd.DataFrame],
    shap_values: Optional[pd.DataFrame],
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate JSON explain for a single event.
    
    Args:
        event_row: Row from events.parquet with event metadata.
        policy_scores: DataFrame with per_event_policy_scores (optional).
        predictions: DataFrame with event_policy_predictions (optional).
        clusters: DataFrame with client_clusters (optional).
        shap_values: DataFrame with SHAP values (optional).
        feature_names: List of feature names for SHAP top-3 (optional).
    
    Returns:
        Dictionary with ranking_modello, ranking_policy, cluster_label, shap_top3.
    """
    doc_id = event_row.get("numero_documento", "n.d.")
    cliente = event_row.get("cliente", "n.d.")
    event_key = event_row.get("event_key")
    doc_id_str = str(doc_id)
    cliente_str = str(cliente)
    
    # Model ranking (p_i score)
    p_model = event_row.get("p_model", None)
    ranking_modello = {
        "numero_documento": doc_id,
        "cliente": cliente,
        "p_model": float(p_model) if pd.notna(p_model) else None,
        "score_binned": int(event_row.get("score_binned", -1)) if pd.notna(event_row.get("score_binned")) else None,
    }
    
    # Policy ranking
    ranking_policy: Dict[str, Any] = {}
    if policy_scores is not None and not policy_scores.empty:
        doc_col = _find_column(
            policy_scores,
            [
                "numero_documento",
                "documento",
                "doc_id",
                "document_number",
                "invoice_id",
            ],
        )
        cliente_col = _find_column(
            policy_scores,
            ["cliente", "client", "customer", "cliente_id", "id_cliente"],
        )
        event_col = _find_column(policy_scores, ["event_key", "evento", "event_id"])

        mask: Optional[pd.Series] = None
        if doc_col and cliente_col:
            mask = (
                policy_scores[doc_col].astype(str) == doc_id_str
            ) & (policy_scores[cliente_col].astype(str) == cliente_str)
        elif event_col and event_key is not None:
            mask = policy_scores[event_col].astype(str) == str(event_key)

        if mask is not None:
            event_match = policy_scores.loc[mask]
            if not event_match.empty:
                row = event_match.iloc[0]
                ranking_policy = {
                    "fifo_score": float(row.get("fifo_score", 0.0)),
                    "lifo_score": float(row.get("lifo_score", 0.0)),
                    "largest_score": float(row.get("largest_score", 0.0)),
                    "same_period_score": float(row.get("same_period_score", 0.0)),
                    "eomfirst_score": float(row.get("eomfirst_score", 0.0)),
                    "match_rate": float(row.get("match_rate", 0.0)),
                }

    if predictions is not None and not predictions.empty:
        doc_col = _find_column(
            predictions,
            [
                "numero_documento",
                "documento",
                "doc_id",
                "document_number",
                "invoice_id",
            ],
        )
        cliente_col = _find_column(
            predictions,
            ["cliente", "client", "customer", "cliente_id", "id_cliente"],
        )
        event_col = _find_column(predictions, ["event_key", "evento", "event_id"])

        mask: Optional[pd.Series] = None
        if doc_col and cliente_col:
            mask = (
                predictions[doc_col].astype(str) == doc_id_str
            ) & (predictions[cliente_col].astype(str) == cliente_str)
        elif event_col and event_key is not None:
            mask = predictions[event_col].astype(str) == str(event_key)

        if mask is not None:
            pred_match = predictions.loc[mask]
            if not pred_match.empty:
                ranking_policy["predicted_policy"] = str(
                    pred_match.iloc[0].get("predicted_policy", pred_match.iloc[0].get("policy_pred", "n.d."))
                )
    
    # Cluster label
    cluster_label = "n.d."
    if clusters is not None:
        cluster_match = clusters[clusters["cliente"] == cliente]
        if not cluster_match.empty:
            cluster_label = str(cluster_match.iloc[0].get("cluster_display", "n.d."))
    
    # SHAP top-3
    shap_top3 = []
    if shap_values is not None and feature_names is not None:
        # Assume shap_values is a DataFrame with columns = feature_names, indexed same as event_row
        # For simplicity, pass shap_row as Series
        try:
            if isinstance(shap_values, pd.Series):
                shap_row = shap_values
            elif isinstance(shap_values, pd.DataFrame):
                # Find matching row by doc_id
                shap_match = shap_values[shap_values.index == event_row.name]
                if not shap_match.empty:
                    shap_row = shap_match.iloc[0]
                else:
                    shap_row = None
            else:
                shap_row = None
            
            if shap_row is not None and isinstance(shap_row, pd.Series):
                # Get top-3 by absolute value
                shap_abs = shap_row.abs().sort_values(ascending=False)  # type: ignore[call-overload]
                top_feats = shap_abs.head(3).index
                for feat in top_feats:
                    val_raw = shap_row.loc[feat]  # type: ignore[call-overload]
                    val_float = float(val_raw) if not pd.isna(val_raw) else 0.0
                    feat_val_raw = event_row.get(feat, 0.0)
                    feat_val_float = float(feat_val_raw) if not pd.isna(feat_val_raw) else None
                    shap_top3.append({
                        "feature": str(feat),
                        "shap_value": val_float,
                        "feature_value": feat_val_float,
                    })
        except Exception:
            pass  # Graceful fallback if SHAP not available
    
    return {
        "ranking_modello": ranking_modello,
        "ranking_policy": ranking_policy,
        "cluster_label": cluster_label,
        "shap_top3": shap_top3,
    }


def export_dataframe_to_csv(df: pd.DataFrame, filename: str = "export.csv") -> bytes:
    """Export DataFrame to CSV bytes for st.download_button.
    
    Args:
        df: DataFrame to export.
        filename: Suggested filename (ignored, used in caller).
    
    Returns:
        CSV bytes.
    """
    return df.to_csv(index=False).encode("utf-8")


def generate_deeplink(
    page: str,
    filters: Dict[str, Any],
) -> str:
    """Generate deep-link URL with filter state encoded in query string.
    
    Args:
        page: Current page name (e.g., "Pattern Card").
        filters: Dictionary of filter key-value pairs.
    
    Returns:
        URL string with query parameters.
    """
    # Encode filters as query string
    query_params = {"page": page}
    for key, value in filters.items():
        if value is not None:
            if isinstance(value, (list, tuple)):
                query_params[key] = ",".join(str(v) for v in value)
            else:
                query_params[key] = str(value)
    
    # Build URL (Streamlit uses query params API)
    query_string = urlencode(query_params)
    return f"?{query_string}"


def restore_filters_from_deeplink() -> Dict[str, Any]:
    """Restore filter state from URL query parameters.
    
    Returns:
        Dictionary of filter key-value pairs.
    """
    try:
        query_params = st.query_params
        filters = {}
        for key, value in query_params.items():
            if key == "page":
                continue
            # Try to parse comma-separated lists
            if "," in value:
                filters[key] = value.split(",")
            else:
                filters[key] = value
        return filters
    except Exception:
        return {}


def copy_to_clipboard_button(text: str, label: str = "📋 Copy Deep-Link") -> None:
    """Display button that copies text to clipboard.
    
    Args:
        text: Text to copy.
        label: Button label.
    """
    # Streamlit doesn't have native clipboard API, use JavaScript workaround
    # Generate a button that triggers JS copy
    st.markdown(
        f"""
        <button onclick="navigator.clipboard.writeText('{text}')">
            {label}
        </button>
        """,
        unsafe_allow_html=True,
    )
    st.caption("(Click button to copy URL to clipboard)")


def save_export_to_disk(
    content: bytes,
    filename: str,
    data_dir: Path,
    is_mock: bool = False,
) -> Path:
    """Save export file to disk (for audit trail in mock mode).
    
    Args:
        content: File content as bytes.
        filename: Target filename (e.g., "export_20231015_123456.csv").
        data_dir: Root data directory.
        is_mock: If True, save to data_mock/exports/, else data/exports/.
    
    Returns:
        Path to saved file.
    """
    if is_mock:
        export_dir = data_dir.parent / "data_mock" / "exports"
    else:
        export_dir = data_dir / "exports"
    
    export_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = export_dir / filename
    output_path.write_bytes(content)
    
    return output_path


def generate_export_filename(prefix: str, extension: str = "csv") -> str:
    """Generate timestamped export filename.
    
    Args:
        prefix: Filename prefix (e.g., "confusion_matrix").
        extension: File extension without dot (e.g., "csv", "json").
    
    Returns:
        Filename like "confusion_matrix_20231015_143022.csv".
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"
