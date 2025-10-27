"""Pagina Policy IQ: matrice di confusione, analisi uplift e riquadri esplicativi."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.utils.export import export_dataframe_to_csv, generate_deeplink

logger = logging.getLogger(__name__)


# ============================================================================
# Data Loading
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_policy_confusion_matrix(data_dir: Path) -> pd.DataFrame:
    """Load policy confusion matrix from curated data."""
    path = data_dir / "curated" / "event_policy_confusion.parquet"
    if not path.exists():
        logger.warning(f"Policy confusion matrix not found: {path}")
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded policy confusion matrix: {df.shape}")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_policy_metrics(data_dir: Path) -> pd.DataFrame:
    """Load policy uplift metrics from curated data."""
    path = data_dir / "curated" / "event_policy_metrics.parquet"
    if not path.exists():
        logger.warning(f"Policy metrics not found: {path}")
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded policy metrics: {df.shape}")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_per_event_policy_scores(data_dir: Path) -> pd.DataFrame:
    """Load per-event policy scores from curated data."""
    path = data_dir / "curated" / "per_event_policy_scores.parquet"
    if not path.exists():
        logger.warning(f"Per-event policy scores not found: {path}")
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded per-event policy scores: {df.shape}")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_event_policy_predictions(data_dir: Path) -> pd.DataFrame:
    """Load event policy predictions from curated data."""
    path = data_dir / "curated" / "event_policy_predictions.parquet"
    if not path.exists():
        logger.warning(f"Event policy predictions not found: {path}")
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded event policy predictions: {df.shape}")
    return df


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_policy_confusion_matrix(confusion_df: pd.DataFrame) -> go.Figure:
    """
    Create normalized confusion matrix heatmap with percentage annotations.
    
    Args:
        confusion_df: DataFrame with policy_true column and predicted policy columns
        
    Returns:
        Plotly figure with confusion matrix heatmap
    """
    if confusion_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No confusion matrix data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        return fig
    
    # Extract policy names (all columns except policy_true)
    policy_cols = [col for col in confusion_df.columns if col != "policy_true"]
    
    # Build matrix with policy_true as rows
    matrix = confusion_df[policy_cols].values
    y_labels = confusion_df["policy_true"].tolist()
    x_labels = policy_cols
    
    # Create annotations with percentages
    annotations = []
    for i, y_label in enumerate(y_labels):
        for j, x_label in enumerate(x_labels):
            value = matrix[i, j]
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f"{value:.1%}" if value > 0 else "",
                    showarrow=False,
                    font=dict(color="white" if value > 0.5 else "black", size=12)
                )
            )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=x_labels,
        y=y_labels,
        colorscale="Blues",
        text=[[f"{val:.1%}" for val in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="Reale: %{y}<br>Predetta: %{x}<br>Tasso: %{z:.2%}<extra></extra>",
        colorbar=dict(title="Tasso", tickformat=".0%")
    ))
    
    fig.update_layout(
        title="Matrice di confusione (normalizzata per riga)",
        xaxis_title="Policy predetta",
        yaxis_title="Policy reale",
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        height=400,
        width=700,
        margin=dict(l=100, r=50, t=80, b=80)
    )
    
    return fig


def render_uplift_table(metrics_df: pd.DataFrame) -> None:
    """
    Render uplift table comparing policy-guided recall@k vs baseline.
    
    Args:
        metrics_df: DataFrame with segment_type, segment_value, uplift metrics
    """
    if metrics_df.empty:
        st.warning("Nessuna metrica di uplift disponibile.")
        return
    
    st.subheader("Analisi uplift: policy guidata vs baseline FIFO")
    
    # Format the table
    display_df = metrics_df.copy()
    
    # Rename columns for clarity
    display_df = display_df.rename(columns={
        "segment_type": "Tipo segmento",
        "segment_value": "Valore segmento",
        "n_events": "# Eventi",
        "baseline_recall": "Recall baseline@k",
        "predicted_recall": "Recall policy@k",
        "uplift_absolute": "Uplift assoluto",
        "uplift_relative": "Uplift %"
    })
    
    # Format numeric columns
    if "Recall baseline@k" in display_df.columns:
        display_df["Recall baseline@k"] = display_df["Recall baseline@k"].apply(lambda x: f"{x:.2%}")
    if "Recall policy@k" in display_df.columns:
        display_df["Recall policy@k"] = display_df["Recall policy@k"].apply(lambda x: f"{x:.2%}")
    if "Uplift assoluto" in display_df.columns:
        display_df["Uplift assoluto"] = display_df["Uplift assoluto"].apply(lambda x: f"{x:+.2%}")
    if "Uplift %" in display_df.columns:
        display_df["Uplift %"] = display_df["Uplift %"].apply(lambda x: f"{x:+.1f}%")
    
    # Display table
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True
    )
    
    # Summary metrics
    if "uplift_relative" in metrics_df.columns:
        global_uplift = metrics_df[metrics_df["segment_type"] == "global"]["uplift_relative"].values
        if len(global_uplift) > 0:
            st.metric(
                "Uplift globale",
                f"{global_uplift[0]:+.1f}%",
                help="Miglioramento relativo rispetto alla policy FIFO"
            )


def extract_dominant_policies(
    policy_scores_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    cliente: str,
    top_n: int = 3
) -> List[Dict]:
    """
    Extract dominant policies for a client with summary stats.
    
    Args:
        policy_scores_df: Per-event policy scores
        predictions_df: Event policy predictions
        cliente: Client identifier
        top_n: Number of top policies to return
        
    Returns:
        List of dicts with policy info (name, recall, stability, example_event)
    """
    # Filter for client
    client_scores = policy_scores_df[policy_scores_df["cliente"] == cliente].copy()
    client_preds = predictions_df[predictions_df["event_key"].str.startswith(cliente)].copy()
    
    if client_scores.empty or client_preds.empty:
        return []
    
    # Identify policy columns (those starting with "policy_")
    policy_cols = [col for col in client_scores.columns if col.startswith("policy_") and col != "policy_true"]
    
    if not policy_cols:
        return []
    
    # Calculate mean recall per policy
    policy_stats = []
    for policy_col in policy_cols:
        policy_name = policy_col.replace("policy_", "")
        
        # Mean recall (from scores)
        mean_recall = client_scores[policy_col].mean()
        
        # Stability: frequency of this policy being dominant
        # Find corresponding proba column in predictions
        proba_col = f"proba_{policy_name}"
        if proba_col in client_preds.columns:
            stability = (client_preds["policy_pred"] == policy_name).mean()
        else:
            stability = 0.0
        
        # Find an example event where this policy was predicted
        example_event = None
        if proba_col in client_preds.columns:
            example_rows = client_preds[client_preds["policy_pred"] == policy_name]
            if not example_rows.empty:
                example_event = example_rows.iloc[0]["event_key"]
        
        policy_stats.append({
            "policy_name": policy_name,
            "mean_recall": mean_recall,
            "stability": stability,
            "example_event": example_event
        })
    
    # Sort by mean recall descending
    policy_stats.sort(key=lambda x: x["mean_recall"], reverse=True)
    
    return policy_stats[:top_n]


def render_policy_tile(policy_info: Dict, predictions_df: pd.DataFrame) -> None:
    """
    Render an explain tile for a dominant policy.
    
    Args:
        policy_info: Dict with policy_name, mean_recall, stability, example_event
        predictions_df: Event policy predictions for micro-example
    """
    policy_name = policy_info["policy_name"]
    mean_recall = policy_info["mean_recall"]
    stability = policy_info["stability"]
    example_event = policy_info["example_event"]
    
    # Define policy rules (human-readable descriptions)
    policy_rules = {
        "fifo": "Paga prima le fatture più vecchie (per data documento)",
        "lifo": "Paga prima le fatture più recenti (per data documento)",
        "same_period": "Paga le fatture dello stesso periodo contabile",
        "largest": "Paga prima gli importi più grandi",
        "smallest": "Paga prima gli importi più piccoli",
        "eomfirst": "Paga prima le fatture con scadenza a fine mese",
        "regmonth": "Paga le fatture dello stesso mese di registrazione",
        "quarter": "Paga le fatture dello stesso trimestre",
        "seqblock": "Paga insieme numeri documento consecutivi",
        "typeprio": "Paga in base alla priorità del tipo documento",
    }
    
    rule_description = policy_rules.get(policy_name, "Policy personalizzata")
    
    # Create tile container with styled HTML
    tile_html = f"""
    <div style="
        background-color: #FFFFFF;
        border-left: 4px solid #0056B3;
        padding: 16px;
        border-radius: 4px;
        margin-bottom: 16px;
    ">
        <h4 style="margin-top: 0; color: #12355B;">{policy_name.upper()}</h4>
        <p style="font-size: 14px; color: #1F2933; margin: 8px 0;">
            <strong>Regola:</strong> {rule_description}
        </p>
        <p style="font-size: 14px; margin: 8px 0;">
            <strong>Recall:</strong> {mean_recall:.1%} | <strong>Stabilità:</strong> {stability:.1%}
        </p>
    </div>
    """
    
    st.markdown(tile_html, unsafe_allow_html=True)
    
    # Micro-example
    if example_event:
        with st.expander(f"📋 Evento di esempio: {example_event}"):
            example_row = predictions_df[predictions_df["event_key"] == example_event]
            if not example_row.empty:
                row = example_row.iloc[0]
                
                st.write(f"**Policy predetta:** {row['policy_pred']}")
                st.write(f"**Policy reale:** {row['policy_true']}")
                st.write(f"**Accuratezza:** {'✅ Corretta' if row['accuracy'] else '❌ Errata'}")
                st.write(f"**Recall media:** {row['mean_recall']:.1%}")
                
                # Show top-k probabilities
                proba_cols = [col for col in row.index if col.startswith("proba_")]
                if proba_cols:
                    proba_data = {col.replace("proba_", ""): row[col] for col in proba_cols}
                    proba_sorted = sorted(proba_data.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    st.write("**Top-3 probabilità policy:**")
                    for policy, proba in proba_sorted:
                        st.write(f"- {policy}: {proba:.1%}")


# ============================================================================
# Main Rendering Function
# ============================================================================

def render_policy_iq(data_dir: Path, hide_regmonth: bool = False) -> None:
    """
    Render Policy IQ page with confusion matrix, uplift analysis, and explain tiles.
    
    Args:
        data_dir: Path to data directory
        hide_regmonth: If True, filter out regmonth policy from analysis
    """
    st.title("📊 Policy IQ — Analisi classificatore delle policy")
    
    st.markdown(
        """
        Questa pagina offre una panoramica sulle prestazioni del **classificatore di policy**:
        - **Matrice di confusione**: confronta policy vere vs. predette.
        - **Analisi uplift**: misura il guadagno rispetto al baseline FIFO per segmento.
        - **Riquadri esplicativi**: mostra le policy dominanti per un cliente con esempi concreti.
        """
    )
    
    # Load data
    with st.spinner("Caricamento dati delle policy..."):
        confusion_df = load_policy_confusion_matrix(data_dir)
        metrics_df = load_policy_metrics(data_dir)
        policy_scores_df = load_per_event_policy_scores(data_dir)
        predictions_df = load_event_policy_predictions(data_dir)
    
    # PROMPT 6: Filter regmonth if requested
    if hide_regmonth:
        if not confusion_df.empty and "true_policy" in confusion_df.columns:
            confusion_df = confusion_df[confusion_df["true_policy"] != "regmonth"].copy()
            if "pred_policy" in confusion_df.columns:
                confusion_df = confusion_df[confusion_df["pred_policy"] != "regmonth"].copy()
        
        if not metrics_df.empty and "policy" in metrics_df.columns:
            metrics_df = metrics_df[metrics_df["policy"] != "regmonth"].copy()
        
        if not policy_scores_df.empty:
            # Filter policy_regmonth column if exists
            regmonth_cols = [c for c in policy_scores_df.columns if "regmonth" in c.lower()]
            if regmonth_cols:
                policy_scores_df = policy_scores_df.drop(columns=regmonth_cols)
    
    if confusion_df.empty and metrics_df.empty and policy_scores_df.empty:
        st.error("Dati di policy non disponibili. Esegui prima la pipeline di analytics.")
        return
    
    # ========================================================================
    # Section 1: Confusion Matrix
    # ========================================================================
    st.header("1. Matrice di confusione")
    st.markdown("Matrice normalizzata per riga che confronta policy reali e predette.")
    
    fig_confusion = plot_policy_confusion_matrix(confusion_df)
    st.plotly_chart(fig_confusion, width="stretch")
    
    # Accuracy metrics
    if not metrics_df.empty:
        global_metrics = metrics_df[metrics_df["segment_type"] == "global"]
        if not global_metrics.empty:
            row = global_metrics.iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if "top1_accuracy" in row:
                    st.metric("Accuratezza Top-1", f"{row['top1_accuracy']:.1%}")
            with col2:
                if "top2_hit_rate" in row:
                    st.metric("Copertura Top-2", f"{row['top2_hit_rate']:.1%}")
            with col3:
                if "n_events" in row:
                    st.metric("Eventi totali", f"{int(row['n_events'])}")
    
    st.divider()
    
    # ========================================================================
    # Section 2: Uplift Analysis
    # ========================================================================
    st.header("2. Analisi uplift")
    st.markdown("Confronta il **recall@k guidato dalla policy** con il baseline FIFO per segmento.")
    
    render_uplift_table(metrics_df)
    
    st.divider()
    
    # ========================================================================
    # Section 3: Explain Tiles for Dominant Policies
    # ========================================================================
    st.header("3. Riquadri esplicativi — Policy dominanti")
    st.markdown("Esplora le policy predominanti per un cliente specifico con esempi.")
    
    # Client selector
    if not policy_scores_df.empty:
        available_clients = sorted(policy_scores_df["cliente"].unique())
        
        if available_clients:
            selected_client_tile = st.selectbox(
                "Seleziona cliente",
                options=available_clients,
                index=0,
                key="policy_iq_client_selector"
            )
            
            # Extract dominant policies
            dominant_policies = extract_dominant_policies(
                policy_scores_df,
                predictions_df,
                str(selected_client_tile),
                top_n=3
            )
            
            if dominant_policies:
                st.markdown(f"### Policy principali per **{selected_client_tile}**")
                
                for policy_info in dominant_policies:
                    render_policy_tile(policy_info, predictions_df)
            else:
                st.info(f"Nessun dato policy disponibile per il cliente {selected_client_tile}.")
        else:
            st.warning("Nessun cliente trovato nei punteggi policy.")
    else:
        st.warning("Punteggi policy non disponibili.")
    
    # === S12: EXPORT & DEEP-LINK ===
    st.divider()
    st.markdown("### 📤 Esporta & audit")
    
    export_cols = st.columns(3)
    
    with export_cols[0]:
        # Export uplift table CSV
        if not metrics_df.empty:
            csv_bytes = export_dataframe_to_csv(metrics_df)
            st.download_button(
                label="📥 Scarica metriche uplift (CSV)",
                data=csv_bytes,
                file_name="policy_iq_uplift_metrics.csv",
                mime="text/csv",
            )
    
    with export_cols[1]:
        # Export confusion matrix CSV
        if not confusion_df.empty:
            csv_bytes = export_dataframe_to_csv(confusion_df)
            st.download_button(
                label="📥 Scarica matrice di confusione (CSV)",
                data=csv_bytes,
                file_name="policy_iq_confusion_matrix.csv",
                mime="text/csv",
            )
    
    with export_cols[2]:
        # Deep-link
        client_filter = ""
        if not policy_scores_df.empty:
            available_clients = sorted(policy_scores_df["cliente"].unique())
            if available_clients:
                client_filter = str(available_clients[0])  # Default to first client
        
        deeplink_url = generate_deeplink(
            page="Policy IQ",
            filters={"selected_client": client_filter},
        )
        if st.button("📋 Copia deep-link", key="deeplink_policyiq"):
            st.code(deeplink_url, language=None)
            st.caption("⚠️ Copia manualmente l'URL sopra (limitazione di Streamlit).")
    
    # Footer
    st.divider()
    st.caption("Policy IQ | Dati da event_policy_metrics, per_event_policy_scores, event_policy_predictions")
    st.caption("Esporta & audit | I CSV rispettano i filtri correnti e il deep-link mantiene la selezione del cliente")
