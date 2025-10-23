"""
Policy IQ Page - Auditable policy classifier with confusion matrix, uplift analysis, and explain tiles.

PROMPT S9 implementation: visualization of policy classifier performance, uplift metrics,
and interpretable tiles for dominant policies.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Rate: %{z:.2%}<extra></extra>",
        colorbar=dict(title="Rate", tickformat=".0%")
    ))
    
    fig.update_layout(
        title="Policy Classifier Confusion Matrix (Row-Normalized)",
        xaxis_title="Predicted Policy",
        yaxis_title="True Policy",
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
        st.warning("No uplift metrics available.")
        return
    
    st.subheader("Uplift Analysis: Policy-Guided vs Baseline (FIFO)")
    
    # Format the table
    display_df = metrics_df.copy()
    
    # Rename columns for clarity
    display_df = display_df.rename(columns={
        "segment_type": "Segment Type",
        "segment_value": "Segment Value",
        "n_events": "# Events",
        "baseline_recall": "Baseline Recall@k",
        "predicted_recall": "Policy Recall@k",
        "uplift_absolute": "Uplift (abs)",
        "uplift_relative": "Uplift (%)"
    })
    
    # Format numeric columns
    if "Baseline Recall@k" in display_df.columns:
        display_df["Baseline Recall@k"] = display_df["Baseline Recall@k"].apply(lambda x: f"{x:.2%}")
    if "Policy Recall@k" in display_df.columns:
        display_df["Policy Recall@k"] = display_df["Policy Recall@k"].apply(lambda x: f"{x:.2%}")
    if "Uplift (abs)" in display_df.columns:
        display_df["Uplift (abs)"] = display_df["Uplift (abs)"].apply(lambda x: f"{x:+.2%}")
    if "Uplift (%)" in display_df.columns:
        display_df["Uplift (%)"] = display_df["Uplift (%)"].apply(lambda x: f"{x:+.1f}%")
    
    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Summary metrics
    if "uplift_relative" in metrics_df.columns:
        global_uplift = metrics_df[metrics_df["segment_type"] == "global"]["uplift_relative"].values
        if len(global_uplift) > 0:
            st.metric(
                "Global Uplift",
                f"{global_uplift[0]:+.1f}%",
                help="Relative improvement over baseline FIFO policy"
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
        "fifo": "Pay oldest invoices first (by document date)",
        "lifo": "Pay newest invoices first (by document date)",
        "same_period": "Pay invoices from the same accounting period",
        "largest": "Pay largest invoices first (by amount)",
        "smallest": "Pay smallest invoices first (by amount)",
        "eomfirst": "Pay end-of-month invoices first",
        "regmonth": "Pay invoices from the same registration month",
        "quarter": "Pay invoices from the same quarter",
        "seqblock": "Pay consecutive document numbers together",
        "typeprio": "Pay by document type priority"
    }
    
    rule_description = policy_rules.get(policy_name, "Custom policy")
    
    # Create tile container with styled HTML
    tile_html = f"""
    <div style="
        background-color: #F8F9FA;
        border-left: 4px solid #007BFF;
        padding: 16px;
        border-radius: 4px;
        margin-bottom: 16px;
    ">
        <h4 style="margin-top: 0; color: #007BFF;">{policy_name.upper()}</h4>
        <p style="font-size: 14px; color: #495057; margin: 8px 0;">
            <strong>Rule:</strong> {rule_description}
        </p>
        <p style="font-size: 14px; margin: 8px 0;">
            <strong>Recall:</strong> {mean_recall:.1%} | <strong>Stability:</strong> {stability:.1%}
        </p>
    </div>
    """
    
    st.markdown(tile_html, unsafe_allow_html=True)
    
    # Micro-example
    if example_event:
        with st.expander(f"📋 Example Event: {example_event}"):
            example_row = predictions_df[predictions_df["event_key"] == example_event]
            if not example_row.empty:
                row = example_row.iloc[0]
                
                st.write(f"**Predicted Policy:** {row['policy_pred']}")
                st.write(f"**True Policy:** {row['policy_true']}")
                st.write(f"**Accuracy:** {'✅ Correct' if row['accuracy'] else '❌ Incorrect'}")
                st.write(f"**Mean Recall:** {row['mean_recall']:.1%}")
                
                # Show top-k probabilities
                proba_cols = [col for col in row.index if col.startswith("proba_")]
                if proba_cols:
                    proba_data = {col.replace("proba_", ""): row[col] for col in proba_cols}
                    proba_sorted = sorted(proba_data.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    st.write("**Top-3 Policy Probabilities:**")
                    for policy, proba in proba_sorted:
                        st.write(f"- {policy}: {proba:.1%}")


# ============================================================================
# Main Rendering Function
# ============================================================================

def render_policy_iq(data_dir: Path) -> None:
    """
    Render Policy IQ page with confusion matrix, uplift analysis, and explain tiles.
    
    Args:
        data_dir: Path to data directory
    """
    st.title("📊 Policy IQ — Auditable Policy Classifier")
    
    st.markdown("""
    This page provides insights into the **policy classifier** performance:
    - **Confusion Matrix**: How well does the classifier predict the best policy?
    - **Uplift Analysis**: Performance gain over baseline (FIFO) by segment
    - **Explain Tiles**: Dominant policies for selected client with examples
    """)
    
    # Load data
    with st.spinner("Loading policy data..."):
        confusion_df = load_policy_confusion_matrix(data_dir)
        metrics_df = load_policy_metrics(data_dir)
        policy_scores_df = load_per_event_policy_scores(data_dir)
        predictions_df = load_event_policy_predictions(data_dir)
    
    if confusion_df.empty and metrics_df.empty and policy_scores_df.empty:
        st.error("No policy data available. Please run the policy analytics pipeline first.")
        return
    
    # ========================================================================
    # Section 1: Confusion Matrix
    # ========================================================================
    st.header("1. Confusion Matrix")
    st.markdown("**Row-normalized** confusion matrix showing true vs predicted policies.")
    
    fig_confusion = plot_policy_confusion_matrix(confusion_df)
    st.plotly_chart(fig_confusion, use_container_width=True)
    
    # Accuracy metrics
    if not metrics_df.empty:
        global_metrics = metrics_df[metrics_df["segment_type"] == "global"]
        if not global_metrics.empty:
            row = global_metrics.iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if "top1_accuracy" in row:
                    st.metric("Top-1 Accuracy", f"{row['top1_accuracy']:.1%}")
            with col2:
                if "top2_hit_rate" in row:
                    st.metric("Top-2 Hit Rate", f"{row['top2_hit_rate']:.1%}")
            with col3:
                if "n_events" in row:
                    st.metric("Total Events", f"{int(row['n_events'])}")
    
    st.divider()
    
    # ========================================================================
    # Section 2: Uplift Analysis
    # ========================================================================
    st.header("2. Uplift Analysis")
    st.markdown("Compare **policy-guided recall@k** vs **baseline (FIFO)** by segment.")
    
    render_uplift_table(metrics_df)
    
    st.divider()
    
    # ========================================================================
    # Section 3: Explain Tiles for Dominant Policies
    # ========================================================================
    st.header("3. Explain Tiles — Dominant Policies")
    st.markdown("Explore dominant policies for a specific client with examples.")
    
    # Client selector
    if not policy_scores_df.empty:
        available_clients = sorted(policy_scores_df["cliente"].unique())
        
        if available_clients:
            selected_client_tile = st.selectbox(
                "Select Client",
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
                st.markdown(f"### Top Policies for **{selected_client_tile}**")
                
                for policy_info in dominant_policies:
                    render_policy_tile(policy_info, predictions_df)
            else:
                st.info(f"No policy data available for client {selected_client_tile}.")
        else:
            st.warning("No clients found in policy scores data.")
    else:
        st.warning("Policy scores data not available.")
    
    # Footer
    st.divider()
    st.caption("Policy IQ — PROMPT S9 | Data from event_policy_metrics, per_event_policy_scores, event_policy_predictions")
