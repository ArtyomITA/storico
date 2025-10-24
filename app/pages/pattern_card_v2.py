"""Pattern Card v2: Enhanced 2-column layout with visual + metrics components."""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import entropy

from src.analytics.pattern_detectors import POLICY_NAMES
from src.report.nlg_patterns import render_pattern_summary
from app.utils.export import export_dataframe_to_csv, generate_deeplink, copy_to_clipboard_button, generate_event_explain_json


def plot_calendar_heatmap_12mo(events: pd.DataFrame) -> go.Figure:
    """
    Create monthly event heatmap for last 12 months.
    
    Shows day-of-week (rows) vs month (columns) with count of events.
    Tooltip displays exact event counts.
    """
    if events.empty or "data_pareggio" not in events.columns:
        return go.Figure().update_layout(
            title="Calendario eventi (ultimi 12 mesi)",
            annotations=[dict(text="Nessun dato disponibile", showarrow=False, x=0.5, y=0.5)]
        )
    
    tmp = events.dropna(subset=["data_pareggio"]).copy()
    if tmp.empty:
        return go.Figure().update_layout(
            title="Calendario eventi (ultimi 12 mesi)",
            annotations=[dict(text="Nessun evento con data pareggio", showarrow=False, x=0.5, y=0.5)]
        )
    
    # Filter last 12 months
    max_date = tmp["data_pareggio"].max()
    min_date = max_date - pd.DateOffset(months=12)
    tmp = tmp[tmp["data_pareggio"] >= min_date].copy()
    
    if tmp.empty:
        return go.Figure().update_layout(
            title="Calendario eventi (ultimi 12 mesi)",
            annotations=[dict(text="Nessun evento negli ultimi 12 mesi", showarrow=False, x=0.5, y=0.5)]
        )
    
    tmp["dow"] = tmp["data_pareggio"].dt.weekday  # 0=Monday
    tmp["year_month"] = tmp["data_pareggio"].dt.to_period("M").astype(str)
    
    pivot = tmp.pivot_table(
        index="dow",
        columns="year_month",
        values="event_key",
        aggfunc="count",
        fill_value=0,
    ).sort_index()
    
    # Sort columns by date
    sorted_cols = sorted(pivot.columns)
    pivot = pivot[sorted_cols]
    
    dow_labels = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]
    
    heatmap = go.Heatmap(
        z=pivot.to_numpy(),
        x=pivot.columns.tolist(),
        y=[dow_labels[i] for i in pivot.index],
        colorscale="Blues",
        colorbar=dict(title="Eventi"),
        hovertemplate="<b>%{y}</b><br>Mese: %{x}<br><b>Eventi: %{z}</b><extra></extra>",
        text=pivot.to_numpy(),
        texttemplate="%{text}",
        textfont=dict(size=10),
    )
    
    fig = go.Figure(data=heatmap)
    fig.update_layout(
        title="Calendario eventi (ultimi 12 mesi)",
        xaxis=dict(title="Anno-Mese"),
        yaxis=dict(title="Giorno settimana"),
        margin=dict(l=80, r=40, t=60, b=60),
        height=350,
    )
    
    return fig


def plot_cadence_histogram_enhanced(gaps: np.ndarray) -> go.Figure:
    """
    Histogram of inter-event gaps with exact value tooltips.
    
    Args:
        gaps: Array of gap days between events
        
    Returns:
        Plotly histogram figure with numerical tooltips
    """
    if gaps.size == 0:
        return go.Figure().update_layout(
            title="Distribuzione gap tra eventi",
            annotations=[dict(text="Nessun gap disponibile", showarrow=False, x=0.5, y=0.5)]
        )
    
    # Create histogram with explicit bins for tooltip accuracy
    counts, bin_edges = np.histogram(gaps, bins=min(30, int(np.sqrt(gaps.size))))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts,
        marker=dict(color="#4ECDC4", line=dict(color="#2C7A7B", width=1)),
        hovertemplate="<b>Gap giorni: %{x:.1f}</b><br>Frequenza: %{y}<extra></extra>",
        name="Frequenza",
    ))
    
    # Add markers for common periodicities (7, 14, 30 days)
    periodicity_targets = {"Settimanale": 7, "Quindicinale": 14, "Mensile": 30}
    for label, days in periodicity_targets.items():
        if days >= gaps.min() and days <= gaps.max():
            fig.add_vline(
                x=days,
                line=dict(color="red", dash="dash", width=1.5),
                annotation=dict(text=label, textangle=-90, yshift=10)
            )
    
    fig.update_layout(
        title="Distribuzione gap tra eventi",
        xaxis=dict(title="Giorni tra eventi"),
        yaxis=dict(title="Frequenza"),
        margin=dict(l=60, r=40, t=60, b=60),
        height=300,
        showlegend=False,
    )
    
    return fig


def plot_acf_enhanced(gaps: np.ndarray, max_lag: int = 10) -> go.Figure:
    """
    Autocorrelation function with numerical tooltips.
    
    Args:
        gaps: Array of gap days
        max_lag: Maximum lag to display
        
    Returns:
        Plotly ACF figure with exact correlation values
    """
    if gaps.size < 3:
        return go.Figure().update_layout(
            title="Autocorrelazione (ACF)",
            annotations=[dict(text="Dati insufficienti per ACF", showarrow=False, x=0.5, y=0.5)]
        )
    
    # Compute ACF manually
    gaps_demeaned = gaps - gaps.mean()
    max_lag = min(max_lag, gaps.size // 2)
    acf_values = []
    
    variance = np.sum(gaps_demeaned ** 2) / gaps.size
    
    for lag in range(max_lag + 1):
        if lag == 0:
            acf_values.append(1.0)
        else:
            covariance = np.sum(gaps_demeaned[:-lag] * gaps_demeaned[lag:]) / gaps.size
            acf_values.append(covariance / variance if variance > 0 else 0.0)
    
    lags = list(range(max_lag + 1))
    
    # Confidence interval (95%)
    confidence_level = 1.96 / np.sqrt(gaps.size)
    
    fig = go.Figure()
    
    # ACF bars
    fig.add_trace(go.Bar(
        x=lags,
        y=acf_values,
        marker=dict(color="#FF9DA7"),
        hovertemplate="<b>Lag: %{x}</b><br>ACF: %{y:.3f}<extra></extra>",
        name="ACF",
    ))
    
    # Confidence bands
    fig.add_hline(
        y=confidence_level,
        line=dict(color="blue", dash="dash", width=1),
        annotation=dict(text="95% CI", x=max_lag * 0.9, yshift=5)
    )
    fig.add_hline(
        y=-confidence_level,
        line=dict(color="blue", dash="dash", width=1),
    )
    
    fig.update_layout(
        title="Autocorrelazione (ACF)",
        xaxis=dict(title="Lag", dtick=1),
        yaxis=dict(title="ACF", range=[-1, 1]),
        margin=dict(l=60, r=40, t=60, b=60),
        height=300,
        showlegend=False,
    )
    
    return fig


def plot_cardinality_with_metrics(events: pd.DataFrame, k_mode: float, k_entropy: float) -> Tuple[go.Figure, Dict[str, float]]:
    """
    Cardinality distribution histogram with k_mode and k_entropy metrics.
    
    Args:
        events: Events dataframe with n_righe_evento column
        k_mode: Most frequent cardinality value
        k_entropy: Shannon entropy of cardinality distribution
        
    Returns:
        Tuple of (histogram figure, metrics dict)
    """
    metrics = {"k_mode": k_mode, "k_entropy": k_entropy}
    
    if events.empty or "n_righe_evento" not in events.columns:
        fig = go.Figure().update_layout(
            title="Distribuzione cardinalità evento",
            annotations=[dict(text="Nessun dato disponibile", showarrow=False, x=0.5, y=0.5)]
        )
        return fig, metrics
    
    k_values = events["n_righe_evento"].dropna()
    
    if k_values.empty:
        fig = go.Figure().update_layout(
            title="Distribuzione cardinalità evento",
            annotations=[dict(text="Nessun valore di cardinalità", showarrow=False, x=0.5, y=0.5)]
        )
        return fig, metrics
    
    # Compute actual mode and entropy if not provided
    if pd.isna(k_mode) or k_mode == 0:
        k_mode = float(k_values.mode().iloc[0]) if not k_values.mode().empty else 1.0
    
    if pd.isna(k_entropy) or k_entropy == 0:
        value_counts = k_values.value_counts(normalize=True)
        k_entropy = float(entropy(value_counts.values)) if len(value_counts) > 0 else 0.0
    
    metrics = {"k_mode": k_mode, "k_entropy": k_entropy}
    
    # Create histogram
    counts = k_values.value_counts().sort_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker=dict(color="#8CD17D", line=dict(color="#4A7B5C", width=1)),
        hovertemplate="<b>Cardinalità k=%{x}</b><br>Frequenza: %{y} eventi<extra></extra>",
        name="Frequenza",
    ))
    
    # Add vertical line at k_mode
    if not pd.isna(k_mode):
        fig.add_vline(
            x=k_mode,
            line=dict(color="red", dash="dash", width=2),
            annotation=dict(text=f"Mode: k={k_mode:.0f}", yshift=10)
        )
    
    fig.update_layout(
        title="Distribuzione cardinalità evento",
        xaxis=dict(title="Numero righe (k)", dtick=1),
        yaxis=dict(title="Frequenza (eventi)"),
        margin=dict(l=60, r=40, t=60, b=60),
        height=300,
        showlegend=False,
    )
    
    return fig, metrics


def plot_policy_bars_enhanced(score_row: pd.Series) -> go.Figure:
    """
    Policy recall bars with exact recall values and confidence tooltips.
    
    Args:
        score_row: Series with policy scores and confidence
        
    Returns:
        Plotly bar chart with numerical tooltips
    """
    policy_data = []
    
    for policy in POLICY_NAMES:
        recall = float(score_row.get(f"{policy}_score", 0.0))
        policy_data.append({
            "policy": policy.upper(),
            "recall": recall,
        })
    
    df = pd.DataFrame(policy_data)
    
    # Get confidence if available
    confidence = float(score_row.get("confidence", 0.0))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df["policy"],
        y=df["recall"],
        marker=dict(color="#59A14F", line=dict(color="#2F5827", width=1)),
        hovertemplate="<b>%{x}</b><br>Recall: %{y:.2%}<br>Confidence: " + f"{confidence:.2f}<extra></extra>",
        name="Recall",
        text=df["recall"].apply(lambda x: f"{x:.1%}"),
        textposition="outside",
    ))
    
    fig.update_layout(
        title=f"Recall medio per policy (confidence: {confidence:.2f})",
        xaxis=dict(title="Policy"),
        yaxis=dict(title="Recall", range=[0, min(1.0, df["recall"].max() * 1.15)], tickformat=".0%"),
        margin=dict(l=60, r=40, t=80, b=60),
        height=350,
        showlegend=False,
    )
    
    return fig


def render_claim_evidence_box(
    selected_client: str,
    score_row: pd.Series,
    pattern_row: Optional[pd.Series] = None
) -> None:
    """
    Render Claim + Evidence section with formatted pattern summary.
    
    Args:
        selected_client: Client identifier
        score_row: Pattern scores for client
        pattern_row: Optional pattern table row with metadata
    """
    st.markdown("### 💡 Claim + Evidence")
    
    if score_row.empty:
        st.info("Nessuna metrica disponibile per generare claim.")
        return
    
    # Call NLG pattern summary
    summary_lines = render_pattern_summary(
        selected_client,
        score_row=score_row,
        pattern_row=pattern_row,
    )
    
    if not summary_lines:
        st.info("Sintesi non disponibile per questo cliente.")
        return
    
    # Render in styled container
    st.markdown("""
    <style>
    .claim-evidence-box {
        background-color: #F0F8FF;
        border-left: 4px solid #4682B4;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .claim-evidence-box ul {
        margin: 5px 0;
        padding-left: 20px;
    }
    .claim-evidence-box li {
        margin: 8px 0;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    claims_html = "<div class='claim-evidence-box'><ul>"
    for line in summary_lines:
        claims_html += f"<li>{line}</li>"
    claims_html += "</ul></div>"
    
    st.markdown(claims_html, unsafe_allow_html=True)


def render_pattern_card_v2(
    selected_client: str,
    events: pd.DataFrame,
    pattern_scores: pd.DataFrame,
    pattern_table_df: pd.DataFrame,
    feature_repository: Optional[pd.DataFrame] = None,
    cat_feature_cols: Optional[Sequence[str]] = None,
    cat_cat_indices: Optional[Sequence[int]] = None,
    auto_feature_cols: Optional[Sequence[str]] = None,
    developer_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render Pattern Card v2 with 2-column layout: visual + metrics + explainability.
    
    Args:
        selected_client: Client identifier
        events: Events dataframe
        pattern_scores: Client pattern scores
        pattern_table_df: Pattern table with labels and metadata
        feature_repository: Optional feature matrix for explainability
        cat_feature_cols: Optional CatBoost feature columns
        cat_cat_indices: Optional CatBoost categorical indices
        auto_feature_cols: Optional AutoGluon feature columns
        developer_metrics: Optional metrics tracking dict
    """
    st.title(f"📊 Pattern Card — Cliente {selected_client}")
    
    # Filter data for selected client
    client_events = events[events["cliente"] == selected_client].copy()
    
    score_row = (
        pattern_scores[pattern_scores["cliente"] == selected_client].iloc[0]
        if not pattern_scores.empty and selected_client in pattern_scores["cliente"].values
        else pd.Series(dtype=float)
    )
    
    pattern_row = (
        pattern_table_df[pattern_table_df["cliente"] == selected_client].iloc[0]
        if not pattern_table_df.empty and selected_client in pattern_table_df["cliente"].values
        else None
    )
    
    if score_row.empty:
        st.warning("Nessuna metrica disponibile per il cliente selezionato.")
        return
    
    # === 2-COLUMN LAYOUT ===
    col_visual, col_metrics = st.columns([1, 1], gap="large")
    
    # LEFT COLUMN: VISUALS
    with col_visual:
        st.markdown("### 📅 Visual Analytics")
        
        # Calendar heatmap (last 12 months)
        st.plotly_chart(
            plot_calendar_heatmap_12mo(client_events),
            use_container_width=True,
            key="calendar_heatmap"
        )
        
        # Cadence histogram
        gaps = client_events["inter_event_gap_days"].dropna().to_numpy(dtype=float)
        st.plotly_chart(
            plot_cadence_histogram_enhanced(gaps),
            use_container_width=True,
            key="cadence_histogram"
        )
        
        # ACF
        st.plotly_chart(
            plot_acf_enhanced(gaps, max_lag=10),
            use_container_width=True,
            key="acf_plot"
        )
    
    # RIGHT COLUMN: METRICS
    with col_metrics:
        st.markdown("### 📊 Metrics Dashboard")
        
        # Cardinality with metrics
        k_mode = float(score_row.get("k_mode", 0.0))
        k_entropy = float(score_row.get("k_entropy", 0.0))
        
        card_fig, card_metrics = plot_cardinality_with_metrics(client_events, k_mode, k_entropy)
        st.plotly_chart(card_fig, use_container_width=True, key="cardinality_hist")
        
        # Metrics cards
        metric_cols = st.columns(2)
        metric_cols[0].metric("🎯 k_mode", f"{card_metrics['k_mode']:.1f}")
        metric_cols[1].metric("📈 k_entropy", f"{card_metrics['k_entropy']:.2f}")
        
        st.markdown("---")
        
        # Policy bars with recall + confidence
        st.plotly_chart(
            plot_policy_bars_enhanced(score_row),
            use_container_width=True,
            key="policy_bars"
        )
        
        # Pattern label card
        if pattern_row is not None:
            st.markdown("#### Pattern riconosciuto")
            pattern_label = str(pattern_row.get("pattern_label", "n.d."))
            confidence = float(pattern_row.get("confidence", 0.0))
            stability = float(pattern_row.get("stability_index", 0.0))
            
            label_cols = st.columns(3)
            label_cols[0].metric("Label", pattern_label)
            label_cols[1].metric("Confidence", f"{confidence:.2f}")
            label_cols[2].metric("Stability", f"{stability:.2f}")
    
    # === CLAIM + EVIDENCE SECTION (full width) ===
    st.markdown("---")
    render_claim_evidence_box(selected_client, score_row, pattern_row)
    
    # === ADDITIONAL METRICS (full width) ===
    st.markdown("---")
    st.markdown("### 📏 Additional Metrics")
    
    extra_cols = st.columns(4)
    extra_cols[0].metric("DOW concentration", f"{float(score_row.get('dow_concentration', 0.0)):.2f}")
    extra_cols[1].metric("DOM concentration", f"{float(score_row.get('dom_concentration', 0.0)):.2f}")
    extra_cols[2].metric("# Eventi", f"{len(client_events)}")
    extra_cols[3].metric("Gap medio", f"{gaps.mean():.1f} giorni" if gaps.size > 0 else "n.d.")
    
    # === EXPLAINABILITY PANEL (full width) ===
    if (feature_repository is not None and not feature_repository.empty 
        and cat_feature_cols is not None and cat_cat_indices is not None):
        render_explainability_panel(
            selected_client=selected_client,
            events=events,
            feature_repository=feature_repository,
            cat_feature_cols=cat_feature_cols,
            cat_cat_indices=cat_cat_indices,
            auto_feature_cols=auto_feature_cols or [],
            developer_metrics=developer_metrics,
        )
    
    # === S12: EXPORT & DEEP-LINK ===
    st.markdown("---")
    st.markdown("### 📤 Export & Audit (PROMPT S12)")
    
    export_cols = st.columns(2)
    
    with export_cols[0]:
        # CSV export for pattern table
        if not pattern_table_df.empty:
            csv_bytes = export_dataframe_to_csv(pattern_table_df)
            st.download_button(
                label="📥 Download Pattern Table CSV",
                data=csv_bytes,
                file_name=f"pattern_table_{selected_client}.csv",
                mime="text/csv",
            )
    
    with export_cols[1]:
        # Deep-link copy
        deeplink_url = generate_deeplink(
            page="Pattern Card",
            filters={"cliente": selected_client},
        )
        if st.button("📋 Copy Deep-Link"):
            st.code(deeplink_url, language=None)
            st.caption("⚠️ Copy the URL above manually (Streamlit limitation).")
    
    st.caption("Export & Audit — PROMPT S12 | CSV exports respect current filters; deep-link preserves cliente selection")


def apply_policy_boost(
    df: pd.DataFrame,
    policy_name: str,
    boost_alpha: float,
    original_probs: pd.Series
) -> pd.DataFrame:
    """
    Apply policy boost to recalculate scores with gating.
    
    Args:
        df: Event dataframe with policy indicator columns
        policy_name: Policy to boost (e.g., "same_period", "eomfirst")
        boost_alpha: Boost multiplier (e.g., 0.2 for 20% boost)
        original_probs: Original model probabilities
        
    Returns:
        DataFrame with boosted probabilities and ranking
    """
    df = df.copy()
    df["p_original"] = original_probs
    df["p_boosted"] = original_probs.copy()
    
    # Apply boost to rows matching policy
    policy_col = policy_name.lower()
    if policy_col in df.columns:
        mask = df[policy_col] == 1
        df.loc[mask, "p_boosted"] = df.loc[mask, "p_original"] * (1 + boost_alpha)
        # Clip to [0, 1]
        df["p_boosted"] = df["p_boosted"].clip(0, 1)
    
    # Recompute ranking
    df["rank_original"] = df["p_original"].rank(ascending=False, method="first").astype(int)
    df["rank_boosted"] = df["p_boosted"].rank(ascending=False, method="first").astype(int)
    df["rank_delta"] = df["rank_original"] - df["rank_boosted"]  # Positive = moved up
    
    return df.sort_values("p_boosted", ascending=False)


def render_explainability_panel(
    selected_client: str,
    events: pd.DataFrame,
    feature_repository: pd.DataFrame,
    cat_feature_cols: Sequence[str],
    cat_cat_indices: Sequence[int],
    auto_feature_cols: Sequence[str],
    developer_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render explainability panel with SHAP top-k and what-if policy gating.
    
    Args:
        selected_client: Client identifier
        events: Events dataframe
        feature_repository: Feature matrix with all invoice-level features
        cat_feature_cols: CatBoost feature column names
        cat_cat_indices: CatBoost categorical feature indices
        auto_feature_cols: AutoGluon feature column names
        developer_metrics: Optional metrics tracking dict
    """
    st.markdown("---")
    st.markdown("### 🔍 Explainability Panel")
    st.caption("Comprendi **perché** il modello propone certe fatture e simula l'effetto di policy gating")
    
    # Filter events for client
    client_events = events[events["cliente"] == selected_client].copy()
    
    if client_events.empty:
        st.info("Nessun evento disponibile per il cliente selezionato.")
        return
    
    # Event selector
    event_options = client_events["event_key"].unique().tolist()
    if not event_options:
        st.info("Nessun event_key disponibile per explainability.")
        return
    
    selected_event = st.selectbox(
        "🎯 Seleziona evento da analizzare",
        event_options,
        key="explainability_event_selector"
    )
    
    # Load event features
    event_features = feature_repository[feature_repository["event_key"] == selected_event].copy()
    
    if event_features.empty:
        st.warning(f"Nessuna feature disponibile per evento {selected_event}.")
        return
    
    # Model selector
    model_choice = st.radio(
        "📊 Modello",
        ["CatBoost", "AutoGluon"],
        horizontal=True,
        key="explainability_model_selector"
    )
    model_type = "catboost" if model_choice == "CatBoost" else "autogluon"
    
    # Load models and compute predictions + SHAP
    from app.streamlit_app import (
        load_catboost_bundle,
        load_autogluon_predictor,
        safe_calibrate
    )
    
    catboost_model, calibrator = None, None
    autogluon_predictor = None
    shap_values = None
    
    if model_type == "catboost":
        catboost_model, calibrator = load_catboost_bundle()
        if catboost_model is None:
            st.error("Modello CatBoost non disponibile.")
            return
        
        from catboost import Pool  # type: ignore
        
        model_input = event_features[cat_feature_cols]
        pool = Pool(model_input, cat_features=list(cat_cat_indices) if cat_cat_indices else None)
        probs = catboost_model.predict_proba(pool)[:, 1]
        probs = safe_calibrate(calibrator, probs)
        
        try:
            shap_values = catboost_model.get_feature_importance(pool, type="ShapValues")
        except Exception as e:
            st.warning(f"SHAP values non disponibili: {e}")
            shap_values = None
        
        event_features["p_model"] = probs
        
    else:  # autogluon
        autogluon_predictor = load_autogluon_predictor()
        if autogluon_predictor is None:
            st.error("Modello AutoGluon non disponibile.")
            return
        
        model_input = event_features[auto_feature_cols]
        proba_df = autogluon_predictor.predict_proba(model_input)
        if isinstance(proba_df, pd.DataFrame):
            if 1 in proba_df.columns:
                probs = proba_df[1].to_numpy()
            else:
                probs = proba_df.iloc[:, -1].to_numpy()
        else:
            probs = np.asarray(proba_df)
        
        event_features["p_model"] = probs
    
    # === TOP-K PROPOSALS TABLE ===
    st.markdown("#### 📋 Top-k Proposte")
    
    k = st.slider("Numero fatture da mostrare (k)", 1, min(20, len(event_features)), 5, key="explainability_k")
    
    top_k = event_features.nlargest(k, "p_model").copy()
    top_k["numero_documento"] = top_k["numero_documento"].astype(str)
    top_k["label_actual"] = top_k["label"].fillna(0).astype(int)
    
    display_df = top_k[["numero_documento", "p_model", "label_actual"]].copy()
    display_df.columns = ["Numero Documento", "Probabilità modello (p_i)", "Label reale"]
    display_df["Probabilità modello (p_i)"] = display_df["Probabilità modello (p_i)"].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # === S12: JSON EXPLAIN DOWNLOAD FOR FIRST EVENT ===
    if not top_k.empty:
        st.markdown("#### 📥 JSON Explain Download (PROMPT S12)")
        st.caption("Download structured JSON explain for first ranked event with model ranking, policy scores, cluster label, SHAP top-3")
        
        first_event_row = top_k.iloc[0]
        
        # Load policy scores and predictions for explain JSON
        policy_scores_df = None
        predictions_df = None
        clusters_df = None
        try:
            base_dir = Path(__file__).resolve().parents[2]
            curated_dir = base_dir / "data" / "curated"
            
            policy_scores_path = curated_dir / "per_event_policy_scores.parquet"
            if policy_scores_path.exists():
                policy_scores_df = pd.read_parquet(policy_scores_path)
            
            predictions_path = curated_dir / "event_policy_predictions.parquet"
            if predictions_path.exists():
                predictions_df = pd.read_parquet(predictions_path)
            
            clusters_path = curated_dir / "client_clusters.parquet"
            if clusters_path.exists():
                clusters_df = pd.read_parquet(clusters_path)
        except Exception:
            pass
        
        # Prepare SHAP values for first event
        shap_series = None
        if model_type == "catboost" and shap_values is not None and cat_feature_cols is not None:
            try:
                doc_id = first_event_row["numero_documento"]
                row_idx = event_features[event_features["numero_documento"] == doc_id].index[0]
                shap_row = shap_values[row_idx, :-1]
                shap_series = pd.Series(data=shap_row, index=list(cat_feature_cols))  # type: ignore[call-overload]
            except Exception:
                pass
        
        feat_names_list = list(cat_feature_cols) if cat_feature_cols is not None else None
        
        explain_json = generate_event_explain_json(
            event_row=first_event_row,
            policy_scores=policy_scores_df,
            predictions=predictions_df,
            clusters=clusters_df,
            shap_values=shap_series,  # type: ignore[arg-type]
            feature_names=feat_names_list,
        )
        
        json_bytes = json.dumps(explain_json, indent=2).encode("utf-8")
        st.download_button(
            label="📥 Download JSON Explain (first event)",
            data=json_bytes,
            file_name=f"explain_event_{first_event_row['numero_documento']}.json",
            mime="application/json",
        )
    
    # === SHAP TOP-3 FOR FIRST 2 INVOICES ===
    if model_type == "catboost" and shap_values is not None:
        st.markdown("#### 🔬 SHAP Explanations (prime 2 fatture)")
        
        for idx, (_, row) in enumerate(top_k.head(2).iterrows()):
            doc_id = row["numero_documento"]
            row_idx = event_features[event_features["numero_documento"] == doc_id].index[0]
            
            # Get SHAP values for this row (exclude last column which is base value)
            shap_row = shap_values[row_idx, :-1]
            
            # Get top-3 features by absolute SHAP value
            ranked_feats = sorted(
                zip(cat_feature_cols, shap_row),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:3]
            
            # Display
            with st.expander(f"📄 Fattura **{doc_id}** (p_i = {row['p_model']:.4f})", expanded=idx == 0):
                shap_df = pd.DataFrame([
                    {
                        "Feature": feat,
                        "Valore": f"{row[feat]:.2f}" if pd.notna(row.get(feat)) and isinstance(row.get(feat), (int, float)) else str(row.get(feat, "n.d.")),
                        "Contributo SHAP": f"{shap_val:+.4f}"
                    }
                    for feat, shap_val in ranked_feats
                ])
                st.dataframe(shap_df, use_container_width=True, hide_index=True)
    
    # === WHAT-IF POLICY GATING ===
    st.markdown("---")
    st.markdown("#### 🎛️ What-If Policy Gating")
    st.caption("Simula l'effetto di boost su policy specifiche: `p_i' = p_i × (1 + α)`")
    
    # Policy boost controls
    st.markdown("**Seleziona policy da boostare:**")
    
    boost_cols = st.columns(3)
    
    with boost_cols[0]:
        same_period_boost = st.checkbox("Same Period", value=False, key="boost_same_period")
        eomfirst_boost = st.checkbox("EOM First", value=False, key="boost_eomfirst")
    
    with boost_cols[1]:
        largest_boost = st.checkbox("Largest", value=False, key="boost_largest")
        fifo_boost = st.checkbox("FIFO", value=False, key="boost_fifo")
    
    with boost_cols[2]:
        lifo_boost = st.checkbox("LIFO", value=False, key="boost_lifo")
    
    # Alpha slider
    alpha = st.slider(
        "Boost multiplier (α)",
        0.0,
        1.0,
        0.2,
        0.05,
        key="boost_alpha",
        help="Percentuale di boost: 0.2 = +20%"
    )
    
    # Apply boosts if any selected
    active_boosts = []
    if same_period_boost:
        active_boosts.append("same_period")
    if eomfirst_boost:
        active_boosts.append("eomfirst")
    if largest_boost:
        active_boosts.append("largest")
    if fifo_boost:
        active_boosts.append("fifo")
    if lifo_boost:
        active_boosts.append("lifo")
    
    if active_boosts:
        st.markdown(f"**Policy attive:** {', '.join([p.upper() for p in active_boosts])} con α={alpha:.2f}")
        
        # Apply composite boost (cumulative)
        boosted_df = event_features.copy()
        boosted_df["p_original"] = boosted_df["p_model"]
        boosted_df["p_boosted"] = boosted_df["p_model"].copy()
        
        for policy in active_boosts:
            policy_col = policy.lower()
            if policy_col in boosted_df.columns:
                mask = boosted_df[policy_col] == 1
                boosted_df.loc[mask, "p_boosted"] = boosted_df.loc[mask, "p_boosted"] * (1 + alpha)
        
        # Clip and rerank
        boosted_df["p_boosted"] = boosted_df["p_boosted"].clip(0, 1)
        boosted_df["rank_original"] = boosted_df["p_original"].rank(ascending=False, method="first").astype(int)
        boosted_df["rank_boosted"] = boosted_df["p_boosted"].rank(ascending=False, method="first").astype(int)
        boosted_df["rank_delta"] = boosted_df["rank_original"] - boosted_df["rank_boosted"]
        
        # Show top-k with ranking changes
        top_k_boosted = boosted_df.nlargest(k, "p_boosted").copy()
        top_k_boosted["numero_documento"] = top_k_boosted["numero_documento"].astype(str)
        
        comparison_df = top_k_boosted[[
            "numero_documento",
            "p_original",
            "p_boosted",
            "rank_original",
            "rank_boosted",
            "rank_delta"
        ]].copy()
        comparison_df.columns = [
            "Numero Documento",
            "p_i (originale)",
            "p_i' (boosted)",
            "Rank originale",
            "Rank boosted",
            "Δ Rank"
        ]
        comparison_df["p_i (originale)"] = comparison_df["p_i (originale)"].apply(lambda x: f"{x:.4f}")
        comparison_df["p_i' (boosted)"] = comparison_df["p_i' (boosted)"].apply(lambda x: f"{x:.4f}")
        
        # Color code rank delta
        def color_rank_delta(val: int) -> str:
            if val > 0:
                return f"background-color: #D4EDDA; color: #155724"  # Green (moved up)
            elif val < 0:
                return f"background-color: #F8D7DA; color: #721C24"  # Red (moved down)
            else:
                return ""
        
        styled_df = comparison_df.style.applymap(
            color_rank_delta,
            subset=["Δ Rank"]
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Summary stats
        moved_up = (boosted_df["rank_delta"] > 0).sum()
        moved_down = (boosted_df["rank_delta"] < 0).sum()
        unchanged = (boosted_df["rank_delta"] == 0).sum()
        
        summary_cols = st.columns(3)
        summary_cols[0].metric("📈 Mosse in alto", moved_up)
        summary_cols[1].metric("📉 Mosse in basso", moved_down)
        summary_cols[2].metric("➡️ Invariate", unchanged)
    else:
        st.info("Seleziona almeno una policy per simulare il boost.")
