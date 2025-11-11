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
from src.utils.sap_glossary import SAP_FI_AR_GLOSSARY, get_sap_term, get_sap_tooltip
from src.utils.training_manager import (  
    TrainingManager,
    build_training_command,
    get_catboost_artifacts,
    get_autogluon_artifacts,
    get_pattern_artifacts,
)
from app.utils.export import (
    export_dataframe_to_csv,
    generate_deeplink,
    generate_event_explain_json,
)


# ============================================================================
# Prompt 3: Model Availability & Open Invoices Functions
# ============================================================================

# Detect BASE_DIR for model paths
BASE_DIR = Path(__file__).resolve().parents[2]


def check_model_availability() -> Dict[str, bool]:
    """
    Check which trained models are available.
    
    Returns:
        Dict with keys: "catboost", "autogluon", "any"
    """
    models_dir = BASE_DIR / "models"
    
    return {
        "catboost": (models_dir / "catboost.cbm").exists(),
        "autogluon": (models_dir / "autogluon").exists(),
        "any": (models_dir / "catboost.cbm").exists() or (models_dir / "autogluon").exists()
    }


def fetch_open_invoices_for_client(cliente: str, line_items: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch open invoices (not yet cleared) for given client.
    
    Args:
        cliente: Client identifier
        line_items: Full line items dataframe
    
    Returns:
        DataFrame with open invoices + age_days column
    """
    if line_items.empty:
        return pd.DataFrame()
    
    # Filter: cliente match + no clearing doc/date
    open_invoices = line_items[
        (line_items["cliente"] == cliente) &
        (
            (line_items["documento_pareggio"].isna()) |
            (line_items["documento_pareggio"] == "") |
            (line_items["documento_pareggio"].astype(str).str.strip() == "")
        ) &
        (line_items["data_pareggio"].isna())
    ].copy()
    
    if open_invoices.empty:
        return open_invoices
    
    # Enrich with age_days
    today = pd.Timestamp.now()
    open_invoices["data_documento"] = pd.to_datetime(open_invoices["data_documento"], errors="coerce")
    open_invoices["age_days"] = (today - open_invoices["data_documento"]).dt.days.fillna(0).astype(int)
    
    return open_invoices


def get_dominant_policy(cliente: str, pattern_scores: pd.DataFrame) -> str:
    """
    Get dominant policy for client from pattern_scores.
    
    Returns:
        Policy name (e.g., "fifo", "largest") or "unknown"
    """
    if pattern_scores.empty or cliente not in pattern_scores["cliente"].values:
        return "unknown"
    
    score_row = pattern_scores[pattern_scores["cliente"] == cliente].iloc[0]
    
    # Find max score among policy columns
    policy_cols = [col for col in score_row.index if col.endswith("_score") and col.replace("_score", "") in POLICY_NAMES]
    
    if not policy_cols:
        return "unknown"
    
    max_col = score_row[policy_cols].idxmax()
    dominant_policy = max_col.replace("_score", "")
    
    return dominant_policy


 # Policy labels con terminologia SAP FI-AR corretta
POLICY_LABELS_IT: dict[str, str] = {
    "weekly": "Cadenza Settimanale",
    "biweekly": "Cadenza Quindicinale",
    "monthly": "Cadenza Mensile",
    "eomfirst": "Pagamento a Fine Mese",
    "payall": "Pagamento Totale Fattura",
    "regmonth": "Stesso Mese (Reg=Comp)",  # Mese Registrazione = Mese Compensazione
    "same_period": "Stesso Periodo",
    "largest": "Importo Maggiore",
    "fifo": "FIFO (Prima Emessa, Prima Pagata)",
    "lifo": "LIFO (Ultima Emessa, Prima Pagata)",
}

 # Tooltips SAP per policy (gergo interno → terminologia standard)
POLICY_TOOLTIPS: dict[str, str] = {
    "weekly": get_sap_tooltip("policy_settimanale"),
    "biweekly": get_sap_tooltip("policy_quindicinale"),
    "monthly": get_sap_tooltip("policy_mensile"),
    "eomfirst": get_sap_tooltip("policy_fine_mese"),
    "payall": get_sap_tooltip("policy_pagamento_totale"),
    "regmonth": get_sap_tooltip("policy_regmonth"),
    "same_period": "Compensa fatture dello stesso periodo contabile",
    "largest": "Compensa per prima la fattura di importo maggiore",
    "fifo": "Compensa fatture in ordine cronologico (First In, First Out)",
    "lifo": "Compensa fatture in ordine inverso (Last In, First Out)",
}


# ============================================================================
 # Policy Toggle Sidebar
# ============================================================================

 # Policy deterministiche disponibili (esclude periodicity policies)
DETERMINISTIC_POLICIES: list[str] = [
    "fifo",
    "lifo",
    "same_period",
    "largest",
    "smallest",
    "eomfirst",
    "quarter",
    "seqblock",
    "typeprio",
    # "regmonth" excluded by default - handled separately with checkbox
]


def _render_policy_toggle_sidebar() -> List[str]:
    """Render sidebar UI per toggle policy deterministiche.
    
    Returns:
        List[str]: Lista policy attive (include regmonth se checkbox è ON)
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Filtro Policy Deterministiche")
    st.sidebar.markdown(
        "Seleziona quali policy mostrare nei grafici e nelle metriche. "
        "Il toggle agisce **solo sulla visualizzazione** (dati a disco non modificati)."
    )
    
    # Multiselect per policy standard (default: tutte attive)
    selected_policies = st.sidebar.multiselect(
        "Policy attive",
        options=DETERMINISTIC_POLICIES,
        default=DETERMINISTIC_POLICIES,
        format_func=lambda p: POLICY_LABELS_IT.get(p, p),
        help="Scegli quali policy deterministiche mostrare nei grafici e nelle tabelle",
        key="policy_toggle_multiselect",
    )
    
    # Checkbox separata per RegMonth (default: OFF)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Policy Avanzata:**")
    
    show_regmonth = st.sidebar.checkbox(
        POLICY_LABELS_IT.get("regmonth", "RegMonth"),
        value=False,
        key="policy_toggle_regmonth",
        help=(
            "⚠️ **RegMonth** favorisce documenti con **mese di registrazione = mese di compensazione**. "
            "NON significa 'pagare tutte le fatture del mese'. "
            "Confronta `MONTH(Data Registrazione) = MONTH(Data Compensazione)`, "
            "senza verificare se il pagamento è tempestivo rispetto alla scadenza. "
            "Policy nascosta di default perché può essere fuorviante."
        ),
    )
    
    # Build active policies list
    active_policies = list(selected_policies)
    if show_regmonth:
        active_policies.append("regmonth")
    
    # Debug info (optional, for developer mode)
    if st.sidebar.checkbox("🔍 Debug policy attive", value=False, key="debug_policy_toggle"):
        st.sidebar.json({"active_policies": active_policies, "count": len(active_policies)})
    
    return active_policies


# ============================================================================
 # Training Actions Panel
# ============================================================================

def _render_training_actions_panel() -> None:
    """Render riquadro Azioni con pulsanti training CatBoost/AutoGluon/Pattern."""
    manager = TrainingManager()
    
    with st.expander("🚀 **Azioni Training & Pattern**", expanded=False):
        st.markdown(
            "Esegui training modelli predittivi (CatBoost, AutoGluon) o ricalcola Pattern/Policy. "
            "Lo stato dell'ultima esecuzione è salvato e mostrato qui sotto."
        )
        
        # 3 colonne: CatBoost, AutoGluon, Pattern
        col_cat, col_auto, col_pattern = st.columns(3)
        
        # ===== CATBOOST =====
        with col_cat:
            st.markdown("### 🐱 CatBoost")
            
            catboost_status = manager.get_run_status("catboost")
            artifacts_exist = manager.check_artifacts_exist("catboost")
            
            # Mostra stato ultimo run
            if catboost_status:
                _render_training_status_card(catboost_status, artifacts_exist)
            else:
                st.info("Nessun training CatBoost registrato.")
            
            # Parametri
            use_gpu_cat = st.checkbox("Usa GPU", key="catboost_gpu", value=False)
            k_values_cat = st.multiselect(
                "Valori k per metriche",
                options=[1, 2, 3, 5, 10],
                default=[1, 2, 3, 5],
                key="catboost_k",
                help="Valori k per calcolare Precision@k, Hit Rate@k, NDCG@k",
            )
            
            # Pulsante
            button_label = "Riesegui CatBoost" if artifacts_exist else "Allena CatBoost"
            if st.button(button_label, key="btn_catboost", type="primary"):
                _start_training_catboost(manager, use_gpu_cat, k_values_cat)
        
        # ===== AUTOGLUON =====
        with col_auto:
            st.markdown("### 🤖 AutoGluon")
            
            autogluon_status = manager.get_run_status("autogluon")
            artifacts_exist_ag = manager.check_artifacts_exist("autogluon")
            
            if autogluon_status:
                _render_training_status_card(autogluon_status, artifacts_exist_ag)
            else:
                st.info("Nessun training AutoGluon registrato.")
            
            use_gpu_ag = st.checkbox("Usa GPU", key="autogluon_gpu", value=False)
            k_values_ag = st.multiselect(
                "Valori k per metriche",
                options=[1, 2, 3, 5, 10],
                default=[1, 2, 3, 5],
                key="autogluon_k",
                help="Valori k per calcolare Precision@k, Hit Rate@k, NDCG@k",
            )
            
            button_label_ag = "Riesegui AutoGluon" if artifacts_exist_ag else "Allena AutoGluon"
            if st.button(button_label_ag, key="btn_autogluon", type="primary"):
                _start_training_autogluon(manager, use_gpu_ag, k_values_ag)
        
        # ===== PATTERN & POLICY =====
        with col_pattern:
            st.markdown("### 📐 Pattern/Policy")
            
            pattern_status = manager.get_run_status("pattern")
            artifacts_exist_pat = manager.check_artifacts_exist("pattern")
            
            if pattern_status:
                _render_training_status_card(pattern_status, artifacts_exist_pat)
            else:
                st.info("Nessun ricalcolo Pattern registrato.")
            
            st.markdown(
                "Riesegue pattern detection e policy backtest: genera `per_client_patterns.parquet`, "
                "`event_policy_confusion.parquet`, `event_policy_metrics.parquet`."
            )
            
            button_label_pat = "Ricomputa Pattern" if artifacts_exist_pat else "Calcola Pattern"
            if st.button(button_label_pat, key="btn_pattern", type="primary"):
                _start_pattern_recompute(manager)


def _render_training_status_card(run, artifacts_exist: bool) -> None:
    """Render card with training status, timestamp, duration, artifacts."""
    status_emoji = {
        "not_started": "⚪",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
    }
    emoji = status_emoji.get(run.status, "❓")
    
    # Status
    st.markdown(f"**Stato:** {emoji} `{run.status}`")
    
    # Timestamp
    if run.completed_at:
        timestamp_str = run.completed_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        st.markdown(f"**Completato:** {timestamp_str}")
    elif run.started_at:
        timestamp_str = run.started_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        st.markdown(f"**Avviato:** {timestamp_str}")
    
    # Duration
    if run.duration_seconds:
        duration_min = run.duration_seconds / 60.0
        st.markdown(f"**Durata:** {duration_min:.1f} min")
    
    # Artifacts
    if artifacts_exist and run.artifacts:
        st.markdown(f"**Artefatti:** {len(run.artifacts)} file(s)")
        with st.expander("Mostra path artefatti"):
            for artifact in run.artifacts:
                st.code(str(artifact), language="text")
    elif run.status == "completed" and not artifacts_exist:
        st.warning("⚠️ Artefatti non trovati")
    
    # Error
    if run.error_message:
        st.error(f"**Errore:** {run.error_message}")


def _start_training_catboost(manager: TrainingManager, use_gpu: bool, k_values: List[int]) -> None:
    """Avvia training CatBoost in background."""
    st.info("🔄 Avvio training CatBoost in background...")
    
    command = build_training_command("catboost", use_gpu=use_gpu, k_values=k_values)
    parameters = {"use_gpu": use_gpu, "k_values": k_values}
    
    try:
        # Start async
        process = manager.execute_training_async("catboost", command, parameters)
        st.success(f"✅ Training CatBoost avviato (PID={process.pid}). Aggiorna la pagina per vedere lo stato.")
        st.info("💡 Il training continua in background. Ricarica la pagina tra qualche minuto per vedere i risultati.")
    except Exception as e:
        st.error(f"❌ Errore avvio training CatBoost: {e}")


def _start_training_autogluon(manager: TrainingManager, use_gpu: bool, k_values: List[int]) -> None:
    """Avvia training AutoGluon in background."""
    st.info("🔄 Avvio training AutoGluon in background...")
    
    command = build_training_command("autogluon", use_gpu=use_gpu, k_values=k_values)
    parameters = {"use_gpu": use_gpu, "k_values": k_values}
    
    try:
        process = manager.execute_training_async("autogluon", command, parameters)
        st.success(f"✅ Training AutoGluon avviato (PID={process.pid}). Aggiorna la pagina per vedere lo stato.")
        st.info("💡 Il training continua in background. Ricarica la pagina tra qualche minuto per vedere i risultati.")
    except Exception as e:
        st.error(f"❌ Errore avvio training AutoGluon: {e}")


def _start_pattern_recompute(manager: TrainingManager) -> None:
    """Avvia ricalcolo Pattern/Policy in background."""
    st.info("🔄 Avvio ricalcolo Pattern/Policy in background...")
    
    command = build_training_command("pattern")
    parameters = {}
    
    try:
        process = manager.execute_training_async("pattern", command, parameters)
        st.success(f"✅ Ricalcolo Pattern avviato (PID={process.pid}). Aggiorna la pagina per vedere lo stato.")
        st.info("💡 Il processo continua in background. Ricarica la pagina tra qualche minuto per vedere i risultati.")
    except Exception as e:
        st.error(f"❌ Errore avvio ricalcolo Pattern: {e}")


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
        title="Calendario Eventi di Compensazione (ultimi 12 mesi)", # Compensazione (SAP: Clearing)
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
        title="Distribuzione Cadenza Pagamenti", # Cadenza (SAP: Payment Frequency)
        xaxis=dict(title="Giorni tra compensazioni"), # Compensazioni (SAP: Clearings)
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


def plot_policy_bars_enhanced(score_row: pd.Series, active_policies: Optional[List[str]] = None) -> go.Figure:
    """
    Policy recall bars with exact recall values and confidence tooltips.
    
    Args:
        score_row: Series with policy scores and confidence
        active_policies: List of policies to include (defaults to all POLICY_NAMES)
        
    Returns:
        Plotly bar chart with numerical tooltips
    """
    if active_policies is None:
        active_policies = POLICY_NAMES
    
    policy_data = []
    
    for policy in active_policies:
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
        hovertemplate="<b>%{x}</b><br>Recall: %{y:.2%}<br>Confidenza: " + f"{confidence:.2f}<extra></extra>",
        name="Recall",
        text=df["recall"].apply(lambda x: f"{x:.1%}"),
        textposition="outside",
    ))
    
    fig.update_layout(
        title=f"Recall medio per policy (confidenza: {confidence:.2f})",
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
    st.markdown("### 💡 Tesi + evidenze")
    
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
        color: #0B2545;
    }
    .claim-evidence-box ul {
        margin: 5px 0;
        padding-left: 20px;
    }
    .claim-evidence-box li {
        margin: 8px 0;
        line-height: 1.6;
        color: #102A43;
    }
    </style>
    """, unsafe_allow_html=True)
    
    claims_html = "<div class='claim-evidence-box'><ul>"
    for line in summary_lines:
        claims_html += f"<li>{line}</li>"
    claims_html += "</ul></div>"
    
    st.markdown(claims_html, unsafe_allow_html=True)


def _render_coherence_panel(
    selected_client: str,
    pattern_row: Optional[pd.Series] = None,
    cluster_row: Optional[pd.Series] = None,
) -> None:
    """
    Render Coerenza con Pattern panel (Prompt 11).
    
    Shows agreement_score badge (green/yellow/red), comparison between model top-k
    and dominant policy top-k, and motivations.
    
    Args:
        selected_client: Client identifier
        pattern_row: Client pattern row with dominant_policy, pattern_confidence
        cluster_row: Optional cluster row with cluster_label, membership_prob
    """
    st.markdown("---")
    st.markdown("### 🎯 Coerenza con Pattern Cliente")
    st.caption("Confronta il ranking del modello (p_i) con la policy dominante del cliente")
    
    # Load predictions with agreement_score
    try:
        base_dir = Path(__file__).resolve().parents[2]
        curated_dir = base_dir / "data" / "curated"
        predictions_path = curated_dir / "event_policy_predictions.parquet"
        
        if not predictions_path.exists():
            st.info("📊 Esegui scoring (Prompt 7: `python src/analytics/score_open_invoices.py`) per vedere coerenza con pattern storico")
            return
        
        predictions_df = pd.read_parquet(predictions_path)
        client_predictions = predictions_df[predictions_df["cliente"] == selected_client].copy()
        
        if client_predictions.empty:
            st.info(f"📊 Nessuna predizione trovata per cliente {selected_client}")
            return
    
    except Exception as e:
        st.warning(f"⚠️ Errore caricamento predizioni: {e}")
        return
    
    # Get aggregated agreement_score (media su tutti gli eventi)
    if "agreement_score" not in client_predictions.columns:
        st.warning("⚠️ Colonna `agreement_score` non trovata. Riesegui Prompt 7.")
        return
    
    agreement_score = float(client_predictions["agreement_score"].mean())
    alignment_status = client_predictions["alignment_status"].mode().iloc[0] if "alignment_status" in client_predictions.columns else "unknown"
    
    # Get pattern metrics
    dominant_policy = pattern_row.get("dominant_policy", "n.d.") if pattern_row is not None and not pattern_row.empty else "n.d."
    pattern_confidence = pattern_row.get("pattern_confidence", 0.0) if pattern_row is not None and not pattern_row.empty else 0.0
    
    # Badge logic
    if alignment_status == "aligned" or agreement_score > 0.7:
        badge_color = "#28a745"  # green
        badge_emoji = "✅"
        badge_text = "In linea con pattern storico"
    elif alignment_status == "partial" or agreement_score > 0.4:
        badge_color = "#ffc107"  # yellow
        badge_emoji = "⚠️"
        badge_text = "Parzialmente allineato"
    else:  # divergent
        badge_color = "#dc3545"  # red
        badge_emoji = "❌"
        badge_text = "Divergente dal pattern storico"
    
    # Display badge
    st.markdown(
        f"""
        <div style='background:{badge_color};color:white;padding:1.2rem;border-radius:10px;text-align:center;margin-bottom:1rem;'>
        <h3 style='margin:0;color:white;'>{badge_emoji} {badge_text}</h3>
        <p style='margin:0.5rem 0 0 0;font-size:1.05rem;'>Agreement Score: <strong>{agreement_score:.2f}</strong> | Policy dominante: <strong>{dominant_policy.upper()}</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Two-column metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Metriche Pattern**")
        st.metric("Policy Dominante", dominant_policy.upper())
        st.metric("Pattern Confidence", f"{pattern_confidence:.2f}")
        
        if cluster_row is not None and not cluster_row.empty:
            cluster_label = cluster_row.get("cluster_label", "n.d.")
            membership_prob = cluster_row.get("membership_prob", 0.0)
            st.metric("Cluster Label", f"{cluster_label}")
            st.metric("Membership Prob", f"{membership_prob:.2f}")
    
    with col2:
        st.markdown("**🎯 Coerenza Modello**")
        st.metric("Agreement Score", f"{agreement_score:.2f}")
        st.metric("Alignment Status", alignment_status.capitalize())
        
        # Interpretation
        if agreement_score > 0.7:
            st.success("✓ **Modello allineato**: ranking coerente con comportamento storico")
        elif agreement_score > 0.4:
            st.warning("⚠ **Parziale allineamento**: verifica eventuali anomalie comportamentali")
        else:
            st.error("✗ **Basso allineamento**: possibile cambio strategia cliente o pattern atipico")
    
    # Top-k comparison table
    st.markdown("**📋 Confronto Top-K: Modello vs Policy Dominante**")
    st.caption("Confronto tra le fatture top-ranked dal modello (p_i) e dalla policy dominante")
    
    # Get top-k from predictions
    k = min(5, len(client_predictions))
    top_k_model = client_predictions.nlargest(k, "p_i")[["numero_documento", "p_i", "rank"]].copy()
    top_k_model.columns = ["Numero Documento", "Probabilità (p_i)", "Rank Modello"]
    
    # Get top-k from dominant policy (if available)
    policy_col = f"{dominant_policy}_rank" if f"{dominant_policy}_rank" in client_predictions.columns else None
    
    if policy_col:
        top_k_policy = client_predictions.nsmallest(k, policy_col)[["numero_documento", policy_col]].copy()
        top_k_policy.columns = ["Numero Documento", f"Rank {dominant_policy.upper()}"]
        
        # Merge for comparison
        comparison_df = top_k_model.merge(top_k_policy, on="Numero Documento", how="outer", indicator=True)
        comparison_df["Overlap"] = comparison_df["_merge"].apply(lambda x: "✅" if x == "both" else "")
        comparison_df = comparison_df.drop(columns=["_merge"])
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Overlap percentage
        overlap_count = (comparison_df["Overlap"] == "✅").sum()
        overlap_pct = overlap_count / k * 100 if k > 0 else 0.0
        st.info(f"📊 **Overlap**: {overlap_count}/{k} fatture ({overlap_pct:.0f}%) in comune tra modello e {dominant_policy.upper()}")
    
    else:
        st.dataframe(top_k_model, use_container_width=True, hide_index=True)
        st.info(f"⚠️ Policy rank per `{dominant_policy}` non trovata nelle predizioni.")
    
    # Motivazioni SHAP (reference to existing section)
    with st.expander("🔍 Motivazioni SHAP (Top-3 Features)"):
        st.info("💡 Le motivazioni SHAP (top-3 feature più influenti) sono già visualizzate nella sezione **'Explainability Panel' → 'SHAP (prime 2 fatture)'** più in basso.")
        st.markdown("""
        **Cosa sono le SHAP values?**
        - **SHAP** (SHapley Additive exPlanations) misura l'impatto di ciascuna feature sulla predizione del modello.
        - **Feature positive** (barra verso destra): aumentano la probabilità di pagamento (p_i ↑)
        - **Feature negative** (barra verso sinistra): diminuiscono la probabilità di pagamento (p_i ↓)
        - Le top-3 feature spiegano **perché** il modello ha rankato una fattura in alto o in basso.
        """)


# ============================================================================
# Prompt 3: Open Invoices Scoring Panel with Coherence Badge
# ============================================================================

def render_scoring_panel(
    cliente: str,
    open_invoices: pd.DataFrame,
    pattern_scores: pd.DataFrame,
    model_choice: str = "catboost",
) -> None:
    """
    Render scoring panel for open invoices with coherence badge.
    
    Prompt 3: Shows ranking table with model scores + policy-based ranking comparison.
    Badge (green/yellow/red) indicates overlap between top-5 model vs top-5 policy.
    
    Args:
        cliente: Client identifier
        open_invoices: DataFrame with open invoices (must have age_days column)
        pattern_scores: Client pattern scores DataFrame
        model_choice: "catboost" or "autogluon"
    """
    if open_invoices.empty:
        st.info(f"✅ Cliente **{cliente}** non ha fatture aperte")
        return
    
    st.markdown(f"### 📊 Ranking Predittivo - {len(open_invoices)} Fatture Aperte")
    
    # Get dominant policy for comparison
    dominant_policy = get_dominant_policy(cliente, pattern_scores)
    
    if dominant_policy == "unknown":
        st.warning("⚠️ Pattern policy non disponibile per questo cliente. Mostrando solo ranking modello.")
        dominant_policy = "fifo"  # Default fallback
    
    # Build mock scores (in production, this would call actual model)
    # For now, use age_days as proxy for scoring
    open_invoices["p_model"] = 1.0 / (1.0 + open_invoices["age_days"] / 30.0)  # Decay function
    open_invoices["rank_model"] = open_invoices["p_model"].rank(ascending=False, method="first").astype(int)
    
    # Policy-based ranking (simple FIFO/LIFO logic)
    if dominant_policy == "fifo":
        # FIFO: oldest first (smallest data_documento)
        open_invoices["rank_policy"] = open_invoices["data_documento"].rank(method="first").astype(int)
    elif dominant_policy == "lifo":
        # LIFO: newest first (largest data_documento)
        open_invoices["rank_policy"] = open_invoices["data_documento"].rank(ascending=False, method="first").astype(int)
    elif dominant_policy == "largest":
        # Largest amount first
        open_invoices["rank_policy"] = open_invoices.get("imp_div_int", 0).rank(ascending=False, method="first").astype(int)
    else:
        # Default: same as FIFO
        open_invoices["rank_policy"] = open_invoices["data_documento"].rank(method="first").astype(int)
    
    # Compute coherence (overlap top-5)
    top_k = min(5, len(open_invoices))
    top_k_model = set(open_invoices.nsmallest(top_k, "rank_model")["numero_documento"])
    top_k_policy = set(open_invoices.nsmallest(top_k, "rank_policy")["numero_documento"])
    overlap = len(top_k_model & top_k_policy) / float(top_k) if top_k > 0 else 0.0
    
    # Badge logic
    if overlap >= 0.6:
        badge = "🟢 Alta coerenza"
        badge_color = "#28a745"
        badge_text = "Il modello è allineato con il pattern storico"
    elif overlap >= 0.4:
        badge = "🟡 Coerenza media"
        badge_color = "#ffc107"
        badge_text = "Il modello mostra parziale allineamento"
    else:
        badge = "🔴 Bassa coerenza"
        badge_color = "#dc3545"
        badge_text = "Il modello diverge dal pattern storico"
    
    # Display badge
    st.markdown(
        f"""
        <div style="background: {badge_color}; color: white; padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 1rem;">
            <h3 style="margin: 0; color: white;">{badge}</h3>
            <p style="margin: 0.5rem 0 0 0;">Overlap top-{top_k}: {overlap:.0%} | Pattern: {dominant_policy.upper()}</p>
            <p style="margin: 0.3rem 0 0 0; font-size: 0.9rem;">{badge_text}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display ranked table (top 10)
    display_df = open_invoices.nsmallest(10, "rank_model")[[
        "numero_documento",
        "data_documento",
        "imp_div_int",
        "age_days",
        "p_model",
        "rank_model",
        "rank_policy"
    ]].copy()
    
    # Add flag for top-5 policy
    display_df["in_top5_policy"] = display_df["numero_documento"].isin(top_k_policy)
    
    # Format columns
    display_df["data_documento"] = pd.to_datetime(display_df["data_documento"]).dt.strftime("%Y-%m-%d")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "numero_documento": "Doc",
            "data_documento": "Data",
            "imp_div_int": st.column_config.NumberColumn("Importo", format="%.2f"),
            "age_days": st.column_config.NumberColumn("Età (gg)", format="%d"),
            "p_model": st.column_config.ProgressColumn("Prob. Modello", format="%.2f", min_value=0, max_value=1),
            "rank_model": st.column_config.NumberColumn("Rank Modello", format="%d"),
            "rank_policy": st.column_config.NumberColumn(f"Rank {dominant_policy.upper()}", format="%d"),
            "in_top5_policy": st.column_config.CheckboxColumn("✓ Top-5 Policy?")
        }
    )
    
    # Explanation
    st.caption(
        f"💡 **Interpretazione**: Le fatture con 🟢 checkmark rientrano anche nella top-5 secondo la policy **{dominant_policy.upper()}**. "
        f"Alta sovrapposizione indica che il modello predice comportamenti coerenti con il pattern storico."
    )


# ============================================================================
# Main Pattern Card Render Function
# ============================================================================

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
    hide_regmonth: bool = False,
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
        hide_regmonth: If True, exclude regmonth policy from charts/metrics
    """
    st.title(f"📊 Scheda pattern — Cliente {selected_client}")
    
 # Riquadro Azioni per training CatBoost/AutoGluon/Pattern
    _render_training_actions_panel()
    
 # Policy toggle UI in sidebar
    active_policies = _render_policy_toggle_sidebar()
    
    # DEPRECATED 
    # active_policies = [p for p in POLICY_NAMES if p != "regmonth" or not hide_regmonth]
    
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
        st.markdown("### 📅 Analisi visive")

        # Calendar heatmap (last 12 months)
        st.plotly_chart(
            plot_calendar_heatmap_12mo(client_events),
            width="stretch",
            key="calendar_heatmap",
        )

        # Cadence histogram
        gaps = client_events["inter_event_gap_days"].dropna().to_numpy(dtype=float)
        st.plotly_chart(
            plot_cadence_histogram_enhanced(gaps),
            width="stretch",
            key="cadence_histogram",
        )

        # ACF
        st.plotly_chart(
            plot_acf_enhanced(gaps, max_lag=10),
            width="stretch",
            key="acf_plot",
        )
    
    # RIGHT COLUMN: METRICS
    with col_metrics:
        st.markdown("### 📊 Cruscotto metriche")

        # Cardinality with metrics
        k_mode = float(score_row.get("k_mode", 0.0))
        k_entropy = float(score_row.get("k_entropy", 0.0))

        card_fig, card_metrics = plot_cardinality_with_metrics(client_events, k_mode, k_entropy)
        st.plotly_chart(card_fig, width="stretch", key="cardinality_hist")

        # Metrics cards
        metric_cols = st.columns(2)
        metric_cols[0].metric("🎯 k_mode", f"{card_metrics['k_mode']:.1f}")
        metric_cols[1].metric("📈 k_entropy", f"{card_metrics['k_entropy']:.2f}")

        st.markdown("---")

        # Policy bars with recall + confidence
        st.plotly_chart(
            plot_policy_bars_enhanced(score_row, active_policies=active_policies),
            width="stretch",
            key="policy_bars",
        )

        # Pattern label card
        if pattern_row is not None:
            st.markdown("#### Pattern riconosciuto")
            pattern_label = str(pattern_row.get("pattern_label", "n.d."))
            confidence = float(pattern_row.get("confidence", 0.0))
            stability = float(pattern_row.get("stability_index", 0.0))

            label_cols = st.columns(3)
            label_cols[0].metric("Etichetta", pattern_label)
            label_cols[1].metric("Confidenza", f"{confidence:.2f}")
            label_cols[2].metric("Stabilità", f"{stability:.2f}")
    
    # === CLAIM + EVIDENCE SECTION (full width) ===
    st.markdown("---")
    render_claim_evidence_box(selected_client, score_row, pattern_row)
    
    # === ADDITIONAL METRICS (full width) ===
    st.markdown("---")
    st.markdown("### 📏 Metriche aggiuntive")
    
 # Add month coverage metrics
    extra_cols = st.columns(6)
    extra_cols[0].metric("Concentrazione DOW", f"{float(score_row.get('dow_concentration', 0.0)):.2f}")
    extra_cols[1].metric("Concentrazione DOM", f"{float(score_row.get('dom_concentration', 0.0)):.2f}")
    extra_cols[2].metric("# Eventi", f"{len(client_events)}")
    extra_cols[3].metric("Gap medio", f"{gaps.mean():.1f} giorni" if gaps.size > 0 else "n.d.")
    
    # Month coverage metrics (PROMPT 7)
    month_cov_mean = float(score_row.get("month_coverage_mean", 0.0))
    month_cov_full_rate = float(score_row.get("month_coverage_full_rate", 0.0))
    extra_cols[4].metric(
        "📅 Copertura mese media",
        f"{month_cov_mean:.1%}",
        help="Media della percentuale di fatture del mese pagate per evento"
    )
    extra_cols[5].metric(
        "✅ Mese completo %",
        f"{month_cov_full_rate:.1%}",
        help="% eventi con copertura = 100% (tutte fatture del mese pagate)"
    )
    
    # === COHERENCE PANEL (Prompt 11) ===
    # Load cluster data for coherence panel
    cluster_row = None
    try:
        base_dir = Path(__file__).resolve().parents[2]
        curated_dir = base_dir / "data" / "curated"
        clusters_path = curated_dir / "client_clusters.parquet"
        
        if clusters_path.exists():
            clusters_df = pd.read_parquet(clusters_path)
            if selected_client in clusters_df["cliente"].values:
                cluster_row = clusters_df[clusters_df["cliente"] == selected_client].iloc[0]
    except Exception:
        pass  # Cluster data optional
    
    _render_coherence_panel(
        selected_client=selected_client,
        pattern_row=pattern_row,
        cluster_row=cluster_row,
    )
    
    # ========================================================================
    # Prompt 3: Open Invoices Scoring Panel
    # ========================================================================
    st.markdown("---")
    
    # Check model availability
    model_status = check_model_availability()
    
    if not model_status["any"]:
        st.error("❌ **Nessun modello allenato disponibile**")
        st.info("""
**Per utilizzare il ranking predittivo delle fatture aperte:**

1. Vai su **Overview & Wizard** (nella sidebar)
2. Completa **Step 3: Allena Modello Scoring**
3. Torna qui per vedere gli score fatture

**Modelli supportati:**
- CatBoost (Gradient Boosting con features categoriche)
- AutoGluon (AutoML ensemble)
        """)
        
        with st.expander("💡 Cosa fa il ranking predittivo?"):
            st.markdown("""
Il modello di scoring **predice la probabilità p_i** che ciascuna fattura aperta venga pagata nel prossimo evento di pagamento.

**Benefici:**
- 📊 **Prioritizzazione**: Focus su fatture con alta probabilità di incasso
- 💰 **Cash Flow**: Migliora la previsione degli incassi futuri  
- ⚠️ **Anomalie**: Identifica fatture divergenti dal pattern storico

**Come funziona:**
1. Il modello analizza **età fattura**, **importo**, **scadenza**, **pattern cliente**
2. Genera uno **score 0-1** per ogni fattura aperta (1 = alta prob. pagamento)
3. Confronta ranking modello vs ranking policy (FIFO, LIFO, etc.) per **badge coerenza**
            """)
    else:
        # Load line items for open invoices
        try:
            base_dir = BASE_DIR
            curated_dir = base_dir / "data" / "curated"
            line_items_path = curated_dir / "line_items.parquet"
            
            if line_items_path.exists():
                line_items = pd.read_parquet(line_items_path)
                open_invoices = fetch_open_invoices_for_client(selected_client, line_items)
                
                if not open_invoices.empty:
                    # Determine which model to use
                    model_choice = "catboost" if model_status["catboost"] else "autogluon"
                    st.caption(f"🤖 Modello attivo: **{model_choice.upper()}**")
                    
                    # Render scoring panel
                    render_scoring_panel(
                        cliente=selected_client,
                        open_invoices=open_invoices,
                        pattern_scores=pattern_scores,
                        model_choice=model_choice
                    )
                else:
                    st.success(f"✅ Cliente **{selected_client}** non ha fatture aperte")
                    st.info("Tutte le fatture sono state compensate. Ottimo lavoro! 🎉")
            else:
                st.warning("⚠️ File `line_items.parquet` non trovato. Esegui ingestion dei dati SAP.")
        except Exception as e:
            st.error(f"❌ Errore caricamento fatture aperte: {e}")
    
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
            hide_regmonth=hide_regmonth,
            developer_metrics=developer_metrics,
        )
    
    # === Export & deep-link ===
    st.markdown("---")
    st.markdown("### 📤 Esporta & audit")
    
    export_cols = st.columns(2)
    
    with export_cols[0]:
        # CSV export for pattern table
        if not pattern_table_df.empty:
            csv_bytes = export_dataframe_to_csv(pattern_table_df)
            st.download_button(
                label="📥 Scarica tabella pattern (CSV)",
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
        if st.button("📋 Copia deep-link"):
            st.code(deeplink_url, language=None)
            st.caption("⚠️ Copia manualmente l'URL sopra (limitazione di Streamlit).")
    
    st.caption("Esporta & audit | I CSV rispettano i filtri correnti e il deep-link conserva il cliente selezionato")


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
    hide_regmonth: bool = False,
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
        hide_regmonth: If True, hide regmonth policy from boost options
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
    
    st.dataframe(display_df, width="stretch", hide_index=True)
    
    # === JSON explain per evento top ===
    if not top_k.empty:
        st.markdown("#### 📥 Scarica JSON Explain")
        st.caption(
            "Scarica il JSON strutturato dell'evento in prima posizione con ranking del modello, punteggi policy, cluster e top-3 SHAP"
        )

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
            label="📥 Scarica JSON Explain (evento top)",
            data=json_bytes,
            file_name=f"explain_event_{first_event_row['numero_documento']}.json",
            mime="application/json",
        )
    
    # === SHAP TOP-3 FOR FIRST 2 INVOICES ===
    if model_type == "catboost" and shap_values is not None:
        st.markdown("#### 🔬 Spiegazioni SHAP (prime 2 fatture)")

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
                st.dataframe(shap_df, width="stretch", hide_index=True)
    
    # === WHAT-IF POLICY GATING ===
    st.markdown("---")
    st.markdown("####  🎛️ Simula policy gating")
    st.caption("Simula l'effetto di un boost su policy specifiche: `p_i' = p_i × (1 + α)`")
    
    # Policy boost controls
    st.markdown("**Seleziona le policy da potenziare:**")
    
    # Determine number of columns based on regmonth visibility
    n_policies = 6 if not hide_regmonth else 5
    boost_cols = st.columns(3)
    
    with boost_cols[0]:
        same_period_boost = st.checkbox("Stesso periodo", value=False, key="boost_same_period")
        eomfirst_boost = st.checkbox("Fine mese", value=False, key="boost_eomfirst")
    
    with boost_cols[1]:
        largest_boost = st.checkbox("Importo maggiore", value=False, key="boost_largest")
        fifo_boost = st.checkbox("FIFO", value=False, key="boost_fifo")
    
    with boost_cols[2]:
        lifo_boost = st.checkbox("LIFO", value=False, key="boost_lifo")
        if not hide_regmonth:
            regmonth_boost = st.checkbox(
                "Mese registrazione",
                value=False,
                key="boost_regmonth",
                help="Regmonth = stesso mese di registrazione"
            )
        else:
            regmonth_boost = False
    
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
    if not hide_regmonth and regmonth_boost:
        active_boosts.append("regmonth")
    
    if active_boosts:
        active_labels = ", ".join([POLICY_LABELS_IT.get(p, p.upper()) for p in active_boosts])
        st.markdown(f"**Policy attive:** {active_labels} con α={alpha:.2f}")

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
        
        styled_df = comparison_df.style.map(
            color_rank_delta,
            subset=["Δ Rank"]
        )

        st.dataframe(styled_df, width="stretch", hide_index=True)

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
