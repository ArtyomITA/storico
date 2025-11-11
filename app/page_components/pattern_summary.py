"""Streamlit page: Pattern Summary with drill-down to Pattern Card.

This page provides a high-level overview of all client payment patterns:
- Table with pattern label, confidence, stability, k_mode, dominant policy recall
- Badge "Insicuro" for high entropy patterns
- Click-to-drill-down: select a row → navigate to Pattern Card with pre-selected client
- Macro-pattern distribution: pie chart and KPI cards
- Filters: pattern label, confidence range, entropy threshold
- Export: CSV download
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.sap_glossary import POLICY_LABELS_IT


LOGGER = logging.getLogger(__name__)

# Pattern label mapping (from make_pattern_table.py logic)
PATTERN_LABEL_MAP = {
    "P_k1": "k=1 (1 Partita)",
    "P_k2": "k=2 (2 Partite)",
    "P_kge3": "k≥3 (3+ Partite)",
    "pattern_k1": "k=1 (1 Partita)",
    "pattern_k2": "k=2 (2 Partite)",
    "pattern_k_ge_3": "k≥3 (3+ Partite)",
}

# Policy names in order of dominance check
POLICY_NAMES = [
    "fifo",
    "lifo",
    "same_period",
    "largest",
    "smallest",
    "eomfirst",
    "regmonth",
    "quarter",
    "seqblock",
    "typeprio",
]

# Entropy threshold for "Insicuro" badge
HIGH_ENTROPY_THRESHOLD = 0.6


def load_pattern_data(data_dir: Path) -> pd.DataFrame:
    """Load client_pattern_scores.parquet from data/curated.
    
    Args:
        data_dir: Path to data directory (contains curated/)
    
    Returns:
        DataFrame with pattern scores for all clients
    """
    curated_dir = data_dir / "curated"
    pattern_scores_path = curated_dir / "client_pattern_scores.parquet"
    
    if not pattern_scores_path.exists():
        raise FileNotFoundError(
            f"Pattern scores not found: {pattern_scores_path}. "
            "Run pattern detection first (Pattern Card → Azioni → Ricomputa Pattern)."
        )
    
    LOGGER.info(f"Loading pattern data from {pattern_scores_path}")
    df = pd.read_parquet(pattern_scores_path)
    
    # Ensure required columns exist
    required_cols = ["cliente", "k_mode", "k_entropy"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Pattern scores missing required columns: {missing}")
    
    return df


def infer_pattern_label(row: pd.Series) -> str:
    """Infer pattern label from probability columns (P_k1, P_k2, P_kge3).
    
    Args:
        row: Pattern scores row with P_k1, P_k2, P_kge3 columns
    
    Returns:
        Human-readable pattern label (e.g., "k=1 (1 Partita)")
    """
    # Try to use probability columns
    prob_cols = ["P_k1", "P_k2", "P_kge3"]
    available_probs = {col: row.get(col, 0.0) for col in prob_cols if col in row.index}
    
    if available_probs:
        max_prob_col = max(available_probs.items(), key=lambda x: x[1])[0]
        return PATTERN_LABEL_MAP.get(max_prob_col, max_prob_col)
    
    # Fallback: check if pattern_label column exists
    if "pattern_label" in row.index:
        raw_label = row["pattern_label"]
        return PATTERN_LABEL_MAP.get(raw_label, raw_label)
    
    # Final fallback: use k_mode
    k_mode = row.get("k_mode", 0)
    if k_mode <= 1:
        return "k=1 (1 Partita)"
    elif k_mode <= 2:
        return "k=2 (2 Partite)"
    else:
        return "k≥3 (3+ Partite)"


def compute_confidence(row: pd.Series) -> float:
    """Compute pattern confidence from probability columns.
    
    Args:
        row: Pattern scores row with P_k1, P_k2, P_kge3 columns
    
    Returns:
        Confidence score (0.0-1.0)
    """
    # Use existing confidence column if available
    if "confidence" in row.index and not pd.isna(row["confidence"]):
        return float(row["confidence"])
    
    # Otherwise compute from probability columns
    prob_cols = ["P_k1", "P_k2", "P_kge3"]
    available_probs = [row.get(col, 0.0) for col in prob_cols if col in row.index]
    
    if available_probs:
        return float(max(available_probs))
    
    # Fallback: use stability_index
    if "stability_index" in row.index:
        return float(row.get("stability_index", 0.5))
    
    return 0.5


def compute_stability(row: pd.Series) -> float:
    """Compute stability index (1 - k_entropy).
    
    Args:
        row: Pattern scores row with k_entropy column
    
    Returns:
        Stability score (0.0-1.0)
    """
    if "stability_index" in row.index and not pd.isna(row["stability_index"]):
        return float(row["stability_index"])
    
    k_entropy = row.get("k_entropy", 0.5)
    return float(max(0.0, 1.0 - k_entropy))


def get_dominant_policy(row: pd.Series) -> Tuple[str, float]:
    """Get dominant policy and its recall score.
    
    Args:
        row: Pattern scores row with policy score columns
    
    Returns:
        Tuple of (policy_name, recall_score)
    """
    policy_scores = {}
    for policy in POLICY_NAMES:
        score_col = f"{policy}_score"
        if score_col in row.index:
            policy_scores[policy] = float(row[score_col])
    
    if not policy_scores:
        return "unknown", 0.0
    
    dominant_policy = max(policy_scores.items(), key=lambda x: x[1])
    return dominant_policy[0], dominant_policy[1]


def is_uncertain(row: pd.Series) -> bool:
    """Check if pattern is uncertain (high entropy).
    
    Args:
        row: Pattern scores row with k_entropy column
    
    Returns:
        True if entropy > HIGH_ENTROPY_THRESHOLD
    """
    k_entropy = row.get("k_entropy", 0.0)
    return k_entropy > HIGH_ENTROPY_THRESHOLD


def build_summary_table(pattern_df: pd.DataFrame) -> pd.DataFrame:
    """Build summary table with pattern label, confidence, stability, etc.
    
    Args:
        pattern_df: Raw pattern scores dataframe
    
    Returns:
        Summary table with display-ready columns
    """
    summary = []
    
    for idx, row in pattern_df.iterrows():
        pattern_label = infer_pattern_label(row)
        confidence = compute_confidence(row)
        stability = compute_stability(row)
        k_mode = row.get("k_mode", 0)
        k_entropy = row.get("k_entropy", 0.0)
        dominant_policy, policy_recall = get_dominant_policy(row)
        uncertain = is_uncertain(row)
        
        summary.append({
            "Cliente": row["cliente"],
            "Pattern": pattern_label,
            "Confidence": confidence,
            "Stabilità": stability,
            "k_mode": k_mode,
            "k_entropy": k_entropy,
            "Policy Dominante": POLICY_LABELS_IT.get(dominant_policy, dominant_policy),
            "Policy Recall": policy_recall,
            "Insicuro": "⚠️ Sì" if uncertain else "✓ No",
        })
    
    return pd.DataFrame(summary)


def render_kpi_cards(summary_df: pd.DataFrame) -> None:
    """Render KPI cards with summary statistics.
    
    Args:
        summary_df: Summary table dataframe
    """
    col1, col2, col3, col4 = st.columns(4)
    
    total_clients = len(summary_df)
    uncertain_clients = len(summary_df[summary_df["Insicuro"] == "⚠️ Sì"])
    certain_clients = total_clients - uncertain_clients
    avg_confidence = summary_df["Confidence"].mean()
    
    with col1:
        st.metric("📊 Clienti Totali", total_clients)
    
    with col2:
        st.metric("✓ Pattern Sicuri", certain_clients, delta=None)
    
    with col3:
        st.metric("⚠️ Pattern Insicuri", uncertain_clients, delta=None)
    
    with col4:
        st.metric("🎯 Confidence Media", f"{avg_confidence:.2%}")


def render_pattern_distribution(summary_df: pd.DataFrame) -> None:
    """Render pie chart with pattern label distribution.
    
    Args:
        summary_df: Summary table dataframe
    """
    st.markdown("### 📊 Distribuzione Macro-Pattern")
    
    pattern_counts = summary_df["Pattern"].value_counts()
    
    if pattern_counts.empty:
        st.warning("Nessun dato disponibile per visualizzare la distribuzione pattern.")
        return
    
    # Pie chart
    fig = px.pie(
        values=pattern_counts.values,
        names=pattern_counts.index,
        title="Distribuzione Pattern per Cardinalità",
        hole=0.3,  # Donut chart
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Clienti: %{value}<br>Percentuale: %{percent}<extra></extra>',
    )
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Bar chart (alternative view)
    with st.expander("📊 Vista Barre"):
        fig_bar = px.bar(
            x=pattern_counts.index,
            y=pattern_counts.values,
            labels={"x": "Pattern", "y": "Numero Clienti"},
            title="Distribuzione Pattern (Barre)",
            color=pattern_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        
        fig_bar.update_traces(
            hovertemplate='<b>%{x}</b><br>Clienti: %{y}<extra></extra>',
        )
        
        fig_bar.update_layout(showlegend=False)
        
        st.plotly_chart(fig_bar, use_container_width=True)


def render_policy_distribution(summary_df: pd.DataFrame) -> None:
    """Render bar chart with dominant policy distribution.
    
    Args:
        summary_df: Summary table dataframe
    """
    st.markdown("### 🎯 Policy Dominanti")
    
    policy_counts = summary_df["Policy Dominante"].value_counts()
    
    if policy_counts.empty:
        st.warning("Nessun dato disponibile per visualizzare la distribuzione policy.")
        return
    
    fig = px.bar(
        x=policy_counts.index,
        y=policy_counts.values,
        labels={"x": "Policy", "y": "Numero Clienti"},
        title="Distribuzione Policy Dominanti",
        color=policy_counts.index,
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Clienti: %{y}<extra></extra>',
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        height=400,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_filters_sidebar(summary_df: pd.DataFrame) -> Dict[str, Any]:
    """Render sidebar filters for pattern summary.
    
    Args:
        summary_df: Summary table dataframe
    
    Returns:
        Dictionary with filter values
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Filtri Pattern")
    
    # Pattern label filter
    pattern_labels = sorted(summary_df["Pattern"].unique())
    selected_patterns = st.sidebar.multiselect(
        "Pattern (Cardinalità)",
        options=pattern_labels,
        default=pattern_labels,
        help="Filtra clienti per pattern label (k=1, k=2, k≥3)",
    )
    
    # Confidence range filter
    min_conf, max_conf = st.sidebar.slider(
        "Confidence Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.05,
        help="Filtra clienti per range di confidence",
    )
    
    # Entropy threshold filter
    entropy_threshold = st.sidebar.slider(
        "Max Entropia (k_entropy)",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="Filtra clienti con entropia <= soglia (entropia alta = pattern instabile)",
    )
    
    # Uncertain clients filter
    show_uncertain_only = st.sidebar.checkbox(
        "Solo Pattern Insicuri",
        value=False,
        help="Mostra solo clienti con entropia alta (⚠️ Insicuro)",
    )
    
    return {
        "selected_patterns": selected_patterns,
        "min_conf": min_conf,
        "max_conf": max_conf,
        "entropy_threshold": entropy_threshold,
        "show_uncertain_only": show_uncertain_only,
    }


def apply_filters(summary_df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply sidebar filters to summary table.
    
    Args:
        summary_df: Summary table dataframe
        filters: Dictionary with filter values from render_filters_sidebar()
    
    Returns:
        Filtered dataframe
    """
    filtered = summary_df.copy()
    
    # Pattern label filter
    if filters["selected_patterns"]:
        filtered = filtered[filtered["Pattern"].isin(filters["selected_patterns"])]
    
    # Confidence range filter
    filtered = filtered[
        (filtered["Confidence"] >= filters["min_conf"]) &
        (filtered["Confidence"] <= filters["max_conf"])
    ]
    
    # Entropy threshold filter
    filtered = filtered[filtered["k_entropy"] <= filters["entropy_threshold"]]
    
    # Uncertain clients filter
    if filters["show_uncertain_only"]:
        filtered = filtered[filtered["Insicuro"] == "⚠️ Sì"]
    
    return filtered


def render_summary_table(summary_df: pd.DataFrame) -> Optional[str]:
    """Render interactive summary table with drill-down capability.
    
    Args:
        summary_df: Summary table dataframe
    
    Returns:
        Selected cliente ID if user clicked a row, else None
    """
    st.markdown("### 📋 Tabella Riepilogativa Clienti")
    
    if summary_df.empty:
        st.warning("Nessun cliente trovato con i filtri selezionati.")
        return None
    
    # Display instructions
    st.info(
        "💡 **Drill-down:** Clicca su una riga per aprire la Pattern Card del cliente selezionato. "
        "Usa i filtri nella sidebar per restringere la vista."
    )
    
    # Display table with pagination
    page_size = 20
    total_rows = len(summary_df)
    total_pages = (total_rows - 1) // page_size + 1
    
    # Page selector
    col_page1, col_page2, col_page3 = st.columns([2, 1, 2])
    with col_page2:
        current_page = st.number_input(
            f"Pagina (1-{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key="pattern_summary_page",
        )
    
    # Slice dataframe for current page
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    page_df = summary_df.iloc[start_idx:end_idx].reset_index(drop=True)
    
    # Display table (hide k_entropy column in display, keep for filtering)
    display_cols = [
        "Cliente", "Pattern", "Confidence", "Stabilità", "k_mode",
        "Policy Dominante", "Policy Recall", "Insicuro"
    ]
    
    # Format confidence and recall as percentages
    display_df = page_df[display_cols].copy()
    display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.1%}")
    display_df["Stabilità"] = display_df["Stabilità"].apply(lambda x: f"{x:.1%}")
    display_df["Policy Recall"] = display_df["Policy Recall"].apply(lambda x: f"{x:.1%}")
    
    # Render table with selection
    st.dataframe(
        display_df,
        use_container_width=True,
        height=600,
    )
    
    # Search box for quick client lookup
    st.markdown("---")
    st.markdown("#### 🔎 Ricerca Rapida Cliente")
    search_query = st.text_input(
        "Cerca Cliente (ID o nome parziale)",
        placeholder="es. DEMO_C01",
        key="pattern_summary_search",
    )
    
    if search_query:
        matches = summary_df[
            summary_df["Cliente"].str.contains(search_query, case=False, na=False)
        ]
        
        if matches.empty:
            st.warning(f"Nessun cliente trovato con query: '{search_query}'")
        else:
            st.success(f"Trovati {len(matches)} clienti:")
            st.dataframe(matches[display_cols], use_container_width=True)
            
            # Drill-down buttons for search results
            st.markdown("**Seleziona cliente:**")
            cols = st.columns(min(len(matches), 5))
            for i, (idx, row) in enumerate(matches.head(5).iterrows()):
                with cols[i]:
                    if st.button(row["Cliente"], key=f"drill_{row['Cliente']}", use_container_width=True):
                        return row["Cliente"]
    
    return None


def render_export_button(summary_df: pd.DataFrame) -> None:
    """Render CSV export button.
    
    Args:
        summary_df: Summary table dataframe to export
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Esporta Dati")
    
    # Prepare CSV
    csv_data = summary_df.to_csv(index=False).encode("utf-8")
    
    st.sidebar.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name="pattern_summary.csv",
        mime="text/csv",
        help="Scarica tabella riepilogativa in formato CSV",
        use_container_width=True,
    )


def render_pattern_summary(data_dir: Path) -> None:
    """Main render function for Pattern Summary page.
    
    Args:
        data_dir: Path to data directory (contains curated/)
    """
    st.title("📊 Riepilogo Pattern Clienti")
    
    st.markdown(
        """
        Questa pagina fornisce una **vista riassuntiva** di tutti i pattern di pagamento clienti:
        - **Pattern Label**: Cardinalità modale (k=1, k=2, k≥3 partite per evento di pareggio)
        - **Confidence**: Probabilità massima per il pattern inferito
        - **Stabilità**: `1 - k_entropy` (alta stabilità = pattern consistente)
        - **Policy Dominante**: Policy deterministica con recall più alto
        - **Badge Insicuro**: ⚠️ se entropia > 0.6 (pattern instabile, difficile da predire)
        
        💡 **Drill-down**: Cerca un cliente nella tabella e clicca per aprire la sua Pattern Card dettagliata.
        """
    )
    
    try:
        # Load pattern data
        pattern_df = load_pattern_data(data_dir)
        
        # Build summary table
        summary_df = build_summary_table(pattern_df)
        
        # Render KPI cards
        render_kpi_cards(summary_df)
        
        st.markdown("---")
        
        # Render filters sidebar
        filters = render_filters_sidebar(summary_df)
        
        # Apply filters
        filtered_df = apply_filters(summary_df, filters)
        
        # Render visualizations (before table, to give overview)
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            render_pattern_distribution(filtered_df)
        
        with col_viz2:
            render_policy_distribution(filtered_df)
        
        st.markdown("---")
        
        # Render summary table with drill-down
        selected_cliente = render_summary_table(filtered_df)
        
        # Handle drill-down
        if selected_cliente:
            st.session_state["drill_cliente"] = selected_cliente
            st.session_state["active_page"] = "Pattern Card"  # Fixed: use proper page name
            st.session_state["drill_source_page"] = "Pattern Summary"  # Prompt 4: Track source
            st.success(f"✓ Cliente **{selected_cliente}** selezionato. Reindirizzamento a Pattern Card...")
            st.rerun()
        
        # Render export button
        render_export_button(filtered_df)
        
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.info(
            "💡 **Suggerimento:** Vai alla pagina **Pattern Card** e clicca su "
            "'Ricomputa Pattern' nel riquadro Azioni per generare i dati necessari."
        )
    except Exception as e:
        st.error(f"❌ Errore durante il caricamento dei pattern: {e}")
        LOGGER.exception("Error rendering pattern summary")
