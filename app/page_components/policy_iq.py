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
 # Policy Toggle (imported from pattern_card_v2)
# ============================================================================

 # Policy deterministiche disponibili (same as pattern_card_v2)
DETERMINISTIC_POLICIES: List[str] = [
    "fifo",
    "lifo",
    "same_period",
    "largest",
    "smallest",
    "eomfirst",
    "quarter",
    "seqblock",
    "typeprio",
]

 # Policy labels (imported from pattern_card_v2 for consistency)
from app.page_components.pattern_card_v2 import POLICY_LABELS_IT


# ============================================================================
 # Policy Explanations (Business-Friendly)
# ============================================================================

POLICY_EXPLANATIONS: Dict[str, Dict[str, str]] = {
    "fifo": {
        "title": "🔹 FIFO — Prima Emessa, Prima Pagata",
        "simple": "Il cliente paga sempre le fatture **più vecchie** per prime.",
        "example": "**Esempio**: Cliente ha fatture del 10/01, 15/01, 20/01. Il 25/01 paga? → Sceglie quella del **10/01** (la più vecchia).",
        "why": "**Quando usarla**: Clienti ordinati che seguono rigidamente l'ordine cronologico di ricezione.",
        "technical": "**Formula**: `min(Data Documento)` tra fatture aperte al momento del pagamento."
    },
    "lifo": {
        "title": "🔸 LIFO — Ultima Emessa, Prima Pagata",
        "simple": "Il cliente paga sempre le fatture **più recenti** per prime.",
        "example": "**Esempio**: Cliente ha fatture del 10/01, 15/01, 20/01. Il 25/01 paga? → Sceglie quella del **20/01** (la più recente).",
        "why": "**Quando usarla**: Clienti che vogliono evitare ritardi sugli ordini recenti, o che ricevono solleciti recenti.",
        "technical": "**Formula**: `max(Data Documento)` tra fatture aperte al momento del pagamento."
    },
    "largest": {
        "title": "💰 Largest — Importo Maggiore Prima",
        "simple": "Il cliente paga sempre le fatture con **importo più alto** per prime.",
        "example": "**Esempio**: Cliente ha fatture da €100, €500, €200. Paga? → Sceglie quella da **€500**.",
        "why": "**Quando usarla**: Clienti che vogliono ridurre l'esposizione totale velocemente, o che negoziano sconti su grandi importi.",
        "technical": "**Formula**: `max(Imp. div. int.)` tra fatture aperte al momento del pagamento."
    },
    "smallest": {
        "title": "🪙 Smallest — Importo Minore Prima",
        "simple": "Il cliente paga sempre le fatture con **importo più basso** per prime.",
        "example": "**Esempio**: Cliente ha fatture da €100, €500, €200. Paga? → Sceglie quella da **€100**.",
        "why": "**Quando usarla**: Clienti che vogliono chiudere tante posizioni piccole per semplificare la contabilità.",
        "technical": "**Formula**: `min(Imp. div. int.)` tra fatture aperte al momento del pagamento."
    },
    "same_period": {
        "title": "📅 Same Period — Stesso Periodo Contabile",
        "simple": "Il cliente paga fatture dello **stesso mese contabile** (Periodo Contabile = mese pagamento).",
        "example": "**Esempio**: Cliente paga il 15/03. Ha fatture con Periodo 01, 02, 03. → Sceglie quelle con Periodo **03** (marzo).",
        "why": "**Quando usarla**: Clienti con processi contabili rigidi che chiudono bilanci mensili.",
        "technical": "**Formula**: `Periodo Contabile = MONTH(Data Compensazione)` tra fatture aperte."
    },
    "eomfirst": {
        "title": "📆 EOM First — Fine Mese Prima",
        "simple": "Il cliente paga fatture emesse a **fine mese** per prime (ultimi 5 giorni).",
        "example": "**Esempio**: Cliente ha fatture del 10/01, 28/01, 15/02. Paga? → Sceglie quella del **28/01** (fine mese).",
        "why": "**Quando usarla**: Clienti che ricevono solleciti batch a fine mese o che chiudono rendiconti periodici.",
        "technical": "**Formula**: `day(Data Documento) >= 26` tra fatture aperte al momento del pagamento."
    },
    "quarter": {
        "title": "📊 Quarter — Stesso Trimestre",
        "simple": "Il cliente paga fatture dello **stesso trimestre** per prime (Q1, Q2, Q3, Q4).",
        "example": "**Esempio**: Cliente paga ad aprile (Q2). Ha fatture Q1 (gen-mar) e Q2 (apr-giu). → Sceglie quelle **Q2**.",
        "why": "**Quando usarla**: Clienti corporate con budget trimestrali o rendiconti quarterly.",
        "technical": "**Formula**: `QUARTER(Data Documento) = QUARTER(Data Compensazione)` tra fatture aperte."
    },
    "seqblock": {
        "title": "🔢 Sequential Block — Blocco Sequenziale",
        "simple": "Il cliente paga fatture in **blocchi sequenziali** (es. tutte del 2023, poi tutte del 2024).",
        "example": "**Esempio**: Cliente ha fatture 2022, 2023, 2024. Prima paga **tutte** del 2022, poi passa al 2023.",
        "why": "**Quando usarla**: Clienti che chiudono periodi contabili in modo sequenziale (es. audit annuali).",
        "technical": "**Formula**: `min(Esercizio)` tra fatture aperte, poi procede al successivo solo quando il precedente è vuoto."
    },
    "typeprio": {
        "title": "📋 Type Priority — Priorità per Tipo Documento",
        "simple": "Il cliente paga prima certi **tipi di documento** (es. fatture prima di note credito).",
        "example": "**Esempio**: Cliente ha fatture (tipo 'RV') e note credito (tipo 'NC'). Paga? → Sceglie prima **tutte le RV**.",
        "why": "**Quando usarla**: Clienti con workflow contabili che richiedono chiusura fatture standard prima di gestire storni.",
        "technical": "**Formula**: `ORDER BY Tipo Documento (priorità fissa: RV > NC > altro)` tra fatture aperte."
    },
    "regmonth": {
        "title": "⚠️ RegMonth — Mese Registrazione = Mese Pagamento",
        "simple": "Il cliente paga fatture con **mese di registrazione = mese del pagamento** (indipendentemente dalla scadenza).",
        "example": "**Esempio**: Cliente paga il 15/03. Ha fatture registrate a feb (02) e mar (03). → Sceglie quelle registrate a **marzo (03)**.",
        "why": "**⚠️ Attenzione**: Questa regola NON verifica scadenze, può portare a pagamenti 'prematuri' se registrazione è recente ma scadenza lontana.",
        "technical": "**Formula**: `MONTH(Data Registrazione) = MONTH(Data Compensazione)` tra fatture aperte."
    }
}


def render_policy_explain_tile(policy_name: str) -> None:
    """Render expander tile with business-friendly explanation for a policy."""
    if policy_name not in POLICY_EXPLANATIONS:
        st.warning(f"Spiegazione non disponibile per policy '{policy_name}'")
        return
    
    expl = POLICY_EXPLANATIONS[policy_name]
    
    with st.expander(expl["title"], expanded=False):
        st.markdown(f"**📘 Cosa fa**: {expl['simple']}")
        st.markdown(expl["example"])
        st.markdown(expl["why"])
        
        # Technical details in collapsed section
        with st.expander("🔬 Dettagli Tecnici (per sviluppatori)", expanded=False):
            st.code(expl["technical"], language="python")


def _render_policy_toggle_sidebar_iq() -> List[str]:
    """Render sidebar UI per toggle policy deterministiche (Policy IQ page).
    
    Returns:
        List[str]: Lista policy attive (include regmonth se checkbox è ON)
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Filtro Policy Deterministiche")
    st.sidebar.markdown(
        "Seleziona quali policy mostrare nella confusion matrix e nei grafici. "
        "Il toggle agisce **solo sulla visualizzazione** (dati a disco non modificati)."
    )
    
    # Multiselect per policy standard (default: tutte attive)
    selected_policies = st.sidebar.multiselect(
        "Policy attive",
        options=DETERMINISTIC_POLICIES,
        default=DETERMINISTIC_POLICIES,
        format_func=lambda p: POLICY_LABELS_IT.get(p, p),
        help="Scegli quali policy deterministiche mostrare nei grafici e nelle tabelle",
        key="policy_iq_toggle_multiselect",
    )
    
    # Checkbox separata per RegMonth (default: OFF)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Policy Avanzata:**")
    
    show_regmonth = st.sidebar.checkbox(
        POLICY_LABELS_IT.get("regmonth", "RegMonth"),
        value=False,
        key="policy_iq_toggle_regmonth",
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
    if st.sidebar.checkbox("🔍 Debug policy attive (IQ)", value=False, key="debug_policy_iq_toggle"):
        st.sidebar.json({"active_policies": active_policies, "count": len(active_policies)})
    
    return active_policies


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
        hide_regmonth: DEPRECATED (PROMPT 6) - replaced by PROMPT 7 policy toggle
    """
    st.title("🎯 Policy IQ — Come pagano i tuoi clienti?")
    
    st.markdown(
        """
        Capire quale **regola di pagamento** usa ogni cliente aiuta a predire quali fatture pagheranno prima:
        - **🎯 Matrice Accuratezza**: Confronta regola reale vs. quella predetta dal modello
        - **📈 Miglioramento vs FIFO**: Quanto guadagni usando il modello invece di FIFO standard
        - **💡 Spiegazioni Semplici**: Cosa significa ogni regola di pagamento con esempi concreti
        """
    )
    
 # Policy toggle UI in sidebar
    active_policies = _render_policy_toggle_sidebar_iq()
    
    # Load data
    with st.spinner("Caricamento dati delle policy..."):
        confusion_df = load_policy_confusion_matrix(data_dir)
        metrics_df = load_policy_metrics(data_dir)
        policy_scores_df = load_per_event_policy_scores(data_dir)
        predictions_df = load_event_policy_predictions(data_dir)
    
 # Filter data based on active_policies (replaces PROMPT 6 hide_regmonth)
    if not confusion_df.empty and "true_policy" in confusion_df.columns:
        confusion_df = confusion_df[
            confusion_df["true_policy"].isin(active_policies)
        ].copy()
        if "pred_policy" in confusion_df.columns:
            confusion_df = confusion_df[
                confusion_df["pred_policy"].isin(active_policies)
            ].copy()
    
    if not metrics_df.empty and "policy" in metrics_df.columns:
        metrics_df = metrics_df[metrics_df["policy"].isin(active_policies)].copy()
    
    if not policy_scores_df.empty:
        # Filter policy score columns based on active_policies
        policy_score_cols = [c for c in policy_scores_df.columns if c.startswith("policy_")]
        active_policy_cols = [
            c for c in policy_score_cols
            if any(p in c for p in active_policies)
        ]
        keep_cols = [c for c in policy_scores_df.columns if not c.startswith("policy_")] + active_policy_cols
        policy_scores_df = policy_scores_df[keep_cols].copy()
    
    if not predictions_df.empty and "pred_policy" in predictions_df.columns:
        predictions_df = predictions_df[
            predictions_df["pred_policy"].isin(active_policies)
        ].copy()
    
    if confusion_df.empty and metrics_df.empty and policy_scores_df.empty:
        st.error("Dati di policy non disponibili. Esegui prima la pipeline di analytics.")
        return
    
    # ========================================================================
    # Section 1: Confusion Matrix (Business-Friendly)
    # ========================================================================
    st.header("🎯 1. Matrice Accuratezza Predizioni")
    st.markdown(
        """
        **Cosa stai vedendo?** Ogni riga mostra quante volte il modello ha indovinato la regola di pagamento reale del cliente.
        
        - **Diagonale verde scuro**: Predizioni corrette ✅
        - **Caselle fuori diagonale**: Confusione tra regole simili (es. FIFO vs LIFO)
        
        💡 **Obiettivo**: Diagonale >70% significa modello affidabile.
        """
    )
    
    fig_confusion = plot_policy_confusion_matrix(confusion_df)
    st.plotly_chart(fig_confusion, use_container_width=True)
    
    # Accuracy metrics with business-friendly labels
    if not metrics_df.empty:
        global_metrics = metrics_df[metrics_df["segment_type"] == "global"]
        if not global_metrics.empty:
            row = global_metrics.iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if "top1_accuracy" in row:
                    accuracy_val = row['top1_accuracy']
                    accuracy_emoji = "🎯" if accuracy_val >= 0.7 else ("⚠️" if accuracy_val >= 0.5 else "❌")
                    st.metric(
                        f"{accuracy_emoji} Precisione Predizioni",
                        f"{accuracy_val:.1%}",
                        help="% di eventi in cui il modello ha indovinato la regola esatta (target: >70%)"
                    )
            with col2:
                if "top2_hit_rate" in row:
                    hitrate_val = row['top2_hit_rate']
                    st.metric(
                        "📊 Copertura Top-2",
                        f"{hitrate_val:.1%}",
                        help="% di eventi in cui la regola vera è tra le prime 2 predette (utile per ambiguità FIFO/LIFO)"
                    )
            with col3:
                if "n_events" in row:
                    st.metric(
                        "📋 Eventi Analizzati",
                        f"{int(row['n_events']):,}",
                        help="Numero totale di pagamenti usati per calcolare la matrice"
                    )
    
    # Export confusion matrix CSV
    if not confusion_df.empty:
        st.markdown("---")
        csv_bytes = export_dataframe_to_csv(confusion_df)
        st.download_button(
            label="📥 Scarica Matrice CSV",
            data=csv_bytes,
            file_name="matrice_accuratezza_policy.csv",
            mime="text/csv",
            help="Esporta la matrice con percentuali per analisi in Excel"
        )
    
    st.divider()
    
    # ========================================================================
    # Section 2: Uplift Analysis (Business-Friendly + Chart)
    # ========================================================================
    st.header("📈 2. Miglioramento rispetto a FIFO Standard")
    st.markdown(
        """
        **Domanda chiave**: Conviene usare il modello o basta applicare FIFO a tutti i clienti?
        
        - **Baseline FIFO**: Predici sempre "Prima Emessa, Prima Pagata" (strategia standard)
        - **Modello Policy IQ**: Usa la regola specifica per cliente (FIFO, LIFO, Largest, etc.)
        - **Uplift %**: Quanto migliora il recall@k con Policy IQ vs FIFO
        
        💡 **Interpretazione**:
        - **+10%**: Policy IQ predice 10% in più di pagamenti corretti
        - **0%**: Nessun vantaggio, FIFO funziona già bene
        - **-5%**: Policy IQ peggiora (raro, indica overfitting)
        """
    )
    
    # Render uplift table
    render_uplift_table(metrics_df)
    
    # Add bar chart for visual comparison
    if not metrics_df.empty:
        st.markdown("#### 📊 Confronto Visivo: Baseline vs Policy IQ")
        
        # Prepare data for chart
        chart_df = metrics_df.copy()
        if "segment_type" in chart_df.columns and "baseline_recall" in chart_df.columns and "predicted_recall" in chart_df.columns:
            # Focus on meaningful segments (global, k buckets)
            chart_df = chart_df[chart_df["segment_type"].isin(["global", "k_bucket"])].copy()
            
            if not chart_df.empty:
                import plotly.express as px
                
                # Reshape for grouped bar chart
                chart_long = chart_df.melt(
                    id_vars=["segment_value"],
                    value_vars=["baseline_recall", "predicted_recall"],
                    var_name="Strategia",
                    value_name="Recall@k"
                )
                
                # Rename for clarity
                chart_long["Strategia"] = chart_long["Strategia"].map({
                    "baseline_recall": "FIFO Standard",
                    "predicted_recall": "Policy IQ"
                })
                
                fig_uplift = px.bar(
                    chart_long,
                    x="segment_value",
                    y="Recall@k",
                    color="Strategia",
                    barmode="group",
                    color_discrete_map={"FIFO Standard": "#94a3b8", "Policy IQ": "#10b981"},
                    labels={"segment_value": "Segmento", "Recall@k": "Recall@k (%)"},
                    title="Recall@k per Segmento: FIFO vs Policy IQ"
                )
                
                fig_uplift.update_yaxis(tickformat=".1%")
                fig_uplift.update_layout(height=400)
                
                st.plotly_chart(fig_uplift, use_container_width=True)
                
                # Key insight
                global_row = metrics_df[metrics_df["segment_type"] == "global"]
                if not global_row.empty and "uplift_relative" in global_row.columns:
                    uplift_val = global_row.iloc[0]["uplift_relative"]
                    if uplift_val > 5:
                        st.success(f"✅ **Uplift globale: +{uplift_val:.1f}%** — Policy IQ migliora significativamente il recall!")
                    elif uplift_val > 0:
                        st.info(f"💡 **Uplift globale: +{uplift_val:.1f}%** — Policy IQ migliora leggermente, potrebbe non giustificare la complessità.")
                    else:
                        st.warning(f"⚠️ **Uplift globale: {uplift_val:.1f}%** — Policy IQ non migliora rispetto a FIFO. Verifica overfitting.")
    
    # Export uplift CSV
    if not metrics_df.empty:
        st.markdown("---")
        csv_bytes_uplift = export_dataframe_to_csv(metrics_df)
        st.download_button(
            label="📥 Scarica Uplift CSV",
            data=csv_bytes_uplift,
            file_name="policy_iq_uplift_metriche.csv",
            mime="text/csv",
            help="Esporta metriche uplift (baseline, predicted, delta) per segmento"
        )
    
    st.divider()
    
    # ========================================================================
    # Section 2.5: Cluster Breakdown (Prompt 12)
    # ========================================================================
    st.header("📊 2.5 Metriche per Cluster (Segmentazione)")
    st.markdown(
        """
        **Perché segmentare?** Clienti diversi seguono regole diverse. I cluster HDBSCAN aggregano clienti simili.
        
        - **Dominant Policy**: Regola più frequente nel cluster
        - **Recall@k Medio**: Quanto bene il modello predice i pagamenti per quel cluster
        - **Uplift vs FIFO**: Miglioramento del cluster rispetto al baseline FIFO
        
        💡 **Utilità**: Scopri se certi cluster hanno pattern più prevedibili (→ targeting campagne DSO).
        """
    )
    
    # Load cluster data
    cluster_summary_path = data_dir / "curated" / "cluster_summary.parquet"
    client_clusters_path = data_dir / "curated" / "client_clusters.parquet"
    
    if cluster_summary_path.exists() and client_clusters_path.exists():
        cluster_summary_df = pd.read_parquet(cluster_summary_path)
        client_clusters_df = pd.read_parquet(client_clusters_path)
        
        if not cluster_summary_df.empty:
            # Display summary table
            st.markdown("**📋 Riepilogo Cluster**")
            
            # Extract relevant columns
            display_cols = []
            if "cluster_label" in cluster_summary_df.columns:
                display_cols.append("cluster_label")
            if "n_clienti" in cluster_summary_df.columns:
                display_cols.append("n_clienti")
            if "dominant_policy" in cluster_summary_df.columns:
                display_cols.append("dominant_policy")
            if "pattern_confidence" in cluster_summary_df.columns:
                display_cols.append("pattern_confidence")
            if "k_mode" in cluster_summary_df.columns:
                display_cols.append("k_mode")
            if "periodicity" in cluster_summary_df.columns:
                display_cols.append("periodicity")
            
            if display_cols:
                cluster_display_df = cluster_summary_df[display_cols].copy()
                
                # Rename for business clarity
                cluster_display_df.columns = cluster_display_df.columns.str.replace("_", " ").str.title()
                
                st.dataframe(cluster_display_df, use_container_width=True, hide_index=True)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🏷️ Cluster Trovati", f"{len(cluster_summary_df)}")
                with col2:
                    total_clients = client_clusters_df["cliente"].nunique() if "cliente" in client_clusters_df.columns else 0
                    st.metric("👥 Clienti Totali", f"{total_clients:,}")
                with col3:
                    if "n_clienti" in cluster_summary_df.columns:
                        avg_cluster_size = cluster_summary_df["n_clienti"].mean()
                        st.metric("📊 Dimensione Media Cluster", f"{avg_cluster_size:.0f}")
                
                # Bar chart: Dominant policy distribution per cluster
                if "dominant_policy" in cluster_summary_df.columns and "cluster_label" in cluster_summary_df.columns:
                    st.markdown("**📊 Policy Dominante per Cluster**")
                    
                    import plotly.express as px
                    
                    policy_cluster_df = cluster_summary_df[["cluster_label", "dominant_policy"]].copy()
                    policy_cluster_df["count"] = 1  # For aggregation
                    
                    fig_cluster_policy = px.bar(
                        policy_cluster_df,
                        x="cluster_label",
                        y="count",
                        color="dominant_policy",
                        labels={"cluster_label": "Cluster", "count": "Count", "dominant_policy": "Policy Dominante"},
                        title="Distribuzione Policy Dominante per Cluster",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    
                    fig_cluster_policy.update_layout(height=350)
                    st.plotly_chart(fig_cluster_policy, use_container_width=True)
                
                # Export cluster summary CSV
                st.markdown("---")
                csv_bytes_cluster = export_dataframe_to_csv(cluster_summary_df)
                st.download_button(
                    label="📥 Scarica Cluster Summary CSV",
                    data=csv_bytes_cluster,
                    file_name="cluster_summary_policy_iq.csv",
                    mime="text/csv",
                    help="Esporta riepilogo cluster con policy dominanti, k_mode, periodicity"
                )
            else:
                st.warning("⚠️ Colonne richieste (cluster_label, dominant_policy) non trovate in cluster_summary.parquet")
        else:
            st.info("📊 Cluster summary vuoto. Esegui HDBSCAN clustering (Prompt 4-5) per generare segmentazione.")
    else:
        st.info("📊 Cluster data non disponibili. Esegui clustering (Prompt 4: `python src/analytics/run_hdbscan.py`) e pattern detection (Prompt 5: `python src/analytics/make_pattern_table.py`) per vedere breakdown per cluster.")
    
    st.divider()
    
    # ========================================================================
    # Section 3: Explain Tiles (Business-Friendly Policy Explanations)
    # ========================================================================
    st.header("💡 3. Cosa Significa Ogni Regola di Pagamento?")
    st.markdown(
        """
        **Guida rapida**: Espandi una regola per capire quando e perché un cliente la usa.
        
        Ogni spiegazione include:
        - 📘 **Descrizione semplice**: Cosa fa la regola in parole chiare
        - 🧪 **Esempio concreto**: Caso d'uso con fatture reali
        - 💼 **Quando usarla**: Scenari tipici per questo comportamento
        - 🔬 **Dettagli tecnici**: Formula esatta (per sviluppatori)
        """
    )
    
    # Tab view for better organization
    explain_tabs = st.tabs([
        "📋 Tutte le Regole",
        "🔍 Per Cliente Specifico"
    ])
    
    # Tab 1: All policies explained
    with explain_tabs[0]:
        st.markdown("### 📚 Dizionario Completo delle Regole")
        
        # Group policies by category
        st.markdown("#### 🕐 Regole Temporali")
        temporal_policies = ["fifo", "lifo", "same_period", "eomfirst", "quarter", "seqblock", "regmonth"]
        for policy in temporal_policies:
            if policy in active_policies:
                render_policy_explain_tile(policy)
        
        st.markdown("#### 💰 Regole Basate su Importo")
        amount_policies = ["largest", "smallest"]
        for policy in amount_policies:
            if policy in active_policies:
                render_policy_explain_tile(policy)
        
        st.markdown("#### 📋 Regole Basate su Tipo Documento")
        type_policies = ["typeprio"]
        for policy in type_policies:
            if policy in active_policies:
                render_policy_explain_tile(policy)
    
    # Tab 2: Client-specific dominant policies
    with explain_tabs[1]:
        st.markdown("### 🎯 Policy Dominanti per Cliente")
        st.markdown("Seleziona un cliente per vedere le sue 3 regole più frequenti con esempi dai suoi dati reali.")
        
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
                    st.markdown(f"#### Top 3 regole per **{selected_client_tile}**:")
                    
                    for policy_info in dominant_policies:
                        render_policy_tile(policy_info, predictions_df)
                else:
                    st.info(f"Nessun dato policy disponibile per il cliente {selected_client_tile}.")
            else:
                st.warning("Nessun cliente trovato nei punteggi policy.")
        else:
            st.warning("Punteggi policy non disponibili.")
    
    # ========================================================================
    # Footer: Deep-link for Audit
    # ========================================================================
    st.divider()
    st.markdown("### � Condividi Analisi")
    
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
    if st.button("📋 Genera Deep-Link", key="deeplink_policyiq"):
        st.code(deeplink_url, language=None)
        st.caption("⚠️ Copia manualmente l'URL sopra per condividere questa vista con colleghi.")
    
    # Footer
    st.divider()
    st.caption("🎯 Policy IQ | Dati: event_policy_metrics, per_event_policy_scores, event_policy_predictions")
    st.caption("📥 CSV export disponibili in ogni sezione | 🔗 Deep-link mantiene filtri attivi")
