"""Client Debugger page for validating individual client predictions.

Step 5 implementation: Operational page for validation.

Sections:
- Section A: Historical Test (80/20 split, semaphore)
- Section B: Next Reconciliation (score current open invoices)
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

LOGGER = logging.getLogger(__name__)

HAS_CATBOOST = False
CatBoostClassifier = None
try:
    from catboost import CatBoostClassifier as _CatBoostClassifier
    CatBoostClassifier = _CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    LOGGER.warning("CatBoost not installed, scoring will be disabled")


def render_client_debugger_page(
    selected_client: str,
    events: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    data_root_func: Callable[[], Path],
) -> None:
    """Render Client Debugger page.
    
    Args:
        selected_client: Currently selected client ID
        events: Events dataframe
        tables: Loaded curated tables
        data_root_func: Function to get data root path
    """
    st.title("🔎 Client Debugger")
    st.markdown("""
    Pagina operativa per validare il singolo cliente.
    Verifica la prevedibilità storica e visualizza lo scoring delle fatture aperte.
    """)
    
    models_dir = data_root_func().parent / "models"
    
    st.info(f"**Cliente selezionato:** `{selected_client}`")
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("📜 Sezione A: Test Storico")
        st.markdown("""
        Prende tutto lo storico *chiuso* del cliente.
        Divide 80% (pattern) e 20% (test).
        """)
        _render_historical_test(selected_client, events, tables, models_dir)
    
    with col_b:
        st.subheader("📋 Sezione B: Prossima Riconciliazione")
        st.markdown("""
        Fatture *attualmente aperte* del cliente.
        Applica il modello CatBoost per predire quali verranno pagate.
        """)
        _render_next_reconciliation(selected_client, tables, models_dir)


def _render_historical_test(
    cliente: str,
    events: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    models_dir: Path,
) -> None:
    """Render historical test section with 80/20 split and semaphore."""
    
    if events.empty:
        st.warning("Nessun evento disponibile.")
        return
    
    client_events = events[events["cliente"].astype(str) == str(cliente)].copy()
    n_events = len(client_events)
    
    if n_events < 5:
        st.warning(f"⚠️ Solo {n_events} eventi per questo cliente. Servono almeno 5 eventi per un test affidabile.")
        st.metric("Eventi totali", n_events)
        return
    
    n_train = int(n_events * 0.8)
    n_test = n_events - n_train
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Eventi Train (80%)", n_train)
    with col2:
        st.metric("Eventi Test (20%)", n_test)
    
    accuracy = _compute_client_accuracy(cliente, tables, models_dir)
    
    if accuracy is None:
        st.info("🔄 Metriche non disponibili. Esegui la pipeline per calcolarle.")
        return
    
    st.markdown("### Valutazione Prevedibilità")
    
    if accuracy >= 0.80:
        st.success(f"🟢 **Prevedibile** - Match Rate: {accuracy:.1%}")
        st.markdown("""
        ✅ Il modello capisce come paga questo cliente.  
        Le previsioni sono affidabili.
        """)
    elif accuracy >= 0.50:
        st.warning(f"🟡 **Parzialmente Prevedibile** - Match Rate: {accuracy:.1%}")
        st.markdown("""
        ⚠️ Il modello ha una comprensione parziale.  
        Usare le previsioni con cautela.
        """)
    else:
        st.error(f"🔴 **Imprevedibile** - Match Rate: {accuracy:.1%}")
        st.markdown("""
        ❌ Il modello non riesce a predire questo cliente.  
        Comportamento di pagamento erratico o dati insufficienti.
        """)
    
    with st.expander("📊 Dettagli metriche", expanded=False):
        client_metrics = tables.get("client_model_metrics_catboost", pd.DataFrame())
        if not client_metrics.empty and "cliente" in client_metrics.columns:
            client_row = client_metrics[client_metrics["cliente"].astype(str) == str(cliente)]
            if not client_row.empty:
                st.dataframe(client_row.T, use_container_width=True)


def _render_next_reconciliation(
    cliente: str,
    tables: Dict[str, pd.DataFrame],
    models_dir: Path,
) -> None:
    """Render next reconciliation section with open invoices scoring."""
    
    line_items = tables.get("line_items", pd.DataFrame())
    if line_items.empty:
        features_df = tables.get("features", pd.DataFrame())
        if not features_df.empty:
            line_items = features_df
        else:
            st.warning("Nessun line item disponibile.")
            return
    
    client_items = line_items[line_items["cliente"].astype(str) == str(cliente)].copy()
    
    if client_items.empty:
        st.info("Nessun dato per questo cliente nei line items.")
        return
    
    if "data_pareggio" in client_items.columns:
        open_mask = client_items["data_pareggio"].isna()
        open_invoices = client_items[open_mask].copy()
    else:
        st.info("Colonna 'data_pareggio' non presente - non posso identificare fatture aperte.")
        open_invoices = pd.DataFrame()
    
    n_open = len(open_invoices)
    
    if n_open == 0:
        st.success("✅ Nessuna fattura aperta per questo cliente.")
        st.markdown("Tutte le fatture sono già state pareggiate.")
        return
    
    st.metric("📄 Fatture Aperte", n_open)
    
    scored_df = _score_open_invoices(cliente, open_invoices, tables, models_dir)
    
    if scored_df is None or scored_df.empty:
        st.warning("Modello non disponibile per lo scoring.")
        display_cols = ["numero_documento", "data_documento", "imp_div_int"]
        display_cols = [c for c in display_cols if c in open_invoices.columns]
        if display_cols:
            st.dataframe(open_invoices[display_cols].head(20), use_container_width=True)
        return
    
    st.markdown("### Previsioni Fatture Aperte")
    st.markdown("""
    **Score 0-5**: probabilità di pagamento nella prossima riconciliazione.
    - 🟢 4-5: Alta probabilità
    - 🟡 2-3: Media probabilità  
    - 🔴 0-1: Bassa probabilità
    """)
    
    st.dataframe(
        scored_df.style.background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=5),
        use_container_width=True,
        height=min(400, 35 * len(scored_df) + 50),
    )
    
    avg_score = scored_df["Score"].mean()
    st.metric("Score medio", f"{avg_score:.2f} / 5")


def _compute_client_accuracy(
    cliente: str,
    tables: Dict[str, pd.DataFrame],
    models_dir: Path,
) -> Optional[float]:
    """Compute accuracy from precomputed metrics.
    
    Returns match_rate or None if not available.
    """
    client_metrics = tables.get("client_model_metrics_catboost", pd.DataFrame())
    
    if client_metrics.empty:
        return None
    
    if "cliente" not in client_metrics.columns:
        return None
    
    client_row = client_metrics[client_metrics["cliente"].astype(str) == str(cliente)]
    
    if client_row.empty:
        return None
    
    for col in ["model_match_rate", "match_rate", "model_recall_at_k", "recall_at_k"]:
        if col in client_row.columns:
            val = client_row[col].iloc[0]
            if pd.notna(val):
                return float(val)
    
    return None


def _score_open_invoices(
    cliente: str,
    open_invoices: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    models_dir: Path,
) -> Optional[pd.DataFrame]:
    """Score open invoices using trained CatBoost model.
    
    Returns DataFrame with columns: Fattura, Importo, Data, Score
    """
    if not HAS_CATBOOST:
        return None
    
    catboost_path = models_dir / "catboost.cbm"
    calibrator_path = models_dir / "calibrator.pkl"
    feature_cols_path = models_dir / "feature_cols.txt"
    
    if not catboost_path.exists():
        LOGGER.warning(f"Model not found at {catboost_path}")
        return None
    
    result = open_invoices.copy()
    
    display_cols = []
    if "numero_documento" in result.columns:
        result["Fattura"] = result["numero_documento"].astype(str)
        display_cols.append("Fattura")
    
    if "imp_div_int" in result.columns:
        result["Importo"] = result["imp_div_int"].apply(
            lambda x: f"€ {x:,.2f}" if pd.notna(x) else "-"
        )
        display_cols.append("Importo")
    
    date_col = None
    for col in ["data_documento", "data_registrazione", "data_fattura"]:
        if col in result.columns:
            date_col = col
            break
    
    if date_col:
        result["Data"] = pd.to_datetime(result[date_col]).dt.strftime("%Y-%m-%d")
        display_cols.append("Data")
    
    if not HAS_CATBOOST or CatBoostClassifier is None:
        result["Score"] = 2.5
        result["Note"] = "CatBoost non installato"
        display_cols.append("Score")
        display_cols.append("Note")
        return result[display_cols].head(30)
    
    try:
        model = CatBoostClassifier()
        model.load_model(str(catboost_path))
        
        calibrator = None
        if calibrator_path.exists():
            with open(calibrator_path, "rb") as f:
                calibrator = pickle.load(f)
        
        feature_cols = None
        if feature_cols_path.exists():
            with open(feature_cols_path, "r") as f:
                feature_cols = [line.strip() for line in f if line.strip()]
        
        if feature_cols:
            available_cols = [c for c in feature_cols if c in open_invoices.columns]
            if len(available_cols) < len(feature_cols) * 0.5:
                LOGGER.warning(f"Only {len(available_cols)}/{len(feature_cols)} feature columns available")
                result["Score"] = 2.5
                result["Note"] = "Feature mancanti"
            else:
                X = open_invoices[available_cols].copy()
                for col in available_cols:
                    if X[col].dtype == "object":
                        X[col] = X[col].fillna("unknown")
                    else:
                        X[col] = X[col].fillna(0)
                
                probs = model.predict_proba(X)[:, 1]
                
                if calibrator is not None:
                    probs = calibrator.transform(probs.reshape(-1, 1)).flatten()
                
                result["Score"] = (probs * 5).round(1)
        else:
            result["Score"] = 2.5
            result["Note"] = "feature_cols.txt non trovato"
    
    except Exception as e:
        LOGGER.error(f"Error scoring invoices: {e}")
        result["Score"] = np.nan
        result["Note"] = f"Errore: {str(e)[:50]}"
    
    display_cols.append("Score")
    if "Note" in result.columns:
        display_cols.append("Note")
    
    return result[display_cols].head(30)
