"""Model Internals page combining Clusters, KPI Dashboard, and Policy IQ.

Step 4 refactoring: Consolidate analytical views into a single tabbed page.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st

from app.page_components.clusters import render_clusters_page
from app.page_components.kpi_dashboard import render_kpi_dashboard_page
from app.page_components.policy_iq import render_policy_iq

LOGGER = logging.getLogger(__name__)


def render_model_internals_page(
    tables: Dict[str, pd.DataFrame],
    clients: List[str],
    data_root_func: Callable[[], Path],
    developer_metrics: Dict[str, Any],
    hide_regmonth: bool = True,
) -> None:
    """Render Model Internals page with tabbed navigation.
    
    Tabs:
    - Cluster Map: HDBSCAN cluster visualization and client assignments
    - Model Metrics: KPI dashboard with ranking metrics
    - Policy Confusion: Policy classification analysis
    """
    st.title("🧠 Model Internals")
    st.markdown("""
    Analisi del modello e dei cluster. Usa i tab per navigare tra le viste.
    """)
    
    if "selected_client_internals" not in st.session_state:
        if clients:
            st.session_state["selected_client_internals"] = clients[0]
    
    st.sidebar.header("Selezione (Model Internals)")
    selected_client = st.sidebar.selectbox(
        "Cliente per Cluster Map",
        clients,
        key="selected_client_internals",
        help="Seleziona un cliente per evidenziarlo nella mappa dei cluster",
    )
    selected_client = str(selected_client)
    
    tab_cluster, tab_kpi, tab_policy = st.tabs([
        "📊 Cluster Map",
        "📈 Model Metrics", 
        "🎯 Policy Confusion"
    ])
    
    with tab_cluster:
        st.subheader("Cluster Map")
        st.markdown("""
        Visualizzazione dei cluster HDBSCAN e assegnazione clienti.
        Ogni cliente è assegnato a un cluster basato sul suo pattern di pagamento storico.
        """)
        render_clusters_page(
            tables=tables,
            selected_client=selected_client,
            clients=clients,
            data_root_func=data_root_func,
        )
    
    with tab_kpi:
        st.subheader("Model Metrics")
        st.markdown("""
        Metriche di ranking: Recall@K, Match Rate, AUC-PR per il modello CatBoost.
        """)
        render_kpi_dashboard_page(
            tables=tables,
            developer_metrics=developer_metrics,
            data_root_func=data_root_func,
        )
    
    with tab_policy:
        st.subheader("Policy Confusion")
        st.markdown("""
        Analisi della classificazione delle policy di pagamento.
        Confronto tra policy predette vs osservate per evento.
        """)
        render_policy_iq(
            data_dir=data_root_func(),
            hide_regmonth=hide_regmonth,
        )
