"""KPI Dashboard page: ranking and calibration metrics with filters and download.

Rewritten for business readability - clear titles, technical details in tooltips.
Includes precision@k, hit_rate@k, nDCG@k, MRR, coverage@k, Brier, ECE, MCE, reliability curves.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils.export import generate_deeplink
from src.utils.sap_glossary import get_sap_term, get_sap_tooltip  


@st.cache_data(show_spinner=False, ttl=600)
def _load_parquet_cached(path: Path) -> pd.DataFrame:
    """Cached parquet loader for KPI data.
    
    st.cache_data with TTL=600s (10 min) for efficient reloading.
    Cache invalidates when file path changes (e.g., switching mock/real data).
    """
    return pd.read_parquet(path)


def _apply_filters(
    df: pd.DataFrame,
    selected_clients: List[str],
    cardinality_bucket: Optional[str],
    segment_type: Optional[str],
) -> pd.DataFrame:
    """Apply filters to KPI dataframe."""
    filtered = df.copy()

    # Filter by segment type if specified
    if segment_type and "aggregation_key" in filtered.columns:
        filtered = filtered[filtered["aggregation_key"] == segment_type]
    elif segment_type and "segment_type" in filtered.columns:
        filtered = filtered[filtered["segment_type"] == segment_type]

    # Filter by selected clients
    if selected_clients:
        if "aggregation_value" in filtered.columns:
            filtered = filtered[
                (filtered["aggregation_key"] == "cliente") & (filtered["aggregation_value"].isin(selected_clients))
                | (filtered["aggregation_key"] != "cliente")
            ]
        elif "segment_value" in filtered.columns:
            filtered = filtered[
                (filtered["segment_type"] == "cliente") & (filtered["segment_value"].isin(selected_clients))
                | (filtered["segment_type"] != "cliente")
            ]

    # Filter by cardinality bucket
    if cardinality_bucket and cardinality_bucket != "all":
        if "aggregation_value" in filtered.columns:
            filtered = filtered[
                (filtered["aggregation_key"] == "cardinality_bucket")
                & (filtered["aggregation_value"] == cardinality_bucket)
                | (filtered["aggregation_key"] != "cardinality_bucket")
            ]
        elif "segment_value" in filtered.columns:
            filtered = filtered[
                (filtered["segment_type"] == "cardinality_bucket") & (filtered["segment_value"] == cardinality_bucket)
                | (filtered["segment_type"] != "cardinality_bucket")
            ]

    return filtered


def _render_ranking_tab(kpi_df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """Render ranking metrics tab with business-friendly language.
    
    Clear titles for business users, technical details in tooltips.
    Shows precision@k, hit_rate@k, nDCG@k, MRR, coverage@k metrics.
    """
    st.markdown("## 🎯 Qualità delle Raccomandazioni")
    st.caption(
        "Misura quanto bene il modello **identifica le partite corrette** tra i documenti aperti. "
        "Metriche più alte → raccomandazioni più accurate."
    )

    # Global metrics
    global_row = filtered_df[filtered_df.get("aggregation_key", "") == "global"]
    if not global_row.empty:
        st.markdown("### 📊 Performance Complessiva")
        st.caption("Metriche aggregate su tutti gli eventi di pagamento.")
        
        # Top-K metrics in columns
        st.markdown("#### Accuratezza Top-K Raccomandazioni")
        
        k_values = [1, 2, 3, 5]
        tabs_k = st.tabs([f"Top {k}" for k in k_values])
        
        for idx, k in enumerate(k_values):
            with tabs_k[idx]:
                col1, col2, col3, col4 = st.columns(4)
                
                precision = global_row.iloc[0].get(f"precision@{k}", 0.0)
                hit_rate = global_row.iloc[0].get(f"hit_rate@{k}", 0.0)
                coverage = global_row.iloc[0].get(f"coverage@{k}", 0.0)
                ndcg = global_row.iloc[0].get(f"ndcg@{k}", 0.0)
                
                with col1:
                    st.metric(
                        f"✅ Precisione",
                        f"{precision:.1%}",
                        help=(
                            f"**Precisione@{k}**: Percentuale di raccomandazioni corrette tra le prime {k} suggerite.\n\n"
                            f"📘 **Interpretazione Business:** Se Precisione@{k} = {precision:.1%}, significa che in media "
                            f"**{precision:.1%} delle prime {k} raccomandazioni** sono effettivamente partite corrette.\n\n"
                            f"🔬 **Dettaglio Tecnico:** `Precision@{k} = (# partite corrette trovate nelle prime {k} posizioni) / {k}`"
                        ),
                    )
                
                with col2:
                    st.metric(
                        f"🎯 Tasso di Successo",
                        f"{hit_rate:.1%}",
                        help=(
                            f"**Hit Rate@{k}** (o Recall@{k}): Percentuale di **tutte le partite corrette** che sono state trovate "
                            f"nelle prime {k} raccomandazioni.\n\n"
                            f"📘 **Interpretazione Business:** Se Hit Rate@{k} = {hit_rate:.1%}, significa che il modello trova "
                            f"**{hit_rate:.1%} di tutte le partite rilevanti** guardando solo le prime {k} suggerimenti.\n\n"
                            f"🔬 **Dettaglio Tecnico:** `Hit_Rate@{k} = (# partite corrette trovate nelle prime {k}) / (# totale partite corrette nell'evento)`"
                        ),
                    )
                
                with col3:
                    st.metric(
                        f"📦 Copertura Eventi",
                        f"{coverage:.1%}",
                        help=(
                            f"**Coverage@{k}**: Percentuale di **eventi di pagamento** dove almeno 1 partita corretta è stata trovata "
                            f"nelle prime {k} raccomandazioni.\n\n"
                            f"📘 **Interpretazione Business:** Se Coverage@{k} = {coverage:.1%}, significa che per **{coverage:.1%} degli eventi** "
                            f"il modello ha suggerito almeno una partita corretta tra le prime {k} opzioni.\n\n"
                            f"🔬 **Dettaglio Tecnico:** `Coverage@{k} = (# eventi con almeno 1 hit nelle prime {k}) / (# totale eventi)`"
                        ),
                    )
                
                with col4:
                    st.metric(
                        f"📈 Rilevanza Posizionale",
                        f"{ndcg:.1%}",
                        help=(
                            f"**nDCG@{k}** (Normalized Discounted Cumulative Gain): Misura la qualità del **ranking** considerando "
                            f"che partite corrette in posizioni più alte sono più utili.\n\n"
                            f"📘 **Interpretazione Business:** Se nDCG@{k} = {ndcg:.1%}, significa che il modello ordina le raccomandazioni "
                            f"con un'efficacia del **{ndcg:.1%}** rispetto all'ordine perfetto. Valori > 80% indicano ranking eccellente.\n\n"
                            f"🔬 **Dettaglio Tecnico:** `nDCG@{k} = DCG@{k} / IDCG@{k}`, dove DCG applica un peso logaritmico decrescente "
                            f"alle posizioni (posizione 1 vale più di posizione 5)."
                        ),
                    )
        
        # MRR metric (standalone)
        st.markdown("---")
        st.markdown("#### 🥇 Posizione della Prima Partita Corretta")
        
        col_mrr1, col_mrr2, col_mrr3 = st.columns([2, 2, 3])
        
        with col_mrr1:
            mrr = global_row.iloc[0].get("mrr", 0.0)
            st.metric(
                "MRR (Rank Reciproco Medio)",
                f"{mrr:.3f}",
                help=(
                    "**Mean Reciprocal Rank (MRR)**: Media del reciproco della posizione della **prima partita corretta**.\n\n"
                    "📘 **Interpretazione Business:** MRR alto (> 0.5) significa che mediamente la prima partita corretta appare "
                    "nelle prime 2 posizioni delle raccomandazioni. MRR = 1.0 → prima posizione sempre corretta.\n\n"
                    "🔬 **Dettaglio Tecnico:** `MRR = mean(1 / rank_prima_partita_corretta)`. "
                    "Es: se la prima corretta è in posizione 1 → 1/1=1.0, posizione 2 → 1/2=0.5, posizione 3 → 1/3=0.33."
                ),
            )
        
        with col_mrr2:
            # Compute avg position from MRR (1/MRR)
            avg_position = 1.0 / mrr if mrr > 0 else 0.0
            st.metric(
                "Posizione Media Prima Hit",
                f"{avg_position:.1f}",
                help=(
                    "Posizione media della prima partita corretta nelle raccomandazioni.\n\n"
                    "Calcolato come `1 / MRR`. Più bassa è la posizione, meglio è (idealmente posizione 1)."
                ),
            )
        
        with col_mrr3:
            # Visual indicator
            if mrr >= 0.7:
                indicator = "🟢 Eccellente"
                desc = "Prima partita corretta quasi sempre nelle prime 2 posizioni"
            elif mrr >= 0.5:
                indicator = "🟡 Buono"
                desc = "Prima partita corretta mediamente nelle prime 2-3 posizioni"
            elif mrr >= 0.3:
                indicator = "🟠 Accettabile"
                desc = "Prima partita corretta mediamente nelle prime 3-4 posizioni"
            else:
                indicator = "🔴 Da Migliorare"
                desc = "Prima partita corretta spesso oltre la 4a posizione"
            
            st.markdown(f"**Valutazione:** {indicator}")
            st.caption(desc)

    # Cardinality buckets
    st.markdown("---")
    bucket_rows = filtered_df[filtered_df.get("aggregation_key", "") == "cardinality_bucket"]
    if not bucket_rows.empty:
        st.markdown("### 📊 Performance per Complessità Evento")
        st.caption(
            "Come varia l'accuratezza in base al **numero di partite** da identificare (cardinalità k). "
            "Eventi con k=1 (singola partita) sono più facili; k≥3 (multiple partite) più complessi."
        )
        
        # Format table with percentages
        display_bucket = bucket_rows.copy()
        for col in ["precision@3", "hit_rate@3", "coverage@3", "ndcg@3", "mrr"]:
            if col in display_bucket.columns:
                display_bucket[col] = display_bucket[col].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            display_bucket[
                [
                    "aggregation_value",
                    "n_events",
                    "precision@3",
                    "hit_rate@3",
                    "coverage@3",
                    "ndcg@3",
                    "mrr",
                ]
            ].rename(
                columns={
                    "aggregation_value": "Cardinalità (k)",
                    "n_events": "Numero Eventi",
                    "precision@3": "✅ Precisione@3",
                    "hit_rate@3": "🎯 Tasso Successo@3",
                    "coverage@3": "📦 Copertura@3",
                    "ndcg@3": "📈 Rilevanza@3",
                    "mrr": "🥇 MRR",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        
        # Visualization: bar chart per cardinality bucket
        st.markdown("#### 📊 Confronto Visuale per Cardinalità")
        
        metric_to_plot = st.selectbox(
            "Seleziona metrica da visualizzare",
            ["precision@3", "hit_rate@3", "coverage@3", "ndcg@3", "mrr"],
            format_func=lambda x: {
                "precision@3": "✅ Precisione@3",
                "hit_rate@3": "🎯 Tasso Successo@3",
                "coverage@3": "📦 Copertura@3",
                "ndcg@3": "📈 Rilevanza@3",
                "mrr": "🥇 MRR",
            }.get(x, x),
            key="metric_selector_bucket",
        )
        
        fig_bucket = px.bar(
            bucket_rows,
            x="aggregation_value",
            y=metric_to_plot,
            labels={
                "aggregation_value": "Cardinalità (k)",
                metric_to_plot: {
                    "precision@3": "Precisione@3",
                    "hit_rate@3": "Tasso Successo@3",
                    "coverage@3": "Copertura@3",
                    "ndcg@3": "Rilevanza@3",
                    "mrr": "MRR",
                }.get(metric_to_plot, metric_to_plot),
            },
            color="aggregation_value",
            color_discrete_sequence=px.colors.qualitative.Set2,
            text=metric_to_plot,
        )
        
        fig_bucket.update_traces(
            texttemplate='%{text:.1%}',
            textposition='outside',
        )
        
        fig_bucket.update_layout(
            showlegend=False,
            yaxis_tickformat='.0%',
            yaxis_range=[0, max(bucket_rows[metric_to_plot]) * 1.15],
            height=400,
        )
        
        st.plotly_chart(fig_bucket, use_container_width=True)

    # Per-client metrics
    st.markdown("---")
    client_rows = filtered_df[filtered_df.get("aggregation_key", "") == "cliente"]
    if not client_rows.empty:
        st.markdown("### 🏢 Performance per Cliente")
        st.caption("Dettaglio accuratezza per singolo cliente. Utile per identificare clienti con pattern predittivi difficili.")

        # Search and filter controls
        col_search, col_sort = st.columns([2, 1])
        
        with col_search:
            search_term = st.text_input(
                "🔍 Cerca cliente (ID o nome parziale)",
                "",
                placeholder="es. DEMO_C01",
                key="client_search_ranking",
            )
        
        display_df = client_rows.copy()
        if search_term:
            display_df = display_df[
                display_df["aggregation_value"].astype(str).str.contains(search_term, case=False, na=False)
            ]
        
        with col_sort:
            sort_col = st.selectbox(
                "Ordina per",
                ["hit_rate@3", "precision@3", "ndcg@3", "mrr", "n_events"],
                format_func=lambda x: {
                    "hit_rate@3": "🎯 Tasso Successo",
                    "precision@3": "✅ Precisione",
                    "ndcg@3": "📈 Rilevanza",
                    "mrr": "🥇 MRR",
                    "n_events": "Numero Eventi",
                }.get(x, x),
                key="sort_ranking",
            )
        
        sort_order = st.radio(
            "Ordine",
            ["Decrescente (migliori prima)", "Crescente (peggiori prima)"],
            horizontal=True,
            key="sort_order_ranking",
        )
        
        display_df = display_df.sort_values(
            by=sort_col,
            ascending=("Crescente" in sort_order),
        )
        
        # Format percentages
        display_table = display_df.copy()
        for col in ["precision@1", "precision@3", "hit_rate@3", "coverage@3", "ndcg@3", "mrr"]:
            if col in display_table.columns:
                display_table[col] = display_table[col].apply(lambda x: f"{x:.1%}")
        
        # Pagination
        page_size = 20
        total_rows = len(display_table)
        total_pages = (total_rows - 1) // page_size + 1 if total_rows > 0 else 1
        
        col_page1, col_page2, col_page3 = st.columns([2, 1, 2])
        with col_page2:
            current_page = st.number_input(
                f"Pagina (1-{total_pages})",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                key="page_ranking_client",
            )
        
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        page_table = display_table.iloc[start_idx:end_idx]
        
        st.dataframe(
            page_table[
                [
                    "aggregation_value",
                    "n_events",
                    "precision@1",
                    "precision@3",
                    "hit_rate@3",
                    "coverage@3",
                    "ndcg@3",
                    "mrr",
                ]
            ].rename(
                columns={
                    "aggregation_value": "Cliente",
                    "n_events": "Eventi",
                    "precision@1": "✅ P@1",
                    "precision@3": "✅ P@3",
                    "hit_rate@3": "🎯 Successo@3",
                    "coverage@3": "📦 Copertura@3",
                    "ndcg@3": "📈 Rilevanza@3",
                    "mrr": "🥇 MRR",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=600,
        )
        
        st.caption(f"Mostrando {start_idx + 1}-{end_idx} di {total_rows} clienti")


def _render_calibration_tab(
    calibration_summary: pd.DataFrame, calibration_bins: pd.DataFrame, filtered_summary: pd.DataFrame
) -> None:
    """Render calibration diagnostics tab with business-friendly language.
    
    Explains calibration in business terms - how trustworthy are model confidence scores?
    Shows Brier, ECE, MCE metrics and reliability curves.
    """
    st.markdown("## 🎯 Affidabilità delle Probabilità Predette")
    st.caption(
        "**Calibrazione** misura se le probabilità del modello sono **affidabili**: "
        "quando il modello dice '80% di confidenza', quella previsione è davvero corretta 80% delle volte? "
        "Calibrazione perfetta → si può **fidarsi dei punteggi** per prendere decisioni."
    )

    # Metrics cards
    st.markdown("### 📊 Metriche di Calibrazione")
    st.caption(
        "Valori più **bassi** sono migliori. Idealmente < 0.05 (5%) indica calibrazione eccellente."
    )
    
    models = filtered_summary["model"].unique()

    for model in models:
        model_summary = filtered_summary[
            (filtered_summary["model"] == model) & (filtered_summary["segment_type"] == "global")
        ]
        if not model_summary.empty:
            st.markdown(f"#### 🤖 Modello: **{model.upper()}**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            brier = model_summary.iloc[0].get("brier_score", 0.0)
            ece = model_summary.iloc[0].get("ece", 0.0)
            mce = model_summary.iloc[0].get("mce", 0.0)
            
            with col1:
                # Brier Score indicator
                if brier <= 0.1:
                    brier_status = "🟢 Eccellente"
                elif brier <= 0.2:
                    brier_status = "🟡 Buono"
                else:
                    brier_status = "🔴 Da Migliorare"
                
                st.metric(
                    "📐 Errore Quadratico Medio",
                    f"{brier:.4f}",
                    help=(
                        "**Brier Score**: Misura l'errore quadratico medio tra le probabilità predette e i risultati effettivi.\n\n"
                        "📘 **Interpretazione Business:** Brier = 0.0 → previsioni perfette. Brier < 0.1 → eccellente. "
                        f"Il modello {model} ha Brier = {brier:.4f}, quindi in media le sue probabilità differiscono "
                        f"dalla realtà di circa **{np.sqrt(brier):.1%}** (radice quadrata del Brier).\n\n"
                        "🔬 **Dettaglio Tecnico:** `Brier = mean((p_predetto - outcome_vero)²)` dove outcome ∈ {0, 1}"
                    ),
                )
                st.caption(brier_status)
            
            with col2:
                # ECE indicator
                if ece <= 0.05:
                    ece_status = "🟢 Eccellente"
                elif ece <= 0.10:
                    ece_status = "🟡 Buono"
                else:
                    ece_status = "🔴 Da Migliorare"
                
                st.metric(
                    "📊 Gap di Calibrazione Medio",
                    f"{ece:.1%}",
                    help=(
                        "**Expected Calibration Error (ECE)**: Media ponderata del gap tra confidence predetta e accuracy osservata, "
                        "calcolata su bin di probabilità.\n\n"
                        "📘 **Interpretazione Business:** Se il modello dice '70% di confidenza', quanto spesso ha davvero ragione? "
                        f"ECE = {ece:.1%} significa che in media c'è un gap di **{ece:.1%}** tra probabilità predette e accuracy reale. "
                        f"Es: modello dice 70%, ma ha ragione solo {0.70 - ece:.1%} delle volte.\n\n"
                        "🔬 **Dettaglio Tecnico:** `ECE = Σ (n_bin/n_tot) * |accuracy_bin - confidence_bin|` su 10 bin"
                    ),
                )
                st.caption(ece_status)
            
            with col3:
                # MCE indicator
                if mce <= 0.10:
                    mce_status = "🟢 Eccellente"
                elif mce <= 0.20:
                    mce_status = "🟡 Accettabile"
                else:
                    mce_status = "🔴 Critico"
                
                st.metric(
                    "⚠️ Gap Massimo (Worst Case)",
                    f"{mce:.1%}",
                    help=(
                        "**Maximum Calibration Error (MCE)**: Il **peggior** gap di calibrazione osservato in qualsiasi bin di probabilità.\n\n"
                        "📘 **Interpretazione Business:** Identifica il 'punto debole' del modello. "
                        f"MCE = {mce:.1%} significa che nel caso peggiore, le probabilità del modello si discostano dalla realtà "
                        f"di **{mce:.1%}**. Es: modello dice 90%, ma ha ragione solo {0.90 - mce:.1%} delle volte (o viceversa).\n\n"
                        "🔬 **Dettaglio Tecnico:** `MCE = max_bin |accuracy_bin - confidence_bin|`"
                    ),
                )
                st.caption(mce_status)
            
            with col4:
                # Overall calibration quality
                if ece <= 0.05 and mce <= 0.10:
                    overall = "🟢 Ben Calibrato"
                    overall_desc = "Probabilità affidabili, puoi fidarti dei punteggi"
                elif ece <= 0.10 and mce <= 0.20:
                    overall = "🟡 Calibrazione Accettabile"
                    overall_desc = "Probabilità utilizzabili, ma con cautela"
                else:
                    overall = "🔴 Calibrazione Scarsa"
                    overall_desc = "Probabilità inaffidabili, usa solo ranking relativo"
                
                st.markdown(f"**Valutazione Complessiva:**")
                st.markdown(f"**{overall}**")
                st.caption(overall_desc)

    # Reliability curves
    st.markdown("---")
    st.markdown("### 📈 Curve di Affidabilità (Reliability Diagrams)")
    st.caption(
        "**Grafico confidence vs accuracy**: La linea tratteggiata grigia rappresenta la **calibrazione perfetta** "
        "(se modello dice 60%, ha ragione 60% delle volte). Curve vicine alla diagonale → buona calibrazione. "
        "Curve sopra la diagonale → modello **sottostima** la confidenza (troppo prudente). "
        "Curve sotto → modello **sovrastima** (troppo sicuro di sé)."
    )

    # Filter controls
    col_model, col_segment = st.columns(2)
    
    with col_model:
        model_filter = st.multiselect(
            "Seleziona modelli da confrontare",
            calibration_bins["model"].unique().tolist(),
            default=calibration_bins["model"].unique().tolist(),
            key="model_filter_calibration",
            help="Confronta calibrazione tra CatBoost, AutoGluon, Champion, etc.",
        )
    
    with col_segment:
        segment_filter = st.selectbox(
            "Livello di aggregazione",
            ["global", "cardinality_bucket", "cliente"],
            format_func=lambda x: {
                "global": "📊 Globale (tutti gli eventi)",
                "cardinality_bucket": "📦 Per Cardinalità (k=1, k=2, etc.)",
                "cliente": "🏢 Per Cliente",
            }.get(x, x),
            key="segment_filter_calibration",
            help="Visualizza calibrazione aggregata o segmentata per complessità/cliente",
        )

    filtered_bins = calibration_bins[
        (calibration_bins["model"].isin(model_filter)) & (calibration_bins["segment_type"] == segment_filter)
    ]

    if not filtered_bins.empty:
        # Create Plotly line chart
        fig = go.Figure()

        # Add diagonal reference line (perfect calibration)
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="📐 Calibrazione Perfetta",
                line=dict(dash="dash", color="gray", width=2),
                showlegend=True,
                hoverinfo="skip",
            )
        )

        # Add reliability curves per model and segment
        colors = px.colors.qualitative.Plotly
        color_idx = 0
        
        for model in filtered_bins["model"].unique():
            for segment_value in filtered_bins["segment_value"].unique():
                segment_bins = filtered_bins[
                    (filtered_bins["model"] == model) & (filtered_bins["segment_value"] == segment_value)
                ].sort_values("pred_mean")

                if segment_bins.empty:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=segment_bins["pred_mean"],
                        y=segment_bins["obs_rate"],
                        mode="lines+markers",
                        name=f"🤖 {model.upper()} - {segment_value}",
                        line=dict(width=2.5, color=colors[color_idx % len(colors)]),
                        marker=dict(size=8, symbol="circle"),
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            "Confidence Predetta: %{x:.1%}<br>"
                            "Accuracy Osservata: %{y:.1%}<br>"
                            "Gap: %{customdata:.1%}<br>"
                            "<extra></extra>"
                        ),
                        customdata=np.abs(segment_bins["pred_mean"] - segment_bins["obs_rate"]),
                    )
                )
                color_idx += 1

        fig.update_layout(
            xaxis_title="💭 Confidence Predetta dal Modello (%)",
            yaxis_title="✅ Accuracy Osservata nella Realtà (%)",
            xaxis=dict(range=[0, 1], tickformat='.0%'),
            yaxis=dict(range=[0, 1], tickformat='.0%'),
            hovermode="closest",
            height=550,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
            ),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation guide
        with st.expander("📖 Come Leggere il Grafico"):
            st.markdown(
                """
                **Punti sulla Diagonale (linea grigia):**  
                Calibrazione perfetta. Es: se il modello dice 70% di confidenza, ha ragione esattamente 70% delle volte.
                
                **Punti Sopra la Diagonale:**  
                Modello **sottostima** la confidenza (troppo prudente). Es: dice 60%, ma ha ragione 75% delle volte → potrebbe essere più sicuro.
                
                **Punti Sotto la Diagonale:**  
                Modello **sovrastima** la confidenza (troppo ottimista). Es: dice 80%, ma ha ragione solo 60% delle volte → troppo sicuro di sé.
                
                **Gap Piccoli (< 5%):**  
                🟢 Eccellente - Puoi fidarti delle probabilità per prendere decisioni automatiche.
                
                **Gap Medi (5-10%):**  
                🟡 Accettabile - Probabilità utilizzabili, ma con margine di errore.
                
                **Gap Grandi (> 10%):**  
                🔴 Problematico - Usa solo il ranking relativo, non le probabilità assolute.
                """
            )
    else:
        st.info("Nessun dato di calibrazione disponibile per i filtri selezionati. Prova a cambiare modello o segmento.")


def _download_csv(df: pd.DataFrame, filename: str) -> None:
    """Provide CSV download button."""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, float_format="%.4f")
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label=f"📥 Download {filename}",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
    )


def render_kpi_dashboard_page(
    tables: Dict[str, pd.DataFrame],
    developer_metrics: Dict[str, Any],
    data_root_func: Optional[Callable[[], Path]] = None,
) -> None:
    """Render KPI Dashboard page with business-friendly ranking and calibration metrics.
    
    Rewritten for business readability with clear titles and technical tooltips.
    Shows precision@k, hit_rate@k, nDCG@k, MRR, coverage@k, Brier, ECE, MCE, reliability curves.
    """
    st.title("📊 Dashboard Qualità Predittiva")
    
    st.markdown(
        """
        Questo dashboard mostra **quanto bene il modello predice le partite corrette** per i pagamenti clienti.
        
        - **🎯 Qualità Raccomandazioni**: Accuratezza delle prime K suggerimenti (Precision, Hit Rate, nDCG, MRR, Coverage)
        - **🎯 Affidabilità Probabilità**: Quanto fidarsi dei punteggi di confidenza (Calibrazione: Brier, ECE, MCE)
        
        💡 **Usa i filtri** nella sidebar per analizzare clienti specifici o bucket di complessità.
        """
    )

    # Load data (support mock mode via data_root_func)
    if data_root_func and callable(data_root_func):
        curated_dir = data_root_func() / "curated"
    else:
        curated_dir = Path(__file__).resolve().parents[2] / "data" / "curated"
    
    kpi_ranking_path = curated_dir / "kpi_ranking.parquet"
    calibration_summary_path = curated_dir / "calibration_summary.parquet"
    calibration_bins_path = curated_dir / "calibration_bins.parquet"

    if not kpi_ranking_path.exists():
        st.error(
            f"❌ File metriche ranking non trovato: `kpi_ranking.parquet`\n\n"
            f"💡 **Soluzione:** Esegui il calcolo delle metriche:\n"
            f"```bash\n"
            f"python -m src.analytics.kpi_ranking\n"
            f"```"
        )
        st.stop()

    if not calibration_summary_path.exists() or not calibration_bins_path.exists():
        st.warning(
            "⚠️ File metriche calibrazione non trovati. Tab Calibrazione sarà vuota.\n\n"
            "💡 **Soluzione (opzionale):** Esegui diagnostica calibrazione:\n"
            "```bash\n"
            "python -m src.analytics.calibration\n"
            "```"
        )
        calibration_summary = pd.DataFrame()
        calibration_bins = pd.DataFrame()
    else:
        calibration_summary = _load_parquet_cached(calibration_summary_path)
        calibration_bins = _load_parquet_cached(calibration_bins_path)

    kpi_ranking = _load_parquet_cached(kpi_ranking_path)

    # Filters panel
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Filtri Dashboard")
    st.sidebar.caption("Filtra metriche per clienti specifici o complessità eventi.")

    # Client multi-select
    available_clients = []
    if "aggregation_value" in kpi_ranking.columns:
        client_rows = kpi_ranking[kpi_ranking["aggregation_key"] == "cliente"]
        available_clients = sorted(client_rows["aggregation_value"].unique().tolist())

    selected_clients = st.sidebar.multiselect(
        "🏢 Clienti",
        available_clients,
        default=[],
        help=(
            "Filtra per clienti specifici. Lascia vuoto per vedere metriche aggregate di tutti i clienti.\n\n"
            "Utile per confrontare performance su clienti difficili vs facili."
        ),
        key="clients_filter_kpi",
    )

    # Cardinality bucket filter
    cardinality_options = ["all", "k=1", "k=2-3", "k>=4"]
    cardinality_bucket = st.sidebar.selectbox(
        "📦 Complessità Eventi (Cardinalità)",
        cardinality_options,
        index=0,
        format_func=lambda x: {
            "all": "Tutti (k=1, k=2, k≥3, k≥4)",
            "k=1": "k=1 (Singola Partita) - Facile",
            "k=2-3": "k=2-3 (2-3 Partite) - Medio",
            "k>=4": "k≥4 (4+ Partite) - Complesso",
        }.get(x, x),
        help=(
            "Filtra eventi per **numero di partite da identificare** (cardinalità k).\n\n"
            "- **k=1**: Eventi con 1 sola partita da trovare (più facili)\n"
            "- **k=2-3**: Eventi con 2-3 partite (complessità media)\n"
            "- **k≥4**: Eventi con 4+ partite (più difficili)\n\n"
            "Utile per capire dove il modello performa meglio/peggio."
        ),
        key="cardinality_filter_kpi",
    )
    
    # Model filter (if multiple models available)
    available_models = []
    if not calibration_summary.empty and "model" in calibration_summary.columns:
        available_models = sorted(calibration_summary["model"].unique().tolist())
    
    if len(available_models) > 1:
        selected_model = st.sidebar.selectbox(
            "🤖 Modello",
            ["all"] + available_models,
            index=0,
            format_func=lambda x: {
                "all": "Tutti i Modelli",
                "catboost": "CatBoost (Gradient Boosting)",
                "autogluon": "AutoGluon (Ensemble)",
                "champion": "Champion (Migliore Modello)",
            }.get(x, x.title()),
            help=(
                "Filtra metriche per modello specifico.\n\n"
                "Utile per confrontare CatBoost vs AutoGluon o analizzare solo il Champion."
            ),
            key="model_filter_kpi",
        )
    else:
        selected_model = "all"

    # Apply filters
    filtered_kpi = _apply_filters(
        kpi_ranking,
        selected_clients,
        cardinality_bucket if cardinality_bucket != "all" else None,
        None,
    )

    filtered_calibration_summary = calibration_summary
    if not calibration_summary.empty:
        if selected_clients:
            filtered_calibration_summary = _apply_filters(
                calibration_summary,
                selected_clients,
                cardinality_bucket if cardinality_bucket != "all" else None,
                None,
            )
        if selected_model != "all" and "model" in filtered_calibration_summary.columns:
            filtered_calibration_summary = filtered_calibration_summary[
                filtered_calibration_summary["model"] == selected_model
            ]

    # Tabs
    st.markdown("---")
    tab1, tab2 = st.tabs(["🎯 Qualità Raccomandazioni", "🎯 Affidabilità Probabilità"])

    with tab1:
        _render_ranking_tab(kpi_ranking, filtered_kpi)
        
        st.markdown("---")
        st.markdown("### 📥 Export Dati Ranking")
        st.caption("Scarica tabella metriche filtrate in formato CSV per analisi esterne (Excel, Python, R, etc.)")
        _download_csv(filtered_kpi, "kpi_ranking_filtered.csv")
        
        # Deep-link
        with st.expander("� Deep-Link Condivisibile"):
            st.caption("Genera URL con filtri correnti per condividere questa vista con colleghi.")
            deeplink_url = generate_deeplink(
                page="KPI Dashboard",
                filters={
                    "selected_clients": selected_clients if selected_clients else [],
                    "cardinality_bucket": cardinality_bucket if cardinality_bucket else "all",
                    "model": selected_model if selected_model != "all" else "all",
                },
            )
            if st.button("📋 Genera Deep-Link Ranking", key="deeplink_ranking"):
                st.code(deeplink_url, language=None)
                st.caption("⚠️ Copia manualmente l'URL sopra (limitazione di Streamlit: clipboard API non supportata).")

    with tab2:
        if not calibration_summary.empty:
            _render_calibration_tab(calibration_summary, calibration_bins, filtered_calibration_summary)
            
            st.markdown("---")
            st.markdown("### 📥 Export Dati Calibrazione")
            st.caption("Scarica metriche e bin di calibrazione per analisi dettagliate.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Metriche Calibrazione (Summary)**")
                _download_csv(filtered_calibration_summary, "calibration_summary_filtered.csv")
            
            with col2:
                st.markdown("**Bin Calibrazione (Curve)**")
                # Filter calibration bins by same criteria
                filtered_calibration_bins = calibration_bins
                if not calibration_bins.empty:
                    # Apply model filter if specified
                    if "model_filter_calibration" in st.session_state:
                        model_filter = st.session_state.get("model_filter_calibration", [])
                        if model_filter:
                            filtered_calibration_bins = filtered_calibration_bins[
                                filtered_calibration_bins["model"].isin(model_filter)
                            ]
                    # Apply segment filter if specified
                    if "segment_filter_calibration" in st.session_state:
                        segment_filter = st.session_state.get("segment_filter_calibration", "global")
                        if segment_filter:
                            filtered_calibration_bins = filtered_calibration_bins[
                                filtered_calibration_bins["segment_type"] == segment_filter
                            ]
                _download_csv(filtered_calibration_bins, "calibration_bins_filtered.csv")
            
            # Deep-link
            with st.expander("� Deep-Link Condivisibile"):
                st.caption("Genera URL con filtri correnti per condividere questa vista con colleghi.")
                deeplink_url_cal = generate_deeplink(
                    page="KPI Dashboard",
                    filters={
                        "selected_clients": selected_clients if selected_clients else [],
                        "cardinality_bucket": cardinality_bucket if cardinality_bucket else "all",
                        "model": selected_model if selected_model != "all" else "all",
                        "tab": "calibration",
                    },
                )
                if st.button("📋 Genera Deep-Link Calibrazione", key="deeplink_calibration"):
                    st.code(deeplink_url_cal, language=None)
                    st.caption("⚠️ Copia manualmente l'URL sopra (limitazione di Streamlit: clipboard API non supportata).")
        else:
            st.info(
                "ℹ️ Nessun dato di calibrazione disponibile.\n\n"
                "La **calibrazione** misura l'affidabilità delle probabilità predette dal modello. "
                "Se il modello dice '70% di confidenza', quella previsione dovrebbe essere corretta 70% delle volte.\n\n"
                "💡 **Per abilitare questa sezione**, esegui:\n"
                "```bash\n"
                "python -m src.analytics.calibration\n"
                "```"
            )
