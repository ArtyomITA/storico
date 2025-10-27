"""KPI Dashboard page: ranking and calibration metrics with filters and download."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.utils.export import generate_deeplink


@st.cache_data(show_spinner=False)
def _load_parquet_cached(path: Path) -> pd.DataFrame:
    """Cached parquet loader for KPI data."""
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
    """Render ranking metrics tab with aggregated and per-client tables."""
    st.subheader("📊 Ranking Metrics")

    # Global metrics
    global_row = filtered_df[filtered_df.get("aggregation_key", "") == "global"]
    if not global_row.empty:
        st.markdown("### Global Metrics")
        cols = st.columns(5)
        k_values = [1, 2, 3, 5]
        for idx, k in enumerate(k_values):
            if idx < len(cols):
                precision = global_row.iloc[0].get(f"precision@{k}", 0.0)
                hit_rate = global_row.iloc[0].get(f"hit_rate@{k}", 0.0)
                coverage = global_row.iloc[0].get(f"coverage@{k}", 0.0)
                cols[idx].metric(
                    f"Precision@{k}",
                    f"{precision:.3f}",
                    help=f"Fraction of top-{k} items that are relevant",
                )
                cols[idx].metric(
                    f"Hit Rate@{k}",
                    f"{hit_rate:.3f}",
                    help=f"Fraction of relevant items found in top-{k}",
                )
                cols[idx].metric(
                    f"Coverage@{k}",
                    f"{coverage:.3f}",
                    help=f"Fraction of events with ≥1 hit in top-{k}",
                )

        mrr = global_row.iloc[0].get("mrr", 0.0)
        st.metric("MRR (Mean Reciprocal Rank)", f"{mrr:.3f}", help="Average reciprocal rank of first relevant item")

    # Cardinality buckets
    bucket_rows = filtered_df[filtered_df.get("aggregation_key", "") == "cardinality_bucket"]
    if not bucket_rows.empty:
        st.markdown("### Cardinality Bucket Metrics")
        st.dataframe(
            bucket_rows[
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
                    "aggregation_value": "Bucket",
                    "n_events": "Events",
                    "precision@3": "Precision@3",
                    "hit_rate@3": "Hit Rate@3",
                    "coverage@3": "Coverage@3",
                    "ndcg@3": "nDCG@3",
                    "mrr": "MRR",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    # Per-client metrics
    client_rows = filtered_df[filtered_df.get("aggregation_key", "") == "cliente"]
    if not client_rows.empty:
        st.markdown("### Per-Client Metrics")

        # Add search box
        search_term = st.text_input("🔍 Search by client", "", key="client_search_ranking")

        display_df = client_rows.copy()
        if search_term:
            display_df = display_df[
                display_df["aggregation_value"].astype(str).str.contains(search_term, case=False, na=False)
            ]

        # Sort options
        sort_col = st.selectbox(
            "Sort by",
            ["hit_rate@3", "precision@3", "mrr", "n_events"],
            key="sort_ranking",
        )
        sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True, key="sort_order_ranking")

        display_df = display_df.sort_values(
            by=sort_col,
            ascending=(sort_order == "Ascending"),
        )

        st.dataframe(
            display_df[
                [
                    "aggregation_value",
                    "n_events",
                    "precision@1",
                    "precision@3",
                    "hit_rate@3",
                    "coverage@3",
                    "mrr",
                ]
            ].rename(
                columns={
                    "aggregation_value": "Cliente",
                    "n_events": "Events",
                    "precision@1": "P@1",
                    "precision@3": "P@3",
                    "hit_rate@3": "Recall@3",
                    "coverage@3": "Cov@3",
                    "mrr": "MRR",
                }
            ),
            width="stretch",
            hide_index=True,
        )


def _render_calibration_tab(
    calibration_summary: pd.DataFrame, calibration_bins: pd.DataFrame, filtered_summary: pd.DataFrame
) -> None:
    """Render calibration diagnostics tab with reliability curves and metrics cards."""
    st.subheader("🎯 Calibration Diagnostics")

    # Metrics cards
    st.markdown("### Calibration Metrics")
    models = filtered_summary["model"].unique()

    for model in models:
        model_summary = filtered_summary[
            (filtered_summary["model"] == model) & (filtered_summary["segment_type"] == "global")
        ]
        if not model_summary.empty:
            st.markdown(f"#### Model: {model.upper()}")
            cols = st.columns(3)
            brier = model_summary.iloc[0].get("brier_score", 0.0)
            ece = model_summary.iloc[0].get("ece", 0.0)
            mce = model_summary.iloc[0].get("mce", 0.0)

            cols[0].metric(
                "Brier Score",
                f"{brier:.4f}",
                help="Mean squared error of probabilistic predictions (lower is better)",
            )
            cols[1].metric(
                "ECE (Expected Calibration Error)",
                f"{ece:.4f}",
                help="Weighted average calibration gap across bins (lower is better)",
            )
            cols[2].metric(
                "MCE (Maximum Calibration Error)",
                f"{mce:.4f}",
                help="Worst-case calibration gap (lower is better)",
            )

    # Reliability curves
    st.markdown("### Reliability Curves")
    st.caption("Predicted confidence vs observed accuracy. Perfect calibration = diagonal line.")

    # Filter bins by selected models
    model_filter = st.multiselect(
        "Select models",
        calibration_bins["model"].unique().tolist(),
        default=calibration_bins["model"].unique().tolist(),
        key="model_filter_calibration",
    )

    segment_filter = st.selectbox(
        "Segment",
        ["global", "cardinality_bucket", "cliente"],
        key="segment_filter_calibration",
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
                name="Perfect Calibration",
                line=dict(dash="dash", color="gray", width=1),
                showlegend=True,
            )
        )

        # Add reliability curves per model and segment
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
                        name=f"{model} - {segment_value}",
                        hovertemplate="<b>%{fullData.name}</b><br>"
                        + "Predicted: %{x:.3f}<br>"
                        + "Observed: %{y:.3f}<br>"
                        + "<extra></extra>",
                    )
                )

        fig.update_layout(
            xaxis_title="Predicted Probability (Confidence)",
            yaxis_title="Observed Frequency (Accuracy)",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            hovermode="closest",
            height=500,
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No calibration data available for selected filters.")


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


def render_kpi_dashboard_page(tables: Dict[str, pd.DataFrame], developer_metrics: Dict[str, Any]) -> None:
    """Render KPI Dashboard page with ranking and calibration metrics."""
    st.title("📊 KPI Dashboard")
    st.caption("Ranking and calibration metrics with filters and download options")

    # Load data
    curated_dir = Path(__file__).resolve().parents[2] / "data" / "curated"
    kpi_ranking_path = curated_dir / "kpi_ranking.parquet"
    calibration_summary_path = curated_dir / "calibration_summary.parquet"
    calibration_bins_path = curated_dir / "calibration_bins.parquet"

    if not kpi_ranking_path.exists():
        st.error(f"Missing kpi_ranking.parquet. Run: python -m src.analytics.kpi_ranking")
        st.stop()

    if not calibration_summary_path.exists() or not calibration_bins_path.exists():
        st.warning("Calibration data not found. Run: python -m src.analytics.calibration")
        calibration_summary = pd.DataFrame()
        calibration_bins = pd.DataFrame()
    else:
        calibration_summary = _load_parquet_cached(calibration_summary_path)
        calibration_bins = _load_parquet_cached(calibration_bins_path)

    kpi_ranking = _load_parquet_cached(kpi_ranking_path)

    # Filters panel
    st.sidebar.header("🔧 Filters")

    # Client multi-select
    available_clients = []
    if "aggregation_value" in kpi_ranking.columns:
        client_rows = kpi_ranking[kpi_ranking["aggregation_key"] == "cliente"]
        available_clients = sorted(client_rows["aggregation_value"].unique().tolist())

    selected_clients = st.sidebar.multiselect(
        "Clients",
        available_clients,
        default=[],
        help="Filter by specific clients (empty = all)",
        key="clients_filter_kpi",
    )

    # Cardinality bucket filter
    cardinality_options = ["all", "k=1", "k=2-3", "k>=4"]
    cardinality_bucket = st.sidebar.selectbox(
        "Bucket di cardinalità",
        cardinality_options,
        index=0,
        key="cardinality_filter_kpi",
    )

    # Apply filters
    filtered_kpi = _apply_filters(
        kpi_ranking,
        selected_clients,
        cardinality_bucket if cardinality_bucket != "all" else None,
        None,
    )

    filtered_calibration_summary = calibration_summary
    if not calibration_summary.empty and selected_clients:
        filtered_calibration_summary = _apply_filters(
            calibration_summary,
            selected_clients,
            cardinality_bucket if cardinality_bucket != "all" else None,
            None,
        )

    # Tabs
    tab1, tab2 = st.tabs(["📊 Ranking", "🎯 Calibration"])

    with tab1:
        _render_ranking_tab(kpi_ranking, filtered_kpi)
        st.markdown("---")
        _download_csv(filtered_kpi, "kpi_ranking_filtered.csv")
        
        # Deep-link
        st.markdown("#### 📋 Deep-Link")
        deeplink_url = generate_deeplink(
            page="KPI Dashboard",
            filters={
                "selected_clients": selected_clients if selected_clients else [],
                "cardinality_bucket": cardinality_bucket if cardinality_bucket else "all",
            },
        )
        if st.button("📋 Copia deep-link (ranking)", key="deeplink_ranking"):
            st.code(deeplink_url, language=None)
            st.caption("⚠️ Copia manualmente l'URL sopra (limitazione di Streamlit).")

    with tab2:
        if not calibration_summary.empty:
            _render_calibration_tab(calibration_summary, calibration_bins, filtered_calibration_summary)
            st.markdown("---")
            _download_csv(filtered_calibration_summary, "calibration_summary_filtered.csv")
            
            # Deep-link
            st.markdown("#### 📋 Deep-Link")
            deeplink_url_cal = generate_deeplink(
                page="KPI Dashboard",
                filters={
                    "selected_clients": selected_clients if selected_clients else [],
                    "cardinality_bucket": cardinality_bucket if cardinality_bucket else "all",
                    "tab": "calibration",
                },
            )
            if st.button("📋 Copia deep-link (calibrazione)", key="deeplink_calibration"):
                st.code(deeplink_url_cal, language=None)
                st.caption("⚠️ Copia manualmente l'URL sopra (limitazione di Streamlit).")
        else:
            st.info("Nessun dato di calibrazione disponibile. Esegui prima la diagnostica di calibrazione.")
