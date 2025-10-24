"""Streamlit page for exploring client clusters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Try to import HDBSCAN (optional dependency)
try:
    import hdbscan  # type: ignore
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# Try to import UMAP (optional dependency)
try:
    import umap  # type: ignore
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

POLICY_METRICS: tuple[str, ...] = (
    "fifo_score",
    "lifo_score",
    "same_period_score",
    "largest_score",
    "smallest_score",
    "eomfirst_score",
    "regmonth_score",
    "quarter_score",
    "seqblock_score",
    "typeprio_score",
)
PERIODICITY_METRICS: tuple[str, ...] = (
    "periodicity_weekly",
    "periodicity_biweekly",
    "periodicity_monthly",
    "dow_concentration",
    "dom_concentration",
)
CARDINALITY_METRICS: tuple[str, ...] = (
    "P_k1",
    "P_k2",
    "P_kge3",
    "k_mode",
    "k_entropy",
)
METRIC_LABELS: dict[str, str] = {
    "fifo_score": "FIFO",
    "lifo_score": "LIFO",
    "same_period_score": "Same-Period",
    "largest_score": "Largest",
    "smallest_score": "Smallest",
    "eomfirst_score": "EOMFirst",
    "regmonth_score": "RegMonth",
    "quarter_score": "Quarter",
    "seqblock_score": "SeqBlock",
    "typeprio_score": "TypePriority",
    "periodicity_weekly": "Settimanale",
    "periodicity_biweekly": "Bi-settimanale",
    "periodicity_monthly": "Mensile",
    "dow_concentration": "DOW concentration",
    "dom_concentration": "DOM concentration",
    "P_k1": "k=1",
    "P_k2": "k=2",
    "P_kge3": "k>=3",
    "k_mode": "k_mode",
    "k_entropy": "k_entropy",
}


def _format_cluster_label(cluster_label: int) -> str:
    return "Noise (-1)" if cluster_label == -1 else f"Cluster {cluster_label}"


def _render_badge(label: str, value: str, *, color: str = "#1f77b4") -> str:
    safe_label = label.replace("<", "&lt;").replace(">", "&gt;")
    safe_value = value.replace("<", "&lt;").replace(">", "&gt;")
    style = (
        "display:inline-block;padding:0.25rem 0.65rem;margin-right:0.4rem;"
        f"border-radius:999px;background:{color};color:#fff;font-size:0.85rem;"
        "font-weight:600;"
    )
    return f"<span style=\"{style}\">{safe_label}: {safe_value}</span>"


def _summarise_metrics(row: pd.Series, metrics: Sequence[str]) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for metric in metrics:
        mean_col = f"{metric}_mean"
        if mean_col not in row:
            continue
        mean_value = float(row.get(mean_col, np.nan))
        entry: dict[str, float | str] = {
            "Metrica": METRIC_LABELS.get(metric, metric),
            "Media": mean_value,
        }
        iqr_col = f"{metric}_iqr"
        if iqr_col in row:
            entry["IQR"] = float(row.get(iqr_col, np.nan))
        records.append(entry)
    return pd.DataFrame(records)


def _generate_automatic_label(
    dominant_pattern: str,
    k_mode: float,
    periodicity_metrics: Dict[str, float]
) -> Tuple[str, float]:
    """
    Generate automatic cluster label from profile metrics.
    
    Returns:
        (label_text, confidence): Auto-generated label and confidence score
    """
    # Extract periodicity
    period_str = "irregular"
    period_conf = 0.0
    if periodicity_metrics:
        weekly = periodicity_metrics.get("periodicity_weekly", 0.0)
        biweekly = periodicity_metrics.get("periodicity_biweekly", 0.0)
        monthly = periodicity_metrics.get("periodicity_monthly", 0.0)
        
        max_period = max(weekly, biweekly, monthly)
        if max_period > 0.3:
            if weekly == max_period:
                period_str = "weekly"
                period_conf = weekly
            elif biweekly == max_period:
                period_str = "bi-weekly"
                period_conf = biweekly
            else:
                period_str = "monthly"
                period_conf = monthly
    
    # Cardinality description
    card_str = f"k~{int(k_mode)}" if not np.isnan(k_mode) else "k~?"
    
    # Pattern description
    pattern_str = dominant_pattern if dominant_pattern and dominant_pattern != "n.d." else "mixed"
    
    # Build label
    label_text = f"{pattern_str.upper()}, {card_str}, {period_str}"
    
    # Confidence: average of pattern strength (assumed 0.7 if present) and periodicity confidence
    pattern_conf = 0.7 if pattern_str != "mixed" else 0.3
    confidence = (pattern_conf + period_conf) / 2.0
    
    return (label_text, confidence)


def _recompute_hdbscan(
    embeddings_df: pd.DataFrame,
    min_cluster_size: int,
    min_samples: int,
    metric: str = "cosine"
) -> Tuple[np.ndarray, int, int]:
    """
    Recompute HDBSCAN clustering with new parameters.
    
    Returns:
        (labels, n_clusters, n_noise): Cluster labels, number of clusters, number of noise points
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan package required for reclustering")
    
    import hdbscan  # type: ignore
    
    # Extract embedding columns (exclude 'cliente')
    embed_cols = [col for col in embeddings_df.columns if col != "cliente"]
    X = embeddings_df[embed_cols].values
    
    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        core_dist_n_jobs=1
    )
    labels = clusterer.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    
    return (labels, n_clusters, n_noise)


def _compute_umap_projection(embeddings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute UMAP 2D projection from embeddings.
    
    Returns:
        DataFrame with columns: cliente, x, y
    """
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn package required for UMAP projection")
    
    import umap  # type: ignore
    
    embed_cols = [col for col in embeddings_df.columns if col != "cliente"]
    X = embeddings_df[embed_cols].values
    
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(X)
    
    result_df = pd.DataFrame({
        "cliente": embeddings_df["cliente"].values,
        "x": embedding_2d[:, 0],
        "y": embedding_2d[:, 1]
    })
    
    return result_df


def _build_neighbors_table(neighbors_row: pd.Series, subject: str) -> pd.DataFrame:
    neighbors = neighbors_row.get("neighbors", []) or []
    distances = neighbors_row.get("distances", []) or []
    rows: list[dict[str, float | str]] = []
    for neighbor, distance in zip(neighbors, distances):
        if neighbor is None or (isinstance(neighbor, float) and np.isnan(neighbor)):
            continue
        neighbor_id = str(neighbor)
        if neighbor_id == subject:
            continue
        dist_value = (
            np.nan
            if distance is None or (isinstance(distance, float) and np.isnan(distance))
            else float(distance)
        )
        rows.append({"Cliente": neighbor_id, "Distanza": dist_value})
        if len(rows) >= 10:
            break
    return pd.DataFrame(rows)


def _rerun_app() -> None:
    rerun = getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()


def render_clusters_page(
    *,
    tables: Dict[str, pd.DataFrame],
    selected_client: str,
    clients: Sequence[str],
) -> None:
    st.title("🔬 Cluster Explorer — Enhanced UX (S11)")

    embeddings = tables.get("client_embedding_2d", pd.DataFrame())
    clusters = tables.get("client_clusters", pd.DataFrame())
    neighbors = tables.get("client_neighbors", pd.DataFrame())
    summaries = tables.get("cluster_summary", pd.DataFrame())
    
    # Load full embeddings for HDBSCAN recalculation if available
    embeddings_full = tables.get("client_embeddings", pd.DataFrame())

    if embeddings.empty:
        st.info("Proiezione 2D `client_embedding_2d.parquet` non disponibile.")
        return
    
    # ========================================================================
    # S11: HDBSCAN Parameter Panel
    # ========================================================================
    with st.sidebar.expander("🔧 HDBSCAN Parameters", expanded=False):
        st.markdown("**Recalculate clustering with custom parameters**")
        
        min_cluster_size = st.slider(
            "min_cluster_size",
            min_value=5,
            max_value=100,
            value=30,
            step=5,
            help="Minimum number of clients to form a cluster"
        )
        
        min_samples = st.slider(
            "min_samples",
            min_value=1,
            max_value=50,
            value=15,
            step=1,
            help="Minimum samples in neighborhood for core point"
        )
        
        # Preview cluster counts
        if HDBSCAN_AVAILABLE and not embeddings_full.empty:
            if st.button("🔄 Preview Clustering", key="preview_hdbscan"):
                with st.spinner("Computing preview..."):
                    try:
                        labels, n_clusters, n_noise = _recompute_hdbscan(
                            embeddings_full,
                            min_cluster_size,
                            min_samples
                        )
                        st.success(f"✅ Preview: {n_clusters} clusters, {n_noise} noise points")
                        
                        # Store in session state for Apply button
                        st.session_state["hdbscan_preview_labels"] = labels
                        st.session_state["hdbscan_preview_params"] = {
                            "min_cluster_size": min_cluster_size,
                            "min_samples": min_samples
                        }
                    except Exception as e:
                        st.error(f"Preview failed: {e}")
            
            # Apply button to persist changes
            if "hdbscan_preview_labels" in st.session_state:
                if st.button("✅ Apply & Save", key="apply_hdbscan"):
                    with st.spinner("Applying changes and saving..."):
                        try:
                            labels = st.session_state["hdbscan_preview_labels"]
                            params = st.session_state["hdbscan_preview_params"]
                            
                            # Update clusters dataframe
                            clusters_updated = embeddings_full[["cliente"]].copy()
                            clusters_updated["cluster_label"] = labels
                            clusters_updated["cluster_label_numeric"] = labels
                            clusters_updated["membership_prob"] = 1.0  # Simplified
                            
                            # Save to parquet
                            from pathlib import Path
                            curated_dir = Path("data/curated")
                            curated_dir.mkdir(parents=True, exist_ok=True)
                            clusters_updated.to_parquet(curated_dir / "client_clusters.parquet")
                            
                            st.success(f"✅ Saved! Clusters updated with params: {params}")
                            st.info("Refresh the page to see updated clusters.")
                        except Exception as e:
                            st.error(f"Save failed: {e}")
        elif not HDBSCAN_AVAILABLE:
            st.warning("HDBSCAN not available. Install with: pip install hdbscan")
        elif embeddings_full.empty:
            st.info("Full embeddings (client_embeddings.parquet) required for recalculation.")
    
    # ========================================================================
    # S11: UMAP/PCA Toggle
    # ========================================================================
    with st.sidebar.expander("📊 Projection Method", expanded=False):
        projection_method = st.radio(
            "2D Projection",
            options=["PCA (default)", "UMAP"],
            index=0,
            key="projection_method_toggle",
            help="Toggle between PCA and UMAP for 2D visualization"
        )
        
        if projection_method == "UMAP" and UMAP_AVAILABLE and not embeddings_full.empty:
            if st.button("🔄 Compute UMAP", key="compute_umap"):
                with st.spinner("Computing UMAP projection..."):
                    try:
                        umap_df = _compute_umap_projection(embeddings_full)
                        st.session_state["umap_projection"] = umap_df
                        st.success("✅ UMAP projection computed!")
                    except Exception as e:
                        st.error(f"UMAP computation failed: {e}")
        elif projection_method == "UMAP" and not UMAP_AVAILABLE:
            st.warning("UMAP not available. Install with: pip install umap-learn")
        elif projection_method == "UMAP" and embeddings_full.empty:
            st.info("Full embeddings required for UMAP computation.")
    
    # Use UMAP projection if available and selected
    if projection_method == "UMAP" and "umap_projection" in st.session_state:
        embeddings = st.session_state["umap_projection"]
        st.caption("📊 Using UMAP projection")
    else:
        st.caption("📊 Using PCA projection (default)")

    plot_df = embeddings.copy()
    if not clusters.empty:
        clusters = clusters.copy()
        if "cluster_label" in clusters.columns:
            label_series = pd.to_numeric(
                clusters["cluster_label"], errors="coerce"
            ).fillna(-1)
        else:
            label_series = pd.Series(-1, index=clusters.index, dtype=float)
        clusters["cluster_label_numeric"] = label_series.astype(int)
        if "membership_prob" not in clusters.columns:
            clusters["membership_prob"] = np.nan
        plot_df = plot_df.merge(
            clusters[["cliente", "cluster_label_numeric", "membership_prob"]],
            on="cliente",
            how="left",
        )
    else:
        plot_df["cluster_label_numeric"] = -1
        plot_df["membership_prob"] = np.nan

    plot_df["cluster_display"] = plot_df["cluster_label_numeric"].apply(
        _format_cluster_label
    )

    available_clusters = sorted(
        plot_df["cluster_label_numeric"].dropna().astype(int).unique().tolist()
    )
    cluster_options = [None] + available_clusters
    selected_cluster = st.sidebar.selectbox(
        "Cluster",
        cluster_options,
        format_func=lambda value: "Tutti"
        if value is None
        else _format_cluster_label(int(value)),
        key="cluster_page_filter",
    )

    filtered_plot = (
        plot_df
        if selected_cluster is None
        else plot_df[plot_df["cluster_label_numeric"] == int(selected_cluster)]
    )

    st.subheader("Mappa 2D")
    if filtered_plot.empty:
        st.info("Nessun punto da visualizzare per il filtro corrente.")
    else:
        # S11: Enrich tooltips with policy scores and match_rate if available
        client_profiles = tables.get("client_profiles", pd.DataFrame())
        if not client_profiles.empty:
            # Merge policy scores for rich tooltips
            tooltip_cols = ["cliente"]
            for col in ["fifo_score", "largest_score", "same_period_score", "match_rate"]:
                if col in client_profiles.columns:
                    tooltip_cols.append(col)
            
            if len(tooltip_cols) > 1:
                filtered_plot = filtered_plot.merge(
                    client_profiles[tooltip_cols],
                    on="cliente",
                    how="left"
                )
        
        # Build hover_data dict dynamically
        hover_data_dict = {
            "cliente": True,
            "cluster_display": True,
            "membership_prob": ":.2f",
        }
        
        # Add policy scores to hover if available
        for col in ["fifo_score", "largest_score", "same_period_score", "match_rate"]:
            if col in filtered_plot.columns:
                hover_data_dict[col] = ":.2f"
        
        fig = px.scatter(
            filtered_plot,
            x="x",
            y="y",
            color="cluster_display",
            hover_data=hover_data_dict,
            labels={
                "x": "Dimensione 1",
                "y": "Dimensione 2",
                "cluster_display": "Cluster",
                "fifo_score": "FIFO",
                "largest_score": "Largest",
                "same_period_score": "Same-Period",
                "match_rate": "Match Rate"
            },
        )
        fig.update_traces(marker=dict(size=8, opacity=0.85))
        selected_point = plot_df[plot_df["cliente"] == selected_client]
        if not selected_point.empty:
            fig.add_trace(
                go.Scatter(
                    x=selected_point["x"],
                    y=selected_point["y"],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color="black",
                        symbol="diamond",
                        line=dict(width=1, color="white"),
                    ),
                    name=f"Cliente {selected_client}",
                    hoverinfo="skip",
                )
            )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Clienti nel cluster")
    if clusters.empty:
        st.info("Tabella `client_clusters.parquet` non disponibile.")
    elif selected_cluster is None:
        st.info("Seleziona un cluster nella sidebar per vedere i clienti associati.")
    else:
        cluster_clients = clusters[
            clusters["cluster_label_numeric"] == int(selected_cluster)
        ].copy()
        if cluster_clients.empty:
            st.info("Nessun cliente associato a questo cluster.")
        else:
            display_df = (
                cluster_clients.assign(
                    cluster_label=lambda df: df["cluster_label_numeric"].apply(
                        _format_cluster_label
                    ),
                    membership_prob=lambda df: df["membership_prob"].astype(float),
                )[["cliente", "cluster_label", "membership_prob"]]
                .sort_values("membership_prob", ascending=False)
            )
            st.dataframe(display_df, use_container_width=True)
            cluster_client_choice = st.selectbox(
                "Seleziona un cliente nel cluster",
                display_df["cliente"].tolist(),
                key="cluster_page_client_select",
            )
            if st.button("Apri Pattern Card", key="cluster_page_open_pattern_card"):
                st.session_state["selected_client"] = cluster_client_choice
                st.session_state["selected_page"] = "Pattern Card"
                _rerun_app()

    st.subheader("Clienti vicini")
    if neighbors.empty:
        st.info("Tabella `client_neighbors.parquet` non disponibile.")
    else:
        default_index = clients.index(selected_client) if selected_client in clients else 0
        neighbor_subject = st.selectbox(
            "Cliente di riferimento",
            clients,
            index=default_index,
            key="cluster_page_neighbor_subject",
        )
        neighbor_row = neighbors[neighbors["cliente"] == neighbor_subject]
        if neighbor_row.empty:
            st.info("Nessun vicino disponibile per il cliente selezionato.")
        else:
            neighbors_table = _build_neighbors_table(
                neighbor_row.iloc[0], neighbor_subject
            )
            if neighbors_table.empty:
                st.info("Nessun vicino disponibile per il cliente selezionato.")
            else:
                st.dataframe(neighbors_table, use_container_width=True)

    st.subheader("Profilo cluster")
    if summaries.empty:
        st.info("File `cluster_summary.parquet` non disponibile.")
    elif selected_cluster is None:
        st.info("Seleziona un cluster per visualizzare il profilo.")
    else:
        summary_row_df = summaries[
            summaries["cluster_label_id"] == int(selected_cluster)
        ]
        if summary_row_df.empty:
            st.info("Nessun profilo disponibile per il cluster selezionato.")
        else:
            summary_row = summary_row_df.iloc[0]
            info_cols = st.columns(3)
            info_cols[0].metric(
                "Cluster", _format_cluster_label(int(selected_cluster))
            )
            info_cols[1].metric(
                "Clienti", f"{int(summary_row.get('size', 0))}"
            )
            info_cols[2].metric(
                "Dominant pattern",
                f"{summary_row.get('dominant_pattern', 'n.d.')}",
            )

            # S11: Generate automatic label if not present
            cluster_label_text = str(
                summary_row.get("cluster_label", summary_row.get("label_text", "n.d."))
            )
            cluster_conf = float(
                summary_row.get(
                    "cluster_confidence", summary_row.get("confidence", 0.0)
                )
            )
            
            # If label is generic or missing, generate automatic label
            if cluster_label_text in ["n.d.", "Cluster", ""] or cluster_conf == 0.0:
                dominant_pattern = str(summary_row.get("dominant_pattern", "mixed"))
                k_mode_val = float(summary_row.get("k_mode_mean", np.nan))
                
                periodicity_dict = {}
                for metric in PERIODICITY_METRICS:
                    mean_col = f"{metric}_mean"
                    if mean_col in summary_row:
                        periodicity_dict[metric] = float(summary_row.get(mean_col, 0.0))
                
                auto_label, auto_conf = _generate_automatic_label(
                    dominant_pattern,
                    k_mode_val,
                    periodicity_dict
                )
                
                # Use auto-generated label
                cluster_label_text = auto_label
                cluster_conf = auto_conf
                
                st.info("🤖 Auto-generated label from cluster profile")
            
            badges = "".join(
                [
                    _render_badge("Label", cluster_label_text),
                    _render_badge(
                        "Confidence",
                        f"{cluster_conf:.2f}",
                        color="#2e7d32" if cluster_conf > 0.5 else "#ed6c02",
                    ),
                ]
            )
            st.markdown(badges, unsafe_allow_html=True)
            st.write(f"**Descrizione**: {cluster_label_text}")

            policy_col, periodicity_col, cardinality_col = st.columns(3)
            policy_df = _summarise_metrics(summary_row, POLICY_METRICS)
            if not policy_df.empty:
                policy_col.write("**Policy**")
                policy_col.dataframe(policy_df, use_container_width=True)

            periodicity_df = _summarise_metrics(
                summary_row, PERIODICITY_METRICS
            )
            if not periodicity_df.empty:
                periodicity_col.write("**Periodicità**")
                periodicity_col.dataframe(
                    periodicity_df, use_container_width=True
                )

            cardinality_df = _summarise_metrics(
                summary_row, CARDINALITY_METRICS
            )
            if not cardinality_df.empty:
                cardinality_col.write("**Cardinalità**")
                cardinality_col.dataframe(
                    cardinality_df, use_container_width=True
                )

            macro_cols = [
                col
                for col in summary_row.index
                if col.startswith("share_tipo_documento_")
                and col.endswith("_mean")
            ]
            if macro_cols:
                macro_records = []
                for mean_col in macro_cols:
                    macro_name = (
                        mean_col.replace("share_tipo_documento_", "")
                        .replace("_mean", "")
                    )
                    macro_records.append(
                        {
                            "Macro-tipo": macro_name,
                            "Media": float(summary_row.get(mean_col, np.nan)),
                            "IQR": float(
                                summary_row.get(
                                    mean_col.replace("_mean", "_iqr"), np.nan
                                )
                            ),
                        }
                    )
                st.write("**Distribuzione macro-tipologie**")
                st.dataframe(pd.DataFrame(macro_records), use_container_width=True)
