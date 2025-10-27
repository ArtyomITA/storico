"""Streamlit page for exploring client clusters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

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
    "same_period_score": "Stesso periodo",
    "largest_score": "Importo maggiore",
    "smallest_score": "Importo minore",
    "eomfirst_score": "Fine mese",
    "regmonth_score": "Mese registrazione",
    "quarter_score": "Trimestre",
    "seqblock_score": "Sequenza documenti",
    "typeprio_score": "Priorità tipo",
    "periodicity_weekly": "Settimanale",
    "periodicity_biweekly": "Bisettimanale",
    "periodicity_monthly": "Mensile",
    "dow_concentration": "Concentrazione DOW",
    "dom_concentration": "Concentrazione DOM",
    "P_k1": "k=1",
    "P_k2": "k=2",
    "P_kge3": "k≥3",
    "k_mode": "k_mode",
    "k_entropy": "k_entropy",
}


def _ensure_list(value: Any) -> list[Any]:
    """Normalize scalars, arrays, or None into a list for safe iteration."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (pd.Series, np.ndarray)):
        return [item for item in value.tolist()]
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return []
    except Exception:
        pass
    return [value]


def _format_cluster_label(cluster_label: int) -> str:
    return "Rumore (-1)" if cluster_label == -1 else f"Cluster {cluster_label}"


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
    neighbors = _ensure_list(neighbors_row.get("neighbors", []))
    distances = _ensure_list(neighbors_row.get("distances", []))
    rows: list[dict[str, float | str]] = []
    for neighbor, distance in zip(neighbors, distances):
        if neighbor is None or (isinstance(neighbor, float) and np.isnan(neighbor)):
            continue
        neighbor_id = str(neighbor)
        if neighbor_id == subject:
            continue
        if distance is None:
            dist_value = np.nan
        else:
            try:
                dist_value = float(distance)
            except (TypeError, ValueError):
                dist_value = np.nan
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
    data_root_func: Any = None,  # Function to get data root (for mock mode)
) -> None:
    st.title("🔬 Cluster Explorer — UX avanzata")

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
    # Parametri HDBSCAN
    # ========================================================================
    with st.sidebar.expander("🔧 Parametri HDBSCAN", expanded=False):
        st.markdown("**Ricalcola il clustering con parametri personalizzati**")
        
        min_cluster_size = st.slider(
            "Dimensione minima cluster (min_cluster_size)",
            min_value=5,
            max_value=100,
            value=30,
            step=5,
            help="Numero minimo di clienti per formare un cluster"
        )
        
        min_samples = st.slider(
            "Campioni minimi (min_samples)",
            min_value=1,
            max_value=50,
            value=15,
            step=1,
            help="Numero minimo di vicini per considerare un punto core"
        )
        
        # Preview cluster counts
        if HDBSCAN_AVAILABLE and not embeddings_full.empty:
            if st.button("🔄 Anteprima clustering", key="preview_hdbscan"):
                with st.spinner("Calcolo dell'anteprima in corso..."):
                    try:
                        labels, n_clusters, n_noise = _recompute_hdbscan(
                            embeddings_full,
                            min_cluster_size,
                            min_samples
                        )
                        st.success(f"✅ Anteprima: {n_clusters} cluster, {n_noise} punti rumorosi")
                        
                        # Store in session state for Apply button
                        st.session_state["hdbscan_preview_labels"] = labels
                        st.session_state["hdbscan_preview_params"] = {
                            "min_cluster_size": min_cluster_size,
                            "min_samples": min_samples
                        }
                    except Exception as e:
                        st.error(f"Anteprima non riuscita: {e}")
            
            # Apply button to persist changes
            if "hdbscan_preview_labels" in st.session_state:
                if st.button("✅ Applica e salva", key="apply_hdbscan"):
                    with st.spinner("Applicazione modifiche e salvataggio..."):
                        try:
                            labels = st.session_state["hdbscan_preview_labels"]
                            params = st.session_state["hdbscan_preview_params"]
                            
                            # Update clusters dataframe
                            clusters_updated = embeddings_full[["cliente"]].copy()
                            clusters_updated["cluster_label"] = labels
                            clusters_updated["cluster_label_numeric"] = labels
                            clusters_updated["membership_prob"] = 1.0  # Simplified
                            
                            # Save to parquet - respect mock mode
                            from pathlib import Path
                            if data_root_func and callable(data_root_func):
                                curated_dir = data_root_func() / "curated"
                            else:
                                curated_dir = Path("data/curated")
                            curated_dir.mkdir(parents=True, exist_ok=True)
                            clusters_updated.to_parquet(curated_dir / "client_clusters.parquet")
                            
                            st.success(f"✅ Salvato! Cluster aggiornati con parametri: {params}")
                            st.info("Aggiorna la pagina per vedere i cluster aggiornati.")
                            
                            # Clear preview from session state
                            del st.session_state["hdbscan_preview_labels"]
                            del st.session_state["hdbscan_preview_params"]
                        except Exception as e:
                            st.error(f"Salvataggio non riuscito: {e}")
        elif not HDBSCAN_AVAILABLE:
            st.warning("HDBSCAN non disponibile. Installa con: pip install hdbscan")
        elif embeddings_full.empty:
            st.info("Sono necessari gli embedding completi (client_embeddings.parquet) per ricalcolare.")
    
    # ========================================================================
    # Selettore proiezione
    # ========================================================================
    with st.sidebar.expander("📊 Metodo di proiezione", expanded=False):
        projection_method = st.radio(
            "Proiezione 2D",
            options=["PCA (predefinito)", "UMAP"],
            index=0,
            key="projection_method_toggle",
            help="Alterna tra PCA e UMAP per la visualizzazione 2D"
        )
        
        if projection_method == "UMAP" and UMAP_AVAILABLE and not embeddings_full.empty:
            if st.button("🔄 Calcola UMAP", key="compute_umap"):
                with st.spinner("Calcolo della proiezione UMAP in corso..."):
                    try:
                        umap_df = _compute_umap_projection(embeddings_full)
                        st.session_state["umap_projection"] = umap_df
                        st.success("✅ Proiezione UMAP generata!")
                    except Exception as e:
                        st.error(f"Calcolo UMAP non riuscito: {e}")
        elif projection_method == "UMAP" and not UMAP_AVAILABLE:
            st.warning("UMAP non disponibile. Installa con: pip install umap-learn")
        elif projection_method == "UMAP" and embeddings_full.empty:
            st.info("Sono necessari gli embedding completi per calcolare UMAP.")
    
    # Use UMAP projection if available and selected
    if projection_method == "UMAP" and "umap_projection" in st.session_state:
        embeddings = st.session_state["umap_projection"]
        st.caption("📊 Visualizzazione con proiezione UMAP")
    else:
        st.caption("📊 Visualizzazione con proiezione PCA (predefinita)")

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
    # Arricchisci i tooltip con i punteggi policy e match_rate se disponibili
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

        label_dict = {
            "x": "Dimensione 1",
            "y": "Dimensione 2",
            "cluster_display": "Cluster",
            "membership_prob": "Probabilità appartenenza",
            "match_rate": "Match rate",
        }
        for metric in ["fifo_score", "largest_score", "same_period_score"]:
            if metric in filtered_plot.columns:
                label_dict[metric] = METRIC_LABELS.get(metric, metric)

        fig = px.scatter(
            filtered_plot,
            x="x",
            y="y",
            color="cluster_display",
            hover_data=hover_data_dict,
            labels=label_dict,
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
        st.plotly_chart(fig, width="stretch")

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
            st.dataframe(display_df, width="stretch")
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
                st.dataframe(neighbors_table, width="stretch")

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
                "Pattern dominante",
                f"{summary_row.get('dominant_pattern', 'n.d.')}",
            )

            # Genera automaticamente l'etichetta se assente
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
                
                # Usa etichetta generata automaticamente
                cluster_label_text = auto_label
                cluster_conf = auto_conf
                
                st.info("🤖 Etichetta generata automaticamente dal profilo del cluster")
            
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
                policy_col.dataframe(policy_df, width="stretch")

            periodicity_df = _summarise_metrics(
                summary_row, PERIODICITY_METRICS
            )
            if not periodicity_df.empty:
                periodicity_col.write("**Periodicità**")
                periodicity_col.dataframe(
                    periodicity_df, width="stretch"
                )

            cardinality_df = _summarise_metrics(
                summary_row, CARDINALITY_METRICS
            )
            if not cardinality_df.empty:
                cardinality_col.write("**Cardinalità**")
                cardinality_col.dataframe(
                    cardinality_df, width="stretch"
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
                st.dataframe(pd.DataFrame(macro_records), width="stretch")
