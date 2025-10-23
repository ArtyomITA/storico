"""Streamlit page for exploring client clusters."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
    st.title("Cluster Explorer")

    embeddings = tables.get("client_embedding_2d", pd.DataFrame())
    clusters = tables.get("client_clusters", pd.DataFrame())
    neighbors = tables.get("client_neighbors", pd.DataFrame())
    summaries = tables.get("cluster_summary", pd.DataFrame())

    if embeddings.empty:
        st.info("Proiezione 2D `client_embedding_2d.parquet` non disponibile.")
        return

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
        fig = px.scatter(
            filtered_plot,
            x="x",
            y="y",
            color="cluster_display",
            hover_data={
                "cliente": True,
                "cluster_display": True,
                "membership_prob": True,
            },
            labels={
                "x": "Dimensione 1",
                "y": "Dimensione 2",
                "cluster_display": "Cluster",
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

            cluster_label_text = str(
                summary_row.get("cluster_label", summary_row.get("label_text", "n.d."))
            )
            cluster_conf = float(
                summary_row.get(
                    "cluster_confidence", summary_row.get("confidence", 0.0)
                )
            )
            badges = "".join(
                [
                    _render_badge("Label", cluster_label_text),
                    _render_badge(
                        "Confidence",
                        f"{cluster_conf:.2f}",
                        color="#2e7d32",
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
