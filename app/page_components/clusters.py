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

 # Training manager imports
from src.utils.training_manager import (
    TrainingManager,
    build_training_command,
    get_hdbscan_artifacts,
)

# Export utilities
from app.utils.export import (
    export_dataframe_to_csv,
    generate_deeplink,
    generate_export_filename,
    save_export_to_disk,
)

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
    "fifo_score": "FIFO (Prima Emessa, Prima Pagata)",  
    "lifo_score": "LIFO (Ultima Emessa, Prima Pagata)",  
    "same_period_score": "Stesso Periodo",  
    "largest_score": "Importo Maggiore",  
    "smallest_score": "Importo Minore",  
    "eomfirst_score": "Pagamento a Fine Mese",  
    "regmonth_score": "Stesso Mese (Reg=Comp)",  
    "quarter_score": "Cadenza Trimestrale",  
    "seqblock_score": "Sequenza Documenti",  
    "typeprio_score": "Priorità per Tipo",  
    "periodicity_weekly": "Cadenza Settimanale",  
    "periodicity_biweekly": "Cadenza Quindicinale",  
    "periodicity_monthly": "Cadenza Mensile",  
    "dow_concentration": "Concentrazione Giorno Settimana",  
    "dom_concentration": "Concentrazione Giorno Mese",  
    "P_k1": "Prob. k=1 Partita",  
    "P_k2": "Prob. k=2 Partite",  
    "P_kge3": "Prob. k≥3 Partite",  
    "k_mode": "Cardinalità Modale",  
    "k_entropy": "Entropia Cardinalità",  
}


# ============================================================================
 # HDBSCAN Training Actions Panel
# ============================================================================

def _render_hdbscan_actions_panel() -> None:
    """Render riquadro Azioni con pulsante HDBSCAN clustering."""
    manager = TrainingManager()
    
    with st.expander("🔬 **Azioni Clustering HDBSCAN**", expanded=False):
        st.markdown(
            "**HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise) "
            "raggruppa clienti con comportamenti simili basandosi sulla **densità locale** degli embedding. "
            "L'algoritmo identifica cluster di forma arbitraria e gestisce automaticamente il **rumore** "
            "(clienti outlier non assegnati a nessun cluster). "
            "Il parametro **stabilità** (stability) indica la robustezza del cluster rispetto a variazioni "
            "nei parametri."
        )
        
        st.markdown("---")
        
        # Stato ultimo run
        hdbscan_status = manager.get_run_status("hdbscan")
        artifacts_exist = manager.check_artifacts_exist("hdbscan")
        
        col_status, col_params = st.columns([1, 1])
        
        with col_status:
            st.markdown("### 📊 Stato Clustering")
            
            if hdbscan_status:
                _render_hdbscan_status_card(hdbscan_status, artifacts_exist)
            else:
                st.info("Nessun clustering HDBSCAN registrato.")
        
        with col_params:
            st.markdown("### ⚙️ Parametri HDBSCAN")
            
 # Expanded parameter range for better control
            min_cluster_size = st.slider(
                "📏 Dimensione minima cluster",
                min_value=2,
                max_value=50,
                value=5,
                step=1,
                key="hdbscan_min_cluster_size",
                help=(
                    "**min_cluster_size**: Numero minimo di clienti per formare un cluster.\n\n"
                    "📘 **Business Interpretation**: Se hai 100 clienti e setti min_cluster_size=10, "
                    "HDBSCAN creerà solo cluster con almeno 10 clienti. Clienti in gruppi più piccoli "
                    "diventeranno 'noise' (outlier).\n\n"
                    "📊 **Linee guida**:\n"
                    "- **2-5**: Molti cluster piccoli, cattura nicchie rare (alta granularità)\n"
                    "- **5-10**: Bilanciato per 50-100 clienti\n"
                    "- **10-20**: Cluster robusti per 100-500 clienti\n"
                    "- **>20**: Solo macro-segmenti dominanti (bassa granularità)\n\n"
                    "⚠️ **Attenzione**: Valori troppo bassi (<3) possono produrre micro-cluster instabili."
                ),
            )
            
            min_samples = st.slider(
                "🎯 Campioni minimi per densità",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                key="hdbscan_min_samples",
                help=(
                    "**min_samples**: Numero minimo di vicini per considerare un punto 'core' (denso).\n\n"
                    "📘 **Business Interpretation**: Se setti min_samples=5, un cliente diventa 'core' "
                    "solo se ha almeno 5 clienti simili vicini nello spazio embedding. Punti non-core "
                    "possono essere assegnati a cluster ma con minor certezza.\n\n"
                    "📊 **Linee guida**:\n"
                    "- **1**: Tutti i punti sono potenzialmente core (più inclusivo)\n"
                    "- **2-5**: Bilanciato, richiede almeno qualche vicino simile\n"
                    "- **5-10**: Conservativo, solo regioni molto dense\n"
                    "- **>10**: Molto restrittivo, più noise\n\n"
                    "💡 **Regola pratica**: Usa min_samples = min_cluster_size / 2 o 3 come punto di partenza."
                ),
            )
            
 # Metric selection
            metric = st.selectbox(
                "📐 Metrica distanza",
                options=["euclidean", "manhattan", "cosine"],
                index=2,  # Default cosine (best for embeddings)
                key="hdbscan_metric",
                help=(
                    "**metric**: Funzione per calcolare distanza tra clienti negli embedding.\n\n"
                    "📊 **Opzioni**:\n"
                    "- **euclidean**: Distanza geometrica classica (√(Σ(x_i - y_i)²)). Buona per dati normalizzati.\n"
                    "- **manhattan**: Somma differenze assolute (Σ|x_i - y_i|). Robusta a outlier.\n"
                    "- **cosine**: Angolo tra vettori (1 - cos(θ)). **Migliore per embeddings** perché ignora magnitudo.\n\n"
                    "💡 **Raccomandazione**: Usa **cosine** per embeddings neurali (default)."
                ),
            )
            
            st.markdown(
                "**💡 Suggerimenti pratici:**\n"
                "- **Dataset piccolo** (50-100 clienti): min_cluster_size=3-5, min_samples=2\n"
                "- **Dataset medio** (100-300 clienti): min_cluster_size=5-10, min_samples=3-5\n"
                "- **Dataset grande** (>300 clienti): min_cluster_size=10-20, min_samples=5-10\n"
                "- Se vedi **troppo noise** (>30%): riduci min_cluster_size o min_samples\n"
                "- Se vedi **troppi cluster** (>10): aumenta min_cluster_size"
            )
        
        # Pulsante
        button_label = "Riesegui HDBSCAN" if artifacts_exist else "Esegui HDBSCAN"
        if st.button(button_label, key="btn_hdbscan", type="primary"):
            _start_hdbscan_clustering(manager, min_cluster_size, min_samples, metric)


def _render_hdbscan_status_card(run, artifacts_exist: bool) -> None:
    """Render card with HDBSCAN clustering status."""
    status_emoji = {
        "not_started": "⚪",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
    }
    emoji = status_emoji.get(run.status, "❓")
    
    st.markdown(f"**Stato:** {emoji} `{run.status}`")
    
    if run.completed_at:
        timestamp_str = run.completed_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        st.markdown(f"**Completato:** {timestamp_str}")
    elif run.started_at:
        timestamp_str = run.started_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        st.markdown(f"**Avviato:** {timestamp_str}")
    
    if run.duration_seconds:
        duration_min = run.duration_seconds / 60.0
        st.markdown(f"**Durata:** {duration_min:.1f} min")
    
    # Parametri
    if run.parameters:
        st.markdown(f"**Parametri:** `min_cluster_size={run.parameters.get('min_cluster_size')}`, "
                   f"`min_samples={run.parameters.get('min_samples')}`")
    
    if artifacts_exist and run.artifacts:
        st.markdown(f"**Artefatti:** {len(run.artifacts)} file(s)")
        with st.expander("Mostra path artefatti"):
            for artifact in run.artifacts:
                st.code(str(artifact), language="text")
    elif run.status == "completed" and not artifacts_exist:
        st.warning("⚠️ Artefatti non trovati")
    
    if run.error_message:
        st.error(f"**Errore:** {run.error_message}")


def _start_hdbscan_clustering(manager: TrainingManager, min_cluster_size: int, min_samples: int, metric: str = "cosine") -> None:
    """Avvia clustering HDBSCAN in background."""
    st.info("🔄 Avvio clustering HDBSCAN in background...")
    
    command = build_training_command(
        "hdbscan",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    parameters = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "metric": metric,
    }
    
    try:
        process = manager.execute_training_async("hdbscan", command, parameters)
        st.success(f"✅ Clustering HDBSCAN avviato (PID={process.pid}). Aggiorna la pagina per vedere lo stato.")
        st.info("💡 Il clustering continua in background. Ricarica la pagina tra qualche minuto per vedere i risultati.")
    except Exception as e:
        st.error(f"❌ Errore avvio clustering HDBSCAN: {e}")


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


def _compute_umap_projection(
    embeddings_df: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean"
) -> pd.DataFrame:
    """
    Compute UMAP 2D projection from embeddings with configurable parameters.
    
    Args:
        embeddings_df: DataFrame with cliente column + embedding features
        n_neighbors: Number of neighbors for manifold approximation (5-50, default 15)
        min_dist: Minimum distance between points in 2D (0.0-1.0, default 0.1)
        metric: Distance metric (euclidean/manhattan/cosine, default euclidean)
    
    Returns:
        DataFrame with columns: cliente, x, y
    """
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn package required for UMAP projection. Install with: pip install umap-learn")
    
    import umap  # type: ignore
    
    embed_cols = [col for col in embeddings_df.columns if col != "cliente"]
    X = embeddings_df[embed_cols].values
    
 # Configurable UMAP parameters
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric
    )
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
    """
    Render Clusters Explorer page with HDBSCAN visualization and drill-down.
    
    Prompt 2: Enhanced with file existence checks, interactive scatter plot,
    and drill-down to Pattern Card.
    """
    st.title("🔬 Cluster Explorer — HDBSCAN Visualization")
    
    # ========================================================================
    # Prompt 2: File Existence Checks with Fallback
    # ========================================================================
    curated_path = data_root_func() / "curated" if data_root_func and callable(data_root_func) else Path("data/curated")
    
    required_files = {
        "clusters": curated_path / "client_clusters.parquet",
        "embeddings_2d": curated_path / "client_embedding_2d.parquet",
        "profiles": curated_path / "client_profiles.parquet",
    }
    
    missing = [name for name, path in required_files.items() if not path.exists()]
    
    if "clusters" in missing or "embeddings_2d" in missing:
        st.warning("⚠️ **Clustering HDBSCAN non ancora eseguito**")
        st.info("""
**Per visualizzare i cluster:**

1. Vai su **Overview & Wizard** (nella sidebar)
2. Completa **Step 2: Segmenta Clienti (HDBSCAN)**
3. Torna qui per esplorare i risultati

**File necessari:**
- ❌ `client_clusters.parquet` (assegnazioni cliente → cluster)
- ❌ `client_embedding_2d.parquet` (proiezione 2D per visualizzazione)

**Oppure** usa il pannello Azioni HDBSCAN qui sotto per eseguire il clustering direttamente.
        """)
        
        # Show HDBSCAN actions panel for direct execution
        _render_hdbscan_actions_panel()
        return
    
    # ========================================================================
    # Load Data
    # ========================================================================
    embeddings = tables.get("client_embedding_2d", pd.DataFrame())
    clusters = tables.get("client_clusters", pd.DataFrame())
    neighbors = tables.get("client_neighbors", pd.DataFrame())
    summaries = tables.get("cluster_summary", pd.DataFrame())
    embeddings_full = tables.get("client_embeddings", pd.DataFrame())
    
    if embeddings.empty:
        st.error("❌ `client_embedding_2d.parquet` non trovato")
        st.info("Esegui Step 2 (HDBSCAN) nel wizard per generare gli embeddings 2D")
        return
    
    if clusters.empty:
        st.error("❌ `client_clusters.parquet` non trovato")
        st.info("Esegui Step 2 (HDBSCAN) nel wizard per generare i cluster")
        return
    
 # Riquadro Azioni HDBSCAN
    _render_hdbscan_actions_panel()
    
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
        
 # Preview cluster counts with visual metrics
        if HDBSCAN_AVAILABLE and not embeddings_full.empty:
            if st.button("🔄 Anteprima clustering", key="preview_hdbscan"):
                with st.spinner("Calcolo dell'anteprima in corso..."):
                    try:
                        labels, n_clusters, n_noise = _recompute_hdbscan(
                            embeddings_full,
                            min_cluster_size,
                            min_samples
                        )
                        
 # Metrics cards
                        total_clients = len(labels)
                        n_clustered = total_clients - n_noise
                        noise_pct = (n_noise / total_clients * 100) if total_clients > 0 else 0
                        clustered_pct = (n_clustered / total_clients * 100) if total_clients > 0 else 0
                        
                        st.markdown("---")
                        st.markdown("### 📊 Metriche Clustering Preview")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "📊 Cluster Trovati",
                                n_clusters,
                                delta=None,
                                help=(
                                    f"Numero di cluster distinti identificati da HDBSCAN.\n\n"
                                    f"Un cluster è un gruppo di clienti con comportamenti simili. "
                                    f"Se vedi troppi cluster (>10), aumenta `min_cluster_size`."
                                )
                            )
                        
                        with col2:
                            # Color code: noise >20% = bad (red), 10-20% = warning (orange), <10% = good (green)
                            if noise_pct > 20:
                                noise_delta = f"⚠️ Alto ({noise_pct:.1f}%)"
                                noise_delta_color = "off"  # Streamlit shows red for negative delta
                            elif noise_pct > 10:
                                noise_delta = f"⚠️ Moderato ({noise_pct:.1f}%)"
                                noise_delta_color = "off"
                            else:
                                noise_delta = f"✓ Basso ({noise_pct:.1f}%)"
                                noise_delta_color = "normal"
                            
                            st.metric(
                                "⚠️ Noise Points",
                                n_noise,
                                delta=noise_delta,
                                delta_color=noise_delta_color,
                                help=(
                                    f"Clienti **outlier** non assegnati a nessun cluster (label = -1).\n\n"
                                    f"**Interpretazione**:\n"
                                    f"- <10% noise: Ottimo, la maggior parte dei clienti è ben clusterizzata\n"
                                    f"- 10-20% noise: Accettabile, alcuni outlier sono normali\n"
                                    f"- >20% noise: Critico, troppi outlier → riduci `min_cluster_size` o `min_samples`\n\n"
                                    f"**Percentuale corrente**: {noise_pct:.1f}%"
                                )
                            )
                        
                        with col3:
                            st.metric(
                                "✅ Clienti Clustered",
                                f"{clustered_pct:.1f}%",
                                delta=f"{n_clustered}/{total_clients}",
                                delta_color="normal",
                                help=(
                                    f"Percentuale di clienti assegnati a cluster validi (non-noise).\n\n"
                                    f"**Obiettivo**: >80% dei clienti clustered per analisi significativa.\n\n"
                                    f"**Valori correnti**:\n"
                                    f"- Clustered: {n_clustered} clienti ({clustered_pct:.1f}%)\n"
                                    f"- Noise: {n_noise} clienti ({noise_pct:.1f}%)"
                                )
                            )
                        
                        if noise_pct > 30:
                            st.error(
                                f"⚠️ **Attenzione**: Noise troppo alto ({noise_pct:.1f}%)!\n\n"
                                f"**Suggerimenti**:\n"
                                f"- Riduci `min_cluster_size` da {min_cluster_size} a {max(2, min_cluster_size // 2)}\n"
                                f"- Riduci `min_samples` da {min_samples} a {max(1, min_samples // 2)}\n"
                                f"- Prova metrica 'manhattan' invece di '{st.session_state.get('hdbscan_metric', 'cosine')}'"
                            )
                        elif noise_pct < 5 and n_clusters > 15:
                            st.warning(
                                f"💡 **Suggerimento**: Hai {n_clusters} cluster con solo {noise_pct:.1f}% noise.\n\n"
                                f"Troppi cluster micro-frammentati? Prova ad aumentare `min_cluster_size` a {min_cluster_size + 5}."
                            )
                        else:
                            st.success(f"✅ Configurazione bilanciata: {n_clusters} cluster, {noise_pct:.1f}% noise.")
                        
                        st.markdown("---")
                        
                        # Store in session state for Apply button
                        st.session_state["hdbscan_preview_labels"] = labels
                        st.session_state["hdbscan_preview_params"] = {
                            "min_cluster_size": min_cluster_size,
                            "min_samples": min_samples,
                            "metric": st.session_state.get("hdbscan_metric", "cosine")
                        }
                        st.session_state["hdbscan_preview_metrics"] = {
                            "n_clusters": n_clusters,
                            "n_noise": n_noise,
                            "noise_pct": noise_pct
                        }
                    except Exception as e:
                        st.error(f"❌ Anteprima non riuscita: {e}")
            
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
 # Selettore proiezione PCA/UMAP migliorato
    # ========================================================================
    with st.sidebar.expander("📊 Metodo di proiezione 2D", expanded=False):
        st.markdown(
            "**Scegli algoritmo** per ridurre embedding ad alta dimensione (es. 768D) a 2D per visualizzazione:\n\n"
            "- **PCA (Principal Component Analysis)**: Veloce, lineare, deterministico\n"
            "- **UMAP (Uniform Manifold Approximation)**: Lento, non-lineare, preserva struttura locale\n\n"
            "💡 **Quando usare**:\n"
            "- **PCA**: Preview rapide, dataset >500 clienti, prima esplorazione\n"
            "- **UMAP**: Analisi dettagliata, dataset <500 clienti, publication-quality viz"
        )
        
        # Disable UMAP if not installed
        if not UMAP_AVAILABLE:
            st.warning(
                "⚠️ **UMAP non disponibile**\n\n"
                "UMAP non è installato. Puoi usare solo PCA.\n\n"
                "**Per installare UMAP**:\n"
                "```bash\n"
                "pip install umap-learn\n"
                "```\n\n"
                "UMAP offre visualizzazioni migliori preservando **struttura locale** dei dati, "
                "ma richiede più tempo di calcolo."
            )
            umap_option_label = "UMAP (non installato)"
            umap_disabled = True
        else:
            umap_option_label = "UMAP (qualità)"
            umap_disabled = False
        
        # Radio button with conditional disable
        if umap_disabled:
            projection_method = st.radio(
                "🎨 Algoritmo proiezione",
                options=["PCA (veloce)"],
                index=0,
                key="projection_method_toggle",
                help="UMAP disabilitato (non installato). Solo PCA disponibile."
            )
        else:
            projection_method = st.radio(
                "🎨 Algoritmo proiezione",
                options=["PCA (veloce)", umap_option_label],
                index=0,
                key="projection_method_toggle",
                help=(
                    "**PCA vs UMAP**:\n\n"
                    "| Criterio | PCA | UMAP |\n"
                    "|----------|-----|------|\n"
                    "| Velocità | ⚡ Veloce (~1s) | 🐢 Lento (~30s) |\n"
                    "| Qualità | 📊 Buona (lineare) | 🎨 Eccellente (non-lineare) |\n"
                    "| Deterministico | ✅ Sempre uguale | ❌ Varia con seed |\n"
                    "| Preserva distanze globali | ✅ Sì | ❌ No |\n"
                    "| Preserva struttura locale | ❌ No | ✅ Sì |\n\n"
                    "💡 **Regola pratica**: Usa PCA per esplorazione rapida, UMAP per analisi finale."
                )
            )
        
        # UMAP parameters (only if UMAP selected and available)
        if projection_method.startswith("UMAP") and UMAP_AVAILABLE and not embeddings_full.empty:
            st.markdown("---")
            st.markdown("### ⚙️ Parametri UMAP")
            
            n_neighbors = st.slider(
                "🏘️ Vicini (n_neighbors)",
                min_value=5,
                max_value=50,
                value=15,
                step=5,
                key="umap_n_neighbors",
                help=(
                    "**n_neighbors**: Numero di vicini per costruire grafo locale.\n\n"
                    "📘 **Business**: Controlla quanto UMAP enfatizza struttura **locale vs globale**.\n\n"
                    "- **5-10**: Focus su dettagli locali, molti micro-cluster\n"
                    "- **15**: Bilanciato (default raccomandato)\n"
                    "- **30-50**: Focus su struttura globale, cluster più grandi\n\n"
                    "💡 **Regola**: Usa 15 come default, aumenta se vedi troppa frammentazione."
                ),
            )
            
            min_dist = st.slider(
                "📏 Distanza minima (min_dist)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                key="umap_min_dist",
                help=(
                    "**min_dist**: Distanza minima tra punti nella proiezione 2D.\n\n"
                    "📘 **Business**: Controlla quanto i punti possono \"compattarsi\".\n\n"
                    "- **0.0**: Massima compattezza, cluster densi e separati (migliore per clustering)\n"
                    "- **0.1**: Bilanciato (default raccomandato)\n"
                    "- **0.5-1.0**: Punti più dispersi, topologia continua (migliore per viz)\n\n"
                    "💡 **Regola**: Usa 0.1 per clustering, 0.3-0.5 per visualizzazione esplorativa."
                ),
            )
            
            umap_metric = st.selectbox(
                "📐 Metrica UMAP",
                options=["euclidean", "manhattan", "cosine"],
                index=2,  # Default cosine
                key="umap_metric",
                help="Stessa logica della metrica HDBSCAN. **Cosine** raccomandato per embeddings.",
            )
            
            if st.button("🔄 Calcola UMAP", key="compute_umap"):
                with st.spinner("Calcolo della proiezione UMAP in corso (può richiedere 30-60 sec)..."):
                    try:
                        umap_df = _compute_umap_projection(
                            embeddings_full,
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            metric=umap_metric
                        )
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
        
        # Prompt 2: Scatter plot click interaction hint
        st.caption("💡 **Tip**: Hover sopra un punto per vedere dettagli cliente. Usa la lista clienti sotto per drill-down.")

    # ========================================================================
    # Prompt 2: Client List with Drill-Down Buttons
    # ========================================================================
    st.subheader("📋 Esplora Clienti per Cluster")
    
    if clusters.empty:
        st.info("Tabella `client_clusters.parquet` non disponibile.")
    elif selected_cluster is None:
        st.info("💡 Seleziona un cluster nella sidebar per vedere i clienti associati.")
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
            
            clients_in_cluster = display_df["cliente"].tolist()
            st.write(f"**{len(clients_in_cluster)} clienti** in {_format_cluster_label(int(selected_cluster))}")
            
            # Prompt 2: Paginated list with drill-down buttons
            clients_per_page = 20
            total_pages = max(1, (len(clients_in_cluster) + clients_per_page - 1) // clients_per_page)
            
            page_col, summary_col = st.columns([0.3, 0.7])
            
            with page_col:
                page = st.number_input(
                    "Pagina",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    key="cluster_clients_page",
                    help=f"Mostra {clients_per_page} clienti per pagina"
                )
            
            with summary_col:
                st.info(f"📄 Pagina {page} di {total_pages} ({len(clients_in_cluster)} clienti totali)")
            
            start_idx = (page - 1) * clients_per_page
            end_idx = min(start_idx + clients_per_page, len(clients_in_cluster))
            
            # Render clients with drill-down buttons
            st.markdown("---")
            for idx, cliente in enumerate(clients_in_cluster[start_idx:end_idx], start=start_idx + 1):
                col1, col2, col3 = st.columns([0.1, 0.6, 0.3])
                
                with col1:
                    st.caption(f"#{idx}")
                
                with col2:
                    # Get membership probability for this client
                    client_row = display_df[display_df["cliente"] == cliente]
                    if not client_row.empty:
                        prob = client_row.iloc[0]["membership_prob"]
                        prob_badge = f"🎯 {prob:.2f}" if not np.isnan(prob) else "n.d."
                    else:
                        prob_badge = "n.d."
                    
                    st.markdown(f"**{cliente}** `{prob_badge}`")
                
                with col3:
                    # Prompt 2 + Prompt 4: Drill-down button to Pattern Card with source tracking
                    if st.button("🔍 Dettagli", key=f"drill_{cliente}_{selected_cluster}"):
                        st.session_state["drill_cliente"] = cliente
                        st.session_state["active_page"] = "Pattern Card"
                        st.session_state["drill_source_page"] = "Clusters"  # Prompt 4: Track source
                        st.rerun()
            
            st.markdown("---")
            
            # Quick jump selector (alternative to pagination)
            with st.expander("⚡ Vai rapidamente a un cliente specifico"):
                cluster_client_choice = st.selectbox(
                    "Seleziona cliente",
                    display_df["cliente"].tolist(),
                    key="cluster_page_client_select",
                )
                if st.button("📊 Apri Pattern Card", key="cluster_page_open_pattern_card"):
                    st.session_state["drill_cliente"] = cluster_client_choice
                    st.session_state["active_page"] = "Pattern Card"
                    st.session_state["drill_source_page"] = "Clusters"  # Prompt 4: Track source
                    st.rerun()

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
    
    # ========================================================================
    # Export & Deep-link
    # ========================================================================
    st.divider()
    st.markdown("### 📤 Esporta & Condividi")
    
    export_cols = st.columns(3)
    
    with export_cols[0]:
        # Export cluster assignments CSV
        if not clusters.empty:
            csv_bytes = export_dataframe_to_csv(clusters, "client_clusters.csv")
            st.download_button(
                label="📥 Scarica Cluster CSV",
                data=csv_bytes,
                file_name=generate_export_filename("client_clusters", "csv"),
                mime="text/csv",
                help="Esporta assegnazioni cliente → cluster con label"
            )
    
    with export_cols[1]:
        # Export cluster summary CSV
        if not summaries.empty:
            csv_bytes_summary = export_dataframe_to_csv(summaries, "cluster_summary.csv")
            st.download_button(
                label="📥 Scarica Summary CSV",
                data=csv_bytes_summary,
                file_name=generate_export_filename("cluster_summary", "csv"),
                mime="text/csv",
                help="Esporta metriche aggregate per cluster"
            )
    
    with export_cols[2]:
        # Deep-link
        deeplink_url = generate_deeplink(
            page="Clusters",
            filters={"selected_client": selected_client},
        )
        if st.button("📋 Genera Deep-Link", key="deeplink_clusters"):
            st.code(deeplink_url, language=None)
            st.caption("⚠️ Copia manualmente l'URL per condividere questa vista.")
    
    # Footer
    st.divider()
    st.caption("🔬 Cluster Explorer | Dati: client_clusters, cluster_summary, client_embedding_2d")
    st.caption("📥 CSV export disponibili | 🔗 Deep-link mantiene cliente selezionato")
