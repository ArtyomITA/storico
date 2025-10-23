"""Streamlit dashboard for visualising client payment patterns."""
from __future__ import annotations

import functools
import importlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.pages.clusters import render_clusters_page
from app.pages.kpi_dashboard import render_kpi_dashboard_page

from src.analytics.pattern_detectors import POLICY_NAMES, infer_pattern_label
from src.report.nlg_patterns import render_pattern_summary
from src.utils.logging_setup import get_current_log_file, log_timed

st.set_page_config(page_title="Pattern Card", page_icon="📊", layout="wide")

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
CURATED_DIR = BASE_DIR / "data" / "curated"
MODELS_DIR = BASE_DIR / "models"
CATBOOST_MODEL_PATH = MODELS_DIR / "catboost.cbm"
CALIBRATOR_PATH = MODELS_DIR / "calibrator.pkl"
AUTOGLUON_DIR = MODELS_DIR / "autogluon"
CHAMPION_PATH = MODELS_DIR / "champion.json"

PERIODICITY_TARGETS = {
    "Settimanale": 7,
    "Quindicinale": 14,
    "Mensile": 30,
}
PERIODICITY_COLUMNS = [
    "periodicity_weekly",
    "periodicity_biweekly",
    "periodicity_monthly",
]

PAGE_OPTIONS = ("Pattern Card", "KPI Dashboard", "Data Explorer", "Clusters")


@st.cache_data(show_spinner=False)
def _load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _section_tracker(metrics: Dict[str, Any], stage_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    sections: List[Dict[str, Any]] = metrics.setdefault("sections", [])

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        timed_func = log_timed(stage_name)(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = timed_func(*args, **kwargs)
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            sections.append({"name": stage_name, "duration_ms": duration_ms})
            return result

        return wrapper

    return decorator


def _record_dataframe(metrics: Dict[str, Any], name: str, df: Optional[pd.DataFrame]) -> None:
    if df is None or not isinstance(df, pd.DataFrame):
        return
    rows, cols = df.shape
    try:
        memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    except Exception:  # pragma: no cover - defensive
        memory_mb = float("nan")
    frames: Dict[str, Dict[str, Any]] = metrics.setdefault("dataframes", {})
    frames[name] = {
        "name": name,
        "rows": int(rows),
        "columns": int(cols),
        "memory_mb": memory_mb,
    }


def _load_warning_log_records(max_items: int = 3) -> Dict[str, Any]:
    log_path = get_current_log_file()
    if log_path is None or not log_path.exists():
        candidate_logs = sorted((BASE_DIR / "logs").glob("run_*.jsonl"), reverse=True)
        log_path = candidate_logs[0] if candidate_logs else None
    if log_path is None or not log_path.exists():
        return {"first": [], "last": [], "path": None}

    warnings_errors: List[Dict[str, Any]] = []
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:  # pragma: no cover - malformed line
                    continue
                if payload.get("level") in {"WARNING", "ERROR"}:
                    warnings_errors.append(payload)
    except FileNotFoundError:  # pragma: no cover - race condition
        return {"first": [], "last": [], "path": None}

    return {
        "first": warnings_errors[:max_items],
        "last": warnings_errors[-max_items:] if warnings_errors else [],
        "path": str(log_path),
    }


def _load_champion_metadata() -> Dict[str, Any]:
    if not CHAMPION_PATH.exists():
        return {}
    try:
        with CHAMPION_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:  # pragma: no cover - malformed json
        return {}

    seed_candidates = [
        data.get("seed"),
        data.get("train_seed"),
        data.get("metadata", {}).get("seed"),
        data.get("config", {}).get("seed"),
    ]
    split_candidates = [
        data.get("split"),
        data.get("split_params"),
        data.get("metadata", {}).get("split"),
        data.get("config", {}).get("split"),
    ]

    seed_value = next((value for value in seed_candidates if value is not None), None)
    split_value = next((value for value in split_candidates if value is not None), None)

    metadata = {
        "model": data.get("model"),
        "trained_at": data.get("trained_at"),
        "seed": seed_value,
        "split": split_value,
    }
    return metadata


def _render_developer_panel(metrics: Dict[str, Any]) -> None:
    st.markdown("---")
    st.header("Developer diagnostics")

    sections = metrics.get("sections", [])
    if sections:
        st.subheader("Section latencies (ms)")
        st.dataframe(pd.DataFrame(sections).sort_values("duration_ms", ascending=False), use_container_width=True)
    else:
        st.info("Nessuna latenza disponibile per questa sessione.")

    df_map: Dict[str, Dict[str, Any]] = metrics.get("dataframes", {})
    if df_map:
        st.subheader("Dataframe footprint")
        st.dataframe(
            pd.DataFrame(df_map.values()).sort_values("memory_mb", ascending=False),
            use_container_width=True,
        )

    log_records = metrics.get("logs")
    if log_records:
        st.subheader("Warnings / Errors dal log JSON")
        log_path = log_records.get("path")
        if log_path:
            st.caption(f"File log: {log_path}")
        cols = st.columns(2)
        cols[0].markdown("**Primi 3**")
        if log_records.get("first"):
            cols[0].json(log_records["first"])
        else:
            cols[0].info("Nessun warning/error iniziale registrato.")
        cols[1].markdown("**Ultimi 3**")
        if log_records.get("last"):
            cols[1].json(log_records["last"])
        else:
            cols[1].info("Nessun warning/error recente registrato.")

    champion_meta = metrics.get("champion")
    st.subheader("Champion model metadata")
    if champion_meta:
        st.json(champion_meta)
    else:
        st.info("Nessun file models/champion.json disponibile o privo di seed/split.")


@st.cache_data(show_spinner=False)
def load_curated_tables() -> Tuple[Dict[str, pd.DataFrame], list[str]]:
    """Load curated parquet tables required for the dashboard."""
    base_dir = CURATED_DIR
    required = {
        "events": "events.parquet",
        "event_items": "event_items.parquet",
        "client_profiles": "client_profiles.parquet",
        "client_pattern_scores": "client_pattern_scores.parquet",
    }
    optional = {
        "pattern_table": "pattern_table.parquet",
        "client_rules_accuracy": "client_rules_accuracy.parquet",
        "client_model_metrics_catboost": "client_model_metrics_catboost.parquet",
        "client_model_metrics_autogluon": "client_model_metrics_autogluon.parquet",
        "client_embedding_2d": "client_embedding_2d.parquet",
        "client_neighbors": "client_neighbors.parquet",
        "client_clusters": "client_clusters.parquet",
        "cluster_summary": "cluster_summary.parquet",
        "line_items": "line_items.parquet",
    }

    tables: Dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for key, filename in required.items():
        path = base_dir / filename
        if path.exists():
            tables[key] = _load_parquet(path)
        else:
            tables[key] = pd.DataFrame()
            missing.append(filename)

    for key, filename in optional.items():
        path = base_dir / filename
        if path.exists():
            tables[key] = _load_parquet(path)
        else:
            tables[key] = pd.DataFrame()

    if "client_pattern_scores" in tables:
        scores = tables["client_pattern_scores"]
        # Backward compatibility: ensure stability index column is named correctly.
        if "0" in scores.columns and "stability_index" not in scores.columns:
            scores = scores.rename(columns={"0": "stability_index"})
            tables["client_pattern_scores"] = scores

    return tables, missing


def enrich_features_for_models(
    features: pd.DataFrame,
    event_items: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Align feature store with the training pipeline for model inference."""
    df = features.copy()
    if df.empty:
        return df

    df["cliente"] = df["cliente"].astype("string")
    df["numero_documento"] = df["numero_documento"].astype("string")

    if not events.empty and "event_key" in df.columns:
        event_dates = (
            events[["event_key", "data_pareggio"]]
            .drop_duplicates()
            .assign(data_pareggio=lambda d: pd.to_datetime(d["data_pareggio"], errors="coerce"))
        )
        df = df.merge(event_dates, on="event_key", how="left")

    if not event_items.empty:
        join_cols = ["cliente", "event_key", "numero_documento"]
        available = [col for col in join_cols if col in event_items.columns]
        if len(available) == len(join_cols):
            enrich = event_items[join_cols + ["periodo_contabile"]].drop_duplicates()
            df = df.merge(enrich, on=join_cols, how="left")

    if "periodo_contabile" not in df.columns:
        df["periodo_contabile"] = pd.NA

    for col in ("tipo_documento", "chiave_contabile", "divisa", "periodo_contabile"):
        if col not in df.columns:
            df[col] = "UNKNOWN"
        df[col] = df[col].fillna("UNKNOWN").astype(str)

    df["macro_tipo"] = df["tipo_documento"].astype(str).str[:1].fillna("U")
    if "imp_div_int" in df.columns:
        df["abs_amount"] = df["imp_div_int"].astype(float).fillna(0.0).abs()
    else:
        df["abs_amount"] = 0.0
    if "age_days" in df.columns:
        df["age_days"] = df["age_days"].astype(float).fillna(0.0)
    else:
        df["age_days"] = 0.0
    if "same_period" in df.columns:
        df["same_period"] = df["same_period"].fillna(0).astype(float)
    else:
        df["same_period"] = 0.0

    return df


def select_catboost_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[int]]:
    target = df["label"].astype(int) if "label" in df.columns else pd.Series(dtype=int)
    drop_cols = ["label", "data_pareggio", "cliente", "event_key", "numero_documento"]
    feature_df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    categorical_cols = [
        col
        for col in ["tipo_documento", "chiave_contabile", "divisa", "periodo_contabile", "macro_tipo"]
        if col in feature_df.columns
    ]
    cat_indices_array = feature_df.columns.get_indexer(categorical_cols)
    cat_indices = [int(idx) for idx in cat_indices_array if idx >= 0]
    return feature_df, target, categorical_cols, cat_indices


def select_autogluon_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    target = df["label"].astype(int) if "label" in df.columns else pd.Series(dtype=int)
    drop_cols = ["label", "data_pareggio", "cliente", "event_key", "numero_documento"]
    feature_df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    return feature_df, target


@st.cache_data(show_spinner=False)
def load_feature_repository() -> Tuple[pd.DataFrame, List[str], List[int], List[str]]:
    features_path = CURATED_DIR / "train_features.parquet"
    if not features_path.exists():
        LOGGER.warning("Feature store not found at %s", features_path)
        return pd.DataFrame(), [], [], []

    features = pd.read_parquet(features_path)
    event_items_path = CURATED_DIR / "event_items.parquet"
    events_path = CURATED_DIR / "events.parquet"

    event_items = pd.read_parquet(event_items_path) if event_items_path.exists() else pd.DataFrame()
    events = pd.read_parquet(events_path) if events_path.exists() else pd.DataFrame()

    enriched = enrich_features_for_models(features, event_items, events)
    cat_matrix, _, _, cat_indices = select_catboost_features(enriched)
    auto_matrix, _ = select_autogluon_features(enriched)
    return enriched, list(cat_matrix.columns), cat_indices, list(auto_matrix.columns)


@st.cache_resource(show_spinner=False)
def load_catboost_bundle() -> Tuple[Optional[Any], Optional[Any]]:
    try:
        from catboost import CatBoostClassifier  # type: ignore
    except ImportError:
        LOGGER.warning("CatBoost non disponibile nel runtime Streamlit.")
        return None, None

    if not CATBOOST_MODEL_PATH.exists():
        LOGGER.info("CatBoost model path %s not found.", CATBOOST_MODEL_PATH)
        return None, None

    model = CatBoostClassifier()
    model.load_model(str(CATBOOST_MODEL_PATH))

    calibrator: Optional[Any] = None
    if CALIBRATOR_PATH.exists():
        try:
            with open(CALIBRATOR_PATH, "rb") as f:
                calibrator = pickle.load(f)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Impossibile caricare il calibratore: %s", exc)
            calibrator = None

    return model, calibrator


@st.cache_resource(show_spinner=False)
def load_autogluon_predictor() -> Optional[Any]:
    try:
        from autogluon.tabular import TabularPredictor  # type: ignore
    except ImportError:
        LOGGER.info("AutoGluon non disponibile nel runtime Streamlit.")
        return None

    if not AUTOGLUON_DIR.exists():
        LOGGER.info("AutoGluon predictor folder %s not found.", AUTOGLUON_DIR)
        return None

    return TabularPredictor.load(str(AUTOGLUON_DIR))


def safe_calibrate(calibrator: Optional[Any], values: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return values
    try:
        return np.asarray(calibrator.predict(values))
    except Exception:  # pragma: no cover - fallback
        LOGGER.warning("Calibratore non applicabile; uso probabilità grezze.")
        return values


def recommend_policy_docs(event_df: pd.DataFrame, policy: Optional[str], k: int) -> List[str]:
    if event_df.empty or not policy:
        return []
    df = event_df.copy()
    if "imp_div_int" in df.columns:
        df["abs_amount"] = df["imp_div_int"].fillna(0.0).abs()
    else:
        df["abs_amount"] = 0.0
    if "age_days" in df.columns:
        df["age_days"] = df["age_days"].fillna(0.0)
    else:
        df["age_days"] = 0.0
    if "same_period" in df.columns:
        df["same_period"] = df["same_period"].fillna(0.0)
    else:
        df["same_period"] = 0.0

    if policy == "same_period":
        ordered = df.sort_values(["same_period", "age_days", "abs_amount"], ascending=[False, False, False])
    elif policy == "fifo":
        ordered = df.sort_values(["age_days", "abs_amount"], ascending=[False, False])
    elif policy == "lifo":
        ordered = df.sort_values(["age_days", "abs_amount"], ascending=[True, False])
    elif policy == "largest":
        ordered = df.sort_values(["abs_amount", "age_days"], ascending=[False, False])
    elif policy == "smallest":
        ordered = df.sort_values(["abs_amount", "age_days"], ascending=[True, False])
    else:
        return []

    top = ordered.head(max(k, 1))["numero_documento"].astype(str)
    return top.tolist()


def top_docs_from_model(
    event_df: pd.DataFrame,
    feature_columns: Sequence[str],
    cat_indices: Sequence[int],
    autop_columns: Sequence[str],
    model_choice: str,
    k: int,
) -> Tuple[List[str], List[Tuple[str, List[Tuple[str, float]]]]]:
    event_df = event_df.reset_index(drop=True).copy()
    if event_df.empty or not feature_columns:
        return [], []

    event_df["numero_documento"] = event_df["numero_documento"].astype(str)
    event_df["row_id"] = np.arange(len(event_df))

    catboost_model, calibrator = load_catboost_bundle() if model_choice == "catboost" else (None, None)
    autogluon_predictor = load_autogluon_predictor() if model_choice == "autogluon" else None

    if model_choice == "catboost" and catboost_model is not None:
        from catboost import Pool  # type: ignore

        model_input = event_df[feature_columns]
        pool = Pool(model_input, cat_features=list(cat_indices) if cat_indices else None)
        probs = catboost_model.predict_proba(pool)[:, 1]
        probs = safe_calibrate(calibrator, probs)
        try:
            shap_values = catboost_model.get_feature_importance(pool, type="ShapValues")
        except Exception:  # pragma: no cover - shap optional
            shap_values = None
        event_df["model_prob"] = probs
        ranked = event_df.sort_values("model_prob", ascending=False)
        top_rows = ranked.head(max(k, 1))
        docs = top_rows["numero_documento"].tolist()
        shap_details: List[Tuple[str, List[Tuple[str, float]]]] = []
        if shap_values is not None:
            for _, row in top_rows.head(2).iterrows():
                idx = int(row["row_id"])
                shap_row = shap_values[idx, :-1]
                ranked_feats = sorted(
                    zip(feature_columns, shap_row),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )[:3]
                shap_details.append((row["numero_documento"], [(name, float(val)) for name, val in ranked_feats]))
        return docs, shap_details

    if model_choice == "autogluon" and autogluon_predictor is not None and autop_columns:
        model_input = event_df[autop_columns]
        proba_df = autogluon_predictor.predict_proba(model_input)
        if isinstance(proba_df, pd.DataFrame):
            if 1 in proba_df.columns:
                probs = proba_df[1].to_numpy()
            else:
                probs = proba_df.iloc[:, -1].to_numpy()
        else:  # pragma: no cover - defensive
            probs = np.asarray(proba_df)
        event_df["model_prob"] = probs
        ranked = event_df.sort_values("model_prob", ascending=False)
        top_rows = ranked.head(max(k, 1))
        docs = top_rows["numero_documento"].tolist()
        return docs, []

    return [], []


def build_event_comparison_table(
    actual_docs: List[str],
    policy_docs: List[str],
    model_docs: List[str],
    policy_label: str,
    model_label: str,
) -> pd.DataFrame:
    max_len = max(len(actual_docs), len(policy_docs), len(model_docs), 1)
    data = {
        "Fatture reali": actual_docs + [""] * (max_len - len(actual_docs)),
        f"Top-k {policy_label}": policy_docs + [""] * (max_len - len(policy_docs)),
        f"Top-k {model_label}": model_docs + [""] * (max_len - len(model_docs)),
    }
    return pd.DataFrame(data)


def format_percentage(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n.d."
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n.d."


def format_decimal(value: Optional[float], digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n.d."
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n.d."


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return float("nan")


def compute_acf(values: np.ndarray, max_lag: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    if values.size < 2:
        return np.array([]), np.array([])

    lags: list[int] = []
    coeffs: list[float] = []
    for lag in range(1, min(max_lag, values.size - 1) + 1):
        x = values[:-lag]
        y = values[lag:]
        if x.size < 2:
            break
        if np.std(x) == 0 or np.std(y) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
        lags.append(lag)
        coeffs.append(corr)
    return np.asarray(lags), np.asarray(coeffs)
def plot_histogram_with_markers(values: Iterable[float], title: str) -> go.Figure:
    value_list = list(values)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=value_list,
            nbinsx=min(30, max(5, int(len(value_list) * 0.5)) or 10),
            marker_color="#4C78A8",
            opacity=0.8,
            hovertemplate="Gap: %{x:.0f} giorni<br>Conteggio: %{y}<extra></extra>",
        )
    )
    for label, marker in PERIODICITY_TARGETS.items():
        fig.add_vline(
            x=marker,
            line=dict(color="#E45756", dash="dash"),
            annotation_text=label,
            annotation_position="top",
        )
    fig.update_layout(
        title=title,
        bargap=0.05,
        xaxis_title="Giorni tra eventi",
        yaxis_title="Frequenza",
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="closest",
    )
    return fig


def plot_acf(values: np.ndarray) -> go.Figure:
    lags, coeffs = compute_acf(values)
    fig = go.Figure()
    if lags.size:
        fig.add_trace(
            go.Bar(
                x=lags,
                y=coeffs,
                marker_color="#F58518",
                hovertemplate="Lag: %{x}<br>Coeff: %{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Autocorrelazione (lag in eventi)",
        xaxis_title="Lag",
        yaxis_title="Coeff.",
        yaxis=dict(range=[-1, 1]),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="closest",
    )
    return fig


def plot_periodicity_gauges(scores: pd.Series) -> Tuple[go.Figure, go.Figure, go.Figure]:
    figs: list[go.Figure] = []
    labels = ["Settimanale", "Quindicinale", "Mensile"]
    for label, column in zip(labels, PERIODICITY_COLUMNS):
        value = float(scores.get(column, 0.0))
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": label},
                gauge={
                    "axis": {"range": [0, 1], "tickformat": ".0%"},
                    "bar": {"color": "#4C78A8"},
                    "steps": [
                        {"range": [0, 0.4], "color": "#E0ECF4"},
                        {"range": [0.4, 0.7], "color": "#9ECAE1"},
                        {"range": [0.7, 1.0], "color": "#3182BD"},
                    ],
                },
                number={"valueformat": ".2f"},
            )
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=220)
        figs.append(fig)
    return figs[0], figs[1], figs[2]


def plot_calendar_heatmap(events: pd.DataFrame) -> go.Figure:
    if events.empty:
        return go.Figure()
    tmp = events.dropna(subset=["data_pareggio"]).copy()
    if tmp.empty:
        return go.Figure()
    tmp["dow"] = tmp["data_pareggio"].dt.weekday
    tmp["month"] = tmp["data_pareggio"].dt.month

    pivot = tmp.pivot_table(
        index="dow",
        columns="month",
        values="event_key",
        aggfunc="count",
        fill_value=0,
    ).sort_index()

    dow_labels = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
    heat = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=[f"M{int(c):02d}" for c in pivot.columns],
            y=[dow_labels[int(i)] for i in pivot.index],
            colorscale="Blues",
            colorbar=dict(title="# eventi"),
            hovertemplate="Giorno: %{y}<br>Mese: %{x}<br># eventi: %{z}<extra></extra>",
        )
    )
    heat.update_layout(margin=dict(l=40, r=40, t=40, b=40), hovermode="closest")
    return heat


def plot_policy_bars(scores: pd.Series) -> go.Figure:
    policies = [float(scores.get(f"{policy}_score", 0.0)) for policy in POLICY_NAMES]
    fig = go.Figure(
        go.Bar(
            x=[policy.upper() for policy in POLICY_NAMES],
            y=policies,
            marker_color="#59A14F",
            hovertemplate="Policy: %{x}<br>Recall medio: %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Recall medio per policy",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="closest",
    )
    return fig


def plot_cardinality_histogram(events: pd.DataFrame) -> go.Figure:
    if events.empty:
        return go.Figure()
    k_values = events["n_righe_evento"].dropna()
    fig = go.Figure(
        go.Histogram(
            x=k_values,
            nbinsx=min(15, k_values.nunique()),
            marker_color="#8CD17D",
            hovertemplate="Cardinalità: %{x}<br>Frequenza: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Distribuzione cardinalità evento",
        xaxis_title="Numero righe (k)",
        yaxis_title="Frequenza",
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="closest",
    )
    return fig


def plot_frequency_bar(series: pd.Series, title: str, color: str = "#B6992D") -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=series.index.tolist(),
            y=series.values.tolist(),
            marker_color=color,
            hovertemplate="Categoria: %{x}<br>Frequenza: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="Frequenza",
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="closest",
    )
    return fig


def derive_macro_tipo(items: pd.DataFrame) -> pd.Series:
    if "macro_tipo" in items.columns:
        macro = items["macro_tipo"].fillna("Sconosciuto").astype(str)
    else:
        macro = items["tipo_documento"].astype(str).str[:1].replace({"": "Sconosciuto", "nan": "Sconosciuto"})
        macro = macro.fillna("Sconosciuto")
    return macro


def run_holdout_validation(cliente: str) -> Optional[float]:
    module_candidates = [
        "src.analytics.rules_backtest",
        "src.analytics.validation",
        "src.analytics.holdout",
        "src.analytics.prompt7",
    ]
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        candidates = [
            "evaluate_holdout_accuracy",
            "validate_holdout_for_client",
            "score_holdout_for_client",
            "run_holdout_validation",
        ]
        for attr in candidates:
            func = getattr(module, attr, None)
            if callable(func):
                try:
                    result = func(cliente)
                except TypeError:
                    # Function signature might differ; try passing keyword.
                    try:
                        result = func(cliente=cliente)
                    except Exception:  # pragma: no cover - defensive
                        continue
                except Exception:
                    continue
                if isinstance(result, dict):
                    for key in ("accuracy", "acc", "score"):
                        if key in result and result[key] is not None:
                            return float(result[key])
                    return None
                if result is None:
                    return None
                try:
                    return float(result)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return None
    return None


def render_data_explorer_page(
    tables: Dict[str, pd.DataFrame],
    default_client: str,
    developer_metrics: Dict[str, Any],
) -> None:
    """Render the interactive data explorer with filters and analytics toggles."""

    st.title("Data Explorer")

    line_items = tables.get("line_items", pd.DataFrame())
    event_items = tables.get("event_items", pd.DataFrame())
    base_df = line_items if not line_items.empty else event_items
    if base_df.empty:
        st.info(
            "Caricare `data/curated/line_items.parquet` o `event_items.parquet` per utilizzare il Data Explorer."
        )
        return

    df = base_df.copy()
    df.columns = [str(col) for col in df.columns]
    _record_dataframe(developer_metrics, "data_explorer_source", df)

    if "cliente" not in df.columns:
        st.info("Il dataset selezionato non contiene la colonna `cliente`.")
        return

    df["cliente"] = df["cliente"].fillna("Sconosciuto").astype(str)
    if "tipo_documento" in df.columns:
        df["tipo_documento"] = df["tipo_documento"].fillna("Sconosciuto").astype(str)
    if "chiave_contabile" in df.columns:
        df["chiave_contabile"] = df["chiave_contabile"].fillna("Sconosciuto").astype(str)

    date_columns = [
        column
        for column in ("data_documento", "data_registrazione", "data_pareggio", "data_acquisizione")
        if column in df.columns
    ]
    for column in date_columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    client_options = sorted(df["cliente"].dropna().unique().tolist())
    if not client_options:
        st.info("Nessun cliente disponibile nel dataset selezionato.")
        return
    default_clients = [default_client] if default_client in client_options else client_options[:1]

    tipo_options: List[str] = []
    if "tipo_documento" in df.columns:
        tipo_options = sorted(df["tipo_documento"].dropna().astype(str).unique().tolist())

    chiave_options: List[str] = []
    if "chiave_contabile" in df.columns:
        chiave_options = sorted(df["chiave_contabile"].dropna().astype(str).unique().tolist())

    tracker = functools.partial(_section_tracker, developer_metrics)

    @tracker("UI.DataExplorer.Filters")
    def _apply_filters() -> Dict[str, Any]:
        st.subheader("Filtri")
        filter_cols = st.columns(3)
        with filter_cols[0]:
            selected_clients = st.multiselect(
                "Cliente",
                options=client_options,
                default=default_clients or client_options,
            )

        with filter_cols[1]:
            if tipo_options:
                selected_tipodoc = st.multiselect("Tipo documento", options=tipo_options, default=tipo_options)
            else:
                selected_tipodoc = []

        with filter_cols[2]:
            if chiave_options:
                selected_chiave = st.multiselect("Chiave contabile", options=chiave_options, default=chiave_options)
            else:
                selected_chiave = []

        date_column: Optional[str] = None
        date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
        if date_columns:
            preferred_order = ["data_documento", "data_pareggio", "data_registrazione", "data_acquisizione"]
            date_column = next((col for col in preferred_order if col in date_columns), date_columns[0])
            date_series = df[date_column].dropna()
            if not date_series.empty:
                min_date = pd.Timestamp(date_series.min()).date()
                max_date = pd.Timestamp(date_series.max()).date()
                date_input_value = st.date_input(
                    "Intervallo date",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    format="YYYY-MM-DD",
                )
                if isinstance(date_input_value, tuple):
                    if len(date_input_value) == 2:
                        start_date, end_date = date_input_value
                    elif len(date_input_value) == 1:
                        start_date = date_input_value[0]
                        end_date = date_input_value[0]
                    else:  # pragma: no cover - defensive
                        start_date = min_date
                        end_date = max_date
                else:
                    start_date = date_input_value
                    end_date = date_input_value
                if start_date > end_date:
                    start_date, end_date = end_date, start_date
                date_range = (
                    pd.Timestamp(start_date),
                    pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
                )

        search_text = st.text_input("Ricerca testuale", placeholder="Cerca in tutte le colonne...")

        filtered_local = df.copy()
        client_filter = selected_clients if selected_clients else client_options
        if client_filter:
            filtered_local = filtered_local[filtered_local["cliente"].isin(client_filter)]

        if selected_tipodoc and "tipo_documento" in filtered_local.columns:
            filtered_local = filtered_local[filtered_local["tipo_documento"].isin(selected_tipodoc)]

        if selected_chiave and "chiave_contabile" in filtered_local.columns:
            filtered_local = filtered_local[filtered_local["chiave_contabile"].isin(selected_chiave)]

        if date_range and date_column:
            start_ts, end_ts = date_range
            filtered_local = filtered_local[filtered_local[date_column].between(start_ts, end_ts)]

        if search_text:
            combined = filtered_local.astype(str).agg(" ".join, axis=1)
            search_mask = combined.str.contains(search_text, case=False, na=False)
            filtered_local = filtered_local[search_mask]

        _record_dataframe(developer_metrics, "data_explorer_filtered", filtered_local)

        return {
            "filtered_df": filtered_local,
            "client_filter": client_filter,
            "date_range": date_range,
        }

    filter_payload = _apply_filters()
    filtered_df: pd.DataFrame = filter_payload["filtered_df"]
    client_filter: List[str] = filter_payload["client_filter"]
    date_range = filter_payload["date_range"]

    st.caption(
        f"Righe mostrate: {len(filtered_df):,} su {len(df):,} (aggiornare i filtri per restringere la vista)."
    )

    table_tab, analytics_tab = st.tabs(["Tabella", "Analitiche"])

    with table_tab:
        @_section_tracker(developer_metrics, "UI.DataExplorer.Table")
        def _render_table() -> None:
            st.dataframe(filtered_df, use_container_width=True)
            if not filtered_df.empty:
                csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv_bytes,
                    file_name="data_explorer_view.csv",
                    mime="text/csv",
                )
            else:
                st.info("Nessuna riga corrisponde ai filtri correnti.")

        _render_table()

    with analytics_tab:
        @_section_tracker(developer_metrics, "UI.DataExplorer.Analytics")
        def _render_analytics() -> None:
            st.subheader("Analitiche rapide")
            show_cadence = st.toggle("Mostra sezione cadenza", value=True, key="toggle_cadence_explorer")
            show_policy = st.toggle("Mostra sezione policy", value=True, key="toggle_policy_explorer")
            show_cardinality = st.toggle("Mostra sezione cardinalità", value=True, key="toggle_cardinality_explorer")

            events_df = tables.get("events", pd.DataFrame()).copy()
            if not events_df.empty:
                events_df["cliente"] = events_df["cliente"].astype(str)
                events_df["data_pareggio"] = pd.to_datetime(events_df["data_pareggio"], errors="coerce")

            relevant_events = events_df
            event_keys: np.ndarray = np.array([])
            if "event_key" in filtered_df.columns:
                event_keys = filtered_df["event_key"].dropna().astype(str).unique()
                if event_keys.size and not events_df.empty:
                    relevant_events = events_df[events_df["event_key"].astype(str).isin(event_keys)]
            if event_keys.size == 0 and not events_df.empty:
                relevant_events = events_df[events_df["cliente"].isin(client_filter)]
            if date_range and not relevant_events.empty:
                start_ts, end_ts = date_range
                relevant_events = relevant_events[relevant_events["data_pareggio"].between(start_ts, end_ts)]

            _record_dataframe(developer_metrics, "data_explorer_events", relevant_events)

            pattern_scores_df = tables.get("client_pattern_scores", pd.DataFrame()).copy()
            if not pattern_scores_df.empty:
                pattern_scores_df["cliente"] = pattern_scores_df["cliente"].astype(str)
                pattern_subset = pattern_scores_df[pattern_scores_df["cliente"].isin(client_filter)]
            else:
                pattern_subset = pd.DataFrame()
            _record_dataframe(developer_metrics, "data_explorer_pattern_scores", pattern_subset)

            pattern_table_df = tables.get("pattern_table", pd.DataFrame()).copy()
            if not pattern_table_df.empty:
                pattern_table_df["cliente"] = pattern_table_df["cliente"].astype(str)
                pattern_table_subset = pattern_table_df[pattern_table_df["cliente"].isin(client_filter)]
            else:
                pattern_table_subset = pd.DataFrame()
            _record_dataframe(developer_metrics, "data_explorer_pattern_table", pattern_table_subset)

            if show_cadence:
                st.markdown("### Cadenza")
                if relevant_events.empty:
                    st.info("Nessun evento disponibile per elaborare la cadenza.")
                else:
                    gaps = relevant_events["inter_event_gap_days"].dropna().to_numpy(dtype=float)
                    cadence_cols = st.columns(2)
                    if gaps.size:
                        cadence_cols[0].plotly_chart(
                            plot_histogram_with_markers(gaps, "Distribuzione gap tra eventi"),
                            use_container_width=True,
                        )
                        cadence_cols[1].plotly_chart(
                            plot_acf(gaps),
                            use_container_width=True,
                        )
                    else:
                        st.info("Non sono presenti gap sufficienti per il grafico.")

                    st.plotly_chart(
                        plot_calendar_heatmap(relevant_events),
                        use_container_width=True,
                    )

            if show_policy:
                st.markdown("### Policy")
                if pattern_subset.empty:
                    st.info("Metriche policy non disponibili per i clienti filtrati.")
                else:
                    score_columns = [
                        f"{policy}_score" for policy in POLICY_NAMES if f"{policy}_score" in pattern_subset.columns
                    ]
                    if score_columns:
                        mean_scores = pattern_subset[score_columns].mean()
                        st.plotly_chart(plot_policy_bars(mean_scores), use_container_width=True)
                    else:
                        st.info("Nessuna colonna di score policy disponibile.")
                    metrics_cols = st.columns(3)
                    metrics_cols[0].metric(
                        "Stability index medio",
                        format_decimal(pattern_subset.get("stability_index", pd.Series(dtype=float)).mean()),
                    )
                    metrics_cols[1].metric(
                        "DOW concentration media",
                        format_decimal(pattern_subset.get("dow_concentration", pd.Series(dtype=float)).mean()),
                    )
                    metrics_cols[2].metric(
                        "DOM concentration media",
                        format_decimal(pattern_subset.get("dom_concentration", pd.Series(dtype=float)).mean()),
                    )

            if show_cardinality:
                st.markdown("### Cardinalità")
                if relevant_events.empty:
                    st.info("Nessun evento per calcolare la cardinalità.")
                else:
                    st.plotly_chart(plot_cardinality_histogram(relevant_events), use_container_width=True)
                if not pattern_table_subset.empty:
                    card_cols = st.columns(2)
                    card_cols[0].metric(
                        "k_mode medio",
                        format_decimal(pattern_table_subset.get("k_mode", pd.Series(dtype=float)).mean(), digits=1),
                    )
                    card_cols[1].metric(
                        "k_entropy media",
                        format_decimal(pattern_table_subset.get("k_entropy", pd.Series(dtype=float)).mean()),
                    )

        _render_analytics()


def render_pattern_card(
    selected_client: str,
    events: pd.DataFrame,
    event_items: pd.DataFrame,
    pattern_scores: pd.DataFrame,
    pattern_table_df: pd.DataFrame,
    rules_metrics_df: pd.DataFrame,
    cat_metrics_df: pd.DataFrame,
    auto_metrics_df: pd.DataFrame,
    feature_repository: pd.DataFrame,
    cat_feature_cols: Sequence[str],
    cat_cat_indices: Sequence[int],
    auto_feature_cols: Sequence[str],
    developer_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    metrics = developer_metrics if developer_metrics is not None else {}
    tracker = functools.partial(_section_tracker, metrics)

    client_events = events[events["cliente"] == selected_client].copy()
    client_items = event_items[event_items["cliente"] == selected_client].copy()
    _record_dataframe(metrics, f"pattern_card_events_{selected_client}", client_events)
    _record_dataframe(metrics, f"pattern_card_items_{selected_client}", client_items)
    score_row = (
        pattern_scores[pattern_scores["cliente"] == selected_client].iloc[0]
        if not pattern_scores.empty and selected_client in pattern_scores["cliente"].values
        else pd.Series(dtype=float)
    )

    st.title(f"Pattern Card — Cliente {selected_client}")

    @tracker("UI.PatternCard.Cadence")
    def _render_cadence_section() -> None:
        st.markdown("### Cadenza")
        gaps = client_events["inter_event_gap_days"].dropna().to_numpy(dtype=float)
        col_hist, col_acf = st.columns(2)
        with col_hist:
            if gaps.size:
                col_hist.plotly_chart(
                    plot_histogram_with_markers(gaps, "Distribuzione gap tra eventi"),
                    use_container_width=True,
                )
            else:
                st.info("Non sono presenti gap sufficienti per il grafico.")
        with col_acf:
            if gaps.size:
                col_acf.plotly_chart(plot_acf(gaps), use_container_width=True)
            else:
                st.info("Autocorrelazione non disponibile.")

        gauge_cols = st.columns(3)
        gauge_figs = plot_periodicity_gauges(score_row)
        for container, fig in zip(gauge_cols, gauge_figs):
            container.plotly_chart(fig, use_container_width=True)

        st.markdown("### Calendario")
        if not client_events.empty:
            heatmap = plot_calendar_heatmap(client_events)
            st.plotly_chart(heatmap, use_container_width=True)
        else:
            st.info("Nessun evento disponibile per il calendario.")

        conc_cols = st.columns(2)
        conc_cols[0].metric("DOW concentration", f"{float(score_row.get('dow_concentration', 0.0)):.2f}")
        conc_cols[1].metric("DOM concentration", f"{float(score_row.get('dom_concentration', 0.0)):.2f}")

    _render_cadence_section()

    @tracker("UI.PatternCard.Policy")
    def _render_policy_section() -> None:
        st.markdown("### Policy match")
        if score_row.empty:
            st.info("Metriche policy non disponibili per il cliente selezionato.")
        else:
            st.plotly_chart(plot_policy_bars(score_row), use_container_width=True)

    _render_policy_section()

    @tracker("UI.PatternCard.Cardinality")
    def _render_cardinality_section() -> None:
        st.markdown("### Cardinalità")
        if client_events.empty:
            st.info("Nessun evento per calcolare la cardinalità.")
        else:
            st.plotly_chart(plot_cardinality_histogram(client_events), use_container_width=True)
        card_cols = st.columns(2)
        card_cols[0].metric("k_mode", f"{float(score_row.get('k_mode', 0.0)):.1f}")
        card_cols[1].metric("k_entropy", f"{float(score_row.get('k_entropy', 0.0)):.2f}")

    _render_cardinality_section()

    @tracker("UI.PatternCard.Documenti")
    def _render_document_section() -> None:
        st.markdown("### Documento")
        if client_items.empty:
            st.info("Nessuna riga documento per il cliente.")
        else:
            macro_counts = derive_macro_tipo(client_items).value_counts().head(10)
            tipo_counts = (
                client_items["tipo_documento"].fillna("Sconosciuto").astype(str).value_counts().head(10)
            )
            doc_cols = st.columns(2)
            doc_cols[0].plotly_chart(
                plot_frequency_bar(macro_counts, "Macro-tipologia documenti", "#FF9DA7"),
                use_container_width=True,
            )
            doc_cols[1].plotly_chart(
                plot_frequency_bar(tipo_counts, "Tipo documento", "#EDC948"),
                use_container_width=True,
            )

    _render_document_section()

    st.markdown("### Modello vs Regole")
    pattern_row: Optional[pd.Series] = None
    pattern_row_df = (
        pattern_table_df[pattern_table_df["cliente"] == selected_client]
        if not pattern_table_df.empty
        else pd.DataFrame()
    )
    rules_row_df = (
        rules_metrics_df[rules_metrics_df["cliente"] == selected_client]
        if not rules_metrics_df.empty
        else pd.DataFrame()
    )

    @tracker("UI.PatternCard.ModelVsRules")
    def _render_model_vs_rules() -> Optional[pd.Series]:
        if pattern_row_df.empty and rules_row_df.empty:
            st.info("Metriche di confronto non disponibili per il cliente selezionato.")
            return None

        pattern_row_local = pattern_row_df.iloc[0] if not pattern_row_df.empty else None
        if pattern_row_local is not None:
            _record_dataframe(metrics, f"pattern_card_pattern_table_{selected_client}", pattern_row_df)
            meta_cols = st.columns(4)
            meta_cols[0].metric("Pattern label", str(pattern_row_local.get("pattern_label", "n.d.")))
            meta_cols[1].metric("Confidence", format_decimal(safe_float(pattern_row_local.get("confidence"))))
            meta_cols[2].metric("k_entropy", format_decimal(safe_float(pattern_row_local.get("k_entropy"))))
            meta_cols[3].metric("Stability index", format_decimal(safe_float(pattern_row_local.get("stability_index"))))

        if rules_row_df.empty:
            st.info("Metriche delle regole non disponibili.")
            return pattern_row_local

        rule_row = rules_row_df.iloc[0]
        policy_dom = str(rule_row.get("policy_dom", "")) if pd.notna(rule_row.get("policy_dom")) else ""
        models_available: List[Tuple[str, str, Dict[str, float]]] = []

        if not cat_metrics_df.empty:
            cat_row_df = cat_metrics_df[cat_metrics_df["cliente"] == selected_client]
            if not cat_row_df.empty:
                cat_row = cat_row_df.iloc[0]
                models_available.append(
                    (
                        "catboost",
                        "Modello A · CatBoost",
                        {
                            "recall": safe_float(cat_row.get("model_recall_at_k")),
                            "match_rate": safe_float(cat_row.get("model_match_rate")),
                        },
                    )
                )
        if not auto_metrics_df.empty:
            auto_row_df = auto_metrics_df[auto_metrics_df["cliente"] == selected_client]
            if not auto_row_df.empty:
                auto_row = auto_row_df.iloc[0]
                models_available.append(
                    (
                        "autogluon",
                        "Modello B · AutoGluon",
                        {
                            "recall": safe_float(auto_row.get("model_recall_at_k")),
                            "match_rate": safe_float(auto_row.get("model_match_rate")),
                        },
                    )
                )

        if not models_available:
            st.info("Nessun modello addestrato disponibile per il confronto.")
            return pattern_row_local

        labels = [entry[1] for entry in models_available]
        selected_label = st.radio("Modello da confrontare", labels, horizontal=True, index=0)
        selected_type, _, selected_metrics = next(entry for entry in models_available if entry[1] == selected_label)

        rules_metrics = {
            "recall": safe_float(rule_row.get("rules_recall_at_k")),
            "match_rate": safe_float(rule_row.get("rules_match_rate")),
        }

        badge_color = "gray"
        badge_text = "Dati non sufficienti"
        if not np.isnan(rules_metrics["match_rate"]) and not np.isnan(selected_metrics["match_rate"]):
            if selected_metrics["match_rate"] > rules_metrics["match_rate"] + 1e-3:
                badge_color = "green"
                badge_text = "Preferisci MODELLO"
            elif rules_metrics["match_rate"] > selected_metrics["match_rate"] + 1e-3:
                badge_color = "blue"
                badge_text = "Preferisci REGOLA"
            else:
                badge_color = "orange"
                badge_text = "Match equivalente"
        st.markdown(f":{badge_color}[{badge_text}]")

        model_short = selected_label.split("·")[0].strip()
        metric_cols = st.columns(4)
        metric_cols[0].metric("Recall@k Regola", format_percentage(rules_metrics["recall"]))
        metric_cols[1].metric("Match rate Regola", format_percentage(rules_metrics["match_rate"]))
        metric_cols[2].metric(f"Recall@k {model_short}", format_percentage(selected_metrics["recall"]))
        metric_cols[3].metric(f"Match rate {model_short}", format_percentage(selected_metrics["match_rate"]))

        st.markdown("#### Ultimi eventi")
        if client_events.empty:
            st.info("Nessun evento disponibile per il confronto.")
        else:
            recent_events = client_events.sort_values("data_pareggio").tail(3)
            for _, event_row in recent_events.iterrows():
                event_key = str(event_row.get("event_key"))
                data_value = event_row.get("data_pareggio")
                if pd.isna(data_value):
                    event_date = pd.NaT
                else:
                    event_date = pd.to_datetime(str(data_value), errors="coerce")
                date_label = event_date.date().isoformat() if pd.notna(event_date) else "data n.d."
                st.markdown(f"**Evento {event_key} — {date_label}**")

                event_features = feature_repository[feature_repository["event_key"] == event_key]
                if event_features.empty:
                    st.info("Feature non disponibili per questo evento.")
                    continue

                actual_docs = (
                    event_features[event_features["label"] == 1]["numero_documento"].astype(str).tolist()
                )
                k = max(len(actual_docs), 1)
                policy_label = policy_dom.upper() if policy_dom else "Regola"
                policy_docs = recommend_policy_docs(event_features, policy_dom, k)
                model_docs, shap_details = top_docs_from_model(
                    event_features,
                    cat_feature_cols,
                    cat_cat_indices,
                    auto_feature_cols,
                    selected_type,
                    k,
                )
                comparison_df = build_event_comparison_table(
                    actual_docs,
                    policy_docs,
                    model_docs,
                    policy_label,
                    model_short,
                )
                st.dataframe(comparison_df, use_container_width=True)

                if selected_type == "catboost" and shap_details:
                    with st.expander("Spiegazioni SHAP (prime 2 fatture)", expanded=False):
                        for doc_id, contributions in shap_details:
                            pretty = ", ".join(f"{feat}: {val:+.3f}" for feat, val in contributions)
                            st.write(f"**{doc_id}** → {pretty}")

        return pattern_row_local

    pattern_row = _render_model_vs_rules()

    @tracker("UI.PatternCard.Summary")
    def _render_summary_section() -> None:
        st.markdown("### Sintesi")
        if score_row.empty:
            st.info("Nessuna metrica disponibile per sintetizzare il comportamento.")
        else:
            label, confidence = infer_pattern_label(score_row)
            st.metric("Pattern riconosciuto", f"{label}", delta=f"Confidence: {confidence:.2f}")
            summary_lines = render_pattern_summary(
                selected_client,
                score_row=score_row,
                pattern_row=pattern_row,
            )
            if summary_lines:
                st.markdown("\n".join(f"- {line}" for line in summary_lines))
            else:
                st.info("Sintesi non disponibile.")

    _render_summary_section()

    @tracker("UI.PatternCard.Validation")
    def _render_validation_section() -> None:
        st.markdown("### Validazione")
        if selected_client and st.button("Valida su hold-out"):
            with st.spinner("Esecuzione validazione hold-out..."):
                accuracy = run_holdout_validation(selected_client)
            if accuracy is None:
                st.info("Funzione di validazione non ancora disponibile. Completare il Prompt 7 per abilitarla.")
            else:
                st.success(f"Accuracy hold-out: {accuracy:.2%}")

    _render_validation_section()


def main() -> None:
    tables, missing = load_curated_tables()
    if missing:
        st.sidebar.warning("File mancanti: " + ", ".join(missing))

    events = tables.get("events", pd.DataFrame())
    if events.empty:
        st.error("Il file events.parquet è necessario per utilizzare la dashboard.")
        st.stop()

    event_items = tables.get("event_items", pd.DataFrame())
    pattern_scores = tables.get("client_pattern_scores", pd.DataFrame())
    pattern_table_df = tables.get("pattern_table", pd.DataFrame())
    rules_metrics_df = tables.get("client_rules_accuracy", pd.DataFrame())
    cat_metrics_df = tables.get("client_model_metrics_catboost", pd.DataFrame())
    auto_metrics_df = tables.get("client_model_metrics_autogluon", pd.DataFrame())

    feature_repository, cat_feature_cols, cat_cat_indices, auto_feature_cols = load_feature_repository()

    clients = sorted(events["cliente"].dropna().astype(str).unique())
    if not clients:
        st.error("Nessun cliente disponibile negli eventi.")
        st.stop()

    page = st.sidebar.radio("Pagina", PAGE_OPTIONS, key="selected_page")

    if "selected_client" not in st.session_state or st.session_state["selected_client"] not in clients:
        st.session_state["selected_client"] = clients[0]

    st.sidebar.header("Selezione")
    selected_client_value = st.sidebar.selectbox("Cliente", clients, key="selected_client")
    selected_client = str(selected_client_value)

    developer_mode = st.sidebar.checkbox("Developer mode", value=False)
    developer_metrics: Dict[str, Any] = {"sections": [], "dataframes": {}}

    if page == "KPI Dashboard":
        @_section_tracker(developer_metrics, "UI.KPI_Dashboard")
        def _render_kpi_dashboard() -> None:
            render_kpi_dashboard_page(
                tables=tables,
                developer_metrics=developer_metrics,
            )

        _render_kpi_dashboard()
        if developer_mode:
            developer_metrics["logs"] = _load_warning_log_records()
            developer_metrics["champion"] = _load_champion_metadata()
            _render_developer_panel(developer_metrics)
        return

    if page == "Data Explorer":
        render_data_explorer_page(
            tables=tables,
            default_client=selected_client,
            developer_metrics=developer_metrics,
        )
        if developer_mode:
            developer_metrics["logs"] = _load_warning_log_records()
            developer_metrics["champion"] = _load_champion_metadata()
            _render_developer_panel(developer_metrics)
        return

    if page == "Clusters":
        @_section_tracker(developer_metrics, "UI.Clusters")
        def _render_clusters() -> None:
            render_clusters_page(
                tables=tables,
                selected_client=selected_client,
                clients=clients,
            )

        _render_clusters()
        if developer_mode:
            developer_metrics["logs"] = _load_warning_log_records()
            developer_metrics["champion"] = _load_champion_metadata()
            _render_developer_panel(developer_metrics)
        return

    render_pattern_card(
        selected_client=selected_client,
        events=events,
        event_items=event_items,
        pattern_scores=pattern_scores,
        pattern_table_df=pattern_table_df,
        rules_metrics_df=rules_metrics_df,
        cat_metrics_df=cat_metrics_df,
        auto_metrics_df=auto_metrics_df,
        feature_repository=feature_repository,
        cat_feature_cols=cat_feature_cols,
        cat_cat_indices=cat_cat_indices,
        auto_feature_cols=auto_feature_cols,
        developer_metrics=developer_metrics,
    )

    if developer_mode:
        developer_metrics["logs"] = _load_warning_log_records()
        developer_metrics["champion"] = _load_champion_metadata()
        _render_developer_panel(developer_metrics)


if __name__ == "__main__":  # pragma: no cover
    main()
