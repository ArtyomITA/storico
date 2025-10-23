"""Generate printable HTML pattern reports for each client."""
from __future__ import annotations

import argparse
import html
import logging
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

from src.analytics.pattern_detectors import POLICY_NAMES
from src.report.nlg_patterns import render_pattern_summary
from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
CURATED_DIR = BASE_DIR / "data" / "curated"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
CATBOOST_MODEL_PATH = MODELS_DIR / "catboost.cbm"
CALIBRATOR_PATH = MODELS_DIR / "calibrator.pkl"
AUTOGLUON_DIR = MODELS_DIR / "autogluon"


def configure_logging(level: int | str | None = logging.INFO) -> Path:
    return setup_logging(level)


def compute_acf(values: np.ndarray, max_lag: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    if values.size < 2:
        return np.array([]), np.array([])
    lags: List[int] = []
    coeffs: List[float] = []
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
            opacity=0.85,
        )
    )
    markers = {
        "Settimanale": 7,
        "Quindicinale": 14,
        "Mensile": 30,
    }
    for label, marker in markers.items():
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
    )
    return fig


def plot_acf(values: np.ndarray) -> go.Figure:
    lags, coeffs = compute_acf(values)
    fig = go.Figure()
    if lags.size:
        fig.add_trace(go.Bar(x=lags, y=coeffs, marker_color="#F58518"))
    fig.update_layout(
        title="Autocorrelazione (lag in eventi)",
        xaxis_title="Lag",
        yaxis_title="Coeff.",
        yaxis=dict(range=[-1, 1]),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


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
        )
    )
    heat.update_layout(margin=dict(l=40, r=40, t=40, b=40), title="Frequenza eventi per giorno")
    return heat


def plot_policy_bars(scores: pd.Series) -> go.Figure:
    values = [float(scores.get(f"{policy}_score", 0.0)) for policy in POLICY_NAMES]
    fig = go.Figure(
        go.Bar(
            x=[policy.upper() for policy in POLICY_NAMES],
            y=values,
            marker_color="#59A14F",
        )
    )
    fig.update_layout(
        title="Recall medio per policy",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def plot_cardinality_histogram(events: pd.DataFrame) -> go.Figure:
    if events.empty:
        return go.Figure()
    k_values = events["n_righe_evento"].dropna()
    fig = go.Figure(
        go.Histogram(
            x=k_values,
            nbinsx=min(15, max(5, k_values.nunique())),
            marker_color="#8CD17D",
        )
    )
    fig.update_layout(
        title="Distribuzione cardinalità evento",
        xaxis_title="Numero righe (k)",
        yaxis_title="Frequenza",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def derive_macro_tipo(items: pd.DataFrame) -> pd.Series:
    if "macro_tipo" in items.columns:
        return items["macro_tipo"].fillna("Sconosciuto").astype(str)
    macro = items["tipo_documento"].astype(str).str[:1]
    return macro.replace({"": "Sconosciuto", "nan": "Sconosciuto"}).fillna("Sconosciuto")
def enrich_features_for_models(
    features: pd.DataFrame,
    event_items: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
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
    return feature_df, target, list(feature_df.columns), cat_indices


def select_autogluon_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    target = df["label"].astype(int) if "label" in df.columns else pd.Series(dtype=int)
    drop_cols = ["label", "data_pareggio", "cliente", "event_key", "numero_documento"]
    feature_df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    return feature_df, target, list(feature_df.columns)


def load_feature_repository() -> Tuple[pd.DataFrame, List[str], List[int], List[str]]:
    features_path = CURATED_DIR / "train_features.parquet"
    if not features_path.exists():
        LOGGER.warning("Feature store non trovato in %s", features_path)
        return pd.DataFrame(), [], [], []
    features = pd.read_parquet(features_path)
    event_items_path = CURATED_DIR / "event_items.parquet"
    events_path = CURATED_DIR / "events.parquet"
    event_items = pd.read_parquet(event_items_path) if event_items_path.exists() else pd.DataFrame()
    events = pd.read_parquet(events_path) if events_path.exists() else pd.DataFrame()
    enriched = enrich_features_for_models(features, event_items, events)
    cat_matrix, _, cat_feature_cols, cat_indices = select_catboost_features(enriched)
    auto_matrix, _, auto_feature_cols = select_autogluon_features(enriched)
    col_intersection = list(cat_matrix.columns)
    return enriched, col_intersection, cat_indices, auto_feature_cols


def load_catboost_bundle() -> Tuple[Optional[Any], Optional[Any]]:
    try:
        from catboost import CatBoostClassifier  # type: ignore
    except ImportError:
        LOGGER.info("CatBoost non disponibile per il report.")
        return None, None
    if not CATBOOST_MODEL_PATH.exists():
        LOGGER.info("Modello CatBoost mancante in %s", CATBOOST_MODEL_PATH)
        return None, None
    model = CatBoostClassifier()
    model.load_model(str(CATBOOST_MODEL_PATH))
    calibrator: Optional[Any] = None
    if CALIBRATOR_PATH.exists():
        try:
            with open(CALIBRATOR_PATH, "rb") as f:
                calibrator = pickle.load(f)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Impossibile caricare il calibratore: %s", exc)
            calibrator = None
    return model, calibrator


def load_autogluon_predictor() -> Optional[Any]:
    try:
        from autogluon.tabular import TabularPredictor  # type: ignore
    except ImportError:
        LOGGER.info("AutoGluon non disponibile per il report.")
        return None
    if not AUTOGLUON_DIR.exists():
        LOGGER.info("Directory AutoGluon non trovata in %s", AUTOGLUON_DIR)
        return None
    return TabularPredictor.load(str(AUTOGLUON_DIR))


def safe_calibrate(calibrator: Optional[Any], values: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return values
    try:
        return np.asarray(calibrator.predict(values))
    except Exception:
        LOGGER.warning("Calibratore non applicabile; uso probabilità grezze.")
        return values


def recommend_policy_docs(event_df: pd.DataFrame, policy: Optional[str], k: int) -> List[str]:
    if event_df.empty or not policy:
        return []
    df = event_df.copy()
    if "imp_div_int" in df.columns:
        df["abs_amount"] = df["imp_div_int"].fillna(0.0).abs()
    if "age_days" in df.columns:
        df["age_days"] = df["age_days"].fillna(0.0)
    if "same_period" in df.columns:
        df["same_period"] = df["same_period"].fillna(0.0)
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


def top_docs_from_catboost(
    event_df: pd.DataFrame,
    feature_columns: Sequence[str],
    cat_indices: Sequence[int],
    model: Any,
    calibrator: Optional[Any],
    k: int,
    shap_top_docs: int = 2,
    shap_top_features: int = 3,
) -> Tuple[List[str], Dict[str, List[Dict[str, Any]]]]:
    from catboost import Pool  # type: ignore

    if model is None or event_df.empty:
        return [], {}
    event_df = event_df.reset_index(drop=True).copy()
    model_input = event_df[feature_columns]
    pool = Pool(model_input, cat_features=list(cat_indices) if cat_indices else None)
    probs = model.predict_proba(pool)[:, 1]
    probs = safe_calibrate(calibrator, probs)
    event_df = event_df.assign(model_prob=probs)
    ranked = event_df.sort_values("model_prob", ascending=False)
    top_rows = ranked.head(max(k, 1))
    docs = top_rows["numero_documento"].astype(str).tolist()
    shap_details: Dict[str, List[Dict[str, Any]]] = {}
    try:
        shap_values = model.get_feature_importance(pool, type="ShapValues")
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            for idx, row in top_rows.head(max(shap_top_docs, 0)).iterrows():
                original_idx = int(idx)
                shap_row = shap_values[original_idx, :-1]
                feature_vector = model_input.iloc[original_idx]
                ranked_indices = np.argsort(np.abs(shap_row))[::-1][: max(shap_top_features, 0)]
                details: List[Dict[str, Any]] = []
                for feature_idx in ranked_indices:
                    feature_name = feature_columns[feature_idx]
                    feature_value = feature_vector.iloc[feature_idx]
                    details.append(
                        {
                            "feature": feature_name,
                            "value": feature_value,
                            "shap": float(shap_row[feature_idx]),
                        }
                    )
                shap_details[str(row["numero_documento"])] = details
        return docs, shap_details
    except Exception:
        return docs, {}


def top_docs_from_autogluon(
    event_df: pd.DataFrame,
    feature_columns: Sequence[str],
    predictor: Any,
    k: int,
) -> List[str]:
    if predictor is None or event_df.empty:
        return []
    model_input = event_df.reset_index(drop=True)[feature_columns]
    proba = predictor.predict_proba(model_input)
    if isinstance(proba, pd.DataFrame):
        if 1 in proba.columns:
            probs = proba[1].to_numpy()
        else:
            probs = proba.iloc[:, -1].to_numpy()
    else:  # pragma: no cover
        probs = np.asarray(proba)
    ranked = event_df.assign(model_prob=probs).sort_values("model_prob", ascending=False)
    return ranked.head(max(k, 1))["numero_documento"].astype(str).tolist()


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


def _extract_threshold_hint(feature_name: str) -> Optional[str]:
    pattern = re.compile(r"(-?\d+(?:\.\d+)?)")
    match = pattern.search(feature_name)
    if match:
        return match.group(1)
    return None


def _format_feature_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n.d."
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def _format_shap_details(entries: Sequence[Dict[str, Any]]) -> str:
    if not entries:
        return "<p class=\"shap-empty\">SHAP non disponibili.</p>"
    items: List[str] = []
    for entry in entries:
        feature_name = str(entry.get("feature", ""))
        feature_label = feature_name.replace("_", " ")
        shap_value = entry.get("shap")
        shap_text = "n.d."
        if isinstance(shap_value, (int, float, np.floating)):
            shap_text = f"{float(shap_value):+.3f}"
        feature_value = _format_feature_value(entry.get("value"))
        threshold = _extract_threshold_hint(feature_name) or "n.d."
        items.append(
            "<li><strong>{label}</strong>: valore {value} · soglia {thr} · SHAP {shap}</li>".format(
                label=html.escape(feature_label),
                value=html.escape(feature_value),
                thr=html.escape(threshold),
                shap=html.escape(shap_text),
            )
        )
    return "<ul class=\"shap-list\">" + "".join(items) + "</ul>"


def _build_highlighted_row(doc: str, actual_set: set[str]) -> str:
    if not doc:
        return "<td class=\"cell-empty\"></td>"
    cls = "match-cell" if doc in actual_set else "mismatch-cell"
    return f"<td class=\"{cls}\">{html.escape(doc)}</td>"


def build_highlighted_comparison_html(
    actual_docs: Sequence[str],
    policy_docs: Sequence[str],
    model_a_docs: Sequence[str],
    model_b_docs: Sequence[str],
    model_a_label: str,
    model_b_label: str,
) -> str:
    actual_list = [doc for doc in actual_docs if doc]
    actual_set = set(actual_list)
    max_len = max(len(actual_list), len(policy_docs), len(model_a_docs), len(model_b_docs), 1)
    rows: List[str] = []
    for idx in range(max_len):
        actual_doc = actual_list[idx] if idx < len(actual_list) else ""
        policy_doc = policy_docs[idx] if idx < len(policy_docs) else ""
        model_a_doc = model_a_docs[idx] if idx < len(model_a_docs) else ""
        model_b_doc = model_b_docs[idx] if idx < len(model_b_docs) else ""
        row_html = (
            "<tr>"
            f"<td class=\"actual-cell\">{html.escape(actual_doc) if actual_doc else ''}</td>"
            f"{_build_highlighted_row(policy_doc, actual_set)}"
            f"{_build_highlighted_row(model_a_doc, actual_set)}"
            f"{_build_highlighted_row(model_b_doc, actual_set)}"
            "</tr>"
        )
        rows.append(row_html)
    header_html = (
        "<thead><tr><th>Fatture reali</th>"
        "<th>Top-k Policy dominante</th>"
        f"<th>Top-k {html.escape(model_a_label)}</th>"
        f"<th>Top-k {html.escape(model_b_label)}</th></tr></thead>"
    )
    body_html = "<tbody>" + "".join(rows) + "</tbody>"
    return "<table class=\"explain-table\">" + header_html + body_html + "</table>"


def build_explainability_cards(
    client_events: pd.DataFrame,
    features_repo: pd.DataFrame,
    policy_name: Optional[str],
    cat_feature_cols: Sequence[str],
    cat_indices: Sequence[int],
    auto_feature_cols: Sequence[str],
    catboost_bundle: Tuple[Optional[Any], Optional[Any]],
    autogluon_predictor: Optional[Any],
    model_label_map: Dict[str, str],
    max_events: int = 3,
) -> List[str]:
    if client_events.empty:
        return []

    recent_events = (
        client_events.copy()
        .assign(data_pareggio=pd.to_datetime(client_events["data_pareggio"], errors="coerce"))
        .sort_values("data_pareggio")
        .tail(max_events)
    )

    cards: List[str] = []
    cat_model, calibrator = catboost_bundle
    model_a_label = model_label_map.get("catboost", "Modello A")
    model_b_label = model_label_map.get("autogluon", "Modello B")

    for _, event_row in recent_events.iterrows():
        event_key = str(event_row.get("event_key"))
        raw_date = event_row.get("data_pareggio")
        event_date = pd.to_datetime(raw_date, errors="coerce") if not pd.isna(raw_date) else pd.NaT
        date_label = event_date.date().isoformat() if pd.notna(event_date) else "data n.d."

        if features_repo.empty:
            event_features = pd.DataFrame()
        else:
            event_features = features_repo[features_repo["event_key"] == str(event_key)]
        if event_features.empty:
            cards.append(
                (
                    "<div class=\"explain-card\">"
                    f"<div class=\"explain-header\"><strong>Evento {html.escape(event_key)}</strong> · {date_label}</div>"
                    "<p class=\"explain-empty\">Feature non disponibili per questo evento.</p>"
                    "</div>"
                )
            )
            continue

        event_features = event_features.reset_index(drop=True).copy()
        event_features["numero_documento"] = event_features["numero_documento"].astype(str)
        actual_docs = (
            event_features[event_features.get("label") == 1]["numero_documento"].astype(str).tolist()
        )
        k = max(len(actual_docs), 3)

        policy_docs = recommend_policy_docs(event_features, policy_name, k)

        cat_docs: List[str] = []
        shap_details: Dict[str, List[Dict[str, Any]]] = {}
        if cat_model is not None and cat_feature_cols:
            cat_docs, shap_details = top_docs_from_catboost(
                event_features,
                cat_feature_cols,
                cat_indices,
                cat_model,
                calibrator,
                k,
            )

        auto_docs: List[str] = []
        if autogluon_predictor is not None and auto_feature_cols:
            auto_docs = top_docs_from_autogluon(
                event_features,
                auto_feature_cols,
                autogluon_predictor,
                k,
            )

        table_html = build_highlighted_comparison_html(
            actual_docs,
            policy_docs,
            cat_docs,
            auto_docs,
            model_a_label,
            model_b_label,
        )

        shap_blocks: List[str] = []
        if shap_details:
            for doc_id in cat_docs[:2]:
                entries = shap_details.get(doc_id, [])
                shap_blocks.append(
                    "<div class=\"shap-card\">"
                    f"<h5>Fattura {html.escape(doc_id)}</h5>"
                    f"{_format_shap_details(entries)}"
                    "</div>"
                )
        else:
            shap_blocks.append("<p class=\"shap-empty\">Spiegazioni SHAP non disponibili.</p>")

        cards.append(
            "<div class=\"explain-card\">"
            f"<div class=\"explain-header\"><strong>Evento {html.escape(event_key)}</strong> · {date_label}</div>"
            f"{table_html}"
            "<div class=\"shap-wrapper\">"
            f"<h4>Spiegazioni modello A ({html.escape(model_a_label)})</h4>"
            + "".join(shap_blocks)
            + "</div>"
            "</div>"
        )

    return cards


def figure_to_html(fig: go.Figure, include_plotlyjs: bool = False) -> str:
    return to_html(
        fig,
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        default_height="420px",
    )


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


def load_curated_tables() -> Dict[str, pd.DataFrame]:
    filenames = {
        "events": "events.parquet",
        "event_items": "event_items.parquet",
        "client_pattern_scores": "client_pattern_scores.parquet",
        "pattern_table": "pattern_table.parquet",
        "client_rules_accuracy": "client_rules_accuracy.parquet",
        "client_model_metrics_catboost": "client_model_metrics_catboost.parquet",
        "client_model_metrics_autogluon": "client_model_metrics_autogluon.parquet",
    }
    tables: Dict[str, pd.DataFrame] = {}
    for key, filename in filenames.items():
        path = CURATED_DIR / filename
        if path.exists():
            tables[key] = pd.read_parquet(path)
            LOGGER.info("Caricato %s (%d righe)", filename, len(tables[key]))
        else:
            LOGGER.warning("File %s non trovato", path)
            tables[key] = pd.DataFrame()
    return tables


def render_client_report(
    cliente: str,
    tables: Dict[str, pd.DataFrame],
    features_repo: pd.DataFrame,
    cat_feature_cols: Sequence[str],
    cat_indices: Sequence[int],
    auto_feature_cols: Sequence[str],
    catboost_bundle: Tuple[Optional[Any], Optional[Any]],
    autogluon_predictor: Optional[Any],
) -> str:
    events = tables["events"]
    event_items = tables["event_items"]
    pattern_scores_df = tables["client_pattern_scores"]
    pattern_table_df = tables["pattern_table"]
    rules_df = tables["client_rules_accuracy"]
    cat_metrics_df = tables["client_model_metrics_catboost"]
    auto_metrics_df = tables["client_model_metrics_autogluon"]

    client_events = events[events["cliente"] == cliente].copy()
    client_events["data_pareggio"] = pd.to_datetime(client_events["data_pareggio"], errors="coerce")
    client_items = event_items[event_items["cliente"] == cliente].copy()
    pattern_row = pattern_table_df[pattern_table_df["cliente"] == cliente]
    pattern_scores_row = pattern_scores_df[pattern_scores_df["cliente"] == cliente]
    rules_row = rules_df[rules_df["cliente"] == cliente]
    cat_row = cat_metrics_df[cat_metrics_df["cliente"] == cliente]
    auto_row = auto_metrics_df[auto_metrics_df["cliente"] == cliente]

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    events_used = len(client_events)

    gaps = client_events["inter_event_gap_days"].dropna().to_numpy(dtype=float)
    hist_fig = plot_histogram_with_markers(gaps, "Distribuzione gap tra eventi")
    acf_fig = plot_acf(gaps)
    calendar_fig = plot_calendar_heatmap(client_events)
    policy_fig = plot_policy_bars(pattern_scores_row.iloc[0]) if not pattern_scores_row.empty else go.Figure()
    cardinality_fig = plot_cardinality_histogram(client_events)

    figures_html = {
        "hist": figure_to_html(hist_fig, include_plotlyjs=True),
        "acf": figure_to_html(acf_fig),
        "calendar": figure_to_html(calendar_fig),
        "policy": figure_to_html(policy_fig),
        "cardinality": figure_to_html(cardinality_fig),
    }

    pattern_series = pattern_row.iloc[0] if not pattern_row.empty else None
    score_series = pattern_scores_row.iloc[0] if not pattern_scores_row.empty else None

    if not features_repo.empty and "cliente" in features_repo.columns:
        client_feature_repo = features_repo[features_repo["cliente"] == cliente].copy()
    else:
        client_feature_repo = pd.DataFrame()
    if not client_feature_repo.empty and "event_key" in client_feature_repo.columns:
        client_feature_repo["event_key"] = client_feature_repo["event_key"].astype(str)
    if not client_feature_repo.empty and "numero_documento" in client_feature_repo.columns:
        client_feature_repo["numero_documento"] = client_feature_repo["numero_documento"].astype(str)

    pattern_label = (
        str(pattern_series.get("pattern_label", "N/A")) if pattern_series is not None else "N/A"
    )
    confidence = float(pattern_series.get("confidence", np.nan)) if pattern_series is not None else np.nan
    k_mode = float(pattern_series.get("k_mode", np.nan)) if pattern_series is not None else np.nan
    k_entropy = float(pattern_series.get("k_entropy", np.nan)) if pattern_series is not None else np.nan
    stability = float(pattern_series.get("stability_index", np.nan)) if pattern_series is not None else np.nan
    summary_lines = render_pattern_summary(
        cliente,
        score_row=score_series,
        pattern_row=pattern_series,
    )
    def _markdown_to_html(phrase: str) -> str:
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", phrase)

    if summary_lines:
        summary_items = "".join(
            f"<li>{_markdown_to_html(line)}</li>" for line in summary_lines
        )
        summary_html = f"<ul class=\"summary-list\">{summary_items}</ul>"
    else:
        summary_html = "<p class=\"summary-empty\">Sintesi non disponibile.</p>"

    rules_metrics = {
        "policy_dom": rules_row.iloc[0]["policy_dom"] if not rules_row.empty else "",
        "recall": rules_row.iloc[0]["rules_recall_at_k"] if not rules_row.empty else np.nan,
        "match_rate": rules_row.iloc[0]["rules_match_rate"] if not rules_row.empty else np.nan,
    }
    models_metrics: List[Tuple[str, str, Dict[str, float]]] = []
    if not cat_row.empty:
        models_metrics.append(
            (
                "catboost",
                "Modello A · CatBoost",
                {
                    "recall": cat_row.iloc[0]["model_recall_at_k"],
                    "match_rate": cat_row.iloc[0]["model_match_rate"],
                },
            )
        )
    if not auto_row.empty:
        models_metrics.append(
            (
                "autogluon",
                "Modello B · AutoGluon",
                {
                    "recall": auto_row.iloc[0]["model_recall_at_k"],
                    "match_rate": auto_row.iloc[0]["model_match_rate"],
                },
            )
        )

    recent_event_row = client_events.sort_values("data_pareggio").tail(1)
    comparison_sections: List[str] = []
    if not recent_event_row.empty:
        event_key = recent_event_row.iloc[0]["event_key"]
        event_date = recent_event_row.iloc[0]["data_pareggio"]
        if client_feature_repo.empty:
            event_features = pd.DataFrame()
        else:
            event_features = client_feature_repo[client_feature_repo["event_key"] == str(event_key)]
        actual_docs = event_features[event_features.get("label") == 1]["numero_documento"].astype(str).tolist()
        k = max(len(actual_docs), 1)
        policy_docs = recommend_policy_docs(event_features, rules_metrics["policy_dom"], k)
        cat_model, calibrator = catboost_bundle
        for model_code, model_label, metrics in models_metrics:
            if model_code == "catboost" and cat_model is not None:
                model_docs, _ = top_docs_from_catboost(
                    event_features,
                    cat_feature_cols,
                    cat_indices,
                    cat_model,
                    calibrator,
                    k,
                )
            elif model_code == "autogluon" and autogluon_predictor is not None:
                model_docs = top_docs_from_autogluon(
                    event_features,
                    auto_feature_cols,
                    autogluon_predictor,
                    k,
                )
            else:
                model_docs = []
            comparison_df = build_event_comparison_table(
                actual_docs,
                policy_docs,
                model_docs,
                (rules_metrics["policy_dom"] or "Regola").upper(),
                model_label.split("·")[0].strip(),
            )
            comparison_sections.append(
                f"<h4>{model_label} — Evento {event_key} ({event_date.date().isoformat() if pd.notna(event_date) else 'data n.d.'})</h4>"
                + comparison_df.to_html(index=False, classes="comparison-table", justify="center")
            )
    else:
        comparison_sections.append("<p>Nessun evento disponibile per illustrare il confronto.</p>")

    model_label_map = {code: label.split("·")[0].strip() for code, label, _ in models_metrics}
    explainability_cards = build_explainability_cards(
        client_events=client_events,
        features_repo=client_feature_repo,
        policy_name=rules_metrics["policy_dom"] or None,
        cat_feature_cols=cat_feature_cols,
        cat_indices=cat_indices,
        auto_feature_cols=auto_feature_cols,
        catboost_bundle=catboost_bundle,
        autogluon_predictor=autogluon_predictor,
        model_label_map=model_label_map,
    )
    explainability_html = (
        "".join(explainability_cards)
        if explainability_cards
        else "<p class=\"explain-empty\">Nessuna telemetria disponibile per gli eventi recenti.</p>"
    )

    badge_text = "Nessun dato"
    badge_class = "badge-neutral"
    rule_match = rules_metrics["match_rate"]
    best_model = max(models_metrics, key=lambda item: item[2]["match_rate"]) if models_metrics else None
    if best_model:
        best_match = best_model[2]["match_rate"]
        if not np.isnan(best_match) and not np.isnan(rule_match):
            if best_match > rule_match + 1e-3:
                badge_text = "Preferisci MODELLO"
                badge_class = "badge-model"
            elif rule_match > best_match + 1e-3:
                badge_text = "Preferisci REGOLA"
                badge_class = "badge-rule"
            else:
                badge_text = "Prestazioni equivalenti"
                badge_class = "badge-neutral"
        else:
            badge_text = "Dati insufficienti"
    elif not np.isnan(rule_match):
        badge_text = "Solo regole disponibili"

    metrics_rows = [
        (
            "Regole",
            rules_metrics["policy_dom"].upper() if rules_metrics["policy_dom"] else "-",
            format_percentage(rules_metrics["recall"]),
            format_percentage(rules_metrics["match_rate"]),
        )
    ]
    for model_code, model_label, metrics in models_metrics:
        metrics_rows.append(
            (
                model_label,
                "-",
                format_percentage(metrics["recall"]),
                format_percentage(metrics["match_rate"]),
            )
        )
    metrics_table = (
        pd.DataFrame(metrics_rows, columns=["Sistema", "Policy", "Recall@k", "Match rate"])
        .to_html(index=False, classes="metrics-table", justify="center")
    )

    html = f"""
<!DOCTYPE html>
<html lang=\"it\">
<head>
<meta charset=\"utf-8\">
<title>Pattern Card — Cliente {cliente}</title>
<link rel=\"preconnect\" href=\"https://cdn.plot.ly\"/>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 24px; background-color: #f7f7f9; color: #222; }}
header {{ margin-bottom: 32px; }}
section {{ background: #ffffff; padding: 24px; margin-bottom: 24px; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
h1 {{ margin-bottom: 8px; }}
h2 {{ margin-top: 0; }}
.badge {{ display: inline-block; padding: 6px 12px; border-radius: 999px; font-weight: 600; font-size: 0.9rem; color: #fff; }}
.badge-model {{ background: #2ca02c; }}
.badge-rule {{ background: #1f77b4; }}
.badge-neutral {{ background: #ff7f0e; }}
.metadata {{ color: #666; font-size: 0.9rem; }}
.grid-two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px; align-items: stretch; }}
.plot-container {{ background: #fafafa; padding: 12px; border-radius: 8px; }}
.metrics-table, .comparison-table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
.metrics-table th, .metrics-table td, .comparison-table th, .comparison-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
.metrics-table th {{ background: #f0f0f0; }}
.comparison-table th {{ background: #f9f9f9; }}
.summary-box {{ background: #f0f5ff; border-left: 4px solid #3f51b5; padding: 16px; border-radius: 8px; }}
.summary-list {{ margin: 0; padding-left: 20px; }}
.summary-list li {{ margin-bottom: 6px; line-height: 1.4; }}
.summary-empty {{ color: #666; margin: 0; }}
.explain-section {{ display: flex; flex-direction: column; gap: 16px; }}
.explain-card {{ background: #ffffff; border-radius: 10px; border: 1px solid #e5e7eb; padding: 16px; box-shadow: inset 0 0 0 1px rgba(63,81,181,0.05); }}
.explain-header {{ display: flex; justify-content: space-between; align-items: baseline; font-size: 0.95rem; color: #374151; margin-bottom: 12px; }}
.explain-table {{ width: 100%; border-collapse: collapse; margin-bottom: 12px; }}
.explain-table th, .explain-table td {{ border: 1px solid #d1d5db; padding: 8px; text-align: center; font-size: 0.9rem; }}
.explain-table th {{ background: #eef2ff; font-weight: 600; }}
.actual-cell {{ background: #f8fafc; font-weight: 600; color: #1e3a8a; }}
.match-cell {{ background: #e8f5e9; color: #1b5e20; font-weight: 600; }}
.mismatch-cell {{ background: #ffebee; color: #b71c1c; font-weight: 600; }}
.cell-empty {{ background: #f9fafb; color: #9ca3af; }}
.shap-wrapper {{ display: flex; flex-wrap: wrap; gap: 12px; }}
.shap-card {{ flex: 1 1 240px; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; background: linear-gradient(180deg, #fafaff 0%, #ffffff 55%); }}
.shap-card h5 {{ margin: 0 0 8px; font-size: 0.95rem; color: #1f2937; }}
.shap-list {{ margin: 0; padding-left: 18px; font-size: 0.88rem; color: #374151; }}
.shap-empty {{ color: #6b7280; font-style: italic; }}
.explain-empty {{ color: #6b7280; margin: 8px 0 0; }}
</style>
</head>
<body>
<header>
  <h1>Pattern Card — Cliente {cliente}</h1>
  <div class=\"metadata\">Generato il {generated_at} · Eventi analizzati: {events_used}</div>
</header>
<section>
  <h2>Cadenza</h2>
  <div class=\"grid-two\">
    <div class=\"plot-container\">{figures_html['hist']}</div>
    <div class=\"plot-container\">{figures_html['acf']}</div>
  </div>
  <div class=\"plot-container\" style=\"margin-top:24px;\">{figures_html['calendar']}</div>
</section>
<section>
    <span class="badge {badge_class}">{badge_text}</span>
    <div style="margin-top: 16px;">{summary_html}</div>
    <div class=\"plot-container\">{figures_html['cardinality']}</div>
  </div>
  <div class=\"grid-two\" style=\"margin-top: 16px;\">
    <div class=\"summary-box\">
      <strong>k_mode:</strong> {format_decimal(k_mode, 1)}<br/>
      <strong>k_entropy:</strong> {format_decimal(k_entropy, 3)}<br/>
      <strong>Indice di stabilità:</strong> {format_decimal(stability, 3)}
    </div>
    <div class=\"summary-box\">
      <strong>Pattern riconosciuto:</strong> {pattern_label}<br/>
      <strong>Confidence:</strong> {format_decimal(confidence, 2)}
    </div>
  </div>
</section>
<section>
  <h2>Pattern sintetico</h2>
    <span class="badge {badge_class}">{badge_text}</span>
    <div style="margin-top: 16px;">{summary_html}</div>
</section>
<section>
    <h2>Explainability snippets</h2>
    <div class="explain-section">{explainability_html}</div>
</section>
<section>
  <h2>Confronto Regole vs Modelli</h2>
  {metrics_table}
  {''.join(comparison_sections)}
</section>
</body>
</html>
"""
    return html


def save_report(cliente: str, html: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / f"{cliente}.html"
    output_path.write_text(html, encoding="utf-8")
    LOGGER.info("Salvato report per %s in %s", cliente, output_path)
    return output_path


def generate_reports(clients: Optional[List[str]] = None) -> List[Path]:
    tables = load_curated_tables()
    features_repo, cat_feature_cols, cat_indices, auto_feature_cols = load_feature_repository()
    catboost_bundle = load_catboost_bundle()
    autogluon_predictor = load_autogluon_predictor()

    events = tables["events"]
    if events.empty:
        raise ValueError("Il file events.parquet è richiesto per generare i report.")
    available_clients = sorted(events["cliente"].dropna().astype(str).unique())
    target_clients = clients or available_clients

    output_paths: List[Path] = []
    for cliente in target_clients:
        LOGGER.info("Generazione report per %s", cliente)
        html = render_client_report(
            cliente,
            tables,
            features_repo,
            cat_feature_cols,
            cat_indices,
            auto_feature_cols,
            catboost_bundle,
            autogluon_predictor,
        )
        if len(output_paths) < 3:
            preview_lines = render_pattern_summary(cliente, tables=tables)
            if preview_lines:
                formatted_block = "\n".join(f"  - {line}" for line in preview_lines)
                LOGGER.info("Sintesi rapida %s:\n%s", cliente, formatted_block)
        output_paths.append(save_report(cliente, html))
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HTML pattern reports per client")
    parser.add_argument(
        "--clients",
        nargs="*",
        help="Lista opzionale di clienti da elaborare. Se omesso, tutti i clienti disponibili.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    generate_reports(args.clients)


if __name__ == "__main__":  # pragma: no cover
    main()
