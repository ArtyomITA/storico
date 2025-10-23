"""Generate per-client embedding combining SHAP, policy scores, and profile features.

The script loads the champion model, computes mean absolute SHAP values over
recent positive examples for each client, and concatenates the resulting vector
with policy scores and profile attributes. Outputs are written to the curated
folder alongside a JSON metadata file describing the feature ordering.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import shap
from autogluon.tabular import TabularPredictor
from catboost import CatBoostClassifier, Pool
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.models import train_autogluon as auto
from src.models import train_catboost as cat
from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_SAMPLES = 200
DEFAULT_OUTPUT_DIR = Path("data/curated")


@dataclass
class Champion:
    """Champion model metadata."""

    model: str
    trained_at: str


@dataclass
class ShapResult:
    """Output of the SHAP computation stage."""

    values: pd.DataFrame
    available: bool


def configure_logging(level: int | str | None = logging.INFO) -> Path:
    """Initialise logging using the shared handlers."""

    return setup_logging(level)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(
        description="Compute client embeddings from SHAP, policy scores, and profiles.",
    )
    parser.add_argument(
        "--champion",
        type=Path,
        default=Path("models/champion.json"),
        help="Path to champion metadata JSON.",
    )
    parser.add_argument(
        "--train-features",
        type=Path,
        default=Path("data/curated/train_features.parquet"),
        help="Path to train_features parquet.",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("data/curated/events.parquet"),
        help="Path to events parquet (for recency ordering).",
    )
    parser.add_argument(
        "--event-items",
        type=Path,
        default=Path("data/curated/event_items.parquet"),
        help="Path to event_items parquet for feature enrichment.",
    )
    parser.add_argument(
        "--pattern-scores",
        type=Path,
        default=Path("data/curated/client_pattern_scores.parquet"),
        help="Path to client pattern scores parquet.",
    )
    parser.add_argument(
        "--client-profiles",
        type=Path,
        default=Path("data/curated/client_profiles.parquet"),
        help="Path to client profile parquet.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Maximum number of positive samples per client for SHAP computation.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for embeddings and metadata.",
    )
    return parser.parse_args(argv)


def resolve_path(path: Path, project_root: Path) -> Path:
    """Resolve relative paths against the project root."""
    return path if path.is_absolute() else project_root / path


def load_champion(path: Path) -> Champion:
    """Read champion metadata from JSON."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    model = str(payload.get("model"))
    trained_at = str(payload.get("trained_at", ""))
    LOGGER.info("Loaded champion model: %s", model)
    return Champion(model=model, trained_at=trained_at)


def load_tables(paths: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    """Load required parquet tables into memory."""
    tables: Dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        LOGGER.info("Loading %s from %s", name, path)
        tables[name] = pd.read_parquet(path)
    return tables


def select_positive_samples(
    features: pd.DataFrame,
    events: pd.DataFrame,
    max_samples: int,
) -> pd.DataFrame:
    """Return up to ``max_samples`` positive rows per client ordered by recency."""
    events_local = events[["event_key", "data_pareggio"]].copy()
    events_local["data_pareggio"] = pd.to_datetime(
        events_local["data_pareggio"], errors="coerce"
    )

    merged = features.merge(events_local, on="event_key", how="left")
    positives = merged[merged["label"] == 1].copy()
    if positives.empty:
        LOGGER.warning("No positive examples found for SHAP computation.")
        return positives

    positives["data_pareggio"] = pd.to_datetime(
        positives["data_pareggio"], errors="coerce"
    )
    positives["_sort_key"] = positives["data_pareggio"].fillna(pd.Timestamp.min)
    positives = (
        positives.sort_values(["cliente", "_sort_key"], ascending=[True, False])
        .groupby("cliente", group_keys=False)
        .head(max_samples)
    )
    positives = positives.drop(columns="_sort_key")
    return positives


def normalise_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """Apply L2 normalisation across rows, guarding against zero vectors."""
    if df.empty:
        return df
    values = df.to_numpy(dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalised = values / norms
    return pd.DataFrame(normalised, index=df.index, columns=df.columns)


def compute_shap_catboost(
    sample_df: pd.DataFrame,
    model_dir: Path,
) -> ShapResult:
    """Compute per-client SHAP vectors for the CatBoost champion."""
    if sample_df.empty:
        return ShapResult(values=pd.DataFrame(), available=True)

    model_path = model_dir / "catboost.cbm"
    if not model_path.exists():
        raise FileNotFoundError(f"CatBoost model not found at {model_path}")

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    feature_df, _, _, cat_indices = cat.select_features(sample_df)
    pool = Pool(data=feature_df, cat_features=cat_indices)
    shap_values_raw = model.get_feature_importance(pool, type="ShapValues")  # type: ignore[arg-type]
    shap_values = np.asarray(shap_values_raw)
    shap_matrix = shap_values[:, :-1]  # last column is expected value
    shap_df = pd.DataFrame(
        np.abs(shap_matrix),
        index=sample_df.index,
        columns=feature_df.columns,
    )
    client_means = shap_df.groupby(sample_df["cliente"]).mean()
    client_means = normalise_vectors(client_means)
    client_means = client_means.add_prefix("shap_")
    return ShapResult(values=client_means, available=True)


def _autogluon_tree_model_name(predictor: TabularPredictor, candidates: Iterable[str]) -> Optional[str]:
    """Return the first tree-based model present in the leaderboard order."""
    tree_keywords = (
        "lightgbm",
        "catboost",
        "xgboost",
        "randomforest",
        "extratrees",
        "weightedensemble_l2",
    )
    for name in candidates:
        lower = name.lower()
        if any(keyword in lower for keyword in tree_keywords):
            if "weightedensemble" in lower:
                # Prefer base learners over ensembles
                continue
            return name
    return None


def compute_shap_autogluon(
    sample_df: pd.DataFrame,
    predictor_dir: Path,
) -> ShapResult:
    """Compute SHAP vectors for the AutoGluon champion if possible."""
    if sample_df.empty:
        return ShapResult(values=pd.DataFrame(), available=True)

    predictor = TabularPredictor.load(str(predictor_dir))
    leaderboard = predictor.leaderboard(silent=True)
    model_names: List[str] = leaderboard["model"].tolist()
    model_name = _autogluon_tree_model_name(predictor, model_names)

    if model_name is None:
        LOGGER.warning("No tree-based AutoGluon model suitable for SHAP found. Skipping SHAP block.")
        return ShapResult(values=pd.DataFrame(), available=False)

    LOGGER.info("Using AutoGluon model '%s' for SHAP computation.", model_name)
    model = predictor._trainer.load_model(model_name)

    transformed = predictor.transform_features(sample_df, model=model_name)
    shap_explainer = shap.TreeExplainer(model.model)
    shap_values = shap_explainer.shap_values(transformed)
    if isinstance(shap_values, list):  # pragma: no cover - older SHAP versions
        shap_matrix = np.abs(np.asarray(shap_values[-1]))
    else:
        shap_matrix = np.abs(shap_values)

    shap_df = pd.DataFrame(
        shap_matrix,
        index=sample_df.index,
        columns=transformed.columns,
    )

    client_means = shap_df.groupby(sample_df["cliente"]).mean()
    client_means = normalise_vectors(client_means)
    client_means = client_means.add_prefix("shap_")
    return ShapResult(values=client_means, available=True)


def prepare_policy_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix policy score columns and index by client."""
    if df.empty:
        return df.set_index("cliente") if "cliente" in df.columns else df
    renamed = df.rename(
        columns={
            col: col if col.startswith("policy_") else f"policy_{col}"
            for col in df.columns
            if col != "cliente"
        }
    )
    # Harmonise accidental numeric column names
    renamed = renamed.rename(columns={"policy_0": "policy_extra_0"})
    return renamed.set_index("cliente")


def prepare_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix profile columns and index by client."""
    if df.empty:
        return df.set_index("cliente") if "cliente" in df.columns else df
    renamed = df.rename(
        columns={
            col: col if col.startswith("profile_") else f"profile_{col}"
            for col in df.columns
            if col != "cliente"
        }
    )
    return renamed.set_index("cliente")


def assemble_embeddings(
    shap_result: ShapResult,
    pattern_scores: pd.DataFrame,
    profiles: pd.DataFrame,
    all_clients: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Combine SHAP, policy, and profile blocks into a single embedding matrix."""
    embedding = pd.DataFrame(index=pd.Index(all_clients, name="cliente"))

    if not shap_result.values.empty:
        embedding = embedding.join(shap_result.values, how="left")
    else:
        LOGGER.info("SHAP block is empty; clients will rely on policy/profile data.")

    if shap_result.available and not shap_result.values.empty:
        shap_clients = shap_result.values.index
        embedding["shap_available"] = embedding.index.isin(shap_clients).astype(int)
    else:
        embedding["shap_available"] = 0

    if not pattern_scores.empty:
        embedding = embedding.join(pattern_scores, how="left")
    if not profiles.empty:
        embedding = embedding.join(profiles, how="left")

    embedding = embedding.fillna(0.0)
    feature_columns = embedding.columns.tolist()
    return embedding, feature_columns


def apply_scaler_and_pca(
    embedding: pd.DataFrame,
    target_dim: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Scale the embedding matrix and project it via PCA."""
    if embedding.empty:
        raise ValueError("Cannot compute PCA on an empty embedding matrix.")

    feature_columns = embedding.columns.tolist()
    if not feature_columns:
        raise ValueError("No feature columns available for PCA transformation.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(embedding.values)

    n_samples, n_features = scaled.shape
    n_components = min(target_dim, n_features, n_samples)
    if n_components == 0:
        raise ValueError("Embedding matrix has zero dimensionality after scaling.")

    pca = PCA(n_components=n_components, random_state=42)
    projected = pca.fit_transform(scaled)

    component_columns = [f"embedding_pc{idx:02d}" for idx in range(1, n_components + 1)]
    projected_df = pd.DataFrame(projected, index=embedding.index, columns=component_columns)

    explained_variance_ratio = pca.explained_variance_ratio_.tolist()
    cumulative_variance = float(np.sum(pca.explained_variance_ratio_))

    meta = {
        "feature_columns": feature_columns,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance": cumulative_variance,
        "n_components": n_components,
    }
    return projected_df, meta


def ensure_output_dir(path: Path) -> None:
    """Create output directory if required."""
    path.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    champion_path = resolve_path(args.champion, project_root)
    champion = load_champion(champion_path)

    table_paths = {
        "train_features": resolve_path(args.train_features, project_root),
        "events": resolve_path(args.events, project_root),
        "event_items": resolve_path(args.event_items, project_root),
        "pattern_scores": resolve_path(args.pattern_scores, project_root),
        "client_profiles": resolve_path(args.client_profiles, project_root),
    }
    tables = load_tables(table_paths)

    if champion.model == "catboost":
        enriched = cat.enrich_features(tables["train_features"], tables["event_items"])
    elif champion.model == "autogluon":
        enriched = auto.enrich_features(tables["train_features"], tables["event_items"])
    else:
        raise ValueError(f"Unsupported champion model: {champion.model}")

    positives = select_positive_samples(enriched, tables["events"], args.max_samples)

    if champion.model == "catboost":
        shap_result = compute_shap_catboost(positives, project_root / "models")
    else:
        shap_result = compute_shap_autogluon(positives, project_root / "models" / "autogluon")

    pattern_scores = prepare_policy_scores(tables["pattern_scores"])
    profiles = prepare_profiles(tables["client_profiles"])

    client_index = sorted(
        {
            *shap_result.values.index.tolist(),
            *pattern_scores.index.tolist(),
            *profiles.index.tolist(),
        }
    )
    raw_embeddings, feature_columns = assemble_embeddings(
        shap_result=shap_result,
        pattern_scores=pattern_scores,
        profiles=profiles,
        all_clients=client_index,
    )

    transformed_embeddings, pca_meta = apply_scaler_and_pca(
        raw_embeddings, target_dim=64
    )

    embeddings = transformed_embeddings.reset_index().rename(columns={"index": "cliente"})
    column_order = embeddings.columns.tolist()

    output_dir = resolve_path(args.out_dir, project_root)
    ensure_output_dir(output_dir)
    embeddings_path = output_dir / "client_embeddings.parquet"
    embeddings.to_parquet(embeddings_path, compression="snappy", index=False)

    meta_path = output_dir / "client_embeddings_meta.json"
    meta_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "columns": column_order,
        "shap_model": champion.model,
        "shap_available": bool(shap_result.available and not shap_result.values.empty),
        "raw_feature_columns": feature_columns,
        "pca_components": pca_meta["n_components"],
        "explained_variance_ratio": pca_meta["explained_variance_ratio"],
        "cumulative_variance": pca_meta["cumulative_variance"],
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False))

    LOGGER.info("Wrote embeddings to %s", embeddings_path)
    LOGGER.info("Wrote metadata to %s", meta_path)

    print(f"Embedding dimension: {pca_meta['n_components']}")
    print(f"Cumulative explained variance: {pca_meta['cumulative_variance']:.4f}")
    print("Sample embeddings (top 3 rows):")
    print(embeddings.head(3).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
