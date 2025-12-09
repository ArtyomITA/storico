"""Generate interpretable summaries for HDBSCAN clusters."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from numbers import Real
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples

from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)

POLICY_COLUMNS: Tuple[str, ...] = (
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
PERIODICITY_COLUMNS: Tuple[str, ...] = (
    "periodicity_weekly",
    "periodicity_biweekly",
    "periodicity_monthly",
    "dow_concentration",
    "dom_concentration",
)
CADENCE_PRIORITY_COLUMNS: Tuple[str, ...] = (
    "periodicity_monthly",
    "periodicity_weekly",
    "periodicity_biweekly",
)
CARDINALITY_COLUMNS: Tuple[str, ...] = (
    "P_k1",
    "P_k2",
    "P_kge3",
    "k_mode",
    "k_entropy",
)

HUMAN_LABELS: Dict[str, str] = {
    "periodicity_weekly": "Settimanale",
    "periodicity_biweekly": "Bi-settimanale",
    "periodicity_monthly": "Mensile",
    "dow_concentration": "DOW concentrato",
    "dom_concentration": "DOM concentrato",
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
    "P_k1": "k=1",
    "P_k2": "k=2",
    "P_kge3": "k>=3",
    "k_mode": "k_mode",
    "k_entropy": "k_entropy",
}

CARDINALITY_SHORT_LABELS: Dict[str, str] = {
    "P_k1": "1",
    "P_k2": "2",
    "P_kge3": "≥3",
}
CLUSTER_OUTPUT = Path("data/curated/cluster_summary.parquet")

def configure_logging(level: int | str | None = logging.INFO) -> Path:
    return setup_logging(level)

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create cluster profiles from client embeddings and metadata.",
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=Path("data/curated/client_clusters.parquet"),
        help="Parquet with cluster assignments (cliente, cluster_label, membership_prob).",
    )
    parser.add_argument(
        "--pattern-scores",
        type=Path,
        default=Path("data/curated/client_pattern_scores.parquet"),
        help="Per-client pattern and policy scores parquet.",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("data/curated/client_profiles.parquet"),
        help="Per-client profile features parquet.",
    )
    parser.add_argument(
        "--embedding-2d",
        type=Path,
        default=Path("data/curated/client_embedding_2d.parquet"),
        help="Optional 2D embedding projection parquet (cliente, x, y).",
    )
    parser.add_argument(
        "--neighbors",
        type=Path,
        default=Path("data/curated/client_neighbors.parquet"),
        help="Optional nearest-neighbour distances parquet for density fallback.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=CLUSTER_OUTPUT,
        help="Destination parquet file for cluster summaries.",
    )
    return parser.parse_args(argv)

def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path

def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_parquet(path)

def read_optional_table(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        LOGGER.warning("Optional input not found and will be skipped: %s", path)
        return None
    return pd.read_parquet(path)

def identify_macro_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col.startswith("share_tipo_documento_")]

def compute_iqr(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    q75 = series.quantile(0.75)
    q25 = series.quantile(0.25)
    return float(q75 - q25)

def prepare_dataset(
    clusters: pd.DataFrame,
    pattern_scores: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    clusters_local = clusters.copy()
    if "cluster_label" in clusters_local.columns:
        clusters_local["cluster_label"] = pd.to_numeric(
            clusters_local["cluster_label"], errors="coerce"
        ).fillna(-1)
    else:
        clusters_local["cluster_label"] = -1
    clusters_local["cluster_label"] = clusters_local["cluster_label"].astype(int)

    pattern_scores = pattern_scores.drop(columns=[col for col in ("0",) if col in pattern_scores.columns])
    merged = (
        clusters_local.merge(pattern_scores, on="cliente", how="left")
        .merge(profiles, on="cliente", how="left")
    )
    missing = merged[merged.isna().any(axis=1)]
    if not missing.empty:
        LOGGER.warning("Dropping %s clients with incomplete profiling data.", len(missing))
        merged = merged.dropna()
    return merged

def fetch_indicator_columns(df: pd.DataFrame) -> List[str]:
    macro_cols = identify_macro_columns(df)
    indicator_cols = list(
        {
            *POLICY_COLUMNS,
            *PERIODICITY_COLUMNS,
            *CARDINALITY_COLUMNS,
            *tuple(macro_cols),
        }
    )
    indicator_cols = [col for col in indicator_cols if col in df.columns]
    return indicator_cols

def compute_silhouette_map(
    data: pd.DataFrame,
    embedding: Optional[pd.DataFrame],
) -> Dict[str, float]:
    if embedding is None or embedding.empty:
        return {}

    merged = (
        data[["cliente", "cluster_label"]]
        .merge(embedding[["cliente", "x", "y"]], on="cliente", how="inner")
        .dropna(subset=["cluster_label", "x", "y"])
    )
    if merged.empty:
        return {}

    merged["cluster_label"] = pd.to_numeric(merged["cluster_label"], errors="coerce")
    merged = merged.dropna(subset=["cluster_label"])
    merged["cluster_label"] = merged["cluster_label"].astype(int)

    counts = merged["cluster_label"].value_counts()
    valid_labels = counts[counts > 1].index.tolist()
    merged = merged[merged["cluster_label"].isin(valid_labels)]
    merged = merged[merged["cluster_label"] != -1]

    if merged.empty or merged["cluster_label"].nunique() < 2:
        return {}

    try:
        scores = silhouette_samples(
            merged[["x", "y"]].to_numpy(), merged["cluster_label"].to_numpy()
        )
    except ValueError as exc:
        LOGGER.warning("Unable to compute silhouette scores: %s", exc)
        return {}

    clients = merged["cliente"].tolist()
    value_array = np.asarray(scores, dtype=float)
    return {cliente: float(value) for cliente, value in zip(clients, value_array.tolist())}

def compute_density_map(neighbors: Optional[pd.DataFrame]) -> Dict[str, float]:
    if neighbors is None or neighbors.empty:
        return {}
    density: Dict[str, float] = {}
    for row in neighbors.itertuples(index=False):
        distances = getattr(row, "distances", [])
        if distances is None:
            continue
        try:
            distance_list = list(distances)
        except TypeError:
            continue
        finite_dists = [
            float(dist)
            for dist in distance_list
            if isinstance(dist, Real) and not np.isnan(float(dist)) and float(dist) >= 0.0
        ]
        if not finite_dists:
            continue
        mean_dist = float(np.mean(finite_dists[:5]))
        density[str(getattr(row, "cliente"))] = float(1.0 / (1.0 + mean_dist))
    return density

def policy_dispersion_confidence(policy_means: Dict[str, float]) -> float:
    scores = [value for value in policy_means.values() if not np.isnan(value)]
    if not scores:
        return 0.0
    ordered = sorted(scores, reverse=True)
    if len(ordered) == 1:
        dispersion = ordered[0]
    else:
        dispersion = ordered[0] - ordered[1]
    return float(np.clip(dispersion, 0.0, 1.0))

def derive_cluster_confidence(
    members: Iterable[str],
    silhouette_map: Dict[str, float],
    density_map: Dict[str, float],
    policy_means: Dict[str, float],
) -> float:
    silhouette_values = [
        float(silhouette_map[cliente])
        for cliente in members
        if cliente in silhouette_map and not np.isnan(silhouette_map[cliente])
    ]
    if silhouette_values:
        mean_silhouette = float(np.mean(silhouette_values))
        normalised = (mean_silhouette + 1.0) / 2.0
        return float(np.clip(normalised, 0.0, 1.0))

    density_values = [
        float(density_map[cliente])
        for cliente in members
        if cliente in density_map and not np.isnan(density_map[cliente])
    ]
    if density_values:
        return float(np.clip(float(np.mean(density_values)), 0.0, 1.0))

    return policy_dispersion_confidence(policy_means)

def select_cadence_label(cluster_means: pd.Series) -> Optional[str]:
    best_col: Optional[str] = None
    best_value = -np.inf
    for column in CADENCE_PRIORITY_COLUMNS:
        if column not in cluster_means:
            continue
        value = float(cluster_means[column])
        if np.isnan(value):
            continue
        if value > best_value:
            best_value = value
            best_col = column
    if best_col is None or best_value <= 0:
        return None
    return HUMAN_LABELS.get(best_col, best_col)

def select_policy_label(cluster_means: pd.Series) -> Optional[str]:
    best_col: Optional[str] = None
    best_value = -np.inf
    for column in POLICY_COLUMNS:
        if column not in cluster_means:
            continue
        value = float(cluster_means[column])
        if np.isnan(value):
            continue
        if value > best_value:
            best_value = value
            best_col = column
    if best_col is None or best_value <= 0:
        return None
    return HUMAN_LABELS.get(best_col, best_col)

def build_cardinality_tag(cluster_means: pd.Series) -> Optional[str]:
    values: List[Tuple[str, float]] = []
    for column in ("P_k1", "P_k2", "P_kge3"):
        if column not in cluster_means:
            continue
        value = float(cluster_means[column])
        if np.isnan(value):
            continue
        values.append((column, value))
    if not values:
        return None
    values.sort(key=lambda item: item[1], reverse=True)
    if len(values) >= 2 and values[1][1] > 0:
        first = CARDINALITY_SHORT_LABELS.get(values[0][0], values[0][0])
        second = CARDINALITY_SHORT_LABELS.get(values[1][0], values[1][0])
        return f"k={first}–{second}"
    first = CARDINALITY_SHORT_LABELS.get(values[0][0], values[0][0])
    return f"k={first}"

def compose_cluster_label(
    cadence_label: Optional[str],
    policy_label: Optional[str],
    cardinality_tag: Optional[str],
) -> str:
    parts: List[str] = []
    if cadence_label and policy_label:
        parts.append(f"{cadence_label}+{policy_label}")
    else:
        if cadence_label:
            parts.append(cadence_label)
        if policy_label:
            parts.append(policy_label)
    if cardinality_tag:
        parts.append(cardinality_tag)
    if not parts:
        return "Profilo neutro"
    if len(parts) == 1:
        return parts[0]
    headline = parts[0]
    tail = ", ".join(parts[1:])
    return f"{headline}, {tail}" if tail else headline
def create_summary(
    data: pd.DataFrame,
    indicator_cols: List[str],
    output_path: Path,
    silhouette_map: Dict[str, float],
    density_map: Dict[str, float],
) -> pd.DataFrame:
    summaries: List[Dict[str, object]] = []

    grouped = data.groupby("cluster_label", dropna=False)
    for cluster_label, group in grouped:
        if isinstance(cluster_label, Real):
            label_value = int(float(cast(Real, cluster_label)))
        else:
            label_value = -1

        record: Dict[str, object] = {
            "cluster_label_id": label_value,
            "size": int(len(group)),
        }

        for col in indicator_cols:
            mean_value = float(group[col].mean()) if not group[col].empty else float("nan")
            record[f"{col}_mean"] = mean_value
            record[f"{col}_iqr"] = compute_iqr(group[col])

        cluster_means = group[indicator_cols].mean()
        policy_means = {
            column: float(cluster_means.get(column, float("nan")))
            for column in POLICY_COLUMNS
            if column in cluster_means
        }
        cadence_label = select_cadence_label(cluster_means)
        dominant_policy = select_policy_label(cluster_means)
        cardinality_tag = build_cardinality_tag(cluster_means)

        if cluster_label == -1:
            cluster_label_text = "Noise (non-clustered)"
            dominant_pattern = "Noise"
            confidence = 0.0
        else:
            cluster_label_text = compose_cluster_label(
                cadence_label,
                dominant_policy,
                cardinality_tag,
            )
            dominant_pattern = dominant_policy or "n.d."
            members = group["cliente"].astype(str).tolist()
            confidence = derive_cluster_confidence(
                members,
                silhouette_map,
                density_map,
                policy_means,
            )

        record["label_text"] = cluster_label_text
        record["cluster_label"] = cluster_label_text
        record["dominant_pattern"] = dominant_pattern
        record["confidence"] = confidence
        record["cluster_confidence"] = confidence

        summaries.append(record)

    summary_df = pd.DataFrame(summaries)
    summary_df["generated_at"] = datetime.now(timezone.utc).isoformat()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(output_path, compression="snappy", index=False)
    LOGGER.info("Cluster summary saved to %s", output_path)
    return summary_df

def print_samples(summary_df: pd.DataFrame, max_rows: int = 3) -> None:
    preview = summary_df.sort_values("size", ascending=False).head(max_rows)
    cols_to_show = ["cluster_label_id", "size", "cluster_label", "cluster_confidence"]
    extra_cols = [col for col in summary_df.columns if col.endswith("_mean")]
    cols_to_show.extend(extra_cols[:3])
    print("Cluster summaries (top rows):")
    print(preview[cols_to_show].to_string(index=False))

def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    clusters_path = resolve_path(args.clusters, project_root)
    pattern_path = resolve_path(args.pattern_scores, project_root)
    profiles_path = resolve_path(args.profiles, project_root)
    embedding_path = resolve_path(args.embedding_2d, project_root)
    neighbors_path = resolve_path(args.neighbors, project_root)
    output_path = resolve_path(args.out, project_root)

    clusters = read_table(clusters_path)
    pattern_scores = read_table(pattern_path)
    profiles = read_table(profiles_path)
    embedding_2d = read_optional_table(embedding_path)
    neighbors = read_optional_table(neighbors_path)

    dataset = prepare_dataset(clusters, pattern_scores, profiles)
    if dataset.empty:
        raise ValueError("Merged dataset is empty; ensure inputs contain overlapping clients.")

    indicator_cols = fetch_indicator_columns(dataset)
    if not indicator_cols:
        raise ValueError("No indicator columns found for summarisation.")

    silhouette_map = compute_silhouette_map(dataset, embedding_2d)
    density_map = compute_density_map(neighbors)

    summary_df = create_summary(
        dataset,
        indicator_cols,
        output_path,
        silhouette_map,
        density_map,
    )
    print_samples(summary_df)

if __name__ == "__main__":
    main()

