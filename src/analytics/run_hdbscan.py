"""Run HDBSCAN clustering over client embeddings."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer, StandardScaler

from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)

try:
    import hdbscan
except ImportError as exc:
    raise ImportError(
        "hdbscan package is required. Install it with 'pip install hdbscan'."
    ) from exc

def configure_logging(level: int | str | None = logging.INFO) -> Path:
    return setup_logging(level)

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster client embeddings with HDBSCAN and persist results.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/curated/client_embeddings.parquet"),
        help="Path to the client embeddings parquet file.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=30,
        help="Minimum cluster size parameter for HDBSCAN.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=15,
        help="Minimum samples parameter for HDBSCAN.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        help="Distance metric for HDBSCAN (e.g., cosine, euclidean).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/curated"),
        help="Output directory for cluster assignments and metadata.",
    )
    return parser.parse_args(argv)

def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path

def load_embeddings(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError("Embeddings table is empty; cannot cluster.")
    if "cliente" not in df.columns:
        raise ValueError("Embeddings parquet must include a 'cliente' column.")
    return df

def scale_features(features: pd.DataFrame, metric: str) -> np.ndarray:
    if features.empty:
        raise ValueError("Feature matrix is empty after dropping identifier columns.")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    scaled = np.nan_to_num(scaled, nan=0.0)
    if metric == "cosine":
        normalizer = Normalizer(norm="l2")
        scaled = normalizer.fit_transform(scaled)
    return scaled

def run_hdbscan(
    data: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
) -> hdbscan.HDBSCAN:
    if data.shape[0] < 2:
        raise ValueError("At least two samples are required for clustering.")
    metric_to_use = metric
    if metric == "cosine":
        metric_to_use = "euclidean"
        LOGGER.info(
            "Cosine metric requested; using Euclidean distance on L2-normalised embeddings."
        )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric_to_use,
        prediction_data=True,
    )
    clusterer.fit(data)
    return clusterer

def summarise_clusters(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, object]:
    total = len(labels)
    noise_mask = labels == -1
    noise_pct = float(np.sum(noise_mask) / total * 100) if total else 0.0
    unique_labels, counts = np.unique(labels[~noise_mask], return_counts=True)
    cluster_count = int(len(unique_labels))
    cluster_sizes = dict(sorted(zip(unique_labels, counts), key=lambda item: item[1], reverse=True))
    top5 = list(cluster_sizes.items())[:5]

    LOGGER.info("Clusters discovered: %s", cluster_count)
    LOGGER.info("Noise proportion: %.2f%%", noise_pct)
    if top5:
        LOGGER.info("Top-5 cluster sizes: %s", top5)
    else:
        LOGGER.info("Top-5 cluster sizes: [] (all points considered noise)")

    return {
        "cluster_count": cluster_count,
        "noise_pct": noise_pct,
        "top_clusters": top5,
    }

def write_outputs(
    output_dir: Path,
    clusters_df: pd.DataFrame,
    params: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters_path = output_dir / "client_clusters.parquet"
    clusters_df.to_parquet(clusters_path, compression="snappy", index=False)
    LOGGER.info("Wrote cluster assignments to %s", clusters_path)

    params_path = output_dir / "hdbscan_params.json"
    params_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **params,
    }
    params_path.write_text(json.dumps(params_payload, indent=2, ensure_ascii=False))
    LOGGER.info("Wrote parameter metadata to %s", params_path)

def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    embeddings_path = resolve_path(args.embeddings, project_root)
    output_dir = resolve_path(args.out_dir, project_root)

    embeddings = load_embeddings(embeddings_path)
    clientes = embeddings["cliente"].astype(str).tolist()
    feature_matrix = embeddings.drop(columns=["cliente"])
    scaled = scale_features(feature_matrix, metric=args.metric)

    clusterer = run_hdbscan(
        data=scaled,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.metric,
    )

    labels = clusterer.labels_
    probabilities = getattr(clusterer, "probabilities_", np.ones_like(labels, dtype=float))

    clusters_df = pd.DataFrame(
        {
            "cliente": clientes,
            "cluster_label": labels.astype(int),
            "membership_prob": probabilities.astype(float),
        }
    )

    summary = summarise_clusters(labels, probabilities)
    write_outputs(
        output_dir=output_dir,
        clusters_df=clusters_df,
        params={
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "metric": args.metric,
            "summary": summary,
        },
    )

    print("Cluster assignments (first 5 rows):")
    print(clusters_df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()

