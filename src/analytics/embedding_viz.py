"""Generate 2D embeddings and nearest-neighbour lists for clients."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

from src.utils.logging_setup import configure_logging as setup_logging

LOGGER = logging.getLogger(__name__)

def configure_logging(level: int | str | None = logging.INFO) -> Path:
    """Configure shared logging if not already set."""

    return setup_logging(level)

@dataclass
class ProjectionResult:
    """Container for dimensionality reduction results."""

    coordinates: np.ndarray
    method: str

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the embedding visualisation pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate 2D projection and cosine nearest neighbours for client embeddings.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/curated/client_embeddings.parquet"),
        help="Path to the client embeddings parquet file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/curated"),
        help="Directory where projection and neighbour files will be written.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=10,
        help="Number of nearest neighbours to compute per client (including padding).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dimensionality reduction.",
    )
    parser.add_argument(
        "--umap",
        action="store_true",
        help="Attempt UMAP projection (falls back to PCA if unavailable).",
    )
    return parser.parse_args(argv)

def resolve_path(path: Path, project_root: Path) -> Path:
    """Resolve relative paths against the repository root."""
    return path if path.is_absolute() else project_root / path

def load_embeddings(path: Path) -> pd.DataFrame:
    """Load client embeddings and guard against empty datasets."""
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError("Client embeddings table is empty; cannot compute projections.")
    required_columns = {"cliente"}
    if not required_columns.issubset(df.columns):
        raise ValueError("'cliente' column missing from embeddings parquet.")
    return df

def reduce_dimensionality(
    features_scaled: np.ndarray,
    features_normalised: np.ndarray,
    use_umap: bool,
    seed: int,
) -> ProjectionResult:
    """Reduce feature space to 2D using UMAP (if requested) or PCA."""
    if use_umap:
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=seed, metric="cosine")
            coords = reducer.fit_transform(features_normalised)
            return ProjectionResult(coordinates=coords, method="umap")
        except ImportError:
            LOGGER.warning("UMAP not installed; falling back to PCA.")
        except Exception as exc:
            LOGGER.warning("UMAP failed (%s); falling back to PCA.", exc)

    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(features_scaled)
    return ProjectionResult(coordinates=coords, method="pca")

def compute_neighbours(
    clients: Sequence[str],
    features_normalised: np.ndarray,
    n_neighbors: int,
) -> pd.DataFrame:
    """Compute cosine nearest neighbours and return as a DataFrame."""
    sample_count = len(clients)
    if sample_count == 0:
        raise ValueError("No clients available for neighbour computation.")

    k = min(max(n_neighbors, 1), sample_count)
    nbrs = NearestNeighbors(metric="cosine", n_neighbors=k)
    nbrs.fit(features_normalised)
    distances, indices = nbrs.kneighbors(features_normalised)

    rows: List[dict[str, object]] = []
    for client, dists, idxs in zip(clients, distances, indices):
        neighbour_ids: List[str] = []
        neighbour_dists: List[float] = []
        for dist, idx in zip(dists, idxs):
            candidate = clients[int(idx)]
            if candidate == client:
                continue
            neighbour_ids.append(candidate)
            neighbour_dists.append(float(dist))
            if len(neighbour_ids) >= n_neighbors:
                break

        while len(neighbour_ids) < n_neighbors:
            neighbour_ids.append("<NA>")
            neighbour_dists.append(float("nan"))

        rows.append(
            {
                "cliente": client,
                "neighbors": neighbour_ids,
                "distances": neighbour_dists,
            }
        )

    return pd.DataFrame(rows)

def write_outputs(
    output_dir: Path,
    projection_df: pd.DataFrame,
    neighbours_df: pd.DataFrame,
) -> Tuple[Path, Path]:
    """Persist projection and neighbour dataframes to parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    projection_path = output_dir / "client_embedding_2d.parquet"
    neighbours_path = output_dir / "client_neighbors.parquet"

    projection_df.to_parquet(projection_path, compression="snappy", index=False)
    neighbours_df.to_parquet(neighbours_path, compression="snappy", index=False)

    LOGGER.info("Saved 2D projection to %s", projection_path)
    LOGGER.info("Saved neighbour index to %s", neighbours_path)

    return projection_path, neighbours_path

def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    embeddings_path = resolve_path(args.embeddings, project_root)
    output_dir = resolve_path(args.out_dir, project_root)

    embeddings = load_embeddings(embeddings_path)
    clients = embeddings["cliente"].astype(str).tolist()
    feature_df = embeddings.drop(columns="cliente")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_df)
    features_normalised = normalize(features_scaled)

    projection = reduce_dimensionality(
        features_scaled=features_scaled,
        features_normalised=features_normalised,
        use_umap=args.umap,
        seed=args.seed,
    )

    projection_df = pd.DataFrame(
        {
            "cliente": clients,
            "x": projection.coordinates[:, 0],
            "y": projection.coordinates[:, 1],
        }
    )

    neighbours_df = compute_neighbours(
        clients=clients,
        features_normalised=features_normalised,
        n_neighbors=args.neighbors,
    )

    projection_path, neighbours_path = write_outputs(
        output_dir=output_dir,
        projection_df=projection_df,
        neighbours_df=neighbours_df,
    )

    print(f"Projection method: {projection.method.upper()}")
    print("Client embedding 2D (first 3 rows):")
    print(projection_df.head(3).to_string(index=False))
    print("\nClient neighbours (first 3 rows):")
    preview = neighbours_df.head(3).copy()
    preview["neighbors"] = preview["neighbors"].apply(lambda lst: lst[:3])
    preview["distances"] = preview["distances"].apply(lambda lst: lst[:3])
    print(preview.to_string(index=False))

if __name__ == "__main__":
    main()

