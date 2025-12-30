"""
Real-world dataset utilities for benchmarking.

Supports downloading and loading standard benchmark datasets:
- SIFT1M: 1M SIFT image descriptors (128D)
- GIST1M: 1M GIST image descriptors (960D)
- GloVe: Word embeddings (50-300D)

These datasets are widely used in vector search research papers.
"""

import os
import urllib.request
import tarfile
import zipfile
import struct
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


# Dataset configurations
DATASETS = {
    "sift1m": {
        "name": "SIFT1M",
        "description": "1M SIFT image descriptors (128D)",
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        "filename": "sift.tar.gz",
        "dimension": 128,
        "files": {
            "base": "sift/sift_base.fvecs",
            "query": "sift/sift_query.fvecs",
            "groundtruth": "sift/sift_groundtruth.ivecs",
        }
    },
    "gist1m": {
        "name": "GIST1M",
        "description": "1M GIST image descriptors (960D)",
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
        "filename": "gist.tar.gz",
        "dimension": 960,
        "files": {
            "base": "gist/gist_base.fvecs",
            "query": "gist/gist_query.fvecs",
            "groundtruth": "gist/gist_groundtruth.ivecs",
        }
    },
    "glove-50d": {
        "name": "GloVe 50D",
        "description": "GloVe word embeddings (50D)",
        "url": "http://nlp.stanford.edu/data/glove.6B.zip",
        "filename": "glove.6B.zip",
        "dimension": 50,
        "files": {
            "base": "glove.6B.50d.txt",
        }
    },
}


def get_data_dir() -> Path:
    """Get the directory for storing benchmark datasets."""
    # Store in tests/benchmarks/data/
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def download_dataset(dataset_name: str, force: bool = False) -> None:
    """
    Download a benchmark dataset.

    Args:
        dataset_name: Name of dataset (e.g., "sift1m", "gist1m", "glove-50d")
        force: If True, re-download even if already exists

    Example:
        >>> download_dataset("sift1m")
        Downloading SIFT1M...
        Download complete: /path/to/sift.tar.gz
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    config = DATASETS[dataset_name]
    data_dir = get_data_dir()
    filepath = data_dir / config["filename"]

    # Check if already downloaded
    if filepath.exists() and not force:
        print(f"Dataset already downloaded: {filepath}")
        return

    # Download
    print(f"Downloading {config['name']} from {config['url']}...")
    print(f"This may take a while (large file)...")

    try:
        urllib.request.urlretrieve(config["url"], filepath)
        print(f"✓ Download complete: {filepath}")

        # Extract if archive
        if filepath.suffix in ['.gz', '.tar', '.zip']:
            print(f"Extracting {filepath.name}...")
            if filepath.suffix == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zf:
                    zf.extractall(data_dir)
            else:
                with tarfile.open(filepath, 'r:gz') as tf:
                    tf.extractall(data_dir)
            print(f"✓ Extraction complete")

    except Exception as e:
        print(f"✗ Download failed: {e}")
        if filepath.exists():
            filepath.unlink()
        raise


def read_fvecs(filepath: str) -> np.ndarray:
    """
    Read .fvecs file format (used by SIFT/GIST datasets).

    Format: [dim (4 bytes)] [vector components (dim * 4 bytes)] [repeat...]
    Each vector is: int32 dimension + float32 values

    Args:
        filepath: Path to .fvecs file

    Returns:
        Array of vectors (n_vectors x dimension)
    """
    vectors = []

    with open(filepath, 'rb') as f:
        while True:
            # Read dimension (int32)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break

            dim = struct.unpack('i', dim_bytes)[0]

            # Read vector components (dim * float32)
            vec_bytes = f.read(dim * 4)
            vec = struct.unpack(f'{dim}f', vec_bytes)
            vectors.append(vec)

    return np.array(vectors, dtype=np.float32)


def read_ivecs(filepath: str) -> np.ndarray:
    """
    Read .ivecs file format (used for ground truth).

    Same format as .fvecs but with int32 values instead of float32.

    Args:
        filepath: Path to .ivecs file

    Returns:
        Array of integer vectors (n_vectors x dimension)
    """
    vectors = []

    with open(filepath, 'rb') as f:
        while True:
            # Read dimension (int32)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break

            dim = struct.unpack('i', dim_bytes)[0]

            # Read vector components (dim * int32)
            vec_bytes = f.read(dim * 4)
            vec = struct.unpack(f'{dim}i', vec_bytes)
            vectors.append(vec)

    return np.array(vectors, dtype=np.int32)


def read_glove(filepath: str, max_vectors: Optional[int] = None) -> Tuple[np.ndarray, list]:
    """
    Read GloVe text file format.

    Format: word float1 float2 ... floatN

    Args:
        filepath: Path to GloVe .txt file
        max_vectors: Optional limit on number of vectors to read

    Returns:
        Tuple of (vectors, words)
    """
    vectors = []
    words = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_vectors and i >= max_vectors:
                break

            parts = line.strip().split()
            word = parts[0]
            vec = [float(x) for x in parts[1:]]

            words.append(word)
            vectors.append(vec)

    return np.array(vectors, dtype=np.float32), words


def load_dataset(
    dataset_name: str,
    subset: str = "small",
    download_if_missing: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load a real-world benchmark dataset.

    Args:
        dataset_name: Name of dataset (e.g., "sift1m", "gist1m")
        subset: Size to load:
            - "tiny": 1k base vectors, 100 queries
            - "small": 10k base vectors, 100 queries
            - "medium": 100k base vectors, 1k queries
            - "full": All vectors
        download_if_missing: Auto-download if not present

    Returns:
        Tuple of (base_vectors, query_vectors, ground_truth)
        ground_truth is None if not available

    Example:
        >>> vectors, queries, gt = load_dataset("sift1m", subset="small")
        >>> print(f"Loaded {len(vectors)} base vectors, {len(queries)} queries")
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    config = DATASETS[dataset_name]
    data_dir = get_data_dir()

    # Check if dataset exists
    base_path = data_dir / config["files"]["base"]
    if not base_path.exists():
        if download_if_missing:
            print(f"Dataset not found. Downloading {config['name']}...")
            download_dataset(dataset_name)
        else:
            raise FileNotFoundError(
                f"Dataset not found: {base_path}\n"
                f"Run download_dataset('{dataset_name}') first"
            )

    # Load base vectors
    print(f"Loading {config['name']} base vectors...")
    if dataset_name.startswith("glove"):
        base_vectors, _ = read_glove(str(base_path))
    else:
        base_vectors = read_fvecs(str(base_path))

    # Load query vectors
    query_path = data_dir / config["files"].get("query", "")
    if query_path.exists():
        print(f"Loading query vectors...")
        query_vectors = read_fvecs(str(query_path))
    else:
        # Generate random queries from base vectors
        print(f"Generating query vectors from base...")
        n_queries = 100
        indices = np.random.choice(len(base_vectors), n_queries, replace=False)
        query_vectors = base_vectors[indices]

    # Load ground truth if available
    gt_path = data_dir / config["files"].get("groundtruth", "")
    ground_truth = None
    if gt_path.exists():
        print(f"Loading ground truth...")
        ground_truth = read_ivecs(str(gt_path))

    # Apply subset sampling
    subset_sizes = {
        "tiny": (1_000, 100),
        "small": (10_000, 100),
        "medium": (100_000, 1_000),
        "full": (len(base_vectors), len(query_vectors))
    }

    if subset not in subset_sizes:
        raise ValueError(f"Unknown subset: {subset}. Use: {list(subset_sizes.keys())}")

    n_base, n_query = subset_sizes[subset]
    n_base = min(n_base, len(base_vectors))
    n_query = min(n_query, len(query_vectors))

    base_vectors = base_vectors[:n_base]
    query_vectors = query_vectors[:n_query]

    if ground_truth is not None:
        ground_truth = ground_truth[:n_query]

    print(f"✓ Loaded {len(base_vectors)} base vectors, {len(query_vectors)} queries")
    return base_vectors, query_vectors, ground_truth


def list_available_datasets() -> None:
    """Print information about available datasets."""
    print("Available Benchmark Datasets:")
    print("=" * 60)

    for name, config in DATASETS.items():
        data_dir = get_data_dir()
        base_path = data_dir / config["files"]["base"]
        status = "✓ Downloaded" if base_path.exists() else "✗ Not downloaded"

        print(f"\n{name}:")
        print(f"  Name: {config['name']}")
        print(f"  Description: {config['description']}")
        print(f"  Dimension: {config['dimension']}")
        print(f"  Status: {status}")

    print("\n" + "=" * 60)
    print(f"Data directory: {get_data_dir()}")
    print("\nTo download: download_dataset('dataset_name')")
    print("To load: load_dataset('dataset_name', subset='small')")


if __name__ == "__main__":
    # Show available datasets
    list_available_datasets()
