import pytest

from src.index import BruteForceIndex, HNSWIndex, SegmentedHNSWIndex, VectorIndex


def test_index_package_exports_public_index_types():
    assert issubclass(BruteForceIndex, VectorIndex)
    assert HNSWIndex.__name__ == "HNSWIndex"
    assert SegmentedHNSWIndex.__name__ == "SegmentedHNSWIndex"


def test_hnsw_index_rejects_dot_product_metric():
    with pytest.raises(ValueError, match="Unsupported metric"):
        HNSWIndex(metric="dot_product")
