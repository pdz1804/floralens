"""Unit tests for the in-memory cosine vector store."""
import numpy as np
import pytest

from ml.index.vector_store import VectorStore


def test_query_ranks_by_cosine_similarity():
    store = VectorStore(model_version="test")
    store.add("a", np.array([1.0, 0.0, 0.0]), {"label": 0})
    store.add("b", np.array([0.9, 0.1, 0.0]), {"label": 0})  # close to query
    store.add("c", np.array([0.0, 1.0, 0.0]), {"label": 1})  # orthogonal
    store.add("d", np.array([-1.0, 0.0, 0.0]), {"label": 2})  # opposite

    results = store.query(np.array([1.0, 0.0, 0.0]), top_k=4)
    ids_in_order = [r.id for r in results]
    assert ids_in_order == ["a", "b", "c", "d"]
    assert results[0].score == pytest.approx(1.0)
    assert results[-1].score == pytest.approx(-1.0)


def test_query_excludes_ids():
    store = VectorStore(model_version="test")
    store.add("self", np.array([1.0, 0.0]), {"label": 0})
    store.add("other", np.array([0.99, 0.01]), {"label": 0})

    results = store.query(np.array([1.0, 0.0]), top_k=5, exclude_ids={"self"})
    assert [r.id for r in results] == ["other"]


def test_query_top_k_limits_results():
    store = VectorStore(model_version="test")
    for i in range(10):
        store.add(f"id{i}", np.array([1.0, i * 0.01]), {"label": i})
    results = store.query(np.array([1.0, 0.0]), top_k=3)
    assert len(results) == 3


def test_add_rejects_duplicate_id():
    store = VectorStore(model_version="test")
    store.add("a", np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        store.add("a", np.array([0.0, 1.0]))


def test_add_rejects_zero_vector():
    store = VectorStore(model_version="test")
    with pytest.raises(ValueError):
        store.add("a", np.array([0.0, 0.0]))


def test_empty_store_query_returns_empty_list():
    store = VectorStore(model_version="test")
    assert store.query(np.array([1.0, 0.0]), top_k=5) == []


def test_vectors_are_normalized_defensively():
    store = VectorStore(model_version="test")
    store.add("a", np.array([2.0, 0.0, 0.0]))  # not unit norm
    results = store.query(np.array([1.0, 0.0, 0.0]), top_k=1)
    assert results[0].score == pytest.approx(1.0)
