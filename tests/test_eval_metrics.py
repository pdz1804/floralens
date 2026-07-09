"""Unit tests for retrieval metrics against hand-built fixtures with known answers."""
from ml.eval.metrics import (
    average_precision_at_k,
    mean_over_queries,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_hit_and_miss():
    relevance = [False, True, False, False, False]
    assert recall_at_k(relevance, 1) == 0.0
    assert recall_at_k(relevance, 2) == 1.0
    assert recall_at_k(relevance, 5) == 1.0

    no_hits = [False, False, False]
    assert recall_at_k(no_hits, 3) == 0.0


def test_precision_at_k():
    relevance = [True, True, False, False, True]
    assert precision_at_k(relevance, 1) == 1.0
    assert precision_at_k(relevance, 2) == 1.0
    assert precision_at_k(relevance, 4) == 0.5
    assert precision_at_k(relevance, 5) == 3 / 5


def test_average_precision_at_k_known_value():
    # Relevant at positions 1 and 3 (1-indexed): precisions are 1/1 and 2/3.
    relevance = [True, False, True, False]
    ap = average_precision_at_k(relevance, 4)
    expected = (1 / 1 + 2 / 3) / 2
    assert abs(ap - expected) < 1e-9


def test_average_precision_at_k_no_relevant_is_zero():
    relevance = [False, False, False]
    assert average_precision_at_k(relevance, 3) == 0.0


def test_average_precision_at_k_perfect_ranking_is_one():
    relevance = [True, True, True]
    assert average_precision_at_k(relevance, 3) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank([False, False, True, False]) == 1 / 3
    assert reciprocal_rank([True, False, False]) == 1.0
    assert reciprocal_rank([False, False, False]) == 0.0


def test_mean_over_queries():
    assert mean_over_queries([1.0, 0.0, 1.0, 1.0]) == 0.75
    assert mean_over_queries([]) == 0.0


def test_k_le_zero_raises():
    import pytest

    with pytest.raises(ValueError):
        recall_at_k([True], 0)
    with pytest.raises(ValueError):
        precision_at_k([True], -1)
    with pytest.raises(ValueError):
        average_precision_at_k([True], 0)
