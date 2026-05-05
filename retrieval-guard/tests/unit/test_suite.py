"""Unit tests for the benchmark test suite."""

import pytest
from retrieval_guard.benchmark.suite import get_suite, BUILTIN_PAIRS, Category


def test_builtin_pairs_exist():
    assert len(BUILTIN_PAIRS) > 0


def test_all_categories_present():
    categories = {p.category for p in BUILTIN_PAIRS}
    assert "negation" in categories
    assert "role_reversal" in categories
    assert "spatial" in categories
    assert "binding" in categories


def test_filter_by_category():
    negation_pairs = get_suite(categories=["negation"])
    assert all(p.category == "negation" for p in negation_pairs)
    assert len(negation_pairs) > 0


def test_pair_fields_populated():
    for pair in BUILTIN_PAIRS:
        assert pair.query
        assert pair.positive
        assert pair.hard_negative
        assert pair.category
        assert pair.id
        # positive and hard_negative must differ
        assert pair.positive != pair.hard_negative


def test_ids_unique():
    ids = [p.id for p in BUILTIN_PAIRS]
    assert len(ids) == len(set(ids)), "Pair IDs must be unique"
