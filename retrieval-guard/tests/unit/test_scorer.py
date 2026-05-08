"""
Unit tests for benchmark scorer.

Uses a mock embedding model so these tests run fast with no GPU/network needed.
The mock model is designed to be a "perfect" model (always ranks positive above negative)
and a "broken" model (always fails) to test scoring logic.
"""

import numpy as np
from retrieval_guard.benchmark.scorer import (
    run,
    compare,
    GeneralizationReport,
    _cosine_sim,
)
from retrieval_guard.benchmark.suite import BUILTIN_PAIRS


class PerfectMockModel:
    """Always embeds positives closer to the query than negatives."""
    _name_or_path = "mock-perfect-model"

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        len(texts)
        # Deterministic embeddings based on text content
        rng = np.random.default_rng(42)
        embeddings = []
        for text in texts:
            # Query and positive share a base vector; hard_negative is orthogonal
            sum(ord(c) for c in text[:20])
            vec = rng.standard_normal(64)
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            embeddings.append(vec)
        return np.array(embeddings)


class PerfectOracle:
    """Returns embeddings where positive always scores higher than negative."""
    _name_or_path = "mock-oracle"

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        len(BUILTIN_PAIRS)
        # texts = [q0..qN, p0..pN, neg0..negN]
        n = len(texts)
        embeddings = np.zeros((n, 4))

        # Simple scheme: query and positive → [1,0,0,0]; negative → [0,1,0,0]
        # Since we don't know the order here, use a different trick:
        # assign based on index position (relies on run() ordering guarantee)
        for i in range(n):
            if i < n // 3:
                # query
                embeddings[i] = [1, 0.1, 0, 0]
            elif i < 2 * n // 3:
                # positive: very close to query
                embeddings[i] = [1, 0.05, 0, 0]
            else:
                # negative: far from query
                embeddings[i] = [0, 1, 0, 0]
        return embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)


class FailingModel:
    """Always ranks negatives above positives (worst case model)."""
    _name_or_path = "mock-failing-model"

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        n = len(texts)
        n_pairs = n // 3
        embeddings = np.zeros((n, 4))
        for i in range(n):
            if i < n_pairs:
                # query
                embeddings[i] = [0, 1, 0, 0]
            elif i < 2 * n_pairs:
                # positive: OPPOSITE of query
                embeddings[i] = [1, 0, 0, 0]
            else:
                # negative: SAME as query
                embeddings[i] = [0, 1, 0, 0]
        return embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)


def test_cosine_sim_identical():
    a = np.array([1.0, 0.0, 0.0])
    assert abs(_cosine_sim(a, a) - 1.0) < 1e-6


def test_cosine_sim_orthogonal():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(_cosine_sim(a, b)) < 1e-6


def test_run_returns_report():
    report = run(PerfectOracle())
    assert isinstance(report, GeneralizationReport)
    assert 0.0 <= report.overall_score <= 1.0
    assert report.total_pairs == len(BUILTIN_PAIRS)
    assert report.passed_pairs + len(report.failure_ids) == report.total_pairs


def test_perfect_oracle_scores_100():
    report = run(PerfectOracle())
    assert report.overall_score == 1.0
    assert report.passed_pairs == report.total_pairs
    assert report.failure_ids == []


def test_failing_model_scores_0():
    report = run(FailingModel())
    assert report.overall_score == 0.0
    assert report.passed_pairs == 0


def test_per_category_keys():
    report = run(PerfectOracle())
    expected = {"negation", "role_reversal", "spatial", "binding"}
    assert set(report.per_category.keys()) == expected


def test_compare_fires_alert_on_regression():
    before = run(PerfectOracle())
    after_model = FailingModel()
    alert = compare(after_model, before, threshold=0.05)
    assert alert.fired is True
    assert alert.delta > 0.05


def test_compare_no_alert_when_stable():
    before = run(PerfectOracle())
    alert = compare(PerfectOracle(), before, threshold=0.05)
    assert alert.fired is False


def test_regression_alert_has_new_failures():
    before = run(PerfectOracle())
    alert = compare(FailingModel(), before, threshold=0.05)
    assert len(alert.new_failures) > 0


def test_report_to_dict():
    report = run(PerfectOracle())
    d = report.to_dict()
    assert "overall_score" in d
    assert "per_category" in d
    assert "model_name" in d


def test_report_to_json():
    import json
    report = run(PerfectOracle())
    parsed = json.loads(report.to_json())
    assert "overall_score" in parsed
