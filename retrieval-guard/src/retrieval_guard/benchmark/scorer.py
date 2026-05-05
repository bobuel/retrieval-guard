"""
Generalization score computation.

For each NearMissPair the model passes if:
  sim(query, positive) > sim(query, hard_negative)

GeneralizationReport contains:
  - overall_score: fraction of pairs where the model ranks positive above negative
  - per_category: breakdown by category
  - failures: list of pair IDs where the model failed
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .suite import NearMissPair, get_suite, Category

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@dataclass
class GeneralizationReport:
    model_name: str
    overall_score: float                    # 0.0–1.0
    per_category: dict[str, float]          # category → score
    failure_ids: list[str]                  # pair IDs where model failed
    total_pairs: int
    passed_pairs: int

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "overall_score": round(self.overall_score, 4),
            "per_category": {k: round(v, 4) for k, v in self.per_category.items()},
            "failure_ids": self.failure_ids,
            "total_pairs": self.total_pairs,
            "passed_pairs": self.passed_pairs,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class RegressionAlert:
    fired: bool
    delta: float                            # before_score - after_score (positive = regression)
    before_score: float
    after_score: float
    threshold: float
    model_name: str
    new_failures: list[str]                 # pair IDs that newly failed after fine-tuning
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "fired": self.fired,
            "delta": round(self.delta, 4),
            "before_score": round(self.before_score, 4),
            "after_score": round(self.after_score, 4),
            "threshold": self.threshold,
            "model_name": self.model_name,
            "new_failures": self.new_failures,
            "recommendation": self.recommendation,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def run(
    model: "SentenceTransformer",
    categories: list[Category] | None = None,
    batch_size: int = 32,
) -> GeneralizationReport:
    """
    Benchmark any sentence-transformers-compatible model.

    Args:
        model: A SentenceTransformer (or compatible) model with an .encode() method.
        categories: Optional filter — only run specific test categories.
        batch_size: Batch size for embedding computation.

    Returns:
        GeneralizationReport with overall score and per-category breakdown.
    """
    pairs = get_suite(categories)

    # Collect all texts for batched encoding
    queries    = [p.query         for p in pairs]
    positives  = [p.positive      for p in pairs]
    negatives  = [p.hard_negative for p in pairs]

    all_texts = queries + positives + negatives
    embeddings = model.encode(all_texts, batch_size=batch_size, show_progress_bar=False)

    n = len(pairs)
    q_embs  = embeddings[:n]
    p_embs  = embeddings[n:2*n]
    neg_embs = embeddings[2*n:]

    category_results: dict[str, list[bool]] = {}
    failure_ids: list[str] = []

    for i, pair in enumerate(pairs):
        sim_pos = _cosine_sim(q_embs[i], p_embs[i])
        sim_neg = _cosine_sim(q_embs[i], neg_embs[i])
        passed  = sim_pos > sim_neg

        cat = pair.category
        if cat not in category_results:
            category_results[cat] = []
        category_results[cat].append(passed)

        if not passed:
            failure_ids.append(pair.id)

    per_category = {
        cat: round(sum(results) / len(results), 4)
        for cat, results in category_results.items()
    }

    passed_pairs = n - len(failure_ids)
    overall_score = passed_pairs / n if n > 0 else 0.0

    model_name = getattr(model, "_model_card_vars", {}).get("model_name", str(model.__class__.__name__))
    # Try to get name from common SentenceTransformer attributes
    for attr in ("_name_or_path", "model_name_or_path"):
        val = getattr(model, attr, None)
        if val:
            model_name = str(val)
            break

    return GeneralizationReport(
        model_name=model_name,
        overall_score=overall_score,
        per_category=per_category,
        failure_ids=failure_ids,
        total_pairs=n,
        passed_pairs=passed_pairs,
    )


def compare(
    model: "SentenceTransformer",
    before_report: GeneralizationReport,
    threshold: float = 0.05,
    categories: list[Category] | None = None,
    batch_size: int = 32,
) -> RegressionAlert:
    """
    Compare current model state against a baseline report.
    Fires a RegressionAlert if the generalization score dropped by more than `threshold`.

    Args:
        model: The (now fine-tuned) model to evaluate.
        before_report: GeneralizationReport from before fine-tuning.
        threshold: Alert fires if delta > this value (default: 0.05 = 5%).
        categories: Optional category filter (should match the before_report).
        batch_size: Encoding batch size.

    Returns:
        RegressionAlert (check .fired to see if regression was detected).
    """
    after_report = run(model, categories=categories, batch_size=batch_size)

    delta = before_report.overall_score - after_report.overall_score
    fired = delta > threshold

    new_failures = [fid for fid in after_report.failure_ids if fid not in before_report.failure_ids]

    recommendation = ""
    if fired:
        pct = round(delta * 100, 1)
        recommendation = (
            f"Generalization dropped {pct}% (threshold: {round(threshold*100,1)}%). "
            "Consider enabling TwoStagePipeline verification or reducing fine-tuning epochs. "
            "New failures concentrated in: "
            + ", ".join(
                cat for cat, score in after_report.per_category.items()
                if score < before_report.per_category.get(cat, 1.0) - threshold
            )
        )
    else:
        recommendation = "No significant regression detected. Model is safe to deploy."

    return RegressionAlert(
        fired=fired,
        delta=delta,
        before_score=before_report.overall_score,
        after_score=after_report.overall_score,
        threshold=threshold,
        model_name=after_report.model_name,
        new_failures=new_failures,
        recommendation=recommendation,
    )
