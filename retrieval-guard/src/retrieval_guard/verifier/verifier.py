"""
Token-level Transformer verifier (Stage 2).

Uses a cross-encoder to compare query vs candidate at the token level,
rejecting structural near-misses that a dense retriever would accept.

The default model is a lightweight cross-encoder (~50M params) fine-tuned
for structural near-miss rejection:
    "cross-encoder/ms-marco-MiniLM-L-6-v2"  (general relevance — good baseline)

For production use, a model fine-tuned on structural near-miss pairs is
strongly preferred. See pretrained/ for the recommended model card.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

DEFAULT_VERIFIER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class StructuralVerifier:
    """
    Wraps a cross-encoder to score (query, candidate) pairs.

    score() returns a float in (−∞, +∞). Higher = more relevant.
    Pairs below `rejection_threshold` are considered structural near-misses
    and are rejected.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        rejection_threshold: Margin used for rejection. In the pipeline, docs
            scoring more than this below the top candidate are rejected.
            Default 0.1 is calibrated for cross-encoder/ms-marco-MiniLM-L-6-v2.
            Default is 0.0 — tune on your domain.
        device: "cpu", "cuda", or "mps". Auto-detected if None.
        max_length: Max token length for the verifier input.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_VERIFIER_MODEL,
        rejection_threshold: float = 0.1,
        device: Optional[str] = None,
        max_length: int = 256,
    ):
        self.model_name = model_name_or_path
        self.rejection_threshold = rejection_threshold
        self.max_length = max_length

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        logger.info(f"Loading verifier: {model_name_or_path} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

    def score(self, query: str, candidate: str) -> float:
        """Return a relevance score for a (query, candidate) pair."""
        features = self.tokenizer(
            query,
            candidate,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        features = {k: v.to(self.device) for k, v in features.items()}

        with torch.no_grad():
            logits = self.model(**features).logits

        # Most cross-encoders output a single logit for binary relevance
        return float(logits[0][0] if logits.shape[-1] == 1 else logits[0].mean())

    def score_batch(self, query: str, candidates: list[str]) -> list[float]:
        """Score a batch of candidates against the same query."""
        if not candidates:
            return []

        features = self.tokenizer(
            [query] * len(candidates),
            candidates,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        features = {k: v.to(self.device) for k, v in features.items()}

        with torch.no_grad():
            logits = self.model(**features).logits

        if logits.shape[-1] == 1:
            return [float(logit[0]) for logit in logits]
        return [float(logit.mean()) for logit in logits]

    def is_near_miss(self, query: str, candidate: str) -> bool:
        """Return True if the candidate should be rejected as a structural near-miss."""
        return self.score(query, candidate) < self.rejection_threshold
