"""
TwoStagePipeline — drop-in retriever wrapper.

Stage 1: Standard dense retrieval (any retriever with a .get_relevant_documents() method)
Stage 2: StructuralVerifier filters out structural near-misses at the token level

Compatible with:
  - LangChain BaseRetriever (via adapters/langchain.py)
  - LlamaIndex BaseRetriever (via adapters/llamaindex.py)
  - Any raw callable that returns (text, score) pairs

Usage:
    from retrieval_guard.verifier import TwoStagePipeline
    from retrieval_guard.verifier.verifier import StructuralVerifier

    verifier = StructuralVerifier()
    pipeline = TwoStagePipeline(retriever=my_retriever, verifier=verifier)
    results = pipeline.retrieve(query="Is the system secure?", top_k=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from .verifier import StructuralVerifier

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    text: str
    score: float                     # Stage 1 (dense) score
    verifier_score: Optional[float]  # Stage 2 score (None if not verified)
    accepted: bool                   # False = rejected by verifier
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@runtime_checkable
class Retriever(Protocol):
    """Minimal retriever protocol. Anything with retrieve() or get_relevant_documents() works."""
    def retrieve(self, query: str, top_k: int = 10) -> list[Any]: ...


class TwoStagePipeline:
    """
    Drop-in wrapper that adds Stage 2 structural verification to any retriever.

    Args:
        retriever: Any retriever. Supports:
            - Objects with .retrieve(query, top_k) → list of RetrievedDocument
            - LangChain retrievers with .get_relevant_documents(query)
            - LlamaIndex retrievers with .retrieve(query)
            - Raw callables: fn(query, top_k) → list[str | (str, float)]
        verifier: A StructuralVerifier instance.
        top_k: Number of candidates to fetch from Stage 1.
        verification_mode: "full" | "light" | "none"
            - full: verify all candidates (default)
            - light: only verify candidates with Stage 1 score below `light_threshold`
            - none: skip Stage 2 (passthrough — useful for A/B testing)
        light_threshold: Score below which candidates are verified in "light" mode.
        rejection_threshold: Override the verifier's default rejection threshold.
    """

    def __init__(
        self,
        retriever: Any,
        verifier: StructuralVerifier,
        top_k: int = 10,
        verification_mode: str = "full",
        light_threshold: float = 0.5,
        rejection_threshold: Optional[float] = None,
    ):
        self.retriever = retriever
        self.verifier = verifier
        self.top_k = top_k
        self.verification_mode = verification_mode
        self.light_threshold = light_threshold

        if rejection_threshold is not None:
            self.verifier.rejection_threshold = rejection_threshold

        self._retriever_type = self._detect_retriever_type()

    def _detect_retriever_type(self) -> str:
        """Detect retriever interface."""
        if callable(self.retriever) and not hasattr(self.retriever, "__class__"):
            return "callable"
        # LangChain v0.2+ uses invoke(); older versions use get_relevant_documents()
        if hasattr(self.retriever, "invoke") and hasattr(self.retriever, "vectorstore"):
            return "langchain_invoke"
        if hasattr(self.retriever, "get_relevant_documents"):
            return "langchain"
        if hasattr(self.retriever, "retrieve"):
            return "retrieve_protocol"
        if callable(self.retriever):
            return "callable"
        raise ValueError(
            f"Unsupported retriever type: {type(self.retriever)}. "
            "Must have .invoke(), .retrieve(), .get_relevant_documents(), or be callable."
        )

    def _call_retriever(self, query: str) -> list[RetrievedDocument]:
        """Normalize retriever output to list[RetrievedDocument]."""
        if self._retriever_type == "langchain_invoke":
            docs = self.retriever.invoke(query)
            return [
                RetrievedDocument(
                    text=d.page_content,
                    score=d.metadata.get("score", 1.0),
                    verifier_score=None,
                    accepted=True,
                    metadata=dict(d.metadata),
                )
                for d in docs[: self.top_k]
            ]

        if self._retriever_type == "langchain":
            docs = self.retriever.get_relevant_documents(query)
            return [
                RetrievedDocument(
                    text=d.page_content,
                    score=d.metadata.get("score", 1.0),
                    verifier_score=None,
                    accepted=True,
                    metadata=d.metadata,
                )
                for d in docs[: self.top_k]
            ]

        if self._retriever_type == "retrieve_protocol":
            raw = self.retriever.retrieve(query, top_k=self.top_k)
            return self._normalize_raw(raw)

        if self._retriever_type == "callable":
            raw = self.retriever(query, self.top_k)
            return self._normalize_raw(raw)

        raise ValueError(f"Unknown retriever type: {self._retriever_type}")

    def _normalize_raw(self, raw: list) -> list[RetrievedDocument]:
        results = []
        for item in raw[: self.top_k]:
            if isinstance(item, RetrievedDocument):
                results.append(item)
            elif isinstance(item, str):
                results.append(RetrievedDocument(text=item, score=1.0, verifier_score=None, accepted=True))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                results.append(RetrievedDocument(text=str(item[0]), score=float(item[1]), verifier_score=None, accepted=True))
            elif hasattr(item, "node") and hasattr(item, "score"):
                # LlamaIndex NodeWithScore
                results.append(RetrievedDocument(
                    text=item.node.get_content(),
                    score=float(item.score or 1.0),
                    verifier_score=None,
                    accepted=True,
                ))
            else:
                results.append(RetrievedDocument(text=str(item), score=1.0, verifier_score=None, accepted=True))
        return results

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedDocument]:
        """
        Run the two-stage pipeline.

        Args:
            query: The search query.
            top_k: Override default top_k for this call.

        Returns:
            List of RetrievedDocument, with .accepted=False for rejected near-misses.
            Only accepted documents are returned (rejected ones are filtered out).
        """
        if top_k is not None:
            self.top_k = top_k

        # Stage 1: Dense retrieval
        candidates = self._call_retriever(query)
        logger.debug(f"Stage 1 returned {len(candidates)} candidates")

        if self.verification_mode == "none":
            return candidates

        # Stage 2: Structural verification
        texts_to_verify = []
        indices_to_verify = []

        for i, doc in enumerate(candidates):
            if self.verification_mode == "full":
                texts_to_verify.append(doc.text)
                indices_to_verify.append(i)
            elif self.verification_mode == "light" and doc.score < self.light_threshold:
                texts_to_verify.append(doc.text)
                indices_to_verify.append(i)

        if texts_to_verify:
            scores = self.verifier.score_batch(query, texts_to_verify)
            # Cross-encoder scores are raw logits — use relative threshold (margin from top score)
            max_score = max(scores) if scores else 0.0
            margin = self.verifier.rejection_threshold  # treat threshold as margin below top score
            for idx, score in zip(indices_to_verify, scores):
                candidates[idx].verifier_score = score
                # Accept if score is within `margin` of the top score
                # e.g. threshold=1.0 means reject anything >1.0 logit below the best result
                candidates[idx].accepted = (max_score - score) <= margin

        rejected = sum(1 for d in candidates if not d.accepted)
        logger.debug(f"Stage 2 rejected {rejected}/{len(candidates)} candidates")

        return [d for d in candidates if d.accepted]

    # LangChain BaseRetriever compatibility
    def get_relevant_documents(self, query: str) -> list:
        """LangChain-compatible interface."""
        from langchain_core.documents import Document
        results = self.retrieve(query)
        return [
            Document(page_content=d.text, metadata={**d.metadata, "verifier_score": d.verifier_score})
            for d in results
        ]

    async def aget_relevant_documents(self, query: str) -> list:
        """Async LangChain-compatible interface."""
        return self.get_relevant_documents(query)
