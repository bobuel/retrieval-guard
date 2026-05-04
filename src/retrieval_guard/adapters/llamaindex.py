"""
LlamaIndex adapter for TwoStagePipeline.

Makes TwoStagePipeline a compatible LlamaIndex BaseRetriever.

Usage:
    from retrieval_guard.adapters.llamaindex import GuardedLlamaRetriever
    from retrieval_guard.verifier.verifier import StructuralVerifier

    base_retriever = index.as_retriever(similarity_top_k=10)
    verifier = StructuralVerifier()
    guarded = GuardedLlamaRetriever(retriever=base_retriever, verifier=verifier)

    nodes = guarded.retrieve("Is the system secure?")
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..verifier.verifier import StructuralVerifier
from ..verifier.pipeline import TwoStagePipeline, RetrievedDocument

try:
    from llama_index.core.retrievers import BaseRetriever as LlamaBaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
    _LLAMA_AVAILABLE = True
except ImportError:
    try:
        from llama_index.retrievers import BaseRetriever as LlamaBaseRetriever
        from llama_index.schema import NodeWithScore, TextNode, QueryBundle
        _LLAMA_AVAILABLE = True
    except ImportError:
        _LLAMA_AVAILABLE = False


if _LLAMA_AVAILABLE:

    class GuardedLlamaRetriever(LlamaBaseRetriever):
        """
        LlamaIndex BaseRetriever that wraps TwoStagePipeline.

        Drop-in replacement for any LlamaIndex retriever.
        """

        def __init__(
            self,
            retriever: Any,
            verifier: Optional[StructuralVerifier] = None,
            verifier_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k: int = 10,
            verification_mode: str = "full",
            rejection_threshold: Optional[float] = None,
        ):
            super().__init__()
            if verifier is None:
                verifier = StructuralVerifier(
                    model_name_or_path=verifier_model,
                    rejection_threshold=rejection_threshold or 0.0,
                )
            self._pipeline = TwoStagePipeline(
                retriever=retriever,
                verifier=verifier,
                top_k=top_k,
                verification_mode=verification_mode,
                rejection_threshold=rejection_threshold,
            )

        def _retrieve(self, query_bundle: "QueryBundle") -> List["NodeWithScore"]:
            query = query_bundle.query_str
            results: list[RetrievedDocument] = self._pipeline.retrieve(query)
            nodes = []
            for doc in results:
                node = TextNode(text=doc.text, metadata=doc.metadata)
                nodes.append(NodeWithScore(node=node, score=doc.verifier_score or doc.score))
            return nodes

else:
    class GuardedLlamaRetriever:  # type: ignore
        """Placeholder — install llama-index to use this adapter."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LlamaIndex is not installed. "
                "Install it with: pip install retrieval-guard[llamaindex]"
            )
