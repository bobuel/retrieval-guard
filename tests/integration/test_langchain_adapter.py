"""
Integration test: LangChain adapter.

Tests that GuardedRetriever is a drop-in replacement for a standard LangChain
retriever with no changes to calling code.

Requires: pip install retrieval-guard[langchain]
"""

import pytest

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.schema import Document
        from langchain.schema.retriever import BaseRetriever
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False

pytestmark = pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")


class MockLangChainRetriever:
    """Minimal LangChain-like retriever for testing."""
    def get_relevant_documents(self, query: str):
        return [
            type("Doc", (), {"page_content": f"Result {i}", "metadata": {"score": 0.9 - i * 0.1}})()
            for i in range(5)
        ]


class MockVerifier:
    rejection_threshold = 0.0

    def score(self, query, candidate):
        return 1.0

    def score_batch(self, query, candidates):
        return [1.0] * len(candidates)

    def is_near_miss(self, query, candidate):
        return False


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_guarded_retriever_is_drop_in():
    from retrieval_guard.adapters.langchain import GuardedRetriever

    base = MockLangChainRetriever()
    guarded = GuardedRetriever.__new__(GuardedRetriever)
    # Manually wire up the pipeline for testing without loading real models
    from retrieval_guard.verifier.pipeline import TwoStagePipeline
    guarded._pipeline = TwoStagePipeline(
        retriever=base,
        verifier=MockVerifier(),
        top_k=5,
    )

    docs = guarded.get_relevant_documents("test query")
    assert isinstance(docs, list)
    assert len(docs) > 0


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_guarded_retriever_rejects_near_misses():
    from retrieval_guard.adapters.langchain import GuardedRetriever
    from retrieval_guard.verifier.pipeline import TwoStagePipeline

    class RejectAllVerifier:
        rejection_threshold = 0.0
        def score(self, q, c): return -1.0
        def score_batch(self, q, cs): return [-1.0] * len(cs)
        def is_near_miss(self, q, c): return True

    base = MockLangChainRetriever()
    guarded = GuardedRetriever.__new__(GuardedRetriever)
    guarded._pipeline = TwoStagePipeline(
        retriever=base,
        verifier=RejectAllVerifier(),
        top_k=5,
    )

    docs = guarded.get_relevant_documents("test query")
    assert len(docs) == 0
