"""
Integration test: LangChain adapter.

Tests that GuardedRetriever wraps retrievers correctly and that
_get_relevant_documents (the core implementation) works as expected.

Requires: pip install retrieval-guard[langchain]
"""

import pytest

try:
    from langchain_core.documents import Document  # noqa: F401
    from langchain_core.retrievers import BaseRetriever  # noqa: F401
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

pytestmark = pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")


# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------

class _FakeDoc:
    """Mimics langchain_core.documents.Document interface."""
    def __init__(self, text: str, score: float = 0.9):
        self.page_content = text
        self.metadata = {"score": score}


class MockLangChainRetriever:
    """Minimal LangChain-like retriever that returns _FakeDoc objects."""

    def get_relevant_documents(self, query: str):
        return [_FakeDoc(f"Result {i}", score=0.9 - i * 0.1) for i in range(5)]


class PassAllVerifier:
    """Accept everything: margin=999 means all docs within 999 of top score pass."""
    rejection_threshold = 999.0

    def score(self, query, candidate):
        return 1.0

    def score_batch(self, query, candidates):
        return [1.0] * len(candidates)

    def is_near_miss(self, query, candidate):
        return False


class TopOneVerifier:
    """Only top scorer passes: margin=0.0, descending scores so only rank-0 passes."""
    rejection_threshold = 0.0

    def score(self, query, candidate):
        return -1.0

    def score_batch(self, query, candidates):
        # Descending: [1.0, 0.0, -1.0, ...] — only index 0 within 0.0 of max
        return [1.0 - float(i) for i in range(len(candidates))]

    def is_near_miss(self, query, candidate):
        return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_guarded_retriever_returns_documents():
    """_get_relevant_documents returns a non-empty list when verifier passes all."""
    from retrieval_guard.adapters.langchain import GuardedRetriever
    from retrieval_guard.verifier.pipeline import TwoStagePipeline

    guarded = GuardedRetriever.__new__(GuardedRetriever)
    guarded._pipeline = TwoStagePipeline(
        retriever=MockLangChainRetriever(),
        verifier=PassAllVerifier(),
        top_k=5,
    )

    docs = guarded._get_relevant_documents("test query")
    assert isinstance(docs, list)
    assert len(docs) > 0


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_guarded_retriever_returns_langchain_documents():
    """Each returned item is a LangChain Document with page_content string."""
    from langchain_core.documents import Document
    from retrieval_guard.adapters.langchain import GuardedRetriever
    from retrieval_guard.verifier.pipeline import TwoStagePipeline

    guarded = GuardedRetriever.__new__(GuardedRetriever)
    guarded._pipeline = TwoStagePipeline(
        retriever=MockLangChainRetriever(),
        verifier=PassAllVerifier(),
        top_k=5,
    )

    docs = guarded._get_relevant_documents("test query")
    for doc in docs:
        assert isinstance(doc, Document)
        assert isinstance(doc.page_content, str)
        assert len(doc.page_content) > 0


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_guarded_retriever_only_top_passes_at_zero_margin():
    """With margin=0.0 and descending scores, only the top-scoring doc passes."""
    from retrieval_guard.adapters.langchain import GuardedRetriever
    from retrieval_guard.verifier.pipeline import TwoStagePipeline

    guarded = GuardedRetriever.__new__(GuardedRetriever)
    guarded._pipeline = TwoStagePipeline(
        retriever=MockLangChainRetriever(),
        verifier=TopOneVerifier(),
        top_k=5,
    )

    docs = guarded._get_relevant_documents("test query")
    # Only the top-scorer (score=1.0) is within 0.0 margin of max (1.0)
    assert len(docs) == 1


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_verifier_scores_in_metadata():
    """Returned Documents carry verifier_score and stage1_score in metadata."""
    from retrieval_guard.adapters.langchain import GuardedRetriever
    from retrieval_guard.verifier.pipeline import TwoStagePipeline

    guarded = GuardedRetriever.__new__(GuardedRetriever)
    guarded._pipeline = TwoStagePipeline(
        retriever=MockLangChainRetriever(),
        verifier=PassAllVerifier(),
        top_k=5,
    )

    docs = guarded._get_relevant_documents("test query")
    assert len(docs) > 0
    for doc in docs:
        assert "verifier_score" in doc.metadata
        assert "stage1_score" in doc.metadata


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_import_flag_accessible():
    """The _LANGCHAIN_AVAILABLE flag is exposed by the adapter module."""
    import retrieval_guard.adapters.langchain as mod
    assert hasattr(mod, "_LANGCHAIN_AVAILABLE")
    assert mod._LANGCHAIN_AVAILABLE is True
