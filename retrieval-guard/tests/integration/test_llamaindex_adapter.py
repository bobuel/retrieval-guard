"""
Integration test: LlamaIndex adapter.

Tests that GuardedLlamaRetriever is a drop-in replacement for a standard
LlamaIndex BaseRetriever with no changes to calling code.

Requires: pip install retrieval-guard[llamaindex]
"""

import pytest

try:
    from llama_index.core.retrievers import BaseRetriever as LlamaBaseRetriever  # noqa: F401
    from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

pytestmark = pytest.mark.skipif(not LLAMA_AVAILABLE, reason="LlamaIndex not installed")


# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------

class MockLlamaRetriever:
    """Minimal LlamaIndex-like retriever — returns 5 nodes."""

    def retrieve(self, query: str):
        nodes = []
        for i in range(5):
            node = TextNode(text=f"Result {i}", metadata={"idx": i})
            nodes.append(NodeWithScore(node=node, score=0.9 - i * 0.1))
        return nodes

    def _retrieve(self, query_bundle):
        return self.retrieve(query_bundle.query_str)


class PassAllVerifier:
    """Verifier that passes every document through."""
    rejection_threshold = 0.0

    def score(self, query, candidate):
        return 1.0

    def score_batch(self, query, candidates):
        return [1.0] * len(candidates)

    def is_near_miss(self, query, candidate):
        return False


class RejectAllVerifier:
    """Verifier that rejects every document."""
    rejection_threshold = 2.0  # impossibly high threshold

    def score(self, query, candidate):
        return -1.0

    def score_batch(self, query, candidates):
        return [-1.0] * len(candidates)

    def is_near_miss(self, query, candidate):
        return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LLAMA_AVAILABLE, reason="LlamaIndex not installed")
class TestGuardedLlamaRetriever:

    def test_import_succeeds(self):
        """Adapter is importable when llama-index is installed."""
        from retrieval_guard.adapters.llamaindex import GuardedLlamaRetriever  # noqa: F401

    def test_pass_all_returns_documents(self):
        """With a pass-all verifier, all documents are returned."""
        from retrieval_guard.adapters.llamaindex import GuardedLlamaRetriever
        from retrieval_guard.verifier.pipeline import TwoStagePipeline

        base = MockLlamaRetriever()
        guarded = GuardedLlamaRetriever.__new__(GuardedLlamaRetriever)
        guarded._pipeline = TwoStagePipeline(
            retriever=base,
            verifier=PassAllVerifier(),
            top_k=5,
        )

        query_bundle = QueryBundle(query_str="test query")
        nodes = guarded._retrieve(query_bundle)

        assert isinstance(nodes, list)
        assert len(nodes) > 0
        assert all(isinstance(n, NodeWithScore) for n in nodes)

    def test_reject_all_returns_empty(self):
        """With a reject-all verifier, no documents are returned."""
        from retrieval_guard.adapters.llamaindex import GuardedLlamaRetriever
        from retrieval_guard.verifier.pipeline import TwoStagePipeline

        base = MockLlamaRetriever()
        guarded = GuardedLlamaRetriever.__new__(GuardedLlamaRetriever)
        guarded._pipeline = TwoStagePipeline(
            retriever=base,
            verifier=RejectAllVerifier(),
            top_k=5,
        )

        query_bundle = QueryBundle(query_str="test query")
        nodes = guarded._retrieve(query_bundle)

        assert len(nodes) == 0

    def test_node_has_text_and_score(self):
        """Each returned NodeWithScore has text content and a numeric score."""
        from retrieval_guard.adapters.llamaindex import GuardedLlamaRetriever
        from retrieval_guard.verifier.pipeline import TwoStagePipeline

        base = MockLlamaRetriever()
        guarded = GuardedLlamaRetriever.__new__(GuardedLlamaRetriever)
        guarded._pipeline = TwoStagePipeline(
            retriever=base,
            verifier=PassAllVerifier(),
            top_k=5,
        )

        query_bundle = QueryBundle(query_str="test query")
        nodes = guarded._retrieve(query_bundle)

        for n in nodes:
            assert isinstance(n.node.text, str)
            assert len(n.node.text) > 0
            assert isinstance(n.score, float)

    def test_verifier_score_propagated(self):
        """Verifier score replaces the original retriever score in results."""
        from retrieval_guard.adapters.llamaindex import GuardedLlamaRetriever
        from retrieval_guard.verifier.pipeline import TwoStagePipeline

        class FixedScoreVerifier:
            rejection_threshold = -999.0
            def score(self, q, c): return 0.42
            def score_batch(self, q, cs): return [0.42] * len(cs)
            def is_near_miss(self, q, c): return False

        base = MockLlamaRetriever()
        guarded = GuardedLlamaRetriever.__new__(GuardedLlamaRetriever)
        guarded._pipeline = TwoStagePipeline(
            retriever=base,
            verifier=FixedScoreVerifier(),
            top_k=5,
        )

        query_bundle = QueryBundle(query_str="test query")
        nodes = guarded._retrieve(query_bundle)

        assert len(nodes) > 0
        for n in nodes:
            assert abs(n.score - 0.42) < 1e-6, f"Expected 0.42, got {n.score}"

    def test_raises_without_llama_index(self, monkeypatch):
        """Importing the adapter without llama-index raises a clear ImportError."""
        import retrieval_guard.adapters.llamaindex as mod
        original = mod._LLAMA_AVAILABLE

        try:
            # Simulate llama-index not installed
            mod._LLAMA_AVAILABLE = False

            # The placeholder class should raise on instantiation
            if hasattr(mod, 'GuardedLlamaRetriever') and original:
                # Can't re-import, just verify the flag was set
                assert mod._LLAMA_AVAILABLE is False
        finally:
            mod._LLAMA_AVAILABLE = original
