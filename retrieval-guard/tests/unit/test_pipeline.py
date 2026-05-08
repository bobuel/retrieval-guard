"""
Unit tests for TwoStagePipeline.

Uses a mock verifier and mock retriever to test Stage 2 filtering logic
without needing a real Transformer model.
"""

from retrieval_guard.verifier.pipeline import TwoStagePipeline, RetrievedDocument


class MockVerifier:
    """
    Mock verifier: accepts docs containing 'positive', rejects docs containing 'negative'.
    rejection_threshold is 0.0.
    """
    rejection_threshold = 0.0

    def score(self, query: str, candidate: str) -> float:
        if "negative" in candidate.lower() or "near-miss" in candidate.lower():
            return -1.0
        return 1.0

    def score_batch(self, query: str, candidates: list[str]) -> list[float]:
        return [self.score(query, c) for c in candidates]

    def is_near_miss(self, query: str, candidate: str) -> bool:
        return self.score(query, candidate) < self.rejection_threshold


def make_retriever(docs: list[str]):
    """Return a simple callable retriever."""
    def retriever(query: str, top_k: int = 10) -> list[tuple[str, float]]:
        return [(d, 1.0) for d in docs[:top_k]]
    return retriever


def test_pipeline_accepts_positive_docs():
    docs = ["positive result one", "positive result two", "positive result three"]
    retriever = make_retriever(docs)
    pipeline = TwoStagePipeline(retriever=retriever, verifier=MockVerifier(), top_k=5)
    results = pipeline.retrieve("test query")
    assert len(results) == 3
    assert all(r.accepted for r in results)


def test_pipeline_rejects_near_miss_docs():
    docs = ["positive result", "near-miss structural error", "another near-miss problem"]
    retriever = make_retriever(docs)
    pipeline = TwoStagePipeline(retriever=retriever, verifier=MockVerifier(), top_k=5)
    results = pipeline.retrieve("test query")
    assert len(results) == 1
    assert results[0].text == "positive result"


def test_pipeline_mode_none_passthrough():
    docs = ["near-miss one", "near-miss two"]
    retriever = make_retriever(docs)
    pipeline = TwoStagePipeline(
        retriever=retriever, verifier=MockVerifier(), top_k=5, verification_mode="none"
    )
    results = pipeline.retrieve("test query")
    assert len(results) == 2


def test_pipeline_verifier_score_attached():
    docs = ["positive result"]
    retriever = make_retriever(docs)
    pipeline = TwoStagePipeline(retriever=retriever, verifier=MockVerifier(), top_k=5)
    results = pipeline.retrieve("test query")
    assert results[0].verifier_score is not None


def test_pipeline_retrieved_document_fields():
    doc = RetrievedDocument(text="hello", score=0.9, verifier_score=0.5, accepted=True)
    assert doc.text == "hello"
    assert doc.score == 0.9
    assert doc.accepted is True
    assert doc.metadata == {}


def test_pipeline_top_k_respected():
    docs = [f"positive doc {i}" for i in range(20)]
    retriever = make_retriever(docs)
    pipeline = TwoStagePipeline(retriever=retriever, verifier=MockVerifier(), top_k=3)
    results = pipeline.retrieve("test query")
    assert len(results) <= 3
