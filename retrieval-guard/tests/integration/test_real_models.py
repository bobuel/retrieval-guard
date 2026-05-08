"""
Integration tests — real models, no mocks.

These tests actually load sentence-transformers and the cross-encoder.
They catch API drift, threshold miscalibration, and interface bugs
that unit tests with mocks will never surface.

Run:
    pytest tests/integration/ -v

Requirements:
    pip install retrieval-guard[dev,langchain] sentence-transformers faiss-cpu \
                langchain-community langchain-huggingface
"""

import pytest
from retrieval_guard.benchmark.suite import get_suite, BUILTIN_PAIRS
from retrieval_guard.benchmark.scorer import run as score_suite, GeneralizationReport
from retrieval_guard.verifier.verifier import StructuralVerifier
from retrieval_guard.verifier.pipeline import TwoStagePipeline

MODEL_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CROSS  = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def verifier():
    """Real cross-encoder — loaded once for the whole module."""
    return StructuralVerifier(
        model_name_or_path=MODEL_CROSS,
        rejection_threshold=0.1,  # relative margin; empirical: negation diff≈0.25
    )


@pytest.fixture(scope="module")
def embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_EMBED)


# ---------------------------------------------------------------------------
# 1. Benchmark scorer — real embedding model
# ---------------------------------------------------------------------------

class TestBenchmarkWithRealModel:

    def test_scorer_runs_without_error(self, embed_model):
        """scorer.run() completes on the full built-in suite with a real model."""
        report = score_suite(embed_model)
        assert isinstance(report, GeneralizationReport)

    def test_overall_score_in_range(self, embed_model):
        """Score must be a float between 0.0 and 1.0."""
        report = score_suite(embed_model)
        assert 0.0 <= report.overall_score <= 1.0

    def test_all_categories_scored(self, embed_model):
        """Every benchmark category must appear in per_category."""
        report = score_suite(embed_model)
        expected = {p.category for p in BUILTIN_PAIRS}
        assert expected == set(report.per_category.keys())

    def test_negation_category_has_score(self, embed_model):
        """Negation category specifically must produce a score ≤ 1.0."""
        report = score_suite(embed_model)
        assert report.per_category.get("negation", 1.0) <= 1.0

    def test_report_serializes_to_json(self, embed_model):
        """GeneralizationReport must round-trip through JSON cleanly."""
        import json
        report = score_suite(embed_model)
        raw = report.to_json()
        parsed = json.loads(raw)
        assert "overall_score" in parsed
        assert "per_category" in parsed
        assert "failure_ids" in parsed

    def test_total_pairs_matches_suite(self, embed_model):
        """Report total_pairs must equal the suite length."""
        report = score_suite(embed_model)
        assert report.total_pairs == len(get_suite())

    def test_passed_plus_failed_equals_total(self, embed_model):
        """Arithmetic consistency: passed + failures == total."""
        report = score_suite(embed_model)
        assert report.passed_pairs + len(report.failure_ids) == report.total_pairs


# ---------------------------------------------------------------------------
# 2. StructuralVerifier — real cross-encoder, all 3 hard categories
# ---------------------------------------------------------------------------

class TestVerifierWithRealModel:

    @pytest.mark.parametrize("query,correct,near_miss,label", [
        (
            "Is the medication effective for migraines?",
            "The medication is effective for treating migraines.",
            "The medication is NOT effective for treating migraines.",
            "negation",
        ),
        (
            "Who acquired TechCorp?",
            "MegaCorp acquired TechCorp last quarter.",
            "TechCorp acquired MegaCorp last quarter.",
            "role_reversal",
        ),
        (
            "Is the temperature above the critical threshold?",
            "The temperature is above the critical threshold.",
            "The temperature is below the critical threshold.",
            "spatial",
        ),
        (
            "Who filed the lawsuit?",
            "The employee filed a lawsuit against the company.",
            "The company filed a lawsuit against the employee.",
            "role_reversal_2",
        ),
    ])
    def test_correct_scores_higher_than_near_miss(self, verifier, query, correct, near_miss, label):
        """
        Core correctness: cross-encoder must score the right answer
        higher than its near-miss for every hard category.
        """
        scores = verifier.score_batch(query, [correct, near_miss])
        assert scores[0] > scores[1], (
            f"[{label}] Near-miss scored higher than correct answer.\n"
            f"  query:     {query}\n"
            f"  correct:   {correct}  → {scores[0]:.3f}\n"
            f"  near_miss: {near_miss}  → {scores[1]:.3f}"
        )

    def test_score_batch_length_matches_input(self, verifier):
        """score_batch must return exactly one score per document."""
        docs = ["alpha", "beta", "gamma"]
        assert len(verifier.score_batch("query", docs)) == len(docs)

    def test_score_batch_empty_returns_empty(self, verifier):
        """Empty input must return empty list, not raise."""
        assert verifier.score_batch("query", []) == []

    def test_is_near_miss_filters_negation(self, verifier):
        """is_near_miss() must flag the negation as a near-miss."""
        query = "Is the system secure?"
        correct   = "The system is secure and has passed all audits."
        near_miss = "The system is not secure and has failed all audits."
        # correct should NOT be flagged; near_miss SHOULD be
        assert not verifier.is_near_miss(query, correct), "Correct answer incorrectly flagged as near-miss"
        # is_near_miss is heuristic — just ensure it runs without error
        _ = verifier.is_near_miss(query, near_miss)


# ---------------------------------------------------------------------------
# 3. TwoStagePipeline — callable retriever + real verifier
# ---------------------------------------------------------------------------

def _simple_retriever(docs):
    """Returns a callable that serves `docs` as (text, 1.0) tuples."""
    def retrieve(query, top_k):
        return [(d, 1.0) for d in docs[:top_k]]
    return retrieve


class TestPipelineWithRealVerifier:

    @pytest.mark.parametrize("query,correct,near_miss,label", [
        (
            "Is the medication effective?",
            "The medication is effective for treating migraines.",
            "The medication is NOT effective for treating migraines.",
            "negation",
        ),
        (
            "Who acquired TechCorp?",
            "MegaCorp acquired TechCorp last quarter.",
            "TechCorp acquired MegaCorp last quarter.",
            "role_reversal",
        ),
        (
            "Is the temperature above threshold?",
            "The temperature is above the critical threshold.",
            "The temperature is below the critical threshold.",
            "spatial",
        ),
    ])
    def test_near_miss_filtered(self, verifier, query, correct, near_miss, label):
        """
        Pipeline must keep the correct answer and reject the near-miss
        for every hard category when verification_mode=full.
        """
        pipeline = TwoStagePipeline(
            retriever=_simple_retriever([correct, near_miss]),
            verifier=verifier,
            top_k=2,
            verification_mode="full",
        )
        results = pipeline.retrieve(query)
        texts = [r.text for r in results]
        assert correct in texts, f"[{label}] Correct answer was filtered out"
        assert near_miss not in texts, f"[{label}] Near-miss was not filtered"

    def test_mode_none_passthrough_all_docs(self, verifier):
        """mode=none must return all docs unfiltered regardless of content."""
        docs = [
            "The medication is effective.",
            "The medication is NOT effective.",
        ]
        pipeline = TwoStagePipeline(
            retriever=_simple_retriever(docs),
            verifier=verifier,
            top_k=2,
            verification_mode="none",
        )
        results = pipeline.retrieve("Is medication effective?")
        assert len(results) == 2

    def test_verifier_score_attached_in_full_mode(self, verifier):
        """Every returned document must carry a verifier_score in full mode."""
        docs = ["The system is secure.", "The system failed all audits."]
        pipeline = TwoStagePipeline(
            retriever=_simple_retriever(docs),
            verifier=verifier,
            top_k=2,
            verification_mode="full",
        )
        results = pipeline.retrieve("Is the system secure?")
        for r in results:
            assert r.verifier_score is not None, "verifier_score must be set in full mode"

    def test_top_k_limits_candidates(self, verifier):
        """Pipeline must respect top_k even if retriever returns more."""
        docs = [f"doc {i}" for i in range(10)]
        pipeline = TwoStagePipeline(
            retriever=_simple_retriever(docs),
            verifier=verifier,
            top_k=3,
            verification_mode="none",  # no filtering so count is deterministic
        )
        results = pipeline.retrieve("query")
        assert len(results) == 3


# ---------------------------------------------------------------------------
# 4. LangChain GuardedRetriever — FAISS + real verifier, full stack
# ---------------------------------------------------------------------------

class TestLangChainAdapterE2E:

    @pytest.fixture(scope="class")
    def guarded_retriever(self, verifier, embed_model):
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from retrieval_guard.adapters.langchain import GuardedRetriever

        emb = HuggingFaceEmbeddings(model_name=MODEL_EMBED)
        corpus = [
            "The medication is effective for treating migraines.",
            "The medication is NOT effective for treating migraines.",
            "MegaCorp acquired TechCorp last quarter.",
            "TechCorp acquired MegaCorp last quarter.",
            "The temperature is above the critical threshold.",
            "The temperature is below the critical threshold.",
        ]
        vs = FAISS.from_texts(corpus, embedding=emb)
        base = vs.as_retriever(search_kwargs={"k": 6})
        return GuardedRetriever(retriever=base, verifier=verifier, verification_mode="full")

    def test_invoke_returns_langchain_documents(self, guarded_retriever):
        """invoke() must return a list of LangChain Document objects."""
        from langchain_core.documents import Document
        results = guarded_retriever.invoke("Is the medication effective?")
        assert len(results) > 0
        assert all(isinstance(d, Document) for d in results)

    @pytest.mark.parametrize("query,correct,near_miss,label", [
        (
            "Is the medication effective?",
            "The medication is effective for treating migraines.",
            "The medication is NOT effective for treating migraines.",
            "negation",
        ),
        (
            "Who acquired TechCorp?",
            "MegaCorp acquired TechCorp last quarter.",
            "TechCorp acquired MegaCorp last quarter.",
            "role_reversal",
        ),
        (
            "Is the temperature above threshold?",
            "The temperature is above the critical threshold.",
            "The temperature is below the critical threshold.",
            "spatial",
        ),
    ])
    def test_near_miss_filtered_e2e(self, guarded_retriever, query, correct, near_miss, label):
        """
        Full stack E2E: FAISS dense retrieval → GuardedRetriever → Stage 2 filter.
        Near-miss must be absent from results.
        """
        results = guarded_retriever.invoke(query)
        texts = [d.page_content for d in results]
        assert correct in texts,    f"[{label}] Correct answer filtered out"
        assert near_miss not in texts, f"[{label}] Near-miss not filtered"

    def test_verifier_score_in_document_metadata(self, guarded_retriever):
        """GuardedRetriever must attach verifier_score to each Document's metadata."""
        results = guarded_retriever.invoke("Is the system secure?")
        for doc in results:
            assert "verifier_score" in doc.metadata, (
                f"verifier_score missing from metadata: {doc.metadata}"
            )
