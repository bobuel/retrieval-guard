"""
retrieval-guard — Detect and prevent retrieval generalization regression in fine-tuned embedding models.

Quick start:
    from sentence_transformers import SentenceTransformer
    import retrieval_guard

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Benchmark before fine-tuning
    baseline = retrieval_guard.benchmark.run(model)
    print(baseline.to_json())

    # ... fine-tune your model ...

    # Check for regression
    alert = retrieval_guard.benchmark.compare(fine_tuned_model, baseline)
    if alert.fired:
        print("REGRESSION DETECTED:", alert.recommendation)

    # Use two-stage pipeline in production
    from retrieval_guard.verifier import TwoStagePipeline, StructuralVerifier
    pipeline = TwoStagePipeline(retriever=my_retriever, verifier=StructuralVerifier())
    results = pipeline.retrieve("Is the system secure?")
"""

from . import benchmark
from . import verifier
from . import adapters
from .reporter import export

__version__ = "0.1.0"

__all__ = [
    "benchmark",
    "verifier",
    "adapters",
    "export",
    "__version__",
]
