# retrieval-guard

> Detect and prevent retrieval generalization regression in fine-tuned embedding models.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/retrieval-guard)](https://pypi.org/project/retrieval-guard/)

---

## The Problem

When enterprise teams fine-tune RAG embedding models for precision — teaching them to distinguish near-identical sentences with opposite meanings — they unintentionally degrade broad retrieval generalization.

[Redis research (April 2026)](https://redis.io) quantified the regression:
- **8–9%** accuracy loss on smaller models
- **Up to 40%** on mid-size production models

The failure is **silent**: fine-tuning metrics measure the task being trained for, not general retrieval across unrelated topics. The regression only surfaces in production — often inside agentic pipelines where a single retrieval error cascades into a chain of wrong actions.

**retrieval-guard** is the missing guard layer.

---

## What It Does

| Feature | Description |
|---------|-------------|
| 📊 **Pre-tuning benchmark** | Establishes a retrieval generalization baseline before fine-tuning |
| 🚨 **Regression detector** | Flags when fine-tuning degrades broad recall beyond a configurable threshold |
| 🛡️ **Two-stage verifier** | Drop-in pipeline wrapper that adds token-level near-miss rejection |
| 🔌 **Framework adapters** | Works with LangChain, LlamaIndex, or raw vector store setups |
| 📄 **Structured reports** | JSON / Markdown / HTML output for CI/CD integration |

---

## Install

```bash
pip install retrieval-guard

# With LangChain adapter
pip install retrieval-guard[langchain]

# With LlamaIndex adapter
pip install retrieval-guard[llamaindex]

# Everything
pip install retrieval-guard[all]
```

---

## Quick Start

### 1. Benchmark before fine-tuning

```python
from sentence_transformers import SentenceTransformer
import retrieval_guard

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
baseline = retrieval_guard.benchmark.run(model)
print(baseline.to_json())
```

Output:
```json
{
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "overall_score": 0.8125,
  "per_category": {
    "negation": 0.8,
    "role_reversal": 0.75,
    "spatial": 0.9,
    "binding": 0.8
  },
  "failure_ids": ["neg_003", "role_001"],
  "total_pairs": 16,
  "passed_pairs": 13
}
```

### 2. Check for regression after fine-tuning

```python
# After fine-tuning your model
fine_tuned = SentenceTransformer("path/to/fine-tuned-model")
alert = retrieval_guard.benchmark.compare(fine_tuned, baseline, threshold=0.05)

if alert.fired:
    print("REGRESSION DETECTED:", alert.recommendation)
    retrieval_guard.export(baseline, alert, format="html", path="report.html")
```

### 3. Drop-in two-stage pipeline (LangChain)

```python
from retrieval_guard.adapters.langchain import GuardedRetriever
from retrieval_guard.verifier import StructuralVerifier

# Your existing retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# One line change — drop-in replacement
guarded = GuardedRetriever(retriever=base_retriever, verifier=StructuralVerifier())

# Use exactly like before
docs = guarded.get_relevant_documents("Is the medication effective?")
```

---

## CLI

```bash
# Benchmark a model and save baseline
retrieval-guard benchmark --model sentence-transformers/all-MiniLM-L6-v2 --output baseline.json

# Compare fine-tuned model against baseline (exits 1 if regression detected)
retrieval-guard compare --model my-fine-tuned-model --baseline baseline.json

# Export saved report to HTML
retrieval-guard report --input baseline.json --format html --output report.html
```

The `compare` command exits with code `1` if a regression alert fires — making it a first-class CI/CD gate.

---

## Test Categories

The benchmark suite covers the four structural near-miss categories identified in the Redis research:

| Category | Example |
|----------|---------|
| **Negation** | "The drug is effective" vs "The drug is NOT effective" |
| **Role reversal** | "Alice acquired Bob" vs "Bob acquired Alice" |
| **Spatial** | "Above the threshold" vs "Below the threshold" |
| **Binding** | "Red car hit blue truck" vs "Blue car hit red truck" |

---

## Architecture

```
retrieval-guard/
├── benchmark/      # Test suite + generalization scorer
├── verifier/       # StructuralVerifier + TwoStagePipeline
├── adapters/       # LangChain + LlamaIndex drop-in wrappers
├── reporter.py     # JSON / Markdown / HTML report export
└── cli.py          # CLI: benchmark, compare, report
```

**Stage 1:** Dense retrieval — your existing vector store, unchanged.
**Stage 2:** Cross-encoder verifier — token-level comparison, rejects structural near-misses.

The verifier runs a small cross-encoder (~22M–110M params) distributed as a pre-trained HuggingFace model. No training required.

---

## What It Does NOT Do

- Replace your vector store — it wraps, not replaces
- Fine-tune embedding models — it evaluates and guards, not trains
- Require changes to your retrieval index or document pipeline
- Provide a managed cloud service or hosted API
- Solve long-context or multi-hop retrieval problems

---

## CI/CD Integration

```yaml
# .github/workflows/model-regression.yml
- name: Check for retrieval regression
  run: |
    retrieval-guard compare \
      --model ${{ env.FINE_TUNED_MODEL }} \
      --baseline baselines/production_baseline.json \
      --threshold 0.05
  # Exits 1 (fails CI) if regression detected
```

---

## Deprecation Policy

This library will be deprecated if a well-maintained project with >500 GitHub stars ships both:
1. The structural near-miss benchmark suite
2. The two-stage verifier as a **standalone library** (not bundled inside a larger RAG framework)

Framework-bundled solutions (LangChain, LlamaIndex plugins) do not count. The gap is the standalone guard.

---

## License

MIT © [Your Organization]

---

## Acknowledgements

Based on structural near-miss regression research by Redis Labs (April 2026).
