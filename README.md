# retrieval-guard

Experimental Python utilities for measuring retrieval regressions and filtering structural near misses before retrieved context reaches an LLM or agent.

> Status: working research prototype. The public API, scoring behavior, and defaults may change. This is not a hosted service or a guarantee of factual correctness.

## Why I built it

Retrieved text can look relevant while changing a detail that matters. I wanted a way to test those near misses and make the filtering step inspectable before another system relies on the result. The design separates benchmarking, verification, and reporting so each can be evaluated on its own.

[More of my work](https://aydoon.com/) · [Runnable examples](examples/)

## What is implemented

| Capability | Current implementation |
| --- | --- |
| Near-miss benchmark | 15 built-in pairs across negation, role reversal, threshold/spatial language, and entity binding |
| Baseline and comparison | Scores an embedding model, saves a structured report, and fires a configurable regression alert against a baseline |
| Two-stage retrieval | Normalizes raw, LangChain, and LlamaIndex retriever results and filters candidates with a cross-encoder verifier |
| Verification modes | Full verification, low-score-only “light” verification, and passthrough mode for comparison |
| Reports | JSON, Markdown, and standalone HTML output |
| Interfaces | Python API, command-line entry point, LangChain adapter, and LlamaIndex adapter |
| Tests | Fast unit tests, adapter integration tests, and a separately triggered real-model suite |

## What remains exploratory

- Domain-specific benchmark packs and calibrated verifier thresholds
- A verifier fine-tuned specifically for structural near misses
- Larger evaluation datasets and statistically meaningful model comparisons
- Performance profiling, async model inference, and production observability
- Stable public APIs, versioned report schemas, and broader framework compatibility

The repository does **not** currently provide a managed API, vector database, embedding fine-tuning pipeline, factuality guarantee, or production support commitment.

## How it works

```text
query
  │
  ▼
existing retriever ──► top-k candidates
                          │
                          ▼
                  cross-encoder verifier
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
          accepted context     filtered near miss
                │
                ▼
            LLM / agent
```

The benchmark measures whether an embedding model ranks a supported passage above a structurally similar hard negative. `compare` reruns the suite and reports regressions. `TwoStagePipeline` wraps an existing retriever and applies a second relevance score before returning context.

## Install for development

Python 3.10–3.12 is supported.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
pytest tests/unit -v --no-cov
```

Install an adapter only when needed:

```bash
python -m pip install -e ".[langchain]"
python -m pip install -e ".[llamaindex]"
```

## Benchmark a model

```bash
retrieval-guard benchmark \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --output baseline.json
```

After changing or fine-tuning a model:

```bash
retrieval-guard compare \
  --model path/to/changed-model \
  --baseline baseline.json \
  --threshold 0.05 \
  --output comparison.json
```

Export a saved result:

```bash
retrieval-guard report --input baseline.json --format markdown
```

## Wrap an existing retriever

```python
from retrieval_guard.verifier import StructuralVerifier, TwoStagePipeline

verifier = StructuralVerifier(
    model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2",
    rejection_threshold=0.1,
)
guarded = TwoStagePipeline(
    retriever=my_retriever,
    verifier=verifier,
    top_k=10,
    verification_mode="full",
)

documents = guarded.retrieve("Is the system secure?")
```

The default cross-encoder is a general relevance baseline; tune and evaluate the threshold on your own domain before relying on it.

## Repository layout

```text
.
├── src/retrieval_guard/       # package, CLI, benchmark, verifier, adapters
├── tests/unit/                # network-free behavior tests
├── tests/integration/         # adapter and real-model integration tests
├── examples/                  # runnable benchmark and pipeline examples
├── pretrained/                # model-card guidance
├── .github/workflows/         # branch CI and manual real-model CI
└── pyproject.toml
```

## Verification

Pull requests, `main`, and `codex/**` branches run linting plus unit and adapter-integration tests on Python 3.10, 3.11, and 3.12. Adapter tests exercise interface behavior with test doubles; they do not measure retrieval quality on real models. The model-download suite is a separate manual workflow, and its results should be reported separately from routine CI.

```bash
ruff check src tests
pytest tests/unit -v --no-cov
pytest tests/integration -m "not model" -v --no-cov
```

See `examples/` for complete scripts and `pretrained/MODEL_CARD.md` for the verifier-training direction.

