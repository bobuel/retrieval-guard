# retrieval-guard

Experimental utilities for evaluating and guarding retrieval behavior in RAG systems.

`retrieval-guard` explores a practical problem in retrieval-augmented applications: a retriever can appear to work well on average while still failing on near-miss queries where small wording differences change meaning. Those failures can be especially costly when retrieval output feeds an agent or automated workflow.

This repository is a prototype, not a production library.

## Status

Experimental. The code and API may change. Treat this as a research/product prototype for retrieval quality, not as a published or supported package.

## What it is trying to do

The project is organized around three ideas:

1. **Benchmark retrieval behavior before and after model or index changes.**
2. **Detect regressions in broad retrieval quality or near-miss handling.**
3. **Wrap retrieval pipelines with an additional verification step before downstream generation.**

## Why this matters

RAG systems often fail quietly. A retrieved passage may look plausible, but it may not actually support the user's question. This is especially risky when queries involve negation, role reversal, thresholds, policy language, legal/medical/product constraints, or other details where a small semantic difference matters.

The goal of this project is to explore a lightweight guard layer between retrieval and answer generation.

## Intended features

| Area | Prototype direction |
| --- | --- |
| Benchmarking | Run small retrieval tests before and after model/index changes |
| Regression checks | Compare a new retrieval run against a saved baseline |
| Near-miss examples | Test categories such as negation, role reversal, spatial/threshold language, and entity binding |
| Pipeline guard | Add a second-stage verifier before passing documents downstream |
| Integrations | Explore adapters for LangChain, LlamaIndex, and raw retrievers |
| Reporting | Output structured JSON/Markdown/HTML reports for review or CI |

## Example usage

```python
from sentence_transformers import SentenceTransformer
import retrieval_guard

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
baseline = retrieval_guard.benchmark.run(model)
print(baseline.to_json())
```

```python
fine_tuned = SentenceTransformer("path/to/fine-tuned-model")
alert = retrieval_guard.benchmark.compare(fine_tuned, baseline, threshold=0.05)

if alert.fired:
    print("Retrieval regression detected:", alert.recommendation)
```

## Example guarded retriever concept

```python
from retrieval_guard.adapters.langchain import GuardedRetriever
from retrieval_guard.verifier import StructuralVerifier

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
guarded = GuardedRetriever(
    retriever=base_retriever,
    verifier=StructuralVerifier(),
)

docs = guarded.get_relevant_documents("Is the medication effective?")
```

## What this does not do

- replace a vector database
- fine-tune embedding models
- guarantee factual correctness
- solve all long-context or multi-hop retrieval failures
- remove the need for evaluation, review, or domain-specific testing
- provide a managed service or hosted API

## Repository structure

```text
retrieval-guard/
├── benchmark/      # retrieval test suite and scoring concepts
├── verifier/       # structural verifier and two-stage pipeline experiments
├── adapters/       # framework integration experiments
├── reporter.py     # structured report output
└── cli.py          # CLI entry points
```

## Development

```bash
cd retrieval-guard
pip install -e .[dev]
pytest
```

## Product perspective

This project is less about publishing another RAG framework and more about asking a product question:

> How can teams notice retrieval quality regressions before users or agents rely on the wrong context?

That question matters for AI systems that need to be trusted in practical workflows.

## License

MIT © Alex Aidun
