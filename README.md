# retrieval-guard

Experimental utilities for evaluating and guarding retrieval behavior in RAG systems.

The active Python package and full project README live in [`retrieval-guard/`](retrieval-guard/README.md).

## Why this exists

RAG systems can fail quietly when retrieved context looks plausible but does not actually support the user's question. `retrieval-guard` explores lightweight benchmarking and verification ideas for catching retrieval regressions and near-miss failures before they flow into downstream generation or agentic workflows.

## Status

Experimental prototype. Not a production library.

## Start here

- [Project README](retrieval-guard/README.md)
- [Python package metadata](retrieval-guard/pyproject.toml)
