# Recommended Verifier Model

## Default (Baseline)

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
**HuggingFace:** https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
**Size:** ~22M parameters
**Use case:** General relevance scoring. Good baseline for Stage 2 verification.

This is what retrieval-guard uses out of the box. It works well for most near-miss rejection
but was not trained specifically for structural near-miss pairs (negation, role reversal, etc.).

---

## Recommended for Production

**Model:** `cross-encoder/ms-marco-electra-base`
**HuggingFace:** https://huggingface.co/cross-encoder/ms-marco-electra-base
**Size:** ~110M parameters
**Use case:** Higher accuracy on structural near-misses due to bidirectional token-level attention.

---

## Training Your Own Verifier

For domain-specific near-miss rejection, fine-tune a cross-encoder on your own
structural near-miss pairs using the `retrieval_guard.benchmark.suite` as a starting template.

Training data format (JSONL):
```json
{"query": "Is the drug effective?", "candidate": "The drug is not effective.", "label": 0}
{"query": "Is the drug effective?", "candidate": "The drug is effective.", "label": 1}
```

Use `sentence-transformers` CrossEncoder training:
```python
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)
model.fit(train_dataloader=..., epochs=3, warmup_steps=100)
model.save("my-retrieval-guard-verifier")
```

Then use:
```python
from retrieval_guard.verifier import StructuralVerifier
verifier = StructuralVerifier("my-retrieval-guard-verifier")
```
