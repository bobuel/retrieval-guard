"""
Example: Drop-in two-stage pipeline with LangChain FAISS retriever.

Shows how to wrap an existing LangChain retriever with zero changes to
calling code — just swap out the retriever.

Run:
    pip install retrieval-guard[langchain] faiss-cpu langchain-community langchain-huggingface
    python examples/two_stage_pipeline.py
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from retrieval_guard.adapters.langchain import GuardedRetriever
from retrieval_guard.verifier.verifier import StructuralVerifier

# 1. Build your standard FAISS retriever
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "The medication is effective for treating migraines.",
    "The medication is NOT effective for treating migraines.",   # negation near-miss
    "The system has passed all security audits.",
    "MegaCorp acquired TechCorp last quarter.",
    "TechCorp acquired MegaCorp last quarter.",                  # role reversal near-miss
    "The temperature is above the critical threshold.",
    "The temperature is below the critical threshold.",          # spatial near-miss
]

vectorstore = FAISS.from_texts(documents, embedding=embedding_model)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

# 2. Wrap it with GuardedRetriever — ONE LINE CHANGE
verifier = StructuralVerifier(rejection_threshold=0.5)
guarded_retriever = GuardedRetriever(
    retriever=base_retriever,
    verifier=verifier,
    verification_mode="full",
)

# 3. Query — near-misses get filtered or flagged
queries = [
    "Is the medication effective?",
    "Who acquired who, MegaCorp or TechCorp?",
    "Is the temperature above or below threshold?",
]

for query in queries:
    print(f"\nQuery: {query}")
    docs = guarded_retriever.invoke(query)
    print(f"Retrieved {len(docs)} documents after Stage 2 filtering:")
    for i, doc in enumerate(docs):
        score = doc.metadata.get('verifier_score')
        score_str = f"{score:.3f}" if score is not None else "N/A"
        print(f"  [{i+1}] (score={score_str}) {doc.page_content}")
