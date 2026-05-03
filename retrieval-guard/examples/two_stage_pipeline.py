"""
Example: Drop-in two-stage pipeline with LangChain FAISS retriever.

Shows how to wrap an existing LangChain retriever with zero changes to
calling code — just swap out the retriever.

Run:
    pip install retrieval-guard[langchain] faiss-cpu
    python examples/two_stage_pipeline.py
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI

from retrieval_guard.adapters.langchain import GuardedRetriever
from retrieval_guard.verifier.verifier import StructuralVerifier

# 1. Build your standard FAISS retriever
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "The medication is effective for treating migraines.",
    "The system has passed all security audits.",
    "The contract allows subletting with landlord approval.",
    "MegaCorp acquired TechCorp last quarter.",
    "The temperature is above the critical threshold.",
]

vectorstore = FAISS.from_texts(documents, embedding=embedding_model)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 2. Wrap it with GuardedRetriever — ONE LINE CHANGE
verifier = StructuralVerifier(rejection_threshold=0.0)
guarded_retriever = GuardedRetriever(
    retriever=base_retriever,
    verifier=verifier,
    verification_mode="full",
)

# 3. Use exactly like a normal LangChain retriever
query = "Is the medication effective?"
docs = guarded_retriever.get_relevant_documents(query)

print(f"Query: {query}")
print(f"Retrieved {len(docs)} documents after Stage 2 filtering:")
for i, doc in enumerate(docs):
    print(f"  [{i+1}] {doc.page_content[:80]}  (verifier_score={doc.metadata.get('verifier_score', 'N/A'):.2f})")

# 4. Drop into a RetrievalQA chain — no other changes needed
# llm = OpenAI(temperature=0)
# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=guarded_retriever)
# answer = qa_chain.run(query)
# print(f"\nAnswer: {answer}")
