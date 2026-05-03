"""
LangChain adapter for TwoStagePipeline.

Makes TwoStagePipeline a fully compatible LangChain BaseRetriever so it can
be dropped into any LangChain chain, agent, or LCEL pipeline without changes.

Usage:
    from retrieval_guard.adapters.langchain import GuardedRetriever
    from retrieval_guard.verifier.verifier import StructuralVerifier

    # Wrap your existing LangChain retriever
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    verifier = StructuralVerifier()
    guarded = GuardedRetriever(retriever=base_retriever, verifier=verifier)

    # Use exactly like a normal LangChain retriever
    docs = guarded.get_relevant_documents("Is the system secure?")

    # Drop into a chain
    from langchain.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=guarded)
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..verifier.verifier import StructuralVerifier
from ..verifier.pipeline import TwoStagePipeline

try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.schema.retriever import BaseRetriever
        from langchain.schema import Document
        from langchain.callbacks.manager import CallbackManagerForRetrieverRun
        _LANGCHAIN_AVAILABLE = True
    except ImportError:
        _LANGCHAIN_AVAILABLE = False


if _LANGCHAIN_AVAILABLE:

    class GuardedRetriever(BaseRetriever):
        """
        LangChain BaseRetriever that wraps TwoStagePipeline.

        Drop-in replacement for any LangChain retriever — no changes to calling code.
        """

        # Pydantic fields (LangChain uses Pydantic v1 or v2 depending on version)
        pipeline: Any = None
        _pipeline: Optional[TwoStagePipeline] = None

        class Config:
            arbitrary_types_allowed = True

        def __init__(
            self,
            retriever: Any,
            verifier: Optional[StructuralVerifier] = None,
            verifier_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k: int = 10,
            verification_mode: str = "full",
            rejection_threshold: Optional[float] = None,
            **kwargs: Any,
        ):
            super().__init__(**kwargs)
            if verifier is None:
                verifier = StructuralVerifier(
                    model_name_or_path=verifier_model,
                    rejection_threshold=rejection_threshold or 0.0,
                )
            self._pipeline = TwoStagePipeline(
                retriever=retriever,
                verifier=verifier,
                top_k=top_k,
                verification_mode=verification_mode,
                rejection_threshold=rejection_threshold,
            )

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional["CallbackManagerForRetrieverRun"] = None,
        ) -> List[Document]:
            results = self._pipeline.retrieve(query)
            return [
                Document(
                    page_content=doc.text,
                    metadata={
                        **doc.metadata,
                        "stage1_score": doc.score,
                        "verifier_score": doc.verifier_score,
                    },
                )
                for doc in results
            ]

        async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional["CallbackManagerForRetrieverRun"] = None,
        ) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)

else:
    class GuardedRetriever:  # type: ignore
        """Placeholder — install langchain or langchain-core to use this adapter."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LangChain is not installed. "
                "Install it with: pip install retrieval-guard[langchain]"
            )
