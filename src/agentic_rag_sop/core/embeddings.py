"""
Embedding Factory.
Supports local multilingual SentenceTransformer as default,
with Google Gemini Embeddings as an optional provider.
"""

from functools import lru_cache
from typing import Optional
from langchain_core.embeddings import Embeddings

from agentic_rag_sop.config.settings import get_settings


@lru_cache(maxsize=4)
def get_embeddings(provider: Optional[str] = None) -> Embeddings:
    """
    Factory function to retrieve embedding model based on settings or parameter.
    
    Default:
        SentenceTransformer: 'paraphrase-multilingual-MiniLM-L12-v2' (Local & Free)
    Optional:
        Google Gemini: 'models/text-embedding-004' (Cloud API)
    """
    settings = get_settings()
    active_provider = (provider or settings.EMBEDDING_PROVIDER).lower()

    if active_provider == "gemini":
        api_key = settings.google_api_key
        if not api_key:
            raise ValueError(
                "Gemini API key is required when EMBEDDING_PROVIDER='gemini'. "
                "Please set GEMINI_API_KEY in your .env file."
            )
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model=settings.GEMINI_EMBEDDING_MODEL,
                google_api_key=api_key,
            )
        except ImportError as e:
            raise ImportError(
                "Could not import GoogleGenerativeAIEmbeddings. "
                "Ensure langchain-google-genai is installed."
            ) from e

    elif active_provider in ["sentence-transformer", "sentence_transformer", "local"]:
        model_name = settings.SENTENCE_TRANSFORMER_MODEL
        try:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except ImportError as e:
            raise ImportError(
                "Could not import HuggingFaceEmbeddings. "
                "Ensure sentence-transformers and langchain-community are installed."
            ) from e

    else:
        raise ValueError(
            f"Unsupported embedding provider: '{active_provider}'. "
            "Supported options are 'sentence-transformer' or 'gemini'."
        )
