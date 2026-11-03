import os
from functools import lru_cache
from pathlib import Path
from typing import Literal
from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Robustly find and load .env from project root
_env_path = find_dotenv(usecwd=True)
if not _env_path:
    # Fallback to project root relative to this file
    _candidate = Path(__file__).resolve().parent.parent.parent.parent / ".env"
    if _candidate.exists():
        _env_path = str(_candidate)

if _env_path:
    load_dotenv(_env_path, override=True)


class Settings(BaseSettings):
    """Central configuration management using pydantic-settings."""

    model_config = SettingsConfigDict(
        env_file=_env_path or ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Gemini LLM Settings
    GEMINI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    LLM_TEMPERATURE: float = 0.0

    # Embedding Settings
    # Default is the requested local multilingual model
    EMBEDDING_PROVIDER: Literal["sentence-transformer", "gemini"] = "sentence-transformer"
    SENTENCE_TRANSFORMER_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
    GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"

    # Vector Store Settings
    CHROMA_PERSIST_DIR: str = "./data/vector_store"
    CHROMA_COLLECTION_NAME: str = "sop_bpmn_knowledge"

    # Ingestion & Chunking
    RAW_DATA_DIR: str = "./data/raw"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # Retrieval Settings
    TOP_K: int = 4
    SEARCH_TYPE: Literal["similarity", "mmr"] = "similarity"

    # Phase 5 Critic & Bounded Self-Repair Settings
    MAX_REPAIR_ITERATIONS: int = 2
    CRITIC_CONFIDENCE_THRESHOLD: float = 0.85

    # Phase 7 Backend API & Security Settings
    DATABASE_URL: str = "sqlite:///./data/icicos_rag.db"
    API_PORT: int = 8000
    API_HOST: str = "127.0.0.1"

    # CORS Security Configuration
    CORS_ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]
    CORS_ORIGIN_REGEX: str | None = r"^https:\/\/(rag-icicos|icicos(-[a-z0-9]+)*)\.vercel\.app$"

    @property
    def google_api_key(self) -> str:
        """Resolve Google API key from either GEMINI_API_KEY or GOOGLE_API_KEY."""
        key = (
            self.GEMINI_API_KEY
            or self.GOOGLE_API_KEY
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        ).strip().strip("'\"")

        # Synchronize to os.environ for google-genai / langchain-google-genai SDK
        if key:
            os.environ["GEMINI_API_KEY"] = key
            os.environ["GOOGLE_API_KEY"] = key

        return key

    @property
    def persist_path(self) -> Path:
        """Return Path object for Chroma persistence directory."""
        path = Path(self.CHROMA_PERSIST_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def raw_path(self) -> Path:
        """Return Path object for raw data directory."""
        path = Path(self.RAW_DATA_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()
