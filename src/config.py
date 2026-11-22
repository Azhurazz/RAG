"""
src/config.py — Unified RAG System
Single settings class for all components.
"""
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI
    openai_api_key: str = Field(...)
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # Vector DB
    vector_db: str = "chroma"
    chroma_persist_dir: str = "./data/chroma_db"
    pinecone_api_key: str = ""
    pinecone_index_name: str = "unified-rag"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Re-ranking
    cohere_api_key: str = ""
    reranker_backend: str = "cross_encoder"
    # English-only defaults. For Thai / multilingual content use e.g.
    # CROSS_ENCODER_MODEL=BAAI/bge-reranker-v2-m3 and COHERE_RERANK_MODEL=rerank-multilingual-v3.0
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cohere_rerank_model: str = "rerank-english-v3.0"

    # Memory
    default_memory_strategy: str = "buffer"
    buffer_window_size: int = 10
    summary_max_tokens: int = 2000
    max_sessions: int = 1000           # oldest idle sessions are evicted beyond this
    session_ttl_minutes: int = 120     # sessions idle longer than this are evicted

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: str = "change-me-in-production"
    cors_origins: List[str] = ["http://localhost:3000"]
    # All file paths sent to the API (PDFs, text files, JSONL, report/output dirs)
    # must resolve inside this directory.
    api_file_root: str = "./data"
    # Allow /ingest to fetch URLs that resolve to private/loopback addresses.
    allow_private_urls: bool = False
    max_kg_depth: int = 4

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "unified-rag"


@lru_cache
def get_settings() -> Settings:
    return Settings()
