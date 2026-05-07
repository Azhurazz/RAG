"""
src/config.py — Unified RAG System
Single settings class for all components.
"""
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Memory
    default_memory_strategy: str = "buffer"
    buffer_window_size: int = 10
    summary_max_tokens: int = 2000

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: str = "change-me-in-production"

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "unified-rag"


@lru_cache
def get_settings() -> Settings:
    return Settings()
