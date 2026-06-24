from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class Settings(BaseModel):
    app_name: str = "recruiter-matchmaking-service"

    api_key: str = ""

    api_prefix: str = "/v1"

    chroma_mode: str = "local"

    chroma_path: Path = Path(".chroma")

    chroma_host: str = ""

    chroma_port: int = 443

    chroma_enable_ssl: bool = True

    chroma_api_key: str = ""

    chroma_tenant: str = ""

    chroma_database: str = ""

    portfolio_collection_name: str = "student_portfolios"
    
    job_collection_name: str = "job_listings"

    embedding_model_name: str = "intfloat/e5-small-v2"

    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    enable_reranker: bool = True

    semantic_top_k: int = 50

    bm25_top_k: int = 50

    rerank_top_k: int = 25

    final_top_k: int = 25


def load_settings() -> Settings:
    return Settings(
        api_key=os.getenv("MATCHING_API_KEY", ""),
        api_prefix=os.getenv("API_PREFIX", "/v1"),
        chroma_mode=os.getenv("CHROMA_MODE", "local"),
        chroma_path=Path(os.getenv("CHROMA_PATH", ".chroma")),
        chroma_host=os.getenv("CHROMA_HOST", ""),
        chroma_port=int(os.getenv("CHROMA_PORT", "443")),
        chroma_enable_ssl=_parse_bool(
            os.getenv("CHROMA_ENABLE_SSL"),
            True,
        ),
        chroma_api_key=os.getenv("CHROMA_API_KEY", ""),
        chroma_tenant=os.getenv("CHROMA_TENANT", ""),
        chroma_database=os.getenv("CHROMA_DATABASE", ""),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "intfloat/e5-small-v2",
        ),
        reranker_model_name=os.getenv(
            "RERANKER_MODEL_NAME",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ),
        enable_reranker=_parse_bool(
            os.getenv("ENABLE_RERANKER"),
            True,
        ),
        semantic_top_k=int(os.getenv("SEMANTIC_TOP_K", "50")),
        bm25_top_k=int(os.getenv("BM25_TOP_K", "50")),
        rerank_top_k=int(os.getenv("RERANK_TOP_K", "25")),
        final_top_k=int(os.getenv("FINAL_TOP_K", "25")),
    )