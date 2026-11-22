"""
src/api/main.py - RAG API
All endpoints share a single RAGPipeline instance.

Notes
- Every /api/v1 route requires the header `X-API-Key: <API_SECRET_KEY>`.
- Endpoints are plain `def` (not `async def`): the pipeline is blocking
  (LLM calls, CrossEncoder, Neo4j), so FastAPI runs them in its threadpool
  instead of freezing the event loop.
- Long jobs (OpenAI fine-tune wait, LoRA training) run in the background.
- File paths are restricted to API_FILE_ROOT; URLs to public hosts.
"""
from __future__ import annotations

import ipaddress
import secrets
import socket
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Literal, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import get_settings
from src.evaluation.ragas_eval import EvalSample
from src.ingestion.chunking import ChunkStrategy
from src.pipeline import RAGPipeline

settings = get_settings()
pipeline: RAGPipeline = None

DEFAULT_SECRET = "change-me-in-production"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    if settings.api_secret_key == DEFAULT_SECRET:
        print("[api] WARNING: API_SECRET_KEY is the default value — set a real secret in .env")
    pipeline = RAGPipeline(auto_eval=False)
    print("[api] RAG API ready")
    yield
    pipeline.close()


app = FastAPI(
    title="RAG System",
    description="Multi-source + Memory + Knowledge Graph + Fine-tuning + Evaluation",
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                   allow_methods=["*"], allow_headers=["*"])


### --- Security helpers ----------------------------------------------- ###
def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_secret_key):
        raise HTTPException(401, "Missing or invalid X-API-Key header")


def _file_root() -> Path:
    return Path(settings.api_file_root).resolve()


def safe_path(p: str, must_exist: bool = False) -> str:
    """Resolve `p` (relative to API_FILE_ROOT) and refuse anything outside it."""
    root = _file_root()
    full = (root / p).resolve()
    if not full.is_relative_to(root):
        raise HTTPException(400, f"Path must be inside API_FILE_ROOT ({settings.api_file_root}): {p}")
    if must_exist and not full.is_file():
        raise HTTPException(400, f"File not found: {p}")
    return str(full)


def safe_url(url: str) -> str:
    """Only http(s) URLs that resolve to public addresses (blocks SSRF to internal services)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(400, f"Only http(s) URLs are allowed: {url}")
    if settings.allow_private_urls:
        return url
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except socket.gaierror:
        raise HTTPException(400, f"Cannot resolve host: {parsed.hostname}") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise HTTPException(400, f"URL resolves to a non-public address: {url}")
    return url


### --- Request/Response Models --------------------------------------- ###
class IngestRequest(BaseModel):
    pdfs: List[str] = []
    urls: List[str] = []
    texts: List[str] = []
    api_endpoints: List[dict] = []
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = Field(512, ge=64, le=8192)

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = Field(None, max_length=200)
    # None -> DEFAULT_MEMORY_STRATEGY from .env (new sessions only)
    memory_strategy: Optional[Literal["buffer", "summary", "vector"]] = None

class EvalSampleIn(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None

class EvalRequest(BaseModel):
    samples: List[EvalSampleIn] = Field(..., min_length=1)
    metrics: Optional[List[str]] = None
    output: str = "reports/eval.json"          # relative to API_FILE_ROOT

class GenerateDataRequest(BaseModel):
    n_per_chunk: int = Field(3, ge=1, le=10)
    max_chunks: Optional[int] = 200
    include_kg: bool = True
    manual_jsonl: Optional[str] = None         # relative to API_FILE_ROOT
    output_dir: str = "training"               # relative to API_FILE_ROOT

class OpenAITrainRequest(BaseModel):
    train_path: str = "training/train_openai.jsonl"
    val_path: str = "training/val_openai.jsonl"
    n_epochs: int = Field(3, ge=1, le=20)

class LoRATrainRequest(BaseModel):
    train_path: str = "training/train_alpaca.jsonl"
    val_path: Optional[str] = "training/val_alpaca.jsonl"
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    use_4bit: bool = True
    n_epochs: int = Field(3, ge=1, le=20)

class SwapModelRequest(BaseModel):
    model_id: str = Field(..., description="Fine-tuned model ID to swap in")


### --- Background job registry ---------------------------------------- ###
_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _set_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(fields)


def _wait_openai_job(job_id: str) -> None:
    from src.finetuning.finetuner import OpenAIFinetuner
    try:
        model_id = OpenAIFinetuner().wait(job_id)
        _set_job(job_id, status="succeeded" if model_id else "failed", model_id=model_id)
    except Exception as e:
        _set_job(job_id, status="error", error=str(e))


def _run_lora_job(job_id: str, req: LoRATrainRequest, train: str, val: Optional[str]) -> None:
    try:
        pipeline.finetune_lora(train_path=train, val_path=val, base_model=req.base_model,
                               use_4bit=req.use_4bit, n_epochs=req.n_epochs)
        _set_job(job_id, status="succeeded", model_path="./outputs/lora_model")
    except Exception as e:
        _set_job(job_id, status="error", error=str(e))


### --- Endpoints ----------------------------------------------- ###
@app.get("/health")
def health():
    return {"status": "ok", "model": pipeline.llm.model_name if pipeline else settings.openai_chat_model}


api = APIRouter(prefix="/api/v1", dependencies=[Depends(verify_api_key)])


@api.post("/ingest")
def ingest(req: IngestRequest):
    if not any([req.pdfs, req.urls, req.texts, req.api_endpoints]):
        raise HTTPException(400, "Provide at least one source")
    pdfs  = [safe_path(p, must_exist=True) for p in req.pdfs]
    texts = [safe_path(p, must_exist=True) for p in req.texts]
    urls  = [safe_url(u) for u in req.urls]
    for ep in req.api_endpoints:
        if "url" not in ep:
            raise HTTPException(400, "Each api_endpoint needs a 'url'")
        safe_url(ep["url"])
    return pipeline.ingest(
        pdfs=pdfs or None, urls=urls or None,
        texts=texts or None, api_endpoints=req.api_endpoints or None,
        chunk_strategy=req.chunk_strategy, chunk_size=req.chunk_size,
    )


@api.post("/build-graph")
@api.post("/build_graph", include_in_schema=False)  # old path kept for compatibility
def build_graph():
    result = pipeline.build_graph()
    if "error" in result:
        raise HTTPException(400, result["error"])
    return {"status": "built", "stats": result}


@api.post("/chat")
def chat(req: ChatRequest):
    try:
        result = pipeline.chat(
            question=req.question,
            session_id=req.session_id,
            memory_strategy=req.memory_strategy,
        )
    except ValueError as e:  # e.g. memory_strategy doesn't match the existing session
        raise HTTPException(400, str(e)) from e
    return result.to_dict()


@api.get("/sessions/{session_id}")
def get_session(session_id: str):
    s = pipeline.session_mgr.get(session_id)
    if not s:
        raise HTTPException(404, f"Session '{session_id}' not found")
    return {**s.to_dict(), "messages": s.get_history_dict()}


@api.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if not pipeline.session_mgr.delete(session_id):
        raise HTTPException(404, f"Session '{session_id}' not found")
    return {"status": "deleted", "session_id": session_id}


@api.post("/evaluate")
def evaluate(req: EvalRequest):
    from src.evaluation.ragas_eval import RAGEvaluator
    output = safe_path(req.output)
    samples = [EvalSample(**s.model_dump()) for s in req.samples]
    try:
        evaluator = RAGEvaluator(metrics=req.metrics) if req.metrics else RAGEvaluator()
        report = evaluator.evaluate(samples)
    except ValueError as e:  # unknown metric names
        raise HTTPException(400, str(e)) from e
    evaluator.save_report(report, output)
    return {"overall_score": report.overall_score, "metrics": report.metrics}


@api.post("/finetune/generate-data")
def generate_data(req: GenerateDataRequest):
    return pipeline.generate_training_data(
        n_per_chunk=req.n_per_chunk, max_chunks=req.max_chunks,
        include_kg=req.include_kg,
        manual_jsonl=safe_path(req.manual_jsonl, must_exist=True) if req.manual_jsonl else None,
        output_dir=safe_path(req.output_dir),
    )


@api.post("/finetune/openai-train")
def openai_train(req: OpenAITrainRequest, background: BackgroundTasks):
    train = safe_path(req.train_path, must_exist=True)
    val = safe_path(req.val_path)
    try:
        job_id = pipeline.finetune_openai(train_path=train, val_path=val,
                                          n_epochs=req.n_epochs, wait=False)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    _set_job(job_id, type="openai", status="running")
    background.add_task(_wait_openai_job, job_id)
    return {"job_id": job_id, "status": "running",
            "message": f"Poll GET /api/v1/finetune/jobs/{job_id}"}


@api.post("/finetune/lora-train")
def lora_train(req: LoRATrainRequest, background: BackgroundTasks):
    train = safe_path(req.train_path, must_exist=True)
    val = safe_path(req.val_path) if req.val_path else None
    with _jobs_lock:
        if any(j.get("type") == "lora" and j.get("status") == "running" for j in _jobs.values()):
            raise HTTPException(409, "A LoRA training job is already running")
    job_id = f"lora-{uuid.uuid4().hex[:8]}"
    _set_job(job_id, type="lora", status="running")
    background.add_task(_run_lora_job, job_id, req, train, val)
    return {"job_id": job_id, "status": "running"}


@api.get("/finetune/jobs/{job_id}")
def finetune_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return {"job_id": job_id, **job}


@api.post("/finetune/swap-model")
def swap_model(req: SwapModelRequest):
    pipeline.swap_model(req.model_id)
    return {"status": "swapped", "model": req.model_id}


@api.get("/graph/stats")
def graph_stats():
    try:
        return pipeline.get_neo4j().get_stats()
    except Exception as e:
        raise HTTPException(503, f"Neo4j unavailable: {e}") from e


@api.get("/graph/entity/{name}")
def entity_neighbors(name: str, depth: int = Query(2, ge=1)):
    depth = min(depth, settings.max_kg_depth)  # deep variable-length paths can overload Neo4j
    try:
        neighbors = pipeline.get_neo4j().get_neighbors(name, depth=depth)
    except Exception as e:
        raise HTTPException(503, f"Neo4j unavailable: {e}") from e
    if not neighbors:
        raise HTTPException(404, f"Entity '{name}' not found")
    return {"entity": name, "depth": depth, "neighbors": neighbors}


app.include_router(api)
