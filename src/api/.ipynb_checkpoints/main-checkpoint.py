"""
src/api/main.py - RAG API
All endpoints share a single RAGPipeline instance
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import List,Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import get_settings
from src.evaluation.ragas_eval import EvalSample
from src.ingestion.chunking import ChunkStrategy
from src.pipeline import RAGPipeline

setting = get_settings()
pipeline: RAGPipeline = None

@asynccontextmanager
async def lifespane(app: FastAPI):
    global pipeline
    pipeline = RAGPipeline(auto_eval=False)
    print("[api] RAG API ready")
    yield
    pipeline.close()

app = FastAPI(
    title="RAG System",
    description="Multi-source + Memory + Knowledge Graph + Fine-tuning + Evaluation",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

### --- Request/Response Models --------------------------------------- ###
class IngestRequest(BaseModel):
    pdfs: List[str] = []
    urls: List[str] = []
    texts: List[str] = []
    api_endpoints: List[dict] = []
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = 512

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3)
    session_id: Optional[str] = None
    memory_strategy: Literal["buffer","summary","vector"] = "buffer"

class EvalRequest(BaseModel):
    samples: List[dict]
    metrics: Optional[str] = None
    output: str = "reports/eval.json"

class GenerateDataRequest(BaseModel):
    n_per_chunk: int = 3
    max_chunk: Optional[List[int]] = 200
    include_kg: bool = True
    manual_json: Optional[str] = None
    output_dir: str = "data"

class OpenAITrainRequest(BaseModel):
    train_path: str = "data/train_openai.jsonl"
    val_path: str = "data/val_openai.jsonl"
    n_epochs: int = 3
    wait: bool = True

class LoRATrainRequest(BaseModel):
    train_path: str = "data/train_alpaca.jsonl"
    val_path: Optional[str] = "data/val_alpaca.jsonl"
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    use_4bit: bool = True
    n_epochs: int = 3

class SwapModelRequest(BaseModel):
    model_id: str = Field(..., description="Fine-tuned model ID to swap in")

### --- Endpoints ----------------------------------------------- ###
@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.openai_chat_model}

@app.post("/api/v1/ingest")
async def ingest(req: IngestRequest):
    if not any([req.pdfs, req.urls, req.texts, req.api_endpoints]):
        raise HTTPException(400, "Provide at least one source")
    return pipeline.ingest(
        pdfs=req.pdfs or None, urls=req.urls or None,
        texts=req.texts or None, api_endpoints=req.api_endpoints or None,
        chunk_strategy=req.chunk_strategy, chunk_size=req.chunk_size,
    )

@app.post("/api/v1/build_graph")
async def build_graph():
    result = pipeline.build_graph()
    if "error" in result:
        raise HTTPException(400, result["error"])
    return {"status": "built", "stats": result}

@app.post("/api/v1/chat")
async def chat(req: ChatRequest):
    result = pipeline.chat(
        question=req.question,
        session_id=req.session_id,
        memory_strategy=req.memory_strategy,
    )
    return result.to_dict()

@app.post("/api/v1/evaluate")
async def evaluate(req: EvalRequest):
    samples = [EvalSample(**s) for s in req.samples]
    evaluator_kwags = {"metrics": req.metrics} if req.metrics else {}
    from src.evaluation.ragas_eval import RAGEvaluator
    evaluator = RAGEvaluator(**evaluator_kwargs)
    report = evaluator.evaluate(samples)
    evaluator.save_report(report, req.output)
    return {"overall_score": report.overall_score, "metrics": report.metrics}

@app.post("/api/v1/finetune/generate-data")
async def generate_data(req: GenerateDataRequest):
    return pipeline.generate_training_data(
        n_per_chunk=req.n_per_chunk, max_chunks=req.max_chunks,
        include_kg=req.include_kg, manual_jsonl=req.manual_jsonl,
        output_dir=req.output_dir,
    )

@app.post("/api/v1/finetune/openai-train")
async def openai_train(req: OpenAITrainRequest):
    result = pipeline.finetune_openai(
        train_path=req.train_path, val_path=req.val_path,
        n_epochs=req.n_epochs, wait=req.wait,
    )
    return {"result": result, "message": "Model ID saved to data/finetuned_model.txt" if req.wait else "Job started"}

@app.post("/api/v1/finetune/lora-train")
async def lora_train(req: LoRATrainRequest):
    pipeline.finetune_lora(
        train_path=req.train_path, val_path=req.val_path,
        base_model=req.base_model, use_4bit=req.use_4bit, n_epochs=req.n_epochs,
    )
    return {"status": "complete", "model_path": "./outputs/lora_model"}

@app.post("/api/v1/finetune/swap-model")
async def swap_model(req: SwapModelReqeust):
    pipeline.swap_model(req.model_id)
    return {"statis": "swapped", "model": req.model_id}

@app.get("/api/v1/graph/stats")
async def graph_stats():
    from src.knowledge_graph.neo4j_client import Neo4jClient
    client = Neo4jClient(); stats = client.get_stats(); client.close()
    return stats

@app.get("/api/v1/graph/entity/{name}")
async def entity_neighbors(name: str, depth: int = 2):
    from src.knowledge_graph.neo4j_client import Neo4jClient
    client = Neo4jClient()
    neighbors = client.get)neighbors(name, depth=depth)
    clinet.close()
    if not neighbors: raise HTTPException(404, f"Entity '{name}' not found")
    return {"entity": name, "neighbors": neighbors}
        