# RAG System

> A production-grade Retrieval-Augmented Generation pipeline combining multi-source ingestion, conversational memory, knowledge graph retrieval, hybrid search, re-ranking, fine-tuning, and RAGAS evaluation.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green)](https://python.langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [How to use](#how-to-use)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Ingest documents](#ingest-documents)
  - [Build the knowledge graph](#build-the-knowledge-graph)
  - [Chat — CLI](#chat--cli)
  - [Chat — Python API](#chat--python-api)
  - [REST API](#rest-api)
  - [Fine-tuning](#fine-tuning)
  - [Evaluation](#evaluation)
  - [Testing](#testing)
- [Environment variables](#environment-variables)
- [Limitations](#limitations)
- [Planned improvements](#planned-improvements)
- [Project structure](#project-structure)
- [CLI reference](#cli-reference)

---

## Overview

RAG (Retrieval-Augmented Generation) grounds language model answers in real documents rather than training memory. The model retrieves relevant context at query time and generates answers from it — dramatically reducing hallucinations and enabling answers from private or up-to-date knowledge bases.

This system unifies everything needed for a production RAG pipeline into a single `UnifiedRAGPipeline` object. Every call to `pipeline.chat()` automatically runs all 8 stages:

1. **Session memory** — loads chat history, isolates users by session ID
2. **Question rewrite** — turns vague follow-ups into self-contained queries
3. **KG entity detection** — extracts named entities from the question
4. **Neo4j graph traversal** — returns subgraph context around those entities
5. **Hybrid retrieval** — dense cosine search + BM25 keyword search → RRF fusion
6. **CrossEncoder reranking** — joint (query, chunk) scoring for best top-5
7. **LLM generation** — GPT-4o with KG context + ranked chunks + chat history
8. **Memory update** — saves exchange to session for the next turn

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                       RAG Pipeline                   │
│                                                      │
│   Session Memory ──► Question Rewriter               │
│         │                    │                       │
│         │            Standalone Query                │
│         │          ┌─────────┴──────────┐            │
│         │          │                    │            │
│         │    KG Entity Lookup    Hybrid Retrieval    │
│         │    (Neo4j traversal)   (Dense + BM25 RRF)  │
│         │          │                    │            │
│         │          └─────────┬──────────┘            │
│         │                    │                       │
│         │            CrossEncoder Rerank             │
│         │                    │                       │
│         │       GPT-4o / Fine-tuned Model            │
│         │                    │                       │
│         └──────── Memory Update ◄───────────────────┘│
│                              │                       │
│                    PipelineResult                    │
└─────────────────────────────────────────────────────┘
```

**Key file:** `src/pipeline.py` — the single entry point for all RAG operations.

---

## Features

### Ingestion

| Feature | Details |
|---|---|
| PDF loading | PyPDFLoader with per-page metadata |
| Web scraping | BeautifulSoup via WebBaseLoader |
| REST API ingestion | httpx + dot-notation field extraction |
| Plain text / Markdown | TextLoader with UTF-8 handling |
| Recursive chunking | Splits on paragraph → sentence → word boundaries |
| Semantic chunking | Embedding-based sentence grouping (SemanticChunker) |
| Token-aware chunking | tiktoken cl100k_base exact token counts |
| Hierarchical chunking | Parent-child pairs for multi-granularity retrieval |
| Chunk overlap | Configurable overlap prevents boundary information loss |

### Retrieval

| Feature | Details |
|---|---|
| Dense vector search | ChromaDB cosine similarity with optional metadata filter |
| MMR search | Maximum Marginal Relevance for result diversity |
| BM25 sparse search | Keyword matching via BM25Okapi over all ingested chunks |
| Reciprocal Rank Fusion | Merges dense + sparse ranked lists: `score = Σ 1/(60 + rank)` |
| CrossEncoder reranking | `ms-marco-MiniLM-L-6-v2` joint query-doc scoring |
| Cohere reranking | `rerank-english-v3.0` API reranker (optional) |
| KG entity extraction | GPT-4o JSON-mode entity and triple extraction |
| Neo4j graph traversal | `RELATED_TO` hops up to configurable depth |
| KG-linked chunk retrieval | `MENTIONED_IN` edges connect entities back to source chunks |

### Memory

| Feature | Details |
|---|---|
| Buffer window memory | Keeps last N messages, fast and lightweight |
| Summary memory | LLM compresses older messages to save tokens |
| Vector memory | ChromaDB-backed semantic recall of past messages |
| Standalone rewriter | Rewrites follow-ups into self-contained queries before retrieval |
| Multi-user session isolation | UUID-keyed `SessionManager` registry |

### Knowledge graph

| Feature | Details |
|---|---|
| Entity extraction | GPT-4o extracts `(name, type, description)` tuples |
| Relation extraction | GPT-4o extracts `(subject, relation, object)` triples |
| Neo4j persistence | Nodes and edges stored with `MERGE` for deduplication |
| Subgraph context | 2-hop traversal produces a narrative for the LLM prompt |
| Entity search | Contains-based name/description search over the graph |

### Fine-tuning

| Feature | Details |
|---|---|
| Auto data generation | GPT-4o generates Q&A pairs from ingested chunks |
| KG-based data generation | Relationship-focused Q&A pairs from Neo4j entity neighborhoods |
| Manual JSONL support | Merge expert-curated examples into the training set |
| Quality scoring filter | Second LLM pass scores each sample, drops below threshold (0.65) |
| OpenAI JSONL export | `messages` format ready for `gpt-4o-mini` fine-tuning |
| Alpaca JSONL export | `instruction/input/output` format for LoRA/HuggingFace |
| OpenAI fine-tuning | Full lifecycle: validate → upload → create job → poll → compare |
| LoRA / QLoRA fine-tuning | 4-bit quantized training on Mistral/LLaMA with PEFT + TRL |
| Model hot-swap | `pipeline.swap_model(model_id)` replaces LLM without restart |

### Evaluation

| Feature | Details |
|---|---|
| RAGAS faithfulness | Is the answer grounded in the retrieved context? |
| RAGAS answer relevancy | Does the answer address the question asked? |
| RAGAS context precision | Are retrieved chunks actually useful? |
| RAGAS context recall | Are all relevant chunks retrieved? |
| Faithfulness heuristic | Fast local check with zero API cost (word overlap proxy) |
| Auto-eval mode | Per-turn faithfulness check when `auto_eval=True` |
| JSON report export | Timestamped report with per-metric scores |
| Rich table display | Colour-coded terminal table with Good / Fair / Poor ratings |

---

## How to use

### Prerequisites

- Python 3.10 or higher
- OpenAI API key (required — used for LLM, embeddings, and entity extraction)
- Docker (optional — for Neo4j knowledge graph)
- CUDA GPU with 8 GB+ VRAM (optional — for LoRA fine-tuning only)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Azhurazz/RAG.git
cd rag

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Copy and configure environment
cp .env.example .env
# Open .env and set OPENAI_API_KEY at minimum

# 4. Start Neo4j (optional but recommended)
docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:5
# Neo4j browser available at http://localhost:7474
```

> **Minimum viable setup:** set only `OPENAI_API_KEY` and skip Neo4j entirely. You get hybrid retrieval + reranking + GPT-4o immediately. KG features silently degrade to empty context and the pipeline continues normally.

---

### Ingest documents

```bash
# Ingest from a web URL
python scripts/cli.py ingest \
  --urls https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)

# Ingest from PDF files
python scripts/cli.py ingest --pdfs handbook.pdf policy.pdf

# Ingest from multiple sources at once with a specific chunking strategy
python scripts/cli.py ingest \
  --urls https://docs.example.com \
  --pdfs report.pdf \
  --strategy hierarchical \
  --chunk-size 512

# Available chunking strategies:
#   recursive    — respects paragraph and sentence boundaries (default)
#   semantic     — groups sentences by embedding similarity
#   token        — splits by exact token count (tiktoken cl100k_base)
#   hierarchical — creates parent-child chunk pairs
```

> **Tip:** use `--strategy hierarchical` for long technical documents. Parent chunks provide full context; child chunks provide precise retrieval hits.

---

### Build the knowledge graph

Run this after ingestion to extract entities and relationships into Neo4j:

```bash
# Extract entities and triples from all ingested chunks
python scripts/cli.py build-graph

# Inspect the result
python scripts/cli.py graph-stats

# Explore an entity and its neighbors
python scripts/cli.py entity "Transformer" --depth 2
```

The graph builder calls GPT-4o on each chunk to extract `(entity, type, description)` nodes and `(subject, relation, object)` triples, then upserts everything into Neo4j with deduplication via `MERGE`.

---

### Chat — CLI

```bash
# Interactive multi-turn mode
python scripts/cli.py chat --session-id my-session --verbose

# Single question
python scripts/cli.py chat "What is the attention mechanism?"

# Example multi-turn conversation (rewriter handles follow-ups automatically)
You: What is attention in transformers?
You: How does it compare to RNNs?
You: Give me a Python code example
```

The `--verbose` flag shows the rewritten standalone question, detected entities, KG context preview, rerank scores, and retrieval breakdown on every turn.

---

### Chat — Python API

```python
from src.pipeline import UnifiedRAGPipeline

# Initialise once
pipeline = UnifiedRAGPipeline(
    vector_k=8,          # retrieve 8 candidate chunks before reranking
    rerank_top_n=5,      # keep top 5 after CrossEncoder reranking
    kg_depth=2,          # traverse 2 hops in Neo4j per entity
    auto_eval=True,      # run faithfulness heuristic per turn
)

# Ingest and build graph
pipeline.ingest(
    urls=["https://docs.example.com"],
    pdfs=["report.pdf"],
    chunk_strategy="hierarchical",
)
pipeline.build_graph()

# Chat — all 8 pipeline stages run automatically
result = pipeline.chat("What is RAG?", session_id="user-1")

print(result.answer)               # LLM-generated answer
print(result.entities_found)       # KG entities detected in query
print(result.kg_context)           # Neo4j subgraph narrative
print(result.sources)              # [{source, via, rerank_score}, ...]
print(result.eval_score)           # faithfulness heuristic (0.0–1.0)
print(result.retrieval_breakdown)  # {kg_docs, hybrid_docs, final_docs}

# Follow-up — session memory is automatic
result2 = pipeline.chat("Can you give a Python example?", session_id="user-1")
```

**`PipelineResult` fields:**

| Field | Type | Description |
|---|---|---|
| `answer` | `str` | LLM-generated answer |
| `session_id` | `str` | Session identifier |
| `turn_count` | `int` | Number of turns in this session |
| `standalone_question` | `str` | Rewritten query sent to retrieval |
| `entities_found` | `list[str]` | Named entities detected in query |
| `kg_context` | `str` | Neo4j subgraph narrative included in prompt |
| `sources` | `list[dict]` | Source, retrieval method, rerank score per chunk |
| `eval_score` | `float \| None` | Faithfulness heuristic (if `auto_eval=True`) |
| `retrieval_breakdown` | `dict` | Count of KG docs, hybrid docs, and final docs |

---

### REST API

```bash
# Start the server
uvicorn src.api.main:app --reload --port 8000

# Interactive API docs at http://localhost:8000/docs
```

**All endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check with active model name |
| `POST` | `/api/v1/ingest` | Ingest from PDFs, URLs, text files, or API endpoints |
| `POST` | `/api/v1/build-graph` | Build Neo4j knowledge graph from ingested chunks |
| `POST` | `/api/v1/chat` | Full unified chat with session memory |
| `GET` | `/api/v1/sessions/{id}` | Session detail and full message history |
| `DELETE` | `/api/v1/sessions/{id}` | Delete a session and clear its memory |
| `POST` | `/api/v1/evaluate` | Run RAGAS evaluation on provided samples |
| `POST` | `/api/v1/finetune/generate-data` | Generate Q&A training data from ingested docs |
| `POST` | `/api/v1/finetune/openai-train` | Upload data and start OpenAI fine-tuning job |
| `POST` | `/api/v1/finetune/lora-train` | Start LoRA/QLoRA training (requires GPU) |
| `POST` | `/api/v1/finetune/swap-model` | Hot-swap the active LLM to a fine-tuned model |
| `GET` | `/api/v1/graph/stats` | Entity, relationship, and chunk counts |
| `GET` | `/api/v1/graph/entity/{name}` | Entity neighborhood traversal from Neo4j |

**Example requests:**

```bash
# Ingest
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://docs.example.com"], "chunk_strategy": "hierarchical"}'

# Chat
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "session_id": "user-1", "memory_strategy": "buffer"}'

# Evaluate
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [{
      "question": "What is RAG?",
      "answer": "RAG retrieves documents before generating answers.",
      "contexts": ["RAG combines retrieval with LLM generation."],
      "ground_truth": "RAG stands for Retrieval-Augmented Generation."
    }]
  }'
```

---

### Fine-tuning

#### OpenAI fine-tuning (cloud, no GPU required)

```bash
# Step 1 — generate training data from your ingested documents
python scripts/cli.py finetune generate-data \
  --n-per-chunk 3 \
  --max-chunks 200 \
  --include-kg
# Outputs: data/train_openai.jsonl, data/val_openai.jsonl
#          data/train_alpaca.jsonl,  data/val_alpaca.jsonl

# Step 2 — upload and start fine-tuning job (~20 min, ~$3 per 1M tokens)
python scripts/cli.py finetune openai-train \
  --epochs 3 \
  --wait
# Returns: ft:gpt-4o-mini-2024-07-18:org:unified-rag:abc123

# Step 3 — hot-swap into the running pipeline (no restart needed)
python scripts/cli.py finetune swap-model ft:gpt-4o-mini-2024-07-18:org:unified-rag:abc123

# Or set permanently in .env:
# OPENAI_CHAT_MODEL=ft:gpt-4o-mini-2024-07-18:org:unified-rag:abc123
```

#### LoRA / QLoRA fine-tuning (local GPU)

```bash
# QLoRA requires CUDA GPU with at least 8 GB VRAM
python scripts/cli.py finetune lora-train \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --use-4bit \
  --epochs 3
# Model saved to: ./outputs/lora_model

# Merge LoRA adapter into base model before deployment
python scripts/cli.py finetune lora-merge --adapter ./outputs/lora_model
```

> **Reducing hallucinations through fine-tuning:** include 10–20% "I don't have enough information to answer this" refusal examples in your manual JSONL. This trains the model to refuse rather than hallucinate when context is insufficient.

---

### Evaluation

```bash
# Run RAGAS evaluation on a JSON dataset
python scripts/cli.py eval data/eval_dataset.json --output reports/eval.json

# Dataset format — JSON array of EvalSample objects:
# [
#   {
#     "question": "What is the attention mechanism?",
#     "answer": "Attention computes query, key, value matrices...",
#     "contexts": ["The attention mechanism uses Q, K, V matrices..."],
#     "ground_truth": "Attention allows tokens to attend to each other."
#   }
# ]
```

**RAGAS metrics:**

| Metric | What it measures | Good threshold |
|---|---|---|
| `faithfulness` | Answer is grounded in retrieved context | > 0.80 |
| `answer_relevancy` | Answer addresses the question asked | > 0.80 |
| `context_precision` | Retrieved chunks are actually useful | > 0.75 |
| `context_recall` | All relevant chunks are retrieved | > 0.75 |

---

### Testing

```bash
# Unit tests — no API key needed, runs in under 5 seconds
pytest tests/ -m unit

# Integration tests — mocked LLM, real ChromaDB, around 30 seconds
pytest tests/ -m integration

# Evaluation tests — real OpenAI API calls, around 5 minutes
OPENAI_API_KEY=sk-... pytest tests/ -m eval

# CI-safe run — skips slow evaluation tests
pytest tests/ -m "not eval"

# Coverage report (generates coverage_html/index.html)
make coverage
```

**Test structure:**

```
tests/
├── conftest.py                       # shared fixtures and markers
├── unit/
│   ├── test_chunking.py              # all 4 chunking strategies
│   ├── test_memory.py                # session manager + standalone rewriter
│   └── test_retrieval_and_eval.py    # reranker + faithfulness heuristic
├── integration/
│   ├── test_pipeline.py              # full pipeline.chat() flow
│   └── test_api.py                   # all FastAPI endpoints
└── eval/
    └── test_ragas_eval.py            # RAGAS metrics + benchmark runner
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI API key. |
| `OPENAI_CHAT_MODEL` | `gpt-4o` | LLM for generation. Swap to fine-tuned model ID after training. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for all vector operations. |
| `VECTOR_DB` | `chroma` | `chroma` (local, zero config) or `pinecone` (cloud, scalable). |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB persistence directory. |
| `PINECONE_API_KEY` | — | Required only when `VECTOR_DB=pinecone`. |
| `PINECONE_INDEX_NAME` | `unified-rag` | Pinecone index name. |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI. |
| `NEO4J_USER` | `neo4j` | Neo4j username. |
| `NEO4J_PASSWORD` | `password` | Neo4j password. **Change this in production.** |
| `COHERE_API_KEY` | — | Optional. Enables Cohere reranker when set. |
| `RERANKER_BACKEND` | `cross_encoder` | `cross_encoder`, `cohere`, or `none`. |
| `DEFAULT_MEMORY_STRATEGY` | `buffer` | `buffer`, `summary`, or `vector`. |
| `BUFFER_WINDOW_SIZE` | `10` | Number of messages kept in buffer window memory. |
| `SUMMARY_MAX_TOKENS` | `2000` | Token limit before summary memory compression triggers. |
| `API_SECRET_KEY` | `change-me` | API authentication key. **Always change in production.** |
| `LANGCHAIN_TRACING_V2` | `false` | Set `true` to enable LangSmith tracing. |
| `LANGCHAIN_API_KEY` | — | Required when `LANGCHAIN_TRACING_V2=true`. |
| `LANGCHAIN_PROJECT` | `unified-rag` | LangSmith project name for grouping traces. |

---

## Limitations

### Retrieval

- **Embedding quality ceiling** — all vector search is bounded by `text-embedding-3-small`. Highly specialised domains (law, medicine, niche sciences) may benefit from domain-specific embeddings or fine-tuned embedding models.
- **BM25 requires full in-memory corpus** — `rank-bm25` loads all chunk texts into RAM at startup. Corpora larger than around 500k chunks become a memory bottleneck. Replace with Elasticsearch or OpenSearch for production scale.
- **CrossEncoder adds latency** — reranking adds 200–800 ms per query depending on corpus size. For latency-critical applications use the Cohere API reranker (`RERANKER_BACKEND=cohere`) which offloads compute to the cloud.
- **KG traversal is depth-limited** — `kg_depth=2` with 20 neighbors per entity covers most queries well. Very deeply nested multi-hop reasoning (4+ hops) may not surface correctly without increasing depth, which increases latency.

### Memory

- **In-process session storage** — `SessionManager` stores sessions in a Python dict. Sessions are lost on server restart. For horizontal scaling, replace with a Redis-backed session store.
- **Buffer window is shallow** — beyond `BUFFER_WINDOW_SIZE` turns, earlier context is silently dropped. Switch to `summary` or `vector` memory strategy for very long conversations.
- **Standalone rewriter adds one LLM call per turn** — from turn 2 onward, every follow-up triggers a GPT-4o rewrite call (~300 ms additional overhead). On turn 1 this is skipped entirely.

### Knowledge graph

- **Entity extraction quality depends on GPT-4o** — the JSON-mode prompt works well for English technical text but may produce lower-quality triples for informal language, non-English text, or highly ambiguous domains.
- **Graph build is slow for large corpora** — each chunk requires one GPT-4o API call. At 500 chunks this takes 5–10 minutes and costs approximately $0.50. Batch processing with rate limiting is recommended for large ingestions.
- **Neo4j is a hard dependency for KG features** — if Neo4j is unavailable, `_kg_context()` and `_kg_docs()` silently return empty and the pipeline continues without graph context. All vector retrieval still works normally.

### Fine-tuning

- **OpenAI fine-tuning supports `gpt-4o-mini` only** — `gpt-4o` is not fine-tunable via the API. The fine-tuned mini model may underperform the base `gpt-4o` on complex multi-step reasoning tasks.
- **LoRA requires a CUDA GPU** — QLoRA (4-bit) needs at least 8 GB VRAM; full LoRA needs 16 GB+. There is no CPU fallback for LoRA training.
- **Training data quality determines fine-tune quality** — the auto-generator uses a 0.65 quality threshold. For domains where GPT-4o struggles to generate good Q&A pairs, manual curation of training examples is necessary.
- **Fine-tuned models do not auto-update** — when new documents are ingested, the fine-tuned model's weights do not change. Periodic retraining is required to keep domain knowledge current.

### Evaluation

- **Full RAGAS costs additional API calls** — evaluating 100 samples across 4 metrics costs approximately $0.50–$1.00 in OpenAI API usage due to metric-specific LLM calls.
- **Faithfulness heuristic is approximate** — the local word-overlap check correlates with RAGAS faithfulness but is not equivalent. Use full RAGAS for accurate offline evaluation before deployment.
- **RAGAS does not catch all hallucination types** — context-consistent hallucinations (wrong facts that happen to appear in the retrieved context) are not detected by grounding metrics alone.

---

## Planned improvements

### Short term

- **Redis session backend** — replace in-process `SessionManager` dict with Redis for stateless horizontal scaling across multiple API instances.
- **Streaming API responses** — expose server-sent events (SSE) on `/api/v1/chat` so frontends receive tokens progressively rather than waiting for the complete response.
- **Document deduplication** — checksum-based check at ingest time prevents duplicate chunks when the same file is re-ingested.
- **Metadata filtering on retrieval** — expose `source_type`, `source`, and `date` filters on `/api/v1/chat` so users can scope retrieval to specific document subsets.
- **Async pipeline** — convert `UnifiedRAGPipeline.chat()` to `async` so the FastAPI server handles concurrent requests without blocking the event loop.

### Medium term

- **Multi-modal ingestion** — add image and table extraction from PDFs using `unstructured.io` so charts, diagrams, and structured tables become retrievable alongside text.
- **Embedding fine-tuning** — `MatryoshkaLoss` training pipeline to fine-tune the embedding model on domain-specific sentence pairs, improving retrieval quality on specialised vocabulary.
- **Graph-aware reranking** — incorporate Neo4j entity relationship scores as additional reranking features, boosting chunks connected to query entities in the knowledge graph.
- **Incremental KG refresh** — trigger entity extraction only for newly ingested chunks rather than rebuilding the entire graph on every `build-graph` call.
- **LangSmith evaluation integration** — wire RAGAS scores into LangSmith experiment tracking so evaluation runs are versioned and comparable across code changes.

### Long term

- **Agentic RAG** — wrap the pipeline in a LangGraph agent that decides when to retrieve, when to search the web via Tavily, and when to ask a clarifying question before answering.
- **Continuous fine-tuning loop** — automatically collect high-confidence (`eval_score > 0.85`) chat exchanges as new training examples and schedule periodic OpenAI fine-tuning jobs to keep the model improving.
- **Multi-language support** — language detection at ingest time routing to language-appropriate embeddings and retrieval chains rather than relying on cross-lingual transfer alone.
- **GraphRAG community detection** — implement community summarisation to answer global questions that require synthesising information across the entire knowledge graph rather than local neighborhoods.
- **Self-RAG** — a retrieval-need classifier that decides per query whether retrieval is necessary, plus a self-critique loop where the model assesses its own answer and re-retrieves if confidence is low.

---

## Project structure

```
rag/
├── src/
│   ├── config.py                    # centralised settings (pydantic-settings)
│   ├── pipeline.py                  # UnifiedRAGPipeline — the single entry point
│   ├── ingestion/
│   │   ├── chunking.py              # 4 chunking strategies
│   │   └── loaders.py               # PDF, Web, REST API, Text loaders
│   ├── retrieval/
│   │   ├── vector_store.py          # ChromaDB + Pinecone abstraction
│   │   ├── hybrid_retriever.py      # Dense + BM25 + RRF fusion
│   │   └── reranker.py              # CrossEncoder + Cohere + Passthrough
│   ├── memory/
│   │   └── session_manager.py       # Session, SessionManager, StandaloneRewriter
│   ├── knowledge_graph/
│   │   ├── neo4j_client.py          # all Cypher queries in one place
│   │   └── entity_extractor.py      # GPT-4o entity/triple extraction + GraphBuilder
│   ├── finetuning/
│   │   └── finetuner.py             # TrainingDataGenerator, OpenAIFinetuner, LoRAFinetuner
│   ├── evaluation/
│   │   └── ragas_eval.py            # EvalSample, EvalReport, RAGEvaluator
│   └── api/
│       └── main.py                  # FastAPI with all 13 endpoints
├── scripts/
│   └── cli.py                       # Typer CLI — ingest, chat, eval, finetune, graph
├── configs/
│   └── config.yaml                  # default pipeline configuration
├── .env.example                     # environment variable template
└── requirements.txt                 # all Python dependencies
```

---

## CLI reference

```
python scripts/cli.py ingest [OPTIONS]
  --urls TEXT           Web URLs to scrape
  --pdfs TEXT           PDF file paths
  --texts TEXT          Plain text file paths
  --strategy TEXT       recursive | semantic | token | hierarchical  [default: recursive]
  --chunk-size INT      Chunk size in characters  [default: 512]

python scripts/cli.py build-graph
python scripts/cli.py graph-stats
python scripts/cli.py entity NAME [--depth INT]

python scripts/cli.py chat [QUESTION] [OPTIONS]
  --session-id TEXT     Session ID for memory persistence
  --memory TEXT         buffer | summary | vector  [default: buffer]
  --verbose             Show retrieval breakdown per turn

python scripts/cli.py eval DATASET [OPTIONS]
  --output PATH         Report output path  [default: reports/eval.json]
  --metrics TEXT        Specific RAGAS metrics to run

python scripts/cli.py finetune generate-data [OPTIONS]
  --n-per-chunk INT     Q&A pairs per chunk  [default: 3]
  --max-chunks INT      Max chunks to process  [default: 200]
  --include-kg          Also generate KG-based samples
  --manual-jsonl PATH   Path to manual JSONL file to merge in

python scripts/cli.py finetune openai-train [OPTIONS]
  --train PATH          Training JSONL  [default: data/train_openai.jsonl]
  --val PATH            Validation JSONL  [default: data/val_openai.jsonl]
  --epochs INT          Training epochs  [default: 3]
  --wait / --no-wait    Wait for completion  [default: wait]

python scripts/cli.py finetune lora-train [OPTIONS]
  --model TEXT          Base model  [default: mistralai/Mistral-7B-Instruct-v0.3]
  --use-4bit            Use QLoRA 4-bit quantization  [default: true]
  --epochs INT          Training epochs  [default: 3]

python scripts/cli.py finetune swap-model MODEL_ID
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
