"""
src/pipeline.py — RAG Pipeline

This is the single entry point for all RAG operations.
No modes. No switches. Every capability runs automatically:

  1. Session memory   — loads chat history, rewrites follow-up questions
  2. KG entity lookup — detects entities in query → Neo4j subgraph context
  3. Hybrid retrieval — dense (ChromaDB) + sparse (BM25) → RRF fusion
  4. Re-ranking       — CrossEncoder or Cohere re-scores merged results
  5. LLM generation   — GPT-4o (or fine-tuned model) with full context
  6. Auto-evaluation  — optional RAGAS faithfulness check after each response
  7. Memory update    — saves exchange back to session

Usage:
    pipeline = RAGPipeline()

    # First ingestion (run once)
    pipeline.ingest(urls=["https://..."], pdfs=["doc.pdf"])
    pipeline.build_graph()

    # Chat — all features active automatically
    result = pipeline.chat("What is attention?", session_id="user-1")
    result = pipeline.chat("Can you explain more?", session_id="user-1")

    # Inspect
    print(result.answer)
    print(result.entities_found)
    print(result.eval_score)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.evaluation.ragas_eval import EvalSample, RAGEvaluator
from src.ingestion.chunking import ChunkConfig, ChunkStrategy
from src.ingestion.loaders import MultiSourceLoader
from src.knowledge_graph.entity_extractor import EntityExtractor
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.memory.session_manager import SessionManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import get_reranker
from src.retrieval.vector_store import VectorStoreManager


# ──────────────────────────────────────────────────────────────────────────────
#  Prompt
# ──────────────────────────────────────────────────────────────────────────────

UNIFIED_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a precise, helpful AI assistant with access to a knowledge base.

Answer the question using the provided context. The context contains:
- Knowledge Graph section: entity relationships extracted from documents
- Retrieved Chunks section: relevant document passages

Guidelines:
- Synthesise both KG relationships and document chunks for a complete answer
- Be factual and grounded in the context
- If context is insufficient, say so clearly
- For follow-up questions, use the conversation history naturally

Context:
{context}"""),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


# ──────────────────────────────────────────────────────────────────────────────
#  Result object
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    answer: str
    session_id: str
    turn_count: int
    standalone_question: str
    entities_found: List[str] = field(default_factory=list)
    kg_context: str = ""
    sources: List[dict] = field(default_factory=list)
    eval_score: Optional[float] = None
    retrieval_breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer":              self.answer,
            "session_id":          self.session_id,
            "turn_count":          self.turn_count,
            "standalone_question": self.standalone_question,
            "entities_found":      self.entities_found,
            "kg_context_preview":  self.kg_context[:300] + "..." if len(self.kg_context) > 300 else self.kg_context,
            "sources":             self.sources,
            "eval_score":          self.eval_score,
            "retrieval_breakdown": self.retrieval_breakdown,
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    One object. All capabilities. Always on.

    Instantiation is fast — heavy components (BM25, CrossEncoder)
    are lazy-loaded on first use.
    """

    def __init__(
        self,
        collection_name: str = "rag",
        vector_k: int = 8,
        rerank_top_n: int = 5,
        kg_depth: int = 2,
        auto_eval: bool = False,
    ):
        self.settings       = get_settings()
        self.collection_name = collection_name
        self.vector_k       = vector_k
        self.rerank_top_n   = rerank_top_n
        self.kg_depth       = kg_depth
        self.auto_eval      = auto_eval

        # Core components — always present
        self.vsm            = VectorStoreManager(collection_name)
        self.session_mgr    = SessionManager()
        self.entity_extractor = EntityExtractor()
        self.reranker       = get_reranker(top_n=rerank_top_n)
        self.evaluator      = RAGEvaluator() if auto_eval else None

        self.llm = ChatOpenAI(
            model=self.settings.openai_chat_model,
            openai_api_key=self.settings.openai_api_key,
            temperature=0.1,
            streaming=True,
        )
        self._chain = UNIFIED_PROMPT | self.llm | StrOutputParser()

        # Lazy-loaded components
        self._hybrid_retriever: Optional[HybridRetriever] = None
        self._neo4j: Optional[Neo4jClient] = None
        self._all_docs: List[Document] = []

    # ──────────────────────────────────────
    #  Ingestion
    # ──────────────────────────────────────

    def ingest(
        self,
        pdfs: List[str] | None = None,
        urls: List[str] | None = None,
        texts: List[str] | None = None,
        api_endpoints: List[dict] | None = None,
        chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        chunk_size: int = 512,
    ) -> dict:
        """Load from any source, chunk, and store in vector DB."""
        config = ChunkConfig(strategy=chunk_strategy, chunk_size=chunk_size)
        loader = MultiSourceLoader(chunk_config=config)
        docs = loader.load_and_chunk(
            pdfs=pdfs, urls=urls, texts=texts, api_endpoints=api_endpoints
        )
        self.vsm.add_documents(docs)
        self._all_docs.extend(docs)
        self._hybrid_retriever = None  # reset so it rebuilds with new docs
        return {"chunks_created": len(docs), "total_chunks": len(self._all_docs)}

    def build_graph(self) -> dict:
        """Build the Neo4j knowledge graph from all ingested documents."""
        from src.knowledge_graph.entity_extractor import GraphBuilder
        if not self._all_docs:
            return {"error": "No documents ingested yet. Call ingest() first."}
        builder = GraphBuilder()
        stats = builder.build(self._all_docs)
        builder.close()
        return stats

    # ──────────────────────────────────────
    #  Main chat method
    # ──────────────────────────────────────

    def chat(
        self,
        question: str,
        session_id: Optional[str] = None,
        memory_strategy: str = None,
    ) -> PipelineResult:
        """
        Full unified pipeline — all features active automatically:
        memory → rewrite → KG → hybrid → rerank → generate → eval → save
        """
        # 1. Session & memory
        session = self.session_mgr.get_or_create(session_id, memory_strategy)
        history = session.get_history()

        # 2. Rewrite follow-up questions using chat history
        standalone_q = session.rewrite(question)

        # 3. KG entity lookup
        entity_names = self.entity_extractor.extract_names(standalone_q)
        kg_context   = self._kg_context(entity_names)
        kg_docs      = self._kg_docs(entity_names)

        # 4. Hybrid retrieval (dense + BM25 + RRF)
        hybrid_docs = self._hybrid_retrieve(standalone_q)

        # 5. Merge + deduplicate KG-linked and hybrid results
        merged = self._merge_docs(kg_docs, hybrid_docs)

        # 6. Re-rank
        final_docs = self.reranker.rerank(standalone_q, merged)

        # 7. Assemble full context
        context = self._build_context(kg_context, final_docs)

        # 8. Generate answer
        answer = self._chain.invoke({
            "context":  context,
            "history":  history,
            "question": question,
        })

        # 9. Auto-evaluation (optional, uses heuristic — no API cost)
        eval_score = None
        if self.auto_eval and self.evaluator:
            sample = EvalSample(
                question=standalone_q,
                answer=answer,
                contexts=[d.page_content for d in final_docs],
            )
            eval_score = self.evaluator.faithfulness_heuristic(sample)

        # 10. Save to session memory
        session.add_exchange(question, answer)

        return PipelineResult(
            answer=answer,
            session_id=session.session_id,
            turn_count=session.turn_count,
            standalone_question=standalone_q,
            entities_found=entity_names,
            kg_context=kg_context,
            sources=[{
                "source":      d.metadata.get("source", "?"),
                "via":         d.metadata.get("retrieval", "vector"),
                "rerank_score": d.metadata.get("rerank_score"),
            } for d in final_docs],
            eval_score=eval_score,
            retrieval_breakdown={
                "kg_docs":     len(kg_docs),
                "hybrid_docs": len(hybrid_docs),
                "final_docs":  len(final_docs),
            },
        )

    # ──────────────────────────────────────
    #  Fine-tuning helpers
    # ──────────────────────────────────────

    def generate_training_data(
        self,
        n_per_chunk: int = 3,
        max_chunks: Optional[int] = None,
        include_kg: bool = True,
        manual_jsonl: Optional[str] = None,
        output_dir: str = "data",
    ) -> dict:
        """Generate training data from ingested docs + KG and export both formats."""
        from src.finetuning.finetuner import TrainingDataGenerator
        gen = TrainingDataGenerator()

        samples = gen.generate_from_chunks(
            self._all_docs, n_per_chunk=n_per_chunk, max_chunks=max_chunks
        )

        if include_kg and self._neo4j_available():
            neo4j = self._get_neo4j()
            kg_samples = gen.generate_from_kg(neo4j, n_pairs=50)
            samples.extend(kg_samples)

        if manual_jsonl:
            manual = gen.generate_manual(manual_jsonl)
            samples.extend(manual)

        train, val = gen.split(samples)
        import os; os.makedirs(output_dir, exist_ok=True)
        gen.save_openai_jsonl(train, f"{output_dir}/train_openai.jsonl")
        gen.save_openai_jsonl(val,   f"{output_dir}/val_openai.jsonl")
        gen.save_alpaca_jsonl(train, f"{output_dir}/train_alpaca.jsonl")
        gen.save_alpaca_jsonl(val,   f"{output_dir}/val_alpaca.jsonl")
        return {"train": len(train), "val": len(val), **gen.stats(samples)}

    def finetune_openai(
        self,
        train_path: str = "data/train_openai.jsonl",
        val_path: str = "data/val_openai.jsonl",
        n_epochs: int = 3,
        wait: bool = True,
    ) -> Optional[str]:
        """Upload data and start an OpenAI fine-tuning job."""
        from src.finetuning.finetuner import OpenAIFinetuner
        ft = OpenAIFinetuner()
        ok, msg = ft.validate(train_path)
        if not ok:
            raise ValueError(f"Invalid training data: {msg}")
        train_id = ft.upload(train_path)
        val_id   = ft.upload(val_path) if val_path else None
        job      = ft.create_job(train_id, val_id, n_epochs=n_epochs)
        if wait:
            model_id = ft.wait(job.id)
            return model_id
        return job.id

    def finetune_lora(
        self,
        train_path: str = "data/train_alpaca.jsonl",
        val_path: Optional[str] = "data/val_alpaca.jsonl",
        base_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        use_4bit: bool = True,
        n_epochs: int = 3,
    ) -> None:
        """Fine-tune a local LLM with LoRA/QLoRA on GPU."""
        from src.finetuning.finetuner import LoRAFinetuner, LoRAConfig
        cfg = LoRAConfig(base_model=base_model, use_4bit=use_4bit, num_epochs=n_epochs)
        LoRAFinetuner(cfg).train(train_path, val_path)

    def swap_model(self, model_id: str) -> None:
        """Hot-swap to a fine-tuned model without restarting."""
        self.llm = ChatOpenAI(
            model=model_id,
            openai_api_key=self.settings.openai_api_key,
            temperature=0.1,
            streaming=True,
        )
        self._chain = UNIFIED_PROMPT | self.llm | StrOutputParser()
        print(f"[pipeline] Model swapped to: {model_id}")

    # ──────────────────────────────────────
    #  Evaluation
    # ──────────────────────────────────────

    def evaluate(self, samples: List[EvalSample], output: str = "reports/eval.json") -> dict:
        """Run full RAGAS evaluation."""
        evaluator = RAGEvaluator()
        report = evaluator.evaluate(samples)
        evaluator.print_report(report)
        evaluator.save_report(report, output)
        return report.metrics

    # ──────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────

    def _hybrid_retrieve(self, query: str) -> List[Document]:
        if not self._hybrid_retriever and self._all_docs:
            self._hybrid_retriever = HybridRetriever(
                vsm=self.vsm, documents=self._all_docs, top_n=self.vector_k
            )
        if self._hybrid_retriever:
            return self._hybrid_retriever.retrieve(query, k=self.vector_k)
        return self.vsm.similarity_search(query, k=self.vector_k)

    def _kg_context(self, entity_names: List[str]) -> str:
        if not entity_names or not self._neo4j_available():
            return ""
        try:
            neo4j = self._get_neo4j()
            return neo4j.get_subgraph_context(entity_names, depth=self.kg_depth)
        except Exception:
            return ""

    def _kg_docs(self, entity_names: List[str]) -> List[Document]:
        if not entity_names or not self._neo4j_available():
            return []
        docs = []
        try:
            neo4j = self._get_neo4j()
            for name in entity_names[:3]:
                for row in neo4j.get_entity_chunks(name, depth=self.kg_depth)[:3]:
                    docs.append(Document(
                        page_content=row["text"],
                        metadata={"source": row["source"], "chunk_id": row["chunk_id"],
                                  "entities": row["entities"], "retrieval": "kg"},
                    ))
        except Exception:
            pass
        return docs

    def _merge_docs(self, kg_docs: List[Document],
                    hybrid_docs: List[Document]) -> List[Document]:
        seen, merged = set(), []
        for doc in kg_docs + hybrid_docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
        return merged

    def _build_context(self, kg_context: str, docs: List[Document]) -> str:
        parts = []
        if kg_context:
            parts.append("=== Knowledge Graph Context ===")
            parts.append(kg_context)
            parts.append("")
        parts.append("=== Retrieved Chunks ===")
        for i, doc in enumerate(docs, 1):
            source   = doc.metadata.get("source", "?")
            via      = doc.metadata.get("retrieval", "vector")
            score    = doc.metadata.get("rerank_score", "")
            score_str = f" | score:{score:.3f}" if isinstance(score, float) else ""
            parts.append(f"[{i}] {source} | via:{via}{score_str}")
            parts.append(doc.page_content)
            parts.append("")
        return "\n".join(parts)

    def _neo4j_available(self) -> bool:
        try:
            from neo4j import GraphDatabase
            return True
        except ImportError:
            return False

    def _get_neo4j(self) -> Neo4jClient:
        if self._neo4j is None:
            self._neo4j = Neo4jClient()
        return self._neo4j

    def close(self) -> None:
        if self._neo4j:
            self._neo4j.close()
