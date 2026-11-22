"""
src/prompts.py
System prompt shared by the RAG pipeline and the OpenAI fine-tuning data,
so a fine-tuned model is trained on the same format it sees at inference.
"""

RAG_SYSTEM_TEMPLATE = """You are a precise, helpful AI assistant with access to a knowledge base.

Answer the question using the provided context. The context contains:
- Knowledge Graph section: entity relationships extracted from documents
- Retrieved Chunks section: relevant document passages

Guidelines:
- Synthesise both KG relationships and document chunks for a complete answer
- Be factual and grounded in the context
- If context is insufficient, say so clearly
- For follow-up questions, use the conversation history naturally

Context:
{context}"""


def format_chunk_context(chunks: list[tuple[str, str]]) -> str:
    """chunks: [(source, text), ...] -> the 'Retrieved Chunks' block used at inference."""
    parts = ["=== Retrieved Chunks ==="]
    for i, (source, text) in enumerate(chunks, 1):
        parts += [f"[{i}] {source}", text, ""]
    return "\n".join(parts)
