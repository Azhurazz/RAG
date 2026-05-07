"""
src/memory/session_manager.py
Session memory (buffer / summary / vector) + standalone question rewriter.
"""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Dict, List, Literal, Optional

from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.config import get_settings

MemoryStrategy = Literal["buffer", "summary", "vector"]

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's question as a fully self-contained standalone question "
     "using the chat history for context. Return ONLY the rewritten question."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


class StandaloneRewriter:
    def __init__(self):
        s = get_settings()
        self._chain = REWRITE_PROMPT | ChatOpenAI(
            model=s.openai_chat_model, openai_api_key=s.openai_api_key, temperature=0
        ) | StrOutputParser()

    def rewrite(self, question: str, history: List[BaseMessage]) -> str:
        if not history:
            return question
        try:
            return self._chain.invoke({"question": question, "history": history})
        except Exception:
            return question


class Session:
    def __init__(self, session_id: str | None = None,
                 memory_strategy: MemoryStrategy = "buffer"):
        s = get_settings()
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.utcnow()
        self.memory_strategy = memory_strategy
        self.turn_count = 0
        self._rewriter = StandaloneRewriter()

        llm = ChatOpenAI(model=s.openai_chat_model,
                         openai_api_key=s.openai_api_key, temperature=0)

        if memory_strategy == "buffer":
            self.memory = ConversationBufferWindowMemory(
                k=s.buffer_window_size, memory_key="chat_history", return_messages=True)
        elif memory_strategy == "summary":
            self.memory = ConversationSummaryBufferMemory(
                llm=llm, max_token_limit=s.summary_max_tokens,
                memory_key="chat_history", return_messages=True)
        elif memory_strategy == "vector":
            from langchain.memory import ConversationVectorStoreRetrieverMemory
            from langchain_chroma import Chroma
            emb = OpenAIEmbeddings(model=s.openai_embedding_model,
                                   openai_api_key=s.openai_api_key)
            store = Chroma(collection_name=f"memory_{self.session_id}",
                           embedding_function=emb,
                           persist_directory=s.chroma_persist_dir + "_memory")
            self.memory = ConversationVectorStoreRetrieverMemory(
                retriever=store.as_retriever(search_kwargs={"k": 5}),
                memory_key="chat_history")
        else:
            raise ValueError(f"Unknown memory strategy: {memory_strategy}")

    def rewrite(self, question: str) -> str:
        return self._rewriter.rewrite(question, self.get_history())

    def add_exchange(self, human: str, ai: str) -> None:
        self.memory.chat_memory.add_user_message(human)
        self.memory.chat_memory.add_ai_message(ai)
        self.turn_count += 1

    def get_history(self) -> List[BaseMessage]:
        return self.memory.chat_memory.messages

    def get_history_dict(self) -> List[dict]:
        return [
            {"role": "human" if isinstance(m, HumanMessage) else "ai",
             "content": m.content}
            for m in self.get_history()
        ]

    def clear(self) -> None:
        self.memory.clear()
        self.turn_count = 0

    def to_dict(self) -> dict:
        return {"session_id": self.session_id,
                "created_at": self.created_at.isoformat(),
                "memory_strategy": self.memory_strategy,
                "turn_count": self.turn_count}


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._default = get_settings().default_memory_strategy

    def get_or_create(self, session_id: str | None = None,
                      memory_strategy: MemoryStrategy | None = None) -> Session:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        s = Session(session_id=session_id,
                    memory_strategy=memory_strategy or self._default)
        self._sessions[s.session_id] = s
        return s

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        s = self._sessions.pop(session_id, None)
        if s: s.clear()

    def list_all(self) -> List[dict]:
        return [s.to_dict() for s in self._sessions.values()]
