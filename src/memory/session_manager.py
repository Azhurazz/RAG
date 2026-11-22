"""
src/memory/session_manager.py
Session memory (buffer / summary / vector) + standalone question rewriter.
"""
from __future__ import annotations
import hashlib
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import List, Literal, Optional

from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory, VectorStoreRetrieverMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
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
        self.created_at = datetime.now(timezone.utc)
        self.last_used = self.created_at
        self.memory_strategy = memory_strategy
        self.turn_count = 0
        self._rewriter = StandaloneRewriter()
        self.log = ChatMessageHistory()

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
            self._make_vector_memory()
        else:
            raise ValueError(f"Unknown memory strategy: {memory_strategy}")

    def _make_vector_memory(self) -> None:
        from langchain_chroma import Chroma
        s = get_settings()
        emb = OpenAIEmbeddings(model=s.openai_embedding_model,
                               openai_api_key=s.openai_api_key)
        # Chroma collection names must be 3-63 chars of [a-zA-Z0-9._-]; arbitrary
        # session ids ("user 1", Thai text, long ids) would crash, so hash them.
        safe = hashlib.sha1(self.session_id.encode("utf-8")).hexdigest()[:32]
        self._store = Chroma(collection_name=f"memory_{safe}",
                             embedding_function=emb,
                             persist_directory=s.chroma_persist_dir + "_memory")
        self.memory = VectorStoreRetrieverMemory(
            retriever=self._store.as_retriever(search_kwargs={"k": 5}),
            memory_key="chat_history", input_key="input")

    def rewrite(self, question: str, history: Optional[List[BaseMessage]] = None) -> str:
        if history is None:
            history = self.get_history(question)
        return self._rewriter.rewrite(question, history)

    def add_exchange(self, human: str, ai: str) -> None:
        self.memory.save_context({"input": human}, {"output": ai})
        self.log.add_user_message(human)
        self.log.add_ai_message(ai)
        self.turn_count += 1

    def get_history(self, query: str = "") -> List[BaseMessage]:
        if self.memory_strategy == "vector":
            text = self.memory.load_memory_variables({"input": query})["chat_history"]
            return [SystemMessage(content=f"Relevant earlier conversation:\n{text}")] if text else []
        return self.memory.load_memory_variables({})["chat_history"]

    def get_history_dict(self) -> List[dict]:
        return [{"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
            for m in self.log.messages]

    def clear(self) -> None:
        if self.memory_strategy == "vector":
            self._store.delete_collection()
            self._make_vector_memory()
        else:
            self.memory.clear()
        self.log.clear()
        self.turn_count = 0

    def destroy(self) -> None:
        """Free the session's storage for good (no empty collection left behind)."""
        if self.memory_strategy == "vector":
            try:
                self._store.delete_collection()
            except Exception as e:
                print(f"[session] could not delete memory collection: {e}")

    def to_dict(self) -> dict:
        return {"session_id": self.session_id,
                "created_at": self.created_at.isoformat(),
                "memory_strategy": self.memory_strategy,
                "turn_count": self.turn_count}


class SessionManager:
    """
    In-memory session registry with LRU + idle-TTL eviction, so random or
    missing session ids can't grow memory without limit.
    """

    def __init__(self, max_sessions: Optional[int] = None,
                 ttl_minutes: Optional[int] = None):
        st = get_settings()
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()
        self._default = st.default_memory_strategy
        self._max = max_sessions or st.max_sessions
        self._ttl = timedelta(minutes=ttl_minutes or st.session_ttl_minutes)
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None = None,
                      memory_strategy: MemoryStrategy | None = None) -> Session:
        with self._lock:
            self._evict_idle()
            if session_id and session_id in self._sessions:
                s = self._sessions[session_id]
                if memory_strategy and memory_strategy != s.memory_strategy:
                    raise ValueError(
                        f"Session '{session_id}' uses memory strategy '{s.memory_strategy}', "
                        f"not '{memory_strategy}'. Use a new session_id to change strategy.")
                s.last_used = datetime.now(timezone.utc)
                self._sessions.move_to_end(session_id)
                return s
            self._make_room()
            s = Session(session_id=session_id,
                        memory_strategy=memory_strategy or self._default)
            self._sessions[s.session_id] = s
            return s

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            s = self._sessions.pop(session_id, None)
        if s:
            s.destroy()
        return s is not None

    def list_all(self) -> List[dict]:
        return [s.to_dict() for s in list(self._sessions.values())]

    def _evict_idle(self) -> None:
        now = datetime.now(timezone.utc)
        for sid in [sid for sid, s in self._sessions.items() if now - s.last_used > self._ttl]:
            self._sessions.pop(sid).destroy()

    def _make_room(self) -> None:
        # OrderedDict is kept in least-recently-used order (move_to_end on access)
        while self._sessions and len(self._sessions) >= self._max:
            _, oldest = self._sessions.popitem(last=False)
            oldest.destroy()
