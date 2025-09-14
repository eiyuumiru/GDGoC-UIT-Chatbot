from __future__ import annotations
from typing import Dict, Any, Optional, List
import os, math, time, json
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langmem import create_memory_manager
from .FullChain import retrieve, _ensure_loaded, ContextFormatter
from .EmbeddingManager import get_encoder

class UserProfile(BaseModel):
    name: Optional[str] = None
    language: Optional[str] = None
    timezone: Optional[str] = None
    tone: Optional[str] = None
    detail_level: Optional[str] = None
    expertise: Optional[str] = None
    presentation: Optional[str] = None

_HIST: Dict[str, ChatMessageHistory] = {}
_SEM: Dict[str, List[Dict[str, Any]]] = {}
_PROFILE: Dict[str, UserProfile] = {}
_EMB = None

def _get_hist(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _HIST:
        _HIST[session_id] = ChatMessageHistory()
    return _HIST[session_id]

def _ensure_emb():
    global _EMB
    if _EMB is None:
        _EMB = get_encoder(batch_size=32, device=os.getenv("EMB_DEVICE", "cpu"), show_tqdm=False)
    return _EMB

def _vec(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], (list, tuple)):
        x = x[0]
    return [float(v) for v in x]

def _dot(a, b):
    return float(sum(x * y for x, y in zip(a, b)))

def _norm(a):
    s = sum(x * x for x in a)
    return math.sqrt(s) if s > 0 else 1e-8

def _cos(a, b):
    return _dot(a, b) / (_norm(a) * _norm(b))

def _sem_add(session_id: str, texts: List[str]):
    if not texts:
        return
    enc = _ensure_emb()
    vecs_raw = enc.embed_documents(texts)
    vecs = [_vec(v) for v in vecs_raw]
    bucket = _SEM.setdefault(session_id, [])
    ts = time.time()
    for t, v in zip(texts, vecs):
        bucket.append({"t": t, "v": v, "ts": ts})

def _sem_recall(session_id: str, query: str, k: int = 6, max_chars: int = 800):
    items = _SEM.get(session_id, [])
    if not items or not query.strip():
        return ""
    enc = _ensure_emb()
    qv = _vec(enc.embed_query(query))
    scored = [(it["t"], _cos(qv, it["v"])) for it in items]
    scored.sort(key=lambda x: x[1], reverse=True)
    s, acc = [], 0
    for t, _ in scored[: max(1, k)]:
        if acc + len(t) > max_chars:
            break
        s.append(t)
        acc += len(t)
    return "\n".join(s)

def _profile_text(p: Optional[UserProfile]) -> str:
    if not p:
        return ""
    d = json.loads(p.model_dump_json(exclude_none=True))
    if not d:
        return ""
    return "\n".join(f"{k}: {v}" for k, v in d.items())

def get_groq_llm(groq_api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2, max_tokens: Optional[int] = None) -> ChatGroq:
    key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Thiếu GROQ_API_KEY")
    if max_tokens is None or max_tokens > 1024:
        max_tokens = 1024
    return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)

SYSTEM_INSTRUCTIONS = (
    "Bạn là trợ lý trả lời về Chương Trình Đào Tạo UIT (Khóa 2025).\n"
    "Nguyên tắc:\n"
    "• Chỉ dùng thông tin đã được cung cấp trong dữ liệu tham chiếu (nếu có). Không bịa.\n"
    "• Nếu dữ liệu chưa đủ để kết luận, nói ngắn gọn rằng chưa đủ và gợi ý cách hỏi cụ thể hơn.\n"
    "• Trả lời tiếng Việt chuẩn, ngắn gọn, mạch lạc; có thể dùng gạch đầu dòng khi phù hợp.\n"
    "• Không đề cập đến quy trình nội bộ, công cụ hay cách bạn có dữ liệu.\n"
    "• Không tạo nhãn Nguồn.\n"
)

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Bạn là tác nhân điều phối cho trợ lý RAG về CTĐT UIT. Viết lại câu hỏi thành câu hỏi độc lập dựa trên lịch sử và ngữ cảnh dài hạn. Nếu câu hỏi liên quan đến CTĐT hãy gọi công cụ 'retrieve' với truy vấn ngắn gọn; nếu không cần dữ liệu thì trả lời trực tiếp ngắn gọn."),
        MessagesPlaceholder("history"),
        ("system", "Hồ sơ người dùng:\n{profile}"),
        ("system", "Ngữ cảnh dài hạn:\n{semantic}"),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTIONS),
        MessagesPlaceholder("history"),
        ("system", "Hồ sơ người dùng:\n{profile}"),
        ("system", "Ngữ cảnh dài hạn:\n{semantic}"),
        ("system", "Dữ liệu tham chiếu:\n{contexts}"),
        ("human", "{question}"),
    ]
)

class RAGService:
    def __init__(self, groq_api_key: Optional[str], model: str, temperature: float, k: int, max_ctx_chars: int):
        _ensure_loaded()
        self.context_formatter = ContextFormatter(max_chars=max_ctx_chars, max_items=k)
        self.llm_decider = get_groq_llm(groq_api_key, model=model, temperature=0.1, max_tokens=256).bind_tools([retrieve])
        self.llm_answer = get_groq_llm(groq_api_key, model=model, temperature=temperature, max_tokens=768)
        self.llm_mem = get_groq_llm(groq_api_key, model=model, temperature=0.1, max_tokens=256)
        self.profile_mgr = create_memory_manager(self.llm_mem, schemas=[UserProfile], instructions="Cập nhật hồ sơ người dùng từ hội thoại. Trả ra JSON các khóa: name, language, timezone, tone, detail_level, expertise, presentation. Bỏ qua khóa không có giá trị.", enable_inserts=True, enable_updates=True, enable_deletes=False)
        self.semantic_mgr = create_memory_manager(self.llm_mem, instructions="Trích tối đa 5 ký ức dài hạn hữu ích cho các lượt sau về CTĐT hoặc sở thích trình bày. Mỗi ký ức 1 câu ngắn, rõ. Dùng tiếng Việt.", enable_inserts=True, enable_updates=True, enable_deletes=False)
        self.planner = RunnableWithMessageHistory(PLANNER_PROMPT | self.llm_decider, lambda sid: _get_hist(sid), input_messages_key="question", history_messages_key="history")
        self.answerer = RunnableWithMessageHistory(ANSWER_PROMPT | self.llm_answer, lambda sid: _get_hist(sid), input_messages_key="question", history_messages_key="history")
        tools = ToolNode([retrieve])
        g = StateGraph(MessagesState)
        g.add_node("plan", self.decide)
        g.add_node("tools", tools)
        g.add_node("answer", self.answer)
        g.add_node("memorize", self.memorize)
        g.add_edge(START, "plan")
        g.add_conditional_edges("plan", tools_condition, {"tools": "tools", END: "answer"})
        g.add_edge("tools", "answer")
        g.add_edge("answer", "memorize")
        g.add_edge("memorize", END)
        self.graph = g.compile()

    def _last_user_text(self, messages: List[Any]) -> str:
        for m in reversed(messages):
            if getattr(m, "type", "") == "human":
                return m.content if isinstance(m.content, str) else ""
        return ""

    def _last_ai_text(self, messages: List[Any]) -> str:
        for m in reversed(messages):
            if getattr(m, "type", "") == "ai":
                return m.content if isinstance(m.content, str) else ""
        return ""

    def _last_tool_chunks(self, messages: List[Any]) -> List[Dict[str, str]]:
        last_tool = None
        for m in reversed(messages):
            if getattr(m, "type", "") == "tool":
                last_tool = m
                break
        chunks = []
        if last_tool is None:
            return chunks
        art = getattr(last_tool, "artifact", None)
        if isinstance(art, list):
            for c in art:
                if isinstance(c, dict):
                    s = (c.get("content") or "").strip()
                    if s:
                        chunks.append({"content": s})
        elif isinstance(art, dict):
            s = (art.get("content") or "").strip()
            if s:
                chunks.append({"content": s})
        elif isinstance(art, str):
            s = art.strip()
            if s:
                chunks.append({"content": s})
        return chunks

    def _format_contexts(self, state_messages: List[Any]) -> str:
        chunks = self._last_tool_chunks(state_messages)
        if not chunks:
            return ""
        fmt = self.context_formatter
        for name in ("format", "__call__", "format_context", "format_chunks", "build", "render"):
            f = getattr(fmt, name, None)
            if callable(f):
                try:
                    return f(chunks)
                except TypeError:
                    return f()
        return "\n\n".join(x.get("content", "") for x in chunks if isinstance(x, dict))

    def decide(self, state: MessagesState, config: Optional[Dict[str, Any]] = None):
        sid = ((config or {}).get("configurable") or {}).get("thread_id") or "default"
        q = self._last_user_text(state["messages"])
        sem = _sem_recall(sid, q)
        profile = _PROFILE.get(sid)
        out = self.planner.invoke({"question": q, "semantic": sem, "profile": _profile_text(profile)}, config={"configurable": {"session_id": sid}})
        return {"messages": [out]}

    def answer(self, state: MessagesState, config: Optional[Dict[str, Any]] = None):
        sid = ((config or {}).get("configurable") or {}).get("thread_id") or "default"
        q = self._last_user_text(state["messages"])
        ctx = self._format_contexts(state["messages"])
        sem = _sem_recall(sid, q)
        profile = _PROFILE.get(sid)
        msgs = self.answerer.invoke({"question": q, "contexts": ctx, "semantic": sem, "profile": _profile_text(profile)}, config={"configurable": {"session_id": sid}})
        return {"messages": [msgs]}

    def memorize(self, state: MessagesState, config: Optional[Dict[str, Any]] = None):
        sid = ((config or {}).get("configurable") or {}).get("thread_id") or "default"
        q = self._last_user_text(state["messages"])
        a = self._last_ai_text(state["messages"])
        ctx = self._format_contexts(state["messages"])
        if not q.strip() and not a.strip():
            return {}
        convo = [{"role": "user", "content": q}]
        if a.strip():
            convo.append({"role": "assistant", "content": a})
        try:
            prof_out = self.profile_mgr.invoke({"messages": convo})
            merged = {}
            for m in prof_out or []:
                c = getattr(m, "content", m)
                if isinstance(c, dict):
                    merged.update({k: v for k, v in c.items() if v not in (None, "", [])})
                elif isinstance(c, str):
                    try:
                        d = json.loads(c)
                        if isinstance(d, dict):
                            merged.update({k: v for k, v in d.items() if v not in (None, "", [])})
                    except Exception:
                        pass
            if merged:
                _PROFILE[sid] = UserProfile(**merged)
        except Exception:
            pass
        try:
            mems = self.semantic_mgr.invoke({"messages": convo})
            texts = []
            for m in mems or []:
                c = getattr(m, "content", m)
                if isinstance(c, str):
                    s = c.strip()
                    if s and s.upper() != "NONE":
                        texts.append(s)
                elif isinstance(c, dict):
                    s = json.dumps(c, ensure_ascii=False)
                    if s:
                        texts.append(s)
            if texts:
                _sem_add(sid, texts)
        except Exception:
            pass
        return {}

    def qa(self, question: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        cfg = {"configurable": {"thread_id": thread_id or "default"}}
        return self.graph.invoke({"messages": [HumanMessage(content=question)]}, config=cfg)

def create_rag_chain(groq_api_key: Optional[str], *, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2, k: int = 6, max_ctx_chars: int = 8000) -> RAGService:
    return RAGService(groq_api_key, model, temperature, k, max_ctx_chars)
