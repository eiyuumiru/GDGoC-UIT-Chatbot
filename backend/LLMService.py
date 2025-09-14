from __future__ import annotations
from typing import Dict, Any, Optional, List
import os, math, time, json
from langchain_groq import ChatGroq
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.prebuilt import ToolNode
from langmem import create_memory_manager
from .FullChain import retrieve, _ensure_loaded, ContextFormatter
from .EmbeddingManager import get_encoder

_HIST: Dict[str, ChatMessageHistory] = {}
_SEM: Dict[str, List[Dict[str, Any]]] = {}
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
    vecs = [_vec(v) for v in _ensure_emb().embed_documents(texts)]
    bucket = _SEM.setdefault(session_id, [])
    ts = time.time()
    for t, v in zip(texts, vecs):
        bucket.append({"t": t, "v": v, "ts": ts})

def _sem_recall(session_id: str, query: str, k: int = 5, max_chars: int = 600):
    items = _SEM.get(session_id, [])
    if not items or not query.strip():
        return ""
    qv = _vec(_ensure_emb().embed_query(query))
    scored = sorted(((it["t"], _cos(qv, it["v"])) for it in items), key=lambda x: x[1], reverse=True)
    out, acc = [], 0
    for t, _ in scored[: max(1, k)]:
        if acc + len(t) > max_chars:
            break
        out.append(t)
        acc += len(t)
    return "\n".join(out)

def _trim_hist(session_id: str, max_messages: int = 10):
    h = _get_hist(session_id)
    msgs = getattr(h, "messages", [])
    if len(msgs) > max_messages:
        h.messages = msgs[-max_messages:]

def get_groq_llm(key: Optional[str] = None, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2, max_tokens: Optional[int] = None) -> ChatGroq:
    k = key or os.getenv("GROQ_API_KEY")
    if not k:
        raise ValueError("Thiếu GROQ_API_KEY")
    if max_tokens is None or max_tokens > 600:
        max_tokens = 600
    return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)

SYSTEM = "Bạn là chuyên gia về ngành và các môn học (theo chương trình đào tạo 2025) tại trường UIT. Chỉ dùng dữ liệu tham chiếu nếu có; nếu thiếu nói chưa đủ và yêu cầu làm rõ. Trả lời ngắn, tiếng Việt. Không thêm nhãn nguồn."

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        MessagesPlaceholder("history"),
        ("human", "Câu hỏi: {question}\n\nDữ liệu tham chiếu:\n{contexts}\n\nKết thúc câu trả lời, nếu trích được ký ức hoặc hồ sơ người dùng, thêm dòng cuối: MEM: {\"mem\": [\"...\"], \"profile\": {\"name\": \"?\", \"language\": \"?\", \"timezone\": \"?\", \"tone\": \"?\", \"detail_level\": \"?\", \"expertise\": \"?\", \"presentation\": \"?\"}}. Nếu không có, bỏ qua dòng MEM.")
    ]
)

class RAGService:
    def __init__(self, groq_api_key: Optional[str], model: str, temperature: float, k: int, max_ctx_chars: int):
        _ensure_loaded()
        self.ctx_fmt = ContextFormatter(max_chars=max_ctx_chars, max_items=k)
        self.llm = get_groq_llm(groq_api_key, model=model, temperature=temperature, max_tokens=400)
        self.use_langmem = os.getenv("USE_LANGMEM", "0") == "1"
        self.mm_profile = create_memory_manager(self.llm, schemas=[], instructions="Trích các trường hồ sơ từ hội thoại dạng JSON phẳng.", enable_inserts=True, enable_updates=True, enable_deletes=False) if self.use_langmem else None
        self.runnable = RunnableWithMessageHistory(PROMPT | self.llm, lambda sid: _get_hist(sid), input_messages_key="question", history_messages_key="history")
        tools = ToolNode([retrieve])
        g = StateGraph(MessagesState)
        g.add_node("tools", tools)
        g.add_node("answer", self.answer_once)
        g.add_node("memorize", self.memorize)
        g.add_edge(START, "answer")
        g.add_edge("answer", "memorize")
        g.add_edge("memorize", END)
        self.graph = g.compile()

    def _retrieve_ctx(self, q: str) -> str:
        try:
            r = retrieve.invoke({"query": q})
        except Exception:
            try:
                r = retrieve({"query": q})
            except Exception:
                r = []
        chunks = []
        if isinstance(r, list):
            for c in r:
                if isinstance(c, dict):
                    s = str(c.get("content") or "").strip()
                    if s:
                        chunks.append({"content": s})
                elif isinstance(c, str):
                    s = c.strip()
                    if s:
                        chunks.append({"content": s})
        for name in ("format", "__call__", "format_context", "format_chunks"):
            f = getattr(self.ctx_fmt, name, None)
            if callable(f):
                try:
                    return f(chunks)
                except TypeError:
                    return f()
        return "\n\n".join(x.get("content", "") for x in chunks if isinstance(x, dict))

    def _extract_mem_line(self, text: str):
        lines = text.rstrip().splitlines()
        mem = {"mem": [], "profile": {}}
        if not lines:
            return text, mem
        last = lines[-1].strip()
        if last.startswith("MEM:"):
            raw = last[4:].strip()
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    if isinstance(obj.get("mem"), list):
                        mem["mem"] = [str(t) for t in obj["mem"] if isinstance(t, str)]
                    if isinstance(obj.get("profile"), dict):
                        mem["profile"] = {k: v for k, v in obj["profile"].items() if v not in (None, "", [])}
                text = "\n".join(lines[:-1]).rstrip()
            except Exception:
                pass
        return text, mem

    def answer_once(self, state: MessagesState, config: Optional[Dict[str, Any]] = None):
        sid = ((config or {}).get("configurable") or {}).get("thread_id") or "default"
        _trim_hist(sid)
        q = ""
        for m in reversed(state["messages"]):
            if getattr(m, "type", "") == "human":
                q = m.content if isinstance(m.content, str) else ""
                break
        ctx = self._retrieve_ctx(q) or _sem_recall(sid, q, 4, 400)
        msg = self.runnable.invoke({"question": q, "contexts": ctx}, config={"configurable": {"session_id": sid}})
        text, mem = self._extract_mem_line(getattr(msg, "content", "") if hasattr(msg, "content") else str(msg))
        cleaned = AIMessage(content=text)
        return {"messages": [cleaned], "mem": mem, "ctx_dump": ctx}

    def memorize(self, state: MessagesState, config: Optional[Dict[str, Any]] = None):
        sid = ((config or {}).get("configurable") or {}).get("thread_id") or "default"
        mem = state.get("mem") or {}
        mem_texts = [t for t in mem.get("mem", []) if isinstance(t, str) and t.strip()]
        if not mem_texts:
            ctx = state.get("ctx_dump") or ""
            lines = [x.strip() for x in ctx.split("\n") if x.strip()]
            mem_texts = lines[:5]
        if mem_texts:
            _sem_add(sid, mem_texts)
        prof = mem.get("profile") or {}
        if prof and self.use_langmem and self.mm_profile is not None:
            try:
                self.mm_profile.invoke({"messages": [{"role": "assistant", "content": json.dumps(prof, ensure_ascii=False)}]})
            except Exception:
                pass
        return {}

    def qa(self, question: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        cfg = {"configurable": {"thread_id": thread_id or "default"}}
        return self.graph.invoke({"messages": [HumanMessage(content=question)]}, config=cfg)

def create_rag_chain(groq_api_key: Optional[str], *, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2, k: int = 6, max_ctx_chars: int = 6000) -> RAGService:
    return RAGService(groq_api_key, model, temperature, k, max_ctx_chars)
