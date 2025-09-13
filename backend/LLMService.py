from __future__ import annotations
from typing import Dict, Any, Optional, List, TypedDict, Annotated
import os, atexit
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from .FullChain import retrieve, _ensure_loaded, ContextFormatter

class RAGState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    summary: str

_CHECK_CM = None
_CHECKPOINTER = None

def get_sqlite_checkpointer():
    global _CHECK_CM, _CHECKPOINTER
    if _CHECKPOINTER is not None:
        return _CHECKPOINTER
    base_dir = os.path.dirname(__file__)
    persist_dir = os.path.join(base_dir, ".index")
    os.makedirs(persist_dir, exist_ok=True)
    persist_path = os.path.join(persist_dir, "graph_memory.sqlite")
    _CHECK_CM = SqliteSaver.from_conn_string(persist_path)
    _CHECKPOINTER = _CHECK_CM.__enter__()
    return _CHECKPOINTER

@atexit.register
def _close_checkpointer():
    global _CHECK_CM
    try:
        if _CHECK_CM is not None:
            _CHECK_CM.__exit__(None, None, None)
    except Exception:
        pass

def get_groq_llm(groq_api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2, max_tokens: Optional[int] = None) -> ChatGroq:
    key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Thiếu GROQ_API_KEY")
    return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)

SYSTEM_INSTRUCTIONS = (
    "Bạn là trợ lý trả lời về Chương Trình Đào Tạo UIT (Khóa 2025).\n"
    "Nguyên tắc:\n"
    "• Chỉ dùng thông tin đã được cung cấp trong dữ liệu tham chiếu (nếu có). Không bịa.\n"
    "• Nếu dữ liệu chưa đủ để kết luận, nói ngắn gọn rằng chưa đủ và gợi ý cách hỏi cụ thể hơn.\n"
    "• Trả lời tiếng Việt ngắn gọn, mạch lạc.\n"
    "• Không đề cập đến quy trình nội bộ hoặc công cụ.\n"
    "• Không tạo hoặc liệt kê 'Nguồn' hay 'Nguồn 1/2/…'.\n"
)


class RAGService:
    def __init__(self, groq_api_key: Optional[str], model: str, temperature: float, k: int, max_ctx_chars: int):
        _ensure_loaded()
        self.llm_planner = get_groq_llm(groq_api_key, model=model, temperature=0.1, max_tokens=7000).bind_tools([retrieve])
        self.llm_answer = get_groq_llm(groq_api_key, model=model, temperature=temperature, max_tokens=7000)
        self.llm_summarizer = get_groq_llm(groq_api_key, model=model, temperature=0.1, max_tokens=1000)
        self.llm_verifier = get_groq_llm(groq_api_key, model=model, temperature=0.1, max_tokens=2000)
        self.context_formatter = ContextFormatter(max_chars=max_ctx_chars, max_items=k)
        tools = ToolNode([retrieve])
        graph_builder = StateGraph(RAGState)
        graph_builder.add_node("rewrite_or_plan", self.rewrite_or_plan)
        graph_builder.add_node("tools", tools)
        graph_builder.add_node("condense_context", self.condense_context)
        graph_builder.add_node("answer", self.answer)
        graph_builder.add_node("verify", self.verify)
        graph_builder.add_node("maybe_summarize", self.maybe_summarize)
        graph_builder.add_edge(START, "rewrite_or_plan")
        graph_builder.add_conditional_edges("rewrite_or_plan", tools_condition, {"tools": "tools", END: "condense_context"})
        graph_builder.add_edge("tools", "condense_context")
        graph_builder.add_edge("condense_context", "answer")
        graph_builder.add_edge("answer", "verify")
        graph_builder.add_edge("verify", "maybe_summarize")
        graph_builder.add_edge("maybe_summarize", END)
        self.saver = get_sqlite_checkpointer()
        self.graph = graph_builder.compile(checkpointer=self.saver)

    def _msg_text(self, x: Any) -> str:
        if isinstance(x, list):
            return " ".join(i.get("text", "") if isinstance(i, dict) else str(i) for i in x)
        return str(x or "")

    def _recent_history_text(self, messages: List[Any], max_turns: int = 6, max_chars: int = 1500) -> str:
        history = []
        for m in messages:
            t = getattr(m, "type", "")
            if t in ("human", "ai"):
                history.append(m)
        tail = history[-max_turns:]
        parts = []
        for m in tail:
            role = "Người dùng" if getattr(m, "type", "") == "human" else "Trợ lý"
            text = self._msg_text(getattr(m, "content", "")).strip()
            if text:
                parts.append(f"- {role}: {text}")
        s = "\n".join(parts)
        if len(s) > max_chars:
            s = s[-max_chars:]
        return s

    def _last_user(self, messages: List[Any]):
        for m in reversed(messages):
            if getattr(m, "type", "") == "human":
                return m
        return None

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

    def _format_context(self, chunks: List[Dict[str, str]]) -> str:
        fmt = self.context_formatter
        for name in ("format", "__call__", "format_context", "format_chunks", "build", "render"):
            f = getattr(fmt, name, None)
            if callable(f):
                try:
                    return f(chunks)
                except TypeError:
                    return f()
        return "\n\n".join(x.get("content", "") for x in chunks if isinstance(x, dict))

    def rewrite_or_plan(self, state: RAGState):
        history_text = self._recent_history_text(state["messages"])
        last_user = self._last_user(state["messages"])
        s1 = SystemMessage(content="Bạn là tác nhân lập kế hoạch cho trợ lý RAG về CTĐT UIT. Viết lại câu hỏi cuối thành câu hỏi độc lập dựa trên lịch sử gần đây và tóm tắt. Nếu câu hỏi liên quan đến CTĐT (môn học, tín chỉ, học kỳ, điều kiện, mô tả, mã môn, quy định), hãy tạo tối đa 3 truy vấn ngắn, đa dạng từ khóa và gọi công cụ 'retrieve' cho từng truy vấn. Nếu không cần dữ liệu, trả lời trực tiếp thật ngắn gọn.")
        s2 = SystemMessage(content=("Tóm tắt: " + (state.get("summary") or "")))
        s3 = SystemMessage(content=("Lịch sử gần đây:\n" + history_text) if history_text else "Lịch sử gần đây: (trống)")
        msgs = [s1, s2, s3]
        if last_user:
            msgs.append(HumanMessage(content=self._msg_text(getattr(last_user, "content", ""))))
        response = self.llm_planner.invoke(msgs)
        return {"messages": [response]}

    def condense_context(self, state: RAGState):
        chunks = self._last_tool_chunks(state["messages"])
        if not chunks:
            return {"messages": [SystemMessage(content="Ngữ liệu:")]}
        raw = self._format_context(chunks)
        sys1 = SystemMessage(content="Rút gọn dữ liệu tham chiếu thành 3-6 ý cốt lõi có thể kiểm chứng, giữ số liệu và ký hiệu liên quan. Không suy diễn.")
        out = self.llm_summarizer.invoke([sys1, HumanMessage(content=raw)]).content
        return {"messages": [SystemMessage(content=f"Ngữ liệu:\n{out.strip()}")]}

    def answer(self, state: RAGState):
        last_user = self._last_user(state["messages"])
        history_text = self._recent_history_text(state["messages"])
        context_msgs = []
        for m in reversed(state["messages"]):
            if isinstance(getattr(m, "content", ""), str) and str(getattr(m, "content", "")).startswith("Ngữ liệu:"):
                context_msgs.append(m)
                break
        sys_msg = SystemMessage(content=SYSTEM_INSTRUCTIONS)
        prompt = [sys_msg]
        if state.get("summary"):
            prompt.append(SystemMessage(content=f"Tóm tắt hội thoại:\n{state['summary']}"))
        if history_text:
            prompt.append(SystemMessage(content=f"Lịch sử hội thoại gần đây:\n{history_text}"))
        if context_msgs:
            prompt.extend(context_msgs)
        if last_user:
            prompt.append(last_user)
        ans = self.llm_answer.invoke(prompt)
        return {"messages": [ans]}

    def verify(self, state: RAGState):
        chunks = self._last_tool_chunks(state["messages"])
        ctx = self._format_context(chunks) if chunks else ""
        last_ai = None
        for m in reversed(state["messages"]):
            if getattr(m, "type", "") == "ai":
                last_ai = m
                break
        if last_ai is None:
            return {}
        if not ctx.strip():
            return {}
        sys1 = SystemMessage(content="Rút gọn dữ liệu tham chiếu thành 3-6 ý cốt lõi có thể kiểm chứng, giữ số liệu và ký hiệu liên quan. Không suy diễn. Không tạo nhãn 'Nguồn' hoặc đánh số nguồn.")
        msgs = [sys1, SystemMessage(content=f"Dữ liệu tham chiếu:\n{ctx}"), HumanMessage(content=self._msg_text(getattr(last_ai, "content", ""))) ]
        rev = self.llm_verifier.invoke(msgs)
        return {"messages": [rev]}

    def maybe_summarize(self, state: RAGState):
        history_text = self._recent_history_text(state["messages"], max_turns=12, max_chars=4000)
        if not history_text:
            return {}
        prev = state.get("summary") or ""
        need = len(history_text) > 1200 or len(prev) < 1
        if not need:
            return {}
        p = [
            SystemMessage(content="Tóm tắt ngắn gọn cuộc hội thoại, giữ thực thể, sở thích, mục tiêu, và bối cảnh cần cho lượt sau."),
            SystemMessage(content="Tóm tắt trước đó: " + prev),
            HumanMessage(content=history_text),
        ]
        s = self.llm_summarizer.invoke(p).content
        if not isinstance(s, str) or not s.strip():
            return {}
        return {"summary": s.strip()}

    def qa(self, question: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        cfg = {"configurable": {"thread_id": thread_id or "default"}}
        return self.graph.invoke({"messages": [HumanMessage(content=question)], "summary": ""}, config=cfg)

def create_rag_chain(groq_api_key: Optional[str], *, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2, k: int = 6, max_ctx_chars: int = 8000) -> RAGService:
    return RAGService(groq_api_key, model, temperature, k, max_ctx_chars)
