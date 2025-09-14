from __future__ import annotations
from typing import Dict, Any, Optional, List
import os
from pathlib import Path
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from .FullChain import retrieve, _ensure_loaded, ContextFormatter

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
        ("system", "Bạn là tác nhân điều phối cho trợ lý RAG về CTĐT UIT. Hãy viết lại câu hỏi cuối thành một câu hỏi độc lập dựa trên lịch sử. Nếu câu hỏi liên quan đến CTĐT (mã môn, tên môn, tín chỉ, tiên quyết, học kỳ, mô tả, quy định), hãy gọi công cụ 'retrieve' với truy vấn ngắn gọn; nếu không cần dữ liệu thì trả lời trực tiếp thật ngắn gọn."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTIONS),
        MessagesPlaceholder("history"),
        ("system", "Dữ liệu tham chiếu:\n{contexts}"),
        ("human", "{question}"),
    ]
)

def _history_store(session_id: str) -> SQLChatMessageHistory:
    base = Path(__file__).resolve().parent / ".index"
    base.mkdir(parents=True, exist_ok=True)
    return SQLChatMessageHistory(session_id=session_id, connection_string=f"sqlite:///{(base / 'chat_history.db').as_posix()}")

class RAGService:
    def __init__(self, groq_api_key: Optional[str], model: str, temperature: float, k: int, max_ctx_chars: int):
        _ensure_loaded()
        self.context_formatter = ContextFormatter(max_chars=max_ctx_chars, max_items=k)
        self.llm_decider = get_groq_llm(groq_api_key, model=model, temperature=0.1, max_tokens=256).bind_tools([retrieve])
        self.llm_answer = get_groq_llm(groq_api_key, model=model, temperature=temperature, max_tokens=768)
        self.planner = RunnableWithMessageHistory(
            PLANNER_PROMPT | self.llm_decider,
            lambda sid: _history_store(sid),
            input_messages_key="question",
            history_messages_key="history",
        )
        self.answerer = RunnableWithMessageHistory(
            ANSWER_PROMPT | self.llm_answer,
            lambda sid: _history_store(sid),
            input_messages_key="question",
            history_messages_key="history",
        )
        tools = ToolNode([retrieve])
        graph = StateGraph(MessagesState)
        graph.add_node("plan", self.decide)
        graph.add_node("tools", tools)
        graph.add_node("answer", self.answer)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", tools_condition, {END: END, "tools": "tools"})
        graph.add_edge("tools", "answer")
        graph.add_edge("answer", END)
        self.graph = graph.compile()

    def _format_contexts(self, state_messages: List[Any]) -> str:
        last_tool = None
        for m in reversed(state_messages):
            if getattr(m, "type", "") == "tool":
                last_tool = m
                break
        chunks = []
        if last_tool is not None:
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
        fmt = self.context_formatter
        for name in ("format", "__call__", "format_context", "format_chunks", "build", "render"):
            f = getattr(fmt, name, None)
            if callable(f):
                try:
                    return f(chunks)
                except TypeError:
                    return f()
        return "\n\n".join(x.get("content", "") for x in chunks if isinstance(x, dict))

    def _last_user_text(self, messages: List[Any]) -> str:
        for m in reversed(messages):
            if getattr(m, "type", "") == "human":
                return m.content if isinstance(m.content, str) else ""
        return ""

    def decide(self, state: MessagesState, config: Optional[Dict[str, Any]] = None):
        sid = ((config or {}).get("configurable") or {}).get("thread_id") or "default"
        q = self._last_user_text(state["messages"])
        out = self.planner.invoke({"question": q}, config={"configurable": {"session_id": sid}})
        return {"messages": [out]}

    def answer(self, state: MessagesState, config: Optional[Dict[str, Any]] = None):
        sid = ((config or {}).get("configurable") or {}).get("thread_id") or "default"
        q = self._last_user_text(state["messages"])
        ctx = self._format_contexts(state["messages"])
        msgs = self.answerer.invoke({"question": q, "contexts": ctx}, config={"configurable": {"session_id": sid}})
        return {"messages": [msgs]}

    def qa(self, question: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        cfg = {"configurable": {"thread_id": thread_id or "default"}}
        return self.graph.invoke({"messages": [HumanMessage(content=question)]}, config=cfg)

def create_rag_chain(groq_api_key: Optional[str], *, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2, k: int = 6, max_ctx_chars: int = 8000) -> RAGService:
    return RAGService(groq_api_key, model, temperature, k, max_ctx_chars)
