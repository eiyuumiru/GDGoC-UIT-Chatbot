from __future__ import annotations
import re
from typing import Callable, List, Dict, Any, Optional
import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from .FullChain import retrieve, _ensure_loaded, ContextFormatter

def get_groq_llm(
    groq_api_key: Optional[str] = None,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> ChatGroq:
    key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Thiếu GROQ_API_KEY")
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


SYSTEM_INSTRUCTIONS = (
    "Bạn là trợ lý trả lời về Chương Trình Đào Tạo UIT (Khóa 2025).\n"
    "Nguyên tắc:\n"
    "• Chỉ dùng thông tin đã được cung cấp trong dữ liệu tham chiếu (nếu có). Không bịa.\n"
    "• Nếu dữ liệu chưa đủ để kết luận, nói ngắn gọn rằng chưa đủ và gợi ý cách hỏi cụ thể hơn.\n"
    "• Trả lời tiếng Việt chuẩn, ngắn gọn, mạch lạc; có thể dùng gạch đầu dòng khi phù hợp.\n"
    "• Không đề cập đến quy trình nội bộ, công cụ hay cách bạn có dữ liệu.\n"
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTIONS),
        (
            "human",
            "Câu hỏi: {question}\n\n"
            "Dữ liệu tham chiếu (có thể trống):\n"
            "----------------\n"
            "{contexts}\n"
            "----------------\n\n"
            "Yêu cầu: Dựa vào dữ liệu trên để trả lời ngắn gọn, rõ ràng. "
            "Nếu dữ liệu không đủ, hãy nói rằng chưa đủ thông tin và gợi ý cách hỏi cụ thể hơn."
        ),
    ]
)


def _collect_tool_chunks_from_state(state: MessagesState) -> List[Dict[str, Any]]:
    """Lấy artifact từ các ToolMessage gần nhất (nếu có)."""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    chunks: List[Dict[str, Any]] = []
    for m in tool_messages:
        if hasattr(m, "artifact") and isinstance(m.artifact, list):
            for c in m.artifact:
                if isinstance(c, dict):
                    content = (c.get("content") or "").strip()
                    if content:
                        chunks.append({"content": content})
    return chunks


def _get_last_user_question(state: MessagesState) -> str:
    for m in reversed(state["messages"]):
        if m.type == "human":
            return str(m.content or "")
    return ""

def create_rag_chain(
    groq_api_key: Optional[str],
    *,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    k: int = 6,
    max_ctx_chars: int = 8000,
) -> Callable[[str], Dict[str, Any]]:

    _ensure_loaded()

    llm_decider = get_groq_llm(
        groq_api_key, model=model, temperature=temperature, max_tokens=7000
    ).bind_tools([retrieve])

    llm_answer = get_groq_llm(
        groq_api_key, model=model, temperature=temperature, max_tokens=7000
    )

    context_formatter = ContextFormatter(max_chars=max_ctx_chars, max_items=k)

    def query_or_response(state: MessagesState):
        """
        Node 1: Cho phép LLM quyết định có cần tool.
        """
        planner_system = SystemMessage(
            content=(
                "Bạn là một trợ lý thông minh cho CTĐT UIT. "
                "Nếu câu hỏi cần chi tiết cụ thể từ dữ liệu chương trình (môn học, tín chỉ, học kỳ, điều kiện, quy định...), "
                "hãy gọi công cụ 'retrieve' với truy vấn ngắn gọn tiếng Việt. "
                "Nếu có thể trả lời ngay, đừng gọi công cụ."
            )
        )
        response = llm_decider.invoke([planner_system] + state["messages"])
        return {"messages": [response]}

    def generate_with_context(state: MessagesState):
        """
        Node 2: Sau khi (có thể) đã gọi tool, tổng hợp trả lời bằng PromptTemplate.
        KHÔNG nhắc đến trích dẫn/nguồn, KHÔNG hiển thị source.
        """
        chunks = _collect_tool_chunks_from_state(state)
        contexts = context_formatter.format_context(chunks)
        print(contexts)  # Debug
        question = _get_last_user_question(state)
        messages = ANSWER_PROMPT.format_messages(question=question, contexts=contexts)
        out = llm_answer.invoke(messages)
        return {"messages": [out]}

    tools = ToolNode([retrieve])

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("query_or_response", query_or_response)
    graph_builder.add_node("tools", tools)
    graph_builder.add_node("generate_with_context", generate_with_context)

    graph_builder.add_edge(START, "query_or_response")
    graph_builder.add_conditional_edges("query_or_response", tools_condition, {END: END, "tools": "tools"})
    graph_builder.add_edge("tools", "generate_with_context")
    graph_builder.add_edge("generate_with_context", END)
    graph = graph_builder.compile()

    def qa(question: str, topk: Optional[int] = None) -> Dict[str, Any]:
        result = graph.invoke({"messages": [HumanMessage(content=question)]})
        return result

    return qa
