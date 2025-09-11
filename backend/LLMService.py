from __future__ import annotations
from typing import Callable, List, Dict, Any, Optional
import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.constants import END
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
from .FullChain import ask_question, _ensure_loaded


graph_builder = StateGraph(MessagesState)

SYSTEM_INSTRUCTIONS = (
    "Bạn là trợ lý trả lời câu hỏi cho chương trình đào tạo UIT (Khóa 2025). "
    "Chỉ trả lời dựa trên NGỮ CẢNH và thông tin được cung cấp; nếu thiếu hãy nói rõ "
    "'không có trong dữ liệu'. Trả lời ngắn gọn, chính xác, tiếng Việt. "
    "Khi có số liệu hoặc quy định, nêu rõ. "
    "Cuối cùng phải có mục 'Nguồn:' liệt kê các nguồn đã dùng theo thứ tự xuất hiện."
)


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

def _format_context(chunks: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    buf, total = [], 0
    for i, r in enumerate(chunks, 1):
        src = r.get("source") or ""
        content = (r.get("content") or "").strip()
        piece = f"<CTX id=\"{i}\" source=\"{src}\">\n{content}\n</CTX>"
        if total + len(piece) > max_chars:
            break
        buf.append(piece)
        total += len(piece)
    return "\n".join(buf)

def _few_shot_examples() -> str:
    ex_ctx = "<CTX id=\"1\" source=\"backend/dataset/majors/cs.md\">Môn CS311 học kỳ 6; tổng số tín chỉ bắt buộc là 3.</CTX>\n<CTX id=\"2\" source=\"backend/dataset/policies/graduation.md\">Điều kiện tốt nghiệp: tích lũy đủ tín chỉ, hoàn thành chuẩn đầu ra.</CTX>"
    ex_user = "CS311 học ở kỳ mấy và có bao nhiêu tín chỉ?"
    ex_out = "CS311 học ở kỳ 6, 3 tín chỉ.\n\nNguồn: [1]"
    ex2_ctx = "<CTX id=\"1\" source=\"backend/dataset/majors/is.md\">Ngành Hệ thống thông tin: tối thiểu 125 tín chỉ.</CTX>"
    ex2_user = "Số tín chỉ tối thiểu ngành Hệ thống thông tin?"
    ex2_out = "Tối thiểu 125 tín chỉ.\n\nNguồn: [1]"
    return f"<EXAMPLES>\n<EXAMPLE>\n<CONTEXTS>\n{ex_ctx}\n</CONTEXTS>\n<QUESTION>{ex_user}</QUESTION>\n<IDEAL_ANSWER>{ex_out}</IDEAL_ANSWER>\n</EXAMPLE>\n<EXAMPLE>\n<CONTEXTS>\n{ex2_ctx}\n</CONTEXTS>\n<QUESTION>{ex2_user}</QUESTION>\n<IDEAL_ANSWER>{ex2_out}</IDEAL_ANSWER>\n</EXAMPLE>\n</EXAMPLES>"

def build_prompt(state: MessagesState):
    rules = (
        "Quy tắc:\n"
        "1) Chỉ dùng thông tin trong CONTEXTS.\n"
        "2) Nếu thiếu dữ liệu cần thiết, trả lời: 'không có trong dữ liệu'.\n"
        "3) Trích nguồn bằng chỉ số [i] theo CTX id đã cho.\n"
        "4) Ưu tiên liệt kê gọn, rõ.\n"
        "5) Đầu ra gồm hai phần:\n"
        "   - Trả lời: phần nội dung chính\n"
        "   - Nguồn: danh sách chỉ số [i] theo thứ tự sử dụng"
    )

    # Lấy messages từ tool (đã thực thi)
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Chuyển ToolMessage thành dict để format
    docs_for_ctx = []
    for m in tool_messages:
        if hasattr(m, "artifact") and isinstance(m.artifact, list):
            for c in m.artifact:
                if isinstance(c, dict):
                    docs_for_ctx.append(c)

    ctx_block = _format_context(docs_for_ctx)
    fewshot = _few_shot_examples()

    conservation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    system_message_content = (
        SYSTEM_INSTRUCTIONS + "\n\n" + rules + "\n\n" + ctx_block + "\n\n" + fewshot
    )
    prompt = [SystemMessage(content=system_message_content)] + conservation_messages
    return prompt

def create_rag_chain(
    groq_api_key: Optional[str],
    *,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    k: int = 6,
    max_ctx_chars: int = 8000,
) -> Callable[[str], Dict[str, Any]]:
    _ensure_loaded()
    llm = get_groq_llm(groq_api_key, model=model, temperature=temperature)

    def query_or_response(state: MessagesState):
        llm_with_tools = llm.bind_tools([ask_question])
        response = llm_with_tools.invoke(state['messages'])
        return {"messages": [response]}

    def generate_with_context(state: MessagesState):
        messages = build_prompt(state)
        out = llm.invoke(messages)
        return {"messages": [out]}

    # ToolNode chỉ nhận tool thật sự
    tools = ToolNode([ask_question])

    # Build graph
    graph_builder.add_node(query_or_response)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate_with_context)

    graph_builder.set_entry_point("query_or_response")
    graph_builder.add_conditional_edges(
        "query_or_response",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate_with_context")
    graph_builder.add_edge("generate_with_context", END)

    graph = graph_builder.compile()

    def qa(question: str, topk: Optional[int] = None) -> Dict[str, Any]:
        for step in graph.stream({"messages": [HumanMessage(content=question)]}):
            print("---- Step ----")
            print(step)
        return graph.invoke({"messages": [HumanMessage(content=question)]})

    return qa
