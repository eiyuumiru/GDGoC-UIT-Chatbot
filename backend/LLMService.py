from __future__ import annotations
from typing import Any, Dict, List, Optional
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from .FullChain import retrieve, _ensure_loaded, ContextFormatter
from .EmbeddingManager import get_encoder
from langchain_groq import ChatGroq
import os

SYSTEM_INSTRUCTIONS = (
    "Bạn là trợ lý trả lời về Chương Trình Đào Tạo UIT (Khóa 2025).\n"
    "Nguyên tắc:\n"
    "• Suy luận từng bước một cách logic trong tâm trí, nhưng chỉ trả lời kết quả cuối cùng.\n"
    "• Chỉ dùng thông tin đã được cung cấp trong dữ liệu tham chiếu và lịch sử hội thoại.\n"
    "• Phân tích câu hỏi để xác định thông tin cần thiết (môn học, ngành, học kỳ, tín chỉ...).\n"
    "• Nếu dữ liệu chưa đủ, giải thích ngắn gọn thiếu gì và gợi ý cách hỏi cụ thể.\n"
    "• Trả lời tiếng Việt chuẩn, ngắn gọn, rõ ràng với gạch đầu dòng khi phù hợp.\n"
    "• Không đề cập đến quy trình nội bộ, công cụ hay quá trình suy luận.\n"
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTIONS),
        (
            "human",
            "Lịch sử hội thoại gần đây:\n{history}\n\n"
            "Câu hỏi hiện tại: {question}\n\n"
            "Dữ liệu tham chiếu (có thể trống):\n"
            "----------------\n"
            "{contexts}\n"
            "----------------\n\n"
            "Hãy suy luận từng bước trong tâm trí:\n"
            "1. Phân tích câu hỏi để xác định thông tin cần tìm\n"
            "2. Kiểm tra dữ liệu tham chiếu và lịch sử có đủ thông tin không\n"
            "3. Đưa ra câu trả lời ngắn gọn, rõ ràng dựa trên phân tích\n\n"
            "Chỉ trả lời kết quả cuối cùng, không mô tả quá trình suy luận."
        ),
    ]
)

def _collect_tool_chunks_from_state(state: MessagesState) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", None) != "tool":
            break
        artifact = getattr(msg, "artifact", [])
        if isinstance(artifact, list):
            for c in artifact:
                content = c.get("content", "").strip()
                if content:
                    chunks.append(c)
    return chunks[::-1]

def _get_last_user_question(state: MessagesState) -> str:
    for m in reversed(state["messages"]):
        if m.type == "human":
            return str(m.content or "")
    return ""

def _format_recent_history(state: MessagesState, max_chars: int = 2000, max_turns: int = 6) -> str:
    buf = []
    for m in state["messages"]:
        if m.type in ("human", "ai"):
            role = "Người dùng" if m.type == "human" else "Trợ lý"
            text = str(m.content or "").strip()
            if text:
                buf.append(f"{role}: {text}")
    if not buf:
        return ""
    tail = buf[-(max_turns*2):]
    joined = []
    total = 0
    for t in tail:
        if total + len(t) > max_chars:
            break
        joined.append(t)
        total += len(t)
    return "\n".join(joined)

class LLMService(ContextFormatter):
    def __init__(self, groq_api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2, max_tokens: Optional[int] = None):
        _ensure_loaded()
        super().__init__()
        self.llm = self.LLM(groq_api_key=groq_api_key, model=model, temperature=temperature, max_tokens=max_tokens)
        self.encoder = self.Encoder()
        self.memory = self.Memory()
        self.graph = self.Graph()

    def LLM(self, groq_api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2, max_tokens: Optional[int] = None) -> ChatGroq:
        key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("Thiếu GROQ_API_KEY")
        return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)
    
    def Encoder(self):
        return get_encoder()
    
    def Memory(self):
        return MemorySaver()
    
    def query_or_response(self, state: MessagesState):
        planner_system = SystemMessage(
            content=(
                "Bạn là trợ lý phân tích câu hỏi về CTĐT UIT. "
                "Suy luận trong tâm trí: "
                "1. Câu hỏi này cần thông tin cụ thể nào? "
                "2. Lịch sử hội thoại có đủ ngữ cảnh không? "
                "3. Nếu cần dữ liệu chi tiết từ chương trình đào tạo, gọi 'retrieve'. "
                "4. Nếu có thể trả lời từ kiến thức chung, không cần gọi công cụ. "
                "Quyết định ngay, không giải thích quá trình."
            )
        )
        llm_with_tools = self.llm.bind_tools([retrieve])
        response = llm_with_tools.invoke([planner_system] + state["messages"])
        return {"messages": [response]}
    
    def generate_with_context(self, state: MessagesState):
        chunks = _collect_tool_chunks_from_state(state)
        contexts = self.format_context(chunks)
        history = _format_recent_history(state)
        messages = ANSWER_PROMPT.format_messages(question=_get_last_user_question(state), contexts=contexts, history=history)
        out = self.llm.invoke(messages)
        return {"messages": [out]}

    def Graph(self):
        tools = ToolNode([retrieve])
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("query_or_response", self.query_or_response)
        graph_builder.add_node("tools", tools)
        graph_builder.add_node("generate_with_context", self.generate_with_context)
        graph_builder.add_edge(START, "query_or_response")
        graph_builder.add_conditional_edges("query_or_response", tools_condition, {END: END, "tools": "tools"})
        graph_builder.add_edge("tools", "generate_with_context")
        graph_builder.add_edge("generate_with_context", END)
        return graph_builder.compile(checkpointer=self.memory)

    def __call__(self, question: str, thread_id: str = "default_session", topk: Optional[int] = None) -> Any:
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke({"messages": [HumanMessage(content=question)]}, config=config)