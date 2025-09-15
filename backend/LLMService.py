from __future__ import annotations
from typing import Any, Dict, List, Optional
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.memory import InMemoryStore
from langchain_core.prompts import ChatPromptTemplate
from .FullChain import retrieve, _ensure_loaded, ContextFormatter
from .EmbeddingManager import get_encoder
from langchain_groq import ChatGroq
import os

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
    """Lấy artifact từ các ToolMessage gần nhất, scan ngắn gọn."""
    chunks: List[Dict[str, Any]] = []
    # chỉ scan các ToolMessage gần nhất (reversed)
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", None) != "tool":
            break
        artifact = getattr(msg, "artifact", [])
        if isinstance(artifact, list):
            for c in artifact:
                content = c.get("content", "").strip()
                if content:
                    chunks.append(c)
    return chunks[::-1]  # giữ thứ tự thời gian

def _get_last_user_question(state: MessagesState) -> str:
    for m in reversed(state["messages"]):
        if m.type == "human":
            return str(m.content or "")
    return ""

class LLMService(ContextFormatter):
    def __init__(self, groq_api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2, max_tokens: Optional[int] = None):
        _ensure_loaded()
        super().__init__()
        self.llm = self.__init_LLM(groq_api_key=groq_api_key, model=model, temperature=temperature, max_tokens=max_tokens)
        self.encoder = self.__init_Encoder()
        self.store = self.__init_Storage()
        self.graph = self.__init_Graph()

    def __init_LLM(self, groq_api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2, max_tokens: Optional[int] = None) -> ChatGroq:
        key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("Thiếu GROQ_API_KEY")
        return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)
    
    def __init_Encoder(self):
        return get_encoder()
    
    def __init_Storage(self):
        return InMemoryStore(
            index={
                "dims": 1536,
                "embed": self.encoder
            }
        )
    
    def query_or_response(self, state: MessagesState):
        """Node 1: Cho phép LLM quyết định có cần tool."""
        planner_system = SystemMessage(
            content=(
                "Bạn là một trợ lý thông minh cho CTĐT UIT. "
                "Nếu câu hỏi cần chi tiết cụ thể từ dữ liệu chương trình (môn học, tín chỉ, học kỳ, điều kiện, quy định...), "
                "hãy gọi công cụ 'retrieve' với truy vấn ngắn gọn tiếng Việt. "
                "Nếu có thể trả lời ngay, đừng gọi công cụ."
            )
        )
        llm_with_tools = self.llm.bind_tools([retrieve])
        response = llm_with_tools.invoke([planner_system] + state["messages"])
        return {"messages": [response]}
    
    def generate_with_context(self, state: MessagesState):
        """Node 2: Sau khi (có thể) đã gọi tool, tổng hợp trả lời bằng PromptTemplate."""
        chunks = _collect_tool_chunks_from_state(state)
        contexts = self.format_context(chunks)
        messages = ANSWER_PROMPT.format_messages(question=_get_last_user_question(state), contexts=contexts)
        out = self.llm.invoke(messages)
        return {"messages": [out]}

    def __init_Graph(self):
        tools = ToolNode([retrieve])
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("query_or_response", self.query_or_response)
        graph_builder.add_node("tools", tools)
        graph_builder.add_node("generate_with_context", self.generate_with_context)
        graph_builder.add_edge(START, "query_or_response")
        graph_builder.add_conditional_edges("query_or_response", tools_condition, {END: END, "tools": "tools"})
        graph_builder.add_edge("tools", "generate_with_context")
        graph_builder.add_edge("generate_with_context", END)
        return graph_builder.compile()
    
    def visualize(self, filename: str = "llm_service_graph"):
        bytes = self.graph.get_graph().draw_mermaid_png()
        with open(f"{filename}.png", "wb") as f:
            f.write(bytes)

    def __call__(self, question: str, topk: Optional[int] = None) -> Any:
        return self.graph.invoke({"messages": [HumanMessage(content=question)]})