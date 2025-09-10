import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import streamlit as st
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from backend.FullChain import build_index
from backend.LLMService import create_rag_chain

st.set_page_config(page_title="UIT RAG Chatbot", page_icon="🎓", layout="centered")
st.header("🎓 UIT RAG Chatbot")

def get_secret_or_input(secret_key: str, label: str, help_url: Optional[str] = None) -> Optional[str]:
    if hasattr(st, "secrets") and secret_key in st.secrets:
        return st.secrets[secret_key]
    if os.getenv(secret_key):
        return os.getenv(secret_key)
    val = st.text_input(label, type="password")
    if not val and help_url:
        st.caption(f"Chưa có key? Xem: {help_url}")
    return val or None

with st.sidebar:
    st.subheader("⚙️ Cấu hình")
    groq_key = st.session_state.get("GROQ_API_KEY") or get_secret_or_input("GROQ_API_KEY", "GROQ API key", "https://console.groq.com/keys")
    if groq_key:
        st.session_state["GROQ_API_KEY"] = groq_key
    model = st.selectbox("Model Groq", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b", "openai/gpt-oss-20b"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    k = st.slider("Số đoạn ngữ cảnh (k)", min_value=3, max_value=12, value=6, step=1)
    do_rebuild = st.toggle("Rebuild index ngay", value=False)
    st.button("🧹 Xóa lịch sử chat", on_click=lambda: st.session_state.pop("messages", None))

@st.cache_resource(show_spinner=False)
def make_chain(groq_api_key: str, model: str, temperature: float, k: int) -> Callable[[str], Dict[str, Any]]:
    return create_rag_chain(groq_api_key=groq_api_key, model=model, temperature=temperature, k=k)

if not st.session_state.get("GROQ_API_KEY"):
    st.warning("Vui lòng nhập GROQ_API_KEY ở Sidebar để bắt đầu.")
    st.stop()

if do_rebuild:
    with st.spinner("Đang rebuild index từ backend/dataset..."):
        stats = build_index(data_dir="backend/dataset")
    st.success(f"Rebuild xong: {stats}", icon="✅")

qa = make_chain(
    groq_api_key=st.session_state["GROQ_API_KEY"],
    model=model,
    temperature=temperature,
    k=k,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Xin chào! Hỏi gì về CTĐT UIT 2025?"}]

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])

def render_sources(srcs: List[str]):
    if not srcs:
        return
    with st.expander("Nguồn tham khảo (retrieved)"):
        for s in srcs:
            st.write(f"- {s}")

prompt = st.chat_input("Đặt câu hỏi của bạn…")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất & tổng hợp..."):
            out = qa(prompt)
            answer = out.get("text", "Không tạo được câu trả lời.")
            st.markdown(answer)
            render_sources(out.get("sources", []))
    st.session_state["messages"].append({"role": "assistant", "content": answer})
