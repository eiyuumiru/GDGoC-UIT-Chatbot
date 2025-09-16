import os
import sys
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import streamlit as st
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from backend.LLMService import LLMService

st.set_page_config(page_title="UIT RAG Chatbot", page_icon="🎓", layout="centered")
st.header("🎓 UIT RAG Chatbot")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

def get_secret_or_input(secret_key: str, label: str, help_url: Optional[str] = None) -> Optional[str]:
    if hasattr(st, "secrets") and secret_key in st.secrets:
        return st.secrets[secret_key]
    if os.getenv(secret_key):
        return os.getenv(secret_key)
    val = st.text_input(label, type="password")
    if not val and help_url:
        st.caption(f"Chưa có key? Xem: {help_url}")
    return val or None

groq_key = st.session_state.get("GROQ_API_KEY") or get_secret_or_input(
    "GROQ_API_KEY", "GROQ API key", "https://console.groq.com/keys"
)
if groq_key:
    st.session_state["GROQ_API_KEY"] = groq_key

if not st.session_state.get("GROQ_API_KEY"):
    st.warning("Vui lòng nhập GROQ_API_KEY để bắt đầu.")
    st.stop()

model = "llama-3.3-70b-versatile"
temperature = 0.5
k = 6

@st.cache_resource(show_spinner=False)
def make_chain(groq_api_key: str, model: str, temperature: float, k: int) -> Callable[[str], Dict[str, Any]]:
    return LLMService(groq_api_key=groq_api_key, model=model, temperature=temperature)

qa = make_chain(
    groq_api_key=st.session_state["GROQ_API_KEY"],
    model=model,
    temperature=temperature,
    k=k,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Xin chào! Hỏi gì về CTĐT UIT 2025?"}]

col_left, col_center, col_right = st.columns([1, 1, 1])
with col_center:
    if st.button("Bắt đầu cuộc trò chuyện mới", use_container_width=True):
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["messages"] = [{"role": "assistant", "content": "Xin chào! Hỏi gì về CTĐT UIT 2025?"}]
        st.rerun()

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])

prompt = st.chat_input("Đặt câu hỏi của bạn…")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất & tổng hợp..."):
            out = qa(prompt, thread_id=st.session_state["session_id"])
            answer = out["messages"][-1].content
            st.markdown(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})