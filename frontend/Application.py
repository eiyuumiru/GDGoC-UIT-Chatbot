import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import streamlit as st
from uuid import uuid4
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from backend.LLMService import create_rag_chain

st.set_page_config(page_title="UIT RAG Chatbot", page_icon="🎓", layout="centered")
st.header("🎓 UIT RAG Chatbot")

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid4())

st.subheader("Bộ nhớ hội thoại")
outer_left, outer_main, outer_right = st.columns([1,8,1])
with outer_main:
    r1c1, r1c2, r1c3 = st.columns([3,1.2,1.6])
    with r1c1:
        st.text_input("Thread ID hiện tại", value=st.session_state["thread_id"], key="thread_id_display", disabled=True, label_visibility="collapsed", placeholder="Thread ID")
    with r1c2:
        if st.button("Tạo thread mới", use_container_width=True):
            st.session_state["thread_id"] = str(uuid4())
            st.session_state["messages"] = [{"role": "assistant", "content": "Đã tạo cuộc trò chuyện mới."}]
            st.rerun()
    with r1c3:
        if st.button("Xoá toàn bộ memory", use_container_width=True):
            db_path = Path(__file__).resolve().parent.parent / "backend" / ".index" / "graph_memory.sqlite"
            try:
                if db_path.exists():
                    db_path.unlink()
            except Exception as e:
                st.error(f"Lỗi: {e}")
            st.session_state["thread_id"] = str(uuid4())
            st.session_state["messages"] = [{"role": "assistant", "content": "Đã xoá bộ nhớ và khởi tạo hội thoại mới."}]
            st.rerun()
    db_path = Path(__file__).resolve().parent.parent / "backend" / ".index" / "graph_memory.sqlite"

def get_secret_or_input(secret_key: str, label: str, help_url: Optional[str] = None) -> Optional[str]:
    if hasattr(st, "secrets") and secret_key in st.secrets:
        return st.secrets[secret_key]
    if os.getenv(secret_key):
        return os.getenv(secret_key)
    val = st.text_input(label, type="password")
    if not val and help_url:
        st.caption(f"Chưa có key? Xem: {help_url}")
    return val or None

groq_key = st.session_state.get("GROQ_API_KEY") or get_secret_or_input("GROQ_API_KEY", "GROQ API key", "https://console.groq.com/keys")
if groq_key:
    st.session_state["GROQ_API_KEY"] = groq_key

if not st.session_state.get("GROQ_API_KEY"):
    st.warning("Vui lòng nhập GROQ_API_KEY để bắt đầu.")
    st.stop()

model = "llama-3.3-70b-versatile"
temperature = 0.2
k = 6

@st.cache_resource(show_spinner=False)
def make_service(groq_api_key: str, model: str, temperature: float, k: int):
    return create_rag_chain(groq_api_key=groq_api_key, model=model, temperature=temperature, k=k)

service = make_service(st.session_state["GROQ_API_KEY"], model, temperature, k)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Xin chào! Hỏi gì về CTĐT UIT 2025?"}]

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
            out = (service.qa if hasattr(service, "qa") else service)(prompt, thread_id=st.session_state["thread_id"])
            answer = out["messages"][-1].content
            st.markdown(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
