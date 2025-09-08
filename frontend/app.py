import streamlit as st
from langchain_groq import ChatGroq
from backend.FullChain import ask_question, create_full_chain

st.set_page_config(page_title="Chat with Document", page_icon=":shark:")
st.header("Chat with Document")

@st.cache_resource
def load_vector_store():
    pass
    
def get_chain(groq_api_key = None):    
    pass

def show_ui(qa, prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(qa, prompt)
                st.markdown(response.content)
        message = {"role": "assistant", "content": response.content}
        st.session_state.messages.append(message)

def get_secret_or_input(secret_key, secret_name, info_link=None):
    if secret_key in st.secrets:
        return st.secrets[secret_key]
    st.write(f"Please provide your {secret_name}")
    secret_value = st.text_input(secret_name, key=f"input_{secret_key}", type="password")
    if secret_value:
        st.session_state[secret_key] = secret_value
    if info_link:
        st.markdown(f"[Get an {secret_name}]({info_link})")
    return secret_value

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Xin chao! Mình có thể giúp gì cho bạn?"}]

def run():
    ready = True

    with st.sidebar:
        if not st.session_state.get("GROQ_API_KEY"):
            st.session_state["GROQ_API_KEY"] = get_secret_or_input('GROQ_API_KEY', "GROQ API key")
        else:
            st.title('💬 RAG Chatbot Project')
            st.write('This chatbot is created by GDG-UIT')
            st.success('GROQ API key already provided!', icon='✅')
            st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
            
    if not st.session_state.get("GROQ_API_KEY"):
        st.warning("Missing GROQ_API_KEY")
        ready = False

    if ready:
        chain = get_chain(groq_api_key=st.session_state["GROQ_API_KEY"])
        st.subheader("Ask me questions about this week's meal plan")
        show_ui(chain, "Xin chao! Mình có thể giúp gì cho bạn?")
    else:
        st.stop()

if __name__ == "__main__":
    run()