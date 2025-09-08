from langchain.memory import ChatMessageHistory


def ask_question(chain, query):
    response = chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": "foo"}}
    )
    return response

def create_full_chain(retriever, groq_api_key=None, chat_memory=ChatMessageHistory()):
    pass