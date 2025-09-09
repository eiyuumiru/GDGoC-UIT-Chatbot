from typing import Dict, Any, List
from langchain.schema import Document

from .Loaders import load_markdown
from .Splitters import split_markdown
from .EmbeddingManager import get_encoder
from .Vectors import build_index as build_vec, load_index as load_vec, get_retriever

def build_index(data_dir: str = "backend/dataset"):
    print("Phase 1/3: load docs ...", flush=True)
    docs = load_markdown(data_dir)
    enc = get_encoder(batch_size=64)

    print("Phase 2/3: semantic split ...", flush=True)
    chunks = split_markdown(docs, embeddings=enc, show_progress=True)

    print(f"Phase 3/3: build index (chunks={len(chunks)}) ...", flush=True)
    db = build_vec(chunks, enc, batch_size=24, show_progress=True)
    return {"docs": len(docs), "chunks": len(chunks), "count": db._collection.count()}

def search(query: str, k: int = 4) -> List[Document]:
    enc = get_encoder()
    db  = load_vec(enc)
    retriever = get_retriever(db, k=k)
    return retriever.invoke(query)

def ask_question(query: str, k: int = 4) -> Dict[str, Any]:
    hits = search(query, k=k)
    return {
        "query": query,
        "results": [
            {"source": h.metadata.get("source"), "content": h.page_content[:800]}
            for h in hits
        ],
    }
