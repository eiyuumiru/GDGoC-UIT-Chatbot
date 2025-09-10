from typing import Dict, Any, List
from langchain.schema import Document

from .Loaders import load_markdown
from .Splitters import split_markdown
from .EmbeddingManager import get_encoder
from .Vectors import build_index as build_vec, load_index as load_vec, get_retriever
from .Retriever import docs_from_chroma, make_hybrid_retriever, CrossEncoderReranker


_DB = None
_ENC = None
_CHUNKS_FOR_BM25 = None
_RERANKER = None

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

def _ensure_loaded():
    global _DB, _ENC, _CHUNKS_FOR_BM25, _RERANKER
    if _ENC is None:
        _ENC = get_encoder(batch_size=24)
    if _DB is None:
        _DB = load_vec(_ENC)
    if _CHUNKS_FOR_BM25 is None:
        _CHUNKS_FOR_BM25 = docs_from_chroma(_DB)
    if _RERANKER is None:
        _RERANKER = CrossEncoderReranker()


def ask_question(query: str, k: int = 6):
    _ensure_loaded()

    k_init = max(30, k * 4)
    weights = (0.65, 0.35)
    ret = make_hybrid_retriever(_CHUNKS_FOR_BM25, _DB, k=k_init, weights=weights)
    cand_docs = ret.invoke(query)

    ranked = _RERANKER.rerank(query, cand_docs, topn=k)

    results = []
    for d, score in ranked:
        results.append({
            "source": d.metadata.get("source", ""),
            "score": float(score),
            "content": d.page_content
        })
    return {"query": query, "k": k, "results": results}