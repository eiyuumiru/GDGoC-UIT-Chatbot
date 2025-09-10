from __future__ import annotations

from typing import Dict, Any, List
from langchain.schema import Document

from .Loaders import load_markdown
from .Splitters import split_markdown
from .EmbeddingManager import get_encoder
from .Vectors import build_index as build_vec, load_index as load_vec
from .Retriever import docs_from_chroma, make_hybrid_retriever, CrossEncoderReranker


_DB = None
_ENC = None
_CHUNKS_FOR_BM25: List[Document] | None = None
_RERANKER: CrossEncoderReranker | None = None

def _refresh_caches(db=None, enc=None):
    global _DB, _ENC, _CHUNKS_FOR_BM25, _RERANKER
    if enc is not None:
        _ENC = enc
    if db is not None:
        _DB = db
        _CHUNKS_FOR_BM25 = None
    if _RERANKER is None:
        _RERANKER = CrossEncoderReranker()

def _ensure_loaded():
    global _DB, _ENC, _CHUNKS_FOR_BM25, _RERANKER
    if _ENC is None:
        _ENC = get_encoder(batch_size=64)
    if _DB is None:
        _DB = load_vec(_ENC)
    if _CHUNKS_FOR_BM25 is None:
        _CHUNKS_FOR_BM25 = docs_from_chroma(_DB)
    if _RERANKER is None:
        _RERANKER = CrossEncoderReranker()

def build_index(
    data_dir: str = "backend/dataset",
    percentile: int = 92,
    enforce_max: int = 850,
    overlap: int = 120,
    batch_size: int = 64,
) -> Dict[str, Any]:
    
    print("Phase 1/3: load docs ...", flush=True)
    docs = load_markdown(data_dir)
    print(f"Docs: {len(docs)}", flush=True)
    print("Phase 2/3: semantic split ...", flush=True)
    enc = get_encoder(batch_size=batch_size)
    chunks = split_markdown(
        docs,
        encoder=enc,
        percentile=percentile,
        enforce_max=enforce_max,
        overlap=overlap,
        show_progress=True,
    )
    print(f"Chunks: {len(chunks)}", flush=True)
    print("Phase 3/3: build index ...", flush=True)
    db = build_vec(chunks, enc, batch_size=batch_size, show_progress=True)
    _refresh_caches(db=db, enc=enc)
    count = getattr(db._collection, "count")() if hasattr(db, "_collection") else None
    return {"docs": len(docs), "chunks": len(chunks), "count": count}

def search(query: str, k: int = 4, mode: str = "dense") -> List[Document]:
    _ensure_loaded()
    if mode == "hybrid":
        k_init = max(30, k * 4)
        ret = make_hybrid_retriever(_CHUNKS_FOR_BM25, _DB, k=k_init, weights=(0.65, 0.35))
    else:
        ret = _DB.as_retriever(search_kwargs={"k": k})
    return ret.invoke(query)


def ask_question(query: str, k: int = 6) -> Dict[str, Any]:
    _ensure_loaded()

    k_init = max(60, k * 6)
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