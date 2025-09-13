from __future__ import annotations
from typing import Dict, Any, List
from langchain.schema import Document
import re
from .Loaders import load_markdown
from .Splitters import split_markdown
from .EmbeddingManager import get_encoder
from .Vectors import build_index as build_vec, load_index as load_vec
from .Retriever import docs_from_chroma, make_hybrid_retriever, CrossEncoderReranker
from langchain_core.tools import tool

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
        try:
            _ENC = get_encoder(batch_size=64)
        except Exception as e:
            raise RuntimeError(f"Failed to load encoder: {e}")
    if _DB is None:
        try:
            _DB = load_vec(_ENC)
        except Exception as e:
            raise RuntimeError(f"Failed to load vector DB: {e}")
    if _CHUNKS_FOR_BM25 is None:
        try:
            _CHUNKS_FOR_BM25 = docs_from_chroma(_DB)
        except Exception as e:
            raise RuntimeError(f"Failed to load BM25 chunks: {e}")
    if _RERANKER is None:
        try:
            _RERANKER = CrossEncoderReranker()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize reranker: {e}")

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
    if _CHUNKS_FOR_BM25 is None:
        raise RuntimeError("BM25 chunks not loaded")
    if _DB is None:
        raise RuntimeError("Vector DB not loaded")
    if mode == "hybrid":
        k_init = max(30, k * 4)
        ret = make_hybrid_retriever(_CHUNKS_FOR_BM25, _DB, k=k_init, weights=(0.6, 0.4))
    else:
        ret = _DB.as_retriever(search_kwargs={"k": k})
    return ret.invoke(query)

@tool(response_format='content_and_artifact')
def retrieve(query: str) -> tuple[str, List[Dict[str, Any]]]:
    """Retrieve information about UIT's academic curriculum (majors, courses, credits, and regulations) from the internal vector database."""
    _ensure_loaded()
    k = 6
    if _CHUNKS_FOR_BM25 is None:
        raise RuntimeError("BM25 chunks not loaded")
    if _RERANKER is None:
        raise RuntimeError("Reranker not initialized")
    k_init = max(60, k * 6)
    weights = (0.3, 0.7)
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
    return query, results

class ContextFormatter:
    def __init__(self, max_chars: int = 8000, max_items: int = 6):
        self.max_chars = max_chars
        self.max_items = max_items

    def _normalize_text(self, piece: str) -> str:
        """
        Chuẩn hóa văn bản mô tả: gom dòng, bỏ ký tự thừa,
        chuyển thành bullet list nếu có dấu chấm phẩy hoặc 'o '.
        """
        text = re.sub(r"\s+", " ", piece).strip()

        if "; " in text or " o " in text:
            parts = re.split(r";|\so\s", text)
            parts = [p.strip(" .") for p in parts if p.strip()]
            return "\n".join(f"- {p}" for p in parts)
        return text

    def _normalize_table(self, piece: str) -> str:
        """
        Chuẩn hóa context dạng bảng môn học sang bảng markdown.
        """
        rows = []
        text = re.sub(r"\s+", " ", piece.replace("|", " "))

        # Regex: Mã môn (chữ + số) + Tên môn + TC + LT + TH
        matches = re.findall(r"([A-Z]{2,}\d+)\s+([^0-9]+?)\s+(\d+)\s+(\d+)\s+(\d+)", text)
        for m in matches:
            ma, ten, tc, lt, th = m
            rows.append((ma, ten.strip(), tc, lt, th))

        if not rows:
            return piece.strip()

        md = "| Mã môn | Tên môn học | TC | LT | TH |\n"
        md += "|--------|-------------|----|----|----|\n"
        for r in rows:
            md += f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |\n"
        return md.strip()

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        buf, total = [], 0
        for i, c in enumerate(chunks[:self.max_items]):
            piece = (c.get("content") or "").strip()
            if not piece:
                continue
            if total + len(piece) > self.max_chars:
                break
            if "|" in piece and re.search(r"[A-Z]{2,}\d+", piece):
                piece_fmt = self._normalize_table(piece)
            else:
                piece_fmt = self._normalize_text(piece)
            buf.append(piece_fmt)
            total += len(piece)
        return "\n\n".join(buf)