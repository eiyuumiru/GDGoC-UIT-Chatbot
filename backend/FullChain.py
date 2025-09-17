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
    min_chunk_chars: int = 320,
    soft_merge_chars: int = 160,
    prefix_headers: bool = True,
    clear_existing: bool = True,
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
        min_chunk_chars=min_chunk_chars,
        soft_merge_chars=soft_merge_chars,
        prefix_headers=prefix_headers,
    )
    print(f"Chunks: {len(chunks)}", flush=True)
    print("Phase 3/3: build index ...", flush=True)
    db = build_vec(chunks, enc, batch_size=batch_size, show_progress=True, clear_existing=clear_existing)
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
        retriever = make_hybrid_retriever(_CHUNKS_FOR_BM25, _DB, k=k, weights=(0.6, 0.4))
        pool = max(40, k * 4)
        candidates = retriever.gather(query, fetch_k=pool)
        return [cand.doc for cand in candidates[:k]]
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
    retriever = make_hybrid_retriever(
        _CHUNKS_FOR_BM25,
        _DB,
        k=k,
        weights=weights,
        pool_multiplier=5.0,
        term_weight=0.4,
    )
    candidates = retriever.gather(query, fetch_k=k_init)
    candidate_docs = [cand.doc for cand in candidates]

    if not candidates:
        fallback = _DB.as_retriever(search_kwargs={"k": k_init}).invoke(query)
        candidate_docs = list(fallback)
        ranked = _RERANKER.rerank(query, candidate_docs, topn=k)
    else:
        ranked = _RERANKER.rerank(query, candidates, topn=k)
        if not ranked and candidate_docs:
            ranked = _RERANKER.rerank(query, candidate_docs, topn=k)

    if not ranked:
        ranked = [(doc, 0.0) for doc in candidate_docs[:k]]

    print(f"[retrieve] query: {query}", flush=True)
    print(f"[retrieve] top_k: {k} | initial_k: {k_init} | candidates: {len(candidate_docs)}", flush=True)
    for idx, (doc, score) in enumerate(ranked, start=1):
        chunk_text = (doc.page_content or "").strip().replace("\n", " ")
        if len(chunk_text) > 300:
            chunk_text = chunk_text[:300].rstrip() + "..."
        source = doc.metadata.get("source", "")
        rerank_info = (doc.metadata or {}).get("_rerank", {})
        extra = ""
        if rerank_info:
            extra = (
                f" | ce={float(rerank_info.get('cross_encoder', 0.0)):.3f}"
                f" | coarse={float(rerank_info.get('coarse_norm', 0.0)):.3f}"
                f" | hits={rerank_info.get('term_hits', 0)}"
            )
        print(
            f"[retrieve] #{idx} score={float(score):.4f} source={source}{extra} chunk={chunk_text}",
            flush=True,
        )

    results = []
    for doc, score in ranked:
        record = {
            "source": doc.metadata.get("source", ""),
            "score": float(score),
            "content": doc.page_content,
        }
        rerank_info = (doc.metadata or {}).get("_rerank")
        if rerank_info:
            record["rerank"] = rerank_info
        results.append(record)
    return query, results



class ContextFormatter:
    def __init__(self, max_chars: int = 8000, max_items: int = 6):
        self.max_chars = max_chars
        self.max_items = max_items

    def _normalize_text(self, piece: str) -> str:
        """
        Chu��cn hA3a v��n b���n mA' t���: gom dA�ng, b��? kA� t��� th���a,
        chuy���n thA�nh bullet list n���u cA3 d���u ch���m ph��cy ho���c 'o '.
        """
        raw = (piece or "").strip()
        if not raw:
            return ""
        if "\n" in raw:
            lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines() if line.strip()]
            if lines and all(((":" in line) and line.index(":") <= 24) for line in lines):
                return "\n".join(f"- {line}" for line in lines)
            text = " ".join(lines)
        else:
            text = re.sub(r"\s+", " ", raw).strip()

        if "; " in text or " o " in text:
            parts = re.split(r";|\so\s", text)
            parts = [p.strip(" .") for p in parts if p.strip()]
            if parts:
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
