from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional
from collections.abc import Sequence

from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_TOKEN_PATTERN = re.compile(r"[\w\-À-ỹ]+", re.UNICODE)


def _normalize_weights(weights: Sequence[float]) -> list[float]:
    if not weights:
        return [0.5, 0.5]
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Hybrid weights must sum to a positive number.")
    return [float(w) / total for w in weights]


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [tok for tok in _TOKEN_PATTERN.findall(text.lower()) if tok]


@dataclass
class HybridCandidate:
    doc: Document
    lexical_rank: int | None = None
    lexical_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    rrf_score: float = 0.0
    term_hits: int = 0
    lexical_norm: float = 0.0
    dense_norm: float = 0.0
    term_norm: float = 0.0
    coarse_score: float = 0.0


class HybridRetriever:
    def __init__(
        self,
        chunks: Sequence[Document],
        db,
        *,
        default_k: int = 10,
        weights: Sequence[float] = (0.6, 0.4),
        pool_multiplier: float = 4.0,
        rrf_constant: float = 60.0,
        rrf_scale: float = 20.0,
        term_weight: float = 0.3,
        id_keys: Sequence[str] | None = None,
    ):
        self.default_k = max(1, int(default_k))
        self._db = db
        weight_list = _normalize_weights(weights)
        if len(weight_list) < 2:
            weight_list = (weight_list + weight_list)[:2]
        self._weights = weight_list[:2]
        self._rrf_constant = max(float(rrf_constant), 1.0)
        self._rrf_scale = max(float(rrf_scale), 1.0)
        self._term_weight = max(float(term_weight), 0.0)
        self._pool_multiplier = max(float(pool_multiplier), 1.5)
        self._bm25 = BM25Retriever.from_documents(chunks, preprocess_func=_tokenize)
        self._bm25.k = self.default_k
        self._bm25_docs = self._bm25.docs
        self._id_keys = tuple(id_keys or ("_id", "chunk_index", "chunk_base", "source"))
        self._pool_floor = max(
            int(self.default_k * self._pool_multiplier),
            self.default_k,
            40,
        )

    def _doc_id(self, doc: Document) -> str:
        meta = doc.metadata or {}
        for key in self._id_keys:
            value = meta.get(key)
            if value:
                return str(value)
        if getattr(doc, "id", None):
            return str(doc.id)
        return doc.page_content

    def _collect_term_hits(self, doc: Document, tokens: Sequence[str]) -> int:
        if not tokens:
            return 0
        meta = doc.metadata or {}
        haystacks = [(doc.page_content or "").lower()]
        for key in ("section_path", "source", "h1", "h2", "h3", "h4", "title"):
            value = meta.get(key)
            if value:
                haystacks.append(str(value).lower())
        combined = " ".join(haystacks)
        hits = 0
        for token in tokens:
            if token:
                hits += combined.count(token)
        return hits

    def _component_limit(self, fetch_k: Optional[int]) -> int:
        pool = fetch_k or self._pool_floor
        return max(pool, self.default_k)

    def gather(self, query: str, fetch_k: Optional[int] = None) -> list[HybridCandidate]:
        pool = self._component_limit(fetch_k)
        candidates: dict[str, HybridCandidate] = {}
        tokens = _tokenize(query)

        lexical_limit = min(pool, len(self._bm25_docs))
        if lexical_limit and tokens:
            scores = self._bm25.vectorizer.get_scores(tokens)
            try:
                scores_seq = scores.tolist()
            except AttributeError:
                scores_seq = scores
            scores_list = [float(s) for s in scores_seq]
            indexed_scores = sorted(
                enumerate(scores_list),
                key=lambda item: item[1],
                reverse=True,
            )
            for rank, (idx, score) in enumerate(indexed_scores[:lexical_limit], start=1):
                doc = self._bm25_docs[idx]
                doc_id = self._doc_id(doc)
                candidate = candidates.get(doc_id)
                if candidate is None:
                    candidate = HybridCandidate(doc=doc)
                    candidates[doc_id] = candidate
                candidate.lexical_rank = rank
                candidate.lexical_score = score
                candidate.rrf_score += self._weights[0] / (rank + self._rrf_constant)

        if hasattr(self._db, "similarity_search_with_relevance_scores"):
            dense_results = self._db.similarity_search_with_relevance_scores(query, k=pool)
        else:
            dense_docs = self._db.similarity_search(query, k=pool)
            dense_results = [(doc, None) for doc in dense_docs]

        for rank, (doc, score) in enumerate(dense_results, start=1):
            doc_id = self._doc_id(doc)
            candidate = candidates.get(doc_id)
            if candidate is None:
                candidate = HybridCandidate(doc=doc)
                candidates[doc_id] = candidate
            candidate.dense_rank = rank
            if score is not None:
                candidate.dense_score = float(score)
            candidate.rrf_score += self._weights[1] / (rank + self._rrf_constant)

        if not candidates:
            return []

        hit_tokens = [tok for tok in tokens if len(tok) > 2 or tok.isdigit()]
        for candidate in candidates.values():
            candidate.term_hits = self._collect_term_hits(candidate.doc, hit_tokens)

        lexical_max = max((cand.lexical_score or 0.0) for cand in candidates.values())
        dense_max = max((cand.dense_score or 0.0) for cand in candidates.values())
        hits_max = max((cand.term_hits) for cand in candidates.values())

        for cand in candidates.values():
            cand.lexical_norm = (
                (cand.lexical_score or 0.0) / lexical_max if lexical_max > 0 else 0.0
            )
            cand.dense_norm = (
                (cand.dense_score or 0.0) / dense_max if dense_max > 0 else 0.0
            )
            cand.term_norm = (
                cand.term_hits / hits_max if hits_max > 0 else 0.0
            )
            cand.coarse_score = (
                cand.rrf_score * self._rrf_scale
                + 0.5 * cand.lexical_norm
                + 0.35 * cand.dense_norm
                + self._term_weight * cand.term_norm
            )

        limit = fetch_k or pool
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda cand: (cand.coarse_score, cand.lexical_norm, cand.dense_norm),
            reverse=True,
        )
        return sorted_candidates[:limit]

    def invoke(
        self,
        query: str,
        *,
        limit: Optional[int] = None,
        fetch_k: Optional[int] = None,
    ) -> list[Document]:
        limit = max(1, int(limit or self.default_k))
        fetch = max(fetch_k or self._component_limit(None), limit)
        candidates = self.gather(query, fetch_k=fetch)
        return [candidate.doc for candidate in candidates[:limit]]

    __call__ = invoke


def docs_from_chroma(db) -> list[Document]:
    collection = getattr(db, "_collection", None)
    if collection is None:
        return []
    data = collection.get(include=["documents", "metadatas"])
    docs = []
    ids = data.get("ids", [])
    texts = data.get("documents", [])
    metas = data.get("metadatas", [])
    for doc_id, text, meta in zip(ids, texts, metas):
        info = dict(meta or {})
        info.setdefault("_id", doc_id)
        docs.append(Document(page_content=text or "", metadata=info))
    return docs


def make_hybrid_retriever(
    chunks: Sequence[Document],
    db,
    k: int = 10,
    weights: Sequence[float] = (0.6, 0.4),
    pool_multiplier: float = 4.0,
    rrf_constant: float = 60.0,
    rrf_scale: float = 20.0,
    term_weight: float = 0.3,
) -> HybridRetriever:
    return HybridRetriever(
        chunks,
        db,
        default_k=k,
        weights=weights,
        pool_multiplier=pool_multiplier,
        rrf_constant=rrf_constant,
        rrf_scale=rrf_scale,
        term_weight=term_weight,
    )


def _resolve_device(preference: str) -> str:
    if preference != "auto":
        return preference
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        device: str = "auto",
        max_length: int = 512,
        max_chars: int = 4000,
    ):
        self.device = _resolve_device(device)
        self.max_chars = max(int(max_chars), 256)
        self.model = CrossEncoder(
            model_name,
            trust_remote_code=True,
            device=self.device,
            max_length=max_length,
        )

    def _prepare_text(self, doc: Document) -> str:
        text = (doc.page_content or "").strip()
        meta = doc.metadata or {}
        prefix_parts: list[str] = []
        for key in ("section_path", "source"):
            value = meta.get(key)
            if value:
                prefix_parts.append(str(value))
        if prefix_parts:
            prefix = " | ".join(prefix_parts)
            if prefix and not text.lower().startswith(prefix.lower()):
                text = f"{prefix}\n\n{text}"
        if len(text) > self.max_chars:
            text = text[: self.max_chars]
        return text

    def rerank(
        self,
        query: str,
        docs: Sequence[HybridCandidate | Document],
        topn: int = 6,
    ):
        if not docs:
            return []
        if isinstance(docs[0], HybridCandidate):
            candidates = list(docs)  # type: ignore[arg-type]
        else:
            candidates = [HybridCandidate(doc=d) for d in docs]  # type: ignore[list-item]
        query_tokens = {tok for tok in _tokenize(query)}
        pairs = [(query, self._prepare_text(candidate.doc)) for candidate in candidates]
        if not pairs:
            return []
        scores = self.model.predict(pairs)
        raw_scores = [float(score) for score in scores]

        ce_min = min(raw_scores)
        ce_max = max(raw_scores)
        ce_range = ce_max - ce_min
        if ce_range <= 1e-6:
            ce_norms = [0.5 for _ in raw_scores]
        else:
            ce_norms = [(score - ce_min) / ce_range for score in raw_scores]

        coarse_values = [candidate.coarse_score for candidate in candidates]
        coarse_min = min(coarse_values)
        coarse_max = max(coarse_values)
        coarse_range = coarse_max - coarse_min
        if coarse_range <= 1e-6:
            coarse_norms = [0.0 for _ in coarse_values]
        else:
            coarse_norms = [
                (value - coarse_min) / coarse_range for value in coarse_values
            ]

        rrf_values = [candidate.rrf_score for candidate in candidates]
        rrf_min = min(rrf_values)
        rrf_max = max(rrf_values)
        rrf_range = rrf_max - rrf_min
        if rrf_range <= 1e-6:
            rrf_norms = [0.0 for _ in rrf_values]
        else:
            rrf_norms = [(value - rrf_min) / rrf_range for value in rrf_values]

        w_ce = 0.48
        w_coarse = 0.32
        w_lexical = 0.10
        w_dense = 0.05
        w_term = 0.05
        table_boost = 0.08
        source_boost = 0.04
        if any(token and any(ch.isdigit() for ch in token) for token in query_tokens):
            w_ce = 0.36
            w_coarse = 0.38
            w_lexical = 0.14
            w_dense = 0.04
            w_term = 0.08
            table_boost = 0.12
            source_boost = 0.06

        ranked: list[tuple[Document, float]] = []
        for idx, candidate in enumerate(candidates):
            doc = candidate.doc
            text_lower = (doc.page_content or "").lower()
            has_table = "|" in (doc.page_content or "")
            token_hit = any(token and token in text_lower for token in query_tokens)
            structure_bonus = 0.0
            if has_table and token_hit:
                structure_bonus += table_boost
            meta_source = str((doc.metadata or {}).get("source", "")).lower().replace('\\', '/')
            if token_hit and 'majors' in meta_source:
                structure_bonus += source_boost
            final_score = (
                w_ce * ce_norms[idx]
                + w_coarse * coarse_norms[idx]
                + w_lexical * candidate.lexical_norm
                + w_dense * candidate.dense_norm
                + w_term * candidate.term_norm
                + structure_bonus
            )
            info = {
                "final": float(final_score),
                "cross_encoder": raw_scores[idx],
                "cross_encoder_norm": ce_norms[idx],
                "coarse": candidate.coarse_score,
                "coarse_norm": coarse_norms[idx],
                "rrf": candidate.rrf_score,
                "rrf_norm": rrf_norms[idx],
                "lexical_rank": candidate.lexical_rank,
                "lexical_score": candidate.lexical_score,
                "lexical_norm": candidate.lexical_norm,
                "dense_rank": candidate.dense_rank,
                "dense_score": candidate.dense_score,
                "dense_norm": candidate.dense_norm,
                "term_hits": candidate.term_hits,
                "term_norm": candidate.term_norm,
                "structure_bonus": structure_bonus,
            }
            metadata = dict(doc.metadata or {})
            metadata["_rerank"] = info
            doc.metadata = metadata
            ranked.append((doc, float(final_score)))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:topn]
