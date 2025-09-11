from __future__ import annotations
from typing import List
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

def docs_from_chroma(db):
    got = db._collection.get(include=["documents", "metadatas"])
    ids   = got.get("ids", [])
    texts = got.get("documents", []) or []
    metas = got.get("metadatas", []) or []

    docs = []
    for i, t, m in zip(ids, texts, metas):
        m = m or {}
        m.setdefault("_id", i)
        docs.append(Document(page_content=t or "", metadata=m))
    return docs

def make_hybrid_retriever(chunks: List[Document], db, k: int = 10, weights=(0.6, 0.4)):
    bm25 = BM25Retriever.from_documents(chunks); bm25.k = k
    dense = db.as_retriever(search_kwargs={"k": k})
    return EnsembleRetriever(retrievers=[bm25, dense], weights=list(weights))

class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        device: str | None = None,
        max_length: int = 512,
    ):
        self.model = CrossEncoder(
            model_name,
            trust_remote_code=True,
            device="cuda",
            max_length=max_length,
        )

    def rerank(self, query: str, docs, topn: int = 6):
        pairs = [(query, d.page_content[:4000]) for d in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        return ranked[:topn]