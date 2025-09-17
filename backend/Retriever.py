import logging
from collections.abc import Sequence

from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


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
):
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k
    dense = db.as_retriever(search_kwargs={"k": k})
    return EnsembleRetriever(retrievers=[bm25, dense], weights=list(weights))


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        device: str = "cuda",
        max_length: int = 512,
    ):
        self.model = CrossEncoder(
            model_name,
            trust_remote_code=True,
            device=device,
            max_length=max_length,
        )

    def rerank(self, query: str, docs: Sequence[Document], topn: int = 6):
        pairs = [(query, doc.page_content[:4000]) for doc in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda item: float(item[1]), reverse=True)
        return ranked[:topn]
