from __future__ import annotations
from typing import List
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "AITeamVN/Vietnamese_Embedding_v2"

class STEncoder(Embeddings):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        batch_size: int = 64,
        normalize: bool = True,
        max_seq_length: int = 2048,
        show_tqdm: bool = True,
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.encode_kwargs = {
            "batch_size": batch_size,
            "normalize_embeddings": normalize,
            "show_progress_bar": show_tqdm,
            "convert_to_numpy": True,
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embs = self.model.encode(texts, **self.encode_kwargs)
        return embs.tolist() if hasattr(embs, "tolist") else embs

    def embed_query(self, text: str) -> List[float]:
        embs = self.model.encode([text], **self.encode_kwargs)
        v = embs[0]
        return v.tolist() if hasattr(v, "tolist") else v

def get_encoder(
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    device: str = "cuda",
    show_tqdm: bool = True,
) -> Embeddings:
    return STEncoder(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        normalize=True,
        max_seq_length=2048,
        show_tqdm=show_tqdm,
    )
