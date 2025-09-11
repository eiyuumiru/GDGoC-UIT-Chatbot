from __future__ import annotations
from typing import List
from langchain.embeddings.base import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings 

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
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={
                "batch_size": batch_size,
                "normalize_embeddings": normalize,
                "convert_to_numpy": True,
            },
        )
        self.max_seq_length = max_seq_length

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)

def get_encoder(
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    device: str = "cuda",
    show_tqdm: bool = True,
) -> Embeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={
            "batch_size": batch_size,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        },
    )