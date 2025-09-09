from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_MODEL = "BAAI/bge-m3"

@lru_cache(maxsize=4)
def get_encoder(
    model_name: str = DEFAULT_MODEL,
    normalize_embeddings: bool = True,
    batch_size: int = 64,
    device: str = "cuda"
):

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": device,
            "trust_remote_code": False,
        },
        encode_kwargs={
            "normalize_embeddings": normalize_embeddings,
            "batch_size": batch_size,
            "device": device,
        },
    )
