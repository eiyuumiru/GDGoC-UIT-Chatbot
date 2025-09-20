from typing import List
from langchain.schema import Document
from langchain_chroma import Chroma
from tqdm import tqdm
import gc

PERSIST_DIR = "backend/.index/chroma"
COLLECTION  = "uit_edu"

def build_index(
    chunks: List[Document],
    encoder,
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION,
    batch_size: int = 64,
    show_progress: bool = True,
    clear_existing: bool = True,
) -> Chroma:
    db = Chroma(
        embedding_function=encoder,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    if clear_existing:
        try:
            db.delete_collection()
        except Exception:
            pass
        db = Chroma(
            embedding_function=encoder,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    total = len(chunks)
    if total == 0:
        return db

    bar = tqdm(
        total=total,
        desc="Embedding & Indexing",
        unit="doc",
        dynamic_ncols=True,
        mininterval=0.2,
        ascii=True,
        disable=not show_progress
    )

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        db.add_documents(batch)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        bar.update(len(batch))

    bar.close()
    return db

def load_index(
    encoder,
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION,
) -> Chroma:
    return Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=encoder,
    )
    