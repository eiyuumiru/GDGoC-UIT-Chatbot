from typing import List
from pathlib import Path
import shutil, sys, gc
from langchain.schema import Document
from langchain_chroma import Chroma
from tqdm.auto import tqdm

PERSIST_DIR = "backend/.index/chroma"
COLLECTION  = "uit_md_vi"

def build_index(
    chunks: List[Document],
    encoder,
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION,
    batch_size: int = 32,
    show_progress: bool = True,
) -> Chroma:
    db = Chroma(
        embedding_function=encoder,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    total = len(chunks)
    if total == 0:
        return db

    bar = tqdm(
        total=total, disable=not show_progress, desc="Embedding & Indexing",
        unit="doc", dynamic_ncols=True, mininterval=0.2, ascii=True
    )

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        db.add_documents(batch)
        try:
            import torch; torch.cuda.empty_cache()
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

def get_retriever(db: Chroma, k: int = 4):
    return db.as_retriever(search_kwargs={"k": k})

def wipe_index(persist_directory: str = PERSIST_DIR):
    p = Path(persist_directory)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
