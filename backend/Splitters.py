from typing import List
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from tqdm.auto import tqdm

def split_markdown(
    docs: List[Document],
    embeddings,
    show_progress: bool = True,
) -> List[Document]:
    if not docs:
        return []

    sem = SemanticChunker(embeddings)
    out: List[Document] = []

    bar = tqdm(
        total=len(docs),
        desc="Semantic Chunking",
        unit="file",
        dynamic_ncols=True,
        mininterval=0.2,
        ascii=True,
        disable=not show_progress,
    )

    for d in docs:
        out.extend(sem.create_documents([d.page_content], metadatas=[d.metadata or {}]))
        bar.update(1)

    bar.close()
    return out
