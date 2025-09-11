from typing import List, Tuple
from langchain.schema import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

def split_markdown(
    docs: List[Document],
    encoder,
    headers: List[Tuple[str, str]] = (("#","h1"), ("##","h2"), ("###","h3"), ("####","h4")), # type: ignore
    percentile: int = 92,
    enforce_max: int = 850,
    overlap: int = 120,
    show_progress: bool = True,
) -> List[Document]:
    header_split = MarkdownHeaderTextSplitter(headers_to_split_on=list(headers), strip_headers=False)
    sections: List[Document] = []
    for d in docs:
        parts = header_split.split_text(d.page_content)
        for p in parts:
            meta = dict(d.metadata); meta.update(p.metadata or {})
            p.metadata = meta
        sections.extend(parts)

    sem = SemanticChunker(
        encoder,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=percentile,
    )
    chunks: List[Document] = []
    bar = tqdm(
        total=len(sections),
        desc="Semantic Chunking",
        unit="sec",
        dynamic_ncols=True,
        mininterval=0.2,
        ascii=True,
        disable=not show_progress,
    )
    for sec in sections:
        pieces = sem.split_text(sec.page_content)
        for t in pieces:
            chunks.append(Document(page_content=t, metadata=sec.metadata))
        bar.update(1)
    bar.close()

    if enforce_max:
        rc = RecursiveCharacterTextSplitter(chunk_size=enforce_max, chunk_overlap=overlap)
        final_chunks: List[Document] = []
        bar2 = tqdm(
            total=len(chunks),
            desc=f"Max-size pass (≤{enforce_max})",
            unit="chunk",
            dynamic_ncols=True,
            mininterval=0.2,
            ascii=True,
            disable=not show_progress,
        )
        for doc in chunks:
            parts = rc.split_text(doc.page_content)
            for t in parts:
                final_chunks.append(Document(page_content=t, metadata=doc.metadata))
            bar2.update(1)
        bar2.close()
        return final_chunks

    return chunks
