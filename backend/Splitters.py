from typing import Any, Dict, List, Tuple
from langchain.schema import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from .ProcessingData import build_header_path, normalize_structured_text


def _merge_segments(pieces: List[str], min_chars: int, soft_chars: int) -> List[str]:
    merged: List[str] = []
    buffer: List[str] = []
    size = 0
    for piece in pieces:
        text = piece.strip()
        if not text:
            continue
        length = len(text)
        if not buffer:
            buffer = [text]
            size = length
            continue
        if size < min_chars or length < soft_chars:
            buffer.append(text)
            size += length
            continue
        merged.append(" ".join(buffer))
        buffer = [text]
        size = length
    if buffer:
        tail = " ".join(buffer)
        if merged and len(tail) < soft_chars and len(merged[-1]) < min_chars:
            merged[-1] = merged[-1] + " " + tail
        else:
            merged.append(tail)
    return merged


def split_markdown(
    docs: List[Document],
    encoder,
    headers: List[Tuple[str, str]] = (("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")),
    percentile: int = 92,
    enforce_max: int = 850,
    overlap: int = 120,
    show_progress: bool = True,
    min_chunk_chars: int = 320,
    soft_merge_chars: int = 160,
    prefix_headers: bool = True,
) -> List[Document]:
    header_split = MarkdownHeaderTextSplitter(headers_to_split_on=list(headers), strip_headers=False)
    sections: List[Document] = []
    for d in docs:
        parts = header_split.split_text(d.page_content)
        for p in parts:
            meta = dict(d.metadata)
            meta.update(p.metadata or {})
            p.metadata = meta
            p.page_content = normalize_structured_text(p.page_content)
        sections.extend(parts)

    sem = SemanticChunker(
        encoder,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=percentile,
    )
    records: List[Tuple[str, Dict[str, Any], str]] = []
    bar = tqdm(
        total=len(sections),
        desc="Semantic Chunking",
        unit="sec",
        dynamic_ncols=True,
        mininterval=0.2,
        ascii=True,
        disable=not show_progress,
    )
    index = 0
    for sec in sections:
        header_path = build_header_path(sec.metadata or {})
        pieces = sem.split_text(sec.page_content)
        merged = _merge_segments(pieces, min_chunk_chars, soft_merge_chars)
        for body in merged:
            text = body.strip()
            if not text:
                continue
            meta = dict(sec.metadata or {})
            meta["section_path"] = header_path
            meta["chunk_base"] = index
            meta["chunk_size"] = len(text)
            records.append((text, meta, header_path))
            index += 1
        bar.update(1)
    bar.close()

    chunks: List[Document] = []
    if enforce_max:
        rc = RecursiveCharacterTextSplitter(chunk_size=enforce_max, chunk_overlap=overlap)
        bar2 = tqdm(
            total=len(records),
            desc=f"Max-size pass ({enforce_max})",
            unit="chunk",
            dynamic_ncols=True,
            mininterval=0.2,
            ascii=True,
            disable=not show_progress,
        )
        for content, meta, header_path in records:
            base_text = f"{header_path}\n\n{content}" if prefix_headers and header_path else content
            parts = rc.split_text(base_text)
            for idx, part in enumerate(parts):
                text = part.strip()
                if not text:
                    continue
                if prefix_headers and header_path and idx > 0 and not text.startswith(header_path):
                    text = f"{header_path}\n\n{text}"
                meta_part = dict(meta)
                meta_part["chunk_index"] = f"{meta['chunk_base']}-{idx}"
                meta_part["chunk_part"] = idx
                meta_part["chunk_size"] = len(text)
                chunks.append(Document(page_content=text, metadata=meta_part))
            bar2.update(1)
        bar2.close()
        return chunks

    for content, meta, header_path in records:
        text = f"{header_path}\n\n{content}" if prefix_headers and header_path else content
        meta_part = dict(meta)
        meta_part["chunk_index"] = meta["chunk_base"]
        chunks.append(Document(page_content=text, metadata=meta_part))
    return chunks
