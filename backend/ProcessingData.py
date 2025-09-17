from typing import Any, Dict, List, Optional
import re

def build_header_path(meta: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("h1", "h2", "h3", "h4"):
        value = meta.get(key)
        if value:
            parts.append(str(value).strip())
    return " > ".join(parts)


def clean_cell(text: str) -> str:
    cleaned = re.sub(r"`([^`]+)`", r"\1", text or "")
    cleaned = cleaned.replace("**", "").replace("__", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def parse_table_block(lines: List[str]) -> Optional[List[str]]:
    rows: List[List[str]] = []
    for line in lines:
        if "|" not in line:
            continue
        cells = [clean_cell(cell) for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    if len(rows) < 2:
        return None
    header, *body = rows
    if body and all(re.fullmatch(r"[:\- ]*", cell or "") for cell in body[0]):
        body = body[1:]
    width = len(header)
    header = [cell if cell else f"col_{idx + 1}" for idx, cell in enumerate(header)]
    records: List[str] = []
    for row in body:
        if len(row) < width:
            row = row + [""] * (width - len(row))
        if not any(cell.strip() for cell in row):
            continue
        filled = [row[idx].strip() for idx in range(width) if row[idx].strip()]
        if not filled:
            continue
        if len(filled) == 1:
            records.append(filled[0])
            continue
        pairs = [f"{header[idx]}: {row[idx].strip()}" for idx in range(width) if row[idx].strip()]
        if pairs:
            records.append(" ; ".join(pairs))
    return records or None


def normalize_structured_text(text: str) -> str:
    lines = text.splitlines()
    buffer: List[str] = []
    table: List[str] = []

    def flush() -> None:
        nonlocal table
        if not table:
            return
        parsed = parse_table_block(table)
        if parsed:
            buffer.extend(parsed)
        else:
            buffer.extend(table)
        table = []

    for line in lines:
        if "|" in line and not line.strip().startswith("<"):
            table.append(line)
            continue
        flush()
        buffer.append(line)
    flush()
    return "\n".join(buffer)


__all__ = [
    "build_header_path",
    "clean_cell",
    "parse_table_block",
    "normalize_structured_text",
]
