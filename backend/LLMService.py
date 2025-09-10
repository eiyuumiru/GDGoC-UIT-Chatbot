from __future__ import annotations
from typing import Callable, List, Dict, Any, Optional
import os
from langchain_groq import ChatGroq
from .FullChain import ask_question, _ensure_loaded

SYSTEM_INSTRUCTIONS = (
    "Bạn là trợ lý RAG cho CTĐT UIT (Khóa 2025). Chỉ trả lời dựa trên NGỮ CẢNH được cung cấp; nếu thiếu hãy nói rõ 'không có trong dữ liệu'. Trả lời ngắn gọn, chính xác, tiếng Việt. Khi có số liệu hoặc quy định, nêu rõ. Cuối cùng phải có mục 'Nguồn:' liệt kê các nguồn đã dùng theo thứ tự xuất hiện."
)

def get_groq_llm(
    groq_api_key: Optional[str] = None,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> ChatGroq:
    key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Thiếu GROQ_API_KEY")
    return ChatGroq(
        groq_api_key=key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

def _format_context(chunks: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    buf, total = [], 0
    for i, r in enumerate(chunks, 1):
        src = r.get("source") or ""
        content = (r.get("content") or "").strip()
        piece = f"<CTX id=\"{i}\" source=\"{src}\">\n{content}\n</CTX>"
        if total + len(piece) > max_chars:
            break
        buf.append(piece)
        total += len(piece)
    return "\n".join(buf)

def _few_shot_examples() -> str:
    ex_ctx = "<CTX id=\"1\" source=\"backend/dataset/majors/cs.md\">Môn CS311 học kỳ 6; tổng số tín chỉ bắt buộc là 3.</CTX>\n<CTX id=\"2\" source=\"backend/dataset/policies/graduation.md\">Điều kiện tốt nghiệp: tích lũy đủ tín chỉ, hoàn thành chuẩn đầu ra.</CTX>"
    ex_user = "CS311 học ở kỳ mấy và có bao nhiêu tín chỉ?"
    ex_out = "CS311 học ở kỳ 6, 3 tín chỉ.\n\nNguồn: [1]"
    ex2_ctx = "<CTX id=\"1\" source=\"backend/dataset/majors/is.md\">Ngành Hệ thống thông tin: tối thiểu 125 tín chỉ.</CTX>"
    ex2_user = "Số tín chỉ tối thiểu ngành Hệ thống thông tin?"
    ex2_out = "Tối thiểu 125 tín chỉ.\n\nNguồn: [1]"
    return f"<EXAMPLES>\n<EXAMPLE>\n<CONTEXTS>\n{ex_ctx}\n</CONTEXTS>\n<QUESTION>{ex_user}</QUESTION>\n<IDEAL_ANSWER>{ex_out}</IDEAL_ANSWER>\n</EXAMPLE>\n<EXAMPLE>\n<CONTEXTS>\n{ex2_ctx}\n</CONTEXTS>\n<QUESTION>{ex2_user}</QUESTION>\n<IDEAL_ANSWER>{ex2_out}</IDEAL_ANSWER>\n</EXAMPLE>\n</EXAMPLES>"

def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rules = (
        "Quy tắc:\n"
        "1) Chỉ dùng thông tin trong CONTEXTS.\n"
        "2) Nếu thiếu dữ liệu cần thiết, trả lời: 'không có trong dữ liệu'.\n"
        "3) Trích nguồn bằng chỉ số [i] theo CTX id đã cho.\n"
        "4) Ưu tiên liệt kê gọn, rõ.\n"
        "5) Đầu ra gồm hai phần:\n"
        "   - Trả lời: phần nội dung chính\n"
        "   - Nguồn: danh sách chỉ số [i] theo thứ tự sử dụng"
    )
    ctx_block = _format_context(contexts)
    fewshot = _few_shot_examples()
    user = (
        f"<TASK>\n{rules}\n</TASK>\n"
        f"<CONTEXTS>\n{ctx_block}\n</CONTEXTS>\n"
        f"<QUESTION>\n{question}\n</QUESTION>\n"
        f"{fewshot}\n"
        f"<OUTPUT_FORMAT>\nTrả lời...\n\nNguồn: [i, ...]\n</OUTPUT_FORMAT>"
    )
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user},
    ]

def generate_with_context(
    llm: ChatGroq,
    question: str,
    contexts: List[Dict[str, Any]],
    *,
    max_ctx_chars: int = 8000,
) -> str:
    messages = build_prompt(question, contexts)
    out = llm.invoke(messages)
    return getattr(out, "content", str(out))

def create_rag_chain(
    groq_api_key: Optional[str],
    *,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    k: int = 6,
    max_ctx_chars: int = 8000,
) -> Callable[[str], Dict[str, Any]]:
    _ensure_loaded()
    llm = get_groq_llm(groq_api_key, model=model, temperature=temperature)
    def qa(question: str, topk: Optional[int] = None) -> Dict[str, Any]:
        _k = int(topk or k)
        ret = ask_question(question, k=_k)
        contexts = ret.get("results", [])
        if not contexts:
            return {"text": "Không tìm thấy thông tin phù hợp trong dữ liệu.", "sources": [], "raw": ret}
        text = generate_with_context(llm, question, contexts, max_ctx_chars=max_ctx_chars)
        seen, srcs = set(), []
        for r in contexts:
            s = r.get("source") or ""
            if s and s not in seen:
                seen.add(s)
                srcs.append(s)
        return {"text": text, "sources": srcs, "raw": ret}
    return qa
