"""
Task 10 — Generation Có Citation (+ Streaming & Confidence).

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt (mỗi chunk có nhãn [n] để LLM cite)
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "Tôi không thể xác minh thông tin này từ nguồn hiện có"

Bổ sung cho UI thân thiện:
    - generate_with_citation_stream(): stream từng token cho hiệu ứng typing
    - compute_confidence(): chấm độ tin cậy câu trả lời (Cao / Trung bình / Thấp)
    - parse_citations(): trích các nhãn [n] trong câu trả lời để highlight nguồn

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
import re
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# LLM model (OpenRouter model ID). Đổi sang model ":free" nếu chưa có credit,
# ví dụ "meta-llama/llama-3.1-8b-instruct:free".
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

# Câu trả lời chuẩn khi không đủ evidence — dùng để chấm confidence.
CANNOT_VERIFY = "Tôi không thể xác minh thông tin này từ nguồn hiện có"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = f"""Bạn là trợ lý trả lời câu hỏi về chính sách thương mại điện tử và hỗ trợ
khách hàng (thanh toán, đổi trả, giao hàng, quyền riêng tư, quy định người bán).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt.
2. Mỗi khẳng định phải có trích dẫn ngay sau bằng SỐ THỨ TỰ tài liệu, ví dụ: [1], [2].
   Chỉ dùng đúng số [n] có trong context, không tự tạo số mới.
3. Nếu context không đủ thông tin → trả lời đúng nguyên văn: "{CANNOT_VERIFY}".
4. Trả lời bằng tiếng Việt, ngắn gọn, có cấu trúc rõ ràng (gạch đầu dòng khi cần).
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context."""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)
    """
    if len(chunks) <= 2:
        return list(chunks)

    front = chunks[::2]   # index 0, 2, 4 -> đặt ở đầu (giữ chunk tốt nhất ở đầu)
    back = chunks[1::2]   # index 1, 3    -> đặt ở cuối (đảo ngược)
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.

    Mỗi chunk có nhãn [n] (theo THỨ TỰ NGUỒN GỐC trong `sources`, không theo thứ tự
    reorder) để citation [n] mà LLM sinh ra khớp với source card số n trên UI.

    Args:
        chunks: List of {'content', 'metadata', 'score', 'citation_id'?}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        # citation_id được gán trước reorder để giữ mapping ổn định với sources UI
        cid = chunk.get("citation_id", i)
        meta = chunk.get("metadata", {}) or {}
        source = meta.get("source", f"Nguồn {cid}")
        doc_type = meta.get("type", "unknown")
        context_parts.append(
            f"[{cid}] (Nguồn: {source} | Loại: {doc_type})\n"
            f"{chunk.get('content', '').strip()}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# CITATIONS & CONFIDENCE
# =============================================================================

def parse_citations(answer: str) -> list[int]:
    """Trích các nhãn citation [n] xuất hiện trong câu trả lời (unique, sorted)."""
    found = {int(m) for m in re.findall(r"\[(\d+)\]", answer or "")}
    return sorted(found)


def compute_confidence(answer: str, sources: list[dict], retrieval_source: str) -> dict:
    """
    Chấm độ tin cậy của câu trả lời dựa trên nhiều tín hiệu.

    Tín hiệu:
        - Điểm retrieval cao nhất (evidence có liên quan không?)
        - Có trích dẫn [n] hợp lệ không (câu trả lời có bám nguồn không?)
        - Câu trả lời có phải "không xác minh được" không
        - Nguồn là 'hybrid' hay 'pageindex' fallback

    Returns:
        {
            'score': float 0..1,
            'level': 'high' | 'medium' | 'low',
            'label': 'Cao' | 'Trung bình' | 'Thấp',
            'reasons': list[str],
        }
    """
    reasons: list[str] = []

    # Trường hợp LLM tự nhận không đủ dữ liệu → confidence thấp ngay
    if answer and CANNOT_VERIFY.lower() in answer.lower():
        return {
            "score": 0.15,
            "level": "low",
            "label": "Thấp",
            "reasons": ["Mô hình không tìm thấy đủ bằng chứng trong nguồn."],
        }

    if not sources:
        return {
            "score": 0.1,
            "level": "low",
            "label": "Thấp",
            "reasons": ["Không có tài liệu nào được truy xuất."],
        }

    # 1) Tín hiệu điểm retrieval (clamp 0..1)
    scores = [float(s.get("score", 0) or 0) for s in sources]
    max_score = max(scores) if scores else 0.0
    retrieval_signal = max(0.0, min(1.0, max_score))
    if retrieval_signal >= 0.6:
        reasons.append("Tài liệu truy xuất bám sát câu hỏi.")
    elif retrieval_signal < 0.35:
        reasons.append("Độ liên quan của tài liệu ở mức trung bình.")

    # 2) Tín hiệu độ phủ citation
    cited = parse_citations(answer)
    valid_cited = [c for c in cited if 1 <= c <= len(sources)]
    if valid_cited:
        coverage = min(1.0, len(valid_cited) / min(3, len(sources)))
        reasons.append(f"Câu trả lời trích dẫn {len(valid_cited)} nguồn.")
    else:
        coverage = 0.0
        reasons.append("Câu trả lời chưa gắn trích dẫn rõ ràng.")

    # 3) Phạt nếu phải fallback sang PageIndex
    fallback_penalty = 0.15 if retrieval_source == "pageindex" else 0.0
    if fallback_penalty:
        reasons.append("Dùng cơ chế fallback (PageIndex) thay vì hybrid search.")

    score = 0.6 * retrieval_signal + 0.4 * coverage - fallback_penalty
    score = max(0.0, min(1.0, score))

    if score >= 0.66:
        level, label = "high", "Cao"
    elif score >= 0.4:
        level, label = "medium", "Trung bình"
    else:
        level, label = "low", "Thấp"

    return {"score": round(score, 3), "level": level, "label": label, "reasons": reasons}


# =============================================================================
# LLM CLIENT — multi-provider (OpenRouter → OpenAI)
# =============================================================================

def _get_client_and_model():
    """
    Trả về (client, model_id) theo API key sẵn có.

    Ưu tiên OpenRouter (nhiều model :free), fallback sang OpenAI thuần.
    """
    from openai import OpenAI

    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key and or_key.strip() and not or_key.startswith("sk-or-v1-..."):
        client = OpenAI(api_key=or_key, base_url="https://openrouter.ai/api/v1")
        return client, LLM_MODEL

    oa_key = os.getenv("OPENAI_API_KEY")
    if oa_key and oa_key.strip():
        client = OpenAI(api_key=oa_key)
        # Nếu LLM_MODEL có tiền tố provider "openai/", cắt bỏ cho API OpenAI thuần
        model = LLM_MODEL.split("/", 1)[-1] if LLM_MODEL.startswith("openai/") else "gpt-4o-mini"
        return client, model

    raise RuntimeError(
        "Chưa cấu hình API key. Đặt OPENROUTER_API_KEY hoặc OPENAI_API_KEY trong .env"
    )


def _build_messages(query: str, sources: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Chuẩn bị messages cho LLM từ danh sách sources đã retrieve.

    Returns:
        (messages, sources) — sources đã gán citation_id ổn định.
    """
    # Gán citation_id theo thứ tự retrieve GỐC để [n] khớp source card số n trên UI
    for i, s in enumerate(sources, 1):
        s.setdefault("citation_id", i)

    reordered = reorder_for_llm(sources)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nCâu hỏi: {query}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    return messages, sources


# =============================================================================
# GENERATION — non-streaming
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation (non-streaming).

    Returns:
        {
            'answer': str,
            'sources': list[dict],       # đã có 'citation_id'
            'retrieval_source': str,     # 'hybrid' | 'pageindex' | 'none'
            'citations': list[int],      # các [n] xuất hiện trong answer
            'confidence': dict,          # {score, level, label, reasons}
        }
    """
    chunks = retrieve(query, top_k=top_k)
    retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"

    if not chunks:
        return {
            "answer": CANNOT_VERIFY + ".",
            "sources": [],
            "retrieval_source": "none",
            "citations": [],
            "confidence": compute_confidence(CANNOT_VERIFY, [], "none"),
        }

    messages, sources = _build_messages(query, chunks)
    client, model = _get_client_and_model()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    answer = response.choices[0].message.content or ""

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
        "citations": parse_citations(answer),
        "confidence": compute_confidence(answer, sources, retrieval_source),
    }


# =============================================================================
# GENERATION — streaming (cho hiệu ứng typing trên UI)
# =============================================================================

def prepare_generation(query: str, top_k: int = TOP_K) -> dict:
    """
    Bước chuẩn bị (retrieve + build prompt) tách riêng để UI có thể:
        1. Hiển thị "đang truy xuất..." trong lúc gọi hàm này
        2. Sau đó stream câu trả lời qua stream_answer()

    Returns:
        {'sources', 'retrieval_source', 'messages'}
    """
    chunks = retrieve(query, top_k=top_k)
    retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"
    if not chunks:
        return {"sources": [], "retrieval_source": "none", "messages": None}

    messages, sources = _build_messages(query, chunks)
    return {"sources": sources, "retrieval_source": retrieval_source, "messages": messages}


def stream_answer(messages: list[dict]):
    """
    Generator yield từng đoạn text từ LLM (tương thích st.write_stream).

    Dùng cùng prepare_generation():
        prep = prepare_generation(query)
        answer = st.write_stream(stream_answer(prep["messages"]))
    """
    if not messages:
        yield CANNOT_VERIFY + "."
        return

    client, model = _get_client_and_model()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


def generate_with_citation_stream(query: str, top_k: int = TOP_K):
    """
    Phiên bản streaming trọn gói cho CLI/test: yield text, kết thúc trả về dict metadata.

    Cách dùng:
        gen = generate_with_citation_stream(q)
        for token in gen:
            print(token, end="")
        meta = gen.value  # không có sẵn với for-loop; dùng prepare + stream trên UI
    """
    prep = prepare_generation(query, top_k=top_k)
    answer_parts = []
    for token in stream_answer(prep["messages"]):
        answer_parts.append(token)
        yield token

    answer = "".join(answer_parts)
    # Trả metadata cuối qua StopIteration.value
    return {
        "answer": answer,
        "sources": prep["sources"],
        "retrieval_source": prep["retrieval_source"],
        "citations": parse_citations(answer),
        "confidence": compute_confidence(answer, prep["sources"], prep["retrieval_source"]),
    }


if __name__ == "__main__":
    test_queries = [
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Làm sao để yêu cầu đổi trả hay hoàn tiền?",
        "Cần chuẩn bị bằng chứng gì khi yêu cầu hoàn tiền?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        conf = result["confidence"]
        print(f"\n[Confidence: {conf['label']} ({conf['score']}) | "
              f"Sources: {len(result['sources'])} | via {result['retrieval_source']}]")
