"""
RAG Chatbot — E-commerce Support
Streamlit UI kết nối Retrieval (Task 9) + Generation có Citation (Task 10).

Tính năng UI/UX:
    • Streaming câu trả lời (hiệu ứng typing)
    • Thanh Confidence (Cao / Trung bình / Thấp) kèm lý do
    • Source cards có highlight nguồn được trích dẫn [n]
    • Câu hỏi gợi ý, chỉnh top_k, xoá hội thoại

Chạy:
    streamlit run app.py
"""

import os
import re
import sys
import html
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="E-commerce Support RAG Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# STYLES — giao diện thân thiện, có dark mode
# =============================================================================

st.markdown(
    """
    <style>
      /* Confidence badge */
      .conf-badge {
          display:inline-flex; align-items:center; gap:6px;
          padding:4px 12px; border-radius:999px;
          font-size:0.82rem; font-weight:600; line-height:1;
          border:1px solid transparent;
      }
      .conf-high   { background:#e7f6ec; color:#1a7f37; border-color:#a6e0b8; }
      .conf-medium { background:#fff4e0; color:#9a6700; border-color:#f2cf8b; }
      .conf-low    { background:#fdecec; color:#b42318; border-color:#f3b4ad; }

      .conf-track { height:8px; border-radius:999px; background:#e6e8eb;
                    overflow:hidden; margin:8px 0 4px; }
      .conf-fill  { height:100%; border-radius:999px; }
      .fill-high  { background:linear-gradient(90deg,#1a7f37,#2ea043); }
      .fill-medium{ background:linear-gradient(90deg,#bf8700,#e0a800); }
      .fill-low   { background:linear-gradient(90deg,#b42318,#e5533d); }

      /* Source card */
      .src-card {
          border:1px solid rgba(128,128,128,0.25); border-radius:12px;
          padding:12px 14px; margin-bottom:10px;
          background:rgba(128,128,128,0.04);
      }
      .src-card.cited { border-color:#2ea043; background:rgba(46,160,67,0.06); }
      .src-head { display:flex; align-items:center; gap:8px; flex-wrap:wrap;
                  margin-bottom:6px; font-weight:600; }
      .src-num  { display:inline-flex; align-items:center; justify-content:center;
                  min-width:22px; height:22px; border-radius:6px; font-size:0.78rem;
                  background:#57606a; color:#fff; }
      .src-card.cited .src-num { background:#2ea043; }
      .chip { font-size:0.72rem; padding:2px 8px; border-radius:999px;
              background:rgba(128,128,128,0.18); font-weight:500; }
      .chip.cited { background:#2ea043; color:#fff; }
      .chip.hybrid { background:#1f6feb; color:#fff; }
      .chip.pageindex { background:#8250df; color:#fff; }
      .src-body { font-size:0.86rem; opacity:0.85; line-height:1.5;
                  max-height:120px; overflow:auto; }
      .conf-reasons { font-size:0.8rem; opacity:0.8; margin-top:2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# HELPERS
# =============================================================================

def render_confidence(conf: dict):
    """Render thanh confidence + badge + lý do."""
    if not conf:
        return
    level = conf.get("level", "low")
    label = conf.get("label", "Thấp")
    score = float(conf.get("score", 0) or 0)
    icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(level, "🔴")
    pct = int(round(score * 100))

    st.markdown(
        f"""
        <div>
          <span class="conf-badge conf-{level}">{icon} Độ tin cậy: {label} · {pct}%</span>
          <div class="conf-track"><div class="conf-fill fill-{level}" style="width:{pct}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    reasons = conf.get("reasons") or []
    if reasons:
        st.markdown(
            "<div class='conf-reasons'>• " + "<br>• ".join(html.escape(r) for r in reasons) + "</div>",
            unsafe_allow_html=True,
        )


def render_sources(sources: list, cited: list, retrieval_source: str):
    """Render các source card, highlight nguồn đã được trích dẫn."""
    if not sources:
        return
    cited_set = set(cited or [])
    n_cited = len(cited_set & {s.get("citation_id", i) for i, s in enumerate(sources, 1)})
    origin = retrieval_source or "hybrid"

    with st.expander(f"📚 Nguồn tham khảo · {len(sources)} tài liệu · {n_cited} được trích dẫn", expanded=False):
        for i, src in enumerate(sources, 1):
            cid = src.get("citation_id", i)
            meta = src.get("metadata", {}) or {}
            source_name = html.escape(str(meta.get("source", "Không rõ")))
            doc_type = html.escape(str(meta.get("type", "unknown")))
            score = float(src.get("score", 0) or 0)
            body = html.escape((src.get("content", "") or "")[:400]).replace("\n", " ")
            is_cited = cid in cited_set

            origin_chip = (
                f"<span class='chip {origin}'>{'🧭 PageIndex' if origin=='pageindex' else '🔀 Hybrid'}</span>"
            )
            cited_chip = "<span class='chip cited'>✓ đã trích dẫn</span>" if is_cited else ""

            st.markdown(
                f"""
                <div class="src-card {'cited' if is_cited else ''}">
                  <div class="src-head">
                    <span class="src-num">{cid}</span>
                    <span>{source_name}</span>
                    <span class="chip">{doc_type}</span>
                    <span class="chip">score {score:.3f}</span>
                    {origin_chip}{cited_chip}
                  </div>
                  <div class="src-body">{body}…</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def highlight_citations(answer: str) -> str:
    """Bọc các nhãn [n] thành mã inline để dễ thấy trong câu trả lời."""
    return re.sub(r"\[(\d+)\]", r"`[\1]`", answer or "")

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("🛒 E-commerce Support RAG")
    st.caption("Trợ lý hỏi đáp chính sách TMĐT & hỗ trợ khách hàng (đổi trả, thanh toán, bảo mật, người bán)")

    st.divider()
    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?",
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Làm sao để đổi phương thức thanh toán đơn hàng?",
        "Quy định về đăng bán sản phẩm cho người bán?",
        "Cách mua hàng trên Shopee của quốc gia khác?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:20]}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)

    if st.button("🗑️ Xoá hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("**Kiến trúc:** Hybrid Retrieval (Semantic + BM25) → RRF Rerank → PageIndex Fallback → LLM Generation có Citation")

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# MAIN
# =============================================================================

st.title("🛒 E-commerce Support RAG Chatbot")
st.caption("Hỏi đáp chính sách e-commerce với trích dẫn nguồn và độ tin cậy minh bạch")

# Lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(highlight_citations(msg["content"]))
            if msg.get("confidence"):
                render_confidence(msg["confidence"])
            render_sources(msg.get("sources", []), msg.get("citations", []), msg.get("retrieval_source", "hybrid"))
        else:
            st.markdown(msg["content"])

# =============================================================================
# QUERY HANDLING
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi của bạn về chính sách/hỗ trợ e-commerce...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        answer, sources, retrieval_source, citations, confidence = "", [], "none", [], None
        try:
            from src.task10_generation import (
                prepare_generation, stream_answer, parse_citations, compute_confidence,
            )

            with st.status("🔎 Đang truy xuất tài liệu liên quan...", expanded=False) as status:
                prep = prepare_generation(query, top_k=top_k)
                sources = prep["sources"]
                retrieval_source = prep["retrieval_source"]
                if sources:
                    status.update(label=f"✅ Tìm thấy {len(sources)} tài liệu · đang soạn câu trả lời...", state="running")
                else:
                    status.update(label="⚠️ Không tìm thấy tài liệu phù hợp", state="complete")

            # Streaming câu trả lời (hiệu ứng typing)
            answer = st.write_stream(stream_answer(prep["messages"]))
            citations = parse_citations(answer)
            confidence = compute_confidence(answer, sources, retrieval_source)

            render_confidence(confidence)
            render_sources(sources, citations, retrieval_source)

        except NotImplementedError:
            answer = ("⚠️ **Pipeline chưa hoàn tất.** Hãy hoàn thành các Task 4–8 (chunking, "
                      "semantic/lexical search, rerank, PageIndex) để `retrieve()` và `generate_with_citation()` "
                      "chạy end-to-end.")
            st.markdown(answer)
        except Exception as e:
            answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "citations": citations,
        "retrieval_source": retrieval_source,
        "confidence": confidence,
    })
