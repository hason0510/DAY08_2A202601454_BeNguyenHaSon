"""
RAG Chatbot — E-commerce Support
Streamlit UI kết nối Retrieval (Task 9) + Generation có Citation (Task 10).

Phong cách giao diện:
    • Chủ đề sáng, modern, thân thiện — nhiều khoảng trắng, bo góc mềm
    • Gradient nhấn indigo → violet → hồng, bề mặt xám xanh nhạt
    • Lời chào gradient + thẻ gợi ý bấm được · ô nhập dạng viên thuốc

Tính năng:
    • Streaming câu trả lời + typing indicator ba chấm
    • Confidence bar + lý do · Source cards highlight trích dẫn [n]
    • Hành động: tạo lại · phản hồi 👍/👎 · sao chép · dấu thời gian

Chạy:
    streamlit run app.py
"""

import re
import sys
import html
from datetime import datetime
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
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

USER_AVATAR = "🧑"
BOT_AVATAR = "🤖"

# =============================================================================
# STYLES — chủ đề sáng, modern, thân thiện
# =============================================================================

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

      /* ---- Design tokens (light only) ---- */
      :root {
          --p1:#6366f1; --p2:#8b5cf6; --p3:#ec4899;                 /* gradient stops */
          --grad:linear-gradient(120deg, var(--p1) 0%, var(--p2) 55%, var(--p3) 100%);
          --bg:#f6f8fc; --surface:#ffffff; --surface-2:#eef1f8;
          --text:#1f2937; --muted:#6b7280; --border:#e6e9f0;
          --ok:#16a34a; --warn:#d97706; --bad:#dc2626;
          --radius:18px;
          --shadow:0 8px 24px -12px rgba(79,70,229,0.28);
      }

      html, body, [class*="css"] { font-family:'Inter',system-ui,sans-serif; }
      .stApp { background:var(--bg); color:var(--text); }
      #MainMenu, footer, header[data-testid="stHeader"] { visibility:hidden; height:0; }
      .block-container { padding-top:1rem; padding-bottom:7rem; max-width:920px; }

      /* ---- Top brand bar ---- */
      .topbar { display:flex; align-items:center; gap:10px; margin:2px 0 6px; }
      .topbar .mark {
          width:38px; height:38px; border-radius:12px; display:inline-flex;
          align-items:center; justify-content:center; font-size:1.2rem; color:#fff;
          background:var(--grad); box-shadow:var(--shadow); }
      .topbar .name { font-size:1.1rem; font-weight:800; color:var(--text); letter-spacing:-0.3px; }
      .topbar .tag { font-size:0.72rem; font-weight:700; color:var(--p1);
          background:rgba(99,102,241,0.1); padding:4px 11px; border-radius:999px; margin-left:auto; }

      /* ---- Greeting (welcome) ---- */
      .greet { font-weight:800; font-size:2.5rem; line-height:1.15; letter-spacing:-1px;
          margin:20px 0 4px;
          background:var(--grad); -webkit-background-clip:text; background-clip:text; color:transparent; }
      .greet-sub { color:var(--muted); font-size:1.05rem; margin-bottom:22px; }

      /* ---- Suggestion cards (main-area buttons) ---- */
      .block-container .stButton>button {
          background:var(--surface); color:var(--text);
          border:1px solid var(--border); border-radius:var(--radius);
          text-align:left; font-weight:500; font-size:0.92rem; line-height:1.45;
          padding:16px 18px; min-height:116px; white-space:normal;
          box-shadow:0 2px 8px -6px rgba(31,41,55,0.25);
          transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease; }
      .block-container .stButton>button:hover {
          transform:translateY(-3px); border-color:var(--p2); box-shadow:var(--shadow); }
      .block-container .stButton>button:focus { box-shadow:0 0 0 2px var(--p1) inset; }

      /* ---- Chat messages ---- */
      .stChatMessage { animation:msgIn .3s ease both; background:transparent; }
      @keyframes msgIn { from{opacity:0; transform:translateY(8px)} to{opacity:1; transform:none} }
      [data-testid="stChatMessageAvatarAssistant"] {
          background:var(--grad) !important; color:#fff !important; }

      /* ---- Typing indicator ---- */
      .typing { display:inline-flex; gap:6px; padding:8px 2px; align-items:center; }
      .typing span { width:9px; height:9px; border-radius:50%; background:var(--grad);
          animation:blink 1.2s infinite ease-in-out; }
      .typing span:nth-child(2){ animation-delay:.2s } .typing span:nth-child(3){ animation-delay:.4s }
      @keyframes blink { 0%,80%,100%{transform:scale(.55);opacity:.4} 40%{transform:scale(1);opacity:1} }

      /* ---- Confidence ---- */
      .conf-badge { display:inline-flex; align-items:center; gap:6px; padding:5px 13px;
          border-radius:999px; font-size:0.82rem; font-weight:600; line-height:1; border:1px solid transparent; }
      .conf-high   { background:rgba(22,163,74,0.1);  color:var(--ok);   border-color:rgba(22,163,74,0.3); }
      .conf-medium { background:rgba(217,119,6,0.1);  color:var(--warn); border-color:rgba(217,119,6,0.3); }
      .conf-low    { background:rgba(220,38,38,0.1);  color:var(--bad);  border-color:rgba(220,38,38,0.3); }
      .conf-track { height:8px; border-radius:999px; background:var(--surface-2);
          overflow:hidden; margin:10px 0 4px; }
      .conf-fill  { height:100%; border-radius:999px; transition:width .6s ease; }
      .fill-high  { background:linear-gradient(90deg,#15803d,#22c55e); }
      .fill-medium{ background:linear-gradient(90deg,#b45309,#f59e0b); }
      .fill-low   { background:linear-gradient(90deg,#b91c1c,#ef4444); }
      .conf-reasons { font-size:0.8rem; color:var(--muted); margin-top:4px; line-height:1.6; }

      /* ---- Source cards ---- */
      .src-card { border:1px solid var(--border); border-radius:14px; padding:14px 16px;
          margin-bottom:12px; background:var(--surface); box-shadow:0 2px 8px -8px rgba(31,41,55,0.3);
          transition:transform .15s ease, box-shadow .15s ease; }
      .src-card:hover { transform:translateY(-2px); box-shadow:var(--shadow); }
      .src-card.cited { border-color:rgba(22,163,74,0.5); }
      .src-head { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:8px; font-weight:600; }
      .src-num  { display:inline-flex; align-items:center; justify-content:center; min-width:24px; height:24px;
          border-radius:8px; font-size:0.78rem; font-weight:700; color:#fff; background:var(--grad); }
      .src-card.cited .src-num { background:linear-gradient(135deg,#15803d,#22c55e); }
      .chip { font-size:0.72rem; padding:3px 9px; border-radius:999px; font-weight:600;
          background:var(--surface-2); color:var(--muted); }
      .chip.cited { background:rgba(22,163,74,0.14); color:var(--ok); }
      .chip.hybrid { background:rgba(99,102,241,0.14); color:var(--p1); }
      .chip.pageindex { background:rgba(139,92,246,0.16); color:var(--p2); }
      .src-body { font-size:0.86rem; color:var(--muted); line-height:1.55; max-height:130px;
          overflow:auto; padding-right:4px; }

      /* ---- Citation code + meta ---- */
      .stChatMessage code { font-family:'JetBrains Mono',monospace; font-weight:600;
          background:rgba(99,102,241,0.12); color:var(--p1); padding:1px 6px; border-radius:6px; }
      .msg-time { font-size:0.72rem; color:var(--muted); margin-top:6px; }
      .meta-line { font-size:0.76rem; color:var(--muted); margin:2px 0 8px; }

      /* ---- Sidebar ---- */
      section[data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--border); }
      section[data-testid="stSidebar"] .stButton>button {
          background:var(--surface-2); border:1px solid transparent; border-radius:12px; text-align:left;
          font-size:0.85rem; font-weight:500; color:var(--text); padding:9px 14px;
          transition:all .15s ease; }
      section[data-testid="stSidebar"] .stButton>button:hover {
          background:#fff; border-color:var(--p2); color:var(--p2); transform:translateX(2px); }
      .side-brand { font-weight:800; font-size:1.1rem; letter-spacing:-0.3px;
          background:var(--grad); -webkit-background-clip:text; background-clip:text; color:transparent; }
      .side-label { font-size:0.72rem; font-weight:700; text-transform:uppercase;
          letter-spacing:0.6px; color:var(--muted); margin:2px 0 6px; }

      /* ---- Chat input pill ---- */
      div[data-testid="stChatInput"] {
          border-radius:28px !important; background:var(--surface) !important;
          border:1px solid var(--border) !important; box-shadow:0 4px 16px -8px rgba(31,41,55,0.25);
          transition:box-shadow .2s ease, border-color .2s ease; }
      div[data-testid="stChatInput"]:focus-within {
          border-color:transparent !important;
          box-shadow:0 0 0 2px var(--p1), 0 6px 20px -6px rgba(99,102,241,0.4); }
      div[data-testid="stChatInput"] textarea { font-size:0.95rem; color:var(--text); }
    </style>
    """,
    unsafe_allow_html=True,
)

TYPING_HTML = "<div class='typing'><span></span><span></span><span></span></div>"

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


def regenerate_last():
    """Bỏ câu trả lời cuối và hỏi lại câu hỏi gần nhất."""
    msgs = st.session_state.messages
    if msgs and msgs[-1]["role"] == "assistant":
        msgs.pop()
    if msgs and msgs[-1]["role"] == "user":
        st.session_state.pending_query = msgs.pop()["content"]
    st.rerun()


def render_answer_actions(idx: int, msg: dict):
    """Hàng hành động dưới câu trả lời cuối: tạo lại · feedback · sao chép."""
    c1, c2, _ = st.columns([1.4, 1.4, 6])
    with c1:
        if st.button("🔁 Tạo lại", key=f"regen_{idx}", use_container_width=True):
            regenerate_last()
    with c2:
        if hasattr(st, "feedback"):
            st.feedback("thumbs", key=f"fb_{idx}")
    with st.expander("📋 Sao chép câu trả lời"):
        st.code(msg.get("content", ""), language=None)


def render_message(idx: int, msg: dict, is_last: bool):
    """Render một tin nhắn trong lịch sử."""
    avatar = BOT_AVATAR if msg["role"] == "assistant" else USER_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant":
            st.markdown(highlight_citations(msg["content"]))
            if msg.get("confidence"):
                render_confidence(msg["confidence"])
            render_sources(msg.get("sources", []), msg.get("citations", []), msg.get("retrieval_source", "hybrid"))
            if msg.get("time"):
                st.markdown(f"<div class='msg-time'>🕐 {msg['time']}</div>", unsafe_allow_html=True)
            if is_last:
                render_answer_actions(idx, msg)
        else:
            st.markdown(msg["content"])
            if msg.get("time"):
                st.markdown(f"<div class='msg-time'>🕐 {msg['time']}</div>", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("<div class='side-brand'>🛍️ E-commerce Support RAG</div>", unsafe_allow_html=True)
    st.caption("Trợ lý hỏi đáp chính sách TMĐT & hỗ trợ khách hàng (đổi trả, thanh toán, bảo mật, người bán)")

    st.divider()
    st.markdown("<div class='side-label'>💡 Câu hỏi gợi ý</div>", unsafe_allow_html=True)
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
    st.markdown("<div class='side-label'>⚙️ Thiết lập</div>", unsafe_allow_html=True)
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
# TOP BAR
# =============================================================================

st.markdown(
    """
    <div class="topbar">
      <span class="mark">🛍️</span>
      <span class="name">E-commerce Support</span>
      <span class="tag">RAG Chatbot</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# WELCOME SCREEN (khi chưa có hội thoại)
# =============================================================================

if not st.session_state.messages:
    st.markdown("<div class='greet'>Xin chào 👋</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='greet-sub'>Mình có thể giúp gì cho bạn về chính sách thương mại điện tử hôm nay?</div>",
        unsafe_allow_html=True,
    )

    card_prompts = [
        ("🔄", "Chính sách đổi trả", suggestions[0]),
        ("💳", "Phương thức thanh toán", suggestions[1]),
        ("🛠️", "Đổi phương thức thanh toán", suggestions[2]),
        ("🏪", "Quy định người bán", suggestions[3]),
    ]
    cols = st.columns(2)
    for i, (icon, title, prompt) in enumerate(card_prompts):
        with cols[i % 2]:
            if st.button(f"{icon}  **{title}**\n\n{prompt}", key=f"card_{i}", use_container_width=True):
                st.session_state["pending_query"] = prompt
                st.rerun()

# =============================================================================
# LỊCH SỬ CHAT
# =============================================================================

for i, msg in enumerate(st.session_state.messages):
    render_message(i, msg, is_last=(i == len(st.session_state.messages) - 1))

# =============================================================================
# QUERY HANDLING
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi của bạn về chính sách/hỗ trợ e-commerce…")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    st.session_state.messages.append({
        "role": "user", "content": query, "time": datetime.now().strftime("%H:%M"),
    })
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(query)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        answer, sources, retrieval_source, citations, confidence = "", [], "none", [], None
        thinking = st.empty()
        try:
            from src.task10_generation import (
                prepare_generation, stream_answer, parse_citations, compute_confidence,
            )

            # Typing indicator ba chấm trong lúc truy xuất
            thinking.markdown(TYPING_HTML, unsafe_allow_html=True)
            prep = prepare_generation(query, top_k=top_k)
            sources = prep["sources"]
            retrieval_source = prep["retrieval_source"]
            thinking.empty()

            if sources:
                origin_label = "🧭 PageIndex" if retrieval_source == "pageindex" else "🔀 Hybrid"
                st.markdown(
                    f"<div class='meta-line'>{origin_label} · tìm thấy {len(sources)} tài liệu liên quan</div>",
                    unsafe_allow_html=True,
                )

            # Streaming câu trả lời (hiệu ứng typing)
            answer = st.write_stream(stream_answer(prep["messages"]))
            citations = parse_citations(answer)
            confidence = compute_confidence(answer, sources, retrieval_source)

            render_confidence(confidence)
            render_sources(sources, citations, retrieval_source)

        except NotImplementedError:
            thinking.empty()
            answer = ("⚠️ **Pipeline chưa hoàn tất.** Hãy hoàn thành các Task 4–8 (chunking, "
                      "semantic/lexical search, rerank, PageIndex) để `retrieve()` và `generate_with_citation()` "
                      "chạy end-to-end.")
            st.markdown(answer)
        except Exception as e:
            thinking.empty()
            answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "citations": citations,
        "retrieval_source": retrieval_source,
        "confidence": confidence,
        "time": datetime.now().strftime("%H:%M"),
    })
    st.rerun()
