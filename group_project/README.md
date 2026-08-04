# Bài Tập Nhóm — E-commerce Support RAG Chatbot

**Lab 08 — RAG Pipeline v2: Retrieval Hybrid, Vectorless Fallback & Generation có Citation**

---

## 1. Thành Viên & Phân Công

| # | Họ và tên | MSSV | Role | Nhiệm vụ chính | Trạng thái |
|---|---|---|---|---|---|
| 1 | **Bế Nguyễn Hà Sơn** | 2A202601454 | 👑 Role 1 — Team Leader & RAG Architect | Kiến trúc pipeline, **Task 9** (Retrieval Pipeline + Fallback logic), quản lý Git & tích hợp, hiệu chỉnh ngưỡng fallback | ✅ |
| 2 | **Phạm Tung Dương** | 2A202601404 | ⚙️ Role 2 — Data Engineering & Scraping | **Task 1** (thu thập PDF chính sách), **Task 2** (crawl help center), **Task 3** (convert Markdown) | ✅ |
| 3 | **Nguyễn Xuân Quân** | 2A202601976 | 🔵 Role 3 — Vector DB & Dense Search | **Task 4** (Chunking & ChromaDB Indexing), **Task 5** (Semantic Search) | ✅ |
| 4 | **Phạm Trung Hiếu** | 2A202601834 | 🟠 Role 4 — Sparse Retrieval & Fallback | **Task 6** (BM25), **Task 7** (RRF Reranking), **Task 8** (PageIndex Vectorless) | ✅ |
| 5 | **Nguyễn Thành Vinh** | 2A202601556 | 🎨 Role 5 — Frontend UI & Integration | **Task 10** (Generation + Citation), Streamlit Chatbot `app.py` | ✅ |
| 6 | **Hồ Lương An** | 2A202601332 | 📊 Role 6 — Evaluation & Benchmark QA | `golden_dataset.json`, RAGAS benchmark, báo cáo `results.md` | ✅ |

---

## 2. Kiến Trúc Hệ Thống

```
   10 tài liệu nguồn (5 PDF chính sách + 5 bài help center)
                        │
                        ▼  Task 3 — MarkItDown
              data/standardized/*.md
                        │
                        ▼  Task 4 — RecursiveCharacterTextSplitter (800 / overlap 100)
                   191 chunks
                        │
                        ▼  text-embedding-3-small (1536 chiều)
                   ChromaDB  (collection: ecommerce_support_docs)


                    ┌─→ Task 5: Semantic Search (cosine) ──┐
   Câu hỏi ─────────┤                                       ├─→ Task 7: RRF (k=60) ─→ top_k
                    └─→ Task 6: BM25 (rank-bm25) ──────────┘             │
                                                                          │
        cosine cao nhất < 0.37 ──→ Task 8: PageIndex Fallback ───────────┤
                                    (vectorless, đọc theo cây mục lục)    │
                                                                          ▼
                                             Task 10: Reorder (front + back[::-1])
                                                          │
                                                          ▼
                                       gpt-4o-mini ─→ Câu trả lời + [Nguồn: ...]
                                                          │
                                                          ▼
                                              app.py (Streamlit Chatbot)
```

### Ba quyết định kiến trúc

| Quyết định | Lý do |
|---|---|
| **Hybrid** thay vì chỉ Semantic | Semantic giỏi hiểu ý nhưng trượt từ khoá đặc thù (mã voucher, số điều khoản); BM25 ngược lại. Kết hợp để bù trừ |
| Gộp bằng **RRF** thay vì cộng điểm | Cosine ∈ [0,1] còn BM25 là điểm thô 0→20+, **không cộng thẳng được**. RRF chỉ dùng thứ hạng nên gộp công bằng |
| Có **ngưỡng fallback** | Không đủ tài liệu liên quan thì chuyển PageIndex thay vì đưa đoạn rác cho LLM — cơ chế chống bịa |

---

## 3. Cấu Hình Chốt

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `CHUNK_SIZE` | 800 ký tự | Đủ chứa trọn một điều khoản |
| `CHUNK_OVERLAP` | 100 ký tự | Tránh cắt đôi câu ở ranh giới |
| Embedding | `text-embedding-3-small` | OpenAI, **1536 chiều** |
| Vector store | ChromaDB (cosine) | 191 chunks |
| Sparse retriever | `rank-bm25` (BM25Okapi) | Tokenize regex `[a-z0-9]+` |
| RRF `k` | 60 | Cormack et al. 2009 |
| **Ngưỡng fallback** | **0.37** | **Hiệu chỉnh bằng đo thực tế — xem mục 4** |
| LLM | `gpt-4o-mini` | Qua `OPENAI_API_KEY` |

---

## 4. Hiệu Chỉnh Ngưỡng Fallback (đo thực tế)

Nhóm **không copy giá trị mẫu 0.48** trong đề bài. Thay vào đó đo điểm cosine top-1 trên 22 câu hỏi thật, chia 3 nhóm:

| Nhóm câu hỏi | Số câu | Khoảng điểm | Trung bình |
|---|---|---|---|
| Trong domain (tiếng Việt) | 10 | **0.445 – 0.770** | 0.647 |
| Trong domain (tiếng Anh) | 4 | **0.299 – 0.486** | 0.413 |
| Ngoài domain (thời tiết, nấu ăn, bóng đá…) | 8 | **0.147 – 0.337** | 0.273 |

**Cửa sổ an toàn: 0.34 – 0.39.** Chọn **0.37**.

Với ngưỡng này: **0/8** câu ngoài domain lọt lưới (so với **3/8** nếu để ngưỡng 0.30), và **14/14** câu trong domain vẫn đi nhánh hybrid.

### Hai phát hiện đáng chú ý

1. **Câu hỏi tiếng Anh cho điểm thấp hơn hẳn tiếng Việt** (TB 0.413 vs 0.647) vì corpus toàn tiếng Việt. Truy vấn tiếng Anh rơi vào cùng vùng điểm với câu ngoài domain — đây là nguyên nhân khiến hai cụm chồng lấn nhẹ.
2. **Câu hỏi về sàn khác lại cho điểm cao**: *"Lazada có chính sách đổi trả không?"* = **0.591**, cao hơn cả câu tiếng Anh hợp lệ. Ngưỡng cosine **không giải quyết được** trường hợp này, vì về mặt ngữ nghĩa "chính sách đổi trả Lazada" thật sự gần với nội dung "chính sách đổi trả Shopee". Cần lọc theo metadata nguồn mới xử lý được.

---

## 5. Hướng Dẫn Chạy

```bash
# 1. Môi trường (Python 3.11+)
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
pip install -r requirements.txt

# 2. Cấu hình — tạo .env ở thư mục gốc, điền OPENAI_API_KEY=sk-...
cp .env.example .env

# 3. Chuẩn hoá dữ liệu (nếu data/standardized/ trống)
python -m src.task3_convert_markdown

# 4. Build vector index — BẮT BUỘC, chroma_db/ không commit lên git
python -m src.task4_chunking_indexing

# 5. Chạy chatbot
streamlit run app.py

# 6. Chạy đánh giá RAGAS
python -m group_project.evaluation.eval_pipeline
```

### Kiểm tra bài cá nhân

```bash
python -m pytest tests/test_individual.py -v      # kỳ vọng: 35 passed
```

---

## 6. Kết Quả Đánh Giá

Chi tiết đầy đủ: [`evaluation/results.md`](evaluation/results.md) — chạy trên **toàn bộ 20/20 câu**.

| Metric | A: Hybrid + Rerank | B: Dense-only | Chênh lệch |
|---|---|---|---|
| Faithfulness | 0.5163 | **0.5521** | −0.036 |
| Answer Relevancy | **0.3088** | 0.3044 | +0.004 |
| Context Recall | **0.6083** | 0.5833 | +0.025 |
| Context Precision | 0.7875 | **0.7951** | −0.008 |

**Kết luận:** trên 20 câu, Hybrid **không** tốt hơn Dense-only — mỗi bên thắng 2 metric, mọi khoảng cách < 0.04 (trong biên độ nhiễu). Kết quả này **bác bỏ** kết luận "+9.9%" của lần chạy 3 câu trước đó, khi toàn bộ mức tăng đến từ một câu duy nhất.

**Phát hiện chính:** 10/20 câu bị 0 điểm vì LLM trả về `"Tôi không thể xác minh thông tin này từ nguồn hiện có"`. Đã truy nguyên:

- **Không phải do ngưỡng fallback** — đo cosine cả 20 câu đều nằm trong 0.533–0.772, **0/20 câu** bị đẩy sang PageIndex.
- Nguyên nhân là **prompt chống bịa quá chặt** (2 câu có Context Recall = 1.0 mà vẫn từ chối) và **golden dataset có đáp án không tồn tại trong corpus** (cụm "tiền mặt" ở câu 9 không xuất hiện lần nào trong tài liệu nguồn).

**Điểm sáng:** loại 10 câu bị từ chối, 10 câu còn lại đạt Faithfulness ~0.96 và Context Recall ~0.97 — khi hệ thống chịu trả lời, nó trả lời rất chính xác.

---

## 7. Hạn Chế Đã Ghi Nhận

| Hạn chế | Ảnh hưởng | Hướng khắc phục |
|---|---|---|
| Corpus nhỏ (10 tài liệu / 191 chunk) | Cỡ mẫu chưa đủ để kết luận mạnh về A/B | Mở rộng nguồn thu thập |
| Golden dataset đánh giá trên số câu ít | Chênh lệch vài % chưa chắc có ý nghĩa thống kê | Chạy đủ 15–20 câu |
| Tokenize BM25 chưa tách từ tiếng Việt | "trả hàng" bị tách thành 2 token rời | Dùng `underthesea` |
| Không phân biệt được câu hỏi về sàn khác | Hỏi về Lazada vẫn trả lời bằng tài liệu Shopee | Lọc theo metadata nguồn |
| Chưa đo riêng tác động của Reordering | Không biết bước này đóng góp bao nhiêu | Chạy A/B riêng cho biến này |
| Tài liệu mâu thuẫn chưa được phát hiện | LLM tự chọn, không cảnh báo | Thêm metadata ngày ban hành, ưu tiên bản mới |

---

## 8. Ghi Chú Kỹ Thuật — 3 lỗi đã xử lý

| Lỗi | Biểu hiện | Nguyên nhân & cách sửa |
|---|---|---|
| **`UnicodeEncodeError: '✓'`** | Task 4 chạy "không báo lỗi" nhưng ChromaDB **0 chunk**, `semantic_search` trả rỗng | Console Windows dùng codepage cp1252, không in được ký tự `✓` → `print()` ném exception làm chết pipeline ngay sau bước load. Vá ở `src/__init__.py`: ép `stdout/stderr` sang UTF-8 |
| **BM25 không khớp truy vấn tiếng Anh** | `lexical_search("payment methods")` trả rỗng dù corpus có chữ "payment" | `.split()` giữ nguyên dấu câu và gạch dưới → frontmatter `topic: "payment_methods"` cho ra token `"payment_methods"`. Đổi sang regex `[a-z0-9]+` tách đúng thành `payment` + `methods` |
| **`rerank()` luôn ném `NotImplementedError`** | Task 7 và Task 10 hỏng dây chuyền | Nhánh `method="rrf"` (mặc định) chưa nối vào `rerank_rrf`. Sửa: bọc `rerank_rrf([candidates], top_k)` |

---

## 9. Cấu Trúc Bàn Giao

```
├── app.py                              ← Streamlit Chatbot
├── src/task1..task10                   ← Pipeline 10 bước
├── data/landing/                       ← Dữ liệu thô (PDF + JSON)
├── data/standardized/                  ← Markdown đã chuẩn hoá
├── tests/test_individual.py            ← 35 test (bài cá nhân)
├── group_project/
│   ├── README.md                       ← Báo cáo nhóm (file này)
│   └── evaluation/
│       ├── golden_dataset.json         ← Bộ câu hỏi kiểm thử
│       ├── eval_pipeline.py            ← Script chạy RAGAS
│       └── results.md                  ← Bảng điểm A/B + phân tích
└── ROLE_GUIDE.md · GROUP_WORKFLOW.md · DEMO_SCRIPT.md · TASK2_CRAWL.md
```

> ⚠️ `chroma_db/` **không commit** (SQLite binary, git không merge được). Mỗi máy tự chạy `python -m src.task4_chunking_indexing`.
> ⚠️ `.env` **không commit** (chứa `OPENAI_API_KEY`).
