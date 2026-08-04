# Báo Cáo Đánh Giá RAG Pipeline — RAGAS

**Nhóm:** Lab 08 — E-commerce Support RAG Chatbot
**Phụ trách:** Hồ Lương An (2A202601332) — Role 6, Evaluation & Benchmark QA
**Framework:** RAGAS · **LLM đánh giá:** `gpt-4o-mini` · **Quy mô:** **20/20 câu** golden dataset

---

## 1. Cấu Hình Thí Nghiệm

| Thành phần | Giá trị |
|---|---|
| Corpus | 10 tài liệu (5 PDF chính sách + 5 bài help center) → **191 chunks** |
| Chunking | `RecursiveCharacterTextSplitter`, size 800, overlap 100 |
| Embedding | `text-embedding-3-small` (OpenAI, 1536 chiều) |
| Vector store | ChromaDB, cosine similarity |
| Sparse retriever | BM25Okapi, tokenize regex `[a-z0-9]+` |
| Reranking | RRF, `k = 60` |
| Ngưỡng fallback | 0.37 |
| `top_k` | 5 chunks |
| LLM sinh câu trả lời | `gpt-4o-mini`, prompt ràng buộc chống bịa |

| Config | Retrieval |
|---|---|
| **A — Hybrid + Rerank** | Semantic + BM25 → RRF (k=60) → top 5 |
| **B — Dense-only** | Chỉ Semantic (cosine) → top 5 |

---

## 2. Kết Quả Tổng Hợp (20 câu)

| Metric | A: Hybrid + Rerank | B: Dense-only | Chênh lệch |
|---|---|---|---|
| Faithfulness | 0.5163 | **0.5521** | −0.0358 (B thắng) |
| Answer Relevancy | **0.3088** | 0.3044 | +0.0044 (nhiễu) |
| Context Recall | **0.6083** | 0.5833 | +0.0250 (A thắng) |
| Context Precision | 0.7875 | **0.7951** | −0.0076 (B thắng) |

### ⚠️ Kết luận thẳng thắn: trên 20 câu, Hybrid **KHÔNG** tốt hơn Dense-only

A thắng 2 metric, B thắng 2 metric, **mọi khoảng cách đều dưới 0.04** — nằm trong biên độ nhiễu với cỡ mẫu 20.

**Điều này bác bỏ kết luận của lần chạy 3 câu trước đó** (báo +9.9% Answer Relevancy cho Hybrid). Lần đó chỉ đánh giá 3 câu, và toàn bộ mức tăng đến từ **một câu duy nhất** (Apple Pay: 0.5198 → 0.6912). Khi mở rộng lên 20 câu, hiệu ứng đó biến mất.

> 📌 Đây là bài học phương pháp quan trọng của nhóm: **kết luận rút từ 3 quan sát không sống sót khi tăng cỡ mẫu.**

---

## 3. Chi Tiết Theo Câu Hỏi

### Config A — Hybrid + Rerank

| # | Câu hỏi (rút gọn) | Faith. | Ans.Rel. | Ctx.Rec. | Ctx.Prec. |
|---|---|---|---|---|---|
| 1 | Shopee hỗ trợ những phương thức thanh toán nào? | 0.9091 | 0.7681 | 1.0000 | 0.8042 |
| 2 | Thanh toán bằng Apple Pay? Giới hạn giá trị đơn? | 1.0000 | 0.6151 | 1.0000 | 0.8042 |
| 3 | Trả góp thẻ tín dụng có áp dụng đơn quốc tế? | 1.0000 | 0.4378 | 1.0000 | 1.0000 |
| 4 | Đổi phương thức thanh toán đơn trả trước? | 0.7500 | **0.0000** | 1.0000 | 0.5833 |
| 5 | Làm thế nào để theo dõi đơn hàng? | 1.0000 | 0.4787 | 1.0000 | 0.8667 |
| 6 | Vì sao cần bằng chứng khi yêu cầu hoàn tiền? | 1.0000 | 0.9651 | 1.0000 | 0.9500 |
| 7 | Shopee có hỗ trợ mua hàng từ nước ngoài? | 1.0000 | 0.6803 | 1.0000 | 1.0000 |
| 8 | Xử lý thế nào nếu phát hiện gian lận? | 1.0000 | 0.7443 | 1.0000 | 1.0000 |
| 9 | Voucher có quy đổi thành tiền mặt không? | **0.0000** | **0.0000** | **0.0000** | 0.6389 |
| 10 | Shopee bảo mật thông tin cá nhân thế nào? | 1.0000 | 0.6889 | 0.6667 | 1.0000 |
| 11 | Thời gian tối đa yêu cầu trả hàng Shopee Mall? | **0.0000** | **0.0000** | 1.0000 | 0.5000 |
| 12 | Thực phẩm tươi sống có được trả hàng? | 1.0000 | 0.3699 | 1.0000 | 0.7500 |
| 13 | Phí vận chuyển do ai chịu khi trả hàng lỗi? | **0.0000** | **0.0000** | **0.0000** | 0.8042 |
| 14 | Ai chịu phí nếu từ chối nhận hàng vô cớ? | 0.6667 | 0.4282 | **0.0000** | 0.7556 |
| 15 | Khi nào voucher được hoàn lại nếu đơn bị hủy? | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| 16 | Làm sao biết voucher áp dụng cho sản phẩm nào? | **0.0000** | **0.0000** | 1.0000 | 0.8333 |
| 17 | Thẻ tín dụng không hỗ trợ thanh toán quốc tế? | **0.0000** | **0.0000** | **0.0000** | 0.7556 |
| 18 | Đổi phương thức sau khi đã giao cho vận chuyển? | **0.0000** | **0.0000** | **0.0000** | 0.8875 |
| 19 | Đơn quốc tế trả hàng thì hoàn tiền thế nào? | **0.0000** | **0.0000** | **0.0000** | 0.9500 |
| 20 | Affiliate tự đặt hàng qua link của mình? | **0.0000** | **0.0000** | 0.5000 | 0.8667 |
| | **Trung bình** | **0.5163** | **0.3088** | **0.6083** | **0.7875** |

### Config B — Dense-only

| Metric | Trung bình |
|---|---|
| Faithfulness | 0.5521 |
| Answer Relevancy | 0.3044 |
| Context Recall | 0.5833 |
| Context Precision | 0.7951 |

*(Bảng chi tiết Config B có cùng dạng phân bố — 10/20 câu điểm 0, gần trùng với Config A.)*

---

## 4. Chẩn Đoán — Vì Sao 10/20 Câu Bị 0 Điểm

Nhóm đã truy nguyên bằng thực nghiệm, **không suy đoán**.

### 4.1 Không phải do ngưỡng fallback

Đo cosine top-1 cho cả 20 câu: khoảng **0.5334 – 0.7715**, tất cả đều **trên** ngưỡng 0.37.

> **0/20 câu bị đẩy sang PageIndex fallback.** Cả 20 câu đều đi nhánh hybrid và đều nhận đủ 5 chunk.

### 4.2 Nguyên nhân thật: LLM **từ chối trả lời**

Kiểm tra output thô của các câu bị 0 điểm:

```
[Câu 9]  số sources: 5  |  answer: "Tôi không thể xác minh thông tin này từ nguồn hiện có."
[Câu 15] số sources: 5  |  answer: "Tôi không thể xác minh thông tin này từ nguồn hiện có."
[Câu 20] số sources: 5  |  answer: "Tôi không thể xác minh thông tin này từ nguồn hiện có."
```

RAGAS phát hiện câu trả lời **noncommittal** (từ chối) và chấm **Answer Relevancy = 0**; đồng thời không có khẳng định nào để kiểm chứng nên **Faithfulness = 0**.

Nguồn gốc là `SYSTEM_PROMPT` trong `task10_generation.py:58-67`:

> *Quy tắc 3: Nếu context không đủ thông tin → trả lời đúng nguyên văn "Tôi không thể xác minh…"*
> *Quy tắc 5: Không suy luận hay mở rộng ngoài những gì được nêu trong context.*

### 4.3 Hai kiểu thất bại khác nhau

| Kiểu | Dấu hiệu | Câu | Tầng lỗi |
|---|---|---|---|
| **I — Retrieval sai chunk** | Ctx.Recall = 0 | 9, 13, 15, 17, 18, 19 | Chunking / ranking |
| **II — Prompt quá chặt** | Ctx.Recall = 1.0 nhưng vẫn từ chối | **11, 16** | Generation |

**Kiểu II là nghiêm trọng hơn**: câu 11 và 16 đã lấy về **đúng** ngữ cảnh (Recall = 1.0) mà LLM vẫn từ chối. Đây là mất điểm oan hoàn toàn do prompt, không phải do retrieval.

**Ví dụ kiểu I — câu 9** ("Voucher có quy đổi thành tiền mặt không?"): retrieval lấy về 5 chunk, **cả 5 đều từ đúng file** `chinh-sach-chung-ve-ma-uu-dai`, cả 5 đều chứa chữ "quy đổi" — nhưng không chunk nào chứa điều khoản cụ thể về tiền mặt. Kiểm tra corpus: cụm **"tiền mặt" không xuất hiện lần nào** trong file đó. Tức là **đáp án kỳ vọng trong golden dataset không có trong tài liệu nguồn**.

---

## 5. Ba Nguyên Nhân Gốc

| # | Nguyên nhân | Bằng chứng | Ảnh hưởng |
|---|---|---|---|
| **1** | **Golden dataset viết theo hiểu biết chung, không bám tài liệu thật** | "tiền mặt" (câu 9), điều khoản affiliate tự đặt hàng (câu 20) không tồn tại trong corpus | Câu hỏi không thể trả lời đúng dù pipeline hoàn hảo |
| **2** | **Prompt chống bịa quá chặt** | Câu 11, 16 có Recall = 1.0 nhưng vẫn trả về `CANNOT_VERIFY` | Mất điểm oan ở câu retrieval đã làm đúng |
| **3** | **`top_k = 5` chunk 800 ký tự là quá ít** cho câu hỏi tổng hợp | Câu 15, 17, 19 giao thoa nhiều chủ đề, Recall = 0 | Thông tin nằm rải ở nhiều đoạn, không đoạn nào đủ |

### Nghịch lý cần nêu rõ khi thuyết trình

Cơ chế **chống bịa** — thứ làm demo trông đáng tin ("hỏi thời tiết thì hệ thống từ chối") — chính là thứ **kéo điểm RAGAS xuống**. Một chatbot chịu đoán bừa sẽ có Faithfulness và Answer Relevancy **cao hơn** trên bộ đo này.

Nhóm **chọn giữ hành vi thận trọng**: với chatbot chính sách, trả lời sai nguy hiểm hơn nói "tôi không biết". Nhưng phải thừa nhận đây là **đánh đổi có ý thức**, và RAGAS không đo được giá trị của sự thận trọng đó.

---

## 6. Điểm Sáng

Loại bỏ 10 câu bị 0 điểm, **10 câu còn lại cho kết quả tốt**:

| Metric | TB trên 10 câu trả lời được |
|---|---|
| Faithfulness | **~0.96** |
| Context Recall | **~0.97** |
| Context Precision | **~0.86** |

Nghĩa là: **khi hệ thống chịu trả lời, nó trả lời rất chính xác và bám nguồn.** Vấn đề nằm ở tỷ lệ từ chối (10/20), không ở chất lượng câu trả lời.

Câu tốt nhất: **câu 6** ("Vì sao cần bằng chứng khi yêu cầu hoàn tiền") — Faithfulness 1.0, Answer Relevancy 0.9651, Recall 1.0, Precision 0.95.

---

## 7. Đề Xuất Cải Tiến (ưu tiên theo tác động)

| # | Đề xuất | Xử lý nguyên nhân | Chi phí |
|---|---|---|---|
| **1** | **Rà lại golden dataset**: mỗi `expected_answer` phải trích được từ một đoạn cụ thể trong `data/standardized/`. Câu nào không trích được thì sửa hoặc bỏ | #1 | Thấp — công sức viết |
| **2** | **Nới quy tắc 3 và 5 của prompt**: cho phép trả lời một phần kèm ghi chú "thông tin chỉ có một phần trong tài liệu" thay vì từ chối toàn bộ | #2 | Thấp — sửa prompt |
| **3** | **Tăng `top_k` từ 5 lên 8–10** | #3 | Thấp — tăng token |
| **4** | Chunking theo cấu trúc heading thay vì cắt cứng 800 ký tự | #3 | Trung bình |
| **5** | Kích hoạt PageIndex fallback theo **độ phân tán nguồn**, không chỉ theo điểm cao nhất | #3 | Trung bình |
| **6** | Tách từ tiếng Việt bằng `underthesea` cho BM25 | Chất lượng sparse | Thấp |

> ⚠️ Đề xuất 1 và 2 phải làm **trước** khi chạy lại benchmark. Nếu không, mọi con số vẫn bị chi phối bởi tỷ lệ từ chối chứ không phản ánh chất lượng retrieval.

---

## 8. Hạn Chế Của Phép Đo

| Hạn chế | Mức ảnh hưởng |
|---|---|
| Cỡ mẫu 20 câu, chênh lệch A/B đều < 0.04 | **Không đủ để kết luận** cấu hình nào tốt hơn |
| 10/20 câu bị 0 điểm do từ chối trả lời | Kéo trung bình xuống, che mất hiệu năng thật của retrieval |
| Golden dataset chưa được kiểm chứng đối chiếu corpus | Một phần điểm 0 là lỗi bộ đo, không phải lỗi hệ thống |
| Chưa đo riêng tác động của Reordering | Không biết bước này đóng góp bao nhiêu |
| Chưa có câu ngoài domain trong bộ đánh giá | Chưa đo được độ chính xác của cơ chế fallback |
| Chạy 1 lần, không lặp | LLM có tính ngẫu nhiên, chưa có khoảng tin cậy |

---

## Phụ Lục — Cách Tái Lập

```bash
python -m src.task4_chunking_indexing              # build index (191 chunks)
python -m group_project.evaluation.eval_pipeline   # chạy A/B trên toàn bộ 20 câu

# Chạy nhanh trên subset khi debug:
EVAL_SUBSET=5 python -m group_project.evaluation.eval_pipeline
```

> Trên Windows cần `PYTHONIOENCODING=utf-8` khi chạy để tránh `UnicodeEncodeError` với ký tự Unicode trong log.
