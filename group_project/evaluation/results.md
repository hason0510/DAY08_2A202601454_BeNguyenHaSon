# RAG Evaluation Results

## Framework sử dụng

> RAGAS (Retrieval Augmented Generation Assessment)

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.9285 | 0.9500 | -0.0215 |
| Answer Relevance | 0.6850 | 0.6841 | +0.0009 |
| Context Recall | 0.8000 | 0.8000 | +0.0000 |
| Context Precision | 0.7617 | 0.8219 | -0.0603 |
| **Average** | **0.7938** | **0.8140** | **-0.0202** |

---

## A/B Comparison Analysis

**Config A:**
> Pipeline đầy đủ: Semantic Search + Lexical Search → RRF Fusion → Reranking → PageIndex Fallback → LLM Generation. Kết hợp cả dense (vector) lẫn sparse (BM25) retrieval để đa dạng hoá kết quả.

**Config B:**
> Dense-only: Chỉ sử dụng Semantic Search (cosine similarity) trực tiếp, không qua Lexical Search, không RRF merge, không reranking. Pipeline tối giản để so sánh baseline.

**Kết luận:**
> Config B (Dense-only) cho kết quả tốt hơn. Config B đạt điểm trung bình cao hơn 0.0202. Semantic search thuần cho kết quả tốt hơn trong trường hợp này, có thể do corpus nhỏ và embedding model đã capture đủ semantic meaning.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Tôi có thể thay đổi phương thức thanh toán cho đơn... | 1.00 | 0.00 | 0.00 | Retrieval | Retriever không tìm đủ context liên quan |
| 2 | Làm thế nào để theo dõi tình trạng đơn hàng của tô... | 0.73 | 0.85 | 1.00 | Unknown | Cần phân tích thêm |
| 3 | Shopee hỗ trợ những phương thức thanh toán nào? | 0.91 | 0.90 | 1.00 | Unknown | Cần phân tích thêm |

---

## Recommendations

### Cải tiến 1
**Action:** Tinh chỉnh prompt template để yêu cầu LLM trả lời trực tiếp câu hỏi trước, sau đó mới bổ sung chi tiết.
**Expected impact:** Tăng Answer Relevancy, câu trả lời sát hơn với câu hỏi được đặt.

### Cải tiến 2
**Action:** Tăng golden dataset lên 30+ câu hỏi với các dạng câu hỏi đa dạng hơn (multi-hop, so sánh, yes/no) để đánh giá toàn diện hơn.
**Expected impact:** Đánh giá chính xác hơn hiệu năng pipeline trên nhiều loại câu hỏi.

### Cải tiến 3
**Action:** Thay thế RRF bằng cross-encoder reranker (Jina v2 hoặc Qwen3-Reranker) để rerank dựa trên semantic similarity thực sự.
**Expected impact:** Cải thiện cả Context Precision và Faithfulness nhờ context chất lượng hơn.

