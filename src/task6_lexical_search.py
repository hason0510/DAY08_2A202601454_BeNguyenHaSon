"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""
"""
Task 6 — Lexical Search Module (BM25).

Cài đặt trước khi chạy:
    pip install rank-bm25 numpy
"""

import sys
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi

# Xử lý import để tương thích cả khi chạy trực tiếp file và khi chạy qua module
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent))
    from task4_chunking_indexing import load_documents, chunk_documents
else:
    from src.task4_chunking_indexing import load_documents, chunk_documents

# Biến toàn cục lưu trữ corpus và model BM25
CORPUS: list[dict] = []
bm25_model = None


def init_bm25():
    """
    Nạp dữ liệu thật từ Task 4 và xây dựng BM25 Index trên các chunks.
    Điều này đảm bảo Lexical Search và Semantic Search dùng chung một tập dữ liệu,
    giúp việc reranking (RRF) ở Task 7 chính xác.
    """
    global CORPUS, bm25_model
    
    print("Đang khởi tạo BM25 Index từ dữ liệu gốc...")
    
    # Load và chunk tài liệu giống hệt cách ChromaDB làm ở Task 4
    docs = load_documents()
    if not docs:
        print("⚠ Không tìm thấy tài liệu nào để build BM25. Hãy kiểm tra data/standardized/.")
        return
        
    CORPUS = chunk_documents(docs)
    
    # Tokenize đơn giản bằng cách tách từ (split) và chuyển thành chữ thường (lower)
    tokenized_corpus = [doc["content"].lower().split() for doc in CORPUS]
    bm25_model = BM25Okapi(tokenized_corpus)
    print(f"✓ BM25 Index đã sẵn sàng với {len(CORPUS)} chunks.")


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global bm25_model
    
    # Tự động init nếu chưa có model
    if bm25_model is None:
        init_bm25()
        
    # Nếu vẫn None (do không có data), trả về list rỗng
    if bm25_model is None:
        return []

    tokenized_query = query.lower().split()
    scores = bm25_model.get_scores(tokenized_query)
    
    # Lấy ra index của top_k kết quả có điểm cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        # Chỉ lấy những kết quả có điểm BM25 > 0 (tức là có chứa ít nhất 1 từ khóa)
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
            
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING TASK 6: BM25 LEXICAL SEARCH (REAL DATA)")
    print("=" * 60)
    
    test_query = "phương thức thanh toán shopee"
    print(f"\nQuery: '{test_query}'\n")
    
    results = lexical_search(test_query, top_k=3)
    
    if not results:
        print("Không tìm thấy kết quả phù hợp hoặc chưa có dữ liệu.")
    else:
        for r in results:
            print(f"Score: [{r['score']:.3f}]")
            print(f"Metadata: {r['metadata']}")
            print(f"Content: {r['content'][:150]}...\n")
            print("-" * 40)
