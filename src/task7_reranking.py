"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

"""
Task 7 — Reranking Module.

Sử dụng RRF (Reciprocal Rank Fusion) để gộp và xếp hạng lại kết quả từ nhiều 
nguồn tìm kiếm (Semantic + Lexical) mà không cần dùng đến API Key.
"""

from typing import Optional


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """Rerank candidates sử dụng cross-encoder model."""
    raise NotImplementedError("Implement rerank_cross_encoder nếu dùng Jina/Qwen")


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse."""
    raise NotImplementedError("Implement rerank_mmr nếu dùng MMR")


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker, ví dụ: [dense_results, sparse_results])
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores = {}  
    content_map = {}  
    
    # Duyệt qua từng danh sách kết quả (từ BM25, từ Vector Store, v.v.)
    for ranked_list in ranked_lists:
        # rank bắt đầu từ 1
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            
            # Cộng dồn điểm RRF dựa trên thứ hạng
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            
            # Lưu trữ toàn bộ dữ liệu của document để trả về sau
            if key not in content_map:
                content_map[key] = item.copy()
    
    # Sắp xếp lại dựa trên điểm RRF từ cao xuống thấp
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content]
        item["score"] = score  # Ghi đè điểm cũ bằng điểm RRF
        results.append(item)
    
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        # RRF vốn để gộp NHIỀU ranked list. Interface chung này chỉ nhận một
        # danh sách candidates, nên bọc thành list-của-list: RRF với 1 ranker
        # giữ nguyên thứ tự và gán lại điểm theo thứ hạng 1/(k + rank).
        # Task 9 vẫn gọi thẳng rerank_rrf([dense_results, sparse_results]).
        return rerank_rrf([candidates], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING TASK 7: RRF RERANKING")
    print("=" * 60)
    
    # Dữ liệu giả định trả về từ Semantic Search (Mô hình Vector)
    dense_dummy = [
        {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 0.85, "metadata": {"source": "dense"}},
        {"content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam", "score": 0.72, "metadata": {"source": "dense"}},
        {"content": "Quy định đăng bán sản phẩm dành cho người bán", "score": 0.65, "metadata": {"source": "dense"}},
    ]
    
    # Dữ liệu giả định trả về từ Lexical Search (BM25)
    sparse_dummy = [
        {"content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam", "score": 2.15, "metadata": {"source": "sparse"}},
        {"content": "Chính sách bảo mật thông tin khách hàng", "score": 1.85, "metadata": {"source": "sparse"}},
        {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 1.10, "metadata": {"source": "sparse"}},
    ]

    print("Đang tiến hành dung hợp (Fusion) kết quả từ Dense và Sparse...\n")
    
    # Chạy RRF gộp 2 danh sách lại
    results = rerank_rrf([dense_dummy, sparse_dummy], top_k=4)
    
    for i, r in enumerate(results, 1):
        print(f"Rank {i}:")
        print(f"  - Content: {r['content']}")
        print(f"  - RRF Score: {r['score']:.5f}")
        print("-" * 40)
