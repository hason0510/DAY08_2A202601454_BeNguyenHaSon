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

from rank_bm25 import BM25Okapi
import numpy as np
from pathlib import Path

def _load_corpus_from_standardized():
    docs = []
    standardized_dir = Path(__file__).parent.parent / "data" / "standardized"
    for md_file in standardized_dir.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        # Dummy customer_role
        role = "buyer"
        if "seller" in md_file.name: role = "seller"
        elif "privacy" in md_file.name: role = "both"
        docs.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type, "customer_role": role}
        })
    return docs

CORPUS = _load_corpus_from_standardized()
_tokenized_corpus = [doc["content"].lower().split() for doc in CORPUS]
bm25 = BM25Okapi(_tokenized_corpus) if _tokenized_corpus else None

def build_bm25_index(corpus: list[dict]):
    global CORPUS, bm25
    CORPUS = corpus
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    if not bm25:
        return []
        
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
