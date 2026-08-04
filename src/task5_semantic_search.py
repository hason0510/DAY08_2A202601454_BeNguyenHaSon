"""
Task 5 — Semantic Search Module.

Kỹ thuật cốt lõi:
    - Dense Retrieval (Semantic Search): Mã hóa cả câu query thành 1 vector dày đặc
      bằng cùng embedding model đã dùng lúc indexing (text-embedding-3-small),
      sau đó tính Cosine Similarity giữa query vector và toàn bộ chunk vector trong DB.
      Khác với BM25 (chỉ khớp từ khóa), semantic search hiểu được NGHĨA của câu —
      ví dụ query 'làm sao hoàn tiền' vẫn tìm được chunk chứa 'chính sách trả hàng'.

    - HyDE (Hypothetical Document Embeddings): Thay vì embed câu hỏi ngắn ngủn,
      dùng LLM sinh ra một 'câu trả lời giả định' dài hơn, đặc thù hơn, rồi embed
      câu trả lời đó. Về mặt toán học, vector của 'câu trả lời giả định' sẽ nằm
      gần hơn với vector của 'tài liệu thực' trong không gian embedding so với
      vector của 'câu hỏi' — vì cả hai đều là văn phong khẳng định, cùng miền ngữ nghĩa.
      Theo nghiên cứu gốc (Gao et al., 2022), HyDE cải thiện recall lên 5-10%.
"""
import sys
from pathlib import Path

# Tương thích cả khi chạy `python src/task5...` lẫn `pytest tests/`
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent))
    from task4_chunking_indexing import get_embedding_model, get_chroma_collection
else:
    from src.task4_chunking_indexing import get_embedding_model, get_chroma_collection

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os

def semantic_search_basic(query: str, top_k: int = 10) -> list[dict]:
    model = get_embedding_model()
    # Với LangChain OpenAI embeddings, dùng hàm embed_query cho 1 câu text
    query_vector = model.embed_query(query)
    
    collection = get_chroma_collection()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    
    output = []
    # ChromaDB trả về mảng 2 chiều (bởi vì ta có thể query nhiều câu cùng lúc)
    if results and results["documents"] and len(results["documents"][0]) > 0:
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            # Cosine distance chuyển sang similarity score (0 đến 1)
            score = max(0.0, 1.0 - dist)
            output.append({"content": doc, "score": round(score, 4), "metadata": meta})
    
    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]

def generate_hypothetical_answer(query: str) -> str:
    """HyDE: Dùng LLM tạo câu trả lời giả định"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Bạn là nhân viên CSKH sàn thương mại điện tử. Hãy viết một đoạn văn (khoảng 3 câu) đóng vai trò là câu trả lời giả định cho vấn đề sau:\nKhách hàng hỏi: {query}\nTrả lời giả định:"
    )
    chain = prompt | llm
    response = chain.invoke({"query": query})
    return response.content

def semantic_search(query: str, top_k: int = 10, use_hyde: bool = False) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa (Hỗ trợ cờ HyDE).
    """
    if use_hyde:
        print(f"\n🔄 [HyDE] Đang tạo câu trả lời giả định cho: '{query}'...")
        fake_answer = generate_hypothetical_answer(query)
        print(f"💡 Fake answer: {fake_answer}\n")
        
        # Đem fake answer đi tìm kiếm vector thay vì câu hỏi gốc
        return semantic_search_basic(fake_answer, top_k)
    else:
        return semantic_search_basic(query, top_k)

if __name__ == "__main__":
    print("=== TEST BASIC SEARCH ===")
    results_basic = semantic_search("làm sao để hoàn tiền", top_k=2, use_hyde=False)
    for r in results_basic:
        print(f"[{r['score']:.3f}] {r['content'][:80]}...")

    print("\n=== TEST HYDE SEARCH (BONUS) ===")
    results_hyde = semantic_search("làm sao để hoàn tiền", top_k=2, use_hyde=True)
    for r in results_hyde:
        print(f"[{r['score']:.3f}] {r['content'][:80]}...")
