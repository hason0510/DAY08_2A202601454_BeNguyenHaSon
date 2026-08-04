"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

"""
Task 8 — PageIndex Vectorless RAG.

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Lưu trữ danh sách Document ID sau khi upload thành công để dùng cho việc query
UPLOADED_DOC_IDS = []


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    global UPLOADED_DOC_IDS
    
    if not PAGEINDEX_API_KEY:
        print("⚠ Missing PAGEINDEX_API_KEY. Bỏ qua bước upload.")
        return

    try:
        from pageindex.client import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    except ImportError:
        print("⚠ Chưa cài đặt thư viện pageindex. Hãy chạy: pip install pageindex")
        return

    # Quét toàn bộ file markdown trong thư mục
    if STANDARDIZED_DIR.exists():
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            print(f"Đang upload: {md_file.name}...")
            try:
                # Gửi tài liệu lên PageIndex
                resp = client.submit_document(str(md_file))
                
                # Trích xuất doc_id từ response
                doc_id = resp.get("doc_id") or resp.get("id")
                if doc_id:
                    UPLOADED_DOC_IDS.append(doc_id)
                    print(f"  ✓ Uploaded thành công: {md_file.name} -> {doc_id}")
            except Exception as e:
                print(f"  ✗ Lỗi khi upload {md_file.name}: {e}")
    else:
        print(f"⚠ Không tìm thấy thư mục {STANDARDIZED_DIR}. Vui lòng kiểm tra lại Task 3.")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'
        }
    """
    if not PAGEINDEX_API_KEY or not UPLOADED_DOC_IDS:
        print("⚠ Không thể search: Thiếu API Key hoặc chưa có tài liệu nào được upload.")
        return []

    try:
        from pageindex.client import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    except ImportError:
        return []

    results = []
    
    # Thực hiện search trên tài liệu đầu tiên trong danh sách đã upload
    target_doc_id = UPLOADED_DOC_IDS[0]
    
    try:
        # Gửi query
        resp = client.submit_query(doc_id=target_doc_id, query=query)
        retrieval_id = resp.get("retrieval_id") or resp.get("id")
        
        if not retrieval_id:
            print("  ✗ Không nhận được retrieval_id từ API.")
            return []

        # Polling (chờ API xử lý xong) - thử tối đa 10 lần, mỗi lần cách nhau 2 giây
        retrieval_data = None
        for _ in range(10):
            retrieval_data = client.get_retrieval(retrieval_id)
            if retrieval_data.get("status") == "completed":
                break
            time.sleep(2)

        if not retrieval_data or retrieval_data.get("status") != "completed":
            print("  ✗ Query timeout hoặc xử lý thất bại từ server.")
            return []

        # Parse kết quả từ "retrieved_nodes"
        # Vì PageIndex không trả về điểm similarity cụ thể, ta sẽ tự gán điểm giảm dần theo thứ hạng
        current_score = 0.95
        
        for node in retrieval_data.get("retrieved_nodes", []):
            for group in node.get("relevant_contents", []):
                for item in group:
                    if len(results) >= top_k:
                        break
                        
                    results.append({
                        "content": item.get("relevant_content", ""),
                        "score": round(current_score, 3),
                        "metadata": {"section": item.get("section_title", "N/A")},
                        "source": "pageindex",
                    })
                    current_score -= 0.05
                    
            if len(results) >= top_k:
                break
                
    except Exception as e:
        print(f"  ✗ Quá trình tìm kiếm PageIndex gặp lỗi: {e}")

    return results[:top_k]


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING TASK 8: PAGEINDEX VECTORLESS RAG")
    print("=" * 60)
    
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env trước khi test.")
        print("  Đăng ký lấy key tại: https://pageindex.ai/")
    else:
        print("1. Uploading documents...")
        upload_documents()

        print("\n2. Test query:")
        test_query = "danh sách sản phẩm cấm đăng bán"
        print(f"Query: '{test_query}'\n")
        
        results = pageindex_search(test_query, top_k=3)
        
        if not results:
            print("Không có kết quả trả về.")
        else:
            for i, r in enumerate(results, 1):
                print(f"Result {i}:")
                print(f"  - Score: {r['score']:.3f} | Source: {r['source']}")
                print(f"  - Metadata: {r['metadata']}")
                print(f"  - Content: {r['content'][:150]}...")
                print("-" * 40)
