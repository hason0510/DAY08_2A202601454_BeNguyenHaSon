"""
Task 4 — Chunking & Indexing vào Vector Store.

Kiến trúc và Lý do kỹ thuật:

[1] Chunking — RecursiveCharacterTextSplitter:
    Thuật toán này cắt văn bản theo hệ thống PHÂN CẤP (recursive), ưu tiên cắt tại
    các dấu phân cách tự nhiên theo đú́ng ngữ pháp: \n\n (hết đoạn) -> \n -> . -> ' '.
    Kết quả: mỗi chunk luôn là một đơn vị ngữ nghĩa hoàn chỉnh (không bị cắt giữa câu).

[2] Chunk Size = 800 ký tự:
    Chính sách thương mại điện tử có cấu trúc mục rõ ràng (điều khoản 1, 2, 3...).
    800 ký tự (~150-200 từ) bão hòa được 1-2 điều khoản hoàn chỉnh mà không bị
    loãng ngữ nghĩa. Size nhỏ (< 400): bị đứt ý. Size lớn (> 1500): chunk chứa
    quá nhiều chủ đề khác nhau, vector bị 'trung bình hóa', similarity giảm mạnh.

[3] Chunk Overlap = 100 ký tự:
    Phần 'gối' giữa 2 chunk liên tiếp. Giả sử 1 điều khoản dài bị cắt ó giữa, 100
    ký tự (~2 câu) đủ để chunk thứ 2 vẫn giữ lại được ngữ cảnh chuyển giao.
    Tránh trưởng hợp hệ thống đọc thấy 'Vì vậy...' nhưng không biết 'vì' gì.

[4] Embedding Model — Hỗ trợ 2 provider, chọn qua biến EMBEDDING_PROVIDER trong .env:
    a) EMBEDDING_PROVIDER=openai  → text-embedding-3-small (1536 dim)
       Ưu điểm: Không tải model về máy, nhanh, rẻ ($0.02/1M tokens)
       Cần: OPENAI_API_KEY trong .env
    b) EMBEDDING_PROVIDER=sentence_transformers  → BAAI/bge-m3 (1024 dim)
       Ưu điểm: Chạy offline hoàn toàn, tối ưu cho tiếng Việt + tiếng Anh
       Cần: pip install sentence-transformers (tải ~1.5GB về máy lần đầu)
    ⚠️  Nếu đổi provider: XÓA thư mục chroma_db/ rồi reindex lại vì dimension khác nhau!
    Mặc định hiện tại: openai (checkpoint yêu cầu bge-m3 → đổi trong .env nếu cần)

[5] Vector Store — ChromaDB:
    - Persistent local (không cần Docker, không cần kết nối internet sau khi index)
    - Hỗ trợ Cosine Similarity search native (metadata={'hnsw:space': 'cosine'})
    - Thiết lập collection 1 lần, tất cả thành viên nhóm có thể query ngay lập tức
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load biến môi trường từ .env
load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# =============================================================================
# CONFIGURATION — Lý do kỹ thuật được giải thích trong module docstring phía trên
# =============================================================================

# [2] Chunk Size: 800 ký tự bão hòa 1-2 điều khoản chính sách, không bị loãng ngữ nghĩa
CHUNK_SIZE = 800
# [3] Chunk Overlap: 100 ký tự giữ ngữ cảnh chuyển giao giữa các chunk liên tiếp
CHUNK_OVERLAP = 100
# [1] Chunking Method: Recursive bảo đảm không cắt giữa câu, giữ nguyên ngữ nghĩa
CHUNKING_METHOD = "recursive"

# [4] Embedding Provider: đọc từ .env để cả nhóm đổi model mà không sửa code
# Đặt EMBEDDING_PROVIDER=sentence_transformers trong .env để dùng BAAI/bge-m3 (1024 dim)
# Đặt EMBEDDING_PROVIDER=openai trong .env để dùng text-embedding-3-small (1536 dim)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")

if EMBEDDING_PROVIDER == "sentence_transformers":
    EMBEDDING_MODEL = "BAAI/bge-m3"
    EMBEDDING_DIM = 1024
else:
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIM = 1536

# [5] Vector Store: ChromaDB — local persistent, không cần Docker
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "ecommerce_support_docs"

# =============================================================================
# HELPERS CHO TASK 5 TÁI SỬ DỤNG
# =============================================================================
def get_embedding_model():
    """Trả về embedding model theo EMBEDDING_PROVIDER trong .env."""
    if EMBEDDING_PROVIDER == "sentence_transformers":
        # Cần: pip install sentence-transformers
        # Lần đầu chạy sẽ tự tải BAAI/bge-m3 về ~/.cache (~1.5GB)
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # Chuẩn hóa về unit vector, tối ưu Cosine
        )
    else:
        # Cần: OPENAI_API_KEY trong .env
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)

def get_chroma_collection():
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection

# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.
    """
    documents = []
    # Đảm bảo thư mục tồn tại
    STANDARDIZED_DIR.mkdir(parents=True, exist_ok=True)
    
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        # Phân loại dựa trên tên thư mục chứa nó nếu có
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents

def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo RecursiveCharacterTextSplitter
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn (openai hoặc sentence_transformers).
    """
    model = get_embedding_model()
    texts = [c["content"] for c in chunks]

    print(f"Đang tạo embedding bằng [{EMBEDDING_PROVIDER.upper()}] model: {EMBEDDING_MODEL}...")
    embeddings = model.embed_documents(texts)
    
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks

def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào ChromaDB.
    """
    if not chunks:
        print("Không có chunks nào để index.")
        return
        
    collection = get_chroma_collection()
    
    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    
    print("Đang lưu vào ChromaDB...")
    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )

def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")
    
    if not docs:
        print("⚠️ Chưa có file nào trong data/standardized/. Hãy chờ Task 3 hoàn thành hoặc tạo file fake để test.")
        return

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
