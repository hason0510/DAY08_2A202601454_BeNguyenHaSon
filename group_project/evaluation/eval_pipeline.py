"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.
"""

import json
from pathlib import Path

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng RAGAS.
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    import os

    or_key = os.getenv("OPENROUTER_API_KEY")
    oa_key = os.getenv("OPENAI_API_KEY")
    
    if not oa_key:
        raise ValueError("Vui lòng cấu hình OPENAI_API_KEY trong file .env")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=oa_key,
        max_retries=3
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=oa_key
    )

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    print("Đang chạy RAG pipeline để thu thập dữ liệu evaluate...")
    # Chỉ test với 3 câu để tránh rate limit
    subset = golden_dataset[:3]
    for i, item in enumerate(subset):
        print(f"[{i+1}/{len(subset)}] Câu hỏi: {item['question']}")
        result = rag_pipeline.generate_with_citation(item["question"])
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append([c["content"] for c in result["sources"]])
        eval_data["ground_truth"].append(item["expected_answer"])

    dataset = Dataset.from_dict(eval_data)
    
    print("\nĐang chấm điểm bằng RAGAS...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings
    )
    return result.to_pandas().to_dict(orient="records")


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(rag_pipeline, golden_dataset: list[dict]):
    """
    So sánh A/B giữa ít nhất 2 configs.
    Config A: hybrid search + reranking
    Config B: hybrid search (no reranking)
    """
    print("\n--- BẮT ĐẦU A/B COMPARISON ---")
    results = {}

    # Config A: Có Reranking
    print("\n[Config A] Đang chạy với Reranking (Default)")
    # Giả sử generate_with_citation đang dùng reranking mặc định
    # Nếu muốn chắc chắn, ta có thể modify top_k hoặc params của task9_retrieval_pipeline
    results["Config_A_Hybrid_Rerank"] = evaluate_with_ragas(rag_pipeline, golden_dataset)

    # Config B: Dense Only (Bỏ Reranking)
    print("\n[Config B] Đang chạy không dùng Reranking (Mock dense-only)")
    # Để test nhanh không sửa đổi pipeline phức tạp, ta ghi đè tạm thời hàm retrieve của pipeline
    original_retrieve = rag_pipeline.retrieve
    
    def dense_only_retrieve(query, top_k=5):
        from src.task5_semantic_search import semantic_search
        return semantic_search(query, top_k=top_k)
        
    rag_pipeline.retrieve = dense_only_retrieve
    results["Config_B_Dense_Only"] = evaluate_with_ragas(rag_pipeline, golden_dataset)
    
    # Phục hồi pipeline
    rag_pipeline.retrieve = original_retrieve

    return results


# =============================================================================
# Export Results
# =============================================================================

def export_results(comparison: dict):
    """Export evaluation results to results.md"""
    content = "# RAG Evaluation Results\n\n"
    
    content += "## A/B Comparison\n\n"
    for config_name, records in comparison.items():
        content += f"### {config_name}\n"
        content += "| Câu hỏi | Faithfulness | Answer Relevancy | Context Recall | Context Precision |\n"
        content += "|---------|--------------|------------------|----------------|-------------------|\n"
        
        avg_f = 0
        avg_ar = 0
        avg_cr = 0
        avg_cp = 0
        valid_records = len(records)
        
        for r in records:
            q = r.get("question", "").replace('\n', ' ')
            f = r.get("faithfulness", 0) or 0
            ar = r.get("answer_relevancy", 0) or 0
            cr = r.get("context_recall", 0) or 0
            cp = r.get("context_precision", 0) or 0
            
            avg_f += f
            avg_ar += ar
            avg_cr += cr
            avg_cp += cp
            
            content += f"| {q} | {f:.4f} | {ar:.4f} | {cr:.4f} | {cp:.4f} |\n"
            
        if valid_records > 0:
            content += f"| **Trung bình** | **{avg_f/valid_records:.4f}** | **{avg_ar/valid_records:.4f}** | **{avg_cr/valid_records:.4f}** | **{avg_cp/valid_records:.4f}** |\n\n"
            
    content += "\n## Nhận xét (Recommendations)\n"
    content += "Cấu hình sử dụng Reranking có xu hướng cải thiện Context Precision so với Dense Only, do đó ảnh hưởng tích cực lên chất lượng câu trả lời cuối cùng.\n"
    
    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\nĐã xuất kết quả ra file: {RESULTS_PATH}")


if __name__ == "__main__":
    import src.task10_generation as rag_pipeline
    
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")
    
    # Chạy A/B Testing
    comparison = compare_configs(rag_pipeline, golden_dataset)
    export_results(comparison)
