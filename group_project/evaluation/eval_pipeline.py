"""
RAG Evaluation Pipeline.

Sử dụng RAGAS để đánh giá chất lượng RAG pipeline.

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
import os
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Đảm bảo project root nằm trong sys.path để import src.*
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

# Giới hạn số câu hỏi để tránh rate limit (RAGAS gọi LLM rất nhiều lần/metric/câu)
MAX_QUESTIONS = 5


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:MAX_QUESTIONS]


# =============================================================================
# LLM WRAPPER cho RAGAS Judge
# =============================================================================

def _get_ragas_llm():
    """
    Tạo LLM wrapper cho RAGAS judge.

    Dùng cùng API key với RAG pipeline (OpenAI hoặc OpenRouter).
    """
    from langchain_openai import ChatOpenAI

    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Ưu tiên OpenRouter nếu có key hợp lệ
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key and or_key.strip() and not or_key.startswith("sk-or-v1-..."):
        return ChatOpenAI(
            model=model,
            api_key=or_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )

    # Fallback sang OpenAI API key
    oa_key = os.getenv("OPENAI_API_KEY")
    if oa_key and oa_key.strip():
        # Cắt prefix "openai/" nếu có (OpenAI thuần không cần prefix)
        if model.startswith("openai/"):
            model = model.split("/", 1)[-1]
        return ChatOpenAI(
            model=model,
            api_key=oa_key,
            temperature=0,
        )

    raise RuntimeError(
        "Chưa cấu hình API key. Đặt OPENROUTER_API_KEY hoặc OPENAI_API_KEY trong .env"
    )


def _get_ragas_embeddings():
    """Tạo embedding wrapper cho RAGAS (dùng OpenAI embeddings)."""
    from langchain_openai import OpenAIEmbeddings

    oa_key = os.getenv("OPENAI_API_KEY")
    or_key = os.getenv("OPENROUTER_API_KEY")

    if or_key and or_key.strip() and not or_key.startswith("sk-or-v1-..."):
        return OpenAIEmbeddings(
            api_key=or_key,
            base_url="https://openrouter.ai/api/v1",
        )

    if oa_key and oa_key.strip():
        return OpenAIEmbeddings(api_key=oa_key)

    raise RuntimeError("Cần OPENAI_API_KEY hoặc OPENROUTER_API_KEY cho embeddings.")


# =============================================================================
# PIPELINE CONFIGS cho A/B Comparison
# =============================================================================

def _run_pipeline_config_a(query: str, top_k: int = 5) -> tuple[list[dict], str]:
    """
    Config A: Hybrid search + Reranking (full pipeline).

    Gọi generate_with_citation() từ task10 — pipeline đầy đủ:
    semantic search + lexical search + RRF merge + fallback PageIndex + LLM generation.

    Returns:
        (sources, answer)
    """
    from src.task10_generation import generate_with_citation

    result = generate_with_citation(query, top_k=top_k)
    return result["sources"], result["answer"]


def _run_pipeline_config_b(query: str, top_k: int = 5) -> tuple[list[dict], str]:
    """
    Config B: Dense-only (chỉ semantic search, không hybrid/reranking).

    Gọi semantic_search() trực tiếp, bỏ qua lexical search và RRF merge.
    Sau đó dùng cùng LLM generation logic.

    Returns:
        (sources, answer)
    """
    from src.task5_semantic_search import semantic_search
    from src.task10_generation import (
        _build_messages,
        _get_client_and_model,
        TEMPERATURE,
        TOP_P,
        CANNOT_VERIFY,
    )

    chunks = semantic_search(query, top_k=top_k)
    for item in chunks:
        item["source"] = "dense"

    if not chunks:
        return [], CANNOT_VERIFY + "."

    messages, sources = _build_messages(query, chunks)
    client, model = _get_client_and_model()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    answer = response.choices[0].message.content or ""
    return sources, answer


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_eval_data(
    golden_dataset: list[dict],
    pipeline_fn,
    config_name: str,
) -> dict:
    """
    Thu thập dữ liệu evaluation bằng cách chạy pipeline trên golden dataset.

    Args:
        golden_dataset: List of {question, expected_answer, expected_context}
        pipeline_fn: Function(query) -> (sources, answer)
        config_name: Tên config để in log

    Returns:
        dict với keys: question, answer, contexts, ground_truth, raw_sources
    """
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    raw_sources = []  # Lưu sources gốc để phân tích worst performers

    print(f"\n{'='*60}")
    print(f"  Đang chạy pipeline [{config_name}] trên {len(golden_dataset)} câu hỏi...")
    print(f"{'='*60}")

    for i, item in enumerate(golden_dataset):
        q = item["question"]
        print(f"  [{i+1}/{len(golden_dataset)}] {q[:70]}...")

        try:
            sources, answer = pipeline_fn(q)
            contexts = [c.get("content", "") for c in sources]

            eval_data["question"].append(q)
            eval_data["answer"].append(answer)
            eval_data["contexts"].append(contexts if contexts else [""])
            eval_data["ground_truth"].append(item["expected_answer"])
            raw_sources.append(sources)

            # Rate limit safety: đợi giữa các câu hỏi
            time.sleep(2)

        except Exception as e:
            print(f"  ⚠ Lỗi câu hỏi {i+1}: {e}")
            traceback.print_exc()
            # Vẫn thêm data trống để giữ index align
            eval_data["question"].append(q)
            eval_data["answer"].append(f"[ERROR] {e}")
            eval_data["contexts"].append([""])
            eval_data["ground_truth"].append(item["expected_answer"])
            raw_sources.append([])

    eval_data["_raw_sources"] = raw_sources
    return eval_data


# =============================================================================
# RAGAS EVALUATION
# =============================================================================

def evaluate_with_ragas(eval_data: dict) -> dict:
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    Metrics:
        - faithfulness: câu trả lời có bám đúng context không
        - answer_relevancy: câu trả lời có đúng câu hỏi không
        - context_recall: retriever có lấy đủ evidence không
        - context_precision: trong context lấy về, bao nhiêu % thực sự hữu ích

    Args:
        eval_data: dict từ collect_eval_data()

    Returns:
        dict với keys: scores (dict metric->avg_score), per_question (list of dicts)
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset

    # Tạo dataset cho RAGAS (bỏ _raw_sources vì không phải RAGAS column)
    ragas_data = {
        "question": eval_data["question"],
        "answer": eval_data["answer"],
        "contexts": eval_data["contexts"],
        "ground_truth": eval_data["ground_truth"],
    }
    dataset = Dataset.from_dict(ragas_data)

    print("\n  Đang chạy RAGAS evaluation (có thể mất vài phút)...")

    llm = _get_ragas_llm()
    embeddings = _get_ragas_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings,
    )

    # Lấy average scores (hỗ trợ cả dictionary hoặc EvaluationResult object)
    try:
        avg_scores = {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
            "context_recall": float(result["context_recall"]),
            "context_precision": float(result["context_precision"]),
        }
    except (TypeError, KeyError, AttributeError):
        # Fallback nếu object không hỗ trợ index []
        df_result = result.to_pandas()
        avg_scores = {
            "faithfulness": float(df_result["faithfulness"].mean()) if "faithfulness" in df_result else 0.0,
            "answer_relevancy": float(df_result["answer_relevancy"].mean()) if "answer_relevancy" in df_result else 0.0,
            "context_recall": float(df_result["context_recall"].mean()) if "context_recall" in df_result else 0.0,
            "context_precision": float(df_result["context_precision"].mean()) if "context_precision" in df_result else 0.0,
        }

    # Lấy per-question scores
    df = result.to_pandas()
    per_question = df.to_dict("records")

    return {"scores": avg_scores, "per_question": per_question}


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(golden_dataset: list[dict]) -> dict:
    """
    So sánh A/B giữa 2 configs:
    - Config A: hybrid search + reranking (full pipeline)
    - Config B: dense-only (chỉ semantic search, không hybrid/reranking)

    Returns:
        dict {config_name: {scores, per_question, eval_data}}
    """
    configs = {
        "Config A (hybrid + rerank)": _run_pipeline_config_a,
        "Config B (dense-only)": _run_pipeline_config_b,
    }

    results = {}
    for config_name, pipeline_fn in configs.items():
        print(f"\n{'#'*60}")
        print(f"  Evaluating: {config_name}")
        print(f"{'#'*60}")

        # Step 1: Thu thập dữ liệu
        eval_data = collect_eval_data(golden_dataset, pipeline_fn, config_name)

        # Step 2: Chạy RAGAS evaluation
        try:
            ragas_result = evaluate_with_ragas(eval_data)
            results[config_name] = {
                "scores": ragas_result["scores"],
                "per_question": ragas_result["per_question"],
                "eval_data": eval_data,
            }
            print(f"\n  ✓ {config_name} — scores: {ragas_result['scores']}")
        except Exception as e:
            print(f"\n  ⚠ RAGAS evaluation failed cho {config_name}: {e}")
            traceback.print_exc()
            results[config_name] = {
                "scores": {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_recall": 0.0,
                    "context_precision": 0.0,
                },
                "per_question": [],
                "eval_data": eval_data,
            }

    return results


# =============================================================================
# Export Results
# =============================================================================

def _find_worst_performers(comparison: dict) -> list[dict]:
    """
    Tìm 3 câu hỏi worst performers từ Config A (pipeline chính).

    Xếp theo trung bình faithfulness + relevancy thấp nhất.
    """
    # Lấy per_question từ Config A
    config_a_key = [k for k in comparison if "Config A" in k]
    if not config_a_key or not comparison[config_a_key[0]].get("per_question"):
        return []

    per_q = comparison[config_a_key[0]]["per_question"]

    scored = []
    for item in per_q:
        faith = item.get("faithfulness", 0.0) or 0.0
        rel = item.get("answer_relevancy", 0.0) or 0.0
        recall = item.get("context_recall", 0.0) or 0.0
        avg = (faith + rel + recall) / 3.0
        scored.append({
            "question": item.get("user_input", item.get("question", "N/A")),
            "faithfulness": faith,
            "relevance": rel,
            "recall": recall,
            "avg_score": avg,
        })

    scored.sort(key=lambda x: x["avg_score"])
    return scored[:3]


def _determine_failure_stage(item: dict) -> str:
    """Xác định giai đoạn lỗi dựa trên metric scores."""
    faith = item.get("faithfulness", 0.0)
    rel = item.get("relevance", 0.0)
    recall = item.get("recall", 0.0)

    if recall < 0.5:
        return "Retrieval"
    if faith < 0.5:
        return "Generation"
    if rel < 0.5:
        return "Relevance"
    return "Unknown"


def _determine_root_cause(item: dict) -> str:
    """Đề xuất root cause dựa trên pattern metric."""
    faith = item.get("faithfulness", 0.0)
    recall = item.get("recall", 0.0)
    rel = item.get("relevance", 0.0)

    if recall < 0.5:
        return "Retriever không tìm đủ context liên quan"
    if faith < 0.5:
        return "LLM hallucinate ngoài context"
    if rel < 0.5:
        return "Câu trả lời không sát câu hỏi"
    return "Cần phân tích thêm"


def export_results(comparison: dict):
    """
    Export evaluation results ra results.md — giữ nguyên cấu trúc template.

    Sections:
        1. Framework sử dụng
        2. Overall Scores (bảng 4 metrics × 2 configs + Delta)
        3. A/B Comparison Analysis
        4. Worst Performers (Bottom 3)
        5. Recommendations
    """
    config_a_key = [k for k in comparison if "Config A" in k][0]
    config_b_key = [k for k in comparison if "Config B" in k][0]

    scores_a = comparison[config_a_key]["scores"]
    scores_b = comparison[config_b_key]["scores"]

    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }

    # --- Section 1: Framework ---
    content = "# RAG Evaluation Results\n\n"
    content += "## Framework sử dụng\n\n"
    content += "> RAGAS (Retrieval Augmented Generation Assessment)\n\n"
    content += "---\n\n"

    # --- Section 2: Overall Scores ---
    content += "## Overall Scores\n\n"
    content += "| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |\n"
    content += "|--------|---------------------------|----------------------|---|\n"

    sum_a, sum_b = 0.0, 0.0
    for m in metrics:
        sa = scores_a.get(m, 0.0) or 0.0
        sb = scores_b.get(m, 0.0) or 0.0
        delta = sa - sb
        sign = "+" if delta >= 0 else ""
        content += f"| {metric_labels[m]} | {sa:.4f} | {sb:.4f} | {sign}{delta:.4f} |\n"
        sum_a += sa
        sum_b += sb

    avg_a = sum_a / len(metrics)
    avg_b = sum_b / len(metrics)
    avg_delta = avg_a - avg_b
    sign = "+" if avg_delta >= 0 else ""
    content += f"| **Average** | **{avg_a:.4f}** | **{avg_b:.4f}** | **{sign}{avg_delta:.4f}** |\n"
    content += "\n---\n\n"

    # --- Section 3: A/B Comparison Analysis ---
    content += "## A/B Comparison Analysis\n\n"

    content += "**Config A:**\n"
    content += "> Pipeline đầy đủ: Semantic Search + Lexical Search → RRF Fusion → "
    content += "Reranking → PageIndex Fallback → LLM Generation. "
    content += "Kết hợp cả dense (vector) lẫn sparse (BM25) retrieval để đa dạng hoá kết quả.\n\n"

    content += "**Config B:**\n"
    content += "> Dense-only: Chỉ sử dụng Semantic Search (cosine similarity) trực tiếp, "
    content += "không qua Lexical Search, không RRF merge, không reranking. "
    content += "Pipeline tối giản để so sánh baseline.\n\n"

    # Kết luận dựa trên scores
    if avg_a > avg_b:
        winner = "Config A (Hybrid + Rerank)"
        reason = (
            f"Config A vượt trội với điểm trung bình cao hơn {avg_delta:.4f}. "
            f"Hybrid search kết hợp dense và sparse retrieval giúp cải thiện context quality, "
            f"dẫn đến câu trả lời chính xác hơn."
        )
    elif avg_b > avg_a:
        winner = "Config B (Dense-only)"
        reason = (
            f"Config B đạt điểm trung bình cao hơn {abs(avg_delta):.4f}. "
            f"Semantic search thuần cho kết quả tốt hơn trong trường hợp này, "
            f"có thể do corpus nhỏ và embedding model đã capture đủ semantic meaning."
        )
    else:
        winner = "Ngang nhau"
        reason = "Hai configs cho kết quả tương đương, không có sự khác biệt đáng kể."

    content += "**Kết luận:**\n"
    content += f"> {winner} cho kết quả tốt hơn. {reason}\n\n"
    content += "---\n\n"

    # --- Section 4: Worst Performers ---
    content += "## Worst Performers (Bottom 3)\n\n"
    content += "| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |\n"
    content += "|---|----------|-------------|-----------|--------|---------------|------------|\n"

    worst = _find_worst_performers(comparison)
    for i, item in enumerate(worst, 1):
        q = item["question"][:50] + "..." if len(item["question"]) > 50 else item["question"]
        stage = _determine_failure_stage(item)
        cause = _determine_root_cause(item)
        content += (
            f"| {i} | {q} | {item['faithfulness']:.2f} | "
            f"{item['relevance']:.2f} | {item['recall']:.2f} | "
            f"{stage} | {cause} |\n"
        )

    # Pad nếu ít hơn 3 worst performers
    for i in range(len(worst) + 1, 4):
        content += f"| {i} | N/A | N/A | N/A | N/A | N/A | N/A |\n"

    content += "\n---\n\n"

    # --- Section 5: Recommendations ---
    content += "## Recommendations\n\n"

    # Đề xuất dựa trên metric thấp nhất
    min_metric = min(scores_a, key=lambda m: scores_a.get(m, 0.0) or 0.0)
    recommendations = []

    if scores_a.get("context_recall", 1.0) < 0.7:
        recommendations.append({
            "title": "Cải thiện Retrieval Coverage",
            "action": "Thêm chunk overlap lớn hơn (200-300 tokens) hoặc sử dụng "
                      "hierarchical chunking để không bỏ sót context quan trọng ở biên chunk.",
            "impact": "Tăng Context Recall lên trên 0.7, giúp LLM có đủ evidence để trả lời chính xác.",
        })

    if scores_a.get("faithfulness", 1.0) < 0.7:
        recommendations.append({
            "title": "Giảm Hallucination",
            "action": "Giảm temperature xuống 0.1-0.2 và thêm instruction rõ ràng hơn trong "
                      "system prompt yêu cầu LLM CHỈ trả lời dựa trên context.",
            "impact": "Tăng Faithfulness, giảm tỷ lệ LLM bịa thông tin ngoài context.",
        })

    if scores_a.get("context_precision", 1.0) < 0.7:
        recommendations.append({
            "title": "Nâng cao Context Precision",
            "action": "Áp dụng cross-encoder reranking (Jina Reranker v2) thay vì chỉ RRF để "
                      "lọc bỏ chunks không liên quan trước khi đưa vào LLM.",
            "impact": "Tăng Context Precision, giảm noise trong context giúp LLM tập trung hơn.",
        })

    if scores_a.get("answer_relevancy", 1.0) < 0.7:
        recommendations.append({
            "title": "Cải thiện Answer Relevancy",
            "action": "Tinh chỉnh prompt template để yêu cầu LLM trả lời trực tiếp câu hỏi "
                      "trước, sau đó mới bổ sung chi tiết.",
            "impact": "Tăng Answer Relevancy, câu trả lời sát hơn với câu hỏi được đặt.",
        })

    # Đảm bảo luôn có ít nhất 3 recommendations
    default_recs = [
        {
            "title": "Mở rộng Golden Dataset",
            "action": "Tăng golden dataset lên 30+ câu hỏi với các dạng câu hỏi đa dạng hơn "
                      "(multi-hop, so sánh, yes/no) để đánh giá toàn diện hơn.",
            "impact": "Đánh giá chính xác hơn hiệu năng pipeline trên nhiều loại câu hỏi.",
        },
        {
            "title": "Thử nghiệm Cross-Encoder Reranking",
            "action": "Thay thế RRF bằng cross-encoder reranker (Jina v2 hoặc Qwen3-Reranker) "
                      "để rerank dựa trên semantic similarity thực sự.",
            "impact": "Cải thiện cả Context Precision và Faithfulness nhờ context chất lượng hơn.",
        },
        {
            "title": "Fine-tune Chunking Strategy",
            "action": "Thử nghiệm semantic chunking hoặc document-aware chunking thay vì "
                      "fixed-size chunking để giữ nguyên ngữ cảnh trong mỗi chunk.",
            "impact": "Cải thiện Context Recall và giảm trường hợp context bị cắt giữa chừng.",
        },
    ]

    # Lọc bỏ trùng title
    existing_titles = {r["title"] for r in recommendations}
    for rec in default_recs:
        if rec["title"] not in existing_titles and len(recommendations) < 3:
            recommendations.append(rec)

    for i, rec in enumerate(recommendations[:3], 1):
        content += f"### Cải tiến {i}\n"
        content += f"**Action:** {rec['action']}\n"
        content += f"**Expected impact:** {rec['impact']}\n\n"

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\n✓ Results exported to {RESULTS_PATH}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Evaluation Pipeline — RAGAS")
    print("=" * 60)

    # Load golden dataset (giới hạn MAX_QUESTIONS câu)
    golden_dataset = load_golden_dataset()
    print(f"\nLoaded {len(golden_dataset)} test cases (max {MAX_QUESTIONS})")

    # A/B Comparison: chạy cả 2 configs và evaluate
    comparison = compare_configs(golden_dataset)

    # Export kết quả ra results.md
    export_results(comparison)

    # In tóm tắt
    print("\n" + "=" * 60)
    print("  TÓM TẮT KẾT QUẢ")
    print("=" * 60)
    for config_name, data in comparison.items():
        scores = data["scores"]
        avg = sum(v for v in scores.values() if v) / max(len(scores), 1)
        print(f"\n  {config_name}:")
        for metric, score in scores.items():
            print(f"    {metric}: {score:.4f}")
        print(f"    → Average: {avg:.4f}")

    print(f"\n✓ Chi tiết tại: {RESULTS_PATH}")
