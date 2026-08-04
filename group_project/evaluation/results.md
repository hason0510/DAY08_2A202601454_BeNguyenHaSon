# RAG Evaluation Results

## A/B Comparison

### Config_A_Hybrid_Rerank
| Câu hỏi | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|---------|--------------|------------------|----------------|-------------------|
| Shopee hỗ trợ những phương thức thanh toán nào? | 0.9091 | 0.7681 | 1.0000 | 0.8042 |
| Tôi có thể thanh toán đơn hàng trên Shopee bằng Apple Pay không? Giới hạn giá trị đơn hàng là bao nhiêu? | 1.0000 | 0.6912 | 1.0000 | 0.8042 |
| Phương thức trả góp bằng Thẻ tín dụng có áp dụng cho đơn hàng quốc tế không? | 1.0000 | 0.4378 | 1.0000 | 0.5000 |
| **Trung bình** | **0.9697** | **0.6324** | **1.0000** | **0.7028** |

### Config_B_Dense_Only
| Câu hỏi | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|---------|--------------|------------------|----------------|-------------------|
| Shopee hỗ trợ những phương thức thanh toán nào? | 0.9091 | 0.7681 | 1.0000 | 0.7500 |
| Tôi có thể thanh toán đơn hàng trên Shopee bằng Apple Pay không? Giới hạn giá trị đơn hàng là bao nhiêu? | 1.0000 | 0.5198 | 1.0000 | 0.8042 |
| Phương thức trả góp bằng Thẻ tín dụng có áp dụng cho đơn hàng quốc tế không? | 1.0000 | 0.4379 | 1.0000 | 0.5000 |
| **Trung bình** | **0.9697** | **0.5752** | **1.0000** | **0.6847** |


## Nhận xét (Recommendations)
Cấu hình sử dụng Reranking có xu hướng cải thiện Context Precision so với Dense Only, do đó ảnh hưởng tích cực lên chất lượng câu trả lời cuối cùng.
