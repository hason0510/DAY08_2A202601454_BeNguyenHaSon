# Hướng dẫn dữ liệu của dự án

File này mô tả chức năng của các thư mục và tệp dữ liệu được sử dụng trong hệ thống E-commerce Support RAG Chatbot.

> Đây là tài liệu hướng dẫn, không phải dữ liệu nguồn dùng để tạo embedding hoặc trả lời người dùng.

## 1. Cấu trúc dữ liệu

```text
data/
├── landing/
│   ├── legal/          # Tài liệu chính sách gốc dạng PDF
│   └── news/           # Bài hỗ trợ khách hàng đã crawl dạng JSON
└── standardized/
    ├── legal/          # Nội dung PDF đã chuyển sang Markdown
    └── news/           # Nội dung JSON đã chuẩn hóa sang Markdown
```

### `data/landing`

Đây là vùng chứa dữ liệu thô vừa được thu thập. Các file trong thư mục này được giữ gần với nội dung nguồn nhất để có thể kiểm tra, chuyển đổi hoặc xử lý lại khi cần.

- `landing/legal`: chứa các chính sách Shopee dạng PDF như bảo mật, trả hàng/hoàn tiền, phí vận chuyển, mã ưu đãi và chống gian lận tiếp thị liên kết.
- `landing/news`: chứa bài viết hỗ trợ khách hàng dạng JSON. Mỗi file có URL, tiêu đề, ngày crawl, nhóm người dùng, chủ đề và nội dung Markdown.

### `data/standardized`

Đây là vùng dữ liệu đã được chuẩn hóa sang Markdown. Các file tại đây được dùng làm đầu vào cho bước chia đoạn, tạo embedding, lưu vào ChromaDB và tìm kiếm RAG.

- `standardized/legal`: phiên bản Markdown được chuyển đổi từ các PDF trong `landing/legal`.
- `standardized/news`: phiên bản Markdown được tạo từ các JSON trong `landing/news`, kèm metadata YAML ở đầu file.

## 2. Chức năng các file trong `landing/news`

| File nguồn | Chủ đề | Chức năng trong chatbot |
|---|---|---|
| `01_payment_methods.json` | Phương thức thanh toán | Cung cấp thông tin về ShopeePay, thẻ ngân hàng, QR, COD, Apple Pay, Google Pay và SPayLater. |
| `02_change_payment_method.json` | Đổi phương thức thanh toán | Hướng dẫn đổi hình thức thanh toán cho đơn hàng trả trước. |
| `03_track_order.json` | Theo dõi đơn hàng | Hướng dẫn xem trạng thái lấy hàng, giao hàng và tra cứu bằng mã vận đơn. |
| `04_refund_evidence.json` | Bằng chứng hoàn tiền | Mô tả quy trình trả hàng/hoàn tiền và cách bổ sung hình ảnh hoặc video làm bằng chứng. |
| `05_cross_border_purchase.json` | Mua hàng xuyên biên giới | Giải thích điều kiện mua hàng trên nền tảng Shopee của quốc gia khác. |

Các file Markdown tương ứng trong thư mục này có tên:

- `01-payment-methods.md`
- `02-change-payment-method.md`
- `03-track-order.md`
- `04-refund-evidence.md`
- `05-cross-border-purchase.md`

## 3. Ý nghĩa metadata

Mỗi file Markdown chuẩn hóa có thể chứa các trường sau:

| Trường | Ý nghĩa |
|---|---|
| `title` | Tiêu đề của tài liệu hoặc bài hỗ trợ. |
| `source_url` | URL bài viết gốc trên trang trợ giúp Shopee. |
| `source_file` | Tên file nguồn trong thư mục `landing`. |
| `source_type` | Loại dữ liệu, thường là `legal` hoặc `news`. |
| `customer_role` | Nhóm người dùng liên quan: `buyer`, `seller` hoặc `both`. |
| `topic` | Chủ đề dùng để phân loại và lọc dữ liệu khi retrieval. |
| `date_crawled` | Thời điểm bài viết được crawl. |
| `converted_at` | Thời điểm dữ liệu được chuyển sang Markdown. |

## 4. Dữ liệu được sử dụng trong RAG như thế nào?

```text
File Markdown
    ↓
Chia thành các đoạn nhỏ (chunking)
    ↓
Tạo vector embedding
    ↓
Lưu vào ChromaDB
    ↓
Tìm kiếm semantic kết hợp BM25
    ↓
Rerank các đoạn liên quan
    ↓
LLM tạo câu trả lời có trích dẫn nguồn
```

## 5. Quy tắc cập nhật dữ liệu

1. Không sửa trực tiếp dữ liệu thô trong `landing` nếu chưa lưu lại nguồn ban đầu.
2. Dùng UTF-8 để tránh lỗi tiếng Việt.
3. Mỗi tài liệu chuẩn hóa cần có nguồn và metadata rõ ràng.
4. Không đưa API key, thông tin đăng nhập hoặc dữ liệu cá nhân vào repository.
5. Sau khi thêm hoặc sửa tài liệu Markdown, cần chạy lại bước chunking và indexing để cập nhật ChromaDB.
6. Không index file `README.md` này vì nó chỉ dùng để giải thích cấu trúc dữ liệu.
