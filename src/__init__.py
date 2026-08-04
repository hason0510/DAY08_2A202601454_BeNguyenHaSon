"""Day 8 — RAG Pipeline v2: Chính sách thương mại điện tử & hỗ trợ khách hàng (E-commerce Support)."""

import sys

# Console Windows mặc định dùng codepage cp1252, không in được ký tự Unicode
# như '✓', '⚠', '→' — mọi print() chứa chúng sẽ ném UnicodeEncodeError và làm
# CHẾT pipeline giữa chừng (ví dụ: Task 4 crash ngay sau bước load, khiến
# ChromaDB được tạo nhưng rỗng 0 chunk).
# Ép stdout/stderr sang UTF-8 để toàn nhóm chạy được trên Windows.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
