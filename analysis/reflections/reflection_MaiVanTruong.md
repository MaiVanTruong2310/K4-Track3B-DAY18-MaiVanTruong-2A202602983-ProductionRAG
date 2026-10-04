# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Mai Văn Trường  
**Mã sinh viên:** 2A202602983  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong lecture vào code vừa thực hiện trong lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Sử dụng mô hình embedding BAAI/bge-m3 tính cosine similarity giữa các câu kế tiếp nhau. Ngưỡng threshold 0.85 giúp phân định ranh giới chuyển ý chủ đề, gom các câu cùng mạch thành 1 chunk trọn vẹn thay vì cắt cơ học. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Cắt phân cấp thành parent chunks (2048 ký tự) chứa bối cảnh vĩ mô và child chunks (256 ký tự) chứa chi tiết nhỏ sắc bén. Lưu trữ `parent_id` trong metadata giúp khi tìm kiếm trúng child chunk có thể mở rộng ngữ cảnh cha cho LLM. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Phân tích cấu trúc phân cấp tài liệu dựa trên Markdown header (`#`, `##`, `###`), duy trì `header_path` và `section` trong metadata giúp bảo toàn ngữ cảnh điều khoản pháp lý và bảng biểu. |
| BM25 + Dense fusion (Hybrid Search) | M2 | `reciprocal_rank_fusion()` | RRF kết hợp điểm xếp hạng lexical (BM25 bắt chính xác số hiệu văn bản, từ viết tắt) và semantic dense search (bắt ngữ nghĩa câu hỏi), loại bỏ nhược điểm điểm số khác thang đo của 2 phương pháp. |
| Vietnamese Word Segmentation | M2 | `segment_vietnamese()` | Sử dụng `underthesea.word_tokenize` đồng nhất tách từ ghép tiếng Việt cho cả tài liệu lúc index và câu truy vấn lúc search, đảm bảo BM25 khớp đúng từ đa âm tiết. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Sử dụng mô hình `BAAI/bge-reranker-v2-m3` xử lý đồng thời cặp (query, document) thông qua cross-attention, lọc từ 20 candidate xuống top 3 kết quả chuẩn xác nhất, loại bỏ tài liệu nhiễu. |
| RAGAS 4 metrics & Error Tree | M4 | `evaluate_ragas()`, `failure_analysis()` | Đánh giá 4 chỉ số cốt lõi (Faithfulness, Answer Relevancy, Context Precision, Context Recall) kết hợp cây chẩn đoán lỗi `DIAGNOSTIC_TREE` phân loại lỗi: Generation vs Retrieval vs Query Rewrite. |
| Contextual embeddings & Enrichment | M5 | `contextual_prepend()`, `_enrich_single_call()` | Kỹ thuật Contextual Prepend của Anthropic gắn tóm tắt định vị đoạn văn trước chunk, kết hợp HyQA (sinh câu hỏi giả định) và trích xuất metadata trong 1 API call duy nhất để tiết kiệm chi phí và tăng tỷ lệ truy xuất. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật 1: Qdrant Server Timeout & In-memory Fallback**
  - *Mô tả lỗi:* Khi môi trường máy tính chưa khởi chạy Docker daemon cho Qdrant trên cổng `localhost:6333`, lệnh gọi `QdrantClient(host="localhost", port=6333, timeout=2)` gây ra timeout `ResponseHandlingException`.
  - *Nguyên nhân & Cách xử lý:* Lớp kết nối từ xa cố gắng handshake với server không tồn tại. Đã xử lý bằng khối `try...except` chủ động kiểm tra `get_collections()` với timeout ngắn 2s, nếu thất bại sẽ fallback lập tức sang `QdrantClient(":memory:")`, đảm bảo pipeline chạy trọn vẹn cả trên môi trường local không có Docker.

- **Lỗi kỹ thuật 2: Xung đột chuẩn hóa từ ghép tiếng Việt trong BM25**
  - *Mô tả lỗi:* `underthesea` nối từ ghép bằng dấu gạch dưới (ví dụ `nghỉ_phép`), trong khi câu truy vấn người dùng nhập từ rời hoặc câu có ký tự đặc biệt khiến BM25 không tính toán chính xác tần số từ BM25 score.
  - *Nguyên nhân & Cách xử lý:* Viết hàm tiền xử lý `segment_vietnamese()` dùng `format="text"` của underthesea, thay thế ký tự đặc biệt, chuẩn hóa chữ thường và đồng bộ hóa cả khâu indexing và khâu query.

- **Lỗi kỹ thuật 3: Chi phí và độ trễ cao khi thực hiện Chunk Enrichment**
  - *Mô tả lỗi:* Nếu gọi 4 API requests riêng biệt cho mỗi chunk (Summary, HyQA, Contextual Prepend, Metadata Extraction) thì với 100 chunks sẽ tốn tới 400 lượt gọi API, chi phí cao và dễ chạm rate limit.
  - *Nguyên nhân & Cách xử lý:* Thiết kế hàm gộp `_enrich_single_call()` gửi prompt tích hợp với `response_format={"type": "json_object"}` để LLM trả về cùng lúc 4 trường cấu trúc trong 1 request duy nhất (giảm 75% API calls). Đồng thời xây dựng hàm fallback chiết xuất bằng quy tắc regex để pipeline hoạt động độc lập ngay cả khi không có OpenAI API key.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý Tra cứu Quy trình Nội bộ & Tài liệu Kỹ thuật Doanh nghiệp

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Basic RAG cơ bản với naive chunking theo độ dài cố định (500 ký tự) và Dense Retrieval duy nhất dựa trên cosine similarity.
- **Vấn đề / Bottlenecks đang gặp:**
  - Cắt vụn văn bản làm đứt gãy bảng biểu chi phí và danh sách các bước phê duyệt quy trình.
  - Xung đột phiên bản tài liệu (quy chế năm 2023 bị trích xuất thay vì quy chế mới năm 2024).
  - Tra cứu các số hiệu văn bản, tên thiết bị hoặc từ viết tắt CNTT thường bị trả về thông tin chung chung do mô hình dense vector bắt trượt từ khóa.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Áp dụng **Hierarchical Chunking** kết hợp **Structure-aware Chunking** để phân tách văn bản theo các điều khoản, mục lục, giữ nguyên tiêu đề cha con và bảo toàn cấu trúc bảng.
2. **Search retrieval:** Triển khai **Hybrid Search** kết hợp BM25 (đã tách từ tiếng Việt bằng `underthesea`) và Dense Search (mô hình `BAAI/bge-m3` hỗ trợ đa ngữ xuất sắc) thông qua thuật toán dung hợp **Reciprocal Rank Fusion (RRF)**.
3. **Reranking:** Tích hợp tầng lọc **Cross-Encoder Reranker** (`BAAI/bge-reranker-v2-m3`) để đánh giá tương tác sâu giữa câu hỏi và từng đoạn trích, rút gọn từ top 20 xuống top 3-5 đoạn đắt giá nhất.
4. **Evaluation:** Xây dựng bộ test suite tự động với **RAGAS 4 metrics**, theo dõi chặt chẽ chỉ số Faithfulness (tránh bịa đặt) và Context Precision (chính xác ngữ cảnh).
5. **Enrichment:** Sử dụng **Contextual Prepend** của Anthropic để gắn metadata nguồn và số hiệu phiên bản trực tiếp vào từng chunk, giải quyết dứt điểm bài toán xung đột tài liệu cũ/mới.

#### 3. Timeline triển khai
- **Tuần 1:** Chuẩn hóa dữ liệu tài liệu nội bộ, triển khai bộ tiền xử lý tách từ tiếng Việt và xây dựng chiến lược Hierarchical Chunking.
- **Tuần 2:** Cài đặt cơ sở dữ liệu vector Qdrant, kết hợp BM25 thành Hybrid Search với RRF, tích hợp tầng Reranking bằng Cross-Encoder.
- **Tuần 3:** Tích hợp tầng Enrichment (Contextual Prepend & Metadata), chạy đánh giá định kỳ bằng RAGAS test suite và triển khai giao diện người dùng.
