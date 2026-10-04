# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Mai Văn Trường  
**Mã sinh viên:** 2A202602983  
**Khóa:** K4 - Track 3B  
**Ngày thực hiện:** 04/10/2026  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 0.7746 | 0.7140 | -0.0606 |
| Context Precision | 0.8667 | 0.8917 | +0.0250 |
| Context Recall | 0.8467 | 0.7666 | -0.0801 |

> **Nhận xét tổng quan:**  
> - **Context Precision** của Production RAG đạt **0.8917**, tăng so với bản Baseline nhờ sự kết hợp của **Hierarchical Chunking**, **Hybrid Search (BM25 + Dense)** và đặc biệt là tầng **Cross-Encoder Reranking** đưa đúng đoạn trích dẫn quan trọng lên vị trí ưu tiên số 1 (rank 1).
> - **Faithfulness** đạt điểm tuyệt đối **1.0000**, đảm bảo hệ thống không bịa đặt (hallucination) mà luôn bám sát vào ngữ cảnh được cung cấp.
> - Các chỉ số còn lại gặp thách thức chủ yếu ở các câu hỏi phức hợp đa bước (multi-hop) và xung đột phiên bản tài liệu (văn bản cũ v2023 vs văn bản mới v2024).

---

## Bottom-5 Failures

### #1
- **Question:** "Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?"
- **Expected:** Hạn mức từ trên 50 triệu đến 200 triệu thuộc thẩm quyền phê duyệt của Phó Tổng Giám đốc (VP) hoặc Tổng Giám đốc theo Quy chế tài chính.
- **Got:** Hệ thống trích xuất nhầm đoạn văn về hạn mức dưới 50 triệu (thuộc thẩm quyền Giám đốc phòng ban / Director) lên trên đầu.
- **Worst metric:** Context Precision (0.5000)
- **Error Tree:** Output chưa chuẩn xác → Context đúng nằm ở rank 2 thay vì rank 1 → Query rõ ràng → Lỗi tại khâu Rerank / Phân biệt hạn mức số.
- **Phân tích 4 câu hỏi:**
  1. *Câu trả lời của mô hình có đúng không?* Chưa chính xác hoàn toàn do bị định hướng bởi đoạn trích dẫn xếp đầu tiên về hạn mức 5-50 triệu.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Có chứa đoạn đúng về hạn mức trên 50 triệu, nhưng bị xếp sau (rank 2).
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Không cần. Câu hỏi đã nêu cụ thể số tiền "55 triệu".
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module 2 (Search) & Module 3 (Rerank): Cần bổ sung metadata filter cho khoảng giá trị tài chính hoặc fine-tune Cross-Encoder để hiểu ngữ nghĩa khoảng giá trị số `(50M, 200M]`.

---

### #2
- **Question:** "Bao lâu phải đổi mật khẩu một lần?"
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải thay đổi mỗi 120 ngày. Chính sách cũ v1.0 (90 ngày) đã hết hiệu lực.
- **Got:** Đoạn văn trích xuất chứa cả văn bản cũ v1.0 (90 ngày) và bị xếp lên đầu do từ khóa mật khẩu trùng khớp cao.
- **Worst metric:** Context Precision (0.3333)
- **Error Tree:** Output có nguy cơ nhầm lẫn phiên bản → Context chứa tài liệu cũ hết hiệu lực → Query không nêu rõ năm ban hành → Lỗi ở Metadata Filtering / Contextual Prepend.
- **Phân tích 4 câu hỏi:**
  1. *Câu trả lời của mô hình có đúng không?* Nguy cơ cao trả lời 90 ngày nếu không có cơ chế phân biệt phiên bản có hiệu lực.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Có chứa bản v2.0 (120 ngày) nhưng lại bị xếp sau bản v1.0 (90 ngày).
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Người dùng hỏi thông thường sẽ không ghi năm. Pipeline phải tự động ưu tiên tài liệu mới nhất.
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module 5 (Enrichment - Contextual Prepend): Gắn rõ `[Tài liệu v2.0 có hiệu lực thay thế v1.0]` vào đầu chunk và Module 2: Thêm bộ lọc metadata `status: active`.

---

### #3
- **Question:** "Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?"
- **Expected:** Theo chính sách v2024 hiện hành, thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm.
- **Got:** Cả 2 đoạn trích dẫn (3 năm vs 5 năm) đều lọt vào top 3 nhưng đoạn cũ v2023 có BM25 score cao nên tranh chấp vị trí đầu.
- **Worst metric:** Context Precision (0.5000)
- **Error Tree:** Output xung đột điều khoản → Context chứa cả văn bản thay thế và bị thay thế → Query ngắn gọn → Lỗi ở Tầng Enrichment / Temporal Rerank.
- **Phân tích 4 câu hỏi:**
  1. *Câu trả lời của mô hình có đúng không?* Dễ gây phân vân hoặc đưa ra cả 2 con số mâu thuẫn nếu prompt không có chỉ thị xử lý xung đột thời gian.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Có, cả 2 đoạn trích dẫn đều có mặt trong top-3 contexts.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Có thể viết lại thành "theo quy định mới nhất hiện hành", tuy nhiên hệ thống RAG cần chủ động giải quyết.
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module 5 (Enrichment) kết hợp Module 2: Gắn nhãn phiên bản tài liệu (`version: 2024`, `effective_date`) và áp dụng Time-decay reranking score.

---

### #4
- **Question:** "Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?"
- **Expected:** Hạn mức 30 triệu (5-50 triệu) do Giám đốc phòng ban (Director) duyệt; mua sắm thiết bị CNTT cần xác nhận cấu hình kỹ thuật từ phòng CNTT và ít nhất 3 báo giá.
- **Got:** Câu trả lời chỉ tập trung vào một vế (hoặc thẩm quyền phê duyệt 30tr, hoặc quy trình xác nhận CNTT).
- **Worst metric:** Answer Relevancy (0.4400)
- **Error Tree:** Output thiếu ý → Context bị phân tán ở 2 tài liệu độc lập (Quy chế tài chính và Quy trình CNTT) → Query phức hợp đa mục tiêu (Multi-hop) → Lỗi ở Query Decomposition / Retrieval Coverage.
- **Phân tích 4 câu hỏi:**
  1. *Câu trả lời của mô hình có đúng không?* Đúng một phần, bị thiếu vế thứ hai về thủ tục kỹ thuật phòng CNTT.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Chỉ trích xuất được 1 trong 2 tài liệu liên quan do giới hạn top_k.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Rất cần! Đây là câu hỏi phức hợp (multi-hop/multi-aspect). Cần phân rã (Query Decomposition) thành 2 câu hỏi con riêng biệt.
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module tiền xử lý câu hỏi: Bổ sung Query Rewriter / Sub-query Generator trước khi truy vấn search engine.

---

### #5
- **Question:** "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"
- **Expected:** Nghỉ phép: 15 ngày cơ bản + 3 ngày thâm niên = 18 ngày phép. Lương Senior (P3-P4): từ 20 đến 35 triệu VNĐ/tháng.
- **Got:** Đoạn trích dẫn chỉ lấy được chính sách nghỉ phép, bỏ sót hoàn toàn bảng thang lương chức danh Senior.
- **Worst metric:** Context Recall (0.5217)
- **Error Tree:** Output thiếu dữ liệu lương → Context bỏ sót tài liệu thang lương → Query ghép 2 chủ đề hoàn toàn độc lập (nghỉ phép và bảng lương) → Lỗi ở Retrieval Recall / Multi-hop Search.
- **Phân tích 4 câu hỏi:**
  1. *Câu trả lời của mô hình có đúng không?* Chỉ trả lời được số ngày phép, không nêu được mức lương Senior.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Không chứa đủ. Bị thiếu chunk trích từ tài liệu thang bảng lương.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Có. Cần tách thành 2 truy vấn độc lập: (1) Số ngày phép của nhân viên 9 năm thâm niên; (2) Mức lương ngạch Senior.
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module 1 (Chunking) & Module 2 (Search): Cần kỹ thuật Multi-hop Retrieval hoặc Agentic RAG cho phép gọi tìm kiếm nhiều vòng lặp (iterative retrieval).

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
> *"Bao lâu phải đổi mật khẩu một lần?"* (Ca điển hình về **Xung đột phiên bản tài liệu — Version Conflict**)

**Error Tree walkthrough:**
1. **Output đúng?** $\rightarrow$ **KHÔNG**. Nếu trích nhầm tài liệu cũ, mô hình trả lời "90 ngày", trong khi câu trả lời đúng theo quy chế hiện hành v2.0 là "120 ngày".
2. **Context đúng?** $\rightarrow$ **CÓ một phần**. Cả đoạn văn bản v1.0 và v2.0 đều được nạp vào, nhưng đoạn cũ v1.0 lại chiếm vị trí ưu tiên cao hơn do tần số từ khóa mật độ cao.
3. **Query rewrite OK?** $\rightarrow$ Người dùng chỉ hỏi một câu ngắn gọn tự nhiên. Truy vấn không chỉ rõ năm ban hành.
4. **Fix ở bước:**  
   - **Fix tại Module 5 (Enrichment):** Gắn Contextual Prepend vào đầu chunk:  
     `"Trích từ Chính sách Bảo mật Thông tin v2.0 (ban hành 2024, thay thế toàn bộ bản v1.0). [Nội dung chunk]"`  
     Khi đó, Cross-Encoder reranker và LLM sẽ nhận biết ngay văn bản này là phiên bản hiện hành.
   - **Fix tại Module 2 (Search Indexing):** Đánh chỉ mục trường metadata `status: "active"` / `status: "deprecated"`. Khi người dùng tra cứu chính sách áp dụng, hệ thống mặc định lọc `filter={"status": "active"}`.

**Nếu có thêm 1 giờ, sẽ optimize:**
- **Giải pháp 1 (Metadata Temporal Filtering):** Thêm trường `valid_from`, `valid_to` vào metadata của chunk trong M1/M5, tự động loại bỏ các tài liệu đã hết hiệu lực trước khi đưa vào RRF.
- **Giải pháp 2 (Query Decomposition Agent):** Tích hợp một module tiền xử lý phân rã câu hỏi phức hợp thành các câu hỏi đơn cho các câu hỏi đa bước (như câu hỏi mua laptop 30tr hoặc lương Senior kèm thâm niên phép).
- **Giải pháp 3 (Parent Document Retrieval):** Khi child chunk khớp trúng câu trả lời, tự động trả về toàn bộ parent chunk tương ứng để LLM nắm trọn vẹn ngữ cảnh xung quanh điều khoản.
