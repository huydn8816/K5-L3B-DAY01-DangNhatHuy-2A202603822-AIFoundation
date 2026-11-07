# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603822
- **Họ và tên:** Đặng Nhật Huy
- **SEED** (in ở ô đầu notebook): 5074

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

Bộ kiểm tra trong `lab1_report.json`: **15/15 đạt** (10 TODO từ 1.1 đến 4.4, cộng 5 phép tính tay).

## 2. Kết quả chính

Bảng so sánh cuối Phần 5. Phần 4 (LLM) chưa chạy với API thật nên không có hàng LLM.

| Cách | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.936 | 0.974 | 0.067 | 0.0 |
| 2. BoW + MLP (deep learning) | 0.964 | 1.000 | 0.213 | 0.0 |
| 3a. Encoder (MiniLM) + kNN | 0.924 | 1.000 | 19.113 | 0.0 |
| 3b. Encoder (MiniLM) + LogReg | 0.965 | 1.000 | 6.960 | 0.0 |
| 3c. Encoder (MiniLM) + MLP head | 0.954 | 1.000 | 7.924 | 0.0 |

Tập test có 775 tin (100 spam). SUB chỉ có 40 tin (20 spam, 20 ham), nên 1 tin sai làm F1 trên SUB thay đổi khoảng 2.5 điểm %. Vì vậy so sánh chất lượng dựa vào cột F1 (test).

## 3. Dự đoán và thực tế

**1. Độ trễ của kNN (`tradeoff`)**
- Đoán: kNN sẽ nằm bên phải vạch SLA 50 ms.
- Thực tế: p95 của kNN là 19 ms, đạt SLA.
- Vì sao lệch: tôi nghĩ so sánh với toàn bộ 3.619 vector train sẽ chậm. Thực tế đó chỉ là một phép nhân ma trận 3.619 × 384, rất nhanh. Phần tốn thời gian chủ yếu là encode câu bằng MiniLM, giống 3b và 3c. kNN chỉ thật sự chậm khi bộ nhớ lên tới hàng triệu tin.

**2. kNN trên 50 case khó (`hard_eval`)**
- Đoán: kNN khoảng 55%, kém hơn hoặc bằng 3b và 3c.
- Thực tế: kNN đạt 72%, cao nhất trong 5 model. 3b và 3c chỉ đạt 64%.
- Vì sao lệch: kNN không học một đường biên chung mà dựa vào láng giềng gần nhất, nên bắt tốt các kiểu spam có vài mẫu giống trong tập train: 100% ở nhóm "Thao túng / lừa đảo", 83% ở "Né từ khoá". Đổi lại, nó rất kém ở "Nói về spam" (25%), vì tin kể về giải thưởng nằm gần các tin spam về giải thưởng.

**3. F1 của BoW + LogReg (`ml_f1`)**
- Đoán: F1 khoảng 0.8, precision cao hơn recall.
- Thực tế: F1 = 0.936, precision = 1.000, recall = 0.88. Chiều precision > recall đúng, nhưng con số thấp hơn nhiều so với thực tế.
- Vì sao lệch: spam trong bộ dữ liệu SMS này phần lớn chứa từ khoá rất rõ ("call", "txt", "free", "claim"), nên chỉ một mô hình tuyến tính trên bag-of-words đã đủ tách tốt. Tôi đánh giá thấp độ dễ của dữ liệu cùng phân phối.

## 4. Phân tích 50 case khó

| Model | F1 test thường (SUB) | F1 50 case khó | Accuracy 50 case khó |
| --- | --- | --- | --- |
| 1. BoW + LogReg | 0.974 | 0.378 | 54% |
| 2. BoW + MLP | 1.000 | 0.368 | 52% |
| 3a. Emb + kNN | 1.000 | 0.759 | 72% |
| 3b. Emb + LogReg | 1.000 | 0.640 | 64% |
| 3c. Emb + MLP head | 1.000 | 0.625 | 64% |

**BoW + MLP tụt nhiều nhất:** F1 từ 1.000 xuống 0.368. Các nhóm khó nhất với nó là "Spam giọng tự nhiên" (0%), "Né từ khoá" (17%) và "Thao túng / lừa đảo" (25%). Điểm trên tập test thường quá lạc quan vì tập test cùng phân phối với tập train, và notebook chỉ bỏ trùng tuyệt đối nên các tin spam mẫu chỉ khác số điện thoại vẫn có thể nằm ở cả hai tập.

**Case #12** (Spam giọng tự nhiên): *"hi there, sorry to bother you, my friend gave me your number, I'm new in town and looking for s…"*. Cả 5 model đều sai. BoW nhìn văn bản như các token rời rạc, và các từ ở đây ("sorry", "friend", "my", "new", "town") là từ của tin nhắn bạn bè, không có từ khoá spam nào. Embedding nhìn nghĩa cả câu, mà giọng lịch sự, thân mật đặt vector gần vùng ham. Mồi nhử thật sự (dụ vào trang hẹn hò) nằm ở ý định, không nằm ở từ ngữ.

**Case #47** (Nói về spam): *"lol I just got a text saying I won £1000 and to call some 0906 number, people still fall for th…"*. Đây là tin ham, nhưng 3c đoán spam. Câu chứa nguyên văn nội dung lừa đảo ("won £1000", "call some 0906 number"), nên vector nghĩa của nó gần các tin spam về giải thưởng. Embedding biết câu nói về giải thưởng, nhưng không phân biệt được người viết đang quảng cáo hay đang cảnh báo.

## 5. Thử thách cá nhân

**Đề #5 (Embedding):** viết 4 tin nhắn của riêng mình (2 spam né từ khoá, 2 ham chứa "free"/"win"), so dự đoán BoW với Embedding + LogReg và giải thích từng tin.

**Cách làm:** chạy cả hai model trên 4 tin, in nhãn thật, dự đoán, xác suất spam `p`, và các từ BoW thực sự nhìn thấy cùng trọng số của chúng.

| # | Tin | Thật | BoW | Emb + LogReg |
| --- | --- | --- | --- | --- |
| 1 | Giả thông báo giao hàng, yêu cầu xác nhận địa chỉ trong 24 giờ | spam | ham ✗ (p = 0.31) | ham ✗ (p = 0.03) |
| 2 | "Y0u h4ve been ch0sen for a c@sh rew4rd…" (leetspeak) | spam | ham ✗ (p = 0.41) | ham ✗ (p = 0.11) |
| 3 | "Are you free tonight? Let's grab dinner…" | ham | ham ✓ (p = 0.04) | ham ✓ (p = 0.00) |
| 4 | "Did we win the match yesterday?…" | ham | ham ✓ (p = 0.01) | ham ✓ (p = 0.00) |

Cả hai model đều đạt accuracy 0.5 và F1 = 0, vì không bắt được tin spam nào.

**Giải thích:**
- **Tin 1:** BoW nhận ra 20/25 từ, có vài từ đẩy về spam ("your" 0.82, "or", "our", "co") nhưng phần lớn là từ trung tính. Embedding thấy câu giống một thông báo giao hàng thật, vì dữ liệu train (SMS cũ) gần như không có lừa đảo giao hàng.
- **Tin 2:** BoW chỉ nhận ra 8/15 từ. Các từ mang tín hiệu spam ("c@sh", "rew4rd", "T3xt") nằm ngoài vocab, chỉ còn "to", "now", "for" kéo xác suất lên 0.41. Embedding sai dù đạt 83% ở nhóm "Né từ khoá" của bộ 50 case. Giả thuyết: tin này thiếu các dấu hiệu spam quen thuộc trong dữ liệu (số điện thoại, mã ngắn, ký hiệu £).
- **Tin 3 và 4:** "free" (0.9) và "win" (0.38) có trọng số dương, nhưng các từ còn lại ("tonight", "dinner", "match", "work") kéo mạnh về ham. Embedding nhìn nghĩa cả câu (rủ đi ăn, hỏi kết quả trận đấu) nên đoán ham rất chắc.

**Kết luận:** tin ham chứa từ khoá dễ với cả hai model, nhưng spam né từ khoá lừa được cả hai. Embedding còn sai tự tin hơn BoW. Embedding hiểu nghĩa câu, nhưng bộ phân loại phía trên chỉ học "spam trông như thế nào" từ dữ liệu cũ.

## 6. Phương án cho sản phẩm thật

Yêu cầu: 10 triệu SMS/ngày (khoảng 116 tin/s trung bình), p95 < 50 ms/tin.

- **Độ trễ:** cả 5 cách đều đạt p95 < 50 ms. Chậm nhất là kNN với 19 ms, nhưng kNN chỉ xử lý khoảng 128 tin/s theo lô, sát mức trung bình 116 tin/s, nên sẽ quá tải giờ cao điểm nếu chỉ chạy một máy.
- **Chọn 3b (Encoder + LogReg):** F1 test 0.965 (cao nhất), p95 khoảng 7 ms, xử lý khoảng 3.000 tin/s. BoW + MLP có F1 gần bằng (0.964) và nhanh hơn nhiều (0.21 ms), nhưng chỉ đạt 52% trên 50 case khó, so với 64% của 3b. Tôi chấp nhận độ trễ cao hơn để có mô hình bền hơn trước các kiểu spam mới.
- **Nếu dùng LLM:** với gpt-6-luna (100 token vào, 1 token ra), 10 triệu tin/ngày tốn khoảng $105/ngày, tức khoảng **$3.150/tháng**. Model lớn đắt gấp khoảng 10 lần. Gọi API qua mạng cũng khó đạt p95 < 50 ms.
- **Thiết kế lai (router):** chuẩn hoá đầu vào và gắn cờ prompt injection, cho mọi tin qua 3b, rồi chỉ gửi các tin lưỡng lự (0.3 ≤ p ≤ 0.7) cho LLM với prompt phòng thủ. Nếu khoảng 5% tin là lưỡng lự, chi phí LLM giảm xuống khoảng $160/tháng, và p95 vẫn do đường nhanh quyết định. Đánh đổi: router không cứu được các ca 3b sai mà rất tự tin, như tin lừa giao hàng trong thử thách (p = 0.03), nên vẫn cần theo dõi và train lại định kỳ.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời / viết gì | Tôi đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| 1 (TODO 1.4) | Đánh giá hàm `metrics` đã đúng chưa | Chỉ ra lỗi chưa chuyển list sang `np.asarray` và dòng `print` thừa | Sửa, chạy lại test 1.4 ✅ |
| 1 (câu hỏi) | Precision/recall, Giải thích 1, câu 1.B, Bắt lỗi 1, Checkpoint A | Giải thích, gợi ý câu trả lời, nhận xét bài tôi viết | Tự viết lại Giải thích 1; đối chiếu với output thật |
| 2 (TODO 2.1, 2.2) | "Code phần 2" | Viết `build_mlp` và 5 dòng vòng lặp train | Chạy test 2.1, 2.2 ✅ |
| 2 (câu hỏi) | Tính nhẩm, dự đoán 2.A/2.B, câu 2.A, Giải thích 2 | Viết đáp án và gợi ý dự đoán | Đối chiếu với output (epoch val loss thấp nhất, F1 MLP) |
| 3 (TODO 3.1–3.2) | Lỗi `TypeError` ở `cosine_similarity` / `top_k_similar` | Sửa: thêm `np.asarray`, chuẩn hoá từng dòng của M | Chạy lại test 3.1–3.2 ✅ |
| 3 (câu hỏi) | Thí nghiệm 3.B, Câu hỏi 3, Bắt lỗi 2, Checkpoint B | Giải thích kết quả, viết câu trả lời | Đối chiếu với output 3.B (6/6 và 5/6) |
| 4 (TODO 4.1–4.4) | "Làm thí nghiệm 4A" | Viết `call_openai`, prompt phân loại, `parse_label`, `llm_classify_batch` | Test 4.1–4.4 bằng API giả lập ✅; chưa chạy API thật vì không có key |
| 4 (câu hỏi) | Gợi ý dự đoán 4A, 4B, Câu hỏi 4, Bắt lỗi 3 | Gợi ý dự đoán và câu trả lời | Chưa có kết quả LLM thật để đối chiếu |
| 5 | Pareto, SLA, Câu hỏi 5, 50 case khó, MY_CASES, tính tay, phản ánh cuối bài | Giải thích khái niệm, viết câu trả lời, đáp số tính tay | Đối chiếu với biểu đồ, bảng và bộ kiểm tra hash ✅ |
| Thử thách (Đề #5) | Nhận xét code của tôi, sau đó "giải nốt" | Nhận xét hai bản nháp, rồi viết code hoàn chỉnh và phân tích kết quả | Tôi chọn 4 tin và viết bản nháp; chạy code của AI và xem output |
| Báo cáo này | Viết file báo cáo cá nhân | Soạn toàn bộ nội dung từ số liệu trong notebook và `lab1_report.json` | Đọc lại, đối chiếu số liệu |

**Điều tôi hiểu ra hôm nay:** điểm cao trên tập test cùng phân phối không chứng minh model đã hiểu thế nào là spam. Cả năm model đều đạt F1 trên 0.92, nhưng tụt xuống 52–72% khi gặp các kiểu spam mới viết tay. Mỗi cách nhìn văn bản (token rời rạc, vector nghĩa, đọc như người) có một kiểu điểm mù riêng, nên chọn mô hình cho sản phẩm cần cân cả độ trễ, chi phí và độ bền trước dữ liệu mới, không chỉ một con số F1.
