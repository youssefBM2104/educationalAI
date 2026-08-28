# Evaluation — Exam Generation

Đánh giá offline chất lượng đề thi do pipeline `exam_generation` sinh ra.

---

## 1. Triết lý thiết kế

### 1.1. Ba tầng, rẻ trước — đắt sau

| Tầng | Chi phí | Trả lời câu hỏi |
|---|---|---|
| **Deterministic** | miễn phí, chính xác tuyệt đối | Đề có **hợp lệ về cấu trúc** không? |
| **LLM-as-judge (G-Eval)** | tốn API call | Đề có **HAY** không? |
| *(chưa có)* Human + psychometrics | tốn người | Đề có **thật sự dạy được** không? |

> **Nguyên tắc:** cái gì code `assert` được thì **tuyệt đối không đốt LLM**. Đếm số câu, kiểm tra 4 đáp án A–D, so path length với difficulty — đó là việc của `if`, không phải của GPT.

### 1.2. Judge phải HELD-OUT

Pipeline đã có `judge_agent` bên trong (relevance / grounding / distractor / path coverage). **Judge của eval phải KHÁC nó** — nếu dùng lại chính con judge đó thì đề nào cũng "đạt", vì **vừa đá bóng vừa thổi còi**.

Nên:

- **Pipeline** chạy trên **NVIDIA NIM** (gemma / llama-3.1).
- **Judge của eval** chạy trên **Google Gemini** — **khác hẳn nhà cung cấp**.

Hai lợi ích:

1. **Tránh self-preference bias** — model không chấm output do chính họ nhà nó sinh ra.
2. **Gemini có constrained decoding native** → structured output **không bao giờ trả `None`** (khác NVIDIA, vốn hay trả `None` giữa chừng làm sập cả eval).

### 1.3. Tách SINH ĐỀ khỏi CHẤM ĐỀ

Sinh đề thì **chậm và hay lỗi** (503 / `None` / timeout). Chấm đề thì **nhanh và ổn định**.

Nên runner **lưu đề ra JSON** ngay sau khi sinh. Mọi lần chấm sau chỉ cần `--from file.json` → **không sinh lại, không dính lỗi sinh đề nữa**, và **thử nghiệm rubric mới thoải mái** trên cùng một bộ đề (so sánh công bằng).

---

## 2. `judge.py` — lõi G-Eval

**Vai trò:** biến một **rubric viết bằng tiếng Anh** thành **điểm số + lý do**.

### Cách hoạt động (theo công thức G-Eval)

```
criteria (tiêu chí)
   + evaluation_steps (các bước chấm, ép model suy luận CoT)
   + nội dung cần chấm
        ↓  LLM judge (Gemini, temperature = 0)
   reason + score (1–5)
        ↓  chuẩn hoá
   score ∈ [0, 1]
```

### Hai quyết định thiết kế đáng chú ý

**① `fields` — kiểm soát judge được THẤY GÌ**

Đây là điểm cốt lõi, và cũng là lý do **không dùng thư viện DeepEval**: DeepEval khoá cứng các field (`input` / `actual_output` / `context`), không cho giấu field theo từng metric.

Ví dụ quyết định: metric **Question Naturalness** **CỐ Ý không cho judge xem knowledge graph**.

> **Vì sao?** Học sinh làm bài **cũng không thấy KG**. Nếu judge được xem KG, nó sẽ thấy câu hỏi "hợp lý" và cho điểm cao — kể cả khi đó là câu **chỉ giải được nếu nhìn vào graph**. Giấu KG đi thì judge mới phát hiện ra câu hỏi là một **câu đố duyệt đồ thị**, không phải câu hỏi môn học.

**② Throttle 4.5s/call**

Gemini free tier giới hạn ~15 request/phút. Không throttle thì chạy được nửa chừng là bị chặn.

### ⚠️ Hạn chế đã biết: chưa có probability weighting

G-Eval gốc đọc **log-probability của token điểm** rồi tính kỳ vọng:

$$\text{score} = \sum_{s=1}^{5} s \cdot P(s)$$

→ ra điểm **mượt** (vd `3.7`), phản ánh **độ không chắc chắn** của judge.

Bản hiện tại chỉ lấy **số nguyên** ⇒ đang lấy **mode** thay vì **mean of posterior** — đúng cái mà paper G-Eval nói là **yếu**.

**Lý do:** `ChatGoogleGenerativeAI` **không expose logprobs**. Hai đường khắc phục:

| Cách | Mô tả | Đánh giá |
|---|---|---|
| **OpenAI + logprobs** | đọc `top_logprobs` → tính đúng công thức | exact, nhưng cần key trả phí |
| **Monte-Carlo trên Gemini** | `temperature=1`, lấy *k* mẫu, **trung bình** | ước lượng **không chệch** của cùng đại lượng; free |

*(Chưa triển khai — sẽ nâng cấp.)*

---

## 3. `deterministic.py` — kiểm tra cấu trúc

**Vai trò:** bắt các lỗi **cứng**, miễn phí, không cần LLM.

Mỗi check ở đây sinh ra từ **một lỗi có thật đã quan sát được** khi chạy pipeline — không phải check cho có.

### Cấp câu hỏi — `check_question()`

| Check | Lỗi thật đã gặp |
|---|---|
| **`kg_path` lẫn relation type** | Generator sinh `['mutex', 'DEFINES', 'synchronized java keyword']` — `DEFINES` là **quan hệ**, không phải concept. Lỗi này còn **thổi phồng path length** ⇒ tính sai độ khó. |
| **Path length vs difficulty** | Câu gắn nhãn `easy` nhưng path dài **7 concept** (dải cho phép là 1–2). Độ khó **được định nghĩa bằng path length**, nên vi phạm dải = sai độ khó. |
| **Đúng 4 đáp án A–D** | Schema có thể lệch. |
| **`correct_answer` ∈ choices** | Đáp án đúng phải nằm trong các lựa chọn. |
| **Trùng nội dung đáp án** | Hai option chữ giống nhau = câu hỏi hỏng. |
| **Essay có model answer + marking scheme** | Thiếu là không chấm được. |

### Cấp bộ đề — `check_exam_set()`

| Check | Lỗi thật đã gặp |
|---|---|
| **Số câu == target** | Judge fail → `give_up` ⇒ hụt câu ("xin 30, trả 24"). |
| **Trùng câu hỏi** | — |
| **Trùng `kg_path`** | Q1 và Q2 **cùng một path** ⇒ hai câu na ná nhau. Đây là lỗi **tinh vi hơn** trùng chữ: câu **khác chữ** nhưng **cùng nội dung suy luận**. |
| **Phân bố độ khó** | Kiểm tra tỉ lệ 40 / 40 / 20 (easy / medium / hard). |

> Lưu ý: khi đo path length, code **loại bỏ relation type trước khi đếm** — nếu không, câu bị lỗi ① sẽ kéo theo báo sai lỗi ②.

---

## 4. `metrics_exam.py` — bốn rubric G-Eval

Mỗi rubric nhắm vào **một kiểu hỏng cụ thể đã quan sát được**, không phải câu hỏi chung chung "đề này có tốt không".

| Metric | Bắt lỗi gì | Judge được thấy |
|---|---|---|
| **Question Naturalness** | Câu hỏi kiểu *"Chuỗi concept nào dẫn tới X?"* với đáp án là **đường đi KG thô** (`A → B → C`). Học sinh **không bao giờ thấy graph** ⇒ câu vô dụng. | `query`, `question`, `options` — **KHÔNG có KG** *(cố ý)* |
| **Groundedness** | Bịa fact không có trong chunks. | + `chunks`, `explanation` |
| **Distractor Quality** | Distractor ngớ ngẩn (loại được ngay), hoặc **hai đáp án cùng đúng**. Thưởng cho **near-miss** — concept gần giống, rẽ sai đúng một bước. | + `chunks`, `kg_relations` |
| **Difficulty Appropriateness** | Câu `hard` mà thật ra chỉ **tra một câu trong chunk**; câu `easy` mà phải **suy luận nhiều bước**. Bổ sung cho check path-length **ở tầng ngữ nghĩa** — *path dài chưa chắc câu khó*. | + `difficulty` |

Essay bỏ metric Distractor (không có đáp án nhiễu).

---

## 5. `run_exam_eval.py` — runner

```
[sinh đề hoặc nạp JSON]
        ↓
① Deterministic  →  liệt kê lỗi cấu trúc  (miễn phí)
        ↓
② G-Eval         →  điểm từng metric, từng câu  (Gemini)
        ↓
③ Summary        →  điểm trung bình mỗi metric + tổng số lỗi cấu trúc
```

```bash
# sinh đề mới + lưu + chấm
python -m backend.eval.run_exam_eval

# chấm lại đề đã lưu — không sinh lại, không tốn LLM sinh đề
python -m backend.eval.run_exam_eval --from outputs/eval/exam_latest.json

# đổi yêu cầu
python -m backend.eval.run_exam_eval --query "Create 3 essay questions about deadlock"
```

Cần `GOOGLE_API_KEY` trong `.env` (lấy free tại https://aistudio.google.com/apikey).

---

## 6. Còn thiếu gì để đăng paper

Nếu dùng bộ eval này cho paper, reviewer **chắc chắn sẽ đòi** hai thứ dưới đây — hiện **chưa có**:

| Thiếu | Vì sao bắt buộc |
|---|---|
| **Human validation của judge** | Không được lập luận *"LLM chấm 0.85 nên hệ thống tốt"* — đó là **vòng tròn**. Phải lấy mẫu ~50–100 câu, **người chấm tay**, rồi báo cáo **correlation** (Spearman / Cohen's κ) giữa judge và người. |
| **Baseline & ablation** | Điểm `0.85` **vô nghĩa nếu không so với gì**. Cần: baseline LLM one-shot (không multi-agent, không KG), và ablation **bỏ KG** / **bỏ judge** / **bỏ solver**. |
| *(bonus)* **Psychometrics** | Nếu thử được với học sinh thật: **item difficulty (p-value)**, **discrimination index**, **distractor efficiency**. Đây là **ground truth thật** — mạnh hơn mọi LLM-judge, và validate trực tiếp cho metric Difficulty & Distractor. |

**Điểm mạnh nên nêu rõ trong paper:** judge là **held-out provider** (Gemini ≠ NVIDIA) ⇒ tránh được self-preference bias — nhiều paper bỏ sót chi tiết này.

**Đặt tên rubric neo vào literature** (tránh bị chê "tự chế metric để tự khen"):

| Metric của mình | Chiều chuẩn trong literature AQG |
|---|---|
| Question Naturalness | Fluency / Well-formedness |
| Groundedness | Faithfulness |
| Distractor Quality | Distractor Plausibility / Efficiency |
| Difficulty Appropriateness | Difficulty Calibration |
| *(nên thêm)* | **Answerability** — solver độc lập giải có ra đáp án không |

**Cite:** Liu et al. 2023 — *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*.
