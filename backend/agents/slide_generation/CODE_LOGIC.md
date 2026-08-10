# Logic code — pipeline sinh slide

Tài liệu mô tả **toàn bộ logic** của 5 file: `_common.py`, `composer_agent.py`, `slide_planner_agent.py`,
`verification_agent.py`, `export.py`. (Bản thiết kế tổng thể xem [instruction.md](instruction.md); các phần
Export vượt ngoài instruction xem [CHANGES_beyond_instruction.md](CHANGES_beyond_instruction.md).)

## Luồng tổng thể

```
             rag_chunks + kg_context (từ RAG service)
                              │
   init ──► composer ──► planner ──► verification ──┬─(pass)─► export ──► .pptx/.pdf
                 ▲                                  │
                 └──── retry_control ◄──(fail)──────┘
                       (patch heading lỗi, đếm attempt, giữ best_attempt)
```

- **composer** (Stage 1): chunks → heading + nội dung + chọn ảnh. Model `gpt-5`.
- **planner** (Stage 2): heading → slide (chọn layout, điền slot, mang ảnh). Model `gpt-5-mini`.
- **verification** (Stage 3): **chỉ gate nội dung** (grounding + coverage). Model `gpt-5-mini`.
- **retry_control** (Stage 4): thất bại → vá heading lỗi qua composer, cap attempt, giữ `best_attempt`.
- **export** (Stage 5): render ra `.pptx`/`.pdf`, **chỗ duy nhất đọc `image_base64`**.

Dữ liệu chảy trên `LectureState` (xem [state.py](../state.py)): `composer_output` → `lecture_slides` →
`verification_feedback`/`retry_scope` → `lecture_output_path`.

Nguyên tắc bất di bất dịch: **`image_base64` KHÔNG bao giờ vào prompt LLM** — mọi stage LLM chỉ thấy
`image_id` + `vlm_description` + `page_number`. Chỉ Export đọc bytes.

---

## 1. `_common.py` — helper dùng chung

Không có LLM, thuần hàm tiện ích. Chia 4 nhóm.

### Xử lý ảnh (an toàn base64)
- `_IMAGE_PROMPT_FIELDS = ("image_id", "vlm_description", "page_number")` — **danh sách trắng** field ảnh
  được phép đưa vào prompt. `image_base64` **cố tình vắng mặt** → đây là điểm chốt của nguyên tắc.
- `image_reasoning_view(image)` — rút 1 ảnh về đúng các field an toàn.
- `images_of_chunk(chunk)` — lấy `chunk["images"]`.
- `build_image_index(chunks)` — dựng map `image_id → {image_base64, mime_type}`. **Chỉ Export gọi** — đây là
  chỗ duy nhất giữ base64.
- `available_images(chunks)` — mọi ảnh ở dạng an-toàn-prompt (không base64), cho composer chọn.

### Concept / chunk
- `concept_ids(chunk)` — lấy id concept từ `covers_concepts` (chịu cả dạng `{"id":...}` lẫn string).
- `all_concepts(chunks)` — tập concept toàn bộ, **giữ thứ tự, không trùng** (dùng dict làm ordered-set).
  Đây là "mẫu số" cho coverage check ở Stage 3.

### Định dạng prompt (không base64)
- `format_chunks_for_prompt(chunks)` — render chunk cho LLM: text + concept + **mô tả ảnh** (không bytes).
- `format_relations(kg)` — render quan hệ KG dạng `A --[TYPE]--> B`.

### Template mặc định
- `DEFAULT_TEMPLATE` — **thư viện layout dạng dict** (KHÁC với template `.pptx` trực quan). Mỗi layout có
  `id` + `purpose` + `slots`: `title`, `title_bullets`, `two_column_image`, `three_card`, `stat`,
  `image_caption`. `max_words_per_bullet=16`. Planner đọc cái này để chọn layout; `id` là **hợp đồng** với
  Export (Export map `id` này sang layout thật của `.pptx`).

---

## 2. `composer_agent.py` — Stage 1: nội dung & cấu trúc

**Vai trò:** biến chunks thành **heading + nội dung viết sẵn + chọn ảnh minh hoạ**. Model `gpt-5`.

### Schema (Pydantic, ép qua function-calling)
- `Heading{heading_id, title, source_concepts, content_points, image_refs}`.
- `ComposerResult{headings: [Heading]}`.

### Hàm
- `_coverage_check(chunks, headings)` — **tính bằng code, không tin LLM**: `all_concepts` (mọi concept trong
  chunks) và `concepts_represented` (concept đã được gán vào heading nào đó). Là bằng chứng cứng cho Stage 3.
- `_base_prompt(state)` — prompt lượt đầu: nêu quan hệ KG (`PART_OF`/`DEFINES` → cùng heading,
  `PREREQUISITE` → thứ tự), chunks, **ảnh dạng an toàn**, luật (viết nội dung bám chunk, **chỉ gắn ảnh thật
  sự minh hoạ**, không thay ảnh bằng bullet mô tả, mọi concept phải thuộc 1 heading).
- `_retry_prompt(state)` — prompt retry: `_base_prompt` + yêu cầu **chỉ regenerate heading trong
  `retry_scope.heading_ids`**, giữ id ổn định; issue `heading_id=null` = **tạo heading mới** cho concept mồ côi.
- `composer_agent(state)` — điều phối:
  1. Xác định `is_retry` (có `retry_scope.heading_ids` hoặc đã có feedback + composer_output cũ).
  2. Gọi LLM ra `new_headings`.
  3. **Nếu retry:** merge — thay heading trong scope + heading vừa đổi id, **giữ nguyên phần còn lại**.
  4. Tính lại `coverage_check`, trả `composer_output`.

**Điểm mấu chốt:** owns *cái gì được dạy, tổ chức ra sao, mỗi mục nói gì, ảnh nào* — KHÔNG quyết số slide/layout.

---

## 3. `slide_planner_agent.py` — Stage 2: bố cục slide

**Vai trò:** biến heading thành **deck slide-by-slide**: chọn layout theo `purpose`, điền slot, mang ảnh sang.
Model `gpt-5-mini` (map layout là việc cơ học, suy luận nặng đã ở composer).

### Schema
- `Slide{slide_id, heading_id, layout, title, bullets, slots, image_refs}`.
  - `bullets`: chỉ cho layout dạng bullet; `[]` cho stat/card/image.
  - `slots`: nội dung cho layout không-bullet, key theo tên slot của layout (`cards`, `stat`, `caption`...).
- `PlannerResult{slides: [Slide]}`.

### Hàm
- `_prompt(headings, template)` — bảo LLM khớp **SHAPE nội dung → PURPOSE layout**: 3 mục song song →
  `three_card`; 1 con số → `stat`; hình là điểm chính → `image_caption`; concept + hình phụ →
  `two_column_image`; còn lại → `title_bullets`. Luật: tôn trọng `max_words_per_bullet`; **heading dài thì
  TÁCH nhiều slide cùng `heading_id`, không cắt xén**; mang `image_refs` sang, không tự bịa hình, không mô tả
  hình bằng chữ.
- `_plan(headings, template)` — gọi LLM, trả list dict slide.
- `slide_planner_agent(state)` — điều phối:
  - **Lượt đầu:** plan toàn bộ heading.
  - **Retry có scope:** chỉ rebuild slide của heading trong `retry_scope.heading_ids`, **giữ nguyên slide còn
    lại** (`kept + rebuilt`).

**Điểm mấu chốt:** thuần bố cục. Field `layout` mà planner sinh ra chính là cái Export dùng để chọn khuôn
template thật.

---

## 4. `verification_agent.py` — Stage 3: kiểm định (chỉ gate nội dung)

**Vai trò:** **chỉ gate NỘI DUNG**. Template & ảnh vẫn kiểm nhưng **chỉ là ghi chú tư vấn**, không bao giờ
làm rớt deck hay kích retry. Model `gpt-5-mini`. `CONTENT_THRESHOLD = 0.8`.

### Schema
- `HeadingVerdict{heading_id, grounded: bool, issue}` — nội dung có bám chunk không.
- `ImageNote{heading_id, image_id, issue}` — ảnh liên quan mạnh mà không lên slide nào (tư vấn).
- `VerifyResult{heading_verdicts, image_notes}`.

### Hàm
- `_prompt(state)` — chỉ hỏi LLM 2 điều: (1) mỗi heading có **grounded** vào chunks không, (2) tư vấn: ảnh liên
  quan nào bị bỏ sót. Đưa chunks + heading + (slide đã có ảnh nào).
- `_template_notes(state)` — **deterministic, tư vấn**: bullet nào vượt `max_words_per_bullet` thì gắn note
  `type="template"`.
- `verification_agent(state)` — điều phối, gom 4 loại issue:
  1. **Grounding** (content, GATE): đếm heading grounded → `content_score = grounded / tổng heading`.
     Heading không grounded → issue `type="content"`.
  2. **Coverage gap** (content, GATE, deterministic): `all_concepts − concepts_represented`; concept mồ côi →
     issue `type="content", heading_id=null`.
  3. **Template density** (tư vấn) — từ `_template_notes`.
  4. **Image relevance** (tư vấn) — từ `image_notes`.
  - `passed = content_score >= 0.8 AND không còn coverage gap`.
  - `retry_scope.heading_ids` = mọi issue `type="content"` (retry **luôn về Composer**).

**Điểm mấu chốt:** chỉ nội dung mới làm rớt → không có `retry_target`, luôn quay về composer. Template/ảnh chỉ
để giáo viên tự chỉnh tay.

> Lưu ý (ngoài instruction): Verification bắt **coverage-gap** (concept không vào heading nào), NHƯNG **không**
> bắt được ca planner sinh slide **rỗng** dù concept vẫn nằm trong heading — ca đó Export vá bằng backfill (mục 5).

---

## 5. `export.py` — Stage 5: render ra file

**Vai trò:** render deck (đã pass, hoặc `best_attempt` nếu hết retry) ra `.pptx`/`.pdf`. **Chỗ DUY NHẤT đọc
`image_base64`**. Không LLM.

### Tên file & phẳng hoá nội dung
- `_safe_stem(name)` — làm sạch query thành tên file hợp lệ (bỏ ký tự cấm, gộp khoảng trắng).
- `_slide_lines(slide)` — **phẳng hoá** nội dung 1 slide thành list dòng: bullets + slot (subtitle / stat+giải
  thích / cards / caption). Dùng cho cả pptx, pdf, và kiểm tra "slide rỗng".

### Lưới an toàn (ngoài instruction)
- `_backfill_empty_slides(slides, headings)` — slide nào **rỗng nội dung mà composer còn `content_points`** →
  điền content_points vào `bullets`. Chống ca **planner làm rớt nội dung**. Deterministic. Trả số slide đã vá.

### Đọc ảnh
- `_decode(image_index, image_id)` — giải base64 (cắt tiền tố `data:` nếu có) → `BytesIO`. Không thấy / lỗi →
  **log & bỏ qua**, không làm chết export.

### Mở & dọn template
- `_open_pptx(template)` — mở template `.pptx` (path hoặc bytes) làm nền, **xoá slide demo** bằng cả
  `part.drop_rel(rid)` **và** gỡ khỏi `sldIdLst` (tránh part mồ côi → PowerPoint đòi repair). Không template →
  deck trắng mặc định.

### Khớp layout theo "chữ ký placeholder" (ngoài instruction)
- `_WANT` — map **kiểu slide của planner** → `(min_body, cần_picture, ưu_tiên_center_title)`.
- `_roles(placeholders)` — chia placeholder thành `(titles, bodies, pictures)`, **bỏ qua chrome**
  (date/footer/số trang).
- `_catalog(prs)` — "chữ ký" mỗi layout template: đếm số ô title/body/picture + có center-title không.
- `_choose(cat, semantic, need_pic)` — chọn layout template hợp kiểu slide bằng **chấm điểm** (không theo TÊN,
  tên tùy hứng mỗi template). `need_pic=True` (slide có ảnh) → **ép layout có ô picture**. Luôn có fallback,
  không vỡ.

### Điền nội dung theo role
- `_card_line(card)` — 1 card → `"label: text"`, **chịu nhiều tên key** (`label/title/heading/name`,
  `text/description/body/detail`) phòng planner đặt khác chuẩn.
- `_write_lines(ph, lines)` — ghi list dòng vào 1 placeholder, bật `word_wrap` + **autofit** (co chữ vừa khung,
  không tràn).
- `_fill_bodies(slide, semantic, bodies)` — phân nội dung vào các ô body: `three_card` + nhiều ô → **mỗi card 1
  ô** (ô cuối gom phần dư); còn lại → dồn `_slide_lines` vào ô đầu. **Ô không có nội dung thì bỏ qua** (không
  đánh dấu used → bị dọn) → hết "Click to add".

### Đặt ảnh vào ô picture
- `_ph_box(slide, ph)` — tọa độ `(left, top, w, h)` của placeholder, **lấy từ layout nếu slide kế thừa None**.
- `_place_image(slide, slide, image_index, pics)` — **không dùng `insert_picture`** (lỗi trên vài template).
  Thay bằng: đọc box ô picture → `add_picture` **fit vào box, giữ tỉ lệ, canh giữa**. Không có ô picture →
  đặt góc phải an toàn. Để ô picture rỗng cho cleanup dọn.

### Render pptx
- `_render_pptx(slides, image_index, out_path, template)` — mỗi slide:
  1. `need_pic` = slide có ảnh giải mã được không.
  2. `_choose(...)` chọn layout → `add_slide`.
  3. Điền **title** (ô title), **body** (`_fill_bodies`), **ảnh** (`_place_image`).
  4. **Xoá mọi placeholder không điền** (diệt "Click to add"); **giữ hình trang trí** (không phải placeholder).

### Render pdf
- `_render_pdf(...)` — reportlab: title + bullet (`_slide_lines`) + ảnh góc phải. **PDF bỏ qua template** (chỉ
  pptx mới áp template hình thức).

### Điều phối
- `export_agent(state)`:
  1. Chọn deck: pass → `lecture_slides` + `composer_output`; fail → `best_attempt` (fallback state).
     `export_status` = `"verified"` | `"best_attempt_unverified"`.
  2. **`_backfill_empty_slides`** (dùng `composer.headings`).
  3. Dựng `image_index`; xuất ra `%TEMP%/slide_export/<stem>.<fmt>`.
  4. pdf → `_render_pdf`; pptx → `_render_pptx(..., template=state["slide_template_pptx"])`.
  5. Trả `lecture_output_path`, `lecture_output_bytes`, `export_status`, `final_output`.

---

## Phụ lục — nối dây (`slides_graph.py`)

- `init_node` — đặt `attempt=0`, `max_attempts` (mặc định 3), `best_attempt=None`.
- Cạnh cứng: `init → composer → planner → verification`.
- `route_after_verify`: `verification_passed` → **export**, else → **retry_control**.
- `retry_control_node`: cập nhật `best_attempt` (nếu `content_score` cao hơn), `attempt += 1`.
- `route_after_retry`: `attempt >= max_attempts` → **give_up (export best)**, else → **again (về composer,
  đọc `retry_scope`)**.
- `export → END`.

Retry là **vá cục bộ** (chỉ heading lỗi + slide của chúng regenerate), nên mỗi vòng rẻ dù chạm cả composer lẫn
planner.
