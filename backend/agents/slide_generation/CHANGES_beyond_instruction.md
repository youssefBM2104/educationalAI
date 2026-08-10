# Những thay đổi ngoài `instruction.md` (Export & áp template hình thức)

`instruction.md` mô tả **logic pipeline** (5 stage, schema, invariant) và dừng Export ở mức
*"Render to pptx/pdf, decode & embed image"* (mục 5). Nó **không** nói gì về việc **áp một template
`.pptx` hình thức thật** (theme, layout, placeholder, hình trang trí). Tất cả mục dưới đây là vấn đề
**phát sinh khi chạy thực tế với template thật** và cách đã xử lý. File này là bản ghi những phần đó.

Tất cả thay đổi nằm ở [`export.py`](export.py) trừ khi ghi chú khác.

---

## 0. State mới: template hình thức theo từng request

**Ngoài instruction.** `instruction.md` chỉ có `slide_template` = *dict thư viện layout* (mô tả slot cho
planner). Nó **không có** khái niệm template `.pptx` trực quan.

- Thêm field `slide_template_pptx` vào `LectureState` ([state.py](../state.py)): đường dẫn `.pptx`
  (hoặc bytes) **mỗi user gửi kèm query** → mỗi người một mẫu riêng.
- Export dựng deck **trên nền template này** để thừa hưởng theme (màu/font/nền) + layout + hình trang trí.
- `slide_template` (dict layout-library) và `slide_template_pptx` (mẫu trực quan) là **hai thứ khác nhau**,
  cùng tồn tại.

---

## 1. Chọn layout template theo "chữ ký placeholder" (tách nội dung ↔ hình thức)

**Vấn đề:** một `.pptx` có hàng chục layout, mỗi layout một thiết kế/màu. Nếu ép mọi slide vào **1 layout**
→ deck **chỉ 1 tông màu**, phí sự đa dạng của template.

**Đã làm:** `_catalog()` + `_choose()` map **kiểu cấu trúc slide của planner** (`title`, `title_bullets`,
`three_card`, `stat`, `two_column_image`, `image_caption`) → layout template có **chữ ký placeholder** phù
hợp (số ô title / body / picture), **KHÔNG match theo tên layout** (tên tùy hứng mỗi template). Bảng `_WANT`
khai báo mỗi kiểu cần bao nhiêu body / có cần ô picture / có ưu tiên center-title.

**Kết quả:** mỗi kiểu slide đáp xuống một layout khác nhau của template → **deck nhiều tông màu + giữ hình
trang trí**, template lạ vẫn chạy (thiếu khuôn lý tưởng thì fallback về layout bullet, không vỡ).

---

## 2. Điền theo ROLE + xóa placeholder rỗng (diệt "Click to add")

**Vấn đề:** cách cũ **vẽ textbox đè** lên placeholder → placeholder rỗng vẫn hiện chữ nhắc
*"Bấm để thêm tiêu đề/nội dung"* + icon bảng/ảnh che nội dung. Trên nền tối, textbox tự vẽ còn bị **chữ đen
tàng hình**.

**Đã làm:** điền thẳng vào placeholder của template theo **role** (`_roles()` chia title / body / picture,
bỏ qua chrome date/footer/number): title ← tiêu đề, body ← bullets/slot. Nhờ điền vào placeholder nên chữ
**thừa hưởng màu/font theo theme** (đọc được trên mọi nền). Sau đó **xóa mọi placeholder không điền**
(`ph._element.getparent().remove(...)`) → hết "Click to add". **Hình trang trí (không phải placeholder) giữ
nguyên.**

---

## 3. Đặt ảnh vào đúng "ô picture" của template

**Vấn đề:** ảnh bị **floating đè lên title/body**. Hai nguyên nhân:
1. Planner gán heading có ảnh vào layout **không có ô picture** → ảnh không có chỗ.
2. `PicturePlaceholder.insert_picture()` của python-pptx **lỗi trên một số template** (`'NoneType'…'ph'`).

**Đã làm:**
- Slide có ảnh **ép chọn layout có ô picture** (`_choose(..., need_pic=True)`).
- Bỏ `insert_picture`. Thay bằng `_place_image()`: **đọc tọa độ ô picture** (từ layout nếu slide kế thừa
  `None` — `_ph_box()`), rồi `add_picture` **fit vào đúng khung, giữ tỉ lệ, canh giữa**. Ô picture rỗng để
  cleanup xóa. Template không có ô picture → fallback đặt góc phải an toàn.

---

## 4. Dọn slot thừa khi nội dung ít hơn ô (three_card)

**Vấn đề:** layout 2 cột có thể có **4 ô body** nhưng chỉ có 3 card → ô thứ 4 nhận nội dung rỗng nhưng vẫn bị
đánh dấu "đã dùng" → **không bị xóa** → lại hiện "Click to add".

**Đã làm:** trong `_fill_bodies()`, ô nào **không có nội dung thì bỏ qua** (không đánh dấu used) → cleanup xóa.

---

## 5. Lưới an toàn: slide rỗng do planner làm rớt nội dung

**Vấn đề (không thuộc invariant của instruction):** Slide Planner (LLM) thỉnh thoảng sinh slide có title mà
**bullets/cards rỗng** → slide trống. Đây **không phải coverage-gap** (khái niệm vẫn nằm trong heading của
composer), nên Verification ở Stage 3 **không bắt được** — instruction.md giả định content-gap luôn là coverage.

**Đã làm:** `_backfill_empty_slides()` chạy trong `export_agent`: slide nào **rỗng nội dung mà composer vẫn còn
`content_points`** cho heading đó → **lấy `content_points` điền vào bullets**. Deterministic, không phụ thuộc
planner may rủi. Có log `backfilled N empty slide(s)`.

Phụ trợ: `_card_line()` chịu được nhiều tên key (`label/title/heading/name`, `text/description/body/detail`)
phòng planner đặt tên slot khác chuẩn.

---

## 6. Strip slide demo đúng cách (tránh PowerPoint báo hỏng file)

**Vấn đề:** template mang sẵn slide demo; xóa thiếu quan hệ (`r:id`) để lại **part mồ côi** → PowerPoint đòi
"repair" khi mở.

**Đã làm:** `_open_pptx()` xóa slide demo bằng cả `part.drop_rel(rid)` **và** gỡ khỏi `sldIdLst` → không còn
orphan, mở sạch.

---

## 7. Giới hạn đã biết — template kiểu SlidesGo/Canva (CHƯA xử lý)

Có **2 họ template**:
- **Design-on-layout (Microsoft):** thiết kế nằm ở master/layout → slide mới tự thừa hưởng → **chạy tốt**.
- **Design-on-slide (SlidesGo/Canva):** thiết kế (các khối blob màu) nằm **trên từng slide demo** dưới dạng
  group shapes, **layout trống trơn**. Pipeline xóa slide demo → tạo slide mới từ layout trống → **ra trắng,
  mất màu**.

**Hiện trạng:** chưa hỗ trợ họ thứ hai. Muốn hỗ trợ phải đổi cách render sang **clone slide demo có sẵn thiết
kế rồi thay chữ** (clone XML + copy quan hệ ảnh) — thay đổi lớn, còn để ngỏ chờ quyết định.

---

## Tóm tắt vấn đề đã giải quyết

| # | Vấn đề thực tế | Trạng thái |
|---|---|---|
| 0 | Template hình thức theo từng user (`slide_template_pptx`) | ✅ |
| 1 | Deck chỉ 1 tông màu (ép 1 layout) | ✅ map theo chữ ký placeholder |
| 2 | "Click to add" + icon che nội dung | ✅ điền theo role + xóa ô rỗng |
| 3 | Ảnh floating đè chữ / `insert_picture` lỗi | ✅ đặt vào ô picture, fit + canh giữa |
| 4 | Ô body thừa hiện "Click to add" | ✅ bỏ qua ô rỗng |
| 5 | Slide trống do planner rớt nội dung | ✅ backfill từ `content_points` |
| 6 | PowerPoint đòi repair file | ✅ strip demo + drop_rel |
| 7 | Template SlidesGo/Canva mất màu | ⏳ chưa hỗ trợ (cần clone slide demo) |
