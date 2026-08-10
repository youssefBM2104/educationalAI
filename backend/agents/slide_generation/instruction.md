# Slide generation pipeline — implementation instructions

## Scope

This document covers the lecture slide generation module only: everything that happens **after** content has already been retrieved. The upstream RAG/KG module is treated as a black box — it populates `rag_chunks` and `kg_context` on the base `AgentState`, and nothing in this document re-fetches or re-derives that data. Composer and Verification work only with whatever the RAG module already returned for the request.

## Actual RAG output shape

Confirmed from a real RAG service payload. This is what `rag_chunks` and `kg_context` actually look like, and it shapes every stage below:

```json
{
  "query": "deadlock situation cooperating activities circularly block themselves",
  "course_id": "test1",
  "reranked": true,
  "chunks": [
    {
      "chunk_id": "ce6e7769-...",
      "score": 7.526,
      "text": "## Deadlock ... [Figure: The image shows a diagram ...]",
      "document_id": "c47044d2-...",
      "chunk_index": 38,
      "covers_concepts": [ { "id": "Deadlock" }, { "id": "Cooperating Activities" }, "..." ],
      "source": "vector",
      "images": [
        {
          "image_id": "eb9909ec-...",
          "page_number": 8,
          "bbox": { "left": 632.2, "top": 437.9, "right": 897.0, "bottom": 259.6 },
          "image_base64": "/9j/4AAQSkZJRg... (JPEG, can be tens of KB)",
          "mime_type": "image/jpeg",
          "vlm_description": "[Figure: The image shows a flowchart ...]"
        }
      ]
    }
  ],
  "kg_context": {
    "concepts": ["Deadlock", "Mutex", "Critical Section", "..."],
    "relations": [
      { "from": "Locks To Solve Mutual Exclusion", "type": "PART_OF", "to": "Passive Waiting Solutions" },
      { "from": "Wholesale Price", "type": "PREREQUISITE", "to": "Retail Price" }
    ]
  }
}
```

Key facts about this shape that drive the design:

- **No `topic_id`.** Each chunk carries `covers_concepts` — fine-grained concept names, often a dozen per chunk. Grouping chunks into headings means clustering over `covers_concepts`, not looking up a single tag.
- **`kg_context` is a flat graph, not a hierarchy.** A `concepts` list plus a `relations` list of typed edges (`PART_OF`, `DEFINES`, `APPLIES_TO`, `ILLUSTRATES`, `PREREQUISITE`, `EXTENDS`). Heading structure is inferred from graph structure each time — there is no pre-built tree.
- **`source: "vector"` vs `"kg"`** maps onto primary vs. supplementary content. Vector chunks are direct semantic hits (with a rerank `score` that can be strongly negative — e.g. `-11.19` — since the payload is reranked but not thresholded). KG chunks are graph-expansion hits. Never drop either based on score.
- **Images arrive inline as base64, not as a storage path.** Each image object is `{ image_id, page_number, bbox, image_base64, mime_type, vlm_description }`. There is no `image_path` — the actual image bytes travel *inside the state* as `image_base64`. `vlm_description` is a text rendering of the image for the LLM to reason about; it is NOT a substitute for showing the image.
- **Many images are irrelevant or decorative.** Real payloads include institutional logos, unrelated webpage screenshots, and off-topic GUI captures alongside the genuinely illustrative diagrams. Composer MUST filter images for relevance — this is load-bearing, not optional polish.

### Handling the heavy `image_base64` field

`image_base64` can be tens of KB per image (one observed at ~51KB as text). The images are NOT persisted anywhere — they exist only inline in the RAG payload — so the base64 must travel through the pipeline with the state; there is nothing to fetch from a database, and it cannot be discarded and re-fetched later.

**Design choice:** base64 stays inline on each image object for the whole pipeline (no separate image index, no intake split). This is the simplest fit for a stateless server, where state is serialized between stage hops anyway — the bytes were going to ride in that serialized state regardless.

**The one non-negotiable rule that replaces the old "strip at intake" step:** `image_base64` must NEVER be serialized into an LLM prompt. It stays in the state object, but when constructing the messages for any LLM stage (Composer, Slide Planner, Verification), exclude the `image_base64` field and send only the reasoning fields (`image_id`, `vlm_description`, `page_number`). A naïve `json.dumps(state)` into a prompt would sweep the base64 in — every prompt-construction site must explicitly drop it. Only Export reads `image_base64`.

Why this matters: feeding tens-of-KB blobs to the model on every call wastes tokens and, on an image-heavy request (the sample had 6 images), can blow the context window for no benefit — the model reasons about images from `vlm_description`, never from the bytes.

> Note: because images are non-persisted inline, a deck cannot be re-exported later without re-running the RAG query. If re-export or caching becomes a requirement, persist images at ingestion so `image_id` becomes a fetchable key — a future change, out of scope here.

---

## State schema

```python
class LectureState(AgentState):

    # --- Lecture request ---
    output_format: NotRequired[Literal["pptx", "pdf"]]
    slide_template: NotRequired[dict]        # lecturer-provided template spec

    # (No image index: image_base64 stays inline on each image object in
    #  rag_chunks. It rides through serialized state between stateless hops but
    #  is excluded when building any LLM prompt. Only Export reads it.)

    # --- Content Composer / Structure output ---
    composer_output: NotRequired[dict]
    # { headings: [ { heading_id, title, source_concepts, content_points, image_refs } ],
    #   coverage_check: { all_concepts, concepts_represented } }

    # --- Slide Planner output ---
    lecture_slides: NotRequired[dict]
    # { slides: [ { slide_id, heading_id, title, bullets, image_refs } ] }

    # --- Verification output ---
    verification_passed: NotRequired[bool]     # gated on content only
    verification_feedback: NotRequired[dict]
    # { content_score,
    #   issues: [ { type: "content"|"template"|"image", heading_id?, slide_id?, image_id?, issue } ] }
    #   ("content" issues gate + drive retries; "template"/"image" are advisory notes for the lecturer)
    retry_scope: NotRequired[dict]            # { heading_ids: [...] } — always Composer, content only

    # --- Loop control ---
    attempt: NotRequired[int]
    max_attempts: NotRequired[int]
    best_attempt: NotRequired[dict]          # { composer_output, lecture_slides, score }

    # --- Export output ---
    lecture_output_path: NotRequired[str]
    lecture_output_bytes: NotRequired[bytes]
    export_status: NotRequired[Literal["verified", "best_attempt_unverified"]]
```

`image_refs` everywhere in `composer_output` and `lecture_slides` is a list of `image_id` strings — never inline base64. The base64 lives on the original image objects inside `rag_chunks`; Export resolves an `image_id` back to its bytes by looking it up there.

---

## Stage 1 — Content Composer / Structure (chunks → composer_output)

**Purpose:** turn the retrieved chunks into a heading structure with fully-written content per heading, and select which images belong on which heading. Owns *what the deck covers, how it's organized, what each section says, and which figures illustrate it* — not slide count or layout.

**Inputs:** `rag_chunks` (image objects carry inline `image_base64`, but it is excluded when building this stage's prompt — Composer sees only `image_id` + `vlm_description`), `kg_context.concepts` + `.relations`, `query`; on retry: `verification_feedback` + `retry_scope.heading_ids`.

**Process (first pass):**
1. Build candidate headings by clustering concepts, using `kg_context.relations` — `PART_OF`/`DEFINES` edges suggest same heading; `PREREQUISITE` edges suggest ordering.
2. Assign chunks to headings via `covers_concepts`. A chunk may feed more than one heading.
3. Treat `source: "vector"` chunks as primary, `source: "kg"` as supplementary. Don't let rerank `score` decide inclusion.
4. **Image selection.** For each image available on a heading's assigned chunks, judge relevance from `vlm_description` against the heading's concepts. Attach only genuinely illustrative images (`image_refs`). Reject decorative/off-topic images — logos, unrelated screenshots, institutional branding — even though they were retrieved. This filter is required: real payloads contain many such images.
5. **Never convert an image into descriptive bullets as a replacement for showing it.** The `vlm_description` informs relevance and may seed a short caption, but the figure itself is carried as an `image_id` in `image_refs` so it can be displayed. Text-that-describes-a-diagram is the failure mode to avoid.
6. Write full content per heading; compress by summarization if oversized, never truncate by dropping chunks on score.
7. Every concept in any chunk's `covers_concepts` must be traceable to a heading (hard invariant, checked in Stage 3).

**Process (scoped retry):** regenerate content/image_refs ONLY for headings in `retry_scope.heading_ids`; copy all other headings through unchanged; recompute `coverage_check` over the merged set; set `retry_scope.heading_ids` to exactly what changed. A `heading_id: null` content issue means CREATE a new heading for an orphaned concept, not revise an existing one — this is a distinct code path from "revise heading h3."

**Output — `composer_output`:**
```json
{
  "headings": [
    {
      "heading_id": "h2",
      "title": "Deadlock detection",
      "source_concepts": ["Deadlock", "Waiting-For Graph", "Cycle In Waiting-For Graph"],
      "content_points": ["A deadlock exists when the waiting-for graph contains a cycle", "..."],
      "image_refs": ["eb9909ec-..."]
    }
  ],
  "coverage_check": {
    "all_concepts": ["Deadlock", "Waiting-For Graph", "Mutual Exclusion", "..."],
    "concepts_represented": ["Deadlock", "Waiting-For Graph", "..."]
  }
}
```

---

## Stage 2 — Slide Planner (composer_output → lecture_slides)

**Purpose:** turn composed content into a slide-by-slide deck, applying the template, including image placement.

**Inputs:** `composer_output`, `slide_template`; on retry: `verification_feedback` + `retry_scope`.

**Process (first pass):**
1. For each heading, pick the layout from `slide_template` whose slots best fit the heading's content (e.g. three parallel items → three-icon or three-card layout; a stat + explanation → chart layout; a diagram + text → two-column). Fill that layout's slots.
2. **Slide count per heading is a soft target (~1–2 slides), not a hard cap.** Most headings fit in one or two slides. If a heading's content genuinely needs more, do NOT truncate to hit the target and do NOT cram beyond a layout's slots — instead signal back that the heading should be **split** (the concept clustering in Stage 1 can divide it into two headings). Splitting preserves coverage while keeping any single heading from sprawling; capping would force dropping content, which violates the coverage invariant.
3. **Image placement.** Carry each heading's `image_refs` onto the slide(s) built from it. Decide layout — an image gets its own slide or shares one with a few bullets — per the chosen layout's image slot. This is layout only; do not re-describe the image in text.
4. **Images only when present.** A slide shows an image only if its heading has a real `image_id` in `image_refs`. If a heading has no relevant image, the slide is text-only — there is no generated or placeholder visual. Nothing to decide beyond "carry the refs the heading already has."

**Process (scoped retry):** after a Composer retry, rebuild slides ONLY for the headings in `retry_scope.heading_ids`; leave all other slides untouched. (There is no template-only retry path — template issues are advisory and never trigger regeneration.)

**Output — `lecture_slides`:**
```json
{
  "slides": [
    { "slide_id": "s3", "heading_id": "h2", "title": "Deadlock detection",
      "bullets": ["A cycle in the waiting-for graph means a deadlock"],
      "image_refs": ["eb9909ec-..."] }
  ]
}
```

---

## Stage 3 — Verification (lecture_slides → verification_passed / feedback / retry_scope)

**Purpose:** gate on **content only**. Template and image checks still run, but only to produce advisory polish notes — they never fail the deck or trigger a retry. The one thing that can't be fixed by eye after export is a missing or ungrounded concept; layout is obvious and editable in the slide tool, so it isn't worth gating on.

**Content score — per heading (the only gating check).** For each heading, confirm `content_points` trace back to its assigned `rag_chunks` (via `source_concepts`). Cross-check `coverage_check`: any concept in `all_concepts` absent from `concepts_represented` is an automatic content failure. Roll per-heading results up into `content_score`; keep per-heading detail in `issues` with `type: "content"`.

**Template notes — advisory only, never gate.** The `slide_template` is a *library of slide layouts* (title slide, two-column title+illustration, three-icon row, three-card, stat/chart, three-image caption). Verification may still flag layout-polish issues — a slot given more items than it holds, a dense slide, an empty required slot — as `{ type: "template", slide_id, issue }`, surfaced to the lecturer as suggestions to tidy the deck. These do NOT affect `verification_passed` and never cause a retry. There is no `template_score` gate.

**Image notes — advisory only, never gate.** If a heading had a strongly-relevant image (per `vlm_description`) that never made it into any slide's `image_refs`, emit `{ type: "image", heading_id, image_id, issue }`. Advisory suggestion only.

**Threshold and routing (single path):**
```
if content_score >= content_threshold:
    verification_passed = True    → Export (advisory notes attached for the lecturer)
else:
    verification_passed = False
    content_issues = [i for i in issues if i.type == "content"]
    retry_scope = { "heading_ids": [i.heading_id for i in content_issues] }
    → Stage 4   # retries always go to Composer; there is no other target
```
Only content can fail, so retries always route to Composer — there is no `retry_target` field. Template and image notes ride along in `issues` for the lecturer but are ignored by the retry logic.

---

## Stage 4 — Retry and best-attempt tracking

Runs whenever `verification_passed == False`.
1. **Update best attempt.** Use `content_score` (the only gating score). If it beats `best_attempt.score`, overwrite `best_attempt` with current `composer_output`, `lecture_slides`, score.
2. `attempt += 1`.
3. **Cap check.**
   - `attempt < max_attempts`: re-run Composer for the headings in `retry_scope.heading_ids`, then re-run Slide Planner for those headings' slides, then re-verify; everything else carries over unchanged.
   - `attempt >= max_attempts`: stop; Export uses `best_attempt.lecture_slides`.

Recommended `max_attempts`: 2–3. Retries are scoped patches (only the flagged headings and their slides regenerate), so each retry is cheap even though it touches both Composer and Slide Planner.

---

## Stage 5 — Export

**Inputs:** the passing `lecture_slides` (or `best_attempt.lecture_slides`), plus `rag_chunks` (which still carry the inline image objects with `image_base64`).

**Process:**
1. Render to `output_format` (`pptx` or `pdf`).
2. **Decode and embed images here and only here.** Build a lookup from the inline image objects across `rag_chunks` (`image_id → { image_base64, mime_type }`). For each `image_id` in a slide's `image_refs`, decode `image_base64` using `mime_type` and embed the actual picture into the slide. This is the only stage that reads the base64.
3. If an `image_id` in `image_refs` can't be found among the `rag_chunks` image objects (shouldn't happen, but guard for it), skip that image and log it rather than failing the whole export.
4. Set `export_status`: `"verified"` for a passing deck, `"best_attempt_unverified"` for the retries-exhausted path — in the latter case surface the last `verification_feedback.issues` alongside the deck.

---

## Summary of hard invariants

- Every concept in `rag_chunks[].covers_concepts` must trace to a heading (Stage 1) then a slide (Stage 2) — checked per-heading in Stage 3 via `coverage_check`.
- Heading structure is inferred from `kg_context.relations` each time — there is no pre-built tree.
- `image_base64` stays inline on the image objects throughout the pipeline, but is excluded at every LLM prompt-construction site (never `json.dumps` the whole state into a prompt) and is read only at Export via `image_id`.
- Images are shown as real pictures (`image_refs` → embedded at Export), never replaced by descriptive text. Decorative/off-topic images are filtered out by Composer.
- A slide shows a real image only when its heading has a relevant `image_id`; otherwise it is text-only. There is no generated or placeholder visual.
- Verification gates on **content only**. Template and image checks produce advisory notes for the lecturer but never fail the deck or trigger a retry.
- Slide count per heading is a soft ~1–2 target; a heading with too much content is **split into two headings**, never truncated to fit the target.
- Only content can fail, so retries always route to Composer (scoped to `heading_ids`), then re-run Slide Planner for those headings' slides. There is no `retry_target` — the path is single. Retries are surgical patches, not full regenerations.
- `best_attempt` tracks both `composer_output` and `lecture_slides`, updated on every failed retry.
- The retry loop has a hard cap; exhausting it always exports the best attempt, labeled honestly via `export_status`.
