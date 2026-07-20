from backend.agents.exam_generation.generator_agent import PATH_LENGTH_RANGES


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _relation_types(kg_context: dict | None) -> set[str]:
    return {
        str(r.get("type", "")).lower()
        for r in (kg_context or {}).get("relations", [])
        if r.get("type")
    }


def check_question(q: dict, question_type: str, kg_context: dict | None) -> list[str]:
    """Return a list of structural violations for one question (empty == clean)."""
    issues: list[str] = []
    path = q.get("kg_path") or []

    # kg_path must be concepts only — the generator sometimes interleaves relation types
    rel_types = _relation_types(kg_context)
    leaked = [c for c in path if str(c).lower() in rel_types]
    if leaked:
        issues.append(f"kg_path contains relation types, not just concepts: {leaked}")

    # Path length is how difficulty is defined, so it must sit inside the band.
    difficulty = q.get("difficulty")
    if difficulty in PATH_LENGTH_RANGES:
        lo, hi = PATH_LENGTH_RANGES[difficulty]
        concepts = [c for c in path if str(c).lower() not in rel_types]
        if not (lo <= len(concepts) <= hi):
            issues.append(
                f"path length {len(concepts)} outside {difficulty} range {(lo, hi)}"
            )

    if question_type == "mcq":
        choices = q.get("choices") or {}
        if sorted(choices) != ["A", "B", "C", "D"]:
            issues.append(f"choices must be exactly A-D, got {sorted(choices)}")
        correct = q.get("correct_answer")
        if correct not in choices:
            issues.append(f"correct_answer {correct!r} is not one of the choices")
        texts = [_norm(t) for t in choices.values()]
        if len(set(texts)) != len(texts):
            issues.append("duplicate option text")
    else:
        if not (q.get("model_answer") or "").strip():
            issues.append("essay question has no model answer")
        if not (q.get("marking_scheme") or "").strip():
            issues.append("essay question has no marking scheme")

    if not (q.get("question") or "").strip():
        issues.append("empty question text")

    return issues


def check_exam_set(result: dict) -> dict:
    """Structural checks across the whole exam set."""
    exam_set = result.get("exam_set") or []
    target = result.get("num_questions_target")
    question_type = result.get("question_type")
    kg_context = result.get("kg_context")

    issues: list[str] = []

    if target is not None and len(exam_set) != target:
        issues.append(f"produced {len(exam_set)} questions, target was {target}")

    # Duplicate questions / duplicate reasoning paths — we saw two questions generated from
    # the identical kg_path, which makes them near-identical for the student.
    seen_q, seen_path = {}, {}
    for i, q in enumerate(exam_set, 1):
        key = _norm(q.get("question", ""))
        if key in seen_q:
            issues.append(f"Q{i} duplicates the question text of Q{seen_q[key]}")
        seen_q[key] = i

        pkey = tuple(str(c).lower() for c in (q.get("kg_path") or []))
        if pkey and pkey in seen_path:
            issues.append(f"Q{i} reuses the kg_path of Q{seen_path[pkey]}: {list(pkey)}")
        seen_path[pkey] = i

    per_question = {
        i: check_question(q, question_type, kg_context)
        for i, q in enumerate(exam_set, 1)
    }

    difficulty_mix: dict[str, int] = {}
    for q in exam_set:
        d = q.get("difficulty", "?")
        difficulty_mix[d] = difficulty_mix.get(d, 0) + 1

    total_issues = len(issues) + sum(len(v) for v in per_question.values())
    return {
        "set_issues": issues,
        "per_question": per_question,
        "difficulty_mix": difficulty_mix,
        "total_issues": total_issues,
    }
