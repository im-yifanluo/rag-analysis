"""Reader-as-judge support scoring (shared).

For each (evidence node, candidate unit) pair the reader model judges, from the
candidate text alone, how well that text supports the information need. The
judgement is parsed, validated against the claimed supporting span, capped, and
cached. The labels double as the training signal for a future learned scorer, so
the interface (`SupportScorer.score`) is model-agnostic.

Shared in `core/` so multiple features can use it without depending on each
other. The prompt version string is kept stable so existing caches stay valid.
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Protocol

from hamlet_qa.core.evidence.schema import EvidenceNode, EvidenceUnit, SupportScore
from hamlet_qa.core.llm_cache import JsonKVCache, stable_hash

TEACHER_PROMPT_VERSION = "reader_support.teacher.v1"

SUPPORT_TYPES = {"none", "related", "partial", "complete", "contradictory"}

SUPPORT_TEACHER_SYSTEM = (
    "You are a strict evidence adjudicator. You judge ONLY from the candidate "
    "text shown to you. You do not use any outside knowledge, you do not answer "
    "the question, and you never invent text that is not present."
)

SUPPORT_TEACHER_TEMPLATE = """Judge how well the CANDIDATE TEXT supports the EVIDENCE NEED.

EVIDENCE NEED:
{need}

CANDIDATE TEXT:
\"\"\"
{unit_text}
\"\"\"

Scoring rubric (support_score):
- 0.0  = irrelevant, or contradicts the need
- 0.25 = same topic/entities but does NOT answer the need
- 0.5  = partial support, useful but incomplete
- 0.75 = mostly supports the need, may need a little local context
- 1.0  = directly and sufficiently supports the need

Rules:
- Judge only from the CANDIDATE TEXT. Do not use outside knowledge of the work.
- Do not answer the overall question; only assess support for THIS need.
- If you claim partial/complete support, quote the exact supporting substring (copied verbatim from the candidate). If you cannot quote it, the support is not there.
- If the text is only topically related but does not answer the need, use support_type "related" and a score near 0.25.

Respond with JSON ONLY in exactly this shape:
{{"support_score": 0.0, "support_type": "none|related|partial|complete|contradictory", "supporting_span": "exact substring or empty", "needs_more_context": false, "explanation": "brief"}}"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def build_teacher_prompt(need: str, unit_text: str) -> str:
    return SUPPORT_TEACHER_TEMPLATE.format(need=need, unit_text=unit_text)


def _coerce_score(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_support_output(raw_output: str) -> dict[str, Any]:
    """Extract the support JSON; report a parse_error instead of raising."""
    match = _JSON_OBJECT.search(raw_output or "")
    if match is None:
        return {"parse_error": "no JSON object found", "fields": None}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        return {"parse_error": f"json decode error: {error}", "fields": None}
    if not isinstance(parsed, dict):
        return {"parse_error": "JSON was not an object", "fields": None}
    return {"parse_error": None, "fields": parsed}


def _is_exact_substring(span: str, unit_text: str) -> bool:
    if not span:
        return False
    if span in unit_text:
        return True
    # Be lenient about whitespace differences only (still source-faithful).
    norm_span = re.sub(r"\s+", " ", span).strip()
    norm_text = re.sub(r"\s+", " ", unit_text).strip()
    return bool(norm_span) and norm_span in norm_text


def validate_and_cap(
    fields: dict[str, Any] | None,
    unit_text: str,
    parse_error: str | None,
) -> dict[str, Any]:
    """Clamp, type-normalize, and apply the span-faithfulness caps."""
    warnings: list[str] = []
    if fields is None:
        return {
            "support_score": 0.0,
            "support_type": "none",
            "supporting_span": "",
            "needs_more_context": False,
            "explanation": "",
            "validation_warnings": ["unparseable teacher output -> score 0.0"],
        }

    score = _coerce_score(fields.get("support_score"))
    if score is None:
        score = 0.0
        warnings.append("missing/invalid support_score -> 0.0")
    support_type = str(fields.get("support_type", "")).strip().lower()
    if support_type not in SUPPORT_TYPES:
        warnings.append(f"unknown support_type '{support_type}'")
        support_type = "none"
    span = str(fields.get("supporting_span", "") or "")
    needs_more = bool(fields.get("needs_more_context", False))
    explanation = str(fields.get("explanation", "")).strip()

    score = max(0.0, min(1.0, score))

    if support_type == "contradictory":
        score = 0.0
    if support_type in {"partial", "complete"} and not span.strip():
        if score > 0.7:
            warnings.append("partial/complete with empty span -> capped at 0.7")
        score = min(score, 0.7)
    if span.strip() and not _is_exact_substring(span, unit_text):
        if score > 0.5:
            warnings.append("supporting_span not found in candidate -> capped at 0.5")
        score = min(score, 0.5)

    return {
        "support_score": score,
        "support_type": support_type,
        "supporting_span": span,
        "needs_more_context": needs_more,
        "explanation": explanation,
        "validation_warnings": warnings,
    }


class SupportScorer(Protocol):
    """Model-agnostic interface; a trained scorer can replace the teacher."""

    def score(
        self, question_text: str, node: EvidenceNode, unit: EvidenceUnit
    ) -> SupportScore: ...


def _call_model(model: Any, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    signature = inspect.signature(model.generate)
    if "max_tokens" in signature.parameters:
        return str(model.generate(system_prompt, user_prompt, max_tokens=max_tokens))
    return str(model.generate(system_prompt, user_prompt))


class ReaderTeacherSupportScorer:
    """Uses the resident reader model as the support teacher, with caching."""

    def __init__(
        self,
        selector_model: Any,
        cache: JsonKVCache,
        max_tokens: int = 384,
        prompt_version: str = TEACHER_PROMPT_VERSION,
    ):
        self.selector_model = selector_model
        self.cache = cache
        self.max_tokens = max_tokens
        self.prompt_version = prompt_version
        self.model_name = str(getattr(selector_model, "model_name", "reader_model"))

    def _cache_key(self, question_text: str, node: EvidenceNode, unit: EvidenceUnit) -> str:
        return stable_hash(
            {
                "question": question_text,
                "node_need": node.need,
                "node_query": node.node_query,
                "unit_text": unit.text,
                "model": self.model_name,
                "prompt_version": self.prompt_version,
            }
        )

    def score(
        self, question_text: str, node: EvidenceNode, unit: EvidenceUnit
    ) -> SupportScore:
        cache_key = self._cache_key(question_text, node, unit)
        cached = self.cache.get(cache_key)
        if cached is not None and "support_score" in cached:
            return SupportScore(
                node_id=node.node_id,
                unit_id=unit.unit_id,
                cache_hit=True,
                **{k: cached[k] for k in (
                    "support_score",
                    "support_type",
                    "supporting_span",
                    "needs_more_context",
                    "explanation",
                    "raw_output",
                    "parse_error",
                    "validation_warnings",
                ) if k in cached},
            )

        prompt = build_teacher_prompt(node.need, unit.text)
        raw_output = _call_model(
            self.selector_model, SUPPORT_TEACHER_SYSTEM, prompt, self.max_tokens
        )
        parsed = parse_support_output(raw_output)
        validated = validate_and_cap(parsed["fields"], unit.text, parsed["parse_error"])
        record = {
            **validated,
            "raw_output": raw_output,
            "parse_error": parsed["parse_error"],
            "prompt_version": self.prompt_version,
        }
        self.cache.set(cache_key, record)
        self.cache.save()
        return SupportScore(
            node_id=node.node_id,
            unit_id=unit.unit_id,
            support_score=validated["support_score"],
            support_type=validated["support_type"],
            supporting_span=validated["supporting_span"],
            needs_more_context=validated["needs_more_context"],
            explanation=validated["explanation"],
            raw_output=raw_output,
            parse_error=parsed["parse_error"],
            validation_warnings=validated["validation_warnings"],
            cache_hit=False,
        )


class TrainedSupportScorer:
    """Placeholder for a learned scorer trained on teacher labels."""

    def __init__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "TrainedSupportScorer is a future drop-in for ReaderTeacherSupportScorer; "
            "no trained checkpoint is wired up yet."
        )

    def score(
        self, question_text: str, node: EvidenceNode, unit: EvidenceUnit
    ) -> SupportScore:  # pragma: no cover - not implemented
        raise NotImplementedError
