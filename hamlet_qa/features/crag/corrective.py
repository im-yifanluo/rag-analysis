"""CRAG core logic (arXiv 2401.15884), ported from the official scripts.

- `extract_strips_from_psg` is an exact port of
  `third_party/CorrectiveRAG/CRAG/scripts/internal_knowledge_preparation.py`
  (fixed_num 50-word windows with <10-word remainder merge; excerption split
  on '?' then '. ' with <=5-word strips appended to the previous strip and 3
  sentence-strips concatenated; selection keeps the whole passage).
- `select_relevant_strips` ports `utils.py::select_relevants`: strips shorter
  than 4 words score -1.0, the rest are evaluator-scored, and the top_n
  strips are joined with '; ' in descending score order. top_n is 3 for
  selection mode and 6 for the decomposed modes (per the official code; the
  paper text is looser).
- `action_from_scores` ports `CRAG_Inference.py::process_flag` (any doc >=
  upper -> correct; elif any >= lower -> ambiguous; else incorrect) and the
  flag -> knowledge mapping (correct -> internal, incorrect -> external,
  ambiguous -> combined "Knowledge1: ... [sep] Knowledge2: ..." from
  `combined_knowledge_preparation.py`).

Documented deviation: the original fine-tuned T5-large evaluator is replaced
by the harness reranker (Qwen3-Reranker-8B) with thresholds recalibrated via
`hamlet_qa/cli/calibrate_crag.py`; web search is replaced by keyword-rewrite
plus BM25 over the whole document (closed single-document corpus).
"""

from __future__ import annotations

from typing import Any, Protocol


class EvaluatorLike(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]:
        ...


def extract_strips_from_psg(psg: str, mode: str = "excerption") -> list[str]:
    if mode == "fixed_num":
        final_strips: list[str] = []
        window_length = 50
        words = psg.split(" ")
        buf: list[str] = []
        for word in words:
            buf.append(word)
            if len(buf) == window_length:
                final_strips.append(" ".join(buf))
                buf = []
        if buf:
            if len(buf) < 10 and final_strips:
                final_strips[-1] += " " + " ".join(buf)
            else:
                final_strips.append(" ".join(buf))
        return final_strips
    if mode == "excerption":
        num_concatenate_strips = 3
        question_strips = psg.split("?")
        origin_strips: list[str] = []
        for question_strip in question_strips:
            origin_strips += question_strip.split(". ")
        strips: list[str] = []
        for strip in origin_strips:
            if strip in strips:
                continue
            if not strips:
                strips.append(strip)
            else:
                if len(strip.split()) > 5:
                    strips.append(strip)
                else:
                    # Official behavior: short strips are appended to the
                    # previous strip without a separator.
                    strips[-1] += strip
        final_strips = []
        buf = []
        for strip in strips:
            buf.append(strip)
            if len(buf) == num_concatenate_strips:
                final_strips.append(" ".join(buf))
                buf = []
        if buf:
            final_strips.append(" ".join(buf))
        return final_strips
    if mode == "selection":
        return [psg]
    raise ValueError(f"Unknown CRAG decompose mode: {mode}")


def top_n_for_mode(mode: str) -> int:
    return 3 if mode == "selection" else 6


def select_relevant_strips(
    strips: list[str],
    query: str,
    evaluator: EvaluatorLike,
    top_n: int,
) -> dict[str, Any]:
    """Score strips and join the top_n by '; ' in descending score order."""
    scored: list[tuple[float, str, int]] = []
    to_score: list[tuple[int, str]] = []
    for index, strip in enumerate(strips):
        if len(strip.split()) < 4:
            scored.append((-1.0, strip, index))
        else:
            to_score.append((index, strip))
    if to_score:
        scores = evaluator.score(query, [strip for _index, strip in to_score])
        for (index, strip), score in zip(to_score, scores):
            scored.append((float(score), strip, index))
    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    selected = ranked[:top_n]
    return {
        "refined_text": "; ".join(item[1] for item in selected),
        "strip_scores": [
            {"index": item[2], "score": item[0], "strip": item[1]}
            for item in ranked
        ],
        "selected_indices": [item[2] for item in selected],
    }


def refine_passages(
    passages: list[str],
    query: str,
    evaluator: EvaluatorLike,
    decompose_mode: str,
) -> dict[str, Any]:
    strips: list[str] = []
    for passage in passages:
        strips += extract_strips_from_psg(passage, mode=decompose_mode)
    return select_relevant_strips(
        strips,
        query,
        evaluator,
        top_n=top_n_for_mode(decompose_mode),
    )


def doc_flags(scores: list[float], upper_threshold: float, lower_threshold: float) -> list[int]:
    return [
        2 if score >= upper_threshold else 1 if score >= lower_threshold else 0
        for score in scores
    ]


def action_from_scores(
    scores: list[float],
    upper_threshold: float,
    lower_threshold: float,
) -> str:
    flags = doc_flags(scores, upper_threshold, lower_threshold)
    if any(flag == 2 for flag in flags):
        return "correct"
    if any(flag == 1 for flag in flags):
        return "ambiguous"
    return "incorrect"


def combine_knowledge(internal: str, external: str) -> str:
    """Official ambiguous-action format from combined_knowledge_preparation.py."""
    return f"Knowledge1: {internal} [sep] Knowledge2: {external}"
