"""Calibrate CRAG action thresholds from an existing run's reranker scores.

The original CRAG uses a fine-tuned T5 evaluator with dataset-tuned score
thresholds. This harness substitutes the Qwen reranker as the evaluator
(documented deviation), so the Correct/Ambiguous/Incorrect thresholds must be
re-derived for the reranker's logit scale. This CLI reads `dense_reranked`
rows from a results.jsonl (each carries the full top-k retrieval trace with
`rerank_score` plus `derived_gold_chunk_ids`), labels candidates gold vs
non-gold within the top `--ndocs`, and proposes:

- upper threshold: smallest observed score whose gold precision at or above
  it meets `--target-precision` (Correct should be safe to trust);
- lower threshold: the `--non-gold-percentile` of non-gold scores (below it,
  candidates are confidently junk, triggering the Incorrect action).

Pure CPU / offline. The chosen values are printed alongside the resulting
per-question action distribution so they can be baked into config defaults.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hamlet_qa.core.config import DEFAULT_CRAG_NDOCS


def load_dense_rows(results_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("treatment") != "dense_reranked":
                continue
            question_id = str(row.get("question_id"))
            if question_id in seen_questions:
                continue
            seen_questions.add(question_id)
            rows.append(row)
    return rows


def labeled_scores(
    rows: list[dict[str, Any]],
    ndocs: int,
) -> tuple[list[tuple[str, float, bool]], list[str]]:
    """Return (question_id, score, is_gold) for top-ndocs candidates."""
    labeled: list[tuple[str, float, bool]] = []
    skipped: list[str] = []
    for row in rows:
        gold = set(row.get("derived_gold_chunk_ids", []))
        trace = row.get("retrieval_trace") or []
        usable = [item for item in trace[:ndocs] if item.get("rerank_score") is not None]
        if not usable or not gold:
            skipped.append(str(row.get("question_id")))
            continue
        for item in usable:
            labeled.append(
                (
                    str(row.get("question_id")),
                    float(item["rerank_score"]),
                    str(item["chunk_id"]) in gold,
                )
            )
    return labeled, skipped


def precision_at_threshold(
    labeled: list[tuple[str, float, bool]],
    threshold: float,
) -> tuple[float, int]:
    kept = [is_gold for _qid, score, is_gold in labeled if score >= threshold]
    if not kept:
        return 0.0, 0
    return sum(kept) / len(kept), len(kept)


def derive_upper_threshold(
    labeled: list[tuple[str, float, bool]],
    target_precision: float,
) -> float | None:
    candidate_scores = sorted({score for _qid, score, _is_gold in labeled})
    for threshold in candidate_scores:
        precision, kept = precision_at_threshold(labeled, threshold)
        if kept and precision >= target_precision:
            return threshold
    return None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("No values to take a percentile of")
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def action_for_scores(scores: list[float], upper: float, lower: float) -> str:
    flags = [2 if score >= upper else 1 if score >= lower else 0 for score in scores]
    if any(flag == 2 for flag in flags):
        return "correct"
    if any(flag == 1 for flag in flags):
        return "ambiguous"
    return "incorrect"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive CRAG thresholds from reranker scores in a past run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        default="runs/qwen_hamlet_probe/results.jsonl",
        help="results.jsonl containing dense_reranked rows with rerank scores.",
    )
    parser.add_argument("--ndocs", type=int, default=DEFAULT_CRAG_NDOCS)
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.9,
        help="Minimum gold precision required at or above the upper threshold.",
    )
    parser.add_argument(
        "--non-gold-percentile",
        type=float,
        default=0.9,
        help="Non-gold score percentile used as the lower threshold.",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    rows = load_dense_rows(results_path)
    if not rows:
        raise SystemExit(f"No dense_reranked rows found in {results_path}")
    labeled, skipped = labeled_scores(rows, args.ndocs)
    if not labeled:
        raise SystemExit("No labeled (score, gold) pairs found in the trace rows.")

    gold_scores = [score for _qid, score, is_gold in labeled if is_gold]
    non_gold_scores = [score for _qid, score, is_gold in labeled if not is_gold]
    upper = derive_upper_threshold(labeled, args.target_precision)
    lower = percentile(non_gold_scores, args.non_gold_percentile)
    if upper is None:
        raise SystemExit(
            f"No threshold reaches gold precision {args.target_precision}; "
            "lower --target-precision."
        )
    if lower > upper:
        lower = upper

    print(f"Run: {results_path}")
    print(f"Questions used: {len(rows) - len(skipped)} (skipped: {skipped or 'none'})")
    print(
        f"Candidates labeled: {len(labeled)} "
        f"(gold {len(gold_scores)}, non-gold {len(non_gold_scores)})"
    )
    precision, kept = precision_at_threshold(labeled, upper)
    print(
        f"\nProposed thresholds:\n"
        f"  upper = {upper:.4f}  (gold precision {precision:.3f} over {kept} kept)\n"
        f"  lower = {lower:.4f}  "
        f"(non-gold p{int(args.non_gold_percentile * 100)})"
    )

    print("\nResulting CRAG action per question:")
    by_question: dict[str, list[float]] = {}
    for question_id, score, _is_gold in labeled:
        by_question.setdefault(question_id, []).append(score)
    for question_id, scores in sorted(by_question.items()):
        action = action_for_scores(scores, upper, lower)
        print(f"  {question_id}: {action} (max score {max(scores):.3f})")

    print(
        "\nBake these into hamlet_qa/core/config.py as "
        "DEFAULT_CRAG_UPPER_THRESHOLD / DEFAULT_CRAG_LOWER_THRESHOLD, "
        "or pass --crag-upper-threshold / --crag-lower-threshold at run time."
    )


if __name__ == "__main__":
    main()
