"""Markdown inspection report for Hamlet QA result JSONL files."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from hamlet_qa.core.io import load_jsonl
from hamlet_qa.metrics.annotate import load_annotations, merge_annotations_into_rows


def load_results_with_annotations(results_path: str | Path) -> list[dict[str, Any]]:
    """Load result rows, merging metric sidecar fields when present."""
    rows = load_jsonl(results_path)
    annotations = load_annotations(results_path)
    if not annotations:
        return rows
    return merge_annotations_into_rows(rows, annotations)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def render_inspection_report(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "# Hamlet QA Inspection Report\n\nNo result rows found.\n"

    by_treatment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_treatment[str(row["treatment"])].append(row)

    lines = [
        "# Hamlet QA Inspection Report",
        "",
        f"Rows: {len(rows)}",
        f"Questions: {len({row['question_id'] for row in rows})}",
        "",
        "## Treatment Summary",
        "",
        "| treatment | rows | mean evidence chunk recall | mean evidence quote recall | mean context tokens | sufficient context rate | mean CI+ fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for treatment, treatment_rows in sorted(by_treatment.items()):
        chunk_recalls = [
            float(row["evidence_chunk_recall"])
            for row in treatment_rows
            if row.get("evidence_chunk_recall") is not None
        ]
        quote_recalls = [
            float(row["evidence_quote_recall"])
            for row in treatment_rows
            if row.get("evidence_quote_recall") is not None
        ]
        context_tokens = [
            float(row["context_tokens"])
            for row in treatment_rows
            if row.get("context_tokens") is not None
        ]
        sufficient_labels = [
            float(row["sufficient_context"])
            for row in treatment_rows
            if row.get("sufficient_context") is not None
        ]
        ci_fractions = [
            float(row["ci_positive_fraction"])
            for row in treatment_rows
            if row.get("ci_positive_fraction") is not None
        ]
        lines.append(
            "| "
            f"{treatment} | {len(treatment_rows)} | "
            f"{_format_float(_mean(chunk_recalls))} | "
            f"{_format_float(_mean(quote_recalls))} | "
            f"{_format_float(_mean(context_tokens))} | "
            f"{_format_float(_mean(sufficient_labels))} | "
            f"{_format_float(_mean(ci_fractions))} |"
        )

    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| question | treatment | context | quote recall | suff | chunks | output preview |",
            "|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        output = (row.get("model_output") or "").replace("\n", " ").strip()
        if len(output) > 80:
            output = output[:77] + "..."
        chunks = ", ".join(row.get("selected_chunk_ids", []))
        if len(chunks) > 80:
            chunks = chunks[:77] + "..."
        sufficient = row.get("sufficient_context")
        lines.append(
            "| "
            f"{row['question_id']} | {row['treatment']} | "
            f"{row.get('context_tokens', 0)} | "
            f"{_format_float(row.get('evidence_quote_recall'))} | "
            f"{'n/a' if sufficient is None else sufficient} | "
            f"{chunks or 'none'} | {output or 'n/a'} |"
        )

    return "\n".join(lines) + "\n"


def write_inspection_report(results_path: str | Path, output_path: str | Path) -> Path:
    rows = load_results_with_annotations(results_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_inspection_report(rows), encoding="utf-8")
    return target
