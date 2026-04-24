"""Markdown inspection report for Hamlet QA result JSONL files."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from hamlet_qa.io_utils import load_jsonl


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
        "| treatment | rows | mean evidence chunk recall | mean evidence quote recall | mean context tokens |",
        "|---|---:|---:|---:|---:|",
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
        lines.append(
            "| "
            f"{treatment} | {len(treatment_rows)} | "
            f"{_format_float(_mean(chunk_recalls))} | "
            f"{_format_float(_mean(quote_recalls))} | "
            f"{_format_float(_mean(context_tokens))} |"
        )

    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| question | treatment | context | quote recall | chunks | output preview |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for row in rows:
        output = (row.get("model_output") or "").replace("\n", " ").strip()
        if len(output) > 80:
            output = output[:77] + "..."
        chunks = ", ".join(row.get("selected_chunk_ids", []))
        if len(chunks) > 80:
            chunks = chunks[:77] + "..."
        lines.append(
            "| "
            f"{row['question_id']} | {row['treatment']} | "
            f"{row.get('context_tokens', 0)} | "
            f"{_format_float(row.get('evidence_quote_recall'))} | "
            f"{chunks or 'none'} | {output or 'n/a'} |"
        )

    return "\n".join(lines) + "\n"


def write_inspection_report(results_path: str | Path, output_path: str | Path) -> Path:
    rows = load_jsonl(results_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_inspection_report(rows), encoding="utf-8")
    return target
