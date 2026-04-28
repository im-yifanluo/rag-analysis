"""Render raw result JSON/JSONL rows into readable Markdown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_result_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = None

    if loaded is not None:
        if isinstance(loaded, list):
            rows: list[dict[str, Any]] = []
            for index, row in enumerate(loaded, start=1):
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object at array index {index - 1} in {source}")
                rows.append(row)
            return rows
        if isinstance(loaded, dict):
            return [loaded]
        raise ValueError(f"Expected JSON object or array in {source}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if not isinstance(row, dict):
            raise ValueError(f"Expected object on line {line_number} in {source}")
        rows.append(row)
    return rows


def code_block(value: Any, language: str = "text") -> str:
    body = "" if value is None else str(value).rstrip()
    fence = "```"
    while fence in body:
        fence += "`"
    return f"{fence}{language}\n{body}\n{fence}"


def bullet_values(values: list[Any] | None) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- {value}" for value in values]


def render_score(score: Any) -> str:
    if isinstance(score, float):
        return f"{score:.6f}"
    return str(score)


def render_required_quotes(row: dict[str, Any]) -> list[str]:
    quotes = row.get("required_quotes_present_in_context")
    if not quotes:
        quotes = row.get("required_evidence_quotes", [])
    if not quotes:
        return ["- none"]

    lines: list[str] = []
    for quote in quotes:
        matched_ids = ", ".join(quote.get("matched_chunk_ids", [])) or "none"
        present = quote.get("present")
        if present is None:
            present_text = "not recorded"
        else:
            present_text = "yes" if present else "no"
        role = quote.get("role", "unknown")
        quote_text = quote.get("quote", "")
        lines.append(f"- present: {present_text}; role: {role}; matched chunks: {matched_ids}")
        lines.append(f"  {quote_text}")
    return lines


def render_retrieval_scores(row: dict[str, Any]) -> list[str]:
    scores = row.get("retrieval_scores") or []
    if not scores:
        return ["- none"]

    lines = []
    for score in scores:
        lines.append(
            "- "
            f"rank {score.get('rank')}: {score.get('chunk_id')} "
            f"(score {render_score(score.get('score'))})"
        )
    return lines


def render_retrieval_trace(row: dict[str, Any], limit: int | None) -> list[str]:
    trace = row.get("retrieval_trace") or []
    if not trace:
        return ["- none"]

    visible = trace if limit is None else trace[:limit]
    lines = []
    for hit in visible:
        location = f"Act {hit.get('act')} Scene {hit.get('scene')}"
        lines.append(
            "- "
            f"rank {hit.get('rank')}: {hit.get('chunk_id')} "
            f"(score {render_score(hit.get('score'))}; {location}; "
            f"global_index {hit.get('global_index')})"
        )
    hidden = len(trace) - len(visible)
    if hidden > 0:
        lines.append(f"- {hidden} more retrieval hits omitted")
    return lines


def render_chunks(row: dict[str, Any]) -> list[str]:
    chunks = row.get("raw_chunks") or []
    if not chunks:
        return ["_No selected context chunks._"]

    lines: list[str] = []
    for chunk in chunks:
        lines.extend(
            [
                (
                    "#### "
                    f"{chunk.get('chunk_id')} "
                    f"(Act {chunk.get('act')} Scene {chunk.get('scene')}, "
                    f"{chunk.get('token_count')} tokens)"
                ),
                "",
                f"Scene title: {chunk.get('scene_title', '')}",
                "",
                code_block(chunk.get("text", ""), "text"),
                "",
            ]
        )
    return lines


def render_prompt_block(row: dict[str, Any]) -> list[str]:
    return [
        "#### System Prompt",
        "",
        code_block(row.get("system_prompt", ""), "text"),
        "",
        "#### User Prompt",
        "",
        code_block(row.get("user_prompt", ""), "text"),
        "",
        "#### Full Prompt",
        "",
        code_block(row.get("full_prompt", ""), "text"),
    ]


def render_row(
    row: dict[str, Any],
    index: int,
    include_prompts: bool,
    retrieval_limit: int | None,
) -> list[str]:
    treatment = row.get("treatment", "unknown_treatment")
    question_id = row.get("question_id", f"row_{index}")
    context_budget = row.get("context_budget", "n/a")

    lines = [
        f"## {index}. {question_id} / {treatment}",
        "",
        f"- run: {row.get('run_name', 'n/a')}",
        f"- timestamp_utc: {row.get('timestamp_utc', 'n/a')}",
        f"- context_budget: {context_budget}",
        f"- context_tokens: {row.get('context_tokens', 'n/a')}",
        f"- prompt_tokens: {row.get('prompt_tokens', 'n/a')}",
        f"- evidence_chunk_recall: {row.get('evidence_chunk_recall', 'n/a')}",
        f"- evidence_quote_recall: {row.get('evidence_quote_recall', 'n/a')}",
        f"- prompt_order: {row.get('prompt_order', 'n/a')}",
        "",
        "### Question",
        "",
        code_block(row.get("question", ""), "text"),
        "",
        "### Expected Answer",
        "",
        code_block(row.get("expected_answer", ""), "text"),
        "",
        "### Model Output",
        "",
        code_block(row.get("model_output", ""), "text"),
        "",
        "### Required Evidence Quotes",
        "",
        *render_required_quotes(row),
        "",
        "### Derived Gold Chunk IDs",
        "",
        *bullet_values(row.get("derived_gold_chunk_ids")),
        "",
        "### Selected Chunk IDs",
        "",
        *bullet_values(row.get("selected_chunk_ids")),
        "",
        "### Retrieval Scores For Selected Chunks",
        "",
        *render_retrieval_scores(row),
        "",
        "### Retrieval Trace",
        "",
        *render_retrieval_trace(row, retrieval_limit),
        "",
        "### Selected Context Chunk Text",
        "",
        *render_chunks(row),
    ]

    if include_prompts:
        lines.extend(["", "### Prompts", "", *render_prompt_block(row), ""])

    return lines


def render_results(
    rows: list[dict[str, Any]],
    source_path: str | Path,
    include_prompts: bool = False,
    retrieval_limit: int | None = 10,
) -> str:
    lines = [
        "# Raw Result Reader",
        "",
        f"Source: {source_path}",
        f"Rows: {len(rows)}",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.extend(render_row(row, index, include_prompts, retrieval_limit))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read raw Hamlet QA result JSON/JSONL and render actual fields as Markdown.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("results", help="Path to results.jsonl or a JSON file.")
    parser.add_argument(
        "--output",
        "-o",
        help="Markdown output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--include-prompts",
        action="store_true",
        help="Include system_prompt, user_prompt, and full_prompt blocks.",
    )
    parser.add_argument(
        "--retrieval-limit",
        type=int,
        default=10,
        help="Retrieval trace rows per result. Use 0 for the full trace.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieval_limit = None if args.retrieval_limit == 0 else args.retrieval_limit
    rows = load_result_rows(args.results)
    rendered = render_results(
        rows,
        args.results,
        include_prompts=args.include_prompts,
        retrieval_limit=retrieval_limit,
    )
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
        print(f"Wrote raw readable results to {target}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
