"""Sufficient-context autorater (ICLR 2025, arXiv 2411.06037).

The prompt reproduces the Appendix C.1 autorater prompt (1-shot, with the
Roald Dahl railway-safety example and the trailing repeat of the
instructions). Documented deviations: the local reader model replaces Gemini
1.5 Pro, and the "Assume the queries have timestamp <TIMESTAMP>." sentence is
dropped (timestamps are meaningless for a fixed literary document).
"""

from __future__ import annotations

import json
import re
from typing import Any

from hamlet_qa.core.prompts import format_context_chunk

SUFFICIENT_CONTEXT_INSTRUCTIONS = (
    "You are an expert LLM evaluator that excels at evaluating a QUESTION and "
    "REFERENCES. Consider the following criteria:\n"
    "Sufficient Context: 1 IF the CONTEXT is sufficient to infer the answer "
    "to the question and 0 IF the CONTEXT cannot be used to infer the answer "
    "to the question\n"
    "First, output a list of step-by-step questions that would be used to "
    "arrive at a label for the criteria. Make sure to include questions about "
    "assumptions implicit in the QUESTION. Include questions about any "
    "mathematical calculations or arithmetic that would be required. Next, "
    "answer each of the questions. Make sure to work step by step through any "
    "required mathematical calculations or arithmetic. Finally, use these "
    "answers to evaluate the criteria. Output the ### EXPLANATION (Text). "
    "Then, use the EXPLANATION to output the ### EVALUATION (JSON)"
)

SUFFICIENT_CONTEXT_EXAMPLE = (
    "EXAMPLE:\n"
    "### QUESTION\n"
    "In which year did the publisher of Roald Dahl's Guide to Railway Safety "
    "cease to exist?\n"
    "### REFERENCES\n"
    "Roald Dahl's Guide to Railway Safety was published in 1991 by the "
    "British Railways Board. The British Railways Board had asked Roald Dahl "
    "to write the text of the booklet, and Quentin Blake to illustrate it, to "
    "help young people enjoy using the railways safely. The British Railways "
    "Board (BRB) was a nationalised industry in the United Kingdom that "
    "operated from 1963 to 2001. Until 1997 it was responsible for most "
    "railway services in Great Britain, trading under the brand name British "
    "Railways and, from 1965, British Rail. It did not operate railways in "
    "Northern Ireland, where railways were the responsibility of the "
    "Government of Northern Ireland.\n"
    "### EXPLANATION\n"
    "The context mentions that Roald Dahl's Guide to Railway Safety was "
    "published by the British Railways Board. It also states that the British "
    "Railways Board operated from 1963 to 2001, meaning the year it ceased to "
    "exist was 2001. Therefore, the context does provide a precise answer to "
    "the question.\n"
    "### JSON\n"
    '{"Sufficient Context": 1}'
)

SUFFICIENT_CONTEXT_PROMPT_TEMPLATE = (
    SUFFICIENT_CONTEXT_INSTRUCTIONS
    + "\n"
    + SUFFICIENT_CONTEXT_EXAMPLE
    + "\nRemember the instructions: "
    + SUFFICIENT_CONTEXT_INSTRUCTIONS
    + "\n### QUESTION\n{question}\n### REFERENCES\n{context}"
)

NO_CONTEXT_PLACEHOLDER = "[no context]"

_JSON_PATTERN = re.compile(
    r'\{[^{}]*"Sufficient Context"\s*:\s*([01])[^{}]*\}'
)
_FALLBACK_PATTERN = re.compile(r'"?Sufficient Context"?\s*:\s*([01])')


def context_text_for_row(row: dict[str, Any]) -> str:
    chunks = row.get("raw_chunks") or []
    if not chunks:
        return NO_CONTEXT_PLACEHOLDER
    return "\n\n".join(format_context_chunk(chunk) for chunk in chunks)


def build_sufficient_context_prompt(question: str, context: str) -> str:
    # str.format would choke on the literal JSON braces in the 1-shot example.
    return SUFFICIENT_CONTEXT_PROMPT_TEMPLATE.replace(
        "{question}", question
    ).replace("{context}", context)


def parse_autorater_output(raw_output: str) -> dict[str, Any]:
    label: int | None = None
    match = _JSON_PATTERN.search(raw_output)
    if match is not None:
        label = int(match.group(1))
    else:
        fallback = _FALLBACK_PATTERN.search(raw_output)
        if fallback is not None:
            label = int(fallback.group(1))
    explanation = ""
    explanation_match = re.search(
        r"###\s*EXPLANATION\s*(.+?)(?:###\s*(?:JSON|EVALUATION)|\Z)",
        raw_output,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if explanation_match is not None:
        explanation = explanation_match.group(1).strip()
    # Verify the JSON is loadable when present (robustness, not required).
    if label is not None and match is not None:
        try:
            json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"label": label, "explanation": explanation}


def compute_sufficient_context_for_row(
    row: dict[str, Any],
    rater_model: Any,
    max_tokens: int = 512,
) -> dict[str, Any]:
    prompt = build_sufficient_context_prompt(
        str(row["question"]),
        context_text_for_row(row),
    )
    try:
        raw_output = rater_model.generate(
            "You are a helpful assistant.",
            prompt,
            max_tokens=max_tokens,
        )
    except TypeError:
        raw_output = rater_model.generate("You are a helpful assistant.", prompt)
    parsed = parse_autorater_output(str(raw_output))
    return {
        "sufficient_context": parsed["label"],
        "sufficient_context_explanation": parsed["explanation"],
        "sufficient_context_raw_output": str(raw_output),
    }
