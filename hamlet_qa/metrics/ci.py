"""Oracle Contextual Influence (CI) value (arXiv 2509.21359).

Faithful to the leave-one-out refiner in
`third_party/InfluenceGuided_RAG/RAG-CSM/flashrag/refiner/refiner.py`:
utility v(S) is the negative mean token cross-entropy loss of the gold answer
given prompt(question, S); the CI value of chunk c_i in selected context C is

    phi_i = loss(C \\ c_i) - loss(C)

so phi_i > 0 means removing c_i hurts (the chunk helps), matching the
official `new_score[j] - total_score[i]` sign convention. Per the paper, the
hyperparameter-free CI selection rule keeps exactly the chunks with positive
CI values, reported here as `ci_positive_chunk_ids`.

Documented deviations: one gold answer (`expected_answer`) instead of a min
over an answer set; the gold answer is a free-form sentence (mean token loss
mitigates length effects); log-likelihoods come from vLLM prompt_logprobs.
"""

from __future__ import annotations

from typing import Any, Protocol

from hamlet_qa.core.prompts import HamletQAPromptBuilder


class CompletionScorerLike(Protocol):
    def format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        ...

    def count_tokens(self, text: str) -> int:
        ...

    def score_completion(self, full_prompt: str, completion: str) -> dict:
        ...


def answer_loss(
    scorer: CompletionScorerLike,
    question: str,
    selected_chunks: list[dict[str, Any]],
    treatment: str,
    expected_answer: str,
) -> float:
    bundle = HamletQAPromptBuilder().build(
        question,
        selected_chunks,
        treatment,
        scorer,
    )
    scored = scorer.score_completion(bundle.full_prompt, expected_answer)
    return -float(scored["mean_logprob"])


def compute_ci_for_row(
    row: dict[str, Any],
    scorer: CompletionScorerLike,
) -> dict[str, Any]:
    """Return CI annotation fields for one result row."""
    chunks = [dict(chunk) for chunk in row.get("raw_chunks") or []]
    if not chunks:
        return {
            "ci_base_loss": None,
            "ci_values": None,
            "ci_positive_chunk_ids": None,
            "ci_positive_fraction": None,
        }
    question = str(row["question"])
    treatment = str(row["treatment"])
    expected_answer = str(row["expected_answer"])

    base_loss = answer_loss(scorer, question, chunks, treatment, expected_answer)
    ci_values: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        remaining = chunks[:index] + chunks[index + 1 :]
        loss_without = answer_loss(
            scorer,
            question,
            remaining,
            treatment,
            expected_answer,
        )
        ci_values.append(
            {
                "chunk_id": str(chunk.get("chunk_id")),
                "phi": loss_without - base_loss,
                "loss_without": loss_without,
            }
        )
    positive_ids = [item["chunk_id"] for item in ci_values if item["phi"] > 0]
    return {
        "ci_base_loss": base_loss,
        "ci_values": ci_values,
        "ci_positive_chunk_ids": positive_ids,
        "ci_positive_fraction": len(positive_ids) / len(ci_values),
    }
