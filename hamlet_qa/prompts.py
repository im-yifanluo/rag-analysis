"""Prompt construction for the Hamlet QA treatments."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "You are a careful research assistant studying Hamlet. Use the provided "
    "context when context is present. If no context is provided, answer from "
    "your own knowledge and be concise."
)


def format_context_chunk(chunk: dict[str, Any]) -> str:
    header = (
        f"[{chunk['chunk_id']} | Act {chunk['act']} Scene {chunk['scene']} | "
        f"{chunk['scene_title']} | {chunk['token_count']} tokens]"
    )
    return f"{header}\n{chunk['text'].strip()}"


def build_user_prompt(question: str, selected_chunks: list[dict[str, Any]]) -> str:
    if not selected_chunks:
        return (
            f"Question: {question}\n\n"
            "Answer the question without document context."
        )

    context = "\n\n".join(format_context_chunk(chunk) for chunk in selected_chunks)
    return (
        "Context chunks:\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer based only on the context chunks above. If the answer is not "
        "supported by the context, say that the provided context does not answer it."
    )


def fallback_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    return f"<system>\n{system_prompt}\n</system>\n<user>\n{user_prompt}\n</user>\n<assistant>\n"


class TokenizerPromptFormatter:
    """Formats prompts with a chat tokenizer without loading a reader model."""

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

    def format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return fallback_chat_prompt(system_prompt, user_prompt)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))
