"""Prompt construction for the Hamlet QA treatments."""

from __future__ import annotations

from typing import Any

CLOSED_BOOK_SYSTEM_PROMPT = (
    "You are a careful research assistant studying Hamlet. No document context "
    "is provided."
)
GROUNDED_SYSTEM_PROMPT = (
    "You are a careful research assistant studying Hamlet. Use the provided "
    "context as evidence."
)
SYSTEM_PROMPT = GROUNDED_SYSTEM_PROMPT


def format_context_chunk(chunk: dict[str, Any]) -> str:
    header = (
        f"[{chunk['chunk_id']} | Act {chunk['act']} Scene {chunk['scene']} | "
        f"{chunk['scene_title']} | {chunk['token_count']} tokens]"
    )
    return f"{header}\n{chunk['text'].strip()}"


def build_user_prompt(
    question: str,
    selected_chunks: list[dict[str, Any]],
    closed_book: bool = False,
) -> str:
    if closed_book:
        return (
            f"Question: {question}\n\n"
            "Answer the question. No document context is provided. Return a concise answer."
        )

    context = "\n\n".join(format_context_chunk(chunk) for chunk in selected_chunks)
    if not context:
        context = "[no context chunks selected]"
    return (
        "Context chunks:\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the provided context. Cite the chunk IDs "
        "that provide the evidence for the answer. If the answer is not supported "
        "by the context, say that the provided context does not answer it."
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
        self.model_max_context = self._resolve_model_max_context()

    def _resolve_model_max_context(self) -> int | None:
        value = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(value, int) and 0 < value < 10_000_000:
            return value
        return None

    def format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if "qwen3" in self.model_name.lower():
                kwargs["enable_thinking"] = False
            try:
                return self.tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                return self.tokenizer.apply_chat_template(messages, **kwargs)
        return fallback_chat_prompt(system_prompt, user_prompt)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))
