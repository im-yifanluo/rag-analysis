"""CRAG corrective query rewriting (keyword extraction).

Ports the popqa few-shot keyword-extraction prompt from
`third_party/CorrectiveRAG/CRAG/scripts/utils.py::extract_keywords` verbatim.
The released code formats the template with double braces, which leaves a
literal "{question}" in the prompt; the intended substitution (used here) is
the actual question. GPT-3.5-turbo is replaced by the local reader model
(documented deviation).
"""

from __future__ import annotations

from typing import Any

from hamlet_qa.core.llm_cache import JsonKVCache, stable_hash


CRAG_KEYWORD_PROMPT = (
    "Extract at most three keywords separated by comma from the following "
    "dialogues and questions as queries for the web search, including topic "
    "background within dialogues and main intent within questions. \n\n"
    "question: What is Henry Feilden's occupation?\n"
    "query: Henry Feilden, occupation\n\n"
    "question: In what city was Billy Carlson born?\n"
    "query: city, Billy Carlson, born\n\n"
    "question: What is the religion of John Gwynn?\n"
    "query: religion of John Gwynn\n\n"
    "question: What sport does Kiribati men's national basketball team play?\n"
    "query: sport, Kiribati men's national basketball team play\n\n"
    "question: {question}\nquery: "
)


def rewrite_query(
    question_text: str,
    rewriter_model: Any,
    cache_path: Any = None,
) -> dict[str, Any]:
    """Return the keyword query used by the corrective re-retrieval."""
    prompt = CRAG_KEYWORD_PROMPT.format(question=question_text)
    model_name = str(getattr(rewriter_model, "model_name", "reader_model"))
    cache = JsonKVCache(cache_path, section="crag_rewrites")
    cache_key = stable_hash({"prompt": prompt, "model": model_name})
    cached = cache.get(cache_key)
    if cached is not None:
        return dict(cached, cache_hit=True)
    raw_output = rewriter_model.generate(
        "You are a helpful assistant.",
        prompt,
    )
    # The few-shot examples end with "query: "; keep only the first line of
    # the completion so trailing chatter does not pollute the BM25 query.
    rewritten = str(raw_output).strip().splitlines()[0].strip() if raw_output else ""
    record = {
        "rewritten_query": rewritten or question_text,
        "raw_output": str(raw_output),
        "rewriter_model": model_name,
    }
    cache.set(cache_key, record)
    cache.save()
    return dict(record, cache_hit=False)
