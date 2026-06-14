"""MacRAG chunk summarization (offline index build).

The persona / instruction / assistant-example prompt is taken verbatim from
`third_party/MacRAG/MacRAG/src/gen_index_macrag.py`. The official pipeline
queries GPT-4o; this harness substitutes the local vLLM reader model
(documented deviation).
"""

from __future__ import annotations

import json
import re
from typing import Any

MACRAG_SUMMARY_PERSONA = '''
You are a senior reporter and have been working in the media industry for a long time.
Your primary objective is for a given text, summarize it to an appropriate length.
As an analytical and systematic thinker, please carefully summarize the documents.
'''

MACRAG_SUMMARY_INSTRUCTION = '''
Use the following steps and respond to user inputs.
Please think step by step and do not restate each step before proceeding.

Step 1:
The user will provide document information.

Step 2:
Please proceed with the summary in the following order based on the given document
    1) Please write the Title and Key words of documents.
    2) Organize subheadings with considering given page_content, but make sure to mention what you're covering. If you have tables or graph then please include the table titles or column information and explain the contents in the subheadings.
    3) Summarize the given page_content while maintaining contexts with specified information in details from the given texts and the detailed information of numbers as much as possible.
    4) Output Length of Summary should be less then length 500. ex) "keypoint" -> length is 8
    5) Please think by Document so if input is two Document information then output also should be two.


Step 3:
Provide the final output in JSON format as follows:
[
    {"Title":"...", "Keywords":"...", "Subheadings":"...", "Summary":"..."}
]
Please be careful about number of output, it depends on the number of input.
Please double check that the keys in the outputs are unique, that is, there is only one of each "Title,"  "Keywords", "Subheadings", and "Summary" as the keys in the output.

Do not display the output from Step 1, Step 2 and provide only the outputs of Step 3.
'''

MACRAG_SUMMARY_ASSISTANT_EXAMPLE = '''
[
    {
    "Title": "Anarchism: Philosophy, History, and Modern Resurgence",
    "Keywords": "Anarchism, Anti-Authority, Stateless Societies, Revolutionary Strategies, Modern Resurgence",
    "Subheadings": "From Enlightenment Roots to Modern Movements: The Historical and Ideological Development of Anarchism",
    "Summary": "Anarchism is a political philosophy opposing all forms of authority, aiming to abolish institutions like the state and capitalism. It advocates for stateless societies and voluntary associations. Modern anarchism emerged from the Enlightenment and was influential in late 19th and early 20th-century workers' struggles, participating in revolutions like the Paris Commune and Spanish Civil War. The movement resurged in the late 20th and early 21st centuries within anti-capitalist and anti-globalization movements, using both revolutionary and evolutionary strategies."
    }
]
'''


def build_summary_user_prompt(chunk_text: str) -> str:
    return (
        MACRAG_SUMMARY_INSTRUCTION
        + "\n\nExample output:\n"
        + MACRAG_SUMMARY_ASSISTANT_EXAMPLE
        + "\n\nDocument information:\npage_content: "
        + chunk_text
    )


def parse_summary_response(raw_output: str) -> dict[str, Any] | None:
    """Parse the Step-3 JSON; return None when no usable record is found."""
    match = re.search(r"\[.*\]|\{.*\}", raw_output, flags=re.DOTALL)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    record = parsed[0] if isinstance(parsed, list) and parsed else parsed
    if not isinstance(record, dict):
        return None
    summary = str(record.get("Summary", "")).strip()
    if not summary:
        return None
    return {
        "title": str(record.get("Title", "")).strip(),
        "keywords": str(record.get("Keywords", "")).strip(),
        "subheadings": str(record.get("Subheadings", "")).strip(),
        "summary": summary,
    }


def summarize_chunk(
    chunk_text: str,
    summarizer_model: Any,
    max_retries: int = 1,
) -> dict[str, Any]:
    """Summarize one chunk; fall back to the first 500 chars on failure."""
    user_prompt = build_summary_user_prompt(chunk_text)
    raw_output = ""
    for _attempt in range(max_retries + 1):
        raw_output = summarizer_model.generate(MACRAG_SUMMARY_PERSONA, user_prompt)
        parsed = parse_summary_response(str(raw_output))
        if parsed is not None:
            return dict(parsed, raw_output=str(raw_output), fallback=False)
    return {
        "title": "",
        "keywords": "",
        "subheadings": "",
        "summary": chunk_text[:500],
        "raw_output": str(raw_output),
        "fallback": True,
    }
