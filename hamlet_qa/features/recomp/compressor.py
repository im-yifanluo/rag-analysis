"""RECOMP compressors (arXiv 2310.04408), ported from the official code.

Extractive: a Contriever-style dual encoder scores query/sentence pairs by
the dot product of mean-pooled embeddings, exactly mirroring
`third_party/RECOMP/recomp/run_extractive_compressor.py::get_contriever_scores`.
The top-N sentences (paper: 5 for HotpotQA) become the compressed context.

Abstractive: a T5 seq2seq checkpoint generates a summary from the input
format used by the official training script
(`train_hf_summarization_model.py`: "Question: {}\\n Document: {}\\n Summary: ").
An empty summary is a valid output (selective augmentation). An optional
prompted mode reuses the reader LLM with the paper's Table 8 GPT-3.5 prompt.

Deviations from the original RECOMP are documented in each treatment's
context_assembly_trace under "deviations".
"""

from __future__ import annotations

import re
from typing import Any, Protocol

# Verbatim from RECOMP paper Table 8 (NQ/TQA prompt used to query GPT-3.5).
RECOMP_PROMPTED_ABSTRACTIVE_PROMPT = (
    "Compress the information in the retrieved documents into a 2-sentence "
    "summary that could be used to answer the question: "
    "Question: {question} Retrieved documents: {documents} "
    "Compressed documents:"
)

# Input format from the official abstractive training/inference script.
RECOMP_ABSTRACTIVE_INPUT_FORMAT = "Question: {question}\n Document: {document}\n Summary: "

_SENTENCE_FALLBACK_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z‘“'\"(\[])")


class SentenceScorerLike(Protocol):
    """Embedding scorer contract; satisfied by ExtractiveCompressor and stubs."""

    model_name: str

    def score_sentences(self, query: str, sentences: list[str]) -> list[float]:
        ...


class Seq2SeqSummarizerLike(Protocol):
    model_name: str

    def summarize(self, source_text: str) -> str:
        ...


def split_sentences(text: str) -> list[str]:
    """Split chunk text into sentences (NLTK when available, regex fallback).

    Newlines are collapsed first because Hamlet is verse: the official code
    operates on Wikipedia prose, so sentence boundaries here are a documented
    fidelity risk either way.
    """
    flattened = re.sub(r"\s+", " ", text).strip()
    if not flattened:
        return []
    try:
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(flattened)
    except (ImportError, LookupError):
        sentences = _SENTENCE_FALLBACK_PATTERN.split(flattened)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


class ExtractiveCompressor:
    """Contriever-style dual-encoder sentence scorer (official checkpoint)."""

    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self._torch = torch

    def _embed(self, texts: list[str]) -> Any:
        # Port of the official mean_pooling: mask-aware mean of token states.
        torch = self._torch
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**inputs)
                token_embeddings = outputs[0]
                mask = inputs["attention_mask"]
                token_embeddings = token_embeddings.masked_fill(
                    ~mask[..., None].bool(), 0.0
                )
                pooled = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
                embeddings.append(pooled.detach().cpu())
        return torch.cat(embeddings, dim=0)

    def score_sentences(self, query: str, sentences: list[str]) -> list[float]:
        if not sentences:
            return []
        embeddings = self._embed([query] + sentences)
        query_embedding = embeddings[0]
        return [
            float(query_embedding @ embeddings[index + 1])
            for index in range(len(sentences))
        ]


class AbstractiveCompressor:
    """Official T5 abstractive compressor checkpoint wrapper."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_source_length: int = 1024,
        max_summary_tokens: int = 512,
    ):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.max_source_length = max_source_length
        self.max_summary_tokens = max_summary_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self._torch = torch

    def summarize(self, source_text: str) -> str:
        with self._torch.no_grad():
            inputs = self.tokenizer(
                source_text,
                max_length=self.max_source_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_summary_tokens,
                num_beams=1,
                do_sample=False,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def compress_extractive(
    query: str,
    input_chunks: list[dict[str, Any]],
    scorer: SentenceScorerLike,
    top_sentences: int,
) -> dict[str, Any]:
    """Score all sentences of the input chunks, keep top-N in score order."""
    sentence_records: list[dict[str, Any]] = []
    for chunk in input_chunks:
        for sentence in split_sentences(str(chunk.get("text", ""))):
            sentence_records.append(
                {"chunk_id": str(chunk.get("chunk_id")), "sentence": sentence}
            )
    scores = scorer.score_sentences(
        query,
        [record["sentence"] for record in sentence_records],
    )
    for record, score in zip(sentence_records, scores):
        record["score"] = float(score)
    ranked = sorted(sentence_records, key=lambda record: -record["score"])
    selected = ranked[: max(0, top_sentences)]
    summary = " ".join(record["sentence"] for record in selected)
    return {
        "summary": summary,
        "selected_sentences": selected,
        "num_input_sentences": len(sentence_records),
        "compressor_model": scorer.model_name,
    }


def compress_abstractive_t5(
    query: str,
    input_chunks: list[dict[str, Any]],
    summarizer: Seq2SeqSummarizerLike,
) -> dict[str, Any]:
    document = " ".join(
        re.sub(r"\s+", " ", str(chunk.get("text", ""))).strip()
        for chunk in input_chunks
    )
    source_text = RECOMP_ABSTRACTIVE_INPUT_FORMAT.format(
        question=query,
        document=document,
    )
    summary = summarizer.summarize(source_text)
    return {
        "summary": summary,
        "compressor_input": source_text,
        "compressor_model": summarizer.model_name,
    }


def build_prompted_abstractive_prompt(
    query: str,
    input_chunks: list[dict[str, Any]],
) -> str:
    documents = " ".join(
        re.sub(r"\s+", " ", str(chunk.get("text", ""))).strip()
        for chunk in input_chunks
    )
    return RECOMP_PROMPTED_ABSTRACTIVE_PROMPT.format(
        question=query,
        documents=documents,
    )
