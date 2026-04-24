# Hamlet QA Failure Analysis

This repo is a small, inspectable harness for studying why long-document QA
fails on one document: `hamlet.txt`.

It deliberately does not include the previous benchmark code. The workflow is:

1. Split Hamlet into stable act/scene chunks.
2. Keep a small editable question set with gold chunk IDs.
3. Run closed-book, gold-evidence, and dense-retrieval treatments.
4. Inspect the full prompt, selected chunks, retrieval scores, evidence recall,
   and model output for every question.

## Setup

```bash
bash setup.sh
source venv/bin/activate
```

The server run uses vLLM for generation. Implementation and tests do not run
reader-model inference.

## Regenerate Chunks

```bash
python -m hamlet_qa.build_chunks \
  --document hamlet.txt \
  --output data/hamlet_chunks.jsonl
```

Defaults:

- tokenizer: `Qwen/Qwen2.5-7B-Instruct`
- chunk size: `256` tokens
- overlap: `64` tokens

Chunk IDs are stable for the default settings, for example
`act03_scene02_chunk004`.

## Run An Experiment

```bash
python -m hamlet_qa.run_experiment \
  --run-name first_hamlet_probe
```

Defaults:

- reader model: `Qwen/Qwen2.5-7B-Instruct`
- embedding model: `Snowflake/snowflake-arctic-embed-m-v1.5`
- temperature: `0.0`
- treatments: `closed_book gold_evidence dense_relevance`
- context budgets: `500 1000 2000`

Outputs are written to `runs/<run_name>/results.jsonl`, with copies of the
config, chunks, and questions used for the run.

To build prompts and traces without vLLM generation:

```bash
python -m hamlet_qa.run_experiment --run-name dry_prompts --prepare-only
```

This still performs dense retrieval when `dense_relevance` is enabled. For a
model-free prompt smoke test, restrict treatments to `closed_book` or
`gold_evidence`.

## Editable Data

- `data/hamlet_chunks.jsonl`: generated default chunks.
- `data/hamlet_questions.json`: starter questions and gold chunk IDs.

If chunking settings change, regenerate chunks and update the question file's
gold chunk IDs.

## Tests

```bash
python -m unittest discover -s tests
```
