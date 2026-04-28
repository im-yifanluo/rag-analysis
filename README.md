# Hamlet QA Failure Analysis

This repo is a small, inspectable harness for studying why long-document QA
fails on one document: `hamlet.txt`.

The workflow is:

1. Split Hamlet into stable act/scene chunks.
2. Keep a small editable question set with required evidence quotes.
3. Derive gold chunk IDs automatically by quote matching.
4. Run closed-book, gold-evidence, dense, and sparse-retrieval treatments.
5. Inspect the full prompt, selected chunks, retrieval scores, evidence recall,
   quote recall, and model output for every question.

## Setup

Recommended on the research servers: use Miniforge/conda so the Python version
is explicit and independent of the system `python`.

```bash
bash setup.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hamlet-qa
```

If activation says `Run 'conda init' before 'conda activate'`, the environment
is fine; your shell just has not loaded conda's activation hook. Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hamlet-qa
```

If `bash setup.sh` says Python 3.12 and conda/mamba were not found, install
Miniforge in your home directory, then rerun setup:

```bash
curl -L -o Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
export PATH="$HOME/miniforge3/bin:$PATH"
bash setup.sh
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate hamlet-qa
```

The repo expects Python `3.12`; `.python-version` pins the local pyenv-style
version to `3.12.3`, and `environment.yml` creates a `hamlet-qa` conda env with
Python `3.12`. If conda/mamba is not available, `setup.sh` falls back to a local
`venv/` only when `python3.12` is installed:

```bash
source venv/bin/activate
```

The server run uses vLLM for generation. Implementation and tests do not run
reader-model inference. Qwen3.5's official model card recommends installing
vLLM from the nightly wheels for serving support; if the PyPI `vllm` package in
`requirements.txt` lags your server, install vLLM with:

```bash
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

## Regenerate Chunks

```bash
python -m hamlet_qa.build_chunks \
  --document hamlet.txt \
  --output data/hamlet_chunks.jsonl
```

Defaults:

- tokenizer: `Qwen/Qwen3.5-9B`
- chunk size: `256` tokens
- overlap: `64` tokens

Chunk IDs are stable for the default settings, for example
`act03_scene02_chunk004`.

The current chunking intentionally stays simple for the first diagnostic run.
TODO: future chunking should use speaker-turn-aware fixed-target chunks with a
hard max token cutoff.

## Run An Experiment

```bash
python -m hamlet_qa.run_experiment \
  --run-name qwen_hamlet_probe
```

Defaults:

- reader model: `Qwen/Qwen3.5-9B`
- tokenizer model: `Qwen/Qwen3.5-9B`
- embedding model: `Qwen/Qwen3-Embedding-8B`
- reranker model: `Qwen/Qwen3-Reranker-8B`
- temperature: `0.0`
- treatments: `closed_book gold_evidence dense_reranked dense_document_order dense_random_order sparse_bm25`
- context budgets: `1000`
- dense retrieval: Qwen embedding vectors in FAISS, then Qwen reranker scores
  define the final dense ranking
- GPU layout: `single`, which stages embedder, reranker, and reader on the
  default CUDA device so a one-GPU run remains intact
- dense prompt order variants: reranker rank, document order, deterministic
  random order
- sparse retrieval: BM25 over chunk text
- random seed: `13`

On a 3xA40 server, opt into the multi-GPU placement explicitly:

```bash
python -m hamlet_qa.run_experiment \
  --run-name qwen_hamlet_probe \
  --gpu-layout a40-3gpu
```

That preset uses `cuda:0` for the embedder, `cuda:1` for the reranker, and
`cuda:2` for the vLLM reader. It changes only model placement, not retrieval
selection, reranking, prompt ordering, or scoring logic.

Outputs are written to `runs/<run_name>/results.jsonl`, with copies of the
config, chunks, input questions, and quote-resolved questions used for the run.
The context budget counts context tokens only. Prompt overhead and generated
tokens are logged separately.

To build prompts and traces without vLLM generation:

```bash
python -m hamlet_qa.run_experiment --run-name dry_prompts --prepare-only
```

This still performs dense retrieval for grounded treatments so relevance
ordering and reranking are available when a dense treatment is selected, and
BM25 when `sparse_bm25` is selected. To disable reranking, pass
`--reranker-model none`. For a model-, embedder-, reranker-, and BM25-free CLI
smoke test, restrict treatments to `closed_book`; tests can inject cached/stub
retrievers.

## Inspect Results

For the interactive browser view, render the JSONL results into a standalone
HTML file:

```bash
python -m hamlet_qa.results_html \
  runs/qwen_hamlet_probe/results.jsonl \
  --output runs/qwen_hamlet_probe/results_viewer.html
```

Open `runs/qwen_hamlet_probe/results_viewer.html` in a browser. The viewer
groups results by question, filters by treatment or text search, and shows the
actual stored model outputs, evidence quotes, selected chunks, retrieval scores,
retrieval traces, prompt text, and expandable evidence chunks. By default it
embeds the chunk text from `hamlet_chunks.jsonl` beside the result file, then
falls back to the `chunks_path` recorded in the run config, then
`data/hamlet_chunks.jsonl`. To choose a specific chunk file:

```bash
python -m hamlet_qa.results_html \
  runs/qwen_hamlet_probe/results.jsonl \
  --chunks runs/qwen_hamlet_probe/hamlet_chunks.jsonl \
  --output runs/qwen_hamlet_probe/results_viewer.html
```

The file pickers in the viewer can load another `results.jsonl` or JSON result
file and another chunk JSONL/JSON file.

For a compact Markdown summary:

```bash
python -m hamlet_qa.inspect_results \
  --results runs/qwen_hamlet_probe/results.jsonl \
  --output runs/qwen_hamlet_probe/inspection.md
```

## Editable Data

- `data/hamlet_chunks.jsonl`: generated default chunks.
- `data/hamlet_questions.json`: starter questions and required evidence quotes.

Gold chunk IDs are derived from `required_evidence_quotes` during validation and
written to `hamlet_questions_resolved.json` inside each run directory. If
chunking settings change, regenerate chunks and confirm the quote validation
still resolves the intended evidence.

## Tests

```bash
python -m unittest discover -s tests
```
