# Hamlet QA Failure Analysis

This repo is a small, inspectable harness for studying why long-document QA
fails on one document: `data/hamlet.txt`.

The workflow is:

1. Split Hamlet into stable act/scene chunks.
2. Keep a small editable question set with required evidence quotes.
3. Derive gold chunk IDs automatically by quote matching.
4. Run closed-book, gold-evidence, dense, sparse-retrieval, and lightweight
   post-retrieval context-assembly treatments.
5. Inspect the full prompt, selected chunks, retrieval scores, evidence recall,
   quote recall, and model output for every question.

## Setup

This repo uses Miniforge/conda as the environment manager so the Python version
and package set are explicit and independent of the system `python`.

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

If `bash setup.sh` says Miniforge/conda was not found, install Miniforge in
your home directory, then rerun setup:

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
version to `3.12.3`, and `environment.yml` creates or updates a `hamlet-qa`
conda env with Python `3.12` plus non-GPU dependencies. `setup.sh` then
installs PyTorch, vLLM, Transformers, SentenceTransformers, and FAISS with
`uv pip install --torch-backend=cu129`. On Linux x86_64, it pins vLLM to the
matching `vllm-0.22.0+cu129` wheel so vLLM and PyTorch load the same CUDA
runtime family. To run without activating the shell:

```bash
conda run -n hamlet-qa python -m unittest discover -s tests
```

The server run uses vLLM for generation. Implementation and tests do not run
reader-model inference. If you see a PyTorch error like `The NVIDIA driver on
your system is too old`, or a vLLM import error like `libcudart.so.13: cannot
open shared object file`, repair only the GPU Python packages inside the
activated env:

```bash
uv pip install \
  --torch-backend=cu129 \
  --reinstall-package torch \
  --reinstall-package vllm \
  torch \
  "https://github.com/vllm-project/vllm/releases/download/v0.22.0/vllm-0.22.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
uv pip uninstall -y torchvision torchaudio
```

This project does not use TorchVision/TorchAudio. If Transformers fails with
`operator torchvision::nms does not exist`, remove the incompatible optional
vision/audio packages from the activated env:

```bash
uv pip uninstall -y torchvision torchaudio
```

## Pipeline Structure

The source code is organized around the experiment boundary this repo studies:
post-retrieval context assembly.

- `hamlet_qa/core/`: stable RAG pipeline code shared by every method. This
  includes config, chunking, question loading, retrieval, prompt construction,
  generation, experiment orchestration, result logging, and the shared
  `ContextAssemblyRequest` / `ContextAssemblyResult` contract.
- `hamlet_qa/features/`: context assembly methods. Each method gets its own
  folder and owns the logic for turning candidate chunks into prompt-ready
  context. Current folders are `baseline/`, `ordering/`, `setr/`, and
  `domain/`.
- `hamlet_qa/features/registry.py`: maps CLI treatment names to feature
  adapters and declares whether a method needs dense retrieval, sparse
  retrieval, or a domain graph.
- `hamlet_qa/cli/` and `hamlet_qa/inspection/`: command-line and result-viewing
  utilities around the core pipeline.

The core pipeline always follows the same path: load chunks/questions, build
retrieval traces, call the registered context assembly feature, build the model
prompt, and log artifacts uniformly. To add a future method such as MMR,
RankRAG, RECOMP, or IRCoT, create a new `hamlet_qa/features/<method>/` folder
with an `assemble_*` adapter and add one `TreatmentSpec` row to
`hamlet_qa/features/registry.py`.

## Regenerate Chunks

```bash
python -m hamlet_qa.cli.build_chunks \
  --document data/hamlet.txt \
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
python -m hamlet_qa.cli.run_experiment \
  --run-name qwen_hamlet_probe
```

Defaults:

- reader model: `Qwen/Qwen3.5-9B`
- tokenizer model: `Qwen/Qwen3.5-9B`
- embedding model: `Qwen/Qwen3-Embedding-8B`
- reranker model: `Qwen/Qwen3-Reranker-8B`
- temperature: `0.0`
- treatments: `closed_book gold_evidence dense_reranked dense_document_order dense_random_order sparse_bm25 setr domain`
- context budgets: `1000`
- dense retrieval: Qwen embedding vectors in FAISS, then Qwen reranker scores
  define the final dense ranking
- top-k candidates: `50`
- GPU layout: `single`, which stages embedder, reranker, and reader on the
  default CUDA device so a one-GPU run remains intact
- dense prompt order variants: reranker rank, document order, deterministic
  random order
- sparse retrieval: BM25 over chunk text
- `setr`: LLM-backed SetR `selection_IRI` set selection over the dense
  candidate list, using the reader model as the selector
- `domain`: Hamlet domain-KG scaffold plus KG-guided dense candidate ordering
- random seed: `13`

On a server where only GPU 0 and GPU 1 are available, use:

```bash
python -m hamlet_qa.cli.run_experiment \
  --run-name qwen_hamlet_probe \
  --gpu-layout a40-2gpu
```

That preset uses `cuda:0` for embedding and reranking, and `cuda:1` for the
vLLM reader and SetR selector.

On a 3xA40 server with GPU 2 available, use:

```bash
python -m hamlet_qa.cli.run_experiment \
  --run-name qwen_hamlet_probe \
  --gpu-layout a40-3gpu
```

That preset uses `cuda:0` for the embedder, `cuda:1` for the reranker, and
`cuda:2` for the vLLM reader. GPU presets change only model placement, not
retrieval selection, reranking, prompt ordering, or scoring logic.

Outputs are written to `runs/<run_name>/results.jsonl`, with copies of the
config, chunks, input questions, and quote-resolved questions used for the run.
The context budget counts context tokens only. Prompt overhead and generated
tokens are logged separately.

To build prompts and traces without vLLM generation:

```bash
python -m hamlet_qa.cli.run_experiment --run-name dry_prompts --prepare-only
```

This still performs dense retrieval for grounded treatments so relevance
ordering and reranking are available when a dense treatment is selected, and
BM25 when `sparse_bm25` is selected. If `setr` is selected, prepare-only still
loads and calls the reader model for SetR context selection; it only skips the
final answer generation. To disable reranking, pass `--reranker-model none`.
For a model-, embedder-, reranker-, and BM25-free CLI smoke test, restrict
treatments to `closed_book`; tests can inject cached/stub retrievers.

## Post-Retrieval Context Assembly

### Baseline And Ordering

`dense_reranked` is the vanilla relevance-ranked dense baseline: retrieve dense
hits, optionally rerank them, and fill the context budget in relevance order.
`sparse_bm25` is the sparse relevance baseline. `closed_book` and
`gold_evidence` are controls.

`dense_document_order` and `dense_random_order` live under
`hamlet_qa/features/ordering/`. They use the same dense candidate set as
`dense_reranked`, then reorder selected chunks by document order or a
deterministic seeded shuffle.

### `setr`

`setr` is an LLM-backed SetR selector using the cloned `third_party/SetR/`
`selection_IRI` prompt and the paper
[Shifting from Ranking to Set Selection for Retrieval Augmented Generation](https://arxiv.org/abs/2507.06838).
The original SetR code centers on the `selection_IRI` prompt: identify the
query's information requirements, map passages to those requirements, then
select a set of passages that covers clear and diverse information.

What this implementation does:

- formats the top dense/reranked candidates as SetR numbered passages; by
  default this is the same `50`-candidate pool used by the other dense
  treatments;
- sends the original `selection_IRI` system/user prompt to the configured
  reader model, which acts as the SetR selector;
- requires the selector output to contain `### Final Selection: [..] [..]`;
- maps selected passage numbers back to chunk IDs and enforces the context
  budget without adding unselected fallback chunks;
- writes cached selector prompts, raw outputs, and parsed selections to
  `data/cache/setr_selector_cache.json` by default;
- logs the selector prompt, selector output, selected passage numbers, and
  selected chunks in `context_assembly_trace`.

This is a runtime adaptation of the original SetR selection prompt. It does not
train a SetR checkpoint. To approximate the paper more closely, use a strong
reader model or a separately fine-tuned SetR selector checkpoint as
`--reader-model`. Use `--setr-max-passages` to control how many retrieved
candidates the selector sees, and `--setr-selector-max-tokens` to control the
selector response budget. The default selector response budget is `4096`,
matching the long selector-response budget used by the cloned SetR scripts.

### `domain`

`domain` is a deterministic Hamlet domain-knowledge-guided assembler. It loads
`data/hamlet_domain_kg.yaml`, detects aliases in the question, expands related
graph nodes, creates a compact `domain_scaffold` pseudo-chunk within the context
budget, then orders dense-retrieved chunks by matches to the expanded graph
nodes before filling the remaining budget.

The editable KG currently includes:

- a short story summary;
- Hamlet, Claudius, Gertrude, Horatio, Ophelia, Polonius, Laertes, and the
  Ghost;
- aliases such as `King -> Claudius`, `Queen -> Gertrude`, and
  `Ghost -> Hamlet's father`;
- events including ghost revelation, antic disposition, Mousetrap, closet
  scene, Polonius death, Ophelia madness, and the duel;
- event aliases and relations such as
  `Mousetrap -> Claudius reaction -> Horatio confirmation`;
- optional evidence-role templates retained for graph editing experiments.

Use `--domain-kg path/to/file.yaml` to point at an edited graph, and
`--context-assembly-cache-dir path/to/cache` to move the SetR selector cache.

## Inspect Results

For the interactive browser view, render the JSONL results into a standalone
HTML file:

```bash
python -m hamlet_qa.inspection.results_html \
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
python -m hamlet_qa.inspection.results_html \
  runs/qwen_hamlet_probe/results.jsonl \
  --chunks runs/qwen_hamlet_probe/hamlet_chunks.jsonl \
  --output runs/qwen_hamlet_probe/results_viewer.html
```

The file pickers in the viewer can load another `results.jsonl` or JSON result
file and another chunk JSONL/JSON file.

For a compact Markdown summary:

```bash
python -m hamlet_qa.cli.inspect_results \
  --results runs/qwen_hamlet_probe/results.jsonl \
  --output runs/qwen_hamlet_probe/inspection.md
```

## Editable Data

- `data/hamlet_chunks.jsonl`: generated default chunks.
- `data/hamlet_questions.json`: starter questions and required evidence quotes.
- `data/hamlet_domain_kg.yaml`: editable Hamlet domain graph used by `domain`
  and optional role templates for `setr`.

Gold chunk IDs are derived from `required_evidence_quotes` during validation and
written to `hamlet_questions_resolved.json` inside each run directory. If
chunking settings change, regenerate chunks and confirm the quote validation
still resolves the intended evidence.

## Tests

```bash
python -m unittest discover -s tests
```
