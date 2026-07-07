# plan_sweep — LLM-Judge Ratings & Trace Analysis

**Run:** `runs/plan_sweep` (8 evidence-planning arms × 10 questions, one model load, budget 1000 tok, greedy Qwen3.5-9B reader).
**Baselines:** `gold_evidence`, `setr`, `dense_reranked`, `closed_book` from `runs/probe_v2` (same questions, same reader, same budget).
**Judge:** claude-opus-4-8, manual per-answer grading against `expected_answer` + required evidence roles.
**Rubric (0–5):** 0 = wrong/hallucinated · 1 = abstained on an answerable question (honest, but a system failure) · 2–3 = partially correct · 4 = correct with minor gaps · 5 = fully correct and grounded. For the unanswerable question, a clean abstention = 5.
**Artifacts:** ratings + rationales merged into each run's `metrics_annotations.jsonl` (`judge_rating`, `judge_rationale`); heatmap at `judge_heatmap.png`; per-row detail in `results_viewer.html`.

---

## 1. Headline ranking (mean judge rating, 10 questions)

| rank | treatment | mean | vs dense_reranked |
|---|---|---|---|
| 1= | gold_evidence (oracle) | **4.85** | +0.35 |
| 1= | setr | **4.85** | +0.35 |
| 3 | dense_reranked | **4.50** | — |
| 4 | plan_fixed_strategy_seq | 4.20 | −0.30 |
| 5 | plan_dynamic_contract | 4.00 | −0.50 |
| 6 | plan_fixed_inforeq_seq | 3.90 | −0.60 |
| 7 | plan_fixed_strategy_par | 3.85 | −0.65 |
| 8 | plan_fixed_inforeq_par | 3.70 | −0.80 |
| 9 | plan_dynamic_strategy | 3.65 | −0.85 |
| 10 | plan_fixed_subq_par | 3.55 | −0.95 |
| 11 | plan_fixed_subq_seq | 3.25 | −1.25 |
| 12 | closed_book | 2.85 | −1.65 |

**Every planning arm currently loses to plain dense retrieval.** But the per-stage metrics show this is *not* because decomposition retrieves worse — it's because the selection stage throws the retrieved evidence away (§3). One catastrophic shared failure (§4.1) also costs every arm ~0.4 mean points.

---

## 2. Where the ratings come from (question-level)

See `judge_heatmap.png`. Key cells:

- **`q_speaker_bitter_cold` is a red stripe: every plan arm scored 1** (abstained). All baselines with retrieval scored 5. This single column accounts for most of the gap to dense_reranked.
- Plan arms **match or beat baselines** on `q_trap_ghost_poison` (the retrieval-trap question — all arms 4.5–5, closed_book 1), `q_state_poisoned_cup` for the inforeq/strategy arms (5 vs closed_book 0), and `q_distractor_claudius_not_punish` for strategy arms (5).
- **closed_book is dead last despite knowing the play**: it aces broad-arc questions from memory (R&G arc 5, mousetrap 5, final deaths 5) but hallucinates specifics (serpent "poured" into the ear; *Claudius* drinks the cup; attributes Francisco's line to Hamlet in a fabricated Act 3 scene). Retrieval's real value on a memorized corpus is *suppressing confident hallucination*, not adding knowledge.
- **gold_evidence is not 5.0**: on `q_mistaken_arras_victim` the reader answered "he thinks it's a *rat*" even though "Is it the King?" was in its context (3.5). Reader-side ceiling, not retrieval.

---

## 3. The funnel: retrieval finds it, selection drops it

Stage-wise means over the 8 arms (from `plan_eval` + inline metrics):

| stage | metric | value |
|---|---|---|
| 1. plan | JSON parse fallbacks | **0 / 80** — decomposition output is fully reliable |
| 1. plan | nodes per question | subq 2.4 · inforeq 2.6 · strategy 2.8 · dyn_contract 2.1 · dyn_strategy 1.6 |
| 2. retrieve | `plan_slot_retrieval_recall` (gold role surfaced by ≥1 node) | **0.78–0.82** |
| 2. retrieve | `plan_gold_chunk_retrieval` (all gold chunks) | 0.74–0.80 |
| 3. select | final `evidence_chunk_recall` | **0.31–0.46** |
| 3. select | chunks in final context | 1.3–2.0 (dense_reranked: 4.1) |
| 3. select | context tokens of 1000 budget | 320–482 (**48% mean utilization even excluding empty rows**; dense_reranked: 954) |

**≈80% of the gold evidence is sitting in the per-node candidate pools, and the greedy selection keeps less than half of it while leaving half the token budget unspent.** Two mechanisms cause this:

1. **`min_support = 0.5` on sigmoid(reranker logit) is a hard floor.** Reranker logits for correct-but-not-perfect matches are often slightly negative (sigmoid < 0.5), so whole candidate pools get filtered. In the worst case *everything* is filtered → empty context (§4.1).
2. **Noisy-OR coverage saturates after one chunk per node.** Once each node has a single covering unit, marginal coverage gain ≈ 0 and greedy stops — there is no "fill remaining budget with next-best evidence" phase. Baselines win simply by *stuffing the budget*.

The experiment's stage-2 hypothesis is actually **confirmed** — decomposed evidence-slot retrieval surfaces the right chunks (0.82 slot recall, including questions where whole-question dense retrieval is distracted). The loss happens entirely in stage 3.

---

## 4. Failure taxonomy (from the traces)

### 4.1 Verbatim-quote lookup breaks per-node dense retrieval (catastrophic, systematic)
`q_speaker_bitter_cold`: every decomposition rewrote the question into a quoted-phrase node query, e.g. `"For this relief much thanks" "'Tis bitter cold" "sick at heart"`. The Qwen embedder (query-prompted, semantic) does not do verbatim string matching: top hits were Act 3.2 / 4.5 / 1.2 — the true chunk (`act01_scene01_chunk001`) never entered the candidate pool. Then `min_support=0.5` filtered even those wrong candidates (best sigmoid ≈ 0.35) → `num_selectable: 0` → **empty context** → "The provided context does not answer the question." 7 of 8 arms emitted literally empty contexts; the 8th (dyn_strategy) had wrong-scene chunks and also abstained.
The full-question dense baseline retrieved the same chunk at rank 1 — the *question wording* ("Who says…") matches dialogue openings; the *bare quote* does not. **Decomposition destroyed the lexical signal.** A BM25/exact-string fallback per node (the corpus already has a BM25 index for CRAG) would fix this class outright.

### 4.2 Event-boundary truncation: setup selected, outcome dropped
`q_state_poisoned_cup`: subq_par/seq and dyn_strategy selected only `act05_scene02_chunk021` ("It is the poison'd cup; it is too late") and **not** the next chunk 022 ("The drink, the drink! I am poison'd… [_Dies._]"). The reader answered, precisely and honestly: *"the provided text ends before she takes the sip."* Same pattern on `q_final_scene_deaths` (inforeq_par kept only the mid-fight chunk → rating 1). Narrative *outcomes live in the chunk after the setup*; single-chunk units with no neighbor expansion systematically cut event arcs mid-action. (`reader_support`'s neighbor-hop units were built for exactly this; the plan executor has no equivalent.)

### 4.3 Coverage-saturation starvation on multi-part questions
`q_fortinbras_campaign` (gold spans 3 scenes / 4 chunks): four arms selected **only** `act04_scene04_chunk001` and correctly reported the other two stages as unanswered; no arm selected more than 2 of 4 gold chunks — while ~800 tokens of budget sat unused. Slot retrieval had surfaced the right chunks (slotR .824). Same mechanism on the R&G arc (each arm missing exactly one of the three stages, a different one per arm).

### 4.4 Chunk-granularity mis-splits corrupt attribution
`q_distractor_claudius_not_punish`: subq arms selected only `chunk002` (reason 2 + the *tail* of reason 1). The reader, seeing "she's so conjunctive to my life and soul" without its subject, attributed it to **Hamlet** ("Hamlet is so conjunctive to Claudius's life and soul") — a wrong answer manufactured by a missing antecedent chunk. Strategy arms selected both chunks → 5.0.

### 4.5 What did *not* fail
- **JSON contracts**: 0 parse fallbacks in 80 rows; the schema'd prompts are robust with this reader.
- **Unanswerable handling**: all 8 arms gave clean abstentions (5.0) — and the two dynamic arms produced *empty contexts* on it, which is arguably the ideal behavior (spend zero tokens on an unanswerable question).
- **The "trap" question**: all arms retrieved 1.5 (Ghost's account) rather than being lured to 3.2 (Mousetrap re-enactment) — decomposed needs ("what poison", "effect on blood") are *good* semantic queries when the answer is content, not verbatim text.

---

## 5. The experimental axes

**Decomposition prompt (the A/B):** `strategy` > `info_requirements` > `subquestions` (means 4.03 / 3.80 / 3.40 across modes). `subquestions` is worst for a mechanical reason: its sub-questions quote/paraphrase the original wording, poor as search queries (§4.1, §4.4). `strategy` emits the most nodes (2.8) and always encodes `depends_on` (10/10 questions), producing the best multi-part coverage (only prompt family to select both Claudius-reasons chunks).

**Parallel vs sequential:** sequential ≥ parallel for strategy (+0.35) and inforeq (+0.20), < for subq (−0.30). 39 reformulations fired across 31 sequential rows. Reformulation helped when the dependent node's query gained an entity (strategy_seq picked up `2.2 chunk005` on Fortinbras after hop-1 evidence); it hurt subq by drifting further from the original wording. Directionally: **hop-wise reformulation helps iff the decomposition style produces entity-seeking (not quote-seeking) nodes.**

**Fixed vs dynamic:** dyn_contract 4.00 lands mid-pack — the planner's *procedure choices* were sensible (greedy_coverage for multi-part, top_per_node for single-fact, sequential exactly once: `bridge_multihop` on the R&G arc; never the teacher scorer). Its weakness is `strategy_contract` (3.65) classifying too many questions as `single` (1.6 nodes avg) — under-decomposition → fewer slots → thinner evidence. The contract mechanism works; the planning *judgment* is prompt-sensitive.

---

## 6. Insights

1. **The bottleneck is selection, not planning.** Stage-2 slot retrieval (0.82) beats the final context (0.31–0.46) by ~2×. Fixing stage 3 is worth far more than any prompt engineering on stage 1.
2. **Budgeted-coverage selection needs a budget-fill phase.** "Cover every node then stop" leaves 52% of tokens unused and loses to naive budget-stuffing. After coverage is reached, keep appending next-best-support units until the budget is actually spent.
3. **`min_support=0.5` is miscalibrated for sigmoid(reranker logits)** and creates empty contexts. Either lower it (~0.25), temperature-scale the logits, or guarantee top-1-per-node survives the filter.
4. **Semantic decomposition needs a lexical safety net.** Quote-attribution/verbatim needs should route to BM25 (already in the codebase) or string search; a trivial heuristic — node query contains a long quoted span → sparse retrieval — covers it.
5. **Units need neighbor expansion.** Setup→outcome arcs span adjacent chunks; ±1-chunk expansion (as in MacRAG/reader_support) would have fixed §4.2 and likely §4.4.
6. **Abstention honesty is a real, measurable win** of the pipeline: zero hallucinated answers across all 80 plan rows (closed_book: 3 confident fabrications in 10). The failure mode is silence, not lies — much easier to fix.
7. **On a memorized corpus, judge ratings alone flatter closed_book** on arc questions; groundedness (citations resolving to supplied chunks) should enter the stage-3 judge rubric when you formalize it with the stronger model.

## 7. Next steps (priority order)

1. **Selection fixes (biggest lever, config-only + ~20 lines):** lower `plan_min_support` → 0.25; add budget-fill after coverage saturation; guarantee each node's top-1 candidate survives filtering. Re-run the sweep — expect plan arms to close most of the 0.3–1.25 gap since the evidence is already in the pools.
2. **Per-node sparse fallback:** route quote-like node queries (regex: quoted span ≥ 4 words) to the existing BM25 index; union candidates before support scoring. Kills the §4.1 failure class.
3. **Neighbor expansion on selected units** (±1 chunk within scene, budget permitting) — fixes §4.2/§4.4.
4. **Re-run and diff:** the sweep is one command now; `plan_eval` + `judge_rating` sidecars give before/after per stage.
5. **Formal LLM-judge (planned):** port this rubric into a stage-3 judge with the stronger model; add a groundedness check (do cited chunk IDs exist in the row's context and contain the quoted text?). The manual ratings in the sidecars are the calibration set.
6. **Optional ablation for the paper:** plan arms with selection fixes vs dense_reranked at budgets 500/750/1000 — the hypothesis worth testing is that planning wins *precisely when the budget is tight* (dense wins now only because it can afford to stuff 954 tokens).
