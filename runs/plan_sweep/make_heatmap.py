"""Regenerate the LLM-judge ratings heatmap for plan_sweep + probe_v2 baselines.

Reads judge_rating from the metrics_annotations.jsonl sidecars and renders a
treatment x question grid. Labels use the intuitive decomposition-prompt names
(split_questions / list_requirements / reason_then_plan). Run from repo root:

    python runs/plan_sweep/make_heatmap.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_ann(path):
    return {(a["question_id"], a["treatment"]): a
            for a in (json.loads(l) for l in open(path) if l.strip())}


ann = load_ann(HERE / "metrics_annotations.jsonl")
ann.update(load_ann(ROOT / "runs" / "probe_v2" / "metrics_annotations.jsonl"))

# (treatment key in data, display label). Arm keys are unchanged; labels use the
# renamed decomposition prompts.
treatments = [
    ("gold_evidence",            "gold evidence (oracle)"),
    ("setr",                     "SetR"),
    ("dense_reranked",           "dense reranked"),
    ("closed_book",              "closed book"),
    ("plan_fixed_subq_par",      "split_questions | parallel"),
    ("plan_fixed_subq_seq",      "split_questions | sequential"),
    ("plan_fixed_inforeq_par",   "list_requirements | parallel"),
    ("plan_fixed_inforeq_seq",   "list_requirements | sequential"),
    ("plan_fixed_strategy_par",  "reason_then_plan | parallel"),
    ("plan_fixed_strategy_seq",  "reason_then_plan | sequential"),
    ("plan_dynamic_contract",    "dynamic | contract_v1"),
    ("plan_dynamic_strategy",    "dynamic | strategy_contract"),
]
questions = [
    ("q_trap_ghost_poison",              "ghost\npoison"),
    ("q_speaker_bitter_cold",            "bitter cold\n(speaker)"),
    ("q_arc_rosencrantz_guildenstern",   "R&G arc\n(3-part)"),
    ("q_bridge_mousetrap_test",          "mousetrap\n(bridge)"),
    ("q_fortinbras_campaign",            "fortinbras\n(temporal)"),
    ("q_state_poisoned_cup",             "poisoned\ncup"),
    ("q_final_scene_deaths",             "final deaths\n(4-part)"),
    ("q_mistaken_arras_victim",          "arras\nvictim"),
    ("q_distractor_claudius_not_punish", "claudius\nreasons"),
    ("q_unanswerable_yorick_wife",       "yorick wife\n(unanswerable)"),
]

M = np.array([[ann[(q, t)]["judge_rating"] for q, _ in questions] for t, _ in treatments], dtype=float)
means = M.mean(axis=1)

fig, ax = plt.subplots(figsize=(13.8, 7.2))
im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=5, aspect="auto")
ax.set_xticks(range(len(questions)), [q for _, q in questions], fontsize=8.5)
ax.set_yticks(range(len(treatments)),
              [f"{lbl}   (mean {mu:.2f})" for (_, lbl), mu in zip(treatments, means)], fontsize=9)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=8.5, color="black")
ax.set_title("LLM-judge answer ratings (0=wrong -> 5=fully correct & grounded)\n"
             "plan_sweep arms vs baselines · Qwen3.5-9B reader · budget 1000 tok · judge: claude-opus-4-8",
             fontsize=11)
for y in (3.5, 9.5):  # separate baselines / fixed arms / dynamic arms
    ax.axhline(y, color="black", linewidth=1.4)
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("judge rating", fontsize=9)
fig.tight_layout()
out = HERE / "judge_heatmap.png"
fig.savefig(out, dpi=180)
print("saved", out)
