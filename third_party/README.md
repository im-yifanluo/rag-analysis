# third_party/ — upstream sources for the ported methods

This directory holds the official repositories (and papers) for the SOTA methods
implemented in `hamlet_qa/features/` and `hamlet_qa/metrics/`. The harness ports
and cites these as the source of truth for faithfulness, but **they are not
committed to this repo** — they are large, contain unused benchmark data, and
are under their own licenses. Everything here except this README is gitignored.

To reproduce the layout the code expects, clone each upstream at the pinned
commit it was reviewed against:

```bash
cd third_party

git clone https://github.com/HuskyInSalt/CRAG            CorrectiveRAG/CRAG
git -C CorrectiveRAG/CRAG checkout de7c2961ae624a1483a138c5798e1f6d0c4fb0e0

git clone https://github.com/SJTU-DMTai/RAG-CSM          InfluenceGuided_RAG/RAG-CSM
git -C InfluenceGuided_RAG/RAG-CSM checkout f2a4a46c9b87add921a2a2ea238a9e4d4e5753f3

git clone https://github.com/Leezekun/MacRAG             MacRAG/MacRAG
git -C MacRAG/MacRAG checkout b1b2812206c8bf02476ba4728243dcb46b1dd0e3

git clone https://github.com/carriex/recomp             RECOMP/recomp
git -C RECOMP/recomp checkout 51d4432151efb3275257a9407dc71d1e5ec6634d

git clone https://github.com/LGAI-Research/SetR          SetR/SetR
git -C SetR/SetR checkout d1ac78b7e41f7c6a2cd8d582934fbba02893f343

git clone https://github.com/hljoren/sufficientcontext   Sufficient_Context/sufficientcontext
git -C Sufficient_Context/sufficientcontext checkout 801b465eb3be5bf5216fbe3301e61daee85ff99c
```

| Directory | Method | Paper | Upstream |
|---|---|---|---|
| `CorrectiveRAG/CRAG` | CRAG | [2401.15884](https://arxiv.org/abs/2401.15884) | https://github.com/HuskyInSalt/CRAG |
| `MacRAG/MacRAG` | MacRAG | [2505.06569](https://arxiv.org/abs/2505.06569) | https://github.com/Leezekun/MacRAG |
| `RECOMP/recomp` | RECOMP | [2310.04408](https://arxiv.org/abs/2310.04408) | https://github.com/carriex/recomp |
| `SetR/SetR` | SetR | [2507.06838](https://arxiv.org/abs/2507.06838) | https://github.com/LGAI-Research/SetR |
| `InfluenceGuided_RAG/RAG-CSM` | Oracle CI value | [2509.21359](https://arxiv.org/abs/2509.21359) | https://github.com/SJTU-DMTai/RAG-CSM |
| `Sufficient_Context/sufficientcontext` | Sufficient-context autorater | [2411.06037](https://arxiv.org/abs/2411.06037) | https://github.com/hljoren/sufficientcontext |

The paper PDFs were downloaded from arXiv into each method's top-level folder
(e.g. `third_party/MacRAG/2505.06569v2.pdf`). They are not required to run the
harness — only to cross-check the prompts/metrics against the published text.

Each upstream keeps its own license; consult the LICENSE/NOTICE in the cloned
repos before redistributing any of this code.
