from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hamlet_qa.core.context import ContextAssemblyRequest
from hamlet_qa.core.llm_cache import JsonKVCache
from hamlet_qa.features.evidence_plan.assembly import (
    assemble_plan_dynamic,
    assemble_plan_fixed,
)
from hamlet_qa.features.evidence_plan.contract import nodes_from_items, parse_contract
from hamlet_qa.features.evidence_plan.executor import execute_plan
from hamlet_qa.features.evidence_plan.planning import decompose, reformulate_query
from hamlet_qa.features.evidence_plan.prompts import (
    get_decomposition_prompt,
    get_followup_prompt,
    get_planner_prompt,
)
from hamlet_qa.features.evidence_plan.retrieve import sigmoid
from hamlet_qa.metrics.plan_eval import compute_plan_eval_for_row
from hamlet_qa.core.evidence.schema import EvidenceNode

MARKERS = ["SUMMON", "ENGLAND", "FATE"]


def make_chunk(chunk_id, global_index, text, token_count=20, scene_id="act01_scene01"):
    return {
        "chunk_id": chunk_id, "global_index": global_index, "act": 1, "scene": 1,
        "scene_id": scene_id, "scene_title": "Scene.", "chunk_in_scene": global_index + 1,
        "start_token": 0, "end_token": token_count, "token_count": token_count, "text": text,
    }


def marker_chunks():
    return {
        "c1": make_chunk("c1", 0, "KING summons them: SUMMON reason is transformation."),
        "c2": make_chunk("c2", 1, "KING sends them: ENGLAND task with a commission."),
        "c3": make_chunk("c3", 2, "AMBASSADOR reports: FATE is that they are dead."),
        "c4": make_chunk("c4", 3, "Unrelated noise about a platform at night."),
    }


class StubNodeRetriever:
    """retrieve(query, top_k): marker-overlap -> high reranker logit, else low."""

    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, top_k):
        rows = []
        for chunk in self.chunks.values():
            hit = any(m in query and m in chunk["text"] for m in MARKERS)
            rows.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "rerank_score": 5.0 if hit else -5.0,
                    "score": 5.0 if hit else -5.0,
                    "global_index": chunk["global_index"],
                    "act": 1, "scene": 1, "scene_title": "Scene.",
                }
            )
        rows.sort(key=lambda r: -r["rerank_score"])
        return rows[:top_k]


class ScriptedReader:
    model_name = "scripted"

    def __init__(self):
        self.calls = []

    def generate(self, system_prompt, user_prompt, max_tokens=None):
        self.calls.append(user_prompt)
        if "THIS STEP'S NEED:" in user_prompt:  # follow-up
            return '{"retrieval_query": "FATE bridged query"}'
        nodes = (
            '"nodes": ['
            '{"node_id": "n1", "need": "summon reason", "node_query": "SUMMON reason", "order_index": 1, "depends_on": []},'
            '{"node_id": "n2", "need": "england task", "node_query": "ENGLAND task", "order_index": 2, "depends_on": []},'
            '{"node_id": "n3", "need": "their fate", "node_query": "FATE", "order_index": 3, "depends_on": []}'
            "]"
        )
        if '"question_type"' in user_prompt:  # planner contract
            return (
                '{"question_type": "independent_multipart", "retrieval_policy": "dense", '
                '"retrieval_mode": "parallel", "selection_policy": "greedy_coverage", '
                '"ordering_policy": "document_order", "support_policy": "reranker", ' + nodes + "}"
            )
        if '"nodes"' in user_prompt:  # decomposition
            return "{" + nodes + "}"
        return "{}"


def node(node_id, query, order=1, depends=None):
    return EvidenceNode(node_id=node_id, need=query, node_query=query, order_index=order, depends_on=depends or [])


class PromptRegistryTests(unittest.TestCase):
    def test_variants_resolve_and_format(self):
        v = get_decomposition_prompt("reason_then_plan")
        self.assertIn("reason_then_plan", v.version)
        rendered = v.template.format(question="Q", catalog="C", max_nodes=5)
        self.assertIn("Q", rendered)
        self.assertIn('"nodes"', rendered)
        self.assertIn("strategy", rendered)
        self.assertTrue(get_planner_prompt("contract_v1").version)
        self.assertTrue(get_followup_prompt("rewrite_with_evidence").version)

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            get_decomposition_prompt("does_not_exist")


class ContractTests(unittest.TestCase):
    DEFAULTS = {
        "retrieval_mode": "parallel", "support_policy": "reranker",
        "selection_policy": "greedy_coverage", "ordering_policy": "document_order",
    }

    def test_valid_contract_parses(self):
        raw = (
            '{"question_type":"bridge_multihop","retrieval_policy":"dense",'
            '"retrieval_mode":"sequential","selection_policy":"top_per_node",'
            '"ordering_policy":"node_order","support_policy":"teacher",'
            '"nodes":[{"node_id":"n1","need":"x","node_query":"qx","order_index":1,"depends_on":[]}]}'
        )
        c = parse_contract(raw, defaults=self.DEFAULTS, max_nodes=5, question_text="Q")
        self.assertEqual(c.retrieval_mode, "sequential")
        self.assertEqual(c.support_policy, "teacher")
        self.assertEqual(len(c.nodes), 1)
        self.assertIsNone(c.parse_error)
        self.assertEqual(c.deviations, [])

    def test_invalid_fields_fall_back_with_deviations(self):
        raw = '{"retrieval_mode":"telepathy","nodes":[{"need":"x","node_query":"q"}]}'
        c = parse_contract(raw, defaults=self.DEFAULTS, max_nodes=5, question_text="Q")
        self.assertEqual(c.retrieval_mode, "parallel")  # default
        self.assertTrue(any("retrieval_mode" in d for d in c.deviations))

    def test_no_nodes_falls_back_to_single_node(self):
        c = parse_contract("garbage", defaults=self.DEFAULTS, max_nodes=5, question_text="Who dies?")
        self.assertEqual(len(c.nodes), 1)
        self.assertIn("Who dies?", c.nodes[0].node_query)
        self.assertIsNotNone(c.parse_error)

    def test_nodes_from_items_skips_bad(self):
        nodes = nodes_from_items([{"need": "ok", "node_query": "q"}, {"no_need": 1}, "bad"], 5)
        self.assertEqual(len(nodes), 1)


class PlanningCallTests(unittest.TestCase):
    def test_decompose_parses_and_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = JsonKVCache(Path(tmp) / "c.json", section="d")
            reader = ScriptedReader()
            out = decompose(
                "Q", "catalog", reader, get_decomposition_prompt("split_questions"),
                cache, max_nodes=5, max_tokens=256,
            )
            self.assertEqual([n.node_id for n in out["nodes"]], ["n1", "n2", "n3"])
            self.assertFalse(out["fallback"])
            # second call hits cache (no new model call)
            reader2 = ScriptedReader()
            out2 = decompose(
                "Q", "catalog", reader2, get_decomposition_prompt("split_questions"),
                cache, max_nodes=5, max_tokens=256,
            )
            self.assertTrue(out2["cache_hit"])
            self.assertEqual(reader2.calls, [])

    def test_decompose_falls_back_on_bad_json(self):
        class Bad:
            model_name = "bad"

            def generate(self, s, u, max_tokens=None):
                return "no json"

        with tempfile.TemporaryDirectory() as tmp:
            cache = JsonKVCache(Path(tmp) / "c.json", section="d")
            out = decompose("Who?", "cat", Bad(), get_decomposition_prompt("split_questions"), cache, max_nodes=5, max_tokens=64)
            self.assertTrue(out["fallback"])
            self.assertEqual(len(out["nodes"]), 1)

    def test_reformulate_parses_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = JsonKVCache(Path(tmp) / "c.json", section="f")
            out = reformulate_query(
                "Q", node("n2", "fate", depends=["n1"]), "evidence so far",
                ScriptedReader(), get_followup_prompt("rewrite_with_evidence"), cache, max_tokens=64,
            )
            self.assertIn("FATE", out["query"])


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.chunks = marker_chunks()
        self.retriever = StubNodeRetriever(self.chunks)
        self.nodes = [node("n1", "SUMMON reason", 1), node("n2", "ENGLAND task", 2), node("n3", "FATE", 3)]

    def _run(self, **overrides):
        kwargs = dict(
            node_retriever=self.retriever, chunk_lookup=self.chunks, context_budget=1000,
            retrieval_mode="parallel", support_policy="reranker", selection_policy="greedy_coverage",
            ordering_policy="document_order", node_top_k=4, support_temp=1.0,
            coverage_threshold=0.85, redundancy_beta=0.15, token_exponent_tau=0.7,
            min_support=0.5, max_selected_units=8,
        )
        kwargs.update(overrides)
        return execute_plan("Q", self.nodes, **kwargs)

    def test_parallel_reranker_covers_all_nodes(self):
        out = self._run()
        self.assertFalse(out["empty"])
        self.assertEqual(set(out["selected_chunk_ids"]), {"c1", "c2", "c3"})  # noise c4 excluded
        cov = out["trace"]["selection"]["final_coverage"]
        self.assertTrue(all(v >= 0.85 for v in cov.values()))

    def test_document_order(self):
        out = self._run(ordering_policy="document_order")
        ids = out["selected_chunk_ids"]
        self.assertEqual(ids, sorted(ids, key=lambda c: {"c1": 0, "c2": 1, "c3": 2, "c4": 3}[c]))

    def test_sequential_calls_reformulate_for_dependent_node(self):
        self.nodes = [node("n1", "SUMMON reason", 1), node("n3", "FATE", 2, depends=["n1"])]
        calls = []

        def reformulate(n, evidence):
            calls.append((n.node_id, evidence))
            return {"query": "FATE bridged", "cache_hit": False}

        out = self._run(retrieval_mode="sequential", reformulate=reformulate)
        self.assertEqual([c[0] for c in calls], ["n3"])  # only the dependent node
        self.assertFalse(out["empty"])

    def test_top_per_node_policy(self):
        out = self._run(selection_policy="top_per_node", max_selected_units=3)
        self.assertEqual(out["trace"]["selection_policy"], "top_per_node")
        self.assertTrue(set(out["selected_chunk_ids"]) <= {"c1", "c2", "c3"})

    def test_empty_when_nothing_clears_min_support(self):
        # min_support above sigmoid(5)=0.993 -> nothing selectable
        out = self._run(min_support=0.999)
        self.assertTrue(out["empty"])
        self.assertEqual(out["selected_chunk_ids"], [])

    def test_teacher_policy_uses_scorer(self):
        class StubTeacher:
            def __init__(self):
                self.calls = 0

            def score(self, q, n, unit):
                from hamlet_qa.core.evidence.schema import SupportScore
                self.calls += 1
                hit = any(m in n.node_query and m in unit.text for m in MARKERS)
                return SupportScore(node_id=n.node_id, unit_id=unit.unit_id,
                                    support_score=1.0 if hit else 0.0, support_type="complete")

        teacher = StubTeacher()
        out = self._run(support_policy="teacher", teacher_scorer=teacher)
        self.assertGreater(teacher.calls, 0)
        self.assertEqual(set(out["selected_chunk_ids"]), {"c1", "c2", "c3"})


class SigmoidTests(unittest.TestCase):
    def test_sigmoid_band(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        self.assertGreater(sigmoid(2.5), 0.9)
        self.assertLess(sigmoid(-5.0), 0.01)


class AssemblyIntegrationTests(unittest.TestCase):
    def _request(self, tmp, treatment):
        chunks = marker_chunks()
        trace = [{"chunk_id": cid, "rank": i + 1} for i, cid in enumerate(chunks)]

        class Q:
            id = "q_test"
            question = "Why summon, what England task, what fate?"

        return ContextAssemblyRequest(
            question=Q(), treatment=treatment, context_budget=1000,
            chunk_lookup=chunks, doc_order_ids=list(chunks), retrieval_trace=trace,
            selector_model=ScriptedReader(),
            feature_handles={"node_retriever": StubNodeRetriever(chunks)},
            feature_params={"plan_cache_path": str(Path(tmp) / "plan_cache.json")},
        )

    def test_plan_fixed_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = assemble_plan_fixed(self._request(tmp, "plan_fixed"))
            self.assertEqual(result.retrieval_method, "plan_fixed")
            self.assertEqual(set(result.selected_chunk_ids), {"c1", "c2", "c3"})
            trace = result.context_assembly_trace
            self.assertEqual(len(trace["decomposition"]["nodes"]), 3)
            self.assertEqual(trace["policies"]["retrieval_mode"], "parallel")
            self.assertTrue(Path(tmp, "plan_cache.json").exists())

    def test_plan_dynamic_uses_contract_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = assemble_plan_dynamic(self._request(tmp, "plan_dynamic"))
            self.assertEqual(result.retrieval_method, "plan_dynamic")
            contract = result.context_assembly_trace["contract"]
            self.assertEqual(contract["retrieval_mode"], "parallel")
            self.assertEqual(contract["selection_policy"], "greedy_coverage")
            self.assertEqual(set(result.selected_chunk_ids), {"c1", "c2", "c3"})

    def test_missing_node_retriever_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = self._request(tmp, "plan_fixed")
            broken = ContextAssemblyRequest(
                question=request.question, treatment="plan_fixed", context_budget=1000,
                chunk_lookup=request.chunk_lookup, doc_order_ids=request.doc_order_ids,
                retrieval_trace=request.retrieval_trace, selector_model=request.selector_model,
                feature_params=request.feature_params,
            )
            with self.assertRaises(ValueError):
                assemble_plan_fixed(broken)


class PlanEvalMetricTests(unittest.TestCase):
    def _plan_row(self):
        return {
            "treatment": "plan_fixed",
            "derived_gold_chunk_ids": ["g1", "g2", "g3"],
            "required_evidence_quotes": [
                {"role": "r1", "matched_chunk_ids": ["g1"]},
                {"role": "r2", "matched_chunk_ids": ["g2"]},
                {"role": "r3", "matched_chunk_ids": ["g3"]},
            ],
            "context_assembly_trace": {
                "method": "plan_fixed",
                "decomposition": {"nodes": [{"node_id": "n1"}, {"node_id": "n2"}, {"node_id": "n3"}], "fallback": False},
                "execution": {
                    "per_node_retrieval": [
                        {"node_id": "n1", "retrieved": [{"chunk_id": "g1"}, {"chunk_id": "x"}]},
                        {"node_id": "n2", "retrieved": [{"chunk_id": "g2"}]},
                        {"node_id": "n3", "retrieved": [{"chunk_id": "y"}]},  # misses g3
                    ]
                },
            },
        }

    def test_stage1_and_stage2_measured(self):
        out = compute_plan_eval_for_row(self._plan_row())
        self.assertTrue(out["plan_eval_applicable"])
        # Stage 1: slots generated
        self.assertEqual(out["plan_num_nodes"], 3)
        self.assertEqual(out["plan_num_gold_roles"], 3)
        self.assertFalse(out["plan_node_fallback"])
        # Stage 2: retrieval found g1, g2 but not g3 -> 2/3 roles
        self.assertAlmostEqual(out["plan_slot_retrieval_recall"], 2 / 3)
        self.assertAlmostEqual(out["plan_gold_chunk_retrieval"], 2 / 3)
        by_role = {d["role"]: d for d in out["plan_slot_detail"]}
        self.assertEqual(by_role["r1"]["retrieved_by_nodes"], ["n1"])
        self.assertFalse(by_role["r3"]["retrieved"])

    def test_non_plan_row_is_not_applicable(self):
        out = compute_plan_eval_for_row({"treatment": "dense_reranked", "context_assembly_trace": {"method": "dense"}})
        self.assertFalse(out["plan_eval_applicable"])
        self.assertIsNone(out["plan_slot_retrieval_recall"])

    def test_plan_dynamic_reads_contract_nodes(self):
        row = self._plan_row()
        row["context_assembly_trace"] = {
            "method": "plan_dynamic",
            "contract": {"nodes": [{"node_id": "n1"}, {"node_id": "n2"}], "parse_error": None},
            "execution": {"per_node_retrieval": [{"node_id": "n1", "retrieved": [{"chunk_id": "g1"}]}]},
        }
        out = compute_plan_eval_for_row(row)
        self.assertTrue(out["plan_eval_applicable"])
        self.assertEqual(out["plan_num_nodes"], 2)
        self.assertAlmostEqual(out["plan_slot_retrieval_recall"], 1 / 3)  # only g1 retrieved


if __name__ == "__main__":
    unittest.main()
