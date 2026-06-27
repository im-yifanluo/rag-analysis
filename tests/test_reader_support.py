from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hamlet_qa.core.context import ContextAssemblyRequest
from hamlet_qa.features.reader_support.assembly import (
    assemble_reader_support,
    greedy_select,
    lexical_prior,
    _order_units,
)
from hamlet_qa.features.reader_support.nodes import (
    NODE_INDUCTION_SYSTEM,
    induce_nodes,
    parse_nodes,
)
from hamlet_qa.features.reader_support.schema import EvidenceNode, EvidenceUnit
from hamlet_qa.features.reader_support.teacher import (
    SUPPORT_TEACHER_SYSTEM,
    parse_support_output,
    validate_and_cap,
)
from hamlet_qa.features.reader_support.units import build_units
from hamlet_qa.core.llm_cache import JsonKVCache
from hamlet_qa.metrics.evidence_role import compute_evidence_role_recall_for_row


def make_chunk(chunk_id, global_index, scene_id, text, token_count=20, act=1, scene=1):
    return {
        "chunk_id": chunk_id,
        "global_index": global_index,
        "act": act,
        "scene": scene,
        "scene_id": scene_id,
        "scene_title": "Test scene.",
        "chunk_in_scene": global_index + 1,
        "start_token": 0,
        "end_token": token_count,
        "token_count": token_count,
        "text": text,
    }


def make_node(node_id, query, order_index=1):
    return EvidenceNode(node_id=node_id, need=f"Find {query}", node_query=query, order_index=order_index)


def make_unit(unit_id, text, primary="c1", token_count=10, order=(0, 0)):
    return EvidenceUnit(
        unit_id=unit_id,
        unit_type="chunk",
        text=text,
        source_chunk_ids=[primary],
        primary_chunk_id=primary,
        token_count=token_count,
        source_order_key=list(order),
        global_index_start=order[0],
        global_index_end=order[0],
    )


class ScriptedSupportModel:
    """Returns node-induction JSON or marker-driven support JSON by system prompt."""

    model_name = "scripted-reader"
    MARKERS = ["SUMMONMARK", "ENGLANDMARK", "FATEMARK"]

    def __init__(self):
        self.calls: list[str] = []

    def generate(self, system_prompt, user_prompt, max_tokens=None):
        self.calls.append(system_prompt)
        if system_prompt == NODE_INDUCTION_SYSTEM:
            return (
                '{"nodes": ['
                '{"node_id": "n1", "need": "Find the SUMMONMARK reason", "node_query": "SUMMONMARK", "order_index": 1, "depends_on": []},'
                '{"node_id": "n2", "need": "Find the ENGLANDMARK task", "node_query": "ENGLANDMARK", "order_index": 2, "depends_on": []},'
                '{"node_id": "n3", "need": "Find the FATEMARK fate", "node_query": "FATEMARK", "order_index": 3, "depends_on": []}'
                "]}"
            )
        if system_prompt == SUPPORT_TEACHER_SYSTEM:
            need_part, _, cand_part = user_prompt.partition("CANDIDATE TEXT:")
            for marker in self.MARKERS:
                if marker in need_part and marker in cand_part:
                    return (
                        '{"support_score": 1.0, "support_type": "complete", '
                        f'"supporting_span": "{marker}", "needs_more_context": false, '
                        '"explanation": "match"}'
                    )
            return (
                '{"support_score": 0.0, "support_type": "none", '
                '"supporting_span": "", "needs_more_context": false, "explanation": "no"}'
            )
        return "{}"


class NodeInductionTests(unittest.TestCase):
    def test_parse_valid_nodes(self):
        nodes = parse_nodes(
            'prose {"nodes": [{"node_id": "n1", "need": "Find the cause", '
            '"node_query": "cause", "order_index": 1, "depends_on": []}]} tail',
            max_nodes=5,
        )
        self.assertIsNotNone(nodes)
        self.assertEqual(nodes[0].node_id, "n1")
        self.assertEqual(nodes[0].need, "Find the cause")

    def test_invalid_output_falls_back_to_single_node(self):
        class BadModel:
            model_name = "bad"

            def generate(self, system_prompt, user_prompt, max_tokens=None):
                return "not json at all"

        with tempfile.TemporaryDirectory() as tmp:
            cache = JsonKVCache(Path(tmp) / "c.json", section="node_induction")
            result = induce_nodes(
                "Who dies?", "(no candidates)", BadModel(), cache, max_nodes=5, max_tokens=64
            )
        self.assertTrue(result["fallback"])
        self.assertEqual(len(result["nodes"]), 1)
        self.assertIn("Who dies?", result["nodes"][0].node_query)


class SupportScoringTests(unittest.TestCase):
    def test_parse_and_clamp(self):
        parsed = parse_support_output('{"support_score": 1.4, "support_type": "complete", "supporting_span": "abc"}')
        capped = validate_and_cap(parsed["fields"], "xx abc yy", parsed["parse_error"])
        self.assertEqual(capped["support_score"], 1.0)  # clamped to [0,1]
        self.assertEqual(capped["support_type"], "complete")

    def test_unsupported_span_caps_and_warns(self):
        parsed = parse_support_output('{"support_score": 0.9, "support_type": "complete", "supporting_span": "NOT PRESENT"}')
        capped = validate_and_cap(parsed["fields"], "candidate text here", parsed["parse_error"])
        self.assertLessEqual(capped["support_score"], 0.5)
        self.assertTrue(capped["validation_warnings"])

    def test_complete_with_empty_span_capped(self):
        parsed = parse_support_output('{"support_score": 1.0, "support_type": "complete", "supporting_span": ""}')
        capped = validate_and_cap(parsed["fields"], "candidate", parsed["parse_error"])
        self.assertLessEqual(capped["support_score"], 0.7)

    def test_contradictory_is_zero(self):
        parsed = parse_support_output('{"support_score": 0.9, "support_type": "contradictory", "supporting_span": ""}')
        capped = validate_and_cap(parsed["fields"], "candidate", parsed["parse_error"])
        self.assertEqual(capped["support_score"], 0.0)


class UnitConstructionTests(unittest.TestCase):
    def setUp(self):
        self.chunks = {
            "c1": make_chunk("c1", 0, "act01_scene01", "BARNARDO.\nWho is there now.\n\nFRANCISCO.\nAnswer me first."),
            "c2": make_chunk("c2", 1, "act01_scene01", "MARCELLUS.\nWhat has appeared again tonight."),
            "c3": make_chunk("c3", 2, "act01_scene02", "KING.\nA different scene begins right here."),
        }
        self.trace = [{"chunk_id": cid} for cid in ("c1", "c2", "c3")]

    def test_raw_chunk_units_generated(self):
        built = build_units(
            self.trace, self.chunks, candidate_chunks=30,
            unit_types=["chunk"], include_neighbors=False, neighbor_hops=1,
            max_unit_tokens=512, max_units_total=200,
        )
        chunk_units = [u for u in built["units"] if u.unit_type == "chunk"]
        self.assertEqual(len(chunk_units), 3)
        self.assertEqual(chunk_units[0].text, self.chunks["c1"]["text"])  # source-extractive

    def test_neighbors_stay_within_scene(self):
        built = build_units(
            self.trace, self.chunks, candidate_chunks=30,
            unit_types=["chunk", "neighbor_left", "neighbor_right"],
            include_neighbors=True, neighbor_hops=1,
            max_unit_tokens=512, max_units_total=200,
        )
        neighbor_units = [u for u in built["units"] if u.unit_type.startswith("neighbor")]
        # c1<->c2 share scene; c3 is a different scene and must not merge across.
        for unit in neighbor_units:
            scene_ids = {self.chunks[cid]["scene_id"] for cid in unit.source_chunk_ids}
            self.assertEqual(len(scene_ids), 1)
        self.assertNotIn("c3", {cid for u in neighbor_units for cid in u.source_chunk_ids})

    def test_max_unit_tokens_drops_oversize(self):
        built = build_units(
            self.trace, self.chunks, candidate_chunks=30,
            unit_types=["chunk"], include_neighbors=False, neighbor_hops=1,
            max_unit_tokens=5, max_units_total=200,
        )
        self.assertEqual(built["units"], [])  # every 20-token chunk exceeds 5
        self.assertEqual(built["dropped"]["oversize"], 3)


class CoverageAndGreedyTests(unittest.TestCase):
    def test_coverage_diminishing_returns(self):
        # Two units each support n1 at 0.7; coverage after both = 1-(0.3*0.3)=0.91.
        nodes = [make_node("n1", "x")]
        u_a = make_unit("a", "alpha text", primary="cA")
        u_b = make_unit("b", "totally different beta words", primary="cB")
        matrix = {("n1", "a"): 0.7, ("n1", "b"): 0.7}
        out = greedy_select(
            [u_a, u_b], nodes, matrix,
            context_budget=1000, beta=0.15, tau=0.7,
            min_unit_score=0.45, coverage_threshold=0.99, max_selected=8,
        )
        final_cov = 1.0 - out["miss"]["n1"]
        self.assertAlmostEqual(final_cov, 0.91, places=5)
        # The second unit's coverage gain is smaller than the first.
        self.assertLess(out["steps"][1]["coverage_gain"], out["steps"][0]["coverage_gain"])

    def test_prefers_complementary_over_redundant(self):
        nodes = [make_node("n1", "x", 1), make_node("n2", "y", 2)]
        # uA and uB redundantly support n1 (same source chunk, similar text);
        # uC uniquely supports n2.
        u_a = make_unit("a", "the summon reason here", primary="cX")
        u_b = make_unit("b", "the summon reason here too", primary="cX")
        u_c = make_unit("c", "england task entirely different", primary="cY")
        matrix = {
            ("n1", "a"): 0.9, ("n1", "b"): 0.9, ("n2", "c"): 0.9,
        }
        out = greedy_select(
            [u_a, u_b, u_c], nodes, matrix,
            context_budget=1000, beta=0.15, tau=0.7,
            min_unit_score=0.45, coverage_threshold=0.85, max_selected=8,
        )
        chosen = {u.unit_id for u in out["selected"]}
        self.assertIn("c", chosen)  # the n2 unit must be selected
        self.assertFalse({"a", "b"}.issubset(chosen))  # not both redundant n1 units

    def test_respects_budget(self):
        nodes = [make_node("n1", "x"), make_node("n2", "y"), make_node("n3", "z")]
        units = [
            make_unit("a", "aaa", primary="cA", token_count=600),
            make_unit("b", "bbb", primary="cB", token_count=600),
        ]
        matrix = {("n1", "a"): 0.9, ("n2", "b"): 0.9}
        out = greedy_select(
            units, nodes, matrix,
            context_budget=1000, beta=0.15, tau=0.7,
            min_unit_score=0.45, coverage_threshold=0.85, max_selected=8,
        )
        self.assertEqual(len(out["selected"]), 1)  # only one 600-token unit fits

    def test_empty_when_all_below_threshold(self):
        nodes = [make_node("n1", "x")]
        units = [make_unit("a", "aaa")]
        matrix = {("n1", "a"): 0.3}
        out = greedy_select(
            units, nodes, matrix,
            context_budget=1000, beta=0.15, tau=0.7,
            min_unit_score=0.45, coverage_threshold=0.85, max_selected=8,
        )
        self.assertEqual(out["selectable"], [])
        self.assertEqual(out["selected"], [])


class OrderingTests(unittest.TestCase):
    def test_anchor_first_then_node_doc_order(self):
        nodes = [make_node("n1", "x", order_index=1), make_node("n2", "y", order_index=2)]
        anchor = make_unit("anchor", "anchor", order=(5, 0))  # strongest support
        early_n1 = make_unit("e1", "e1", order=(1, 0))
        late_n2 = make_unit("l2", "l2", order=(3, 0))
        matrix = {
            ("n1", "anchor"): 0.95,
            ("n1", "e1"): 0.6,
            ("n2", "l2"): 0.6,
        }
        ordered = _order_units([early_n1, late_n2, anchor], nodes, matrix, "anchor_then_node_doc_order")
        self.assertEqual(ordered[0].unit_id, "anchor")
        # remaining ordered by strongest-node order_index: n1 (e1) before n2 (l2)
        self.assertEqual([u.unit_id for u in ordered[1:]], ["e1", "l2"])


class EvidenceRoleMetricTests(unittest.TestCase):
    def test_role_recall_counts_distinct_roles(self):
        row = {
            "required_quotes_present_in_context": [
                {"role": "summon", "present": True},
                {"role": "england", "present": False},
                {"role": "fate", "present": True},
            ]
        }
        out = compute_evidence_role_recall_for_row(row)
        self.assertEqual(out["evidence_roles_total"], 3)
        self.assertEqual(out["evidence_roles_covered"], 2)
        self.assertAlmostEqual(out["evidence_role_recall"], 2 / 3)

    def test_unanswerable_returns_null(self):
        out = compute_evidence_role_recall_for_row({"required_quotes_present_in_context": []})
        self.assertIsNone(out["evidence_role_recall"])


class IntegrationTests(unittest.TestCase):
    def _request(self, tmp):
        chunks = {
            "c1": make_chunk("c1", 0, "act01_scene01", "KING. The SUMMONMARK reason is Hamlet transformation here."),
            "c2": make_chunk("c2", 1, "act01_scene01", "KING. The ENGLANDMARK task sends them to England now."),
            "c3": make_chunk("c3", 2, "act01_scene01", "AMBASSADOR. The FATEMARK fate is that they are dead."),
        }
        trace = [{"chunk_id": cid, "rank": i + 1} for i, cid in enumerate(("c1", "c2", "c3"))]

        class Q:
            id = "q_test"
            question = "Why summon them, what England task, and what fate?"

        return ContextAssemblyRequest(
            question=Q(),
            treatment="reader_support",
            context_budget=1000,
            chunk_lookup=chunks,
            doc_order_ids=["c1", "c2", "c3"],
            retrieval_trace=trace,
            selector_model=ScriptedSupportModel(),
            feature_params={"support_score_cache_path": str(Path(tmp) / "rs_cache.json")},
        )

    def test_assemble_covers_all_nodes_source_faithfully(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = self._request(tmp)
            result = assemble_reader_support(request)
            self.assertEqual(result.retrieval_method, "reader_support")
            joined = " ".join(c["text"] for c in result.selected_chunks)
            for marker in ("SUMMONMARK", "ENGLANDMARK", "FATEMARK"):
                self.assertIn(marker, joined)  # all evidence needs covered, verbatim
            trace = result.context_assembly_trace
            self.assertEqual(len(trace["nodes"]), 3)
            self.assertEqual(trace["coverage"]["num_nodes_covered_above_threshold"], 3)
            self.assertTrue(Path(tmp, "rs_cache.json").exists())  # cached

    def test_assemble_is_cached_on_second_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = self._request(tmp)
            assemble_reader_support(request)
            model2 = ScriptedSupportModel()
            request2 = ContextAssemblyRequest(
                question=request.question,
                treatment="reader_support",
                context_budget=1000,
                chunk_lookup=request.chunk_lookup,
                doc_order_ids=request.doc_order_ids,
                retrieval_trace=request.retrieval_trace,
                selector_model=model2,
                feature_params=request.feature_params,
            )
            assemble_reader_support(request2)
            self.assertEqual(model2.calls, [])  # everything served from cache


if __name__ == "__main__":
    unittest.main()
