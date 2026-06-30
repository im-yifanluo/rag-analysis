"""Shared evidence primitives used by context-assembly features.

Features import these from `core/` so they never depend on one another:
- schema: EvidenceNode, EvidenceUnit, SupportScore
- coverage: greedy_select (budgeted submodular coverage) + lexical_prior
- catalog: build_candidate_catalog
- support_teacher: reader-as-judge support scoring
"""

from hamlet_qa.core.evidence.catalog import build_candidate_catalog
from hamlet_qa.core.evidence.coverage import greedy_select, lexical_prior
from hamlet_qa.core.evidence.schema import EvidenceNode, EvidenceUnit, SupportScore
from hamlet_qa.core.evidence.support_teacher import (
    ReaderTeacherSupportScorer,
    SupportScorer,
    TrainedSupportScorer,
    parse_support_output,
    validate_and_cap,
)

__all__ = [
    "EvidenceNode",
    "EvidenceUnit",
    "SupportScore",
    "greedy_select",
    "lexical_prior",
    "build_candidate_catalog",
    "ReaderTeacherSupportScorer",
    "SupportScorer",
    "TrainedSupportScorer",
    "parse_support_output",
    "validate_and_cap",
]
