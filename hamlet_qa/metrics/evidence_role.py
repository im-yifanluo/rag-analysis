"""Evidence-role recall: did the assembled context cover each evidence role?

Reader-free, computed from the row alone. Each required evidence quote carries a
`role` (e.g. poison_identity, effect_on_blood). A role is covered when any of its
quotes is present in the assembled context — either by selected raw chunk ID or
by verbatim text containment (both already recorded per quote in
`required_quotes_present_in_context`). Unanswerable questions (no required
quotes) return null, not 0.
"""

from __future__ import annotations

from typing import Any


def compute_evidence_role_recall_for_row(row: dict[str, Any]) -> dict[str, Any]:
    quotes = row.get("required_quotes_present_in_context") or []
    if not quotes:
        return {
            "evidence_role_recall": None,
            "evidence_roles_total": 0,
            "evidence_roles_covered": 0,
        }
    covered_by_role: dict[str, bool] = {}
    for index, quote in enumerate(quotes):
        role = str(quote.get("role") or f"quote_{quote.get('quote_index', index)}")
        present = bool(quote.get("present"))
        covered_by_role[role] = covered_by_role.get(role, False) or present
    total = len(covered_by_role)
    covered = sum(1 for is_covered in covered_by_role.values() if is_covered)
    return {
        "evidence_role_recall": covered / total if total else None,
        "evidence_roles_total": total,
        "evidence_roles_covered": covered,
    }
