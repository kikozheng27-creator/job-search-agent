"""Dedicated exhaustive skill-inventory extraction over evidence units.

This is a narrow enumeration stage. It does not score the candidate and
does not own required vs preferred classification.
"""

from __future__ import annotations

from job_search_agent.evidence_units import (
    EvidenceUnit,
    build_evidence_units,
    skill_candidate_units,
)
from job_search_agent.models import (
    SkillExtractionReport,
    SkillExtractionStatus,
    SkillInventory,
)
from job_search_agent.skill_grounding import (
    ground_skill_inventory,
    has_explicit_skill_demand,
    is_contextual_non_requirement,
    lists_from_grounded_unit_names,
    looks_like_multi_skill_unit,
    union_grounded_names,
)


def inventory_coverage(
    inventory: SkillInventory,
    expected_ids: list[str],
) -> dict[str, list[str]]:
    """Compare returned unit ids against the candidate ids supplied."""
    expected = list(expected_ids)
    returned: list[str] = []
    seen: set[str] = set()
    for entry in inventory.units:
        unit_id = entry.source_unit_id
        if unit_id not in seen:
            returned.append(unit_id)
            seen.add(unit_id)

    expected_set = set(expected)
    returned_set = set(returned)
    missing = [unit_id for unit_id in expected if unit_id not in returned_set]
    unexpected = [unit_id for unit_id in returned if unit_id not in expected_set]
    return {
        "expected_unit_ids": expected,
        "returned_unit_ids": returned,
        "missing_unit_ids": missing,
        "unexpected_unit_ids": unexpected,
    }


def detect_unresolved_units(
    candidates: list[EvidenceUnit],
    returned_ids: set[str],
    grounded: dict[str, list[str]],
) -> list[str]:
    """Omitted units, or empty skills where the source clearly names skills."""
    unresolved: list[str] = []

    for unit in candidates:
        if unit.id not in returned_ids:
            unresolved.append(unit.id)
            continue
        if is_contextual_non_requirement(unit.text):
            continue
        names = grounded.get(unit.id, [])
        if has_explicit_skill_demand(unit.text) and not names:
            unresolved.append(unit.id)
            continue
        if (
            has_explicit_skill_demand(unit.text)
            and looks_like_multi_skill_unit(unit.text)
            and len(names) < 2
        ):
            unresolved.append(unit.id)

    return unresolved


def collect_skill_inventory(
    ai_client,
    job_description: str,
    units: list[EvidenceUnit] | None = None,
) -> tuple[list[str], list[str], SkillExtractionReport]:
    """Run primary extraction plus at most one repair pass."""
    from job_search_agent.prompts import (
        build_skill_inventory_prompt,
        build_skill_repair_prompt,
    )

    source_units = units if units is not None else build_evidence_units(job_description)
    candidates = skill_candidate_units(source_units)
    expected_ids = [unit.id for unit in candidates]

    primary = ai_client.extract_skill_inventory(
        build_skill_inventory_prompt(candidates)
    )
    coverage = inventory_coverage(primary, expected_ids)
    grounded = ground_skill_inventory(primary, source_units)
    returned_ids = set(coverage["returned_unit_ids"])
    unresolved = detect_unresolved_units(candidates, returned_ids, grounded)

    repair_attempted = False
    repaired_unit_ids: list[str] = []

    if unresolved:
        repair_attempted = True
        repaired_unit_ids = list(unresolved)
        unresolved_units = [
            unit for unit in candidates if unit.id in set(unresolved)
        ]
        repair = ai_client.extract_skill_inventory(
            build_skill_repair_prompt(unresolved_units)
        )
        repair_coverage = inventory_coverage(repair, repaired_unit_ids)
        repair_grounded = ground_skill_inventory(repair, source_units)
        grounded = union_grounded_names(grounded, repair_grounded)
        returned_ids.update(repair_coverage["returned_unit_ids"])
        coverage["returned_unit_ids"] = [
            unit_id
            for unit_id in expected_ids
            if unit_id in returned_ids
        ] + [
            unit_id
            for unit_id in coverage["returned_unit_ids"]
            if unit_id not in set(expected_ids) and unit_id in returned_ids
        ]
        coverage["missing_unit_ids"] = [
            unit_id for unit_id in expected_ids if unit_id not in returned_ids
        ]
        coverage["unexpected_unit_ids"] = [
            unit_id
            for unit_id in sorted(returned_ids)
            if unit_id not in set(expected_ids)
        ]
        unresolved = detect_unresolved_units(candidates, returned_ids, grounded)

    required, preferred = lists_from_grounded_unit_names(grounded, source_units)
    status = (
        SkillExtractionStatus.UNRESOLVED
        if unresolved
        else SkillExtractionStatus.COMPLETE
    )

    per_unit = {
        unit.id: list(grounded.get(unit.id, []))
        for unit in candidates
    }

    report = SkillExtractionReport(
        status=status,
        expected_unit_ids=expected_ids,
        returned_unit_ids=coverage["returned_unit_ids"],
        missing_unit_ids=coverage["missing_unit_ids"],
        unexpected_unit_ids=coverage["unexpected_unit_ids"],
        unresolved_unit_ids=unresolved,
        repair_attempted=repair_attempted,
        repaired_unit_ids=repaired_unit_ids,
        per_unit_skills=per_unit,
        skills_scored_from_inventory=False,
    )
    return required, preferred, report
