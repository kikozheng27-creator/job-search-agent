"""Ground skill decisions in Python-derived evidence units.

The LLM interprets a fixed set of source units. Python rejects invalid
ids, skills that do not appear in the referenced unit, non-skill
categories, and contextual technology mentions that are not candidate
requirements. A skill name appearing elsewhere in the posting is not
enough.
"""

from __future__ import annotations

import re

from job_search_agent.evidence_units import (
    EvidenceUnit,
    build_evidence_units,
    units_by_id,
)
from job_search_agent.models import (
    JobRequirements,
    SkillClaim,
    SkillInventory,
    SkillLevel,
    UnitSkillDecision,
)
from job_search_agent.skill_scorer import is_non_skill_requirement, normalize_skill


_WHITESPACE = re.compile(r"\s+")
_ALNUM = re.compile(r"[^a-z0-9]+")

_PREFERRED_MARKERS = re.compile(
    r"""
    \b(
        preferred(?:\s+qualifications?)?
        | desired
        | optional
        | bonus
        | plus
        | nice\s+to\s+have
        | a\s+plus
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_REQUIREMENT_MARKERS = re.compile(
    r"""
    \b(
        required
        | must
        | need(?:s|ed)?
        | proficiency
        | proficient
        | qualifications?
        | responsibilit(?:y|ies)
        | experience\s+with
        | demonstrated?
        | expert(?:ise|\s+knowledge)
        | knowledge\s+of
        | minimum
        | should\s+have
        | looking\s+for
        | ability\s+to
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_EXPLICIT_SKILL_DEMAND = re.compile(
    r"""
    \b(
        proficiency\s+in
        | experience\s+with
        | expertise\s+in
        | expert\s+knowledge\s+of
        | knowledge\s+of
        | ability\s+to\s+use
        | capabilities?\s+including
        | proficien(?:cy|t)\s+(?:in|with|using)
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CONTEXTUAL_MENTION = re.compile(
    r"""
    (
        \bour\s+team\s+uses\b
        | \bwe\s+use\b
        | \bthe\s+team\s+uses\b
        | \bwork(?:s|ing)?\s+with\s+.{0,60}\busing\b
        | \banalysts\s+using\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_NON_SKILL_CATEGORIES = {
    "authorization",
    "certification",
    "clearance",
    "course",
    "credential",
    "degree",
    "education",
    "not_a_skill",
    "training",
    "years_of_experience",
}


def _normalize_source(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return _WHITESPACE.sub(" ", text).strip().casefold()


def _alnum(text: str) -> str:
    return _ALNUM.sub("", text.casefold())


def evidence_in_source(evidence: str, job_description: str) -> bool:
    """True only when the quote is a contiguous substring of the posting."""
    quote = _normalize_source(evidence)

    if not quote:
        return False

    return quote in _normalize_source(job_description)


def skill_mentioned_in_evidence(name: str, evidence: str) -> bool:
    """The claimed skill must actually appear in the referenced text."""
    compact_name = _alnum(name)

    if not compact_name:
        return False

    if len(compact_name) <= 2:
        return bool(
            re.search(rf"\b{re.escape(name.strip())}\b", evidence, re.IGNORECASE)
        )

    return compact_name in _alnum(evidence)


def classify_skill_level(evidence: str) -> SkillLevel:
    """Preferred only when the source unit marks the skill as optional."""
    if _PREFERRED_MARKERS.search(_normalize_source(evidence)):
        return SkillLevel.PREFERRED

    return SkillLevel.REQUIRED


def is_contextual_non_requirement(text: str) -> bool:
    """True for technology mentions that are not candidate requirements."""
    normalized = _normalize_source(text)

    if classify_skill_level(text) is SkillLevel.PREFERRED:
        return False

    if _REQUIREMENT_MARKERS.search(normalized):
        return False

    return bool(_CONTEXTUAL_MENTION.search(normalized))


def _is_scored_skill_category(category: str) -> bool:
    return category.strip().casefold() not in _NON_SKILL_CATEGORIES


def _canonical_key(name: str) -> str:
    return normalize_skill(name).replace(" ", "")


def _occurrence_index(name: str, unit_text: str) -> int:
    haystack = unit_text.casefold()
    needle = name.strip().casefold()
    if needle:
        index = haystack.find(needle)
        if index >= 0:
            return index

    compact_haystack = _alnum(unit_text)
    compact_name = _alnum(name)
    if compact_name:
        index = compact_haystack.find(compact_name)
        if index >= 0:
            return index

    return 10**9


def _append_skill(
    name: str,
    level: SkillLevel,
    required: list[str],
    preferred: list[str],
    seen: dict[str, SkillLevel],
) -> None:
    key = _canonical_key(name)
    if not key:
        return

    previous = seen.get(key)

    if previous is SkillLevel.REQUIRED:
        return

    if previous is SkillLevel.PREFERRED:
        if level is SkillLevel.REQUIRED:
            preferred[:] = [
                item
                for item in preferred
                if _canonical_key(item) != key
            ]
            required.append(name)
            seen[key] = SkillLevel.REQUIRED
        return

    seen[key] = level
    if level is SkillLevel.PREFERRED:
        preferred.append(name)
    else:
        required.append(name)


def _accept_skill_name(name: str, category: str, unit_text: str) -> bool:
    stripped = name.strip()
    if not stripped or is_non_skill_requirement(stripped):
        return False
    if not _is_scored_skill_category(category):
        return False
    return skill_mentioned_in_evidence(stripped, unit_text)


def ground_unit_decisions(
    decisions: list[UnitSkillDecision],
    units: list[EvidenceUnit],
) -> tuple[list[str], list[str]]:
    """Build required/preferred lists from per-unit decisions."""
    lookup = units_by_id(units)
    by_unit: dict[str, list[UnitSkillDecision]] = {}

    for decision in decisions:
        if decision.source_unit_id not in lookup:
            continue
        by_unit.setdefault(decision.source_unit_id, []).append(decision)

    required: list[str] = []
    preferred: list[str] = []
    seen: dict[str, SkillLevel] = {}

    for unit in units:
        unit_decisions = by_unit.get(unit.id, [])
        if not unit_decisions:
            continue
        if is_contextual_non_requirement(unit.text):
            continue

        requirement_like = bool(
            _REQUIREMENT_MARKERS.search(_normalize_source(unit.text))
        ) or classify_skill_level(unit.text) is SkillLevel.PREFERRED
        llm_says_requirement = any(
            item.is_candidate_requirement for item in unit_decisions
        )
        if not llm_says_requirement and not requirement_like:
            continue

        level = classify_skill_level(unit.text)
        names: list[str] = []
        seen_in_unit: set[str] = set()

        for decision in unit_decisions:
            for item in decision.skills:
                if not _accept_skill_name(item.name, item.category, unit.text):
                    continue
                from job_search_agent.skill_selection import select_scored_skill

                selected = select_scored_skill(item.name, unit.text)
                if not selected:
                    continue
                key = _canonical_key(selected)
                if not key or key in seen_in_unit:
                    continue
                seen_in_unit.add(key)
                names.append(selected)

        names.sort(key=lambda name: (_occurrence_index(name, unit.text), _canonical_key(name)))

        for name in names:
            _append_skill(name, level, required, preferred, seen)

    return required, preferred


def ground_skill_claims(
    claims: list[SkillClaim],
    job_description: str,
    units: list[EvidenceUnit] | None = None,
) -> tuple[list[str], list[str]]:
    """Ground legacy claim objects. Unit ids beat free-form quotes.

    A skill name appearing somewhere in the posting is not evidence.
    """
    source_units = units if units is not None else build_evidence_units(job_description)
    lookup = units_by_id(source_units)
    required: list[str] = []
    preferred: list[str] = []
    seen: dict[str, SkillLevel] = {}

    for claim in claims:
        name = claim.name.strip()
        if not name or is_non_skill_requirement(name):
            continue
        if not _is_scored_skill_category(claim.category):
            continue

        supporting_texts: list[str] = []

        if claim.source_unit_ids:
            for unit_id in claim.source_unit_ids:
                unit = lookup.get(unit_id)
                if unit is None:
                    continue
                if not skill_mentioned_in_evidence(name, unit.text):
                    continue
                if is_contextual_non_requirement(unit.text):
                    continue
                supporting_texts.append(unit.text)
        elif evidence_in_source(claim.evidence, job_description):
            if not skill_mentioned_in_evidence(name, claim.evidence):
                continue
            if is_contextual_non_requirement(claim.evidence):
                continue
            supporting_texts.append(claim.evidence)
        else:
            continue

        if not supporting_texts:
            continue

        from job_search_agent.skill_selection import select_scored_skill

        selected = None
        for text in supporting_texts:
            selected = select_scored_skill(name, text)
            if selected:
                break
        if not selected:
            continue

        level = SkillLevel.PREFERRED
        if any(
            classify_skill_level(text) is SkillLevel.REQUIRED
            for text in supporting_texts
        ):
            level = SkillLevel.REQUIRED

        _append_skill(selected, level, required, preferred, seen)

    return required, preferred


def has_explicit_skill_demand(text: str) -> bool:
    """True when the unit uses wording that names required/preferred skills."""
    return bool(_EXPLICIT_SKILL_DEMAND.search(_normalize_source(text)))


def looks_like_multi_skill_unit(text: str) -> bool:
    """True when the unit looks like a list of more than one named item."""
    return bool(
        re.search(r",\s*(and\s+)?[A-Za-z]", text)
        or re.search(r"\band\s+[A-Z]", text)
        or re.search(r"\band/or\b", text, re.IGNORECASE)
    )


def ground_skill_inventory(
    inventory: SkillInventory,
    units: list[EvidenceUnit],
) -> dict[str, list[str]]:
    """Map each known unit id to grounded skill names from that unit only."""
    lookup = units_by_id(units)
    grounded: dict[str, list[str]] = {}

    for entry in inventory.units:
        unit = lookup.get(entry.source_unit_id)
        if unit is None:
            continue
        names: list[str] = []
        seen: set[str] = set()
        for mention in entry.skills:
            if not _accept_skill_name(mention.name, mention.category, unit.text):
                continue
            key = _canonical_key(mention.name)
            if not key or key in seen:
                continue
            seen.add(key)
            names.append(mention.name.strip())
        names.sort(
            key=lambda name: (_occurrence_index(name, unit.text), _canonical_key(name))
        )
        if entry.source_unit_id in grounded:
            grounded[entry.source_unit_id] = union_grounded_names(
                {entry.source_unit_id: grounded[entry.source_unit_id]},
                {entry.source_unit_id: names},
            )[entry.source_unit_id]
        else:
            grounded[entry.source_unit_id] = names

    return grounded


def union_grounded_names(
    primary: dict[str, list[str]],
    extra: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Deterministic union. Extra may add grounded names; it cannot remove."""
    merged = {unit_id: list(names) for unit_id, names in primary.items()}

    for unit_id, names in extra.items():
        current = merged.setdefault(unit_id, [])
        seen = {_canonical_key(name) for name in current}
        for name in names:
            key = _canonical_key(name)
            if not key or key in seen:
                continue
            current.append(name)
            seen.add(key)

    return merged


def lists_from_grounded_unit_names(
    grounded: dict[str, list[str]],
    units: list[EvidenceUnit],
) -> tuple[list[str], list[str]]:
    """Python-owned required/preferred lists from per-unit grounded names."""
    required: list[str] = []
    preferred: list[str] = []
    seen: dict[str, SkillLevel] = {}

    for unit in units:
        names = grounded.get(unit.id, [])
        if not names:
            continue
        if is_contextual_non_requirement(unit.text):
            continue
        level = classify_skill_level(unit.text)
        from job_search_agent.skill_selection import select_scored_skill

        for name in names:
            selected = select_scored_skill(name, unit.text)
            if not selected:
                continue
            _append_skill(selected, level, required, preferred, seen)

    return required, preferred


def apply_skill_grounding(
    requirements: JobRequirements,
    claims: list[SkillClaim],
    job_description: str,
    *,
    unit_decisions: list[UnitSkillDecision] | None = None,
    units: list[EvidenceUnit] | None = None,
) -> JobRequirements:
    """Overwrite skill lists from grounded unit decisions or claims."""
    source_units = units if units is not None else build_evidence_units(job_description)

    if unit_decisions:
        required, preferred = ground_unit_decisions(unit_decisions, source_units)
        requirements.required_skills = required
        requirements.preferred_skills = preferred
        return requirements

    if not claims:
        return requirements

    required, preferred = ground_skill_claims(
        claims,
        job_description,
        units=source_units,
    )
    requirements.required_skills = required
    requirements.preferred_skills = preferred
    return requirements
