"""Deterministic skills_score from extracted skill lists and the candidate profile.

The LLM may still extract required_skills and preferred_skills. This module is
the final authority for the skills component score.
"""

from __future__ import annotations

import re

from job_search_agent.candidate import CandidateProfile
from job_search_agent.config_models import SkillsScoringConfig
from job_search_agent.models import JobRequirements


_NON_TOKEN = re.compile(r"[\s,;/|]+")
_PUNCT = re.compile(r"[^a-z0-9+#]+")

# Conservative: drop credential phrasing, keep bare field names like Biostatistics.
_DEGREE_LEAK = re.compile(
    r"""
    \b(
        bachelor'?s?
        | master'?s?
        | ph\.?d\.?
        | doctorate
        | doctoral
        | degree
        | diploma
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def normalize_skill(name: str) -> str:
    collapsed = _NON_TOKEN.sub(" ", name.strip().casefold())
    collapsed = _PUNCT.sub(" ", collapsed)
    return " ".join(collapsed.split())


def is_degree_requirement(name: str) -> bool:
    """True for leaked degree text, false for a domain skill that happens to be a field of study."""
    return bool(_DEGREE_LEAK.search(name))


_NON_SKILL = re.compile(
    r"""
    (
        \bclearance\b
        | \bpolygraph\b
        | \bpoly\b
        | ts/?sci
        | top\s+secret
        | secret\s+clearance
        | certificat\w*
        | \bcertified\b
        | \bcourse\b
        | \btraining\b
        | \bcurriculum\b
        | \bsponsorship\b
        | \bvisa\b
        | \bcitizenship\b
        | work\s+authorization
        | years?\s+of\s+experience
        | pep-?8
        | coding\s+standards?
        | integrated\s+development\s+environment
        | \bides?\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_non_skill_requirement(name: str) -> bool:
    """True for education, clearance, certs, training, authorization, or years."""
    if not name.strip():
        return True

    if is_degree_requirement(name):
        return True

    if re.search(r"[a-z]{2,}\d{3,}", name, re.IGNORECASE):
        return True

    return bool(_NON_SKILL.search(name))


def build_alias_lookup(aliases: dict[str, str]) -> dict[str, str]:
    lookup: dict[str, str] = {}

    for variant, canonical in aliases.items():
        canonical_key = normalize_skill(canonical)
        lookup[normalize_skill(variant)] = canonical_key
        lookup[canonical_key] = canonical_key

    return lookup


def canonical_skill(name: str, lookup: dict[str, str]) -> str:
    key = normalize_skill(name)
    resolved = lookup.get(key, key)
    return resolved.replace(" ", "")


def filter_skill_names(names: list[str]) -> list[str]:
    return [
        name
        for name in names
        if name.strip() and not is_non_skill_requirement(name)
    ]


def candidate_skill_set(
    profile: CandidateProfile,
    lookup: dict[str, str],
) -> set[str]:
    return {
        canonical_skill(name, lookup)
        for name in profile.skills.all_skills()
        if name.strip()
    }


def match_rate(
    extracted: list[str],
    candidate: set[str],
    lookup: dict[str, str],
) -> float:
    """Fraction of posting skills the candidate has. Empty posting list → 1.0."""
    filtered = filter_skill_names(extracted)

    if not filtered:
        return 1.0

    matched = sum(
        1
        for name in filtered
        if canonical_skill(name, lookup) in candidate
    )

    return matched / len(filtered)


def score_skills(
    profile: CandidateProfile,
    requirements: JobRequirements,
    config: SkillsScoringConfig,
) -> int:
    lookup = build_alias_lookup(config.aliases)
    candidate = candidate_skill_set(profile, lookup)

    required = match_rate(requirements.required_skills, candidate, lookup)
    preferred = match_rate(requirements.preferred_skills, candidate, lookup)

    raw = 100.0 * (
        config.required_weight * required
        + config.preferred_weight * preferred
    )

    return int(round(min(100.0, max(0.0, raw))))


def diagnose_skill_list(
    extracted: list[str],
    candidate: set[str],
    lookup: dict[str, str],
) -> dict:
    """Explain one extracted skill list. Does not change the score formula."""
    raw = list(extracted)
    dropped_degree_leaks = [
        name for name in raw if name.strip() and is_degree_requirement(name)
    ]
    filtered = filter_skill_names(raw)
    normalized = [canonical_skill(name, lookup) for name in filtered]
    matched = [
        name
        for name in filtered
        if canonical_skill(name, lookup) in candidate
    ]
    missing = [
        name
        for name in filtered
        if canonical_skill(name, lookup) not in candidate
    ]

    return {
        "raw": raw,
        "normalized": normalized,
        "matched": matched,
        "missing": missing,
        "dropped_degree_leaks": dropped_degree_leaks,
        "match_rate": round(match_rate(extracted, candidate, lookup), 4),
    }


def diagnose_skills(
    profile: CandidateProfile,
    requirements: JobRequirements,
    config: SkillsScoringConfig,
) -> dict:
    """Skill-match breakdown for evaluation. Reuses score_skills for the total."""
    lookup = build_alias_lookup(config.aliases)
    candidate = candidate_skill_set(profile, lookup)
    required = diagnose_skill_list(
        requirements.required_skills,
        candidate,
        lookup,
    )
    preferred = diagnose_skill_list(
        requirements.preferred_skills,
        candidate,
        lookup,
    )

    return {
        "required": required,
        "preferred": preferred,
        "required_match_rate": required["match_rate"],
        "preferred_match_rate": preferred["match_rate"],
        "skills_score": score_skills(profile, requirements, config),
    }
