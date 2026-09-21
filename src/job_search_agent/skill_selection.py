"""Select atomic scored skills from grounded mentions.

A grounded noun is not automatically a scored skill. Python keeps only
the smallest capability concept that the source unit treats as something
the candidate must know, use, apply, or have experience with.
"""

from __future__ import annotations

import re

from job_search_agent.skill_grounding import (
    SkillLevel,
    classify_skill_level,
    skill_mentioned_in_evidence,
)
from job_search_agent.skill_scorer import is_non_skill_requirement, normalize_skill


_QUALIFICATION_LEAD = re.compile(
    r"""
    (
        (?:expert|working)\s+knowledge\s+of
        | (?:expert\s+)?knowledge\s+of
        | experience\s+and\s+knowledge\s+of
        | experience\s+(?:with|in|using)
        | proficien(?:cy|t)\s+(?:in|with|using)
        | expertise\s+in
        | ability\s+to\s+use
        | capabilities?\s+including
        | familiar(?:ity)?\s+with
        | demonstrated?\s+(?:experience\s+and\s+)?knowledge\s+of
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_NAME_QUALIFIER = re.compile(
    r"""
    ^(?:
        (?:expert|working)\s+knowledge\s+of
        | knowledge\s+of
        | experience\s+(?:with|in|using)
        | proficien(?:cy|t)\s+(?:in|with|using)
        | expertise\s+in
        | ability\s+to\s+use
    )\s+
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ENV_PHRASE = re.compile(
    r"""
    (?:managing\s+)?(?:data\s+)?in\s+an?\s+(.+?)\s+environment
    | ^(.+?)\s+environment$
    | ^(.+?)\s+technologies$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_FILLER_PREFIX = re.compile(
    r"^(managing|conducting|developing|using|authoring|debugging|maintaining|refactoring|advanced)\s+",
    re.IGNORECASE,
)

_FILLER_SUFFIX = re.compile(
    r"\s+(concepts?|practices?|functions?|environment|technologies|applications|programs)$",
    re.IGNORECASE,
)

_DISCIPLINE = re.compile(
    r"""
    (
        \bcomputer\s+science\b
        | \bintelligence\s+practices?\b
        | ^intelligence$
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_GENERIC_CONCEPT = re.compile(
    r"""
    ^(
        (?:advanced\s+)?software\s+programs?
        | query\s+techniques?
        | models?
        | data\s+sources?
        | legacy\s+code
        | analytics
        | unstructured\s+data
        | structured\s+data
        | data
        | intelligence
        | analysis
        | software
        | programs?
        | code
    )$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PRESERVE_UPPER = {"sql", "r", "sas", "nlp", "ml", "ide", "vs"}


def atomic_capability_name(name: str) -> str | None:
    """Reduce a mention to the smallest grounded capability concept."""
    raw = " ".join(name.strip().split())
    if not raw:
        return None

    raw = _NAME_QUALIFIER.sub("", raw)
    if re.search(r"\band/or\b", raw, re.IGNORECASE):
        return None

    env_match = _ENV_PHRASE.search(raw)
    if env_match:
        captured = next((group for group in env_match.groups() if group), None)
        if captured:
            raw = captured.strip()

    raw = _FILLER_PREFIX.sub("", raw)
    raw = _FILLER_SUFFIX.sub("", raw)
    raw = " ".join(raw.split())
    if not raw:
        return None
    if len(raw.split()) > 4:
        return None
    return raw


def canonical_display_name(name: str) -> str:
    """Stable display form after case/space/hyphen normalization."""
    original_words = re.sub(r"[-_/]+", " ", name.strip()).split()
    words = normalize_skill(name).split()
    display = []
    for index, word in enumerate(words):
        original = original_words[index] if index < len(original_words) else word
        if word in _PRESERVE_UPPER or (len(word) <= 2 and word.isalpha()):
            display.append(word.upper())
        elif any(char.isupper() for char in original[1:]):
            display.append(original)
        else:
            display.append(word.capitalize())
    return " ".join(display)


def is_discipline_or_generic(name: str) -> bool:
    normalized = normalize_skill(name)
    if not normalized:
        return True
    if _DISCIPLINE.search(normalized):
        return True
    if _GENERIC_CONCEPT.search(normalized):
        return True
    return False


def qualification_spans(unit_text: str) -> list[str]:
    spans: list[str] = []
    for match in _QUALIFICATION_LEAD.finditer(unit_text):
        rest = unit_text[match.end() :]
        cut = re.split(r"(?<=[.!?])\s+", rest, maxsplit=1)[0]
        if cut.strip():
            spans.append(cut)
    return spans


def appears_in_qualification_span(name: str, unit_text: str) -> bool:
    return any(
        skill_mentioned_in_evidence(name, span)
        for span in qualification_spans(unit_text)
    )


def _name_stated_as_requirement(name: str, unit_text: str) -> bool:
    stripped = name.strip()
    if not stripped:
        return False
    escaped = re.escape(stripped)
    return bool(
        re.search(rf"\b{escaped}\s+experience\b", unit_text, re.IGNORECASE)
        or re.search(
            rf"\b{escaped}\s+is\s+(?:required|desired|preferred)\b",
            unit_text,
            re.IGNORECASE,
        )
    )


def is_qualification_supported(name: str, unit_text: str) -> bool:
    """True when the unit treats this name as a candidate qualification."""
    if appears_in_qualification_span(name, unit_text):
        return True
    if _name_stated_as_requirement(name, unit_text):
        return True
    if classify_skill_level(unit_text) is SkillLevel.PREFERRED:
        return skill_mentioned_in_evidence(name, unit_text)
    return False


def select_scored_skill(name: str, unit_text: str) -> str | None:
    """Return a canonical scored skill, or None if the mention should not score."""
    stripped = name.strip()
    if not stripped:
        return None

    atomic = atomic_capability_name(stripped)
    if not atomic:
        return None
    if is_non_skill_requirement(atomic):
        return None
    if is_discipline_or_generic(atomic):
        return None
    if not skill_mentioned_in_evidence(atomic, unit_text) and not skill_mentioned_in_evidence(
        stripped, unit_text
    ):
        return None
    if not (
        is_qualification_supported(atomic, unit_text)
        or is_qualification_supported(stripped, unit_text)
    ):
        return None
    return canonical_display_name(atomic)
