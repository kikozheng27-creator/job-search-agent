"""Deterministic education_score from degree level only.

The LLM may still emit scores.education. This module is the final authority
for the education component score. It does not read field of study, and it
does not add a hard filter.

Degree levels, lowest to highest:

    high_school, associate, bachelor, master, doctorate

Shortfall when the best parseable candidate degree is below the required
level. There was no prior education-score formula in this repository.

    1 level below → 70
    2 levels below → 40
    3 or more levels below, or no parseable candidate degree → 0

A required_degree string that names no known level, or that names several
levels without a clear alternative ("or") or a preferred-only clause, scores
UNPARSED_REQUIRED_DEGREE_SCORE. When several levels are explicit alternatives,
the required level is the lowest one. A preferred degree does not raise it.
"""

from __future__ import annotations

import re

from job_search_agent.candidate import CandidateProfile
from job_search_agent.models import JobRequirements


UNPARSED_REQUIRED_DEGREE_SCORE = 0

_LEVELS = ("high_school", "associate", "bachelor", "master", "doctorate")
_RANK = {level: index for index, level in enumerate(_LEVELS)}

_NO_DEGREE_RANK = -1

# A degree the posting allows experience to replace is not an unconditional
# requirement. This does not score that experience.
_EXPERIENCE_SUBSTITUTE = re.compile(
    r"\bequivalent\b(?:\s+\w+){0,8}\s+experience\b",
    re.IGNORECASE,
)

_WORD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "doctorate",
        re.compile(r"(?<![A-Za-z])ph\.?\s*d\.?(?![A-Za-z])", re.IGNORECASE),
    ),
    (
        "doctorate",
        re.compile(r"\b(?:doctorate|doctoral)\b", re.IGNORECASE),
    ),
    (
        "master",
        re.compile(r"\bmaster(?:['’])?s?\b", re.IGNORECASE),
    ),
    (
        "bachelor",
        re.compile(r"\bbachelor(?:['’])?s?\b", re.IGNORECASE),
    ),
    (
        "associate",
        re.compile(
            r"\bassociates\s+degree\b"
            r"|\bassociate(?:['’]s)?\s+degree\b"
            r"|\bassociate['’]s\b",
            re.IGNORECASE,
        ),
    ),
    (
        "high_school",
        re.compile(
            r"\b(?:high[\s-]+school|secondary[\s-]+school)\b",
            re.IGNORECASE,
        ),
    ),
)

# Two-letter abbreviations are uppercase or dotted so the word "as" does not
# become an associate degree.
_ABBREVIATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("master", re.compile(r"(?<![A-Za-z])M\.?\s*S\.?(?![A-Za-z])")),
    ("master", re.compile(r"(?<![A-Za-z])M\.?\s*A\.?(?![A-Za-z])")),
    ("master", re.compile(r"(?<![A-Za-z])m\.\s*s\.?(?![A-Za-z])", re.IGNORECASE)),
    ("master", re.compile(r"(?<![A-Za-z])m\.\s*a\.?(?![A-Za-z])", re.IGNORECASE)),
    ("bachelor", re.compile(r"(?<![A-Za-z])B\.?\s*S\.?(?![A-Za-z])")),
    ("bachelor", re.compile(r"(?<![A-Za-z])B\.?\s*A\.?(?![A-Za-z])")),
    ("bachelor", re.compile(r"(?<![A-Za-z])b\.\s*s\.?(?![A-Za-z])", re.IGNORECASE)),
    ("bachelor", re.compile(r"(?<![A-Za-z])b\.\s*a\.?(?![A-Za-z])", re.IGNORECASE)),
    ("associate", re.compile(r"(?<![A-Za-z])A\.?\s*A\.?(?![A-Za-z])")),
    ("associate", re.compile(r"(?<![A-Za-z])A\.?\s*S\.?(?![A-Za-z])")),
    ("associate", re.compile(r"(?<![A-Za-z])a\.\s*a\.?(?![A-Za-z])", re.IGNORECASE)),
    ("associate", re.compile(r"(?<![A-Za-z])a\.\s*s\.?(?![A-Za-z])", re.IGNORECASE)),
)


def degree_levels_in(text: str) -> set[str]:
    """Known degree levels explicitly named in text. Empty when none match."""
    found: set[str] = set()
    for level, pattern in (*_WORD_PATTERNS, *_ABBREVIATION_PATTERNS):
        if pattern.search(text):
            found.add(level)
    return found


def _lowest_level(levels: set[str]) -> str:
    return min(levels, key=_RANK.__getitem__)


def _experience_substitutes_degree(clause: str) -> bool:
    """True when this clause offers experience in place of its degree.

    The substitute must sit in the same clause and be an alternative ("or"),
    so a later clause cannot cancel a separately required degree.
    """
    if not degree_levels_in(clause):
        return False
    if not re.search(r"\bor\b", clause, re.IGNORECASE):
        return False
    return bool(_EXPERIENCE_SUBSTITUTE.search(clause))


def normalize_degree_level(text: str) -> str | None:
    """Return the lowest acceptable required level, or None when it is unclear.

    One explicit level is that level, including "Bachelor's or higher".
    Several levels joined by "or" resolve to the lowest. A clause that only
    marks a degree as preferred is ignored. Bare "diploma" and bare
    "Associate" do not match.
    """
    clauses = [part.strip() for part in re.split(r";", text) if part.strip()]
    accepted: list[str] = []

    for clause in clauses:
        levels = degree_levels_in(clause)
        if not levels:
            continue
        preferred_only = bool(re.search(r"\bpreferred\b", clause, re.IGNORECASE)) and not bool(
            re.search(r"\brequired\b", clause, re.IGNORECASE)
        )
        if preferred_only or _experience_substitutes_degree(clause):
            continue
        if len(levels) == 1:
            accepted.append(next(iter(levels)))
            continue
        if re.search(r"\bor\b", clause, re.IGNORECASE):
            accepted.append(_lowest_level(levels))
            continue
        return None

    if not accepted:
        return None

    unique = set(accepted)
    if len(unique) == 1:
        return next(iter(unique))
    if re.search(r"\bor\b", text, re.IGNORECASE):
        return _lowest_level(unique)
    return None


def _no_unconditional_requirement(text: str) -> bool:
    """True when no clause states a degree the candidate must hold."""
    saw_degree = False
    for clause in (part.strip() for part in re.split(r";", text) if part.strip()):
        if not degree_levels_in(clause):
            continue
        saw_degree = True
        if _experience_substitutes_degree(clause):
            continue
        preferred_only = bool(
            re.search(r"\bpreferred\b", clause, re.IGNORECASE)
        ) and not bool(re.search(r"\brequired\b", clause, re.IGNORECASE))
        if not preferred_only:
            return False
    return saw_degree


def _highest_candidate_rank(profile: CandidateProfile) -> int | None:
    ranks = [
        _RANK[level]
        for degree in profile.education
        if (level := normalize_degree_level(degree.degree)) is not None
    ]
    if not ranks:
        return None
    return max(ranks)


def _score_for_gap(gap: int) -> int:
    if gap <= 0:
        return 100
    if gap == 1:
        return 70
    if gap == 2:
        return 40
    return 0


def score_education(
    profile: CandidateProfile,
    requirements: JobRequirements,
) -> int:
    """Degree-level score in [0, 100]. None means the posting states no level."""
    required = requirements.required_degree
    if required is None or _no_unconditional_requirement(required):
        return 100

    required_level = normalize_degree_level(required)
    if required_level is None:
        return UNPARSED_REQUIRED_DEGREE_SCORE

    candidate_rank = _highest_candidate_rank(profile)
    if candidate_rank is None:
        candidate_rank = _NO_DEGREE_RANK

    return _score_for_gap(_RANK[required_level] - candidate_rank)
