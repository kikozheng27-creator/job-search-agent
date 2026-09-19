"""Keep user-facing prose consistent with deterministic sponsorship evidence.

The LLM may invent work-authorization claims. Python already owns the
sponsorship stance; this module drops restriction language that that stance
does not support. Non-authorization text is left alone.
"""

from __future__ import annotations

import re
from enum import Enum

from job_search_agent.models import SponsorshipStance


_WHITESPACE = re.compile(r"\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class AuthorizationClaim(str, Enum):
    STUDENT_STATUS_PROHIBITED = "student_status_prohibited"
    CITIZENSHIP_REQUIRED = "citizenship_required"
    PERMANENT_REQUIRED = "permanent_required"
    SPONSORSHIP_NOT_OFFERED = "sponsorship_not_offered"
    SPONSORSHIP_OFFERED = "sponsorship_offered"
    GENERIC_RESTRICTION = "generic_restriction"


_ALLOWED: dict[SponsorshipStance, frozenset[AuthorizationClaim]] = {
    SponsorshipStance.NOT_MENTIONED: frozenset(),
    SponsorshipStance.OFFERED: frozenset({AuthorizationClaim.SPONSORSHIP_OFFERED}),
    SponsorshipStance.NOT_OFFERED: frozenset(
        {
            AuthorizationClaim.SPONSORSHIP_NOT_OFFERED,
            AuthorizationClaim.GENERIC_RESTRICTION,
        }
    ),
    SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED: frozenset(
        {
            AuthorizationClaim.STUDENT_STATUS_PROHIBITED,
            AuthorizationClaim.PERMANENT_REQUIRED,
            AuthorizationClaim.SPONSORSHIP_NOT_OFFERED,
            AuthorizationClaim.GENERIC_RESTRICTION,
        }
    ),
    SponsorshipStance.CITIZENSHIP_REQUIRED: frozenset(
        {
            AuthorizationClaim.CITIZENSHIP_REQUIRED,
            AuthorizationClaim.STUDENT_STATUS_PROHIBITED,
            AuthorizationClaim.PERMANENT_REQUIRED,
            AuthorizationClaim.SPONSORSHIP_NOT_OFFERED,
            AuthorizationClaim.GENERIC_RESTRICTION,
        }
    ),
}

_STUDENT_TOKEN = re.compile(
    r"\bf-?1\b|\bstem\s+opt\b|\bopt\b(?!-)",
    re.IGNORECASE,
)

_STUDENT_PROHIBITION = re.compile(
    r"""
    (?:
        (?:f-?1|stem\s+opt|\bopt\b)
        .{0,50}
        (?:
            not\s+(?:permitted|eligible|allowed|accepted|open|authorized|considered)
            | ineligible | prohibited | barred
        )
        |
        (?:
            not\s+(?:permitted|eligible|allowed|accepted|open|authorized)
            | ineligible | prohibited | barred | cannot | unable\s+to
        )
        .{0,50}
        (?:f-?1|stem\s+opt|\bopt\b)
        |
        no\s+(?:f-?1|stem\s+opt|opt)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CITIZENSHIP = re.compile(
    r"""
    (?:
        (?:citizens?|permanent\s+residents?|green\s+card)
        .{0,40}
        (?:only|required|must)
        |
        (?:must\s+be|required)
        .{0,40}
        (?:citizen|permanent\s+resident)
        |
        citizenship
        .{0,20}
        required
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PERMANENT = re.compile(
    r"""
    (?:
        permanent(?:ly)?\s+unrestricted
        | unrestricted\s+(?:u\.?\s*s\.?\s+)?work\s+authorization
        | now\s+or\s+in\s+the\s+future
          \s+require(?:\s+visa)?\s+sponsorship
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_NOT_OFFERED = re.compile(
    r"""
    (?:
        (?:do\s+not|does\s+not|will\s+not|cannot|unable\s+to)
        \s+(?:provide|offer|sponsor)
        |
        no\s+(?:visa\s+)?sponsorship
        |
        no\s+h-?1-?b
        |
        h-?1-?b.{0,20}(?:not\s+available|unavailable)
        |
        sponsorship\s+is\s+not\s+available
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_OFFERED = re.compile(
    r"""
    (?:
        (?:will|does|can)\s+sponsor
        | sponsorship\s+is\s+available
        | visa\s+sponsorship\s+(?:is\s+)?(?:available|offered)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_GENERIC_RESTRICTION = re.compile(
    r"""
    (?:
        work\s+authorization
        .{0,40}
        (?:not\s+(?:permitted|eligible|allowed)|required|must)
        |
        (?:visa|sponsorship)
        .{0,40}
        (?:not\s+(?:permitted|eligible|available)|unavailable)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Broad topical detector for LLM prose. Clearance / government language is
# intentionally absent: those are not work-authorization claims.
_AUTH_DOMAIN = re.compile(
    r"""
    \b(
        sponsorship
        | sponsor(?:s|ed|ing)?
        | visas?
        | (?:work|employment|permanent|unrestricted)\s+authorization
        | authorized\s+to\s+work
        | authorization
        | f-?1
        | stem\s+opt
        | opt(?!-)
        | h-?1-?b
        | citizenship
        | citizens?
        | permanent\s+residents?
        | green\s+card
        | immigration(?:\s+status)?
        | work\s+permit
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _normalize(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return _WHITESPACE.sub(" ", text).strip()


def claims_in(text: str) -> frozenset[AuthorizationClaim]:
    """Authorization restriction/offer claims asserted by this snippet."""
    normalized = _normalize(text)

    if not normalized:
        return frozenset()

    claims: set[AuthorizationClaim] = set()
    has_student_token = bool(_STUDENT_TOKEN.search(normalized))
    has_not_offered = bool(_NOT_OFFERED.search(normalized))

    if _STUDENT_PROHIBITION.search(normalized) or (
        has_student_token and has_not_offered
    ):
        claims.add(AuthorizationClaim.STUDENT_STATUS_PROHIBITED)

    if _CITIZENSHIP.search(normalized):
        claims.add(AuthorizationClaim.CITIZENSHIP_REQUIRED)

    if _PERMANENT.search(normalized):
        claims.add(AuthorizationClaim.PERMANENT_REQUIRED)

    if has_not_offered and not has_student_token:
        claims.add(AuthorizationClaim.SPONSORSHIP_NOT_OFFERED)

    if _OFFERED.search(normalized):
        claims.add(AuthorizationClaim.SPONSORSHIP_OFFERED)

    if not claims and _GENERIC_RESTRICTION.search(normalized):
        if has_student_token:
            claims.add(AuthorizationClaim.STUDENT_STATUS_PROHIBITED)
        else:
            claims.add(AuthorizationClaim.GENERIC_RESTRICTION)

    return frozenset(claims)


def in_authorization_domain(text: str) -> bool:
    """True when LLM prose is about work authorization, visas, or sponsorship."""
    return bool(_AUTH_DOMAIN.search(_normalize(text)))


def is_supported(text: str, stance: SponsorshipStance) -> bool:
    if not in_authorization_domain(text):
        return True

    found = claims_in(text)
    if not found:
        found = frozenset({AuthorizationClaim.GENERIC_RESTRICTION})

    return found <= _ALLOWED[stance]


def filter_statements(
    statements: list[str],
    stance: SponsorshipStance,
) -> list[str]:
    return [item for item in statements if is_supported(item, stance)]


def filter_prose(text: str, stance: SponsorshipStance) -> str:
    kept_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_SPLIT.split(stripped)
            if sentence.strip()
        ]
        supported = [
            sentence for sentence in sentences if is_supported(sentence, stance)
        ]

        if supported:
            kept_lines.append(" ".join(supported))

    return "\n".join(kept_lines).strip()


def sanitize_analysis_prose(
    strengths: list[str],
    missing_requirements: list[str],
    reasoning: str,
    stance: SponsorshipStance,
) -> tuple[list[str], list[str], str]:
    """Drop LLM authorization claims that the deterministic stance cannot support."""
    return (
        filter_statements(strengths, stance),
        filter_statements(missing_requirements, stance),
        filter_prose(reasoning, stance),
    )
