"""Deterministic work-authorization classification from quoted sentences.

The LLM may extract sponsorship_language, but this module is the final
authority for sponsorship stance. Patterns are sentence-level; isolated
keywords such as "OPT", "permanent", or "sponsorship" never classify.
"""

from __future__ import annotations

import re

from job_search_agent.models import JobRequirements, SponsorshipStance


_WHITESPACE = re.compile(r"\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# "U.S. citizenship required" must not split after "U.S."
_ABBREVIATION = re.compile(r"\b(u\.s\.a\.|u\.s\.)", re.IGNORECASE)

# Strongest match wins when a posting contains more than one authorization
# sentence, so citizen/PR-only is never overridden by generic no-sponsorship.
_PRECEDENCE = {
    SponsorshipStance.CITIZENSHIP_REQUIRED: 3,
    SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED: 2,
    SponsorshipStance.NOT_OFFERED: 1,
    SponsorshipStance.NOT_MENTIONED: 0,
    SponsorshipStance.OFFERED: 0,
}


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.VERBOSE)


def _normalize(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return _WHITESPACE.sub(" ", text).strip().casefold()


def _mask_abbreviations(text: str) -> str:
    return _ABBREVIATION.sub(lambda match: match.group(0).replace(".", ""), text)


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        for part in _SENTENCE_SPLIT.split(_mask_abbreviations(stripped)):
            normalized = _normalize(part)
            if normalized:
                sentences.append(normalized)

    return sentences


# --- citizenship / permanent residents only --------------------------------

_CITIZENSHIP = [
    _compile(
        r"""
        \b(?:u\.?\s*s\.?|united\s+states)\s+citizens?
        \s+or\s+permanent\s+residents?
        (?:\s+only)?
        """
    ),
    _compile(
        r"""
        \b(?:u\.?\s*s\.?|united\s+states)\s+citizens?
        \s+and\s+permanent\s+residents?
        \s+only\b
        """
    ),
    _compile(r"\b(?:u\.?\s*s\.?|united\s+states)\s+citizens?\s+only\b"),
    _compile(r"\bpermanent\s+residents?\s+only\b"),
    _compile(
        r"""
        \bmust\s+be\s+(?:a\s+)?(?:u\.?\s*s\.?\s+)?citizen
        \s+or\s+(?:a\s+)?permanent\s+resident\b
        """
    ),
    _compile(
        r"""
        \b(?:u\.?\s*s\.?\s+)?citizenship
        \s+or\s+permanent\s+residency
        \s+(?:is\s+)?required\b
        """
    ),
    _compile(
        r"""
        \b(?:u\.?\s*s\.?|united\s+states)\s+citizenship
        \s+(?:is\s+)?required\b
        """
    ),
    _compile(
        r"""
        \bonly\s+to\s+(?:u\.?\s*s\.?|united\s+states)\s+citizens?\b
        """
    ),
]


# --- temporary / student status excluded -----------------------------------

_F1 = r"f-?1(?:\s+visa)?"
_OPT = r"(?:stem\s+)?opt"

_PERMANENT = [
    _compile(
        rf"""
        (?:
            (?:not\s+open\s+to|no|not\s+accepting)\s+(?:any\s+)?{_F1}
            |
            (?:cannot|can\s+not|will\s+not|unable\s+to)
            \s+(?:consider|accept|hire)\s+{_F1}
            |
            {_F1}\s+(?:candidates?|holders?|students?)
            \s+(?:are\s+)?(?:not\s+eligible|ineligible|not\s+accepted|not\s+considered)
        )
        """
    ),
    _compile(
        rf"""
        (?:
            (?:not\s+open\s+to|no|not\s+accepting)\s+(?:any\s+)?{_OPT}
            |
            (?:cannot|can\s+not|will\s+not|unable\s+to)
            \s+(?:consider|accept|hire)\s+{_OPT}
            |
            {_OPT}\s+(?:candidates?|holders?|students?)
            \s+(?:are\s+)?(?:not\s+eligible|ineligible|not\s+accepted|not\s+considered)
        )
        """
    ),
    # "permanent residents" must not match. The negative lookahead blocks
    # that phrasing; "work authorization" is required after "permanent".
    _compile(
        r"""
        \bpermanent(?!\s+residen)
        .{0,40}
        \bwork\s+authorization\b
        """
    ),
    _compile(r"\bpermanent(?:ly)?\s+unrestricted\b"),
    _compile(r"\bunrestricted\s+(?:u\.?\s*s\.?\s+)?work\s+authorization\b"),
    _compile(
        r"""
        \b(?:must\s+)?not\s+now\s+or\s+in\s+the\s+future
        \s+require(?:\s+visa)?\s+sponsorship\b
        """
    ),
    _compile(
        r"""
        \bnow\s+or\s+in\s+the\s+future
        \s+require(?:\s+visa)?\s+sponsorship\b
        """
    ),
]


# --- declines future sponsorship, does not reject current status -----------

_NOT_OFFERED = [
    _compile(
        r"""
        (?:
            (?:do\s+not|does\s+not|will\s+not|cannot|can\s+not|unable\s+to)
            \s+(?:provide|offer)
            |
            no
        )
        .{0,20}
        \bh-?1-?b\b
        """
    ),
    _compile(r"\bh-?1-?b\s+sponsorship\s+(?:is\s+)?(?:not\s+available|unavailable)\b"),
    _compile(
        r"""
        (?:do\s+not|does\s+not|will\s+not|cannot|can\s+not|unable\s+to)
        \s+(?:provide|offer)
        \s+(?:any\s+)?(?:visa\s+)?sponsorship\b
        """
    ),
    _compile(r"\bno\s+visa\s+sponsorship\b"),
    _compile(r"\b(?:visa\s+)?sponsorship\s+is\s+not\s+available\b"),
    _compile(
        r"""
        (?:do\s+not|does\s+not|will\s+not)
        \s+sponsor(?:\s+(?:visas?|candidates?|employees?))?\b
        """
    ),
]


def _matches(patterns: list[re.Pattern[str]], sentence: str) -> bool:
    return any(pattern.search(sentence) for pattern in patterns)


def classify_sentence(sentence: str) -> SponsorshipStance | None:
    """Return a stance for one sentence, or None if it is not evidence."""
    text = _normalize(sentence)

    if not text:
        return None

    if _matches(_CITIZENSHIP, text):
        return SponsorshipStance.CITIZENSHIP_REQUIRED

    if _matches(_PERMANENT, text):
        return SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED

    if _matches(_NOT_OFFERED, text):
        return SponsorshipStance.NOT_OFFERED

    return None


def _quote_in_source(job_description: str, quote: str | None) -> bool:
    """True only when the quote is a contiguous substring of the posting."""
    if not quote or not quote.strip():
        return False

    quoted = _normalize(quote)
    return bool(quoted) and quoted in _normalize(job_description)


def quote_in_source(job_description: str, quote: str | None) -> bool:
    """Public name for evaluation diagnostics. Same rule as classification."""
    return _quote_in_source(job_description, quote)


def classify_sponsorship(
    job_description: str,
    sponsorship_language: str | None = None,
) -> SponsorshipStance:
    """Classify a posting from its sentences. Silence is not_mentioned."""
    sentences = split_sentences(job_description)

    # An LLM quote is evidence only when it actually appears in the posting.
    # Invented phrases — including prompt examples — must not classify.
    if _quote_in_source(job_description, sponsorship_language):
        quoted = _normalize(sponsorship_language)
        if quoted not in sentences:
            sentences.append(quoted)

    strongest = SponsorshipStance.NOT_MENTIONED

    for sentence in sentences:
        stance = classify_sentence(sentence)

        if stance is None:
            continue

        if _PRECEDENCE[stance] > _PRECEDENCE[strongest]:
            strongest = stance

    return strongest


def _supporting_sentence(
    job_description: str,
    stance: SponsorshipStance,
) -> str | None:
    """Return a source sentence that justifies a non-silent stance."""
    for line in job_description.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if classify_sentence(stripped) is stance:
            return stripped

        for part in _SENTENCE_SPLIT.split(_mask_abbreviations(stripped)):
            sentence = part.strip()
            if sentence and classify_sentence(sentence) is stance:
                return sentence

    return None


def apply_sponsorship_classification(
    requirements: JobRequirements,
    job_description: str,
) -> JobRequirements:
    """Overwrite LLM stance. Does not change scores or other extracted fields."""
    if not _quote_in_source(job_description, requirements.sponsorship_language):
        requirements.sponsorship_language = None

    stance = classify_sponsorship(
        job_description,
        requirements.sponsorship_language,
    )
    requirements.sponsorship = stance

    # JobAnalysis revalidates nested requirements. The unquoted-stance guard
    # would otherwise treat a Python-backed stance with no LLM quote as
    # invented and reset it to not_mentioned.
    if (
        stance is not SponsorshipStance.NOT_MENTIONED
        and not (
            requirements.sponsorship_language
            and requirements.sponsorship_language.strip()
        )
    ):
        evidence = _supporting_sentence(job_description, stance)
        if evidence:
            requirements.sponsorship_language = evidence

    return requirements
