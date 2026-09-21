"""Split a posting into numbered, deterministic evidence units.

The LLM must interpret these units. It does not invent source chunks.
Same exact job-description text always yields the same unit ids and text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WHITESPACE = re.compile(r"\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ABBREVIATION = re.compile(
    r"\b(u\.s\.a\.|u\.s\.|e\.g\.|i\.e\.|etc\.|vs\.|mr\.|mrs\.|dr\.|prof\.)",
    re.IGNORECASE,
)
_BULLET_PREFIX = re.compile(r"^([\-\*\u2022\u2013]|[0-9]+[\.)])\s+")
_PUNCTUATION_ONLY = re.compile(r"^[.!?]+$")
_TITLE_LIKE = re.compile(r"^[A-Z*].{0,59}$")

# Units the model must classify. Broad on purpose: omission of a
# skill-bearing unit should be visible, but chrome/nav should not.
_SKILL_CANDIDATE = re.compile(
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
        | preferred
        | desired
        | optional
        | bonus
        | nice\s+to\s+have
        | a\s+plus
        | skills?
        | technolog(?:y|ies)
        | software
        | programming
        | languages?
        | tools?
        | platforms?
        | libraries
        | frameworks?
        | using
        | uses
        | scripts?
        | code
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class EvidenceUnit:
    id: str
    text: str


def _collapse(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return _WHITESPACE.sub(" ", text).strip()


def _protect_abbreviations(text: str) -> str:
    return _ABBREVIATION.sub(lambda match: match.group(0).replace(".", "\0"), text)


def _restore_abbreviations(text: str) -> str:
    return text.replace("\0", ".")


def split_sentences(line: str) -> list[str]:
    """Split one line into sentences without breaking U.S. abbreviations."""
    protected = _protect_abbreviations(line)
    parts = [
        _restore_abbreviations(part).strip()
        for part in _SENTENCE_SPLIT.split(protected)
    ]
    return [part for part in parts if part]


def _is_bullet(line: str) -> bool:
    return bool(_BULLET_PREFIX.match(line))


def _looks_like_heading(line: str) -> bool:
    """A short line ending in a colon is treated as a section heading."""
    if len(line) > 80:
        return False
    return line.endswith(":")


def _is_complete(line: str) -> bool:
    return line.endswith((".", "!", "?", ":"))


def _is_title_like(line: str) -> bool:
    if _looks_like_heading(line) or _is_bullet(line):
        return False
    if line.endswith((".", "!", "?")):
        return False
    return bool(_TITLE_LIKE.match(line))


def _ends_with_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?"))


def _with_heading(heading: str | None, text: str) -> str:
    if not heading or text == heading:
        return text
    if text.startswith(f"{heading} "):
        return text
    return f"{heading} {text}"


def _merge_wrapped_lines(job_description: str) -> list[tuple[str | None, str]]:
    """Rebuild paragraphs from hard-wrapped lines; blank lines break sections."""
    blocks: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        text = " ".join(buffer)
        buffer = []
        blocks.append((current_heading, text))

    for raw_line in job_description.splitlines():
        line = _collapse(raw_line)
        if not line:
            flush()
            current_heading = None
            continue

        if _looks_like_heading(line):
            flush()
            current_heading = line
            blocks.append((None, line))
            continue

        if _is_bullet(line):
            flush()
            blocks.append((current_heading, line))
            continue

        if _PUNCTUATION_ONLY.match(line):
            if buffer:
                buffer[-1] = buffer[-1].rstrip() + line
            elif blocks:
                heading, text = blocks[-1]
                blocks[-1] = (heading, text.rstrip() + line)
            continue

        if buffer and _is_complete(buffer[-1]):
            flush()

        if (
            current_heading
            and not buffer
            and _is_title_like(line)
            and blocks
            and _ends_with_sentence(blocks[-1][1])
        ):
            current_heading = None

        buffer.append(line)

    flush()
    return blocks


def build_evidence_units(job_description: str) -> list[EvidenceUnit]:
    """Convert cleaned JD text into stable, numbered evidence units.

    Prefers existing structure: headings, bullets, reconstructed paragraphs,
    then sentences. Section headings are prefixed onto following content so
    required vs preferred wording is not lost when a bullet is only a name.
    Blank lines and new title-like lines close a heading's scope.
    """
    chunks: list[str] = []

    for heading, text in _merge_wrapped_lines(job_description):
        if _looks_like_heading(text) and heading is None:
            chunks.append(text)
            continue

        prefixed_ready = _with_heading(heading, text)
        if _is_bullet(text):
            chunks.append(prefixed_ready)
            continue

        body = text
        sentences = split_sentences(body)
        if len(sentences) > 1:
            for sentence in sentences:
                chunks.append(_with_heading(heading, sentence))
        else:
            chunks.append(prefixed_ready)

    return [
        EvidenceUnit(id=f"u{index:03d}", text=text)
        for index, text in enumerate(chunks, start=1)
    ]


def is_skill_candidate_unit(unit: EvidenceUnit) -> bool:
    """True when the unit may contain a candidate skill requirement."""
    text = unit.text.strip()
    if len(text) < 8:
        return False
    if _looks_like_heading(text):
        return False
    return bool(_SKILL_CANDIDATE.search(text))


def skill_candidate_units(units: list[EvidenceUnit]) -> list[EvidenceUnit]:
    return [unit for unit in units if is_skill_candidate_unit(unit)]


def format_evidence_units(
    units: list[EvidenceUnit],
    *,
    candidates_only: bool = True,
) -> str:
    selected = skill_candidate_units(units) if candidates_only else units
    return "\n".join(f"{unit.id}: {unit.text}" for unit in selected)


def units_by_id(units: list[EvidenceUnit]) -> dict[str, EvidenceUnit]:
    return {unit.id: unit for unit in units}
