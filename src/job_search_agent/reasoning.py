"""Deterministic reasoning when sanitized LLM prose is empty.

Uses only already-validated JobAnalysis fields. It does not infer new
requirements and does not mention work authorization unless a hard filter
already recorded that fact.
"""

from job_search_agent.models import (
    ComponentScores,
    JobRequirements,
    Recommendation,
    SponsorshipStance,
)

_SKILLS_STRONG = 70
_SKILLS_SOME = 40


def build_fallback_reasoning(
    requirements: JobRequirements,
    scores: ComponentScores,
    recommendation: Recommendation,
    hard_filter_reasons: list[str],
    missing_requirements: list[str],
) -> str:
    """Short explanation from trusted scores, filters, and extracted years."""
    sentences = [_skills_sentence(scores.skills)]

    experience_reason = _experience_filter_reason(hard_filter_reasons)
    years = requirements.minimum_years_experience

    if experience_reason:
        if years is not None:
            sentences.append(
                f"The role requires {years}+ years of experience, which "
                "exceeds the target maximum and triggers a hard-filter SKIP."
            )
        else:
            sentences.append(experience_reason)
    elif years is not None:
        sentences.append(
            f"The posting lists a minimum of {years} years of experience."
        )

    other_reasons = [
        reason
        for reason in hard_filter_reasons
        if reason is not experience_reason
    ]
    if other_reasons:
        sentences.extend(other_reasons)
        if not experience_reason:
            sentences.append("The recommendation is SKIP.")
    elif not hard_filter_reasons:
        sentences.append(f"The recommendation is {recommendation.value}.")
        if missing_requirements:
            sentences.append(
                f"Remaining gaps include {missing_requirements[0]}."
            )

    text = " ".join(sentence.strip() for sentence in sentences if sentence.strip())

    if (
        requirements.sponsorship is SponsorshipStance.NOT_MENTIONED
        and _mentions_authorization(text)
    ):
        text = " ".join(
            sentence
            for sentence in sentences
            if sentence.strip() and not _mentions_authorization(sentence)
        )

    return text.strip()


def _skills_sentence(skills_score: int) -> str:
    if skills_score >= _SKILLS_STRONG:
        return "The candidate has relevant skills for this role."

    if skills_score >= _SKILLS_SOME:
        return "The candidate has some relevant skills for this role."

    return "The candidate has limited overlap with the required skills."


def _experience_filter_reason(hard_filter_reasons: list[str]) -> str | None:
    for reason in hard_filter_reasons:
        if "years of experience" in reason:
            return reason

    return None


def _mentions_authorization(text: str) -> bool:
    lowered = text.casefold()
    return any(
        token in lowered
        for token in (
            "sponsorship",
            "sponsor",
            "visa",
            "authorization",
            "f-1",
            "opt",
            "citizen",
            "green card",
        )
    )
