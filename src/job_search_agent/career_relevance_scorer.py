"""Deterministic career_relevance from structured alignment evidence.

The LLM may still emit scores.career_relevance. This module is the final
authority. It does not read skills, education, years, sponsorship, or location.
"""

from __future__ import annotations

from job_search_agent.candidate import CandidateProfile
from job_search_agent.models import (
    CareerRelevanceEvidence,
    PreferredIndustryRelation,
    TargetFamilyRelation,
)
from job_search_agent.sponsorship_classifier import quote_in_source


_FAMILY_SCORE = {
    TargetFamilyRelation.DIRECT: 100,
    TargetFamilyRelation.ADJACENT: 70,
    TargetFamilyRelation.UNCLEAR: 50,
    TargetFamilyRelation.UNRELATED: 0,
}

_INDUSTRY_SCORE = {
    PreferredIndustryRelation.MATCH: 100,
    PreferredIndustryRelation.MISMATCH: 0,
}

_QUOTED_FAMILY = {
    TargetFamilyRelation.DIRECT,
    TargetFamilyRelation.ADJACENT,
    TargetFamilyRelation.UNRELATED,
}

_QUOTED_INDUSTRY = {
    PreferredIndustryRelation.MATCH,
    PreferredIndustryRelation.MISMATCH,
}

_IGNORED_INDUSTRY = {
    PreferredIndustryRelation.UNKNOWN,
    PreferredIndustryRelation.NOT_APPLICABLE,
}


def validate_career_alignment(
    evidence: CareerRelevanceEvidence,
    profile: CandidateProfile,
    job_description: str,
) -> CareerRelevanceEvidence:
    """Drop non-abstaining relations whose quotes are not in the posting.

    An empty preferred-industry list is NOT_APPLICABLE even if the model
    named a match or mismatch. That absence does not penalize the score.
    """
    family = evidence.target_family_relation
    family_quote = evidence.target_family_evidence
    if family in _QUOTED_FAMILY and not quote_in_source(job_description, family_quote):
        family = TargetFamilyRelation.UNCLEAR
        family_quote = None

    if not profile.preferred_industries:
        industry = PreferredIndustryRelation.NOT_APPLICABLE
        industry_quote = None
    else:
        industry = evidence.preferred_industry_relation
        industry_quote = evidence.preferred_industry_evidence
        if industry is PreferredIndustryRelation.NOT_APPLICABLE:
            industry = PreferredIndustryRelation.UNKNOWN
            industry_quote = None
        elif industry in _QUOTED_INDUSTRY and not quote_in_source(
            job_description,
            industry_quote,
        ):
            industry = PreferredIndustryRelation.UNKNOWN
            industry_quote = None

    return CareerRelevanceEvidence(
        target_family_relation=family,
        target_family_evidence=family_quote,
        preferred_industry_relation=industry,
        preferred_industry_evidence=industry_quote,
    )


def score_career_relevance(evidence: CareerRelevanceEvidence) -> int:
    """Map validated relations to a 0–100 score. Industry is ignored when unknown."""
    family = _FAMILY_SCORE[evidence.target_family_relation]
    industry_relation = evidence.preferred_industry_relation

    if industry_relation in _IGNORED_INDUSTRY:
        raw = float(family)
    else:
        industry = _INDUSTRY_SCORE[industry_relation]
        raw = 0.80 * family + 0.20 * industry

    return int(min(100, max(0, round(raw))))
