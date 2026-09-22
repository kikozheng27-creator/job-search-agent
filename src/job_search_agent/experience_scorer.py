"""Deterministic experience_score from candidate years and the posting minimum.

The LLM may still emit scores.experience. This module is the final authority
for the experience component score. It does not implement the 5+ year hard
filter; that stays in filters.check_experience.
"""

from __future__ import annotations

from job_search_agent.candidate import CandidateProfile
from job_search_agent.models import JobRequirements


def score_experience(
    profile: CandidateProfile,
    requirements: JobRequirements,
) -> int:
    """Years-match score in [0, 100] from the two structured year fields only."""
    required = requirements.minimum_years_experience
    if required is None or required <= 0:
        return 100

    candidate = profile.years_of_experience
    if candidate < 0:
        candidate = 0.0

    if candidate >= required:
        return 100

    score = round(84 * candidate / required)
    return int(min(100, max(0, score)))
