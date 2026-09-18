"""Intended work-authorization policy for an OPT/STEM OPT candidate.

These tests pin Python behavior given an already-classified stance.
Sentence-level classification of posting language is covered in
test_sponsorship_classifier.py.
"""

import pytest
from conftest import make_profile, make_requirements

from job_search_agent.candidate import WorkAuthorization
from job_search_agent.concerns import collect_concerns
from job_search_agent.config_models import ExperienceFilterConfig, FilterConfig
from job_search_agent.filters import check_hard_filters
from job_search_agent.models import SponsorshipStance


def opt_profile():
    return make_profile(
        work_authorization=WorkAuthorization(
            status="F-1 student; eligible for OPT/STEM OPT",
            requires_sponsorship=False,
            is_citizen_or_permanent_resident=False,
            requires_future_sponsorship=True,
        ),
    )


def apply_policy(stance: SponsorshipStance) -> tuple[list[str], list[str]]:
    requirements = make_requirements(
        minimum_years_experience=1,
        sponsorship=stance,
    )
    profile = opt_profile()
    config = FilterConfig(
        experience=ExperienceFilterConfig(
            max_required_years=2,
            tolerance_years=2,
        ),
    )

    return (
        check_hard_filters(requirements, profile, config),
        collect_concerns(requirements, profile),
    )


@pytest.mark.parametrize(
    ("stance", "should_filter", "should_concern"),
    [
        (SponsorshipStance.NOT_MENTIONED, False, False),
        (SponsorshipStance.OFFERED, False, False),
        # Generic "no sponsorship" / "no H-1B" preserves the job and warns.
        (SponsorshipStance.NOT_OFFERED, False, True),
        # Explicit rejection of temporary/student status is a hard filter.
        (SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED, True, False),
        (SponsorshipStance.CITIZENSHIP_REQUIRED, True, False),
    ],
)
def test_opt_policy_matrix(stance, should_filter, should_concern):
    reasons, concerns = apply_policy(stance)

    assert bool(reasons) is should_filter
    assert bool(concerns) is should_concern
    assert not (reasons and concerns)
