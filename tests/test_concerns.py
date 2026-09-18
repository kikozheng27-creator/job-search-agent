import pytest
from conftest import make_profile, make_requirements

from job_search_agent.candidate import WorkAuthorization
from job_search_agent.concerns import collect_concerns
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


def citizen_profile():
    return make_profile(
        work_authorization=WorkAuthorization(
            status="US Citizen",
            requires_sponsorship=False,
            is_citizen_or_permanent_resident=True,
        ),
    )


def test_non_sponsoring_employer_is_a_concern_for_future_sponsorship():
    """The counterpart to the hard filter: preserved, but flagged."""
    concerns = collect_concerns(
        make_requirements(sponsorship=SponsorshipStance.NOT_OFFERED),
        opt_profile(),
    )

    assert len(concerns) == 1
    assert "does not offer visa sponsorship" in concerns[0]
    assert "does not block you now" in concerns[0]


def test_no_concern_when_future_sponsorship_is_not_needed():
    concerns = collect_concerns(
        make_requirements(sponsorship=SponsorshipStance.NOT_OFFERED),
        citizen_profile(),
    )

    assert concerns == []


@pytest.mark.parametrize(
    "stance",
    [
        SponsorshipStance.NOT_MENTIONED,
        SponsorshipStance.OFFERED,
        SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED,
        SponsorshipStance.CITIZENSHIP_REQUIRED,
    ],
)
def test_other_stances_raise_no_sponsorship_concern(stance):
    assert collect_concerns(make_requirements(sponsorship=stance), opt_profile()) == []


def test_requires_future_sponsorship_defaults_to_false():
    authorization = WorkAuthorization(status="US Citizen", requires_sponsorship=False)

    assert authorization.requires_future_sponsorship is False
