import pytest
from conftest import make_profile, make_requirements

from job_search_agent.candidate import LocationPreferences, WorkAuthorization
from job_search_agent.config_models import (
    ExperienceFilterConfig,
    FilterConfig,
    ToggleFilterConfig,
)
from job_search_agent.filters import check_hard_filters
from job_search_agent.models import SponsorshipStance


def build_filters(**overrides) -> FilterConfig:
    defaults = {
        "experience": ExperienceFilterConfig(
            max_required_years=2,
            tolerance_years=2,
        ),
    }

    return FilterConfig(**{**defaults, **overrides})


def run_filters(requirements=None, profile=None, config=None) -> list[str]:
    return check_hard_filters(
        requirements or make_requirements(),
        profile or make_profile(),
        config or build_filters(),
    )


# --- experience -------------------------------------------------------------


@pytest.mark.parametrize("required_years", [0, 1, 2, 3, 4])
def test_experience_within_tolerance_passes(required_years):
    reasons = run_filters(
        make_requirements(minimum_years_experience=required_years),
    )

    assert reasons == []


@pytest.mark.parametrize("required_years", [5, 8, 10])
def test_experience_clearly_above_target_fails(required_years):
    reasons = run_filters(
        make_requirements(minimum_years_experience=required_years),
    )

    assert len(reasons) == 1
    assert "years of experience" in reasons[0]


def test_unstated_experience_does_not_filter():
    reasons = run_filters(
        make_requirements(minimum_years_experience=None),
    )

    assert reasons == []


def test_disabled_experience_filter_never_fires():
    config = build_filters(
        experience=ExperienceFilterConfig(
            enabled=False,
            max_required_years=2,
            tolerance_years=0,
        ),
    )

    reasons = run_filters(
        make_requirements(minimum_years_experience=20),
        config=config,
    )

    assert reasons == []


# --- work authorization -----------------------------------------------------


def needs_sponsorship_profile():
    return make_profile(
        work_authorization=WorkAuthorization(
            status="F-1 STEM OPT",
            requires_sponsorship=True,
            is_citizen_or_permanent_resident=False,
        ),
    )


def test_sponsorship_not_offered_fails_candidate_who_needs_it():
    reasons = run_filters(
        make_requirements(
            minimum_years_experience=1,
            sponsorship=SponsorshipStance.NOT_OFFERED,
        ),
        profile=needs_sponsorship_profile(),
    )

    assert len(reasons) == 1
    assert "sponsorship" in reasons[0]


def test_sponsorship_not_offered_passes_candidate_who_does_not_need_it():
    reasons = run_filters(
        make_requirements(
            minimum_years_experience=1,
            sponsorship=SponsorshipStance.NOT_OFFERED,
        ),
    )

    assert reasons == []


def test_unmentioned_sponsorship_does_not_filter():
    """Silence is the common case and must never disqualify a posting."""
    reasons = run_filters(
        make_requirements(
            minimum_years_experience=1,
            sponsorship=SponsorshipStance.NOT_MENTIONED,
        ),
        profile=needs_sponsorship_profile(),
    )

    assert reasons == []


def test_offered_sponsorship_does_not_filter():
    reasons = run_filters(
        make_requirements(
            minimum_years_experience=1,
            sponsorship=SponsorshipStance.OFFERED,
        ),
        profile=needs_sponsorship_profile(),
    )

    assert reasons == []


def opt_profile():
    """F-1 on OPT: authorized to work now, will need sponsorship later."""
    return make_profile(
        work_authorization=WorkAuthorization(
            status="F-1 student; eligible for OPT/STEM OPT",
            requires_sponsorship=False,
            is_citizen_or_permanent_resident=False,
            requires_future_sponsorship=True,
        ),
    )


@pytest.mark.parametrize(
    ("stance", "should_skip"),
    [
        (SponsorshipStance.NOT_MENTIONED, False),
        (SponsorshipStance.OFFERED, False),
        # Generic "no sponsorship" must not disqualify someone already
        # authorized to work; it is surfaced as a concern instead.
        (SponsorshipStance.NOT_OFFERED, False),
        (SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED, True),
        (SponsorshipStance.CITIZENSHIP_REQUIRED, True),
    ],
)
def test_authorization_policy_for_a_candidate_on_opt(stance, should_skip):
    reasons = run_filters(
        make_requirements(minimum_years_experience=1, sponsorship=stance),
        profile=opt_profile(),
    )

    assert bool(reasons) is should_skip


def test_permanent_authorization_required_passes_permanent_resident():
    reasons = run_filters(
        make_requirements(
            minimum_years_experience=1,
            sponsorship=SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED,
        ),
        profile=make_profile(
            work_authorization=WorkAuthorization(
                status="Green Card",
                requires_sponsorship=False,
                is_citizen_or_permanent_resident=True,
            ),
        ),
    )

    assert reasons == []


def test_permanent_authorization_reason_is_distinct_from_citizenship():
    permanent = run_filters(
        make_requirements(
            minimum_years_experience=1,
            sponsorship=SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED,
        ),
        profile=opt_profile(),
    )

    citizenship = run_filters(
        make_requirements(
            minimum_years_experience=1,
            sponsorship=SponsorshipStance.CITIZENSHIP_REQUIRED,
        ),
        profile=opt_profile(),
    )

    assert "permanent unrestricted" in permanent[0]
    assert "citizenship or permanent residency" in citizenship[0]


def test_citizenship_required_fails_non_resident():
    reasons = run_filters(
        make_requirements(
            minimum_years_experience=1,
            sponsorship=SponsorshipStance.CITIZENSHIP_REQUIRED,
        ),
        profile=needs_sponsorship_profile(),
    )

    assert len(reasons) == 1
    assert "citizenship" in reasons[0]


def test_citizenship_required_passes_permanent_resident():
    reasons = run_filters(
        make_requirements(
            minimum_years_experience=1,
            sponsorship=SponsorshipStance.CITIZENSHIP_REQUIRED,
        ),
        profile=make_profile(
            work_authorization=WorkAuthorization(
                status="Green Card",
                requires_sponsorship=False,
                is_citizen_or_permanent_resident=True,
            ),
        ),
    )

    assert reasons == []


def test_disabled_authorization_filter_never_fires():
    reasons = run_filters(
        make_requirements(
            minimum_years_experience=1,
            sponsorship=SponsorshipStance.NOT_OFFERED,
        ),
        profile=needs_sponsorship_profile(),
        config=build_filters(
            work_authorization=ToggleFilterConfig(enabled=False),
        ),
    )

    assert reasons == []


# --- location ---------------------------------------------------------------


def unavailable_in_texas():
    return make_profile(
        locations=LocationPreferences(
            unavailable=["Texas", "Houston"],
            work_modes=["remote", "hybrid"],
        ),
    )


def test_unavailable_location_fails():
    reasons = run_filters(
        make_requirements(
            minimum_years_experience=1,
            location="Houston, TX",
        ),
        profile=unavailable_in_texas(),
    )

    assert len(reasons) == 1
    assert "Houston" in reasons[0]


def test_matching_is_case_insensitive():
    reasons = run_filters(
        make_requirements(minimum_years_experience=1, location="houston, tx"),
        profile=unavailable_in_texas(),
    )

    assert len(reasons) == 1


def test_available_location_passes():
    reasons = run_filters(
        make_requirements(
            minimum_years_experience=1,
            location="Boston, MA",
        ),
        profile=unavailable_in_texas(),
    )

    assert reasons == []


def test_missing_location_does_not_filter():
    reasons = run_filters(
        make_requirements(minimum_years_experience=1, location=None),
        profile=unavailable_in_texas(),
    )

    assert reasons == []


def test_empty_unavailable_list_disables_location_filtering():
    reasons = run_filters(
        make_requirements(
            minimum_years_experience=1,
            location="Houston, TX",
        ),
    )

    assert reasons == []


def test_remote_role_in_unavailable_location_still_passes():
    """A remote posting nominally based in a blocked city is still workable."""
    reasons = run_filters(
        make_requirements(
            minimum_years_experience=1,
            location="Remote - Houston, TX",
        ),
        profile=unavailable_in_texas(),
    )

    assert reasons == []


# --- combined ---------------------------------------------------------------


def test_multiple_failures_are_all_reported():
    reasons = run_filters(
        make_requirements(
            minimum_years_experience=9,
            location="Houston, TX",
            sponsorship=SponsorshipStance.NOT_OFFERED,
        ),
        profile=make_profile(
            work_authorization=WorkAuthorization(
                status="F-1 STEM OPT",
                requires_sponsorship=True,
            ),
            locations=LocationPreferences(unavailable=["Houston"]),
        ),
    )

    assert len(reasons) == 3


def test_ambiguous_posting_passes_every_filter():
    """A posting with no years, no location and no sponsorship language."""
    reasons = run_filters(
        make_requirements(
            minimum_years_experience=None,
            location=None,
            required_degree=None,
        ),
        profile=needs_sponsorship_profile(),
    )

    assert reasons == []
