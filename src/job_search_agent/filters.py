from job_search_agent.candidate import CandidateProfile, WorkMode
from job_search_agent.config_models import (
    ExperienceFilterConfig,
    FilterConfig,
    ToggleFilterConfig,
)
from job_search_agent.models import JobRequirements, SponsorshipStance


def check_hard_filters(
    requirements: JobRequirements,
    profile: CandidateProfile,
    config: FilterConfig,
) -> list[str]:
    """Return one reason per hard filter the posting fails.

    An empty list means the posting passes. Every rule abstains when the
    posting is ambiguous, so silence in a job description never disqualifies.
    """
    results = [
        check_experience(requirements, config.experience),
        check_work_authorization(
            requirements,
            profile,
            config.work_authorization,
        ),
        check_location(requirements, profile, config.location),
    ]

    return [reason for reason in results if reason is not None]


def check_experience(
    requirements: JobRequirements,
    config: ExperienceFilterConfig,
) -> str | None:
    """Reject only when required experience is clearly above the target level.

    The tolerance exists so a posting asking for slightly more than the target
    still reaches weighted scoring instead of being filtered out.
    """
    if not config.enabled:
        return None

    required_years = requirements.minimum_years_experience

    if required_years is None:
        return None

    if required_years <= config.max_required_years + config.tolerance_years:
        return None

    return (
        f"Requires {required_years}+ years of experience, which is more than "
        f"{config.tolerance_years} year(s) above the target maximum of "
        f"{config.max_required_years}."
    )


def check_work_authorization(
    requirements: JobRequirements,
    profile: CandidateProfile,
    config: ToggleFilterConfig,
) -> str | None:
    """Reject only when the posting rules out the candidate's current status.

    A generic "we do not sponsor" does not disqualify a candidate who is
    already authorized to work; that case is reported as a concern instead.
    Most postings say nothing at all, which extraction reports as
    NOT_MENTIONED and this rule treats as a pass.
    """
    if not config.enabled:
        return None

    authorization = profile.work_authorization
    stance = requirements.sponsorship

    if (
        stance is SponsorshipStance.NOT_OFFERED
        and authorization.requires_sponsorship
    ):
        return (
            "Posting states it does not offer visa sponsorship, and the "
            "candidate cannot begin work without it "
            f"(status: {authorization.status})."
        )

    if (
        stance is SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED
        and not authorization.is_citizen_or_permanent_resident
    ):
        return (
            "Posting requires permanent unrestricted work authorization, "
            "which the candidate does not hold "
            f"(status: {authorization.status})."
        )

    if (
        stance is SponsorshipStance.CITIZENSHIP_REQUIRED
        and not authorization.is_citizen_or_permanent_resident
    ):
        return (
            "Posting requires citizenship or permanent residency, which the "
            f"candidate does not hold (status: {authorization.status})."
        )

    return None


def check_location(
    requirements: JobRequirements,
    profile: CandidateProfile,
    config: ToggleFilterConfig,
) -> str | None:
    """Reject only locations the candidate has explicitly ruled out.

    Abstains when the posting omits a location or when the candidate lists no
    unavailable locations, which keeps this filter opt-in.
    """
    if not config.enabled:
        return None

    location = requirements.location
    unavailable = profile.locations.unavailable

    if not location or not unavailable:
        return None

    normalized = location.casefold()

    if "remote" in normalized and WorkMode.REMOTE in profile.locations.work_modes:
        return None

    for blocked in unavailable:
        if blocked.casefold() in normalized:
            return (
                f"Location '{location}' matches the candidate's unavailable "
                f"location '{blocked}'."
            )

    return None
