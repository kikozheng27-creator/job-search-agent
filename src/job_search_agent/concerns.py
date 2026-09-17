from job_search_agent.candidate import CandidateProfile
from job_search_agent.models import JobRequirements, SponsorshipStance


def collect_concerns(
    requirements: JobRequirements,
    profile: CandidateProfile,
) -> list[str]:
    """Non-blocking warnings reported alongside the recommendation.

    Concerns never change the score, the hard filters, or the
    recommendation. They exist so an issue can be surfaced instead of either
    being silently dropped or being turned into a false-negative filter.
    """
    results = [
        check_future_sponsorship(requirements, profile),
    ]

    return [concern for concern in results if concern is not None]


def check_future_sponsorship(
    requirements: JobRequirements,
    profile: CandidateProfile,
) -> str | None:
    """Flag a non-sponsoring employer for a candidate who will need it later.

    This is the counterpart to the work-authorization hard filter: the
    candidate can take the job today, so it must not be filtered out, but the
    mismatch is worth knowing before applying.
    """
    authorization = profile.work_authorization

    if not authorization.requires_future_sponsorship:
        return None

    if requirements.sponsorship is not SponsorshipStance.NOT_OFFERED:
        return None

    return (
        "Posting states it does not offer visa sponsorship. This does not "
        "block you now, but it will matter once your current work "
        f"authorization ends (status: {authorization.status})."
    )
