from job_search_agent.models import JobRequirements


def check_hard_filters(
    requirements: JobRequirements,
    config: dict,
) -> list[str]:
    reasons = []

    max_years = config["max_required_experience_years"]

    required_years = requirements.minimum_years_experience

    if (
        required_years is not None
        and required_years > max_years
    ):
        reasons.append(
            f"Job requires {required_years}+ years of experience, "
            f"which exceeds the configured maximum of {max_years} years."
        )

    return reasons