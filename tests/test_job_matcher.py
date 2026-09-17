from conftest import (
    DEFAULT_SCORING,
    PERMISSIVE_FILTERS,
    FakeAIClient,
    make_evaluation,
    make_profile,
    make_requirements,
)

from job_search_agent.candidate import WorkAuthorization
from job_search_agent.config_models import (
    ExperienceFilterConfig,
    FilterConfig,
)
from job_search_agent.job_matcher import JobMatcher
from job_search_agent.models import Recommendation, SponsorshipStance


def build_matcher(
    evaluation=None,
    filter_config=None,
) -> JobMatcher:
    return JobMatcher(
        ai_client=FakeAIClient(evaluation or make_evaluation()),
        scoring_config=DEFAULT_SCORING,
        filter_config=filter_config or PERMISSIVE_FILTERS,
    )


def test_job_matcher_calculates_correct_score():
    result = build_matcher().match(
        profile=make_profile(),
        job_description="Example job description",
    )

    assert result.overall_score == 88.0
    assert result.recommendation == Recommendation.STRONGLY_APPLY


def test_job_matcher_preserves_evaluation_details():
    result = build_matcher().match(
        profile=make_profile(),
        job_description="Example job description",
    )

    assert result.job_title == "Biostatistician"
    assert result.company == "Example Pharma"
    assert "SAS" in result.missing_requirements
    assert "Strong R skills" in result.strengths


def test_job_matcher_skips_job_with_too_much_required_experience():
    filter_config = FilterConfig(
        experience=ExperienceFilterConfig(
            enabled=True,
            max_required_years=2,
            tolerance_years=0,
        ),
    )

    result = build_matcher(filter_config=filter_config).match(
        profile=make_profile(),
        job_description="Example",
    )

    assert result.passes_hard_filters is False
    assert result.recommendation == Recommendation.SKIP
    assert len(result.hard_filter_reasons) == 1


def test_job_matcher_records_source_url():
    result = build_matcher().match(
        profile=make_profile(),
        job_description="Example",
        source_url="https://example.com/jobs/1",
    )

    assert result.source_url == "https://example.com/jobs/1"


def test_job_matcher_sends_profile_and_posting_to_the_model():
    ai_client = FakeAIClient(make_evaluation())

    matcher = JobMatcher(
        ai_client=ai_client,
        scoring_config=DEFAULT_SCORING,
        filter_config=PERMISSIVE_FILTERS,
    )

    matcher.match(
        profile=make_profile(),
        job_description="Unique posting text",
    )

    prompt = ai_client.prompts[0]

    assert "Unique posting text" in prompt
    assert "Biostatistics" in prompt
    assert "Do not compute an overall score." in prompt


def test_concerns_are_reported_without_changing_the_verdict():
    """A non-sponsoring employer must not turn a good match into a SKIP."""
    evaluation = make_evaluation(
        requirements=make_requirements(
            sponsorship=SponsorshipStance.NOT_OFFERED,
        ),
    )

    profile = make_profile(
        work_authorization=WorkAuthorization(
            status="F-1 student; eligible for OPT/STEM OPT",
            requires_sponsorship=False,
            is_citizen_or_permanent_resident=False,
            requires_future_sponsorship=True,
        ),
    )

    result = build_matcher(evaluation=evaluation).match(
        profile=profile,
        job_description="Example",
    )

    assert result.passes_hard_filters is True
    assert result.recommendation == Recommendation.STRONGLY_APPLY
    assert result.overall_score == 88.0
    assert len(result.concerns) == 1
    assert "does not offer visa sponsorship" in result.concerns[0]


def test_concerns_are_empty_when_nothing_applies():
    result = build_matcher().match(
        profile=make_profile(),
        job_description="Example",
    )

    assert result.concerns == []


def test_permanent_authorization_requirement_forces_skip():
    evaluation = make_evaluation(
        requirements=make_requirements(
            sponsorship=SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED,
        ),
    )

    profile = make_profile(
        work_authorization=WorkAuthorization(
            status="F-1 student; eligible for OPT/STEM OPT",
            requires_sponsorship=False,
            is_citizen_or_permanent_resident=False,
            requires_future_sponsorship=True,
        ),
    )

    result = build_matcher(evaluation=evaluation).match(
        profile=profile,
        job_description="Example",
    )

    assert result.passes_hard_filters is False
    assert result.recommendation == Recommendation.SKIP
    assert result.concerns == []


def test_job_matcher_keeps_extracted_requirements():
    evaluation = make_evaluation(
        requirements=make_requirements(
            location="Remote",
            preferred_skills=["SAS", "CDISC"],
        ),
    )

    result = build_matcher(evaluation=evaluation).match(
        profile=make_profile(),
        job_description="Example",
    )

    assert result.location == "Remote"
    assert result.requirements.preferred_skills == ["SAS", "CDISC"]
