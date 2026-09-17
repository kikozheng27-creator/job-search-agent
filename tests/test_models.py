import pytest
from conftest import make_analysis, make_requirements, make_scores
from pydantic import ValidationError

from job_search_agent.models import (
    EmploymentType,
    JobEvaluation,
    JobRequirements,
    Recommendation,
    SeniorityLevel,
    SponsorshipStance,
)


def test_valid_job_analysis():
    job = make_analysis(
        overall_score=87.0,
        recommendation=Recommendation.APPLY,
    )

    assert job.job_title == "Biostatistician"
    assert job.company == "Example Pharma"
    assert job.scores.skills == 90
    assert job.scores.education == 100
    assert job.scores.experience == 70
    assert job.scores.career_relevance == 95
    assert job.overall_score == 87.0
    assert job.recommendation == Recommendation.APPLY
    assert "SAS" in job.missing_requirements


# The previous versions of the three tests below omitted most required fields,
# so they passed on the missing fields rather than on the constraint named in
# the test. Each now builds a complete analysis and varies only one field.


def test_overall_score_cannot_exceed_100():
    with pytest.raises(ValidationError) as error:
        make_analysis(overall_score=150)

    assert "overall_score" in str(error.value)


def test_overall_score_cannot_be_negative():
    with pytest.raises(ValidationError) as error:
        make_analysis(overall_score=-10)

    assert "overall_score" in str(error.value)


def test_invalid_recommendation_is_rejected():
    with pytest.raises(ValidationError) as error:
        make_analysis(recommendation="DEFINITELY_DO_IT")

    assert "recommendation" in str(error.value)


def test_component_score_cannot_exceed_100():
    with pytest.raises(ValidationError) as error:
        make_scores(skills=101)

    assert "skills" in str(error.value)


def test_requirements_default_to_unknown_rather_than_guessing():
    requirements = make_requirements()

    assert requirements.employment_type == EmploymentType.UNKNOWN
    assert requirements.seniority_level == SeniorityLevel.UNKNOWN
    assert requirements.sponsorship == SponsorshipStance.NOT_MENTIONED
    assert requirements.salary_range is None


def test_negative_required_years_is_rejected():
    with pytest.raises(ValidationError):
        make_requirements(minimum_years_experience=-1)


# A live run returned sponsorship="not_offered" for a posting containing no
# sponsorship wording, with sponsorship_language null. Acting on that would
# hard-filter a job on invented evidence.


@pytest.mark.parametrize(
    "stance",
    [
        SponsorshipStance.NOT_OFFERED,
        SponsorshipStance.OFFERED,
        SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED,
        SponsorshipStance.CITIZENSHIP_REQUIRED,
    ],
)
def test_sponsorship_stance_without_quoted_language_is_downgraded(stance):
    requirements = JobRequirements(
        company="C",
        job_title="T",
        sponsorship=stance,
        sponsorship_language=None,
    )

    assert requirements.sponsorship is SponsorshipStance.NOT_MENTIONED


@pytest.mark.parametrize("language", ["", "   ", "\n"])
def test_blank_sponsorship_language_is_also_downgraded(language):
    requirements = JobRequirements(
        company="C",
        job_title="T",
        sponsorship=SponsorshipStance.CITIZENSHIP_REQUIRED,
        sponsorship_language=language,
    )

    assert requirements.sponsorship is SponsorshipStance.NOT_MENTIONED


@pytest.mark.parametrize(
    "stance",
    [
        SponsorshipStance.NOT_OFFERED,
        SponsorshipStance.OFFERED,
        SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED,
        SponsorshipStance.CITIZENSHIP_REQUIRED,
    ],
)
def test_quoted_sponsorship_language_preserves_the_stance(stance):
    requirements = JobRequirements(
        company="C",
        job_title="T",
        sponsorship=stance,
        sponsorship_language="We are unable to provide visa sponsorship.",
    )

    assert requirements.sponsorship is stance


def test_not_mentioned_needs_no_language():
    requirements = JobRequirements(company="C", job_title="T")

    assert requirements.sponsorship is SponsorshipStance.NOT_MENTIONED
    assert requirements.sponsorship_language is None


def test_job_evaluation_has_no_overall_score_or_recommendation():
    """The LLM contract must not let the model decide these."""
    fields = set(JobEvaluation.model_fields)

    assert "overall_score" not in fields
    assert "recommendation" not in fields
