import pytest
from pydantic import ValidationError

from job_search_agent.models import JobAnalysis, Recommendation

def test_valid_job_analysis():
    job = JobAnalysis(
    job_title="Biostatistician",
    company="Example Pharma",
    skills_score=90,
    education_score=95,
    experience_score=75,
    career_relevance_score=90,
    overall_score=87.0,
    recommendation=Recommendation.APPLY,
    passes_hard_filters=True,
    hard_filter_reasons=[],
    strengths=[
        "Strong R skills",
        "Relevant statistical background",
    ],
    missing_requirements=[
        "SAS",
    ],
    reasoning="The candidate matches most core requirements.",
    )   

    assert job.job_title == "Biostatistician"
    assert job.company == "Example Pharma"
    assert job.skills_score == 90
    assert job.education_score == 95
    assert job.experience_score == 75
    assert job.career_relevance_score == 90
    assert job.overall_score == 87.0
    assert job.recommendation == Recommendation.APPLY
    assert "SAS" in job.missing_requirements

def test_score_cannot_exceed_100():
    with pytest.raises(ValidationError):
        JobAnalysis(
            job_title="Biostatistician",
            company="Example Pharma",
            overall_score=150,
            recommendation=Recommendation.APPLY,
        )

def test_score_cannot_be_negative():
    with pytest.raises(ValidationError):
        JobAnalysis(
            job_title="Biostatistician",
            company="Example Pharma",
            overall_score=-10,
            recommendation=Recommendation.APPLY,
        )

def test_invalid_recommendation_is_rejected():
    with pytest.raises(ValidationError):
        JobAnalysis(
            job_title="Biostatistician",
            company="Example Pharma",
            overall_score=80,
            recommendation="DEFINITELY_DO_IT",
        )
