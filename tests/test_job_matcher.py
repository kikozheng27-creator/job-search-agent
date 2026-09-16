from job_search_agent.job_matcher import JobMatcher
from job_search_agent.models import (
    CandidateProfile,
    Education,
    JobEvaluation,
    Recommendation,
    Skills,
)

class FakeAIClient:
    def evaluate_job(self, prompt: str) -> JobEvaluation:
        return JobEvaluation(
            job_title="Biostatistician",
            company="Example Pharma",
            skills_score=90,
            education_score=100,
            experience_score=70,
            career_relevance_score=95,
            strengths=[
                "Strong R skills",
                "Relevant statistics background",
            ],
            missing_requirements=[
                "SAS",
            ],
            reasoning="The candidate matches most core requirements.",
        )

def create_test_profile() -> CandidateProfile:
    return CandidateProfile(
        target_roles=[
            "Biostatistician",
            "Data Analyst",
        ],
        education=Education(
            degree="Master's",
            field="Biostatistics",
        ),
        skills=Skills(
            programming=[
                "Python",
                "R",
                "SQL",
            ],
            statistics=[
                "Regression",
                "Survival Analysis",
                "GLM",
            ],
        ),
        experience_level="entry_level",
    )

def test_job_matcher_calculates_correct_score():
    ai_client = FakeAIClient()

    weights = {
        "skills": 0.35,
        "education": 0.20,
        "experience": 0.25,
        "career_relevance": 0.20,
    }

    thresholds = {
        "strongly_apply": 85,
        "apply": 70,
        "maybe": 55,
    }

    matcher = JobMatcher(
        ai_client=ai_client,
        weights=weights,
        thresholds=thresholds,
    )

    profile = create_test_profile()

    result = matcher.match(
        profile=profile,
        job_description="Example job description",
    )

    assert result.overall_score == 88.0
    assert result.recommendation == Recommendation.STRONGLY_APPLY

def test_job_matcher_preserves_evaluation_details():
    matcher = JobMatcher(
        ai_client=FakeAIClient(),
        weights={
            "skills": 0.35,
            "education": 0.20,
            "experience": 0.25,
            "career_relevance": 0.20,
        },
        thresholds={
            "strongly_apply": 85,
            "apply": 70,
            "maybe": 55,
        },
    )

    result = matcher.match(
        profile=create_test_profile(),
        job_description="Example job description",
    )

    assert result.job_title == "Biostatistician"
    assert result.company == "Example Pharma"
    assert "SAS" in result.missing_requirements
    assert "Strong R skills" in result.strengths