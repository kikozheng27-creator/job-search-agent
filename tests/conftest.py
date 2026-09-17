import pytest

from job_search_agent.candidate import (
    CandidateProfile,
    Degree,
    Skills,
    WorkAuthorization,
)
from job_search_agent.config_models import (
    ExperienceFilterConfig,
    FilterConfig,
    ScoringConfig,
    ScoringWeights,
    Thresholds,
)
from job_search_agent.models import (
    ComponentScores,
    JobAnalysis,
    JobEvaluation,
    JobRequirements,
    Recommendation,
    SponsorshipStance,
)


DEFAULT_WEIGHTS = ScoringWeights(
    skills=0.35,
    education=0.20,
    experience=0.25,
    career_relevance=0.20,
)

DEFAULT_THRESHOLDS = Thresholds(
    strongly_apply=85,
    apply=70,
    maybe=55,
)

DEFAULT_SCORING = ScoringConfig(
    weights=DEFAULT_WEIGHTS,
    thresholds=DEFAULT_THRESHOLDS,
)

PERMISSIVE_FILTERS = FilterConfig(
    experience=ExperienceFilterConfig(
        enabled=True,
        max_required_years=10,
        tolerance_years=0,
    ),
)


class FakeAIClient:
    """Stand-in for AIClient so unit tests never reach the OpenAI API."""

    def __init__(self, evaluation: JobEvaluation) -> None:
        self.evaluation = evaluation
        self.prompts: list[str] = []

    def evaluate_job(self, prompt: str) -> JobEvaluation:
        self.prompts.append(prompt)
        return self.evaluation


def make_requirements(**overrides) -> JobRequirements:
    defaults = {
        "company": "Example Pharma",
        "job_title": "Biostatistician",
        "location": "Boston, MA",
        "minimum_years_experience": 5,
        "required_degree": "Master's",
        "required_skills": ["R"],
        "preferred_skills": ["SAS"],
    }

    merged = {**defaults, **overrides}

    # JobRequirements downgrades a sponsorship stance that has no quoted
    # sentence, so supply plausible language whenever a test sets a stance
    # without caring about that guard. Tests for the guard pass it explicitly.
    stance = merged.get("sponsorship")

    if (
        stance is not None
        and stance is not SponsorshipStance.NOT_MENTIONED
        and "sponsorship_language" not in merged
    ):
        merged["sponsorship_language"] = "Quoted from the posting."

    return JobRequirements(**merged)


def make_scores(**overrides) -> ComponentScores:
    defaults = {
        "skills": 90,
        "education": 100,
        "experience": 70,
        "career_relevance": 95,
    }

    return ComponentScores(**{**defaults, **overrides})


def make_evaluation(
    requirements: JobRequirements | None = None,
    scores: ComponentScores | None = None,
) -> JobEvaluation:
    return JobEvaluation(
        requirements=requirements or make_requirements(),
        scores=scores or make_scores(),
        strengths=[
            "Strong R skills",
            "Relevant statistics background",
        ],
        missing_requirements=["SAS"],
        reasoning="The candidate matches most core requirements.",
    )


def make_analysis(**overrides) -> JobAnalysis:
    defaults = {
        "requirements": make_requirements(),
        "scores": make_scores(),
        "overall_score": 88.0,
        "recommendation": Recommendation.STRONGLY_APPLY,
        "passes_hard_filters": True,
        "hard_filter_reasons": [],
        "strengths": ["Strong R skills"],
        "missing_requirements": ["SAS"],
        "reasoning": "The candidate matches most core requirements.",
    }

    return JobAnalysis(**{**defaults, **overrides})


def make_profile(**overrides) -> CandidateProfile:
    defaults = {
        "target_job_families": ["Biostatistician", "Data Analyst"],
        "education": [Degree(degree="Master's", field="Biostatistics")],
        "skills": Skills(
            programming=["Python", "R", "SQL"],
            statistics=["Regression", "Survival Analysis", "GLM"],
        ),
        "experience_level": "entry_level",
        "years_of_experience": 0,
        "work_authorization": WorkAuthorization(
            status="US Citizen",
            requires_sponsorship=False,
            is_citizen_or_permanent_resident=True,
        ),
    }

    return CandidateProfile(**{**defaults, **overrides})


@pytest.fixture
def profile() -> CandidateProfile:
    return make_profile()


@pytest.fixture
def evaluation() -> JobEvaluation:
    return make_evaluation()
