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
    CareerRelevanceEvidence,
    ComponentScores,
    JobAnalysis,
    JobEvaluation,
    JobRequirements,
    Recommendation,
    SkillInventory,
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

    supports_dedicated_skill_extraction = False

    def __init__(
        self,
        evaluation: JobEvaluation,
        skill_inventory: SkillInventory | None = None,
        repair_inventory: SkillInventory | None = None,
        skill_inventories: list[SkillInventory] | None = None,
    ):
        self.evaluation = evaluation
        self.prompts: list[str] = []
        self.skill_prompts: list[str] = []
        responses: list[SkillInventory] = []
        if skill_inventories is not None:
            responses = list(skill_inventories)
        else:
            if skill_inventory is not None:
                responses.append(skill_inventory)
            if repair_inventory is not None:
                responses.append(repair_inventory)
        self._skill_responses = responses
        self._skill_index = 0
        if responses:
            self.supports_dedicated_skill_extraction = True

    def evaluate_job(self, prompt: str) -> JobEvaluation:
        self.prompts.append(prompt)
        return self.evaluation

    def extract_skill_inventory(self, prompt: str) -> SkillInventory:
        self.skill_prompts.append(prompt)
        if self._skill_index >= len(self._skill_responses):
            return SkillInventory(units=[])
        inventory = self._skill_responses[self._skill_index]
        self._skill_index += 1
        return inventory


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
    skill_claims: list | None = None,
    skill_unit_decisions: list | None = None,
    career_alignment=None,
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
        skill_claims=skill_claims or [],
        skill_unit_decisions=skill_unit_decisions or [],
        career_alignment=career_alignment or CareerRelevanceEvidence(),
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
        "target_job_families": ["Example Analyst", "Data Analyst"],
        "education": [Degree(degree="Master's", field="Example Field")],
        "skills": Skills(
            programming=["Python", "R", "SQL"],
            statistics=["Regression", "Survival Analysis", "GLM"],
        ),
        "experience_level": "entry_level",
        "years_of_experience": 0,
        "work_authorization": WorkAuthorization(
            status="Synthetic permanent authorization",
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
