from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Recommendation(str, Enum):
    STRONGLY_APPLY = "STRONGLY_APPLY"
    APPLY = "APPLY"
    MAYBE = "MAYBE"
    SKIP = "SKIP"


class EmploymentType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    TEMPORARY = "temporary"
    UNKNOWN = "unknown"


class SeniorityLevel(str, Enum):
    INTERNSHIP = "internship"
    ENTRY_LEVEL = "entry_level"
    MID_LEVEL = "mid_level"
    SENIOR = "senior"
    LEAD = "lead"
    UNKNOWN = "unknown"


class SponsorshipStance(str, Enum):
    """What the posting says about work authorization.

    The distinction that matters is whether the posting rejects the
    candidate's authorization *now* or only declines to sponsor *later*:

    - NOT_OFFERED covers generic "we do not sponsor" and "no H-1B" wording.
      It says nothing about whether the candidate may work today, so it is a
      future concern rather than a disqualification.
    - PERMANENT_AUTHORIZATION_REQUIRED and CITIZENSHIP_REQUIRED both reject a
      candidate whose current authorization is temporary.
    - NOT_MENTIONED is the deliberate abstain value: most postings say
      nothing, and silence must never trigger a hard filter.
    """

    OFFERED = "offered"
    NOT_OFFERED = "not_offered"
    PERMANENT_AUTHORIZATION_REQUIRED = "permanent_authorization_required"
    CITIZENSHIP_REQUIRED = "citizenship_required"
    NOT_MENTIONED = "not_mentioned"


class SalaryRange(BaseModel):
    minimum: float | None = None
    maximum: float | None = None
    currency: str | None = None
    period: str | None = None


class JobRequirements(BaseModel):
    """Facts extracted from the posting. Extraction only, no judgement."""

    company: str
    job_title: str
    location: str | None = None
    employment_type: EmploymentType = EmploymentType.UNKNOWN

    minimum_years_experience: int | None = Field(default=None, ge=0)
    required_degree: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)

    sponsorship: SponsorshipStance = SponsorshipStance.NOT_MENTIONED
    sponsorship_language: str | None = None

    salary_range: SalaryRange | None = None
    seniority_level: SeniorityLevel = SeniorityLevel.UNKNOWN

    @model_validator(mode="after")
    def unquoted_sponsorship_claims_abstain(self) -> "JobRequirements":
        """Treat a sponsorship stance the posting cannot back up as silence.

        Observed in a live run: the model reported "not_offered" for a posting
        containing no sponsorship wording at all, leaving sponsorship_language
        null. Acting on that would hard-filter a job on invented evidence, so
        an unquoted stance is downgraded to NOT_MENTIONED.
        """
        if self.sponsorship is SponsorshipStance.NOT_MENTIONED:
            return self

        if self.sponsorship_language and self.sponsorship_language.strip():
            return self

        self.sponsorship = SponsorshipStance.NOT_MENTIONED

        return self


class ComponentScores(BaseModel):
    skills: int = Field(ge=0, le=100)
    education: int = Field(ge=0, le=100)
    experience: int = Field(ge=0, le=100)
    career_relevance: int = Field(ge=0, le=100)


class JobEvaluation(BaseModel):
    """Structured output from the LLM.

    Intentionally carries no overall score and no recommendation: both are
    computed deterministically in Python from scoring.yaml.
    """

    requirements: JobRequirements
    scores: ComponentScores

    strengths: list[str]
    missing_requirements: list[str]
    reasoning: str


class JobAnalysis(BaseModel):
    """The final explainable result."""

    requirements: JobRequirements
    scores: ComponentScores

    overall_score: float = Field(ge=0, le=100)
    recommendation: Recommendation

    passes_hard_filters: bool
    hard_filter_reasons: list[str]

    # Non-blocking warnings. Never affect the score or the recommendation.
    concerns: list[str] = Field(default_factory=list)

    strengths: list[str]
    missing_requirements: list[str]
    reasoning: str

    source_url: str | None = None
    analyzed_at: datetime = Field(default_factory=datetime.now)

    @property
    def company(self) -> str:
        return self.requirements.company

    @property
    def job_title(self) -> str:
        return self.requirements.job_title

    @property
    def location(self) -> str | None:
        return self.requirements.location
