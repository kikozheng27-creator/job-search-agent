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


class SkillLevel(str, Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"


class SkillClaim(BaseModel):
    """One LLM-extracted skill plus optional quote and source-unit ids.

    Unit ids are the source of truth when present. A free-form evidence
    quote is optional and never creates a source unit.
    """

    name: str
    level: SkillLevel = SkillLevel.REQUIRED
    evidence: str = ""
    source_unit_ids: list[str] = Field(default_factory=list)
    category: str = "technical_skill"


class UnitSkillItem(BaseModel):
    """A skill the model found inside one evidence unit."""

    name: str
    category: str = "technical_skill"


class UnitSkillDecision(BaseModel):
    """The model's decision for one Python-derived evidence unit."""

    source_unit_id: str
    is_candidate_requirement: bool
    skills: list[UnitSkillItem] = Field(default_factory=list)


class SkillMention(BaseModel):
    """One skill name enumerated from a single evidence unit."""

    name: str
    category: str = "technical_skill"


class UnitSkillInventory(BaseModel):
    """Exhaustive skill names the dedicated extractor found in one unit."""

    source_unit_id: str
    skills: list[SkillMention] = Field(default_factory=list)


class SkillInventory(BaseModel):
    """Narrow structured output for dedicated skill-name enumeration."""

    units: list[UnitSkillInventory] = Field(default_factory=list)


class SkillExtractionStatus(str, Enum):
    """Whether dedicated skill extraction finished with full coverage."""

    COMPLETE = "complete"
    UNRESOLVED = "unresolved"
    NOT_RUN = "not_run"


class SkillExtractionReport(BaseModel):
    """Coverage and completeness of the dedicated skill-inventory stage."""

    status: SkillExtractionStatus = SkillExtractionStatus.NOT_RUN
    expected_unit_ids: list[str] = Field(default_factory=list)
    returned_unit_ids: list[str] = Field(default_factory=list)
    missing_unit_ids: list[str] = Field(default_factory=list)
    unexpected_unit_ids: list[str] = Field(default_factory=list)
    unresolved_unit_ids: list[str] = Field(default_factory=list)
    repair_attempted: bool = False
    repaired_unit_ids: list[str] = Field(default_factory=list)
    per_unit_skills: dict[str, list[str]] = Field(default_factory=dict)
    skills_scored_from_inventory: bool = False


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
    skill_claims: list[SkillClaim] = Field(default_factory=list)
    skill_unit_decisions: list[UnitSkillDecision] = Field(default_factory=list)


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
    skill_extraction: SkillExtractionReport = Field(
        default_factory=SkillExtractionReport
    )

    @property
    def company(self) -> str:
        return self.requirements.company

    @property
    def job_title(self) -> str:
        return self.requirements.job_title

    @property
    def location(self) -> str | None:
        return self.requirements.location
