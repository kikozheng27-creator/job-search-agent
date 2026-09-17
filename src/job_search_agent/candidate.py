from enum import Enum

from pydantic import BaseModel, Field


class WorkMode(str, Enum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"


class ExperienceLevel(str, Enum):
    INTERNSHIP = "internship"
    ENTRY_LEVEL = "entry_level"
    MID_LEVEL = "mid_level"
    SENIOR = "senior"
    LEAD = "lead"


class Degree(BaseModel):
    degree: str
    field: str
    institution: str | None = None
    graduation_year: int | None = None


class Skills(BaseModel):
    programming: list[str] = Field(default_factory=list)
    statistics: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)

    def all_skills(self) -> list[str]:
        return [*self.programming, *self.statistics, *self.tools]


class Project(BaseModel):
    name: str
    description: str
    skills_used: list[str] = Field(default_factory=list)


class WorkExperience(BaseModel):
    title: str
    organization: str
    start_date: str
    end_date: str | None = None
    highlights: list[str] = Field(default_factory=list)


class LocationPreferences(BaseModel):
    preferred: list[str] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)
    work_modes: list[WorkMode] = Field(default_factory=list)


class WorkAuthorization(BaseModel):
    """Drives the work-authorization hard filter, so these values must be accurate."""

    status: str

    # True only when you cannot begin work without the employer filing a
    # petition first. A student on OPT or STEM OPT is already authorized, so
    # this stays False for them even though sponsorship is needed eventually.
    requires_sponsorship: bool

    is_citizen_or_permanent_resident: bool = False

    # True when sponsorship will be needed once current authorization ends.
    # Surfaces as a concern on the analysis, never as a hard filter.
    requires_future_sponsorship: bool = False


class CandidateProfile(BaseModel):
    target_job_families: list[str]
    education: list[Degree]
    skills: Skills
    experience_level: ExperienceLevel
    years_of_experience: float = Field(default=0.0, ge=0)
    work_authorization: WorkAuthorization

    projects: list[Project] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    locations: LocationPreferences = Field(default_factory=LocationPreferences)
    preferred_industries: list[str] = Field(default_factory=list)
