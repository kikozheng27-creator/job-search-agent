from enum import Enum

from pydantic import BaseModel, Field


class Recommendation(str, Enum):
    STRONGLY_APPLY = "STRONGLY_APPLY"
    APPLY = "APPLY"
    MAYBE = "MAYBE"
    SKIP = "SKIP"


class Education(BaseModel):
    degree: str
    field: str


class Skills(BaseModel):
    programming: list[str]
    statistics: list[str]


class CandidateProfile(BaseModel):
    target_roles: list[str]
    education: Education
    skills: Skills
    experience_level: str


class JobRequirements(BaseModel):
    minimum_years_experience: int | None = Field(
        default=None,
        ge=0,
    )

    required_degree: str | None = None
    required_skills: list[str]
    preferred_skills: list[str]


class JobEvaluation(BaseModel):
    job_title: str
    company: str

    requirements: JobRequirements

    skills_score: int = Field(ge=0, le=100)
    education_score: int = Field(ge=0, le=100)
    experience_score: int = Field(ge=0, le=100)
    career_relevance_score: int = Field(ge=0, le=100)

    strengths: list[str]
    missing_requirements: list[str]
    reasoning: str


class JobAnalysis(BaseModel):
    job_title: str
    company: str

    skills_score: int = Field(ge=0, le=100)
    education_score: int = Field(ge=0, le=100)
    experience_score: int = Field(ge=0, le=100)
    career_relevance_score: int = Field(ge=0, le=100)

    overall_score: float = Field(ge=0, le=100)
    recommendation: Recommendation

    passes_hard_filters: bool
    hard_filter_reasons: list[str]

    strengths: list[str]
    missing_requirements: list[str]
    reasoning: str