from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Rejects unknown keys so typos in YAML surface as errors, not silence."""

    model_config = ConfigDict(extra="forbid")


class ScoringWeights(StrictModel):
    skills: float = Field(ge=0)
    education: float = Field(ge=0)
    experience: float = Field(ge=0)
    career_relevance: float = Field(ge=0)

    @model_validator(mode="after")
    def weights_must_sum_to_one(self) -> "ScoringWeights":
        total = (
            self.skills
            + self.education
            + self.experience
            + self.career_relevance
        )

        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Scoring weights must sum to 1.0, but they sum to {total:.3f}."
            )

        return self


class Thresholds(StrictModel):
    strongly_apply: float = Field(ge=0, le=100)
    apply: float = Field(ge=0, le=100)
    maybe: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def thresholds_must_descend(self) -> "Thresholds":
        if not self.strongly_apply > self.apply > self.maybe:
            raise ValueError(
                "Thresholds must be strictly descending, but got "
                f"strongly_apply={self.strongly_apply}, "
                f"apply={self.apply}, maybe={self.maybe}."
            )

        return self


class SkillsScoringConfig(StrictModel):
    """Deterministic skills matching. Aliases map a variant onto a canonical name."""

    required_weight: float = Field(default=0.80, ge=0, le=1)
    preferred_weight: float = Field(default=0.20, ge=0, le=1)
    aliases: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def skill_weights_must_sum_to_one(self) -> "SkillsScoringConfig":
        total = self.required_weight + self.preferred_weight

        if abs(total - 1.0) > 0.001:
            raise ValueError(
                "skills.required_weight and skills.preferred_weight must sum "
                f"to 1.0, but they sum to {total:.3f}."
            )

        return self


class ScoringConfig(StrictModel):
    weights: ScoringWeights
    thresholds: Thresholds
    skills: SkillsScoringConfig = Field(default_factory=SkillsScoringConfig)


class ExperienceFilterConfig(StrictModel):
    enabled: bool = True
    max_required_years: int = Field(ge=0)
    tolerance_years: int = Field(default=0, ge=0)


class ToggleFilterConfig(StrictModel):
    enabled: bool = True


class FilterConfig(StrictModel):
    experience: ExperienceFilterConfig
    work_authorization: ToggleFilterConfig = Field(
        default_factory=ToggleFilterConfig
    )
    location: ToggleFilterConfig = Field(default_factory=ToggleFilterConfig)
