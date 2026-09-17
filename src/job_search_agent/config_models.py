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


class ScoringConfig(StrictModel):
    weights: ScoringWeights
    thresholds: Thresholds


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
