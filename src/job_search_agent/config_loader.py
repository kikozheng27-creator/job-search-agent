from pathlib import Path

import yaml

from job_search_agent.candidate import CandidateProfile
from job_search_agent.config_models import FilterConfig, ScoringConfig


def load_config(path: str | Path) -> dict:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file must contain a YAML mapping: {config_path}"
        )

    return config


def load_candidate_profile(path: str | Path) -> CandidateProfile:
    return CandidateProfile.model_validate(load_config(path))


def load_scoring_config(path: str | Path) -> ScoringConfig:
    return ScoringConfig.model_validate(load_config(path))


def load_filter_config(path: str | Path) -> FilterConfig:
    return FilterConfig.model_validate(load_config(path))
