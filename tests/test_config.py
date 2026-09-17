from pathlib import Path

import pytest
from pydantic import ValidationError

from job_search_agent.config_loader import (
    load_config,
    load_filter_config,
    load_scoring_config,
)
from job_search_agent.config_models import FilterConfig, ScoringConfig


VALID_SCORING = """
weights:
  skills: 0.35
  education: 0.20
  experience: 0.25
  career_relevance: 0.20

thresholds:
  strongly_apply: 85
  apply: 70
  maybe: 55
"""


def write_yaml(tmp_path, contents: str, name: str = "config.yaml"):
    path = tmp_path / name
    path.write_text(contents, encoding="utf-8")

    return path


def test_loads_valid_scoring_config(tmp_path):
    config = load_scoring_config(write_yaml(tmp_path, VALID_SCORING))

    assert config.weights.skills == 0.35
    assert config.thresholds.apply == 70


def test_missing_config_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scoring_config(tmp_path / "does_not_exist.yaml")


def test_non_mapping_config_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_config(write_yaml(tmp_path, "just a string"))


def test_weights_that_do_not_sum_to_one_are_rejected(tmp_path):
    contents = VALID_SCORING.replace("skills: 0.35", "skills: 0.95")

    with pytest.raises(ValidationError) as error:
        load_scoring_config(write_yaml(tmp_path, contents))

    assert "must sum to 1.0" in str(error.value)


def test_missing_weight_is_rejected(tmp_path):
    contents = VALID_SCORING.replace("  education: 0.20\n", "")

    with pytest.raises(ValidationError) as error:
        load_scoring_config(write_yaml(tmp_path, contents))

    assert "education" in str(error.value)


def test_unknown_weight_key_is_rejected(tmp_path):
    contents = VALID_SCORING.replace(
        "  skills: 0.35", "  skils: 0.35\n  skills: 0.35"
    )

    with pytest.raises(ValidationError) as error:
        load_scoring_config(write_yaml(tmp_path, contents))

    assert "skils" in str(error.value)


def test_non_descending_thresholds_are_rejected(tmp_path):
    contents = VALID_SCORING.replace("apply: 70", "apply: 90")

    with pytest.raises(ValidationError) as error:
        load_scoring_config(write_yaml(tmp_path, contents))

    assert "strictly descending" in str(error.value)


def test_threshold_above_100_is_rejected(tmp_path):
    contents = VALID_SCORING.replace("strongly_apply: 85", "strongly_apply: 150")

    with pytest.raises(ValidationError):
        load_scoring_config(write_yaml(tmp_path, contents))


def test_negative_weight_is_rejected(tmp_path):
    contents = VALID_SCORING.replace(
        "skills: 0.35", "skills: -0.35"
    ).replace("education: 0.20", "education: 0.90")

    with pytest.raises(ValidationError):
        load_scoring_config(write_yaml(tmp_path, contents))


def test_loads_valid_filter_config(tmp_path):
    contents = """
experience:
  enabled: true
  max_required_years: 2
  tolerance_years: 2

work_authorization:
  enabled: true

location:
  enabled: false
"""

    config = load_filter_config(write_yaml(tmp_path, contents))

    assert config.experience.max_required_years == 2
    assert config.experience.tolerance_years == 2
    assert config.location.enabled is False


def test_filter_toggles_default_to_enabled(tmp_path):
    contents = """
experience:
  enabled: true
  max_required_years: 2
"""

    config = load_filter_config(write_yaml(tmp_path, contents))

    assert config.experience.tolerance_years == 0
    assert config.work_authorization.enabled is True
    assert config.location.enabled is True


def test_negative_tolerance_is_rejected(tmp_path):
    contents = """
experience:
  enabled: true
  max_required_years: 2
  tolerance_years: -1
"""

    with pytest.raises(ValidationError):
        load_filter_config(write_yaml(tmp_path, contents))


def test_shipped_config_files_are_valid():
    """The configs committed to the repo must always load."""
    config_dir = Path(__file__).resolve().parents[1] / "config"

    scoring = ScoringConfig.model_validate(load_config(config_dir / "scoring.yaml"))
    filters = FilterConfig.model_validate(load_config(config_dir / "filters.yaml"))

    assert scoring.thresholds.strongly_apply > scoring.thresholds.apply
    assert filters.experience.max_required_years >= 0
