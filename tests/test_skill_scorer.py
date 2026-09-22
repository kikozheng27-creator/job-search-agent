from conftest import make_profile, make_requirements

from job_search_agent.config_models import SkillsScoringConfig
from job_search_agent.skill_scorer import (
    diagnose_skills,
    is_degree_requirement,
    score_skills,
)


DEFAULT_SKILLS = SkillsScoringConfig(
    required_weight=0.80,
    preferred_weight=0.20,
    aliases={
        "r programming": "R",
        "sas programming": "SAS",
    },
)


def score(profile=None, requirements=None, config=None) -> int:
    return score_skills(
        profile or make_profile(),
        requirements or make_requirements(),
        config or DEFAULT_SKILLS,
    )


def test_exact_skill_match_scores_full_required_and_missing_preferred():
    """Profile has R; posting requires R and prefers SAS."""
    assert score() == 80


def test_alias_r_programming_matches_r():
    requirements = make_requirements(
        required_skills=["R programming"],
        preferred_skills=[],
    )

    assert score(requirements=requirements) == 100


def test_missing_one_required_skill_lowers_the_score():
    requirements = make_requirements(
        required_skills=["R", "Python", "SAS"],
        preferred_skills=[],
    )

    # 2 of 3 required, no preferred list → 100 * 0.8 * (2/3) + 100 * 0.2 * 1
    assert score(requirements=requirements) == 73


def test_missing_only_preferred_skills_keeps_a_high_score():
    requirements = make_requirements(
        required_skills=["R", "Python"],
        preferred_skills=["SAS"],
    )

    assert score(requirements=requirements) == 80


def test_no_preferred_skills_in_the_posting():
    requirements = make_requirements(
        required_skills=["R"],
        preferred_skills=[],
    )

    assert score(requirements=requirements) == 100


def test_empty_skill_lists_score_100():
    requirements = make_requirements(
        required_skills=[],
        preferred_skills=[],
    )

    assert score(requirements=requirements) == 100


def test_empty_candidate_skills_score_0_when_the_posting_lists_skills():
    profile = make_profile(skills={"programming": [], "statistics": [], "tools": []})
    requirements = make_requirements(
        required_skills=["R"],
        preferred_skills=["SAS"],
    )

    assert score(profile=profile, requirements=requirements) == 0


def test_degree_text_leaking_into_required_skills_is_ignored():
    requirements = make_requirements(
        required_skills=[
            "R",
            "regression",
            "Master's in Biostatistics or Statistics",
        ],
        preferred_skills=["SAS"],
    )

    # Degree leak dropped; remaining required both match; SAS preferred misses.
    assert score(requirements=requirements) == 80
    assert is_degree_requirement("Master's in Biostatistics or Statistics")
    assert is_degree_requirement("PhD in Statistics")
    assert is_degree_requirement("Bachelor's degree")


def test_biostatistics_as_a_competency_is_not_treated_as_a_degree_leak():
    assert not is_degree_requirement("Biostatistics")

    profile = make_profile(
        skills={
            "programming": ["R"],
            "statistics": ["Biostatistics", "Regression"],
            "tools": [],
        },
    )
    requirements = make_requirements(
        required_skills=["Biostatistics", "R"],
        preferred_skills=[],
    )

    assert score(profile=profile, requirements=requirements) == 100


def test_identical_inputs_are_repeatable():
    first = score()
    second = score()

    assert first == second == 80


def test_diagnose_skills_matches_score_and_lists_matches():
    requirements = make_requirements(
        required_skills=["R", "Python", "SAS"],
        preferred_skills=["CDISC"],
    )
    profile = make_profile()
    diagnostic = diagnose_skills(profile, requirements, DEFAULT_SKILLS)

    assert diagnostic["skills_score"] == score(
        profile=profile,
        requirements=requirements,
    )
    assert diagnostic["required"]["raw"] == ["R", "Python", "SAS"]
    assert diagnostic["required"]["matched"] == ["R", "Python"]
    assert diagnostic["required"]["missing"] == ["SAS"]
    assert diagnostic["required"]["match_rate"] == round(2 / 3, 4)
    assert diagnostic["preferred"]["missing"] == ["CDISC"]
    assert diagnostic["preferred"]["matched"] == []
    assert diagnostic["preferred"]["match_rate"] == 0.0


def test_diagnose_skills_is_repeatable_for_identical_inputs():
    requirements = make_requirements(
        required_skills=["R programming", "Python"],
        preferred_skills=[],
    )
    first = diagnose_skills(make_profile(), requirements, DEFAULT_SKILLS)
    second = diagnose_skills(make_profile(), requirements, DEFAULT_SKILLS)

    assert first == second
    assert first["skills_score"] == 100
    assert first["required"]["normalized"] == ["r", "python"]


def test_diagnose_skills_empty_lists_match_rate_is_one():
    requirements = make_requirements(
        required_skills=[],
        preferred_skills=[],
    )
    diagnostic = diagnose_skills(make_profile(), requirements, DEFAULT_SKILLS)

    assert diagnostic["required"]["match_rate"] == 1.0
    assert diagnostic["preferred"]["match_rate"] == 1.0
    assert diagnostic["skills_score"] == 100


def test_score_stays_within_0_and_100():
    assert 0 <= score() <= 100
    assert score(
        requirements=make_requirements(required_skills=[], preferred_skills=[]),
    ) == 100


def test_required_skills_weigh_more_than_preferred_skills():
    required_only_miss = make_requirements(
        required_skills=["SAS"],
        preferred_skills=["R"],
    )
    preferred_only_miss = make_requirements(
        required_skills=["R"],
        preferred_skills=["SAS"],
    )

    assert score(requirements=required_only_miss) < score(
        requirements=preferred_only_miss
    )
    assert score(requirements=required_only_miss) == 20
    assert score(requirements=preferred_only_miss) == 80
