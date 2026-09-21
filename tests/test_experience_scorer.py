from conftest import make_profile, make_requirements

from job_search_agent.candidate import (
    CandidateProfile,
    ExperienceLevel,
    Project,
    WorkExperience,
)
from job_search_agent.experience_scorer import score_experience


def test_unstated_minimum_scores_100():
    assert (
        score_experience(
            make_profile(years_of_experience=0.5),
            make_requirements(minimum_years_experience=None),
        )
        == 100
    )


def test_zero_minimum_scores_100():
    assert (
        score_experience(
            make_profile(years_of_experience=0),
            make_requirements(minimum_years_experience=0),
        )
        == 100
    )


def test_candidate_exactly_meeting_minimum_scores_100():
    assert (
        score_experience(
            make_profile(years_of_experience=2),
            make_requirements(minimum_years_experience=2),
        )
        == 100
    )


def test_candidate_exceeding_minimum_scores_100():
    assert (
        score_experience(
            make_profile(years_of_experience=5),
            make_requirements(minimum_years_experience=2),
        )
        == 100
    )


def test_shortfall_uses_the_deterministic_ratio():
    score = score_experience(
        make_profile(years_of_experience=1),
        make_requirements(minimum_years_experience=2),
    )

    assert score == round(84 * 1 / 2)
    assert score == 42
    assert score < 85


def test_benchmark_half_year_against_seven_years():
    score = score_experience(
        make_profile(years_of_experience=0.5),
        make_requirements(minimum_years_experience=7),
    )

    assert score == round(84 * 0.5 / 7)
    assert score == 6
    assert 0 <= score <= 100


def test_score_is_clamped_to_0_100():
    zero = score_experience(
        make_profile(years_of_experience=0),
        make_requirements(minimum_years_experience=7),
    )

    assert zero == 0
    assert 0 <= zero <= 100


def test_empty_work_history_does_not_change_the_score():
    years = 0.5
    requirements = make_requirements(minimum_years_experience=7)
    empty = make_profile(years_of_experience=years, work_experience=[], projects=[])
    filled = make_profile(
        years_of_experience=years,
        work_experience=[
            WorkExperience(
                title="Research Intern",
                organization="Lab",
                start_date="2024-01",
                end_date="2024-08",
                highlights=["Built a classifier"],
            )
        ],
        projects=[
            Project(
                name="Capstone",
                description="Academic research project",
                skills_used=["Python"],
            )
        ],
    )

    assert empty.work_experience == []
    assert empty.projects == []
    assert score_experience(empty, requirements) == score_experience(
        filled, requirements
    )


def test_experience_level_and_internship_label_do_not_change_the_score():
    requirements = make_requirements(minimum_years_experience=7)
    entry = make_profile(
        years_of_experience=0.5,
        experience_level=ExperienceLevel.ENTRY_LEVEL,
    )
    intern = make_profile(
        years_of_experience=0.5,
        experience_level=ExperienceLevel.INTERNSHIP,
    )

    assert score_experience(entry, requirements) == score_experience(
        intern, requirements
    )


def test_negative_candidate_years_are_treated_as_zero():
    profile = CandidateProfile.model_construct(
        **{
            **make_profile().model_dump(),
            "years_of_experience": -3,
        }
    )
    requirements = make_requirements(minimum_years_experience=7)

    assert profile.years_of_experience == -3
    assert score_experience(profile, requirements) == 0
