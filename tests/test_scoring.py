import pytest
from conftest import DEFAULT_THRESHOLDS, DEFAULT_WEIGHTS, make_scores

from job_search_agent.config_models import ScoringWeights
from job_search_agent.models import Recommendation
from job_search_agent.scoring import (
    calculate_overall_score,
    get_recommendation,
)


def test_calculate_overall_score():
    score = calculate_overall_score(
        make_scores(
            skills=90,
            education=100,
            experience=70,
            career_relevance=95,
        ),
        DEFAULT_WEIGHTS,
    )

    assert score == 88.0


def test_perfect_component_scores_give_100():
    score = calculate_overall_score(
        make_scores(
            skills=100,
            education=100,
            experience=100,
            career_relevance=100,
        ),
        DEFAULT_WEIGHTS,
    )

    assert score == 100.0


def test_zero_component_scores_give_0():
    score = calculate_overall_score(
        make_scores(
            skills=0,
            education=0,
            experience=0,
            career_relevance=0,
        ),
        DEFAULT_WEIGHTS,
    )

    assert score == 0.0


def test_weights_change_the_overall_score():
    scores = make_scores(
        skills=100,
        education=0,
        experience=0,
        career_relevance=0,
    )

    skills_heavy = ScoringWeights(
        skills=1.0,
        education=0.0,
        experience=0.0,
        career_relevance=0.0,
    )

    assert calculate_overall_score(scores, skills_heavy) == 100.0
    assert calculate_overall_score(scores, DEFAULT_WEIGHTS) == 35.0


@pytest.mark.parametrize(
    ("overall_score", "expected"),
    [
        (100.0, Recommendation.STRONGLY_APPLY),
        (85.0, Recommendation.STRONGLY_APPLY),
        (84.9, Recommendation.APPLY),
        (70.0, Recommendation.APPLY),
        (69.9, Recommendation.MAYBE),
        (55.0, Recommendation.MAYBE),
        (54.9, Recommendation.SKIP),
        (0.0, Recommendation.SKIP),
    ],
)
def test_recommendation_thresholds_are_inclusive_at_the_boundary(
    overall_score,
    expected,
):
    assert get_recommendation(overall_score, DEFAULT_THRESHOLDS) == expected
