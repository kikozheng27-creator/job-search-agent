from job_search_agent.config_models import ScoringWeights, Thresholds
from job_search_agent.models import ComponentScores, Recommendation


def calculate_overall_score(
    scores: ComponentScores,
    weights: ScoringWeights,
) -> float:
    overall_score = (
        scores.skills * weights.skills
        + scores.education * weights.education
        + scores.experience * weights.experience
        + scores.career_relevance * weights.career_relevance
    )

    return round(overall_score, 1)


def get_recommendation(
    overall_score: float,
    thresholds: Thresholds,
) -> Recommendation:
    if overall_score >= thresholds.strongly_apply:
        return Recommendation.STRONGLY_APPLY

    if overall_score >= thresholds.apply:
        return Recommendation.APPLY

    if overall_score >= thresholds.maybe:
        return Recommendation.MAYBE

    return Recommendation.SKIP
