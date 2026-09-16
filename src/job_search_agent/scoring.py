def calculate_overall_score(
    skills_score: int,
    education_score: int,
    experience_score: int,
    career_relevance_score: int,
    weights: dict,
) -> float:
    overall_score = (
        skills_score * weights["skills"]
        + education_score * weights["education"]
        + experience_score * weights["experience"]
        + career_relevance_score * weights["career_relevance"]
    )

    return round(overall_score, 1)

from job_search_agent.models import Recommendation

def get_recommendation(
    overall_score: float,
    thresholds: dict,
) -> Recommendation:
    if overall_score >= thresholds["strongly_apply"]:
        return Recommendation.STRONGLY_APPLY

    if overall_score >= thresholds["apply"]:
        return Recommendation.APPLY

    if overall_score >= thresholds["maybe"]:
        return Recommendation.MAYBE

    return Recommendation.SKIP