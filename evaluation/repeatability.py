"""Repeatability evaluation: same profile, same posting, N live analyses.

Does not write tracker rows or processed reports. Does not change scoring.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from job_search_agent.ai_client import AIClient
from job_search_agent.config_loader import load_scoring_config
from job_search_agent.main import analyze_job_description
from job_search_agent.scoring import get_recommendation


DEFAULT_JD = Path("jobs/example.txt")
DEFAULT_RUNS = 10

NUMERIC_FIELDS = (
    "skills_score",
    "education_score",
    "experience_score",
    "career_relevance_score",
    "overall_score",
)

WEIGHTS = {
    "skills_score": 0.35,
    "education_score": 0.20,
    "experience_score": 0.25,
    "career_relevance_score": 0.20,
}


def summarize(values: list[float]) -> dict:
    return {
        "mean": round(statistics.mean(values), 3),
        "minimum": min(values),
        "maximum": max(values),
        "range": round(max(values) - min(values), 3),
        "std_dev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
    }


def main() -> None:
    posting = DEFAULT_JD.read_text(encoding="utf-8")
    client = AIClient()
    scoring = load_scoring_config("config/scoring.yaml")
    runs: list[dict] = []

    for index in range(1, DEFAULT_RUNS + 1):
        analysis = analyze_job_description(
            job_description=posting,
            ai_client=client,
        )
        score_only = get_recommendation(
            analysis.overall_score,
            scoring.thresholds,
        )
        row = {
            "run": index,
            "minimum_years_experience": (
                analysis.requirements.minimum_years_experience
            ),
            "required_skills": analysis.requirements.required_skills,
            "preferred_skills": analysis.requirements.preferred_skills,
            "skills_score": analysis.scores.skills,
            "education_score": analysis.scores.education,
            "experience_score": analysis.scores.experience,
            "career_relevance_score": analysis.scores.career_relevance,
            "overall_score": analysis.overall_score,
            "score_based_recommendation": score_only.value,
            "passes_hard_filters": analysis.passes_hard_filters,
            "hard_filter_reasons": analysis.hard_filter_reasons,
            "final_recommendation": analysis.recommendation.value,
        }
        runs.append(row)
        print(
            f"run {index:02d}  overall={row['overall_score']:5.1f}  "
            f"skills={row['skills_score']:3}  edu={row['education_score']:3}  "
            f"exp={row['experience_score']:3}  rel={row['career_relevance_score']:3}  "
            f"score_rec={row['score_based_recommendation']:<15}  "
            f"final={row['final_recommendation']}",
            flush=True,
        )

    stats = {field: summarize([row[field] for row in runs]) for field in NUMERIC_FIELDS}

    contributions = {}
    for field, weight in WEIGHTS.items():
        variance = statistics.variance([row[field] for row in runs])
        contributions[field] = round((weight**2) * variance, 4)

    dominant = max(contributions, key=contributions.get)

    required_sets = {tuple(row["required_skills"]) for row in runs}
    preferred_sets = {tuple(row["preferred_skills"]) for row in runs}
    experience_values = {row["minimum_years_experience"] for row in runs}

    score_recs = {row["score_based_recommendation"] for row in runs}
    final_recs = {row["final_recommendation"] for row in runs}
    recs_always_match_score = all(
        row["score_based_recommendation"] == row["final_recommendation"]
        for row in runs
    )
    hard_filters_always_pass = all(row["passes_hard_filters"] for row in runs)

    report = {
        "job_description": str(DEFAULT_JD),
        "runs": DEFAULT_RUNS,
        "observations": runs,
        "numeric_summary": stats,
        "variance_contribution_weight_squared_times_variance": contributions,
        "component_most_responsible_for_score_variance": dominant,
        "distinct_score_based_recommendations": sorted(score_recs),
        "distinct_final_recommendations": sorted(final_recs),
        "extracted_facts_changed": {
            "minimum_years_experience": len(experience_values) > 1,
            "minimum_years_experience_values": sorted(
                value for value in experience_values if value is not None
            ),
            "required_skills": len(required_sets) > 1,
            "required_skills_variants": [list(item) for item in required_sets],
            "preferred_skills": len(preferred_sets) > 1,
            "preferred_skills_variants": [list(item) for item in preferred_sets],
        },
        "recommendation_changes_only_from_thresholds": (
            recs_always_match_score and hard_filters_always_pass
        ),
        "hard_filters_always_passed": hard_filters_always_pass,
        "score_based_always_equals_final": recs_always_match_score,
        "thresholds": scoring.thresholds.model_dump(),
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
