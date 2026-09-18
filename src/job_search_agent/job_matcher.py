from job_search_agent.ai_client import AIClient
from job_search_agent.filters import check_hard_filters
from job_search_agent.models import CandidateProfile, JobAnalysis, Recommendation
from job_search_agent.scoring import (
    calculate_overall_score,
    get_recommendation,
)


class JobMatcher:
    def __init__(
        self,
        ai_client: AIClient,
        weights: dict,
        thresholds: dict,
        filter_config: dict,
    ):
        self.ai_client = ai_client
        self.weights = weights
        self.thresholds = thresholds
        self.filter_config = filter_config

    def match(
        self,
        profile: CandidateProfile,
        job_description: str,
    ) -> JobAnalysis:
        prompt = f"""
You are evaluating how well a candidate matches a job.

Candidate profile:
{profile.model_dump_json(indent=2)}

Job description:
{job_description}

Evaluate the candidate objectively.

Extract the job requirements and score each category from 0 to 100:

- skills_score
- education_score
- experience_score
- career_relevance_score

Do not calculate an overall score.
Do not make the final apply/skip recommendation.
"""

        evaluation = self.ai_client.evaluate_job(prompt)

        overall_score = calculate_overall_score(
            skills_score=evaluation.skills_score,
            education_score=evaluation.education_score,
            experience_score=evaluation.experience_score,
            career_relevance_score=evaluation.career_relevance_score,
            weights=self.weights,
        )

        hard_filter_reasons = check_hard_filters(
            evaluation.requirements,
            self.filter_config,
        )

        passes_hard_filters = len(hard_filter_reasons) == 0

        if not passes_hard_filters:
            recommendation = Recommendation.SKIP
        else:
            recommendation = get_recommendation(
                overall_score,
                self.thresholds,
            )

        return JobAnalysis(
            job_title=evaluation.job_title,
            company=evaluation.company,
            skills_score=evaluation.skills_score,
            education_score=evaluation.education_score,
            experience_score=evaluation.experience_score,
            career_relevance_score=evaluation.career_relevance_score,
            overall_score=overall_score,
            recommendation=recommendation,
            passes_hard_filters=passes_hard_filters,
            hard_filter_reasons=hard_filter_reasons,
            strengths=evaluation.strengths,
            missing_requirements=evaluation.missing_requirements,
            reasoning=evaluation.reasoning,
        )