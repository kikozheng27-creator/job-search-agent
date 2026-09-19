from job_search_agent.authorization_claims import sanitize_analysis_prose
from job_search_agent.candidate import CandidateProfile
from job_search_agent.concerns import collect_concerns
from job_search_agent.config_models import FilterConfig, ScoringConfig
from job_search_agent.filters import check_hard_filters
from job_search_agent.models import JobAnalysis, Recommendation
from job_search_agent.prompts import build_evaluation_prompt
from job_search_agent.protocols import JobEvaluator
from job_search_agent.reasoning import build_fallback_reasoning
from job_search_agent.scoring import (
    calculate_overall_score,
    get_recommendation,
)
from job_search_agent.skill_scorer import score_skills
from job_search_agent.sponsorship_classifier import apply_sponsorship_classification


class JobMatcher:
    def __init__(
        self,
        ai_client: JobEvaluator,
        scoring_config: ScoringConfig,
        filter_config: FilterConfig,
    ):
        self.ai_client = ai_client
        self.scoring_config = scoring_config
        self.filter_config = filter_config

    def match(
        self,
        profile: CandidateProfile,
        job_description: str,
        source_url: str | None = None,
    ) -> JobAnalysis:
        prompt = build_evaluation_prompt(profile, job_description)
        evaluation = self.ai_client.evaluate_job(prompt)

        apply_sponsorship_classification(
            evaluation.requirements,
            job_description,
        )

        evaluation.scores.skills = score_skills(
            profile,
            evaluation.requirements,
            self.scoring_config.skills,
        )

        overall_score = calculate_overall_score(
            evaluation.scores,
            self.scoring_config.weights,
        )

        hard_filter_reasons = check_hard_filters(
            evaluation.requirements,
            profile,
            self.filter_config,
        )

        passes_hard_filters = not hard_filter_reasons

        if passes_hard_filters:
            recommendation = get_recommendation(
                overall_score,
                self.scoring_config.thresholds,
            )
        else:
            recommendation = Recommendation.SKIP

        strengths, missing_requirements, reasoning = sanitize_analysis_prose(
            evaluation.strengths,
            evaluation.missing_requirements,
            evaluation.reasoning,
            evaluation.requirements.sponsorship,
        )

        if not reasoning:
            reasoning = build_fallback_reasoning(
                requirements=evaluation.requirements,
                scores=evaluation.scores,
                recommendation=recommendation,
                hard_filter_reasons=hard_filter_reasons,
                missing_requirements=missing_requirements,
            )

        return JobAnalysis(
            requirements=evaluation.requirements,
            scores=evaluation.scores,
            overall_score=overall_score,
            recommendation=recommendation,
            passes_hard_filters=passes_hard_filters,
            hard_filter_reasons=hard_filter_reasons,
            concerns=collect_concerns(evaluation.requirements, profile),
            strengths=strengths,
            missing_requirements=missing_requirements,
            reasoning=reasoning,
            source_url=source_url,
        )
