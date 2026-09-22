from job_search_agent.authorization_claims import sanitize_analysis_prose
from job_search_agent.candidate import CandidateProfile
from job_search_agent.concerns import collect_concerns
from job_search_agent.config_models import FilterConfig, ScoringConfig
from job_search_agent.education_scorer import score_education
from job_search_agent.evidence_units import build_evidence_units
from job_search_agent.experience_scorer import score_experience
from job_search_agent.filters import check_hard_filters
from job_search_agent.models import (
    JobAnalysis,
    Recommendation,
    SkillExtractionReport,
    SkillExtractionStatus,
)
from job_search_agent.prompts import build_evaluation_prompt
from job_search_agent.protocols import JobEvaluator
from job_search_agent.reasoning import build_fallback_reasoning
from job_search_agent.scoring import (
    calculate_overall_score,
    get_recommendation,
)
from job_search_agent.skill_extractor import collect_skill_inventory
from job_search_agent.skill_grounding import apply_skill_grounding
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
        units = build_evidence_units(job_description)
        extraction_report = SkillExtractionReport()
        dedicated_lists = None
        if getattr(self.ai_client, "supports_dedicated_skill_extraction", False):
            required, preferred, extraction_report = collect_skill_inventory(
                self.ai_client,
                job_description,
                units,
            )
            dedicated_lists = (required, preferred)

        prompt = build_evaluation_prompt(profile, job_description)
        evaluation = self.ai_client.evaluate_job(prompt)

        apply_sponsorship_classification(
            evaluation.requirements,
            job_description,
        )

        if dedicated_lists is not None:
            evaluation.requirements.required_skills = dedicated_lists[0]
            evaluation.requirements.preferred_skills = dedicated_lists[1]
        else:
            apply_skill_grounding(
                evaluation.requirements,
                evaluation.skill_claims,
                job_description,
                unit_decisions=evaluation.skill_unit_decisions,
                units=units,
            )

        score_from_inventory = True
        if (
            extraction_report.status is SkillExtractionStatus.UNRESOLVED
            and not evaluation.requirements.required_skills
            and not evaluation.requirements.preferred_skills
        ):
            score_from_inventory = False

        if score_from_inventory:
            evaluation.scores.skills = score_skills(
                profile,
                evaluation.requirements,
                self.scoring_config.skills,
            )

        evaluation.scores.experience = score_experience(
            profile,
            evaluation.requirements,
        )
        evaluation.scores.education = score_education(
            profile,
            evaluation.requirements,
        )

        extraction_report.skills_scored_from_inventory = (
            score_from_inventory
            and extraction_report.status is not SkillExtractionStatus.NOT_RUN
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

        concerns = collect_concerns(evaluation.requirements, profile)
        if extraction_report.status is SkillExtractionStatus.UNRESOLVED:
            unit_ids = ", ".join(extraction_report.unresolved_unit_ids) or "unknown"
            concerns.append(
                "Skill inventory extraction is unresolved for "
                f"{unit_ids}. This is not a genuine empty required-skill list."
            )

        return JobAnalysis(
            requirements=evaluation.requirements,
            scores=evaluation.scores,
            overall_score=overall_score,
            recommendation=recommendation,
            passes_hard_filters=passes_hard_filters,
            hard_filter_reasons=hard_filter_reasons,
            concerns=concerns,
            strengths=strengths,
            missing_requirements=missing_requirements,
            reasoning=reasoning,
            source_url=source_url,
            skill_extraction=extraction_report,
        )
