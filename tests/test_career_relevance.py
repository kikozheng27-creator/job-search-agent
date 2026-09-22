from conftest import (
    PERMISSIVE_FILTERS,
    FakeAIClient,
    make_evaluation,
    make_profile,
    make_requirements,
    make_scores,
)
from job_search_agent.candidate import Degree
from job_search_agent.config_models import ScoringConfig, ScoringWeights, Thresholds
from job_search_agent.career_relevance_scorer import (
    score_career_relevance,
    validate_career_alignment,
)
from job_search_agent.job_matcher import JobMatcher
from job_search_agent.models import (
    CareerRelevanceEvidence,
    PreferredIndustryRelation,
    TargetFamilyRelation,
)
from job_search_agent.config_models import SkillsScoringConfig


POSTING = "We are hiring a Data Analyst to build reports for a hospital."
FAMILY_QUOTE = "We are hiring a Data Analyst to build reports for a hospital."
INDUSTRY_QUOTE = "build reports for a hospital."

DEFAULT_SKILLS = SkillsScoringConfig(
    required_weight=0.80,
    preferred_weight=0.20,
)


def alignment(
    family: TargetFamilyRelation,
    industry: PreferredIndustryRelation,
    *,
    family_quote: str | None = FAMILY_QUOTE,
    industry_quote: str | None = INDUSTRY_QUOTE,
) -> CareerRelevanceEvidence:
    return CareerRelevanceEvidence(
        target_family_relation=family,
        target_family_evidence=family_quote,
        preferred_industry_relation=industry,
        preferred_industry_evidence=industry_quote,
    )


def score_validated(evidence: CareerRelevanceEvidence, profile=None) -> int:
    profile = profile or make_profile(preferred_industries=["Healthcare"])
    validated = validate_career_alignment(evidence, profile, POSTING)
    return score_career_relevance(validated)


def test_direct_without_industry_preference_scores_100():
    evidence = alignment(
        TargetFamilyRelation.DIRECT,
        PreferredIndustryRelation.NOT_APPLICABLE,
        industry_quote=None,
    )

    assert score_validated(evidence, make_profile()) == 100


def test_adjacent_without_industry_preference_scores_70():
    evidence = alignment(
        TargetFamilyRelation.ADJACENT,
        PreferredIndustryRelation.NOT_APPLICABLE,
        industry_quote=None,
    )

    assert score_validated(evidence, make_profile()) == 70


def test_unrelated_without_industry_preference_scores_0():
    evidence = alignment(
        TargetFamilyRelation.UNRELATED,
        PreferredIndustryRelation.NOT_APPLICABLE,
        industry_quote=None,
    )

    assert score_validated(evidence, make_profile()) == 0


def test_unclear_without_industry_preference_scores_50():
    evidence = alignment(
        TargetFamilyRelation.UNCLEAR,
        PreferredIndustryRelation.NOT_APPLICABLE,
        family_quote=None,
        industry_quote=None,
    )

    assert score_validated(evidence, make_profile()) == 50


def test_direct_match_scores_100():
    evidence = alignment(
        TargetFamilyRelation.DIRECT,
        PreferredIndustryRelation.MATCH,
    )

    assert score_validated(evidence) == 100


def test_direct_mismatch_scores_80():
    evidence = alignment(
        TargetFamilyRelation.DIRECT,
        PreferredIndustryRelation.MISMATCH,
    )

    assert score_validated(evidence) == 80


def test_adjacent_match_scores_76():
    evidence = alignment(
        TargetFamilyRelation.ADJACENT,
        PreferredIndustryRelation.MATCH,
    )

    assert score_validated(evidence) == 76


def test_adjacent_mismatch_scores_56():
    evidence = alignment(
        TargetFamilyRelation.ADJACENT,
        PreferredIndustryRelation.MISMATCH,
    )

    assert score_validated(evidence) == 56


def test_unknown_industry_does_not_change_the_family_score():
    profile = make_profile(preferred_industries=["Healthcare"])
    evidence = alignment(
        TargetFamilyRelation.DIRECT,
        PreferredIndustryRelation.UNKNOWN,
        industry_quote=None,
    )
    validated = validate_career_alignment(evidence, profile, POSTING)

    assert validated.preferred_industry_relation is PreferredIndustryRelation.UNKNOWN
    assert score_career_relevance(validated) == 100


def test_empty_preferred_industries_become_not_applicable():
    profile = make_profile(preferred_industries=[])
    evidence = alignment(
        TargetFamilyRelation.DIRECT,
        PreferredIndustryRelation.MISMATCH,
    )
    validated = validate_career_alignment(evidence, profile, POSTING)

    assert (
        validated.preferred_industry_relation
        is PreferredIndustryRelation.NOT_APPLICABLE
    )
    assert score_career_relevance(validated) == 100


def _matcher(evaluation):
    return JobMatcher(
        ai_client=FakeAIClient(evaluation),
        scoring_config=ScoringConfig(
            weights=ScoringWeights(
                skills=0.35,
                education=0.20,
                experience=0.25,
                career_relevance=0.20,
            ),
            thresholds=Thresholds(strongly_apply=85, apply=70, maybe=55),
            skills=DEFAULT_SKILLS,
        ),
        filter_config=PERMISSIVE_FILTERS,
    )


def test_job_matcher_overwrites_raw_llm_career_relevance():
    evidence = alignment(
        TargetFamilyRelation.DIRECT,
        PreferredIndustryRelation.NOT_APPLICABLE,
        industry_quote=None,
    )
    scores = []

    for raw in (0, 30, 75, 100):
        result = _matcher(
            make_evaluation(
                scores=make_scores(career_relevance=raw),
                career_alignment=evidence,
            )
        ).match(make_profile(), POSTING)
        scores.append(result.scores.career_relevance)

    assert scores == [100, 100, 100, 100]


def test_skills_do_not_change_career_relevance():
    evidence = alignment(
        TargetFamilyRelation.ADJACENT,
        PreferredIndustryRelation.NOT_APPLICABLE,
        industry_quote=None,
    )
    evaluation = make_evaluation(career_alignment=evidence)
    baseline = make_profile()
    changed = make_profile(
        skills=baseline.skills.model_copy(
            update={"programming": ["COBOL"], "statistics": [], "tools": []}
        )
    )

    first = _matcher(evaluation).match(baseline, POSTING)
    first_career = first.scores.career_relevance
    first_skills = first.scores.skills
    second = _matcher(evaluation).match(changed, POSTING)

    assert first_career == second.scores.career_relevance == 70
    assert first_skills != second.scores.skills


def test_years_of_experience_do_not_change_career_relevance():
    evidence = alignment(
        TargetFamilyRelation.DIRECT,
        PreferredIndustryRelation.NOT_APPLICABLE,
        industry_quote=None,
    )
    evaluation = make_evaluation(
        requirements=make_requirements(minimum_years_experience=5),
        career_alignment=evidence,
    )

    junior = _matcher(evaluation).match(
        make_profile(years_of_experience=0),
        POSTING,
    )
    junior_career = junior.scores.career_relevance
    junior_experience = junior.scores.experience
    senior = _matcher(evaluation).match(
        make_profile(years_of_experience=12),
        POSTING,
    )

    assert junior_career == senior.scores.career_relevance == 100
    assert junior_experience != senior.scores.experience


def test_degree_level_does_not_change_career_relevance():
    evidence = alignment(
        TargetFamilyRelation.DIRECT,
        PreferredIndustryRelation.NOT_APPLICABLE,
        industry_quote=None,
    )
    evaluation = make_evaluation(
        requirements=make_requirements(required_degree="Master's"),
        career_alignment=evidence,
    )

    bachelor = _matcher(evaluation).match(
        make_profile(education=[Degree(degree="Bachelor's", field="Example Field")]),
        POSTING,
    )
    bachelor_career = bachelor.scores.career_relevance
    bachelor_education = bachelor.scores.education
    doctorate = _matcher(evaluation).match(
        make_profile(education=[Degree(degree="PhD", field="Example Field")]),
        POSTING,
    )

    assert bachelor_career == doctorate.scores.career_relevance == 100
    assert bachelor_education != doctorate.scores.education


def test_unquoted_non_abstaining_alignment_is_downgraded():
    profile = make_profile(preferred_industries=["Healthcare"])
    evidence = alignment(
        TargetFamilyRelation.DIRECT,
        PreferredIndustryRelation.MISMATCH,
        family_quote="Chief Astronaut",
        industry_quote="interstellar mining",
    )
    validated = validate_career_alignment(evidence, profile, POSTING)

    assert validated.target_family_relation is TargetFamilyRelation.UNCLEAR
    assert validated.target_family_evidence is None
    assert validated.preferred_industry_relation is PreferredIndustryRelation.UNKNOWN
    assert validated.preferred_industry_evidence is None
    assert score_career_relevance(validated) == 50

    result = _matcher(
        make_evaluation(scores=make_scores(career_relevance=100), career_alignment=evidence)
    ).match(profile, POSTING)

    assert result.scores.career_relevance == 50
    assert result.scores.career_relevance != 100
    assert any("role-family alignment" in concern for concern in result.concerns)


def test_unclear_family_alignment_adds_a_non_blocking_concern():
    result = _matcher(make_evaluation(scores=make_scores(career_relevance=0))).match(
        make_profile(),
        POSTING,
    )

    assert result.scores.career_relevance == 50
    assert result.passes_hard_filters is True
    assert any(
        "neutral abstention" in concern and "role-family alignment" in concern
        for concern in result.concerns
    )
