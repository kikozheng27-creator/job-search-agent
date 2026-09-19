from conftest import (
    DEFAULT_SCORING,
    FakeAIClient,
    make_profile,
    make_requirements,
    make_scores,
)
from job_search_agent.authorization_claims import (
    AuthorizationClaim,
    claims_in,
    is_supported,
)
from job_search_agent.candidate import WorkAuthorization
from job_search_agent.config_models import ExperienceFilterConfig, FilterConfig
from job_search_agent.job_matcher import JobMatcher
from job_search_agent.models import JobEvaluation, Recommendation, SponsorshipStance


INVENTED_F1 = "work authorization not permitted for F-1 candidates"
INVENTED_SPONSORSHIP = "visa sponsorship is not available for this role"
SUPPORTED_F1 = "F-1 candidates are not eligible"
SUPPORTED_H1B = "employer does not sponsor H-1B"


def opt_profile():
    return make_profile(
        work_authorization=WorkAuthorization(
            status="F-1 student; eligible for OPT/STEM OPT",
            requires_sponsorship=False,
            is_citizen_or_permanent_resident=False,
            requires_future_sponsorship=True,
        ),
    )


def evaluation_with_prose(
    *,
    missing_requirements: list[str],
    strengths: list[str] | None = None,
    reasoning: str = "The candidate matches most core requirements.",
    requirements=None,
    scores=None,
) -> JobEvaluation:
    return JobEvaluation(
        requirements=requirements or make_requirements(),
        scores=scores or make_scores(),
        strengths=strengths or ["Strong R skills"],
        missing_requirements=missing_requirements,
        reasoning=reasoning,
    )


def match(
    evaluation: JobEvaluation,
    job_description: str,
    *,
    profile=None,
    filter_config=None,
):
    matcher = JobMatcher(
        ai_client=FakeAIClient(evaluation),
        scoring_config=DEFAULT_SCORING,
        filter_config=filter_config
        or FilterConfig(
            experience=ExperienceFilterConfig(
                enabled=True,
                max_required_years=10,
                tolerance_years=0,
            ),
        ),
    )

    return matcher.match(
        profile=profile or opt_profile(),
        job_description=job_description,
    )


def test_silence_drops_invented_f1_missing_requirement():
    result = match(
        evaluation_with_prose(
            missing_requirements=["SAS", INVENTED_F1],
            reasoning=(
                "The candidate matches most core requirements. "
                "Work authorization is not permitted for F-1 candidates."
            ),
        ),
        "Example Pharma is hiring a Biostatistician. Strong R required.",
    )

    assert result.requirements.sponsorship is SponsorshipStance.NOT_MENTIONED
    assert INVENTED_F1 not in result.missing_requirements
    assert "SAS" in result.missing_requirements
    assert "F-1" not in result.reasoning
    assert "not permitted" not in result.reasoning
    assert "The candidate matches most core requirements." in result.reasoning


def test_silence_does_not_surface_invented_sponsorship_concern():
    result = match(
        evaluation_with_prose(
            missing_requirements=[INVENTED_SPONSORSHIP],
            strengths=["Does not sponsor visas"],
            reasoning="Sponsorship is unavailable, so this is a poor fit.",
        ),
        "Example Pharma is hiring a Biostatistician. Strong R required.",
    )

    assert result.requirements.sponsorship is SponsorshipStance.NOT_MENTIONED
    assert result.concerns == []
    assert result.missing_requirements == []
    assert result.strengths == []
    assert "sponsorship" not in result.reasoning.lower()


def test_explicit_no_opt_keeps_supported_restriction():
    result = match(
        evaluation_with_prose(
            missing_requirements=["OPT candidates are not eligible"],
            reasoning="OPT candidates are not eligible for this role.",
        ),
        "This position is not open to OPT candidates.",
    )

    assert (
        result.requirements.sponsorship
        is SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED
    )
    assert "OPT candidates are not eligible" in result.missing_requirements
    assert "OPT candidates are not eligible" in result.reasoning
    assert result.passes_hard_filters is False
    assert result.recommendation == Recommendation.SKIP


def test_explicit_no_f1_keeps_supported_restriction():
    result = match(
        evaluation_with_prose(
            missing_requirements=["SAS", SUPPORTED_F1],
            reasoning=(
                "The candidate matches most core requirements. "
                "F-1 candidates are not eligible."
            ),
        ),
        "This position is not open to F-1 candidates.",
    )

    assert (
        result.requirements.sponsorship
        is SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED
    )
    assert SUPPORTED_F1 in result.missing_requirements
    assert "SAS" in result.missing_requirements
    assert "F-1 candidates are not eligible." in result.reasoning
    assert result.passes_hard_filters is False
    assert result.recommendation == Recommendation.SKIP


def test_no_h1b_keeps_generic_concern_but_not_f1_prohibition():
    result = match(
        evaluation_with_prose(
            missing_requirements=[INVENTED_F1, SUPPORTED_H1B],
            reasoning=(
                "The employer does not sponsor H-1B. "
                "F-1 candidates are not eligible."
            ),
        ),
        "We are unable to provide H-1B sponsorship for this position.",
    )

    assert result.requirements.sponsorship is SponsorshipStance.NOT_OFFERED
    assert result.passes_hard_filters is True
    assert result.recommendation == Recommendation.APPLY
    assert len(result.concerns) == 1
    assert "does not offer visa sponsorship" in result.concerns[0]
    assert INVENTED_F1 not in result.missing_requirements
    assert SUPPORTED_H1B in result.missing_requirements
    assert "does not sponsor H-1B" in result.reasoning
    assert "F-1" not in result.reasoning


def test_silence_stance_remains_not_mentioned():
    result = match(
        evaluation_with_prose(missing_requirements=["SAS"]),
        "Example Pharma is hiring a Biostatistician. Strong R required.",
    )

    assert result.requirements.sponsorship is SponsorshipStance.NOT_MENTIONED
    assert result.concerns == []
    assert result.passes_hard_filters is True


def test_experience_hard_filter_still_works_independently():
    filter_config = FilterConfig(
        experience=ExperienceFilterConfig(
            enabled=True,
            max_required_years=2,
            tolerance_years=2,
        ),
    )

    result = match(
        evaluation_with_prose(
            missing_requirements=["SAS", INVENTED_F1],
            requirements=make_requirements(minimum_years_experience=7),
        ),
        (
            "TS/SCI w/ CI Poly. 7 years of relevant experience. "
            "Python, Jupyter, and data science required."
        ),
        filter_config=filter_config,
    )

    assert result.requirements.sponsorship is SponsorshipStance.NOT_MENTIONED
    assert result.passes_hard_filters is False
    assert result.recommendation == Recommendation.SKIP
    assert len(result.hard_filter_reasons) == 1
    assert "7+ years of experience" in result.hard_filter_reasons[0]
    assert INVENTED_F1 not in result.missing_requirements
    assert "SAS" in result.missing_requirements


def test_invented_now_or_future_quote_does_not_add_authorization_hard_filter():
    filter_config = FilterConfig(
        experience=ExperienceFilterConfig(
            enabled=True,
            max_required_years=2,
            tolerance_years=2,
        ),
    )

    result = match(
        evaluation_with_prose(
            missing_requirements=["SAS"],
            requirements=make_requirements(
                minimum_years_experience=7,
                sponsorship=SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED,
                sponsorship_language=(
                    "must not now or in the future require sponsorship"
                ),
            ),
        ),
        (
            "TS/SCI w/ CI Poly. 7 years of relevant experience. "
            "Python, Jupyter, and data science required. "
            "Supporting a National Security customer."
        ),
        filter_config=filter_config,
    )

    assert result.requirements.sponsorship is SponsorshipStance.NOT_MENTIONED
    assert result.requirements.sponsorship_language is None
    assert result.passes_hard_filters is False
    assert result.recommendation == Recommendation.SKIP
    assert len(result.hard_filter_reasons) == 1
    assert "7+ years of experience" in result.hard_filter_reasons[0]
    assert "work authorization" not in result.hard_filter_reasons[0]


def test_overall_score_and_recommendation_stay_deterministic():
    silent_jd = "Example Pharma is hiring a Biostatistician. Strong R required."
    invented = match(
        evaluation_with_prose(missing_requirements=["SAS", INVENTED_F1]),
        silent_jd,
    )
    clean = match(
        evaluation_with_prose(missing_requirements=["SAS"]),
        silent_jd,
    )

    assert invented.overall_score == clean.overall_score == 84.5
    assert invented.recommendation == clean.recommendation == Recommendation.APPLY
    assert invented.scores.skills == clean.scores.skills == 80


def test_generic_no_sponsorship_does_not_become_f1_ban():
    result = match(
        evaluation_with_prose(
            missing_requirements=[INVENTED_F1],
            reasoning="Work authorization not permitted for F-1 candidates.",
        ),
        "We do not provide visa sponsorship.",
    )

    assert result.requirements.sponsorship is SponsorshipStance.NOT_OFFERED
    assert result.passes_hard_filters is True
    assert len(result.concerns) == 1
    assert INVENTED_F1 not in result.missing_requirements
    assert result.reasoning
    assert "F-1" not in result.reasoning
    assert "work authorization" not in result.reasoning.lower()


def test_unit_tests_do_not_call_openai(monkeypatch):
    def fail_openai(*_args, **_kwargs):
        raise AssertionError("unit tests must not call the OpenAI API")

    monkeypatch.setattr("job_search_agent.ai_client.AIClient.evaluate_job", fail_openai)

    match(
        evaluation_with_prose(missing_requirements=["SAS"]),
        "Example job description",
    )


def test_claims_in_classifies_observed_f1_invention():
    found = claims_in(INVENTED_F1)

    assert AuthorizationClaim.STUDENT_STATUS_PROHIBITED in found
    assert not is_supported(INVENTED_F1, SponsorshipStance.NOT_MENTIONED)
    assert not is_supported(INVENTED_F1, SponsorshipStance.NOT_OFFERED)
    assert is_supported(INVENTED_F1, SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED)


def test_non_authorization_text_is_never_stripped():
    assert is_supported("SAS", SponsorshipStance.NOT_MENTIONED)
    assert is_supported("Strong R skills", SponsorshipStance.NOT_MENTIONED)
    assert is_supported(
        "The candidate matches most core requirements.",
        SponsorshipStance.NOT_MENTIONED,
    )


def test_silence_drops_permanent_work_authorization_paraphrase():
    result = match(
        evaluation_with_prose(
            missing_requirements=[
                "SAS",
                "Candidate does not meet the permanent work authorization requirement.",
            ],
            reasoning=(
                "The candidate matches Python requirements. "
                "Candidate does not meet the permanent work authorization requirement."
            ),
        ),
        "TS/SCI w/ CI Poly. 7 years of relevant experience. Python required.",
    )

    assert result.requirements.sponsorship is SponsorshipStance.NOT_MENTIONED
    assert "SAS" in result.missing_requirements
    assert all("authorization" not in item.lower() for item in result.missing_requirements)
    assert "authorization" not in result.reasoning.lower()
    assert "Python requirements" in result.reasoning


def test_silence_drops_authorization_needs_paraphrase():
    result = match(
        evaluation_with_prose(
            missing_requirements=["authorization needs"],
            reasoning=(
                "The candidate is underqualified based on the years of "
                "experience and authorization needs."
            ),
        ),
        "Example Pharma is hiring a Biostatistician. Strong R required.",
    )

    assert result.requirements.sponsorship is SponsorshipStance.NOT_MENTIONED
    assert result.missing_requirements == []
    assert result.reasoning
    assert "authorization" not in result.reasoning.lower()


def test_silence_drops_visa_sponsorship_eligibility_paraphrase():
    result = match(
        evaluation_with_prose(
            missing_requirements=["visa sponsorship eligibility"],
            reasoning="Visa sponsorship eligibility is unclear.",
        ),
        "Example Pharma is hiring a Biostatistician. Strong R required.",
    )

    assert result.requirements.sponsorship is SponsorshipStance.NOT_MENTIONED
    assert result.missing_requirements == []
    assert result.reasoning
    assert "visa" not in result.reasoning.lower()
    assert "sponsorship" not in result.reasoning.lower()
    assert "authorization" not in result.reasoning.lower()


def test_silence_keeps_clearance_requirement_prose():
    clearance = (
        "Position requires TS/SCI clearance, which candidate does not possess."
    )
    result = match(
        evaluation_with_prose(
            missing_requirements=[
                clearance,
                "Candidate does not meet the permanent work authorization requirement.",
            ],
            reasoning=(
                f"{clearance} "
                "The role has authorization needs."
            ),
        ),
        "TS/SCI w/ CI Poly. Supporting a National Security customer.",
    )

    assert result.requirements.sponsorship is SponsorshipStance.NOT_MENTIONED
    assert clearance in result.missing_requirements
    assert all("authorization" not in item.lower() for item in result.missing_requirements)
    assert "TS/SCI clearance" in result.reasoning
    assert "authorization" not in result.reasoning.lower()


def test_explicit_citizenship_keeps_supported_prose():
    result = match(
        evaluation_with_prose(
            missing_requirements=["SAS", "U.S. citizenship is required."],
            reasoning=(
                "The candidate matches most core requirements. "
                "U.S. citizenship is required."
            ),
        ),
        "This position is open only to U.S. citizens.",
    )

    assert result.requirements.sponsorship is SponsorshipStance.CITIZENSHIP_REQUIRED
    assert "U.S. citizenship is required." in result.missing_requirements
    assert "SAS" in result.missing_requirements
    assert "U.S. citizenship is required." in result.reasoning
    assert result.passes_hard_filters is False
    assert result.recommendation == Recommendation.SKIP


def test_empty_sanitized_reasoning_gets_deterministic_fallback():
    result = match(
        evaluation_with_prose(
            missing_requirements=["SAS"],
            reasoning="Work authorization not permitted for F-1 candidates.",
        ),
        "Example Pharma is hiring a Biostatistician. Strong R required.",
    )

    assert result.requirements.sponsorship is SponsorshipStance.NOT_MENTIONED
    assert result.reasoning
    assert "relevant skills" in result.reasoning.lower()
    assert "F-1" not in result.reasoning
    assert "authorization" not in result.reasoning.lower()
    assert "sponsorship" not in result.reasoning.lower()
    assert result.overall_score == 84.5
    assert result.recommendation == Recommendation.APPLY


def test_mixed_authorization_reasoning_fallback_uses_experience_hard_filter():
    filter_config = FilterConfig(
        experience=ExperienceFilterConfig(
            enabled=True,
            max_required_years=2,
            tolerance_years=2,
        ),
    )

    result = match(
        evaluation_with_prose(
            missing_requirements=["SAS"],
            reasoning=(
                "The candidate is underqualified based on the years of "
                "experience and authorization needs."
            ),
            requirements=make_requirements(minimum_years_experience=7),
        ),
        (
            "TS/SCI w/ CI Poly. 7 years of relevant experience. "
            "Python, Jupyter, and data science required."
        ),
        filter_config=filter_config,
    )

    assert result.requirements.sponsorship is SponsorshipStance.NOT_MENTIONED
    assert result.passes_hard_filters is False
    assert result.recommendation == Recommendation.SKIP
    assert result.reasoning
    assert "7+" in result.reasoning
    assert "SKIP" in result.reasoning
    assert "authorization" not in result.reasoning.lower()
    assert "sponsorship" not in result.reasoning.lower()
    assert "F-1" not in result.reasoning
    assert len(result.hard_filter_reasons) == 1
    assert result.overall_score == 84.5


def test_valid_llm_reasoning_is_not_replaced_by_fallback():
    original = "The candidate matches most core requirements."
    result = match(
        evaluation_with_prose(
            missing_requirements=["SAS"],
            reasoning=original,
        ),
        "Example Pharma is hiring a Biostatistician. Strong R required.",
    )

    assert result.reasoning == original
    assert "hard-filter" not in result.reasoning
    assert result.recommendation == Recommendation.APPLY
    assert result.overall_score == 84.5

