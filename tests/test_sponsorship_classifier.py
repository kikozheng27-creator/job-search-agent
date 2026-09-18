from pathlib import Path

from conftest import (
    DEFAULT_SCORING,
    PERMISSIVE_FILTERS,
    FakeAIClient,
    make_evaluation,
    make_profile,
    make_requirements,
)
from job_search_agent.candidate import WorkAuthorization
from job_search_agent.concerns import collect_concerns
from job_search_agent.config_models import ExperienceFilterConfig, FilterConfig
from job_search_agent.filters import check_hard_filters
from job_search_agent.job_matcher import JobMatcher
from job_search_agent.models import Recommendation, SponsorshipStance
from job_search_agent.sponsorship_classifier import (
    apply_sponsorship_classification,
    classify_sentence,
    classify_sponsorship,
)

FIXTURES = Path(__file__).resolve().parents[1] / "jobs" / "authorization"

FIXTURE_STANCES = {
    "01_no_f1.txt": SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED,
    "02_no_opt.txt": SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED,
    "03_no_stem_opt.txt": SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED,
    "04_permanent_unrestricted.txt": (
        SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED
    ),
    "05_citizens_or_pr_only.txt": SponsorshipStance.CITIZENSHIP_REQUIRED,
    "06_no_h1b.txt": SponsorshipStance.NOT_OFFERED,
    "07_generic_no_sponsorship.txt": SponsorshipStance.NOT_OFFERED,
    "08_no_sponsorship_now_or_future.txt": (
        SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED
    ),
    "09_ambiguous.txt": SponsorshipStance.NOT_MENTIONED,
    "10_silence.txt": SponsorshipStance.NOT_MENTIONED,
}


def opt_profile():
    return make_profile(
        work_authorization=WorkAuthorization(
            status="F-1 student; eligible for OPT/STEM OPT",
            requires_sponsorship=False,
            is_citizen_or_permanent_resident=False,
            requires_future_sponsorship=True,
        ),
    )


def opt_policy(requirements):
    config = FilterConfig(
        experience=ExperienceFilterConfig(
            max_required_years=2,
            tolerance_years=2,
        ),
    )
    profile = opt_profile()

    return (
        check_hard_filters(requirements, profile, config),
        collect_concerns(requirements, profile),
    )


def read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_all_authorization_fixtures_classify_as_intended():
    for filename, expected in FIXTURE_STANCES.items():
        posting = read_fixture(filename)
        assert classify_sponsorship(posting) is expected, filename


def test_fixture_classification_does_not_depend_on_llm_stance():
    """Python overwrites the model even when the quote is attached to a wrong stance."""
    posting = read_fixture("01_no_f1.txt")
    requirements = make_requirements(
        minimum_years_experience=1,
        sponsorship=SponsorshipStance.NOT_OFFERED,
        sponsorship_language="This position is not open to F-1 candidates.",
    )

    apply_sponsorship_classification(requirements, posting)

    assert (
        requirements.sponsorship
        is SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED
    )


# --- negative controls ------------------------------------------------------


def test_authorized_to_work_in_the_united_states_is_not_mentioned():
    sentence = "Applicants must be authorized to work in the United States."

    assert classify_sentence(sentence) is None
    assert classify_sponsorship(sentence) is SponsorshipStance.NOT_MENTIONED


def test_no_h1b_is_non_blocking_for_an_opt_candidate():
    posting = read_fixture("06_no_h1b.txt")
    requirements = make_requirements(
        minimum_years_experience=1,
        sponsorship=classify_sponsorship(posting),
        sponsorship_language=(
            "We are unable to provide H-1B sponsorship for this position."
        ),
    )
    reasons, concerns = opt_policy(requirements)

    assert requirements.sponsorship is SponsorshipStance.NOT_OFFERED
    assert reasons == []
    assert len(concerns) == 1


def test_generic_no_sponsorship_is_non_blocking_with_a_concern():
    posting = read_fixture("07_generic_no_sponsorship.txt")
    requirements = make_requirements(
        minimum_years_experience=1,
        sponsorship=classify_sponsorship(posting),
        sponsorship_language="We do not provide visa sponsorship.",
    )
    reasons, concerns = opt_policy(requirements)

    assert requirements.sponsorship is SponsorshipStance.NOT_OFFERED
    assert reasons == []
    assert len(concerns) == 1


def test_silence_is_non_blocking_with_no_concern():
    posting = read_fixture("10_silence.txt")
    requirements = make_requirements(
        minimum_years_experience=1,
        sponsorship=classify_sponsorship(posting),
        sponsorship_language=None,
    )
    reasons, concerns = opt_policy(requirements)

    assert requirements.sponsorship is SponsorshipStance.NOT_MENTIONED
    assert reasons == []
    assert concerns == []


def test_citizen_or_pr_wording_is_not_a_generic_permanent_match():
    sentence = (
        "This position is open to U.S. citizens or permanent residents only."
    )

    assert classify_sentence(sentence) is SponsorshipStance.CITIZENSHIP_REQUIRED
    assert "permanent unrestricted" not in sentence.casefold()


def test_isolated_permanent_does_not_classify():
    sentence = "This is a permanent, full-time biostatistician role."

    assert classify_sponsorship(sentence) is SponsorshipStance.NOT_MENTIONED


def test_isolated_opt_does_not_classify():
    sentence = "The team uses OPT models for dose-finding."

    assert classify_sponsorship(sentence) is SponsorshipStance.NOT_MENTIONED


def test_isolated_sponsorship_does_not_classify():
    sentence = "We discuss professional development and conference sponsorship."

    assert classify_sponsorship(sentence) is SponsorshipStance.NOT_MENTIONED


# --- precedence -------------------------------------------------------------


def test_f1_rejection_outranks_generic_no_sponsorship():
    posting = (
        "We do not provide visa sponsorship.\n"
        "This position is not open to F-1 candidates."
    )

    assert (
        classify_sponsorship(posting)
        is SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED
    )


def test_citizenship_outranks_generic_no_sponsorship():
    posting = (
        "We do not provide visa sponsorship.\n"
        "This position is open to U.S. citizens or permanent residents only."
    )

    assert (
        classify_sponsorship(posting)
        is SponsorshipStance.CITIZENSHIP_REQUIRED
    )


def test_now_or_future_is_not_treated_as_generic_no_sponsorship():
    sentence = (
        "Candidates must not now or in the future require visa sponsorship."
    )

    assert (
        classify_sentence(sentence)
        is SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED
    )


# --- unquoted safeguard still applies before Python classifies -------------


def test_unquoted_llm_stance_does_not_survive_silence_in_the_posting():
    requirements = make_requirements(
        sponsorship=SponsorshipStance.NOT_OFFERED,
        sponsorship_language=None,
    )

    apply_sponsorship_classification(requirements, read_fixture("10_silence.txt"))

    assert requirements.sponsorship is SponsorshipStance.NOT_MENTIONED


def test_classifier_does_not_change_component_scores_in_the_matcher():
    evaluation = make_evaluation(
        requirements=make_requirements(
            minimum_years_experience=1,
            sponsorship=SponsorshipStance.NOT_OFFERED,
            sponsorship_language="This position is not open to F-1 candidates.",
        ),
    )

    matcher = JobMatcher(
        ai_client=FakeAIClient(evaluation),
        scoring_config=DEFAULT_SCORING,
        filter_config=PERMISSIVE_FILTERS,
    )

    result = matcher.match(
        profile=opt_profile(),
        job_description=read_fixture("01_no_f1.txt"),
    )

    assert result.overall_score == 84.5
    assert result.scores.skills == 80
    assert result.recommendation == Recommendation.SKIP
    assert result.passes_hard_filters is False
