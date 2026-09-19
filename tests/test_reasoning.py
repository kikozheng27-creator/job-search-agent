from conftest import make_requirements, make_scores
from job_search_agent.models import Recommendation, SponsorshipStance
from job_search_agent.reasoning import build_fallback_reasoning


def test_fallback_uses_experience_hard_filter_facts():
    text = build_fallback_reasoning(
        requirements=make_requirements(minimum_years_experience=7),
        scores=make_scores(skills=80),
        recommendation=Recommendation.SKIP,
        hard_filter_reasons=[
            "Requires 7+ years of experience, which is more than "
            "2 year(s) above the target maximum of 2."
        ],
        missing_requirements=["SAS"],
    )

    assert "relevant skills" in text.lower()
    assert "7+" in text
    assert "SKIP" in text
    assert "authorization" not in text.lower()


def test_fallback_with_silence_mentions_no_authorization():
    text = build_fallback_reasoning(
        requirements=make_requirements(
            sponsorship=SponsorshipStance.NOT_MENTIONED,
            sponsorship_language=None,
        ),
        scores=make_scores(skills=80),
        recommendation=Recommendation.APPLY,
        hard_filter_reasons=[],
        missing_requirements=["SAS"],
    )

    assert text
    assert "authorization" not in text.lower()
    assert "sponsorship" not in text.lower()
    assert "F-1" not in text
    assert "visa" not in text.lower()
    assert "APPLY" in text
