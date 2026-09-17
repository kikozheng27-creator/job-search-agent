"""Regression tests for prompt instructions changed after the live smoke test.

The live run surfaced two scoring/extraction defects and one classification
gap. These tests pin the instructions that fix them, and pin the behaviors we
deliberately chose *not* to change.

Assertions run against a whitespace-collapsed prompt so they survive the text
being re-wrapped.
"""

from conftest import make_profile

from job_search_agent.prompts import build_evaluation_prompt


def prompt(posting: str = "Some job posting text") -> str:
    return " ".join(build_evaluation_prompt(make_profile(), posting).split())


# --- experience scoring -----------------------------------------------------
# Live run: a candidate with 0.5 years scored 25/100 against a "0-2 years"
# posting, dragging a strong match down to MAYBE.


def test_meeting_the_minimum_earns_a_high_experience_score():
    text = prompt()

    assert "meets or exceeds the stated minimum scores at least 85" in text
    assert "do not deduct for having less than the top of a range" in text


def test_the_live_failure_case_is_called_out_explicitly():
    assert '0.5 years meets a "0-2 years" requirement in full' in prompt()


def test_relevance_still_differentiates_within_the_high_band():
    assert "85-100 band to reflect how relevant" in prompt()


def test_shortfalls_are_scaled_rather_than_flat():
    text = prompt()

    assert "reserve scores below 85 for candidates genuinely short" in text
    assert "scaled by how far short they fall" in text


def test_absent_minimum_does_not_imply_a_requirement():
    assert "posting states no minimum, judge relevance alone" in prompt()


# --- skill normalization ----------------------------------------------------
# Live run: skills came back as prose fragments such as "Strong R programming
# skills" and "SAS preferred".


def test_skills_must_be_normalized_to_concise_items():
    text = prompt()

    assert "Normalize every skill to a concise skill, tool, or competency" in text
    assert "Do not copy prose fragments from the posting" in text


def test_skill_qualifiers_are_stripped():
    text = prompt()

    for qualifier in ("strong", "expert-level", "deep knowledge of", "preferred"):
        assert f'"{qualifier}"' in text


def test_multi_skill_bullets_are_split():
    assert "one entry per skill" in prompt()


# --- deliberately unchanged -------------------------------------------------


def test_employment_type_and_seniority_still_default_to_unknown():
    """We chose not to push the model toward inferring these."""
    assert (
        'employment_type and seniority_level: use "unknown" unless the '
        "posting is explicit" in prompt()
    )


def test_extraction_still_abstains_when_in_doubt():
    assert "This is the default and the correct answer when in doubt." in prompt()


# --- sponsorship classification ---------------------------------------------


def test_all_five_sponsorship_stances_are_documented():
    text = prompt()

    for stance in (
        "not_mentioned",
        "offered",
        "not_offered",
        "permanent_authorization_required",
        "citizenship_required",
    ):
        assert f'"{stance}"' in text


def test_disqualifying_phrasings_are_given_as_examples():
    text = prompt()

    for phrasing in (
        "no F-1 candidates",
        "no OPT or STEM OPT candidates",
        "permanent unrestricted work authorization required",
        "must not now or in the future require sponsorship",
    ):
        assert phrasing in text


def test_generic_no_sponsorship_phrasings_map_to_not_offered():
    text = prompt()

    for phrasing in (
        "we are unable to provide visa sponsorship",
        "no H-1B sponsorship",
        "sponsorship is not available for this role",
    ):
        assert phrasing in text


def test_a_stance_requires_a_quotable_sentence():
    """Live run: the model claimed not_offered with nothing to quote."""
    text = prompt()

    assert (
        'Any answer other than "not_mentioned" requires a sentence you can '
        "quote verbatim in sponsorship_language" in text
    )
    assert "Never infer a stance from the absence of such a sentence." in text


def test_the_two_sponsorship_cases_are_explicitly_contrasted():
    text = prompt()

    assert "the first only declines future sponsorship" in text
    assert "rejects a candidate whose authorization is temporary" in text


# --- unchanged guarantees ---------------------------------------------------


def test_prompt_still_forbids_the_model_scoring_or_deciding():
    text = prompt()

    assert "Do not compute an overall score." in text
    assert "Do not make an apply or skip recommendation." in text


def test_prompt_contains_the_profile_and_the_posting():
    text = prompt("Unique posting body")

    assert "Unique posting body" in text
    assert "Biostatistics" in text
