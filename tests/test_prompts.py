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


def skill_prompt(posting: str = "Proficiency in Python is required.") -> str:
    from job_search_agent.evidence_units import (
        build_evidence_units,
        skill_candidate_units,
    )
    from job_search_agent.prompts import build_skill_inventory_prompt

    units = skill_candidate_units(build_evidence_units(posting))
    return " ".join(build_skill_inventory_prompt(units).split())


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
    text = skill_prompt()

    assert "a concise skill as it would appear on a resume" in text
    assert "Do not copy prose fragments from the posting" in text


def test_skill_qualifiers_are_stripped():
    text = skill_prompt()

    for qualifier in ("strong", "expert-level", "deep knowledge of", "preferred"):
        assert f'"{qualifier}"' in text


def test_skill_extraction_is_a_dedicated_enumeration_task():
    main = prompt("Python is required.")
    dedicated = skill_prompt("Python is required.")

    assert "dedicated skill-inventory extraction step" in main
    assert "Do not enumerate the posting's skill inventory" in main
    assert "u001:" in dedicated
    assert "Do not invent ids" in dedicated
    assert "You enumerate skills" in dedicated
    assert "Do not score education, experience, career relevance, or sponsorship" in dedicated


# --- deliberately unchanged -------------------------------------------------


def test_employment_type_and_seniority_still_default_to_unknown():
    """We chose not to push the model toward inferring these."""
    assert (
        'employment_type and seniority_level: use "unknown" unless the '
        "posting is explicit" in prompt()
    )


def test_extraction_still_abstains_when_in_doubt():
    assert "This is the default and the correct answer when in doubt." in prompt()


def test_null_required_degree_means_no_stated_level():
    text = prompt()

    assert "extract the minimum required degree level" in text
    assert "Use null when the posting does not state a required" in text
    assert "Master's preferred" in text
    assert "equivalent professional experience" in text
    assert "equivalent combination of education and experience" in text
    assert "Do not use null merely because you are uncertain." in text
    assert "Do not raise that level because a higher degree is preferred." in text


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
