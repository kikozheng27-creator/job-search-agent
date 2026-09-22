from conftest import make_profile, make_requirements

from job_search_agent.candidate import Degree
from job_search_agent.education_scorer import (
    UNPARSED_REQUIRED_DEGREE_SCORE,
    normalize_degree_level,
    score_education,
)


def _profile(*degrees: str, field: str = "Biostatistics"):
    return make_profile(
        education=[Degree(degree=degree, field=field) for degree in degrees]
    )


def test_null_required_degree_scores_100():
    assert (
        score_education(
            _profile("Master's"),
            make_requirements(required_degree=None),
        )
        == 100
    )


def test_masters_meets_masters():
    assert (
        score_education(
            _profile("Master's"),
            make_requirements(required_degree="Master's"),
        )
        == 100
    )


def test_masters_exceeds_bachelors():
    assert (
        score_education(
            _profile("Master's"),
            make_requirements(required_degree="Bachelor's"),
        )
        == 100
    )


def test_bachelors_is_one_level_below_masters():
    assert (
        score_education(
            _profile("Bachelor's"),
            make_requirements(required_degree="Master's"),
        )
        == 70
    )


def test_associate_is_two_levels_below_masters():
    assert (
        score_education(
            _profile("associate degree"),
            make_requirements(required_degree="MS"),
        )
        == 40
    )


def test_highest_of_several_degrees_can_meet_the_requirement():
    score = score_education(
        _profile("Bachelor's", "PhD", "associate degree"),
        make_requirements(required_degree="doctorate"),
    )

    assert score == 100


def test_field_of_study_does_not_change_the_score():
    requirements = make_requirements(required_degree="Master's")
    biostatistics = _profile("Master's", field="Biostatistics")
    history = _profile("Master's", field="History")

    assert score_education(biostatistics, requirements) == score_education(
        history, requirements
    )


def test_institution_and_graduation_year_do_not_change_the_score():
    requirements = make_requirements(required_degree="Bachelor's")
    bare = _profile("BS")
    detailed = make_profile(
        education=[
            Degree(
                degree="B.S.",
                field="Statistics",
                institution="Example University",
                graduation_year=2024,
            )
        ]
    )

    assert score_education(bare, requirements) == score_education(
        detailed, requirements
    )
    assert score_education(detailed, requirements) == 100


def test_unparseable_required_degree_uses_the_fallback():
    score = score_education(
        _profile("Master's"),
        make_requirements(required_degree="a quantitative discipline or equivalent"),
    )

    assert score == UNPARSED_REQUIRED_DEGREE_SCORE
    assert score == 0


def test_explicit_alternatives_use_the_lowest_required_level():
    assert normalize_degree_level("Bachelor's or Master's") == "bachelor"
    assert (
        score_education(
            _profile("Bachelor's"),
            make_requirements(required_degree="Bachelor's or Master's"),
        )
        == 100
    )


def test_certifications_are_not_degrees():
    certificate = "Google Data Analytics Professional Certificate"
    assert normalize_degree_level(certificate) is None
    assert (
        score_education(
            _profile(certificate),
            make_requirements(required_degree="Master's"),
        )
        == 0
    )
    assert (
        score_education(
            _profile("Master's"),
            make_requirements(required_degree=certificate),
        )
        == UNPARSED_REQUIRED_DEGREE_SCORE
    )


def test_conservative_aliases_normalize_to_one_level():
    expected = {
        "high school": "high_school",
        "high-school": "high_school",
        "associate degree": "associate",
        "associate's": "associate",
        "AA": "associate",
        "A.S.": "associate",
        "Bachelor's": "bachelor",
        "bachelors": "bachelor",
        "BS": "bachelor",
        "B.S.": "bachelor",
        "BA": "bachelor",
        "B.A.": "bachelor",
        "Master's": "master",
        "masters": "master",
        "MS": "master",
        "M.S.": "master",
        "MA": "master",
        "M.A.": "master",
        "PhD": "doctorate",
        "Ph.D.": "doctorate",
        "doctorate": "doctorate",
        "doctoral": "doctorate",
    }

    for text, level in expected.items():
        assert normalize_degree_level(text) == level, text


def test_lowercase_as_is_not_an_associate_degree():
    assert normalize_degree_level("experience as a analyst") is None


def test_high_school_diploma_is_high_school():
    assert normalize_degree_level("high school diploma") == "high_school"
    assert normalize_degree_level("secondary school") == "high_school"
    assert normalize_degree_level("secondary school diploma") == "high_school"


def test_bare_diploma_and_graduate_diploma_are_unparsed():
    assert normalize_degree_level("diploma") is None
    assert normalize_degree_level("graduate diploma") is None


def test_associate_degree_phrase_is_associate():
    assert normalize_degree_level("associate degree") == "associate"
    assert normalize_degree_level("associates degree") == "associate"


def test_associate_job_title_is_unparsed():
    assert normalize_degree_level("Associate Big Data Engineer") is None


def test_required_bachelors_is_a_required_degree():
    requirements = make_requirements(required_degree="Bachelor's degree required")

    assert normalize_degree_level(requirements.required_degree) == "bachelor"
    assert score_education(_profile("Bachelor's"), requirements) == 100
    assert score_education(_profile(), requirements) == 0


def test_preferred_degree_alone_is_not_required():
    assert (
        score_education(
            _profile(),
            make_requirements(required_degree="Master's preferred"),
        )
        == 100
    )


def test_required_bachelors_ignores_preferred_masters():
    requirements = make_requirements(
        required_degree="Bachelor's required; Master's preferred"
    )

    assert normalize_degree_level(requirements.required_degree) == "bachelor"
    assert score_education(_profile("Bachelor's"), requirements) == 100
    assert score_education(_profile(), requirements) == 0


def test_equivalent_experience_is_not_a_required_degree():
    for wording in (
        "Bachelor's degree or equivalent experience",
        "Bachelor's degree or equivalent professional experience",
        "Bachelor's degree, or equivalent combination of education and experience",
    ):
        assert (
            score_education(
                _profile(),
                make_requirements(required_degree=wording),
            )
            == 100
        ), wording


def test_equivalent_experience_in_another_clause_does_not_cancel_a_requirement():
    preferred = make_requirements(
        required_degree="Bachelor's degree required; equivalent experience preferred"
    )
    later_master = make_requirements(
        required_degree=(
            "Bachelor's degree or equivalent experience; Master's degree required"
        )
    )

    assert normalize_degree_level(preferred.required_degree) == "bachelor"
    assert score_education(_profile("Bachelor's"), preferred) == 100
    assert score_education(_profile(), preferred) == 0

    assert normalize_degree_level(later_master.required_degree) == "master"
    assert score_education(_profile("Master's"), later_master) == 100
    assert score_education(_profile("Bachelor's"), later_master) == 70
    assert score_education(_profile(), later_master) == 0


def test_or_higher_and_preferred_do_not_raise_the_minimum():
    assert normalize_degree_level("Bachelor's degree or higher") == "bachelor"
    assert normalize_degree_level("Bachelor's or higher") == "bachelor"
    assert (
        normalize_degree_level("Bachelor's required; Master's preferred")
        == "bachelor"
    )
    assert normalize_degree_level("Master's required") == "master"
