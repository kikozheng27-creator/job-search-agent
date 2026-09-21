from job_search_agent.evidence_units import (
    build_evidence_units,
    is_skill_candidate_unit,
)


REQUIRED_PREFERRED_POSTING = """
Data Scientist

Required Experience and Qualifications:
- Proficiency in Python is required
- Experience with SQL

Preferred Qualifications:
- SAS is a plus
- Tableau desired
"""


def test_same_job_description_produces_identical_evidence_units():
    first = build_evidence_units(REQUIRED_PREFERRED_POSTING)
    second = build_evidence_units(REQUIRED_PREFERRED_POSTING)

    assert [(unit.id, unit.text) for unit in first] == [
        (unit.id, unit.text) for unit in second
    ]
    assert [unit.id for unit in first] == [
        f"u{index:03d}" for index in range(1, len(first) + 1)
    ]


def test_whitespace_normalization_does_not_change_unit_identity():
    messy = (
        "Required Experience and Qualifications:\n"
        "   -    Proficiency   in   Python   is   required\n"
    )
    clean = (
        "Required Experience and Qualifications:\n"
        "- Proficiency in Python is required\n"
    )

    assert [unit.text for unit in build_evidence_units(messy)] == [
        unit.text for unit in build_evidence_units(clean)
    ]


def test_bullet_structure_preserves_required_versus_preferred_modality():
    units = build_evidence_units(REQUIRED_PREFERRED_POSTING)
    python = next(unit for unit in units if "Python" in unit.text)
    sas = next(unit for unit in units if "SAS" in unit.text)

    assert python.text.startswith("Required Experience and Qualifications:")
    assert "preferred" not in python.text.casefold()
    assert sas.text.startswith("Preferred Qualifications:")
    assert "plus" in sas.text.casefold()


def test_multi_sentence_line_splits_so_modality_is_not_mixed():
    units = build_evidence_units(
        "Proficiency in Python is required. SAS is preferred."
    )

    assert len(units) == 2
    assert units[0].id == "u001"
    assert units[0].text == "Proficiency in Python is required."
    assert units[1].id == "u002"
    assert units[1].text == "SAS is preferred."


def test_abbreviation_does_not_split_a_sentence():
    units = build_evidence_units("U.S. citizenship is required for this role.")

    assert len(units) == 1
    assert "U.S. citizenship is required" in units[0].text


def test_empty_lines_are_not_units():
    units = build_evidence_units("Python is required.\n\n\nR is preferred.\n")

    assert [unit.id for unit in units] == ["u001", "u002"]


def test_hard_wrapped_fragments_merge_into_stable_sentences():
    wrapped = (
        "This is a\n"
        "proposed\n"
        "position, we are offering a salary of\n"
        "$160,000\n"
        "with a\n"
        "$2,000 sign-on bonus, or reimbursable relocation\n"
        ".\n"
    )
    units = build_evidence_units(wrapped)

    assert len(units) == 1
    assert units[0].id == "u001"
    assert units[0].text == (
        "This is a proposed position, we are offering a salary of "
        "$160,000 with a $2,000 sign-on bonus, or reimbursable relocation."
    )


def test_heading_does_not_prefix_a_following_title_section():
    posting = (
        "Required Experience and Qualifications:\n"
        "Demonstrate expert knowledge of Python.\n"
        "Create a Job Alert\n"
        "Apply for this job\n"
    )
    units = build_evidence_units(posting)
    python = next(unit for unit in units if "Python" in unit.text)
    alert = next(unit for unit in units if "Job Alert" in unit.text)

    assert python.text.startswith("Required Experience and Qualifications:")
    assert not alert.text.startswith("Required Experience and Qualifications:")


def test_skill_candidate_units_exclude_chrome_and_keep_requirements():
    posting = (
        "Back to jobs\n"
        "Colorado Springs, CO\n"
        "Required Experience and Qualifications:\n"
        "Demonstrate expert knowledge of Python.\n"
    )
    units = build_evidence_units(posting)
    candidates = [unit.id for unit in units if is_skill_candidate_unit(unit)]
    python = next(unit for unit in units if "Python" in unit.text)

    assert python.id in candidates
    assert not any("Back to jobs" in unit.text for unit in units if is_skill_candidate_unit(unit))
