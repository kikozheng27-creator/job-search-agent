from conftest import (
    PERMISSIVE_FILTERS,
    FakeAIClient,
    make_evaluation,
    make_profile,
    make_requirements,
    make_scores,
)
from job_search_agent.candidate import WorkAuthorization
from job_search_agent.config_models import (
    ExperienceFilterConfig,
    FilterConfig,
    ScoringConfig,
    ScoringWeights,
    SkillsScoringConfig,
    Thresholds,
)
from job_search_agent.evidence_units import build_evidence_units
from job_search_agent.job_matcher import JobMatcher
from job_search_agent.models import (
    Recommendation,
    SkillClaim,
    SkillLevel,
    SponsorshipStance,
    UnitSkillDecision,
    UnitSkillItem,
)
from job_search_agent.skill_grounding import (
    apply_skill_grounding,
    classify_skill_level,
    ground_skill_claims,
    ground_unit_decisions,
)
from job_search_agent.skill_scorer import is_non_skill_requirement, score_skills


DEFAULT_SKILLS = SkillsScoringConfig(
    required_weight=0.80,
    preferred_weight=0.20,
    aliases={"r programming": "R", "python programming": "Python"},
)


def claim(
    name: str,
    evidence: str,
    level: SkillLevel = SkillLevel.REQUIRED,
    source_unit_ids: list[str] | None = None,
    category: str = "technical_skill",
) -> SkillClaim:
    return SkillClaim(
        name=name,
        level=level,
        evidence=evidence,
        source_unit_ids=source_unit_ids or [],
        category=category,
    )


def decision(
    unit_id: str,
    names: list[str],
    is_candidate_requirement: bool = True,
    category: str = "technical_skill",
) -> UnitSkillDecision:
    return UnitSkillDecision(
        source_unit_id=unit_id,
        is_candidate_requirement=is_candidate_requirement,
        skills=[UnitSkillItem(name=name, category=category) for name in names],
    )


def unit_containing(units, snippet: str):
    matches = [unit for unit in units if snippet.casefold() in unit.text.casefold()]
    assert matches, f"no evidence unit contains {snippet!r}"
    return matches[-1]


def matcher(evaluation) -> JobMatcher:
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


def test_grounded_required_python_survives():
    posting = "Proficiency in Python is required. SAS preferred."
    required, preferred = ground_skill_claims(
        [claim("Python", "Proficiency in Python is required.")],
        posting,
    )

    assert required == ["Python"]
    assert preferred == []


def test_grounded_preferred_skill_survives():
    posting = "Proficiency in Python is required. SAS is preferred."
    required, preferred = ground_skill_claims(
        [claim("SAS", "SAS is preferred.", SkillLevel.REQUIRED)],
        posting,
    )

    assert required == []
    assert preferred == ["SAS"]


def test_paraphrased_evidence_is_rejected_without_a_source_unit():
    posting = "Proficiency in Python is required."
    required, preferred = ground_skill_claims(
        [claim("Python", "The candidate must know Python programming.")],
        posting,
    )

    assert required == []
    assert preferred == []


def test_invented_skill_evidence_is_rejected():
    posting = "Proficiency in Python is required."
    required, preferred = ground_skill_claims(
        [
            claim("Python", "Proficiency in Python is required."),
            claim("Kubernetes", "Kubernetes experience is required."),
        ],
        posting,
    )

    assert required == ["Python"]
    assert preferred == []


def test_evidence_that_does_not_mention_the_skill_is_rejected():
    posting = "Proficiency in Python is required. We offer relocation."
    required, preferred = ground_skill_claims(
        [claim("Kubernetes", "Proficiency in Python is required.")],
        posting,
    )

    assert required == []
    assert preferred == []


def test_llm_required_label_cannot_override_preferred_evidence():
    posting = "SAS is preferred for this biostatistician role."
    required, preferred = ground_skill_claims(
        [
            claim(
                "SAS",
                "SAS is preferred for this biostatistician role.",
                SkillLevel.REQUIRED,
            )
        ],
        posting,
    )

    assert required == []
    assert preferred == ["SAS"]
    assert classify_skill_level("SAS is preferred") is SkillLevel.PREFERRED


def test_preferred_evidence_cannot_become_required():
    posting = "R experience is required. Knowledge of SAS is a plus."
    required, preferred = ground_skill_claims(
        [
            claim("R", "R experience is required."),
            claim("SAS", "Knowledge of SAS is a plus.", SkillLevel.REQUIRED),
        ],
        posting,
    )

    assert required == ["R"]
    assert preferred == ["SAS"]


def test_non_skill_categories_are_excluded():
    posting = (
        "TS/SCI w/ CI Poly. Master's degree required. "
        "7 years of relevant experience. "
        "Google Data Analytics Professional Certificate recommended. "
        "Complete required course NSA NETA1400. "
        "Proficiency in Python is required."
    )
    required, preferred = ground_skill_claims(
        [
            claim("TS/SCI", "TS/SCI w/ CI Poly"),
            claim("CI Poly", "TS/SCI w/ CI Poly"),
            claim("Master's degree", "Master's degree required."),
            claim("7 years of experience", "7 years of relevant experience."),
            claim(
                "Google Data Analytics Professional Certificate",
                "Google Data Analytics Professional Certificate recommended.",
            ),
            claim("NSA NETA1400", "Complete required course NSA NETA1400."),
            claim("Python", "Proficiency in Python is required."),
        ],
        posting,
    )

    assert required == ["Python"]
    assert preferred == []
    assert is_non_skill_requirement("TS/SCI")
    assert is_non_skill_requirement("CI Poly")
    assert is_non_skill_requirement(
        "Google Data Analytics Professional Certificate"
    )
    assert is_non_skill_requirement("NSA NETA1400")


def test_aliases_and_formatting_do_not_split_the_same_skill():
    posting = (
        "Demonstrate expert knowledge of Python, and Jupyter Notebooks "
        "and/or JupyterLabs. Experience with machine learning and big-data."
    )
    bullet = (
        "Demonstrate expert knowledge of Python, and Jupyter Notebooks "
        "and/or JupyterLabs."
    )
    ml = "Experience with machine learning and big-data."
    required, preferred = ground_skill_claims(
        [
            claim("Python", bullet),
            claim("Jupyter Notebooks", bullet),
            claim("JupyterLabs", bullet),
            claim("Machine Learning", ml),
            claim("machine learning", ml),
            claim("big-data", ml),
            claim("big data", ml),
        ],
        posting,
    )

    canonical = {
        name.casefold().replace(" ", "").replace("-", "") for name in required
    }
    assert "python" in canonical
    assert "jupyternotebooks" in canonical
    assert "jupyterlabs" in canonical
    assert "machinelearning" in canonical
    assert "bigdata" in canonical
    assert preferred == []
    assert len(required) == 5


def test_scoring_formula_unchanged_for_grounded_lists():
    requirements = make_requirements(
        required_skills=["R", "Python"],
        preferred_skills=["SAS"],
    )

    assert score_skills(make_profile(), requirements, DEFAULT_SKILLS) == 80


def test_matcher_uses_grounded_claims_not_raw_lists():
    posting = "Proficiency in Python is required. SAS is preferred."
    evaluation = make_evaluation(
        requirements=make_requirements(
            required_skills=["Kubernetes", "Python"],
            preferred_skills=[],
        ),
        skill_claims=[
            claim("Python", "Proficiency in Python is required."),
            claim("SAS", "SAS is preferred.", SkillLevel.REQUIRED),
            claim("Kubernetes", "Kubernetes is required."),
        ],
    )

    result = matcher(evaluation).match(make_profile(), posting)

    assert result.requirements.required_skills == ["Python"]
    assert result.requirements.preferred_skills == ["SAS"]
    assert "Kubernetes" not in result.requirements.required_skills
    assert result.scores.skills == 80


def test_failed_claims_do_not_use_a_global_name_fallback():
    posting = "Proficiency in Python is required."
    requirements = make_requirements(
        required_skills=["Python", "Kubernetes"],
        preferred_skills=["SAS"],
    )
    apply_skill_grounding(
        requirements,
        [claim("Kubernetes", "Kubernetes is mandatory.")],
        posting,
    )

    assert requirements.required_skills == []
    assert requirements.preferred_skills == []


def test_empty_skill_claims_leave_legacy_lists_alone():
    requirements = make_requirements(
        required_skills=["R"],
        preferred_skills=["SAS"],
    )
    apply_skill_grounding(requirements, [], "Python is required.")

    assert requirements.required_skills == ["R"]
    assert requirements.preferred_skills == ["SAS"]


def test_jupyter_formatting_variants_match_in_scorer():
    profile = make_profile(
        skills={"programming": ["Jupyter Labs", "Python"], "statistics": [], "tools": []},
    )
    with_spaces = make_requirements(
        required_skills=["JupyterLabs"],
        preferred_skills=[],
    )
    concatenated = make_requirements(
        required_skills=["Jupyter Labs"],
        preferred_skills=[],
    )

    assert score_skills(profile, with_spaces, DEFAULT_SKILLS) == 100
    assert score_skills(profile, concatenated, DEFAULT_SKILLS) == 100


def test_valid_source_unit_reference_grounds_a_required_skill():
    posting = "Proficiency in Python is required."
    units = build_evidence_units(posting)
    python = unit_containing(units, "Python")

    required, preferred = ground_unit_decisions(
        [decision(python.id, ["Python"])],
        units,
    )

    assert required == ["Python"]
    assert preferred == []


def test_explicit_preferred_wording_creates_preferred_skill():
    posting = (
        "Required qualifications:\n"
        "- Proficiency in Python is required\n"
        "Preferred qualifications:\n"
        "- SAS is desired\n"
    )
    units = build_evidence_units(posting)
    sas = unit_containing(units, "SAS")

    required, preferred = ground_unit_decisions(
        [decision(sas.id, ["SAS"])],
        units,
    )

    assert required == []
    assert preferred == ["SAS"]


def test_invalid_source_unit_id_is_rejected():
    posting = "Proficiency in Python is required."
    units = build_evidence_units(posting)

    required, preferred = ground_unit_decisions(
        [decision("u999", ["Python"])],
        units,
    )

    assert required == []
    assert preferred == []


def test_skill_absent_from_referenced_unit_is_rejected():
    posting = "Proficiency in Python is required.\nExperience with SQL is required."
    units = build_evidence_units(posting)
    python = unit_containing(units, "Python")

    required, preferred = ground_unit_decisions(
        [decision(python.id, ["SQL"])],
        units,
    )

    assert required == []
    assert preferred == []


def test_skill_elsewhere_in_jd_is_not_accepted_from_the_wrong_unit():
    posting = (
        "Our office is in Denver and we mention Kubernetes in passing.\n"
        "Proficiency in Python is required."
    )
    units = build_evidence_units(posting)
    denver = unit_containing(units, "Denver")
    python = unit_containing(units, "Python")

    required, preferred = ground_unit_decisions(
        [decision(denver.id, ["Python"])],
        units,
    )
    assert required == []
    assert preferred == []

    required, preferred = ground_unit_decisions(
        [decision(python.id, ["Python"])],
        units,
    )
    assert required == ["Python"]


def test_contextual_technology_mention_is_not_a_requirement():
    posting = "Our team uses Python to analyze logs. You will work with analysts using R."
    units = build_evidence_units(posting)
    python = unit_containing(units, "Python")
    r_unit = unit_containing(units, " analysts using R")

    required, preferred = ground_unit_decisions(
        [
            decision(python.id, ["Python"]),
            decision(r_unit.id, ["R"]),
        ],
        units,
    )

    assert required == []
    assert preferred == []


def test_multiple_skills_in_one_requirement_unit_all_survive():
    posting = "Proficiency in Python and Jupyter Notebooks is required."
    units = build_evidence_units(posting)
    unit = unit_containing(units, "Python")

    required, preferred = ground_unit_decisions(
        [decision(unit.id, ["Python", "Jupyter Notebooks"])],
        units,
    )

    assert required == ["Python", "Jupyter Notebooks"]
    assert preferred == []


def test_same_skill_from_multiple_units_deduplicates_deterministically():
    posting = (
        "Proficiency in Python is required.\n"
        "The candidate must also demonstrate knowledge of Python."
    )
    units = build_evidence_units(posting)
    first = units[0]
    second = units[1]

    required, preferred = ground_unit_decisions(
        [
            decision(second.id, ["python"]),
            decision(first.id, ["Python"]),
        ],
        units,
    )

    assert required == ["Python"]
    assert preferred == []


def test_required_beats_preferred_when_both_are_supported():
    posting = (
        "Required qualifications:\n"
        "- Experience with Python is required\n"
        "Preferred qualifications:\n"
        "- Python is a plus\n"
    )
    units = build_evidence_units(posting)
    required_unit = unit_containing(units, "Experience with Python")
    preferred_unit = unit_containing(units, "Python is a plus")

    required, preferred = ground_unit_decisions(
        [
            decision(preferred_unit.id, ["Python"]),
            decision(required_unit.id, ["Python"]),
        ],
        units,
    )

    assert required == ["Python"]
    assert preferred == []


def test_non_skill_category_is_dropped_from_a_valid_unit():
    posting = "Complete required course NSA NETA1400. Proficiency in Python is required."
    units = build_evidence_units(posting)
    python = unit_containing(units, "Python")

    required, preferred = ground_unit_decisions(
        [
            decision(python.id, ["Python"], category="certification"),
            decision(python.id, ["Python"], category="technical_skill"),
        ],
        units,
    )

    assert required == ["Python"]
    assert preferred == []


def test_requirement_unit_keeps_listed_skills_even_if_flag_is_false():
    posting = "Demonstrate expert knowledge of Python and Jupyter Notebooks."
    units = build_evidence_units(posting)
    unit = unit_containing(units, "Python")

    required, preferred = ground_unit_decisions(
        [decision(unit.id, ["Python", "Jupyter Notebooks"], is_candidate_requirement=False)],
        units,
    )

    assert required == ["Python", "Jupyter Notebooks"]
    assert preferred == []


def test_jupyter_survives_when_the_unit_requires_proficiency():
    posting = "Demonstrate expert knowledge of Python and Jupyter Notebooks."
    units = build_evidence_units(posting)
    unit = unit_containing(units, "Jupyter")

    required, preferred = ground_unit_decisions(
        [decision(unit.id, ["Python", "Jupyter Notebooks"])],
        units,
    )

    assert required == ["Python", "Jupyter Notebooks"]


def test_matcher_uses_unit_decisions_over_raw_lists_and_claims():
    posting = "Proficiency in Python is required. SAS is preferred."
    units = build_evidence_units(posting)
    python = unit_containing(units, "Python")
    sas = unit_containing(units, "SAS")
    evaluation = make_evaluation(
        requirements=make_requirements(
            required_skills=["Kubernetes"],
            preferred_skills=[],
        ),
        skill_claims=[claim("Kubernetes", "Kubernetes is required.")],
        skill_unit_decisions=[
            decision(python.id, ["Python"]),
            decision(sas.id, ["SAS"]),
        ],
    )

    result = matcher(evaluation).match(make_profile(), posting)

    assert result.requirements.required_skills == ["Python"]
    assert result.requirements.preferred_skills == ["SAS"]
    assert result.scores.skills == 80


def test_sponsorship_and_experience_filters_still_work_with_skill_claims():
    posting = (
        "This position is not open to F-1 candidates. "
        "7 years of relevant experience. "
        "Proficiency in Python is required."
    )
    evaluation = make_evaluation(
        requirements=make_requirements(
            minimum_years_experience=7,
            sponsorship=SponsorshipStance.NOT_MENTIONED,
            sponsorship_language=None,
            required_skills=["Python"],
        ),
        skill_claims=[claim("Python", "Proficiency in Python is required.")],
        scores=make_scores(),
    )
    job_matcher = JobMatcher(
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
        filter_config=FilterConfig(
            experience=ExperienceFilterConfig(
                enabled=True,
                max_required_years=2,
                tolerance_years=2,
            ),
        ),
    )

    result = job_matcher.match(
        make_profile(
            work_authorization=WorkAuthorization(
                status="F-1 student; eligible for OPT/STEM OPT",
                requires_sponsorship=False,
                is_citizen_or_permanent_resident=False,
                requires_future_sponsorship=True,
            )
        ),
        posting,
    )

    assert (
        result.requirements.sponsorship
        is SponsorshipStance.PERMANENT_AUTHORIZATION_REQUIRED
    )
    assert result.passes_hard_filters is False
    assert result.recommendation == Recommendation.SKIP
    assert any("7+ years" in reason for reason in result.hard_filter_reasons)
    assert any("permanent" in reason.lower() for reason in result.hard_filter_reasons)
    assert result.requirements.required_skills == ["Python"]
