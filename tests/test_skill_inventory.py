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
from job_search_agent.evidence_units import (
    build_evidence_units,
    skill_candidate_units,
)
from job_search_agent.job_matcher import JobMatcher
from job_search_agent.models import (
    Recommendation,
    SkillExtractionStatus,
    SkillInventory,
    SkillMention,
    SponsorshipStance,
    UnitSkillInventory,
)
from job_search_agent.skill_extractor import (
    collect_skill_inventory,
    detect_unresolved_units,
    inventory_coverage,
)
from job_search_agent.skill_grounding import (
    ground_skill_inventory,
    union_grounded_names,
)
from job_search_agent.skill_scorer import score_skills


DEFAULT_SKILLS = SkillsScoringConfig(
    required_weight=0.80,
    preferred_weight=0.20,
    aliases={"r programming": "R", "python programming": "Python"},
)

PYTHON_JUPYTER_POSTING = (
    "Required Experience and Qualifications:\n"
    "Demonstrate expert knowledge of Python, and Jupyter Notebooks.\n"
    "SAS is preferred.\n"
)


def inventory(*entries: tuple[str, list[str]]) -> SkillInventory:
    return SkillInventory(
        units=[
            UnitSkillInventory(
                source_unit_id=unit_id,
                skills=[SkillMention(name=name) for name in names],
            )
            for unit_id, names in entries
        ]
    )


def filled_inventory(posting: str, names_by_id: dict[str, list[str]]) -> SkillInventory:
    units = skill_candidate_units(build_evidence_units(posting))
    return inventory(
        *((unit.id, names_by_id.get(unit.id, [])) for unit in units)
    )


def complete_empty(posting: str) -> SkillInventory:
    return filled_inventory(posting, {})


def scoring() -> ScoringConfig:
    return ScoringConfig(
        weights=ScoringWeights(
            skills=0.35,
            education=0.20,
            experience=0.25,
            career_relevance=0.20,
        ),
        thresholds=Thresholds(strongly_apply=85, apply=70, maybe=55),
        skills=DEFAULT_SKILLS,
    )


def test_dedicated_extractor_receives_deterministic_candidate_units():
    posting = PYTHON_JUPYTER_POSTING
    candidates = skill_candidate_units(build_evidence_units(posting))
    client = FakeAIClient(
        make_evaluation(),
        skill_inventory=complete_empty(posting),
        repair_inventory=complete_empty(posting),
    )

    collect_skill_inventory(client, posting)

    assert client.skill_prompts
    prompt = client.skill_prompts[0]
    for unit in candidates:
        assert f"{unit.id}:" in prompt
        assert unit.text in prompt


def test_one_result_is_required_for_every_candidate_unit():
    posting = PYTHON_JUPYTER_POSTING
    expected = [unit.id for unit in skill_candidate_units(build_evidence_units(posting))]
    coverage = inventory_coverage(complete_empty(posting), expected)

    assert coverage["missing_unit_ids"] == []
    assert coverage["unexpected_unit_ids"] == []
    assert coverage["returned_unit_ids"] == expected


def test_missing_unit_id_is_detected_as_incomplete():
    posting = PYTHON_JUPYTER_POSTING
    units = skill_candidate_units(build_evidence_units(posting))
    python = next(unit for unit in units if "Python" in unit.text)
    others = [unit.id for unit in units if unit.id != python.id]
    client = FakeAIClient(
        make_evaluation(),
        skill_inventory=inventory(*((unit_id, []) for unit_id in others)),
        repair_inventory=inventory(*((unit_id, []) for unit_id in others)),
    )

    _required, _preferred, report = collect_skill_inventory(client, posting)

    assert python.id in report.missing_unit_ids or python.id in report.unresolved_unit_ids
    assert report.status is SkillExtractionStatus.UNRESOLVED


def test_unknown_unit_id_is_rejected():
    posting = PYTHON_JUPYTER_POSTING
    units = build_evidence_units(posting)
    grounded = ground_skill_inventory(
        inventory(("u099", ["Python"])),
        units,
    )

    assert grounded == {}
    expected = [unit.id for unit in skill_candidate_units(units)]
    coverage = inventory_coverage(inventory(("u099", ["Python"])), expected)
    assert "u099" in coverage["unexpected_unit_ids"]


def test_python_and_jupyter_in_one_unit_both_survive():
    posting = PYTHON_JUPYTER_POSTING
    units = build_evidence_units(posting)
    python_unit = next(unit for unit in units if "Python" in unit.text)
    client = FakeAIClient(
        make_evaluation(),
        skill_inventory=filled_inventory(
            posting, {python_unit.id: ["Python", "Jupyter Notebooks"]}
        ),
    )

    required, preferred, report = collect_skill_inventory(client, posting)

    assert required == ["Python", "Jupyter Notebooks"]
    assert preferred == []
    assert report.status is SkillExtractionStatus.COMPLETE
    assert report.repair_attempted is False


def test_repair_unions_python_only_with_python_and_jupyter():
    posting = PYTHON_JUPYTER_POSTING
    units = build_evidence_units(posting)
    python_unit = next(unit for unit in units if "Python" in unit.text)
    client = FakeAIClient(
        make_evaluation(),
        skill_inventory=inventory((python_unit.id, ["Python"])),
        repair_inventory=inventory(
            (python_unit.id, ["Python", "Jupyter Notebooks"])
        ),
    )

    required, preferred, report = collect_skill_inventory(client, posting)

    assert required == ["Python", "Jupyter Notebooks"]
    assert preferred == []
    assert report.repair_attempted is True
    assert python_unit.id in report.repaired_unit_ids


def test_hallucinated_repair_skill_is_rejected():
    posting = PYTHON_JUPYTER_POSTING
    units = build_evidence_units(posting)
    python_unit = next(unit for unit in units if "Python" in unit.text)
    client = FakeAIClient(
        make_evaluation(),
        skill_inventory=inventory((python_unit.id, ["Python"])),
        repair_inventory=inventory(
            (python_unit.id, ["Python", "Kubernetes"])
        ),
    )

    required, _preferred, _report = collect_skill_inventory(client, posting)

    assert required == ["Python"]
    assert "Kubernetes" not in required


def test_repair_cannot_delete_a_valid_first_pass_skill():
    primary = {"u010": ["Python", "Jupyter Notebooks"]}
    repair = {"u010": ["Python"]}

    merged = union_grounded_names(primary, repair)

    assert merged["u010"] == ["Python", "Jupyter Notebooks"]


def test_only_one_repair_pass_is_allowed():
    posting = PYTHON_JUPYTER_POSTING
    units = skill_candidate_units(build_evidence_units(posting))
    python_unit = next(unit for unit in units if "Python" in unit.text)
    empty = inventory(*((unit.id, []) for unit in units))
    client = FakeAIClient(
        make_evaluation(),
        skill_inventories=[empty, empty, inventory((python_unit.id, ["Python"]))],
    )

    _required, _preferred, report = collect_skill_inventory(client, posting)

    assert len(client.skill_prompts) == 2
    assert report.repair_attempted is True
    assert "Unresolved evidence units" in client.skill_prompts[1]


def test_unresolved_empty_requirement_is_not_a_genuine_zero_inventory():
    posting = PYTHON_JUPYTER_POSTING
    empty = complete_empty(posting)
    evaluation = make_evaluation(scores=make_scores(skills=12))
    matcher = JobMatcher(
        ai_client=FakeAIClient(
            evaluation,
            skill_inventory=empty,
            repair_inventory=empty,
        ),
        scoring_config=scoring(),
        filter_config=PERMISSIVE_FILTERS,
    )

    result = matcher.match(make_profile(), posting)

    assert result.skill_extraction.status is SkillExtractionStatus.UNRESOLVED
    assert result.requirements.required_skills == []
    assert result.scores.skills == 12
    assert result.skill_extraction.skills_scored_from_inventory is False
    assert any("unresolved" in concern.lower() for concern in result.concerns)


def test_genuine_non_skill_unit_may_produce_no_skill():
    posting = "We offer an excellent benefits package, including PTO."
    units = skill_candidate_units(build_evidence_units(posting))
    grounded = {unit.id: [] for unit in units}
    unresolved = detect_unresolved_units(units, {unit.id for unit in units}, grounded)

    assert unresolved == []


def test_preferred_classification_comes_from_source_evidence():
    posting = (
        "Required qualifications:\n"
        "- Proficiency in Python is required\n"
        "Preferred qualifications:\n"
        "- SAS is desired\n"
    )
    units = build_evidence_units(posting)
    python = next(unit for unit in units if "Python" in unit.text)
    sas = next(unit for unit in units if "SAS" in unit.text)
    client = FakeAIClient(
        make_evaluation(),
        skill_inventory=inventory(
            (python.id, ["Python"]),
            (sas.id, ["SAS"]),
        ),
    )

    required, preferred, _report = collect_skill_inventory(client, posting)

    assert required == ["Python"]
    assert preferred == ["SAS"]


def test_non_skill_categories_stay_excluded():
    posting = (
        "Required Experience and Qualifications:\n"
        "Demonstrate expert knowledge of Python.\n"
        "Complete required course NSA NETA1400.\n"
    )
    units = build_evidence_units(posting)
    python = next(unit for unit in units if "Python" in unit.text)
    course = next(unit for unit in units if "NETA1400" in unit.text)
    client = FakeAIClient(
        make_evaluation(),
        skill_inventory=inventory(
            (python.id, ["Python"]),
            (course.id, ["NSA NETA1400"]),
        ),
    )

    required, preferred, _report = collect_skill_inventory(client, posting)

    assert required == ["Python"]
    assert preferred == []


def test_alias_normalization_and_score_skills_formula_unchanged():
    requirements = make_requirements(
        required_skills=["R", "Python"],
        preferred_skills=["SAS"],
    )

    assert score_skills(make_profile(), requirements, DEFAULT_SKILLS) == 80


def test_dedicated_inventory_is_not_overwritten_by_job_evaluation_lists():
    posting = PYTHON_JUPYTER_POSTING
    units = build_evidence_units(posting)
    python_unit = next(unit for unit in units if "Python" in unit.text)
    evaluation = make_evaluation(
        requirements=make_requirements(
            required_skills=["Kubernetes"],
            preferred_skills=["Excel"],
        )
    )
    matcher = JobMatcher(
        ai_client=FakeAIClient(
            evaluation,
            skill_inventory=inventory(
                (python_unit.id, ["Python", "Jupyter Notebooks"])
            ),
        ),
        scoring_config=scoring(),
        filter_config=PERMISSIVE_FILTERS,
    )

    result = matcher.match(make_profile(), posting)

    assert result.requirements.required_skills == ["Python", "Jupyter Notebooks"]
    assert "Kubernetes" not in result.requirements.required_skills
    assert result.requirements.preferred_skills == []


def test_sponsorship_and_experience_filters_still_work_with_dedicated_extraction():
    posting = (
        "This position is not open to F-1 candidates. "
        "7 years of relevant experience. "
        "Demonstrate expert knowledge of Python."
    )
    units = build_evidence_units(posting)
    python_unit = next(unit for unit in units if "Python" in unit.text)
    evaluation = make_evaluation(
        requirements=make_requirements(
            minimum_years_experience=7,
            sponsorship=SponsorshipStance.NOT_MENTIONED,
            sponsorship_language=None,
            required_skills=["Python"],
        ),
        scores=make_scores(),
    )
    matcher = JobMatcher(
        ai_client=FakeAIClient(
            evaluation,
            skill_inventory=inventory((python_unit.id, ["Python"])),
        ),
        scoring_config=scoring(),
        filter_config=FilterConfig(
            experience=ExperienceFilterConfig(
                enabled=True,
                max_required_years=2,
                tolerance_years=2,
            ),
        ),
    )

    result = matcher.match(
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
    assert result.requirements.required_skills == ["Python"]
