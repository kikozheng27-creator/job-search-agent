from job_search_agent.skill_selection import (
    atomic_capability_name,
    canonical_display_name,
    select_scored_skill,
)
from job_search_agent.skill_grounding import (
    ground_skill_claims,
    lists_from_grounded_unit_names,
)
from job_search_agent.evidence_units import build_evidence_units
from job_search_agent.models import SkillClaim, SkillLevel
from job_search_agent.skill_scorer import score_skills
from conftest import make_profile, make_requirements
from job_search_agent.config_models import SkillsScoringConfig


DEFAULT_SKILLS = SkillsScoringConfig(
    required_weight=0.80,
    preferred_weight=0.20,
    aliases={"r programming": "R", "python programming": "Python"},
)


def test_atomic_name_reduces_environment_fragment_to_big_data():
    assert atomic_capability_name("managing data in a big-data environment") == "big-data"
    assert canonical_display_name("big-data") == "Big Data"
    assert canonical_display_name("big data") == "Big Data"


def test_knowledge_of_list_keeps_architecture_and_big_data_drops_disciplines():
    unit = (
        "Required Experience and Qualifications: Demonstrate experience and "
        "knowledge of computer science concepts, data architecture, "
        "intelligence practices, and managing data in a big-data environment."
    )

    assert select_scored_skill("computer science concepts", unit) is None
    assert select_scored_skill("intelligence practices", unit) is None
    assert select_scored_skill("data architecture", unit) == "Data Architecture"
    assert (
        select_scored_skill("managing data in a big-data environment", unit)
        == "Big Data"
    )


def test_expert_knowledge_keeps_python_and_jupyter():
    unit = (
        "Required Experience and Qualifications: Demonstrate expert knowledge "
        "of Python, and Jupyter Notebooks and/or JupyterLabs."
    )

    assert select_scored_skill("Python", unit) == "Python"
    assert select_scored_skill("Jupyter Notebooks", unit) == "Jupyter Notebooks"
    assert select_scored_skill("JupyterLabs", unit) == "JupyterLabs"


def test_incidental_ide_examples_are_dropped_but_proficiency_is_kept():
    incidental = (
        "Author cogent scripts using Python using common Integrated "
        "Development Environments (IDE) such as VS Code, Spyder, PyScript, "
        "or Jupyter Notebooks."
    )
    proficiency = "Proficiency in VS Code and Spyder is required."

    assert select_scored_skill("VS Code", incidental) is None
    assert select_scored_skill("Spyder", incidental) is None
    assert select_scored_skill("Python", incidental) is None
    assert select_scored_skill("VS Code", proficiency) == "VS Code"
    assert select_scored_skill("Spyder", proficiency) == "Spyder"


def test_algorithms_kept_only_as_a_stated_qualification():
    responsibility = (
        "Develop and use advanced software programs, algorithms, "
        "querytechniques, models to solve complex intelligence problems."
    )
    qualification = "Experience with algorithms is required."

    assert select_scored_skill("algorithms", responsibility) is None
    assert select_scored_skill("algorithms", qualification) == "Algorithms"


def test_data_science_role_umbrella_is_dropped_knowledge_is_kept():
    umbrella = (
        "This position is responsible for conducting data science functions "
        "on structured and unstructured data."
    )
    knowledge = "Knowledge of data science is required."

    assert select_scored_skill("data science", umbrella) is None
    assert select_scored_skill("structured data", umbrella) is None
    assert select_scored_skill("data science", knowledge) == "Data Science"


def test_capability_list_keeps_ml_methods_and_canonicalizes_case():
    unit = (
        "Analyze requirements and evaluate technologies for data science "
        "capabilities including Natural Language Processing, Machine Learning, "
        "predictive modeling, statistical analysis, and hypothesis testing."
    )

    assert select_scored_skill("Machine Learning", unit) == "Machine Learning"
    assert select_scored_skill("machine learning", unit) == "Machine Learning"
    assert select_scored_skill("statistical analysis", unit) == "Statistical Analysis"
    assert select_scored_skill("Natural Language Processing", unit) == (
        "Natural Language Processing"
    )


def test_awareness_of_emerging_tech_is_not_a_qualification():
    unit = "Maintain awareness of emerging analytics and big-data technologies."

    assert select_scored_skill("analytics", unit) is None
    assert select_scored_skill("big-data technologies", unit) is None


def test_hyphen_and_case_variants_dedupe_in_final_lists():
    unit = (
        "Demonstrate experience and knowledge of big-data and Machine Learning. "
        "Experience with machine learning and big data is required."
    )
    units = build_evidence_units(unit)
    grounded = {units[0].id: ["big-data", "Machine Learning", "machine learning", "big data"]}

    required, preferred = lists_from_grounded_unit_names(grounded, units)

    assert required == ["Big Data", "Machine Learning"]
    assert preferred == []


def test_preferred_source_still_classifies_sas():
    posting = "SAS is preferred for this biostatistician role."
    required, preferred = ground_skill_claims(
        [SkillClaim(name="SAS", evidence=posting, level=SkillLevel.REQUIRED)],
        posting,
    )

    assert required == []
    assert preferred == ["SAS"]


def test_score_skills_formula_unchanged():
    requirements = make_requirements(
        required_skills=["R", "Python"],
        preferred_skills=["SAS"],
    )

    assert score_skills(make_profile(), requirements, DEFAULT_SKILLS) == 80
