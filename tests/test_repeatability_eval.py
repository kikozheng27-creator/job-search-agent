import sys
from pathlib import Path

from conftest import (
    PERMISSIVE_FILTERS,
    FakeAIClient,
    make_evaluation,
    make_profile,
    make_requirements,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.repeatability import (
    SnapshotEvaluator,
    categorical_summary,
    largest_skills_swing,
    pairwise_jaccard,
    python_skills_consistent_on_identical_inputs,
    record_run,
    summarize,
)
from job_search_agent.config_models import (
    ScoringConfig,
    ScoringWeights,
    SkillsScoringConfig,
    Thresholds,
)
from job_search_agent.job_matcher import JobMatcher
from job_search_agent.models import Recommendation, SponsorshipStance


def test_summarize_reports_all_values_and_spread():
    stats = summarize([7, 9, 100])

    assert stats["values"] == [7, 9, 100]
    assert stats["minimum"] == 7
    assert stats["maximum"] == 100
    assert stats["mean"] == 38.667
    assert stats["std_dev"] > 0


def test_categorical_summary_groups_runs_by_value():
    runs = [
        {"run": 1, "required_skills": ["Python"]},
        {"run": 2, "required_skills": ["Python", "Jupyter"]},
        {"run": 3, "required_skills": ["Python"]},
    ]
    summary = categorical_summary(runs, "required_skills")

    assert summary["unique_count"] == 2
    by_value = {tuple(item["value"]): item for item in summary["frequencies"]}
    assert by_value[("Python",)]["runs"] == [1, 3]
    assert by_value[("Python", "Jupyter")]["runs"] == [2]


def test_identical_skill_lists_are_treated_as_deterministic():
    runs = [
        {
            "run": 1,
            "required_skills": ["Python"],
            "preferred_skills": [],
            "skills_score": 100,
        },
        {
            "run": 2,
            "required_skills": ["Python"],
            "preferred_skills": [],
            "skills_score": 100,
        },
        {
            "run": 3,
            "required_skills": ["Python", "Jupyter"],
            "preferred_skills": [],
            "skills_score": 9,
        },
    ]

    result = python_skills_consistent_on_identical_inputs(runs)

    assert result["identical_skill_lists_always_same_score"] is True
    assert result["distinct_skill_list_pairs"] == 2


def test_skill_swing_traces_extraction_difference():
    def row(run, skills, required, matched, missing, rate):
        return {
            "run": run,
            "skills_score": skills,
            "required_skills": required,
            "preferred_skills": [],
            "skills_diagnostics": {
                "required": {
                    "match_rate": rate,
                    "matched": matched,
                    "missing": missing,
                },
                "preferred": {
                    "match_rate": 1.0,
                    "matched": [],
                    "missing": [],
                },
            },
        }

    runs = [
        row(2, 9, ["Python", "Jupyter"], ["Python"], ["Jupyter"], 0.5),
        row(8, 100, ["Python"], ["Python"], [], 1.0),
    ]
    swing = largest_skills_swing(runs)

    assert swing["changed"] is True
    assert swing["required_skills_differ"] is True
    assert swing["low"]["run"] == 2
    assert swing["high"]["run"] == 8
    assert swing["score_delta"] == 91


def test_snapshot_evaluator_preserves_llm_output_after_matcher_mutation():
    evaluation = make_evaluation(
        requirements=make_requirements(
            required_skills=["Python"],
            sponsorship=SponsorshipStance.NOT_OFFERED,
            sponsorship_language="We do not provide visa sponsorship.",
        ),
    )
    wrapper = SnapshotEvaluator(FakeAIClient(evaluation))
    returned = wrapper.evaluate_job("prompt")

    returned.requirements.required_skills = ["mutated"]
    returned.scores.skills = 1

    assert wrapper.last is not None
    assert wrapper.last.requirements.required_skills == ["Python"]
    assert wrapper.last.scores.skills == 90


def test_record_run_does_not_call_openai_or_the_network():
    scoring = ScoringConfig(
        weights=ScoringWeights(
            skills=0.35,
            education=0.20,
            experience=0.25,
            career_relevance=0.20,
        ),
        thresholds=Thresholds(strongly_apply=85, apply=70, maybe=55),
        skills=SkillsScoringConfig(
            required_weight=0.80,
            preferred_weight=0.20,
            aliases={"python programming": "Python"},
        ),
    )
    evaluation = make_evaluation(
        requirements=make_requirements(
            required_skills=["Python", "Jupyter"],
            preferred_skills=["SAS"],
            minimum_years_experience=7,
            required_degree="Master's",
        ),
    )
    wrapper = SnapshotEvaluator(FakeAIClient(evaluation))
    analysis = JobMatcher(
        ai_client=wrapper,
        scoring_config=scoring,
        filter_config=PERMISSIVE_FILTERS,
    ).match(
        profile=make_profile(),
        job_description="Example posting about Python. Jupyter is also required.",
    )

    row = record_run(
        1,
        analysis,
        wrapper.last,
        "Example posting about Python. Jupyter is also required.",
        make_profile(),
        scoring,
    )

    assert row["required_skills"] == ["Python", "Jupyter"]
    assert row["skills_diagnostics"]["required"]["matched"] == ["Python"]
    assert "Jupyter" in row["skills_diagnostics"]["required"]["missing"]
    assert row["recomputed_skills_score_matches"] is True
    assert row["final_recommendation"] in {item.value for item in Recommendation}
    assert row["python_overwrote_llm_skills_score"] is True
    assert row["evidence_units"]["unit_count"] >= 1
    assert "omitted_unit_ids" in row["evidence_units"]


def test_pairwise_jaccard_on_normalized_skill_sets():
    stats = pairwise_jaccard(
        [
            ["Python", "Big Data"],
            ["python", "big-data"],
            ["Python", "Machine Learning"],
        ]
    )

    assert stats["pair_count"] == 3
    assert stats["maximum"] == 1.0
    assert stats["minimum"] == 1 / 3
    assert stats["mean"] == 0.556
