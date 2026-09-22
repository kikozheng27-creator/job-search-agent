"""Repeatability evaluation: same profile, same posting, N live analyses.

Fetches a job URL once (or reads a local JD), then reuses that exact text
for every run so page-loader variance is isolated from LLM/matcher variance.

Does not write tracker rows. Does not change scoring, prompts, or matcher
business rules. May call the real OpenAI API.

Unit tests must not import AIClient or fetch URLs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from job_search_agent.config_loader import (
    load_candidate_profile,
    load_scoring_config,
)
from job_search_agent.main import analyze_job_description
from job_search_agent.evidence_units import (
    build_evidence_units,
    skill_candidate_units,
)
from job_search_agent.models import JobEvaluation
from job_search_agent.protocols import JobEvaluator
from job_search_agent.scoring import get_recommendation
from job_search_agent.skill_scorer import diagnose_skills
from job_search_agent.sponsorship_classifier import quote_in_source


DEFAULT_URL = (
    "https://job-boards.greenhouse.io/assertiveprofessionals/jobs/4372007009"
)
DEFAULT_RUNS = 10
DEFAULT_OUTPUT = Path("evaluation/output/repeatability.json")
CONFIG_DIR = Path("config")

NUMERIC_FIELDS = (
    "skills_score",
    "education_score",
    "experience_score",
    "career_relevance_score",
    "overall_score",
)

WEIGHTS = {
    "skills_score": 0.35,
    "education_score": 0.20,
    "experience_score": 0.25,
    "career_relevance_score": 0.20,
}

CATEGORICAL_FIELDS = (
    "company",
    "title",
    "location",
    "required_skills",
    "preferred_skills",
    "minimum_years_experience",
    "required_degree",
    "llm_sponsorship_language",
    "validated_sponsorship_language",
    "sponsorship_stance",
    "final_recommendation",
    "hard_filter_reasons",
)


class SnapshotEvaluator:
    """Capture the LLM JobEvaluation before JobMatcher mutates it in place."""

    def __init__(self, inner: JobEvaluator) -> None:
        self.inner = inner
        self.last: JobEvaluation | None = None
        self.supports_dedicated_skill_extraction = getattr(
            inner, "supports_dedicated_skill_extraction", False
        )

    def evaluate_job(self, prompt: str) -> JobEvaluation:
        evaluation = self.inner.evaluate_job(prompt)
        self.last = evaluation.model_copy(deep=True)
        return evaluation

    def extract_skill_inventory(self, prompt: str):
        return self.inner.extract_skill_inventory(prompt)


def summarize(values: list[float]) -> dict:
    return {
        "values": values,
        "mean": round(statistics.mean(values), 3) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "range": round(max(values) - min(values), 3) if values else None,
        "std_dev": (
            round(statistics.stdev(values), 3) if len(values) > 1 else 0.0
        ),
    }


def pairwise_jaccard(lists: list[list[str]]) -> dict:
    """Jaccard similarity on normalized skill sets. Does not change scoring."""
    from job_search_agent.skill_scorer import normalize_skill

    sets = [{normalize_skill(name) for name in names if name.strip()} for names in lists]
    scores: list[float] = []
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            a = sets[left]
            b = sets[right]
            if not a and not b:
                scores.append(1.0)
            else:
                scores.append(len(a & b) / len(a | b))
    return {
        "mean": round(statistics.mean(scores), 3) if scores else None,
        "minimum": min(scores) if scores else None,
        "maximum": max(scores) if scores else None,
        "pair_count": len(scores),
    }


def freeze(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def categorical_summary(runs: list[dict], field: str) -> dict:
    buckets: dict[str, list[int]] = defaultdict(list)

    for row in runs:
        buckets[freeze(row[field])].append(row["run"])

    frequencies = sorted(
        (
            {
                "value": json.loads(key),
                "count": len(run_ids),
                "runs": run_ids,
            }
            for key, run_ids in buckets.items()
        ),
        key=lambda item: (-item["count"], freeze(item["value"])),
    )

    return {
        "unique_count": len(frequencies),
        "frequencies": frequencies,
    }


def recommendation_stability(runs: list[dict]) -> dict:
    counts = Counter(row["final_recommendation"] for row in runs)
    score_counts = Counter(row["score_based_recommendation"] for row in runs)

    return {
        "final": dict(counts),
        "score_based": dict(score_counts),
    }


def hard_filter_stability(runs: list[dict]) -> dict:
    failed = sum(1 for row in runs if not row["passes_hard_filters"])
    reason_counts: dict[str, list[int]] = defaultdict(list)

    for row in runs:
        for reason in row["hard_filter_reasons"]:
            reason_counts[reason].append(row["run"])

    return {
        "runs_failed": failed,
        "runs_passed": len(runs) - failed,
        "reason_frequencies": [
            {
                "reason": reason,
                "count": len(run_ids),
                "runs": run_ids,
            }
            for reason, run_ids in sorted(
                reason_counts.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
        ],
        "exact_reason_lists": categorical_summary(runs, "hard_filter_reasons"),
    }


def python_skills_consistent_on_identical_inputs(runs: list[dict]) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)

    for row in runs:
        key = freeze(
            {
                "required": row["required_skills"],
                "preferred": row["preferred_skills"],
            }
        )
        groups[key].append(row["skills_score"])

    consistent = all(len(set(scores)) == 1 for scores in groups.values())

    return {
        "identical_skill_lists_always_same_score": consistent,
        "distinct_skill_list_pairs": len(groups),
    }


def python_experience_consistent_on_identical_inputs(runs: list[dict]) -> dict:
    """Same candidate years + same extracted min years must yield the same score."""
    groups: dict[str, list[int]] = defaultdict(list)

    for row in runs:
        key = freeze(
            {
                "candidate_years": row.get("candidate_years_of_experience"),
                "minimum_years": row["minimum_years_experience"],
            }
        )
        groups[key].append(row["experience_score"])

    consistent = all(len(set(scores)) == 1 for scores in groups.values())
    production_range = summarize([row["experience_score"] for row in runs])["range"]

    return {
        "identical_year_inputs_always_same_score": consistent,
        "distinct_year_input_pairs": len(groups),
        "production_experience_range": production_range,
    }


def python_education_consistent_on_identical_inputs(runs: list[dict]) -> dict:
    """Same candidate degrees and required_degree must yield the same score."""
    groups: dict[str, list[int]] = defaultdict(list)

    for row in runs:
        key = freeze(
            {
                "candidate_degrees": row.get("candidate_degrees"),
                "required_degree": row.get("required_degree"),
            }
        )
        groups[key].append(row["education_score"])

    consistent = all(len(set(scores)) == 1 for scores in groups.values())
    production_range = summarize([row["education_score"] for row in runs])["range"]

    return {
        "identical_education_inputs_always_same_score": consistent,
        "distinct_education_input_pairs": len(groups),
        "production_education_range": production_range,
    }


def describe_evidence_units(snapshot: JobEvaluation | None, job_description: str) -> dict:
    """Record which source units the model inspected and which produced skills."""
    units = build_evidence_units(job_description)
    candidates = skill_candidate_units(units)
    valid_ids = {unit.id for unit in units}
    candidate_ids = {unit.id for unit in candidates}
    decisions = list(snapshot.skill_unit_decisions) if snapshot else []
    returned_ids = [decision.source_unit_id for decision in decisions]
    returned_set = set(returned_ids)

    classifications = []
    for unit in units:
        matching = [
            decision
            for decision in decisions
            if decision.source_unit_id == unit.id
        ]
        if not matching:
            continue
        skills: list[str] = []
        seen: set[str] = set()
        is_requirement = False
        for decision in matching:
            is_requirement = is_requirement or decision.is_candidate_requirement
            for item in decision.skills:
                key = item.name.strip().casefold()
                if not key or key in seen:
                    continue
                seen.add(key)
                skills.append(item.name)
        classifications.append(
            {
                "id": unit.id,
                "is_candidate_requirement": is_requirement,
                "skills": skills,
            }
        )

    duplicate_ids = sorted(
        {unit_id for unit_id in returned_ids if returned_ids.count(unit_id) > 1}
    )

    return {
        "unit_count": len(units),
        "candidate_unit_count": len(candidates),
        "candidate_unit_ids": [unit.id for unit in candidates],
        "decisions_returned": len(decisions),
        "omitted_unit_ids": [
            unit.id for unit in candidates if unit.id not in returned_set
        ],
        "invalid_unit_ids": [
            unit_id for unit_id in returned_ids if unit_id not in valid_ids
        ],
        "duplicate_unit_ids": duplicate_ids,
        "units_with_skills": [
            item["id"]
            for item in classifications
            if item["is_candidate_requirement"] and item["skills"]
        ],
        "classifications": classifications,
    }


def evidence_unit_inventory(job_description: str) -> list[dict]:
    return [
        {"id": unit.id, "text": unit.text}
        for unit in build_evidence_units(job_description)
    ]


def evidence_unit_stability(runs: list[dict]) -> dict:
    if not runs:
        return {
            "unit_count_stable": True,
            "unit_count_values": [],
            "runs_with_complete_coverage": [],
            "unique_omitted_sets": 0,
            "unique_units_with_skills_sets": 0,
            "list_cardinalities": [],
            "inconsistent_unit_classifications": [],
        }

    unit_counts = sorted({row["evidence_units"]["unit_count"] for row in runs})
    candidate_counts = sorted(
        {row["evidence_units"].get("candidate_unit_count", 0) for row in runs}
    )
    omitted_sets = categorical_summary(
        [
            {"run": row["run"], "omitted": row["evidence_units"]["omitted_unit_ids"]}
            for row in runs
        ],
        "omitted",
    )
    skill_unit_sets = categorical_summary(
        [
            {
                "run": row["run"],
                "units_with_skills": row["evidence_units"]["units_with_skills"],
            }
            for row in runs
        ],
        "units_with_skills",
    )
    cardinalities = sorted({len(row["required_skills"]) for row in runs})
    complete = [
        row["run"]
        for row in runs
        if not row["evidence_units"]["omitted_unit_ids"]
    ]

    signatures: dict[str, dict[str, list[int]]] = {}
    for row in runs:
        for item in row["evidence_units"]["classifications"]:
            signature = freeze(
                {
                    "is_candidate_requirement": item["is_candidate_requirement"],
                    "skills": item["skills"],
                }
            )
            unit_bucket = signatures.setdefault(item["id"], {})
            unit_bucket.setdefault(signature, []).append(row["run"])

    inconsistent = []
    for unit_id, buckets in signatures.items():
        if len(buckets) < 2:
            continue
        inconsistent.append(
            {
                "unit_id": unit_id,
                "unique_signatures": len(buckets),
                "signatures": [
                    {
                        "value": json.loads(signature),
                        "count": len(run_ids),
                        "runs": run_ids,
                    }
                    for signature, run_ids in sorted(
                        buckets.items(),
                        key=lambda pair: (-len(pair[1]), pair[0]),
                    )
                ],
            }
        )
    inconsistent.sort(key=lambda item: item["unit_id"])

    return {
        "unit_count_stable": len(unit_counts) == 1,
        "unit_count_values": unit_counts,
        "candidate_unit_count_stable": len(candidate_counts) == 1,
        "candidate_unit_count_values": candidate_counts,
        "runs_with_complete_coverage": complete,
        "complete_coverage_count": len(complete),
        "unique_omitted_sets": omitted_sets["unique_count"],
        "omitted_sets": omitted_sets["frequencies"],
        "unique_units_with_skills_sets": skill_unit_sets["unique_count"],
        "units_with_skills_sets": skill_unit_sets["frequencies"],
        "required_list_cardinalities": cardinalities,
        "inconsistent_unit_classifications": inconsistent,
        "inconsistent_unit_count": len(inconsistent),
    }


def dedicated_skill_stability(runs: list[dict]) -> dict:
    if not runs:
        return {}

    reports = [row.get("skill_extraction") or {} for row in runs]
    candidate_counts = [
        len(report.get("expected_unit_ids") or []) for report in reports
    ]
    repair_runs = [
        row["run"] for row, report in zip(runs, reports) if report.get("repair_attempted")
    ]
    unresolved_runs = [
        row["run"]
        for row, report in zip(runs, reports)
        if report.get("status") == "unresolved"
    ]
    scored_from_inventory = [
        row["run"]
        for row, report in zip(runs, reports)
        if report.get("skills_scored_from_inventory")
    ]
    empty_scored_100 = [
        row["run"]
        for row in runs
        if row["skills_score"] == 100
        and not row["required_skills"]
        and not row["preferred_skills"]
        and (row.get("skill_extraction") or {}).get("skills_scored_from_inventory")
    ]
    u010_by_run = {
        row["run"]: (row.get("skill_extraction") or {})
        .get("per_unit_skills", {})
        .get("u010")
        for row in runs
    }
    repaired_by_run = {
        row["run"]: (row.get("skill_extraction") or {}).get("repaired_unit_ids")
        for row in runs
    }

    unit_ids = sorted(
        {
            unit_id
            for report in reports
            for unit_id in (report.get("per_unit_skills") or {})
        }
    )
    per_unit = []
    for unit_id in unit_ids:
        summary = categorical_summary(
            [
                {
                    "run": row["run"],
                    "skills": (row.get("skill_extraction") or {})
                    .get("per_unit_skills", {})
                    .get(unit_id, []),
                }
                for row in runs
            ],
            "skills",
        )
        if summary["unique_count"] <= 1:
            continue
        per_unit.append(
            {
                "unit_id": unit_id,
                "unique_count": summary["unique_count"],
                "frequencies": summary["frequencies"],
            }
        )

    return {
        "candidate_unit_count": candidate_counts[0] if candidate_counts else 0,
        "candidate_unit_count_stable": len(set(candidate_counts)) <= 1,
        "runs_needing_repair": repair_runs,
        "repair_count": len(repair_runs),
        "unresolved_runs": unresolved_runs,
        "scored_from_inventory_runs": scored_from_inventory,
        "empty_inventory_scored_100": empty_scored_100,
        "u010_by_run": u010_by_run,
        "repaired_unit_ids_by_run": repaired_by_run,
        "per_unit_skill_frequencies": per_unit,
        "required_jaccard": pairwise_jaccard(
            [row["required_skills"] for row in runs]
        ),
    }


def skill_extraction_traces(runs: list[dict]) -> list[dict]:
    ordered = sorted(runs, key=lambda row: (row["skills_score"], row["run"]))
    traces = []

    for row in ordered:
        required = row["skills_diagnostics"]["required"]
        preferred = row["skills_diagnostics"]["preferred"]
        traces.append(
            {
                "run": row["run"],
                "skills_score": row["skills_score"],
                "required_skills": row["required_skills"],
                "required_match_rate": required["match_rate"],
                "required_matched": required["matched"],
                "required_missing": required["missing"],
                "preferred_skills": row["preferred_skills"],
                "preferred_match_rate": preferred["match_rate"],
                "preferred_matched": preferred["matched"],
                "preferred_missing": preferred["missing"],
            }
        )

    return traces


def largest_skills_swing(runs: list[dict]) -> dict | None:
    if len(runs) < 2:
        return None

    lowest = min(runs, key=lambda row: (row["skills_score"], row["run"]))
    highest = max(runs, key=lambda row: (row["skills_score"], row["run"]))

    if lowest["skills_score"] == highest["skills_score"]:
        return {
            "changed": False,
            "skills_score": lowest["skills_score"],
        }

    return {
        "changed": True,
        "low": _skill_swing_side(lowest),
        "high": _skill_swing_side(highest),
        "score_delta": highest["skills_score"] - lowest["skills_score"],
        "required_skills_differ": (
            lowest["required_skills"] != highest["required_skills"]
        ),
        "preferred_skills_differ": (
            lowest["preferred_skills"] != highest["preferred_skills"]
        ),
    }


def _skill_swing_side(row: dict) -> dict:
    required = row["skills_diagnostics"]["required"]
    preferred = row["skills_diagnostics"]["preferred"]

    return {
        "run": row["run"],
        "skills_score": row["skills_score"],
        "required_skills": row["required_skills"],
        "required_match_rate": required["match_rate"],
        "required_matched": required["matched"],
        "required_missing": required["missing"],
        "preferred_skills": row["preferred_skills"],
        "preferred_match_rate": preferred["match_rate"],
        "preferred_matched": preferred["matched"],
        "preferred_missing": preferred["missing"],
        "formula": (
            "int(round(100 * (0.80 * required_match_rate "
            "+ 0.20 * preferred_match_rate)))"
        ),
        "omitted_unit_ids": row.get("evidence_units", {}).get("omitted_unit_ids"),
        "units_with_skills": row.get("evidence_units", {}).get("units_with_skills"),
        "decisions_returned": row.get("evidence_units", {}).get("decisions_returned"),
    }


def _career_relevance_comparison(runs: list[dict]) -> dict:
    """Separate the raw model integer from the Python career score.

    Identical validated relations must produce a zero production range.
    Relation differences are extraction variance, not scorer variance.
    """
    llm_values = [
        (row.get("llm_scores") or {}).get("career_relevance")
        for row in runs
        if (row.get("llm_scores") or {}).get("career_relevance") is not None
    ]
    production = summarize([row["career_relevance_score"] for row in runs])
    family = categorical_summary(runs, "target_family_alignment")
    industry = categorical_summary(runs, "preferred_industry_alignment")
    relations_stable = family["unique_count"] == 1 and industry["unique_count"] == 1

    return {
        "note": (
            "Python scores career relevance from validated target-family "
            "and preferred-industry relations. The model integer is not "
            "production authority."
        ),
        "llm_score_summary": summarize(llm_values),
        "production_score_summary": production,
        "target_family_alignment": family,
        "preferred_industry_alignment": industry,
        "python_overwrote_llm_career_relevance_score": [
            row["run"]
            for row in runs
            if row.get("python_overwrote_llm_career_relevance_score")
        ],
        "relations_stable": relations_stable,
        "production_range_zero_when_relations_stable": (
            relations_stable and production["range"] in (0, 0.0)
        ),
    }


def llm_score_fields_vs_extraction(runs: list[dict]) -> dict:
    """Separate LLM component scores from extracted requirement fields."""
    years = categorical_summary(runs, "minimum_years_experience")
    degrees = categorical_summary(runs, "required_degree")
    llm_experience = [
        (row.get("llm_scores") or {}).get("experience")
        for row in runs
        if (row.get("llm_scores") or {}).get("experience") is not None
    ]
    production_experience = [row["experience_score"] for row in runs]
    production_summary = summarize(production_experience)
    years_stable = years["unique_count"] == 1
    llm_education = [
        (row.get("llm_scores") or {}).get("education")
        for row in runs
        if (row.get("llm_scores") or {}).get("education") is not None
    ]
    production_education = summarize([row["education_score"] for row in runs])
    candidate_degree_lists = [
        tuple(row.get("candidate_degrees") or []) for row in runs
    ]
    education_inputs_stable = (
        degrees["unique_count"] == 1 and len(set(candidate_degree_lists)) <= 1
    )

    return {
        "experience": {
            "extraction_unique_years": years["unique_count"],
            "extraction_values": years["frequencies"],
            "llm_score_summary": summarize(llm_experience),
            "production_score_summary": production_summary,
            "python_overwrote_llm_experience_score": [
                row["run"]
                for row in runs
                if row.get("python_overwrote_llm_experience_score")
            ],
            "score_varies_with_stable_years": (
                years_stable
                and production_summary["range"] not in (0, 0.0, None)
            ),
            "production_range_zero_when_years_stable": (
                years_stable and production_summary["range"] in (0, 0.0)
            ),
        },
        "education": {
            "extraction_unique_degrees": degrees["unique_count"],
            "extraction_values": degrees["frequencies"],
            "llm_score_summary": summarize(llm_education),
            "production_score_summary": production_education,
            "python_overwrote_llm_education_score": [
                row["run"]
                for row in runs
                if row.get("python_overwrote_llm_education_score")
            ],
            "score_varies_with_stable_degree": (
                education_inputs_stable
                and production_education["range"] not in (0, 0.0, None)
            ),
            "production_range_zero_when_education_inputs_stable": (
                education_inputs_stable
                and production_education["range"] in (0, 0.0)
            ),
        },
        "career_relevance": _career_relevance_comparison(runs),
    }


def root_cause_notes(runs: list[dict]) -> list[str]:
    notes: list[str] = []
    consistency = python_skills_consistent_on_identical_inputs(runs)
    swing = largest_skills_swing(runs)
    skills_lists = categorical_summary(runs, "required_skills")
    preferred_lists = categorical_summary(runs, "preferred_skills")
    llm_vs_extract = llm_score_fields_vs_extraction(runs)

    notes.append(
        "Unstable LLM-extracted fields: required_skills "
        f"({skills_lists['unique_count']} unique lists), "
        f"preferred_skills ({preferred_lists['unique_count']} unique lists), "
        "required_degree "
        f"({categorical_summary(runs, 'required_degree')['unique_count']} "
        "unique values), minimum_years_experience "
        f"({categorical_summary(runs, 'minimum_years_experience')['unique_count']}"
        " unique values), llm_sponsorship_language "
        f"({categorical_summary(runs, 'llm_sponsorship_language')['unique_count']}"
        " unique values)."
    )

    if swing and swing.get("changed"):
        low = swing["low"]
        high = swing["high"]
        notes.append(
            "Skills-score swing is "
            f"{low['skills_score']} (run {low['run']}) -> "
            f"{high['skills_score']} (run {high['run']}), delta "
            f"{swing['score_delta']}."
        )
        notes.append(
            f"Run {low['run']} required_skills={low['required_skills']} "
            f"required_match_rate={low['required_match_rate']} "
            f"matched={low['required_matched']} missing={low['required_missing']}."
        )
        notes.append(
            f"Run {high['run']} required_skills={high['required_skills']} "
            f"required_match_rate={high['required_match_rate']} "
            f"matched={high['required_matched']} missing={high['required_missing']}."
        )
        if swing["required_skills_differ"] or swing["preferred_skills_differ"]:
            notes.append(
                "Primary skills instability originates from required/preferred "
                "skill extraction: the Python formula is the same; only the "
                "extracted lists changed."
            )
        else:
            notes.append(
                "Required and preferred lists were identical on the min/max "
                "skills runs, so extraction does not explain that swing."
            )
    else:
        notes.append("Skills score did not change across runs.")

    if consistency["identical_skill_lists_always_same_score"]:
        notes.append(
            "Deterministic Python skill scoring is consistent: identical "
            "extracted skill lists always produced the same skills_score."
        )
    else:
        notes.append(
            "WARNING: identical extracted skill lists produced different "
            "skills_score values. That would mean unexpected nondeterminism "
            "in Python scoring."
        )

    experience = llm_vs_extract["experience"]
    experience_consistency = python_experience_consistent_on_identical_inputs(runs)
    if experience_consistency["identical_year_inputs_always_same_score"]:
        notes.append(
            "Deterministic Python experience scoring is consistent: identical "
            "candidate years and minimum_years_experience always produced "
            "the same experience_score."
        )
    else:
        notes.append(
            "WARNING: identical candidate years and minimum_years_experience "
            "produced different experience_score values."
        )
    if experience.get("production_range_zero_when_years_stable"):
        notes.append(
            "Production experience score range is 0 while "
            "minimum_years_experience is stable."
        )
    elif experience["score_varies_with_stable_years"]:
        notes.append(
            "Experience score varies while minimum_years_experience is "
            "stable, so that variance comes from the experience scorer, "
            "not from the extracted years field."
        )
    elif experience["extraction_unique_years"] > 1:
        notes.append(
            "minimum_years_experience extraction is itself unstable, which "
            "can affect both the experience score prompt context and the "
            "experience hard filter."
        )

    education = llm_vs_extract["education"]
    education_consistency = python_education_consistent_on_identical_inputs(runs)
    if education_consistency["identical_education_inputs_always_same_score"]:
        notes.append(
            "Deterministic Python education scoring is consistent: identical "
            "candidate degrees and required_degree always produced the same "
            "education_score."
        )
    else:
        notes.append(
            "WARNING: identical candidate degrees and required_degree "
            "produced different education_score values."
        )
    if education.get("production_range_zero_when_education_inputs_stable"):
        notes.append(
            "Production education score range is 0 while validated education "
            "inputs are stable."
        )
    elif education["score_varies_with_stable_degree"]:
        notes.append(
            "Education score varies while required_degree and candidate "
            "degrees are stable, so that variance comes from the education "
            "scorer, not from the extracted degree field."
        )

    career = llm_vs_extract["career_relevance"]
    if career.get("production_range_zero_when_relations_stable"):
        notes.append(
            "Production career relevance range is 0 while validated "
            "target-family and preferred-industry relations are stable. "
            "The raw model integer is not the production score."
        )
    elif career.get("relations_stable"):
        notes.append(
            "WARNING: identical career-alignment relations produced "
            "different production career relevance scores."
        )
    else:
        notes.append(
            "Career relevance is scored in Python from validated "
            "target-family and preferred-industry relations. Production "
            "variance follows those relations, not the raw model integer."
        )

    stance_unique = categorical_summary(runs, "sponsorship_stance")["unique_count"]
    notes.append(
        f"Final sponsorship stance had {stance_unique} unique value(s). "
        "Python overwrites LLM stance from the JD; an invented quote that "
        "is not in the source is ignored."
    )

    if runs and "evidence_units" in runs[0]:
        coverage = evidence_unit_stability(runs)
        notes.append(
            "Evidence-unit coverage: "
            f"{coverage['complete_coverage_count']}/{len(runs)} runs returned "
            "a decision for every unit; "
            f"{coverage['unique_omitted_sets']} unique omitted-unit set(s); "
            f"{coverage['inconsistent_unit_count']} unit(s) received "
            "inconsistent classifications; required-list cardinalities="
            f"{coverage['required_list_cardinalities']}."
        )

    return notes


def load_job_text(url: str | None, jd_path: str | None) -> tuple[str, dict]:
    if jd_path:
        text = Path(jd_path).read_text(encoding="utf-8").strip()
        source = {"type": "file", "path": jd_path}
    elif url:
        from job_search_agent.page_loader import load_job_description_from_url

        text = load_job_description_from_url(url)
        source = {"type": "url", "url": url, "fetched_once": True}
    else:
        raise ValueError("Provide --url or --jd.")

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    source["sha256"] = digest
    source["characters"] = len(text)
    return text, source


def record_run(
    index: int,
    analysis,
    snapshot: JobEvaluation | None,
    job_description: str,
    profile,
    scoring,
) -> dict:
    diagnostics = diagnose_skills(
        profile,
        analysis.requirements,
        scoring.skills,
    )
    score_only = get_recommendation(
        analysis.overall_score,
        scoring.thresholds,
    )
    llm_language = (
        snapshot.requirements.sponsorship_language if snapshot else None
    )
    llm_stance = (
        snapshot.requirements.sponsorship.value if snapshot else None
    )

    return {
        "run": index,
        "company": analysis.company,
        "title": analysis.job_title,
        "location": analysis.location,
        "required_skills": analysis.requirements.required_skills,
        "preferred_skills": analysis.requirements.preferred_skills,
        "minimum_years_experience": (
            analysis.requirements.minimum_years_experience
        ),
        "required_degree": analysis.requirements.required_degree,
        "llm_sponsorship_stance": llm_stance,
        "llm_sponsorship_language": llm_language,
        "llm_sponsorship_language_grounded": quote_in_source(
            job_description,
            llm_language,
        ),
        "validated_sponsorship_language": (
            analysis.requirements.sponsorship_language
        ),
        "sponsorship_stance": analysis.requirements.sponsorship.value,
        "llm_scores": (
            {
                "skills": snapshot.scores.skills,
                "education": snapshot.scores.education,
                "experience": snapshot.scores.experience,
                "career_relevance": snapshot.scores.career_relevance,
            }
            if snapshot
            else None
        ),
        "python_overwrote_llm_skills_score": (
            snapshot.scores.skills != analysis.scores.skills
            if snapshot
            else None
        ),
        "python_overwrote_llm_experience_score": (
            snapshot.scores.experience != analysis.scores.experience
            if snapshot
            else None
        ),
        "python_overwrote_llm_education_score": (
            snapshot.scores.education != analysis.scores.education
            if snapshot
            else None
        ),
        "candidate_years_of_experience": profile.years_of_experience,
        "candidate_degrees": [item.degree for item in profile.education],
        "skills_score": analysis.scores.skills,
        "education_score": analysis.scores.education,
        "experience_score": analysis.scores.experience,
        "llm_career_relevance_score": (
            snapshot.scores.career_relevance if snapshot else None
        ),
        "target_family_alignment": (
            analysis.career_alignment.target_family_relation.value
        ),
        "preferred_industry_alignment": (
            analysis.career_alignment.preferred_industry_relation.value
        ),
        "llm_target_family_alignment": (
            snapshot.career_alignment.target_family_relation.value
            if snapshot
            else None
        ),
        "llm_preferred_industry_alignment": (
            snapshot.career_alignment.preferred_industry_relation.value
            if snapshot
            else None
        ),
        "career_relevance_score": analysis.scores.career_relevance,
        "python_overwrote_llm_career_relevance_score": (
            snapshot.scores.career_relevance != analysis.scores.career_relevance
            if snapshot
            else None
        ),
        "overall_score": analysis.overall_score,
        "score_based_recommendation": score_only.value,
        "final_recommendation": analysis.recommendation.value,
        "passes_hard_filters": analysis.passes_hard_filters,
        "hard_filter_reasons": analysis.hard_filter_reasons,
        "skills_diagnostics": diagnostics,
        "recomputed_skills_score_matches": (
            diagnostics["skills_score"] == analysis.scores.skills
        ),
        "evidence_units": describe_evidence_units(snapshot, job_description),
        "skill_extraction": analysis.skill_extraction.model_dump(),
    }


def print_run_line(row: dict) -> None:
    print(
        f"run {row['run']:02d}  overall={row['overall_score']:5.1f}  "
        f"skills={row['skills_score']:3}  edu={row['education_score']:3}  "
        f"exp={row['experience_score']:3}  "
        f"rel={row['career_relevance_score']:3}  "
        f"req_skills={len(row['required_skills']):2}  "
        f"final={row['final_recommendation']}",
        flush=True,
    )


def print_summary(report: dict) -> None:
    print()
    print("=" * 72)
    print("REPEATABILITY SUMMARY")
    print("=" * 72)
    source = report["job_source"]
    print(
        f"Source: {source.get('url') or source.get('path')}  "
        f"chars={source['characters']}  sha256={source['sha256'][:12]}..."
    )
    print(f"Runs: {report['runs']}")
    print()
    print("Numeric scores")
    for field, stats in report["numeric_summary"].items():
        print(
            f"  {field:24} values={stats['values']}  "
            f"min={stats['minimum']} max={stats['maximum']}  "
            f"mean={stats['mean']} sd={stats['std_dev']}"
        )

    print()
    print("Experience scoring")
    experience = (report.get("llm_scores_versus_extraction") or {}).get(
        "experience"
    ) or {}
    llm_stats = experience.get("llm_score_summary") or {}
    prod_stats = experience.get("production_score_summary") or {}
    print(
        f"  LLM experience           values={llm_stats.get('values')}  "
        f"min={llm_stats.get('minimum')} max={llm_stats.get('maximum')}  "
        f"range={llm_stats.get('range')}"
    )
    print(
        f"  production experience    values={prod_stats.get('values')}  "
        f"min={prod_stats.get('minimum')} max={prod_stats.get('maximum')}  "
        f"range={prod_stats.get('range')}"
    )
    print(
        "  production_range_zero_when_years_stable="
        f"{experience.get('production_range_zero_when_years_stable')}  "
        f"python_overwrote_runs={experience.get('python_overwrote_llm_experience_score')}"
    )

    print()
    print("Education scoring")
    education = (report.get("llm_scores_versus_extraction") or {}).get(
        "education"
    ) or {}
    edu_llm = education.get("llm_score_summary") or {}
    edu_prod = education.get("production_score_summary") or {}
    print(
        f"  LLM education            values={edu_llm.get('values')}  "
        f"min={edu_llm.get('minimum')} max={edu_llm.get('maximum')}  "
        f"range={edu_llm.get('range')}"
    )
    print(
        f"  production education     values={edu_prod.get('values')}  "
        f"min={edu_prod.get('minimum')} max={edu_prod.get('maximum')}  "
        f"range={edu_prod.get('range')}"
    )
    print(
        "  production_range_zero_when_education_inputs_stable="
        f"{education.get('production_range_zero_when_education_inputs_stable')}  "
        f"python_overwrote_runs={education.get('python_overwrote_llm_education_score')}"
    )

    print()
    print("Career relevance scoring")
    career = (report.get("llm_scores_versus_extraction") or {}).get(
        "career_relevance"
    ) or {}
    career_llm = career.get("llm_score_summary") or {}
    career_prod = career.get("production_score_summary") or {}
    print(
        f"  raw LLM career relevance values={career_llm.get('values')}  "
        f"min={career_llm.get('minimum')} max={career_llm.get('maximum')}  "
        f"range={career_llm.get('range')}"
    )
    print(
        f"  target_family_alignment: "
        f"{career.get('target_family_alignment')}"
    )
    print(
        f"  preferred_industry_alignment: "
        f"{career.get('preferred_industry_alignment')}"
    )
    print(
        f"  production career relevance values={career_prod.get('values')}  "
        f"min={career_prod.get('minimum')} max={career_prod.get('maximum')}  "
        f"range={career_prod.get('range')}"
    )
    print(
        "  production_range_zero_when_relations_stable="
        f"{career.get('production_range_zero_when_relations_stable')}  "
        "python_overwrote_runs="
        f"{career.get('python_overwrote_llm_career_relevance_score')}"
    )

    print()
    print("Categorical / extracted fields")
    for field, summary in report["categorical_summary"].items():
        print(f"  {field}: {summary['unique_count']} unique value(s)")
        for item in summary["frequencies"]:
            print(
                f"    n={item['count']} runs={item['runs']}  "
                f"value={item['value']!r}"
            )

    print()
    print("Recommendation stability")
    print(f"  final: {report['recommendation_stability']['final']}")
    print(
        "  score-based (thresholds only): "
        f"{report['recommendation_stability']['score_based']}"
    )

    print()
    print("Hard-filter stability")
    filters = report["hard_filter_stability"]
    print(f"  failed={filters['runs_failed']} passed={filters['runs_passed']}")
    for item in filters["reason_frequencies"]:
        print(
            f"    n={item['count']} runs={item['runs']}  {item['reason']}"
        )

    print()
    print("Evidence-unit coverage")
    coverage = report["evidence_unit_stability"]
    print(
        f"  unit_count_stable={coverage['unit_count_stable']}  "
        f"values={coverage['unit_count_values']}"
    )
    print(
        f"  candidate_unit_count_stable="
        f"{coverage.get('candidate_unit_count_stable')}  "
        f"values={coverage.get('candidate_unit_count_values')}"
    )
    print(
        f"  complete_coverage={coverage['complete_coverage_count']}/{report['runs']}  "
        f"runs={coverage['runs_with_complete_coverage']}"
    )
    print(
        f"  unique omitted-unit sets: {coverage['unique_omitted_sets']}"
    )
    for item in coverage["omitted_sets"]:
        print(
            f"    n={item['count']} runs={item['runs']}  "
            f"omitted={item['value']!r}"
        )
    print(
        f"  unique units-with-skills sets: "
        f"{coverage['unique_units_with_skills_sets']}"
    )
    for item in coverage["units_with_skills_sets"]:
        print(
            f"    n={item['count']} runs={item['runs']}  "
            f"units={item['value']!r}"
        )
    print(
        f"  required list cardinalities: "
        f"{coverage['required_list_cardinalities']}"
    )
    print(
        f"  inconsistent unit classifications: "
        f"{coverage['inconsistent_unit_count']}"
    )
    for item in coverage["inconsistent_unit_classifications"]:
        print(
            f"    {item['unit_id']}: {item['unique_signatures']} signatures"
        )
        for signature in item["signatures"]:
            print(
                f"      n={signature['count']} runs={signature['runs']}  "
                f"{signature['value']!r}"
            )

    print()
    print("Dedicated skill extraction")
    dedicated = report.get("dedicated_skill_stability") or {}
    print(
        f"  candidate_unit_count={dedicated.get('candidate_unit_count')}  "
        f"runs_needing_repair={dedicated.get('runs_needing_repair')}  "
        f"unresolved_runs={dedicated.get('unresolved_runs')}"
    )
    print(f"  repaired_unit_ids: {dedicated.get('repaired_unit_ids_by_run')}")
    print(f"  u010 across runs: {dedicated.get('u010_by_run')}")
    print(
        "  empty inventory scored as 100 from score_skills: "
        f"{dedicated.get('empty_inventory_scored_100')}"
    )
    print(
        f"  skills_scored_from_inventory runs: "
        f"{dedicated.get('scored_from_inventory_runs')}"
    )
    jaccard = dedicated.get("required_jaccard") or {}
    print(
        f"  required Jaccard mean={jaccard.get('mean')}  "
        f"min={jaccard.get('minimum')}  max={jaccard.get('maximum')}"
    )
    for item in dedicated.get("per_unit_skill_frequencies") or []:
        print(
            f"    {item['unit_id']}: {item['unique_count']} unique inventories"
        )
        for freq in item["frequencies"]:
            print(
                f"      n={freq['count']} runs={freq['runs']}  "
                f"{freq['value']!r}"
            )

    print()
    print("Root-cause notes")
    for note in report["root_cause_notes"]:
        print(f"  - {note}")

    swing = report["largest_skills_swing"]
    if swing and swing.get("changed"):
        print()
        print("Largest skills-score swing")
        print(f"  low:  {json.dumps(swing['low'], ensure_ascii=False)}")
        print(f"  high: {json.dumps(swing['high'], ensure_ascii=False)}")


def build_report(
    runs: list[dict],
    job_source: dict,
    job_description: str,
    scoring,
) -> dict:
    stats = {
        field: summarize([row[field] for row in runs]) for field in NUMERIC_FIELDS
    }
    contributions = {}
    for field, weight in WEIGHTS.items():
        values = [row[field] for row in runs]
        variance = statistics.variance(values) if len(values) > 1 else 0.0
        contributions[field] = round((weight**2) * variance, 4)

    dominant = max(contributions, key=contributions.get) if contributions else None

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "job_source": job_source,
        "job_description": job_description,
        "runs": len(runs),
        "observations": runs,
        "numeric_summary": stats,
        "categorical_summary": {
            field: categorical_summary(runs, field)
            for field in CATEGORICAL_FIELDS
        },
        "recommendation_stability": recommendation_stability(runs),
        "hard_filter_stability": hard_filter_stability(runs),
        "skill_extraction_traces": skill_extraction_traces(runs),
        "evidence_unit_inventory": evidence_unit_inventory(job_description),
        "evidence_unit_stability": evidence_unit_stability(runs),
        "dedicated_skill_stability": dedicated_skill_stability(runs),
        "largest_skills_swing": largest_skills_swing(runs),
        "python_skill_scorer": python_skills_consistent_on_identical_inputs(
            runs
        ),
        "python_experience_scorer": python_experience_consistent_on_identical_inputs(
            runs
        ),
        "python_education_scorer": python_education_consistent_on_identical_inputs(
            runs
        ),
        "llm_scores_versus_extraction": llm_score_fields_vs_extraction(runs),
        "root_cause_notes": root_cause_notes(runs),
        "variance_contribution_weight_squared_times_variance": contributions,
        "component_most_responsible_for_score_variance": dominant,
        "thresholds": scoring.thresholds.model_dump(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same job description through the matcher N times and "
            "report extraction/score variance. Fetches --url once."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--url",
        default=None,
        help=f"Public job URL to fetch once (default: {DEFAULT_URL})",
    )
    source.add_argument(
        "--jd",
        help="Local job-description file. Skips network fetch.",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON report path (default: evaluation/output/repeatability.json)",
    )
    parser.add_argument("--config-dir", default=str(CONFIG_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    url = args.url if args.jd else (args.url or DEFAULT_URL)
    posting, job_source = load_job_text(url, args.jd)

    print(
        f"Loaded JD once ({job_source['characters']} chars, "
        f"sha256={job_source['sha256'][:12]}...). "
        f"Reusing this text for {args.runs} analyses.",
        flush=True,
    )

    from job_search_agent.ai_client import AIClient

    client = SnapshotEvaluator(AIClient())
    scoring = load_scoring_config(Path(args.config_dir) / "scoring.yaml")
    profile = load_candidate_profile(
        Path(args.config_dir) / "candidate_profile.yaml"
    )
    runs: list[dict] = []

    for index in range(1, args.runs + 1):
        analysis = analyze_job_description(
            job_description=posting,
            ai_client=client,
            source_url=job_source.get("url"),
            config_dir=args.config_dir,
        )
        row = record_run(
            index,
            analysis,
            client.last,
            posting,
            profile,
            scoring,
        )
        runs.append(row)
        print_run_line(row)

    report = build_report(runs, job_source, posting, scoring)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_summary(report)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
