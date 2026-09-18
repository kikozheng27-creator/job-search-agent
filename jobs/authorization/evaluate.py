"""One-off adversarial evaluation of work-authorization extraction.

Not a product command. Reads fixtures under jobs/authorization/, calls the
real AIClient once per file, and prints a JSON report. Does not save to the
tracker or write processed reports.
"""

from __future__ import annotations

import json
from pathlib import Path

from job_search_agent.ai_client import AIClient
from job_search_agent.config_loader import load_scoring_config
from job_search_agent.main import analyze_job_description
from job_search_agent.scoring import get_recommendation


CASES = [
    {
        "file": "01_no_f1.txt",
        "label": "Explicitly rejects F-1 candidates",
        "sentence": "This position is not open to F-1 candidates.",
        "expected": "hard_filter",
    },
    {
        "file": "02_no_opt.txt",
        "label": "Explicitly rejects OPT candidates",
        "sentence": "We cannot consider OPT candidates for this role.",
        "expected": "hard_filter",
    },
    {
        "file": "03_no_stem_opt.txt",
        "label": "Explicitly rejects STEM OPT candidates",
        "sentence": "STEM OPT candidates are not eligible for this position.",
        "expected": "hard_filter",
    },
    {
        "file": "04_permanent_unrestricted.txt",
        "label": "Requires permanent/unrestricted U.S. work authorization",
        "sentence": (
            "Applicants must have permanent unrestricted U.S. work authorization."
        ),
        "expected": "hard_filter",
    },
    {
        "file": "05_citizens_or_pr_only.txt",
        "label": "U.S. citizens or permanent residents only",
        "sentence": (
            "This position is open to U.S. citizens or permanent residents only."
        ),
        "expected": "hard_filter",
    },
    {
        "file": "06_no_h1b.txt",
        "label": "Explicitly says no H-1B sponsorship",
        "sentence": "We are unable to provide H-1B sponsorship for this position.",
        "expected": "preserve_no_hard_filter",
    },
    {
        "file": "07_generic_no_sponsorship.txt",
        "label": "Generic we do not provide visa sponsorship",
        "sentence": "We do not provide visa sponsorship.",
        "expected": "preserve_with_concern",
    },
    {
        "file": "08_no_sponsorship_now_or_future.txt",
        "label": "No sponsorship now or in the future",
        "sentence": (
            "Candidates must not now or in the future require visa sponsorship."
        ),
        "expected": "hard_filter",
    },
    {
        "file": "09_ambiguous.txt",
        "label": "Ambiguous authorization wording",
        "sentence": "Applicants must be authorized to work in the United States.",
        "expected": "preserve_no_concern",
    },
    {
        "file": "10_silence.txt",
        "label": "No sponsorship/work-authorization wording at all",
        "sentence": None,
        "expected": "preserve_no_concern",
    },
]


def verdict(row: dict) -> str:
    expected = row["expected"]
    filtered = not row["passes_hard_filters"]
    has_concern = bool(row["concerns"])

    if expected == "hard_filter":
        return "PASS" if filtered else "FAIL: expected hard filter"
    if expected == "preserve_no_hard_filter":
        return "PASS" if not filtered else "FAIL: expected preserve"
    if expected == "preserve_with_concern":
        if filtered:
            return "FAIL: expected preserve with concern, got hard filter"
        if not has_concern:
            return "FAIL: expected concern"
        return "PASS"
    if expected == "preserve_no_concern":
        if filtered:
            return "FAIL: expected preserve, no concern, got hard filter"
        if has_concern:
            return "FAIL: expected no concern"
        return "PASS"
    return "OBSERVE"


def main() -> None:
    fixtures = Path("jobs/authorization")
    client = AIClient()
    scoring = load_scoring_config("config/scoring.yaml")
    rows = []

    for case in CASES:
        posting = (fixtures / case["file"]).read_text(encoding="utf-8")
        analysis = analyze_job_description(
            job_description=posting,
            ai_client=client,
            source_url=None,
        )

        score_only = get_recommendation(
            analysis.overall_score,
            scoring.thresholds,
        )
        auth_reasons = [
            reason
            for reason in analysis.hard_filter_reasons
            if any(
                token in reason.lower()
                for token in (
                    "sponsorship",
                    "authorization",
                    "citizenship",
                    "permanent",
                    "residency",
                )
            )
        ]

        row = {
            "file": case["file"],
            "label": case["label"],
            "exact_authorization_sentence": case["sentence"],
            "extracted_stance": analysis.requirements.sponsorship.value,
            "sponsorship_language": analysis.requirements.sponsorship_language,
            "passes_hard_filters": analysis.passes_hard_filters,
            "hard_filter_reasons": analysis.hard_filter_reasons,
            "concerns": analysis.concerns,
            "overall_score": analysis.overall_score,
            "recommendation": analysis.recommendation.value,
            "score_only_recommendation": score_only.value,
            "authorization_changed_recommendation": (
                analysis.recommendation != score_only and bool(auth_reasons)
            ),
            "authorization_changed_score": False,
            "expected": case["expected"],
        }
        row["result"] = verdict(row)
        rows.append(row)

        print(
            f"{row['result']:6}  {row['file']}  "
            f"stance={row['extracted_stance']}  "
            f"filter={'FAIL' if not row['passes_hard_filters'] else 'pass'}  "
            f"concerns={len(row['concerns'])}  "
            f"{row['recommendation']}",
            flush=True,
        )

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
