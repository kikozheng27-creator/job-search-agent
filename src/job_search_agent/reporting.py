import re
from pathlib import Path

from job_search_agent.models import JobAnalysis
from job_search_agent.tracker import TrackedJob


def print_report(result: JobAnalysis) -> None:
    print()
    print("=" * 60)
    print(f"Company: {result.company}")
    print(f"Role: {result.job_title}")

    if result.location:
        print(f"Location: {result.location}")

    print("=" * 60)

    print()
    print(f"Overall Match: {result.overall_score} / 100")
    print(f"Recommendation: {result.recommendation.value}")

    print()
    print("Scores")
    print(f"  Skills:            {result.scores.skills}")
    print(f"  Education:         {result.scores.education}")
    print(f"  Experience:        {result.scores.experience}")
    print(f"  Career Relevance:  {result.scores.career_relevance}")

    if not result.passes_hard_filters:
        print()
        print("Hard Filters: FAILED")
        for reason in result.hard_filter_reasons:
            print(f"  - {reason}")

    if result.concerns:
        print()
        print("Concerns (not blocking)")
        for concern in result.concerns:
            print(f"  - {concern}")

    print()
    print("Strengths")
    for strength in result.strengths:
        print(f"  - {strength}")

    print()
    print("Missing Requirements")
    for requirement in result.missing_requirements:
        print(f"  - {requirement}")

    print()
    print("Reasoning")
    print(result.reasoning)


def print_tracked_jobs(jobs: list[TrackedJob]) -> None:
    if not jobs:
        print("No jobs tracked yet. Run 'analyze --save' to add one.")
        return

    header = (
        f"{'ID':>4}  {'DATE':<10}  {'SCORE':>5}  "
        f"{'RECOMMENDATION':<15}  {'STATUS':<10}  COMPANY / ROLE"
    )

    print()
    print(header)
    print("-" * len(header))

    for job in jobs:
        print(
            f"{job.id:>4}  "
            f"{job.analyzed_at.date().isoformat():<10}  "
            f"{job.match_score:>5.1f}  "
            f"{job.recommendation.value:<15}  "
            f"{job.status.value:<10}  "
            f"{job.company} / {job.job_title}"
        )

    print()
    print(f"{len(jobs)} job(s).")


def save_report(
    result: JobAnalysis,
    output_dir: str = "data/processed",
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    company = make_safe_filename(result.company)
    role = make_safe_filename(result.job_title)

    output_path = directory / f"{company}_{role}.json"

    output_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return output_path


def make_safe_filename(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)

    return value.strip("_")
