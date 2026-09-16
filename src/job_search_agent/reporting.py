import re
from pathlib import Path

from job_search_agent.models import JobAnalysis


def print_report(result: JobAnalysis) -> None:
    print()
    print("=" * 60)
    print(f"Company: {result.company}")
    print(f"Role: {result.job_title}")
    print("=" * 60)

    print()
    print(f"Overall Match: {result.overall_score} / 100")
    print(f"Recommendation: {result.recommendation.value}")

    print()
    print("Scores")
    print(f"  Skills:            {result.skills_score}")
    print(f"  Education:         {result.education_score}")
    print(f"  Experience:        {result.experience_score}")
    print(f"  Career Relevance:  {result.career_relevance_score}")

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