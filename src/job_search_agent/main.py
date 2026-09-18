import argparse
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from job_search_agent.config_loader import (
    load_candidate_profile,
    load_filter_config,
    load_scoring_config,
)
from job_search_agent.job_matcher import JobMatcher
from job_search_agent.logging_config import setup_logging
from job_search_agent.models import JobAnalysis
from job_search_agent.page_loader import load_job_description_from_url
from job_search_agent.protocols import JobEvaluator
from job_search_agent.reporting import (
    print_report,
    print_tracked_jobs,
    save_report,
)
from job_search_agent.tracker import (
    DEFAULT_DATABASE_PATH,
    ApplicationStatus,
    JobTracker,
)


CONFIG_DIR = Path("config")

STATUS_CHOICES = [status.value for status in ApplicationStatus]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="job-search-agent",
        description="Analyze and track job postings against your profile.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze",
        help="Analyze a job description.",
    )

    source = analyze.add_mutually_exclusive_group(required=False)

    source.add_argument(
        "--jd",
        help="Path to a text file containing the job description.",
    )

    source.add_argument(
        "--stdin",
        action="store_true",
        help="Read the job description from standard input.",
    )

    analyze.add_argument(
        "--url",
        help=(
            "Public job-posting URL. Used as the input source when --jd and "
            "--stdin are omitted; otherwise recorded with the analysis."
        ),
    )

    analyze.add_argument(
        "--save",
        action="store_true",
        help="Save the analysis to the local tracker.",
    )

    analyze.add_argument(
        "--status",
        choices=STATUS_CHOICES,
        default=ApplicationStatus.SAVED.value,
        help="Application status to record with --save.",
    )

    analyze.add_argument(
        "--notes",
        help="Free-text notes to record with --save.",
    )

    listing = subparsers.add_parser(
        "list",
        help="List previously analyzed jobs.",
    )

    listing.add_argument(
        "--status",
        choices=STATUS_CHOICES,
        help="Show only jobs with this application status.",
    )

    listing.add_argument(
        "--limit",
        type=int,
        help="Show at most this many jobs.",
    )

    for subparser in (analyze, listing):
        subparser.add_argument(
            "--database",
            default=DEFAULT_DATABASE_PATH,
            help="Path to the tracker database.",
        )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze" and not args.jd and not args.stdin and not args.url:
        parser.error("one of the arguments --jd --stdin --url is required")

    return args


def read_job_description_from_file(path: str) -> str:
    jd_path = Path(path)

    if not jd_path.exists():
        raise FileNotFoundError(f"Job description file not found: {path}")

    job_description = jd_path.read_text(encoding="utf-8").strip()

    if not job_description:
        raise ValueError(f"Job description file is empty: {path}")

    return job_description


def read_job_description_from_stdin() -> str:
    job_description = sys.stdin.read().strip()

    if not job_description:
        raise ValueError("No job description was provided through stdin.")

    return job_description


def get_job_description(args: argparse.Namespace) -> str:
    if args.stdin:
        logging.info("Reading job description from stdin...")
        return read_job_description_from_stdin()

    if args.jd:
        logging.info("Reading job description from file...")
        return read_job_description_from_file(args.jd)

    logging.info("Fetching job description from URL...")
    return load_job_description_from_url(args.url)


def analyze_job_description(
    job_description: str,
    ai_client: JobEvaluator,
    source_url: str | None = None,
    config_dir: str | Path = CONFIG_DIR,
) -> JobAnalysis:
    """Load configuration and run one analysis.

    The AI client is injected so this boundary can be tested with a fake.
    """
    logging.info("Loading configuration...")
    directory = Path(config_dir)

    profile = load_candidate_profile(directory / "candidate_profile.yaml")
    scoring_config = load_scoring_config(directory / "scoring.yaml")
    filter_config = load_filter_config(directory / "filters.yaml")

    matcher = JobMatcher(
        ai_client=ai_client,
        scoring_config=scoring_config,
        filter_config=filter_config,
    )

    logging.info("Analyzing job...")
    return matcher.match(
        profile=profile,
        job_description=job_description,
        source_url=source_url,
    )


def run_analyze(args: argparse.Namespace) -> None:
    job_description = get_job_description(args)

    # Imported here so `list` works without an API key configured.
    from job_search_agent.ai_client import AIClient

    logging.info("Initializing AI client...")

    result = analyze_job_description(
        job_description=job_description,
        ai_client=AIClient(),
        source_url=args.url,
    )

    print_report(result)

    output_path = save_report(result)
    logging.info("Report saved to %s", output_path)

    if not args.save:
        return

    with JobTracker(args.database) as tracker:
        job_id = tracker.save(
            result,
            status=ApplicationStatus(args.status),
            notes=args.notes,
        )

    logging.info("Saved to tracker as job %s in %s", job_id, args.database)


def run_list(args: argparse.Namespace) -> None:
    status = ApplicationStatus(args.status) if args.status else None

    with JobTracker(args.database) as tracker:
        jobs = tracker.list_jobs(status=status, limit=args.limit)

    print_tracked_jobs(jobs)


def main(argv: list[str] | None = None) -> int:
    setup_logging()

    args = parse_args(argv)

    handlers = {
        "analyze": run_analyze,
        "list": run_list,
    }

    try:
        handlers[args.command](args)

    except FileNotFoundError as error:
        logging.error("%s", error)
        return 1

    except ValidationError as error:
        logging.error("Invalid configuration:")
        logging.error("%s", error)
        return 1

    except ValueError as error:
        logging.error("%s", error)
        return 1

    except Exception:
        logging.exception("Unexpected error")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
