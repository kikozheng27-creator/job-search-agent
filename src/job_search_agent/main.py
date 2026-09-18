import argparse
import sys
import logging
from pathlib import Path

from pydantic import ValidationError

from job_search_agent.ai_client import AIClient
from job_search_agent.config_loader import load_config
from job_search_agent.job_matcher import JobMatcher
from job_search_agent.logging_config import setup_logging
from job_search_agent.models import CandidateProfile
from job_search_agent.reporting import print_report, save_report

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze how well a job matches the candidate profile."
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--jd",
        help="Path to a text file containing the job description.",
    )

    group.add_argument(
        "--stdin",
        action="store_true",
        help="Read the job description from standard input.",
    )

    return parser.parse_args()


def read_job_description(path: str) -> str:
    jd_path = Path(path)

    if not jd_path.exists():
        raise FileNotFoundError(
            f"Job description file not found: {path}"
        )

    return jd_path.read_text(encoding="utf-8")

def read_job_description_from_stdin() -> str:
    job_description = sys.stdin.read().strip()

    if not job_description:
        raise ValueError("No job description was provided through stdin.")

    return job_description

def get_job_description(args) -> str:
    if args.stdin:
        logging.info("Reading job description from stdin...")
        return read_job_description_from_stdin()

    logging.info("Reading job description from file...")
    return read_job_description(args.jd)

def run() -> None:
    args = parse_args()

    job_description = get_job_description(args)

    logging.info("Loading candidate profile...")
    raw_profile = load_config("config/candidate_profile.yaml")
    profile = CandidateProfile.model_validate(raw_profile)

    logging.info("Loading scoring configuration...")
    scoring_config = load_config("config/scoring.yaml")

    logging.info("Loading filter configuration...")
    filter_config = load_config("config/filters.yaml")

    logging.info("Initializing AI client...")
    ai_client = AIClient()

    matcher = JobMatcher(
    ai_client=ai_client,
    weights=scoring_config["weights"],
    thresholds=scoring_config["thresholds"],
    filter_config=filter_config,
    )

    logging.info("Analyzing job...")
    result = matcher.match(
        profile=profile,
        job_description=job_description,
    )

    print_report(result)

    output_path = save_report(result)

    logging.info("Report saved to %s", output_path)


def main() -> int:
    setup_logging()

    try:
        run()

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