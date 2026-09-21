import shutil
from pathlib import Path

import pytest
from conftest import (
    FakeAIClient,
    make_analysis,
    make_evaluation,
    make_requirements,
)
from pydantic import ValidationError

from job_search_agent.main import (
    analyze_job_description,
    build_parser,
    get_job_description,
    main,
    parse_args,
)
from job_search_agent.models import Recommendation, SponsorshipStance
from job_search_agent.tracker import ApplicationStatus, JobTracker


REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG_DIR = REPO_ROOT / "config"


def entry_level_evaluation():
    """A posting the shipped filters.yaml lets through to weighted scoring."""
    return make_evaluation(
        requirements=make_requirements(minimum_years_experience=1),
    )


# --- integration boundary: shipped config + extracted requirements ----------


def test_analysis_runs_against_the_shipped_configuration():
    """Exercises the real config files with a fake AI client."""
    ai_client = FakeAIClient(entry_level_evaluation())

    result = analyze_job_description(
        job_description="Example Pharma is hiring a Biostatistician.",
        ai_client=ai_client,
        source_url="https://example.com/jobs/1",
        config_dir=SHIPPED_CONFIG_DIR,
    )

    assert result.company == "Example Pharma"
    assert result.job_title == "Biostatistician"
    assert result.overall_score == 77.5
    assert result.source_url == "https://example.com/jobs/1"
    assert result.passes_hard_filters is True
    assert result.recommendation == Recommendation.APPLY


def test_shipped_configuration_filters_a_senior_posting():
    """A high-scoring posting is still skipped when a hard filter fails.

    The shipped filters allow up to 2 years plus 2 years of tolerance, so a
    posting demanding 5 years is out of range for an entry-level candidate.
    """
    evaluation = make_evaluation(
        requirements=make_requirements(minimum_years_experience=5),
    )

    result = analyze_job_description(
        job_description="Example",
        ai_client=FakeAIClient(evaluation),
        config_dir=SHIPPED_CONFIG_DIR,
    )

    assert result.overall_score == 69.0
    assert result.passes_hard_filters is False
    assert result.recommendation == Recommendation.SKIP
    assert len(result.hard_filter_reasons) == 1


def test_shipped_configuration_allows_a_posting_inside_the_tolerance():
    """3 and 4 years must reach weighted scoring rather than being filtered."""
    for required_years in (3, 4):
        evaluation = make_evaluation(
            requirements=make_requirements(
                minimum_years_experience=required_years
            ),
        )

        result = analyze_job_description(
            job_description="Example",
            ai_client=FakeAIClient(evaluation),
            config_dir=SHIPPED_CONFIG_DIR,
        )

        assert result.passes_hard_filters is True
        assert result.recommendation != Recommendation.SKIP


def test_extracted_requirements_survive_to_the_final_analysis():
    evaluation = make_evaluation(
        requirements=make_requirements(
            minimum_years_experience=1,
            employment_type="full_time",
            seniority_level="entry_level",
            sponsorship=SponsorshipStance.OFFERED,
            sponsorship_language="We do not provide visa sponsorship.",
            salary_range={
                "minimum": 90000,
                "maximum": 110000,
                "currency": "USD",
                "period": "yearly",
            },
        ),
    )

    result = analyze_job_description(
        job_description=(
            "Example Pharma is hiring.\n"
            "We do not provide visa sponsorship."
        ),
        ai_client=FakeAIClient(evaluation),
        config_dir=SHIPPED_CONFIG_DIR,
    )

    requirements = result.requirements

    assert requirements.employment_type.value == "full_time"
    assert requirements.seniority_level.value == "entry_level"
    assert requirements.salary_range.maximum == 110000
    assert requirements.sponsorship_language == "We do not provide visa sponsorship."
    assert requirements.sponsorship is SponsorshipStance.NOT_OFFERED


def test_the_real_profile_is_sent_to_the_model():
    ai_client = FakeAIClient(make_evaluation())

    analyze_job_description(
        job_description="Example",
        ai_client=ai_client,
        config_dir=SHIPPED_CONFIG_DIR,
    )

    assert "Biostatistics" in ai_client.prompts[0]


def test_incomplete_profile_configuration_is_rejected(tmp_path):
    (tmp_path / "candidate_profile.yaml").write_text(
        "target_job_families: []\n", encoding="utf-8"
    )

    with pytest.raises(ValidationError):
        analyze_job_description(
            job_description="Example",
            ai_client=FakeAIClient(make_evaluation()),
            config_dir=tmp_path,
        )


def test_missing_configuration_directory_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_job_description(
            job_description="Example",
            ai_client=FakeAIClient(make_evaluation()),
            config_dir=tmp_path / "nope",
        )


# --- argument parsing -------------------------------------------------------


def test_analyze_requires_a_job_description_source():
    with pytest.raises(SystemExit):
        parse_args(["analyze"])


def test_jd_and_stdin_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["analyze", "--jd", "a.txt", "--stdin"])


def test_url_alone_is_an_accepted_source():
    args = parse_args(["analyze", "--url", "https://example.com/jobs/1"])

    assert args.url == "https://example.com/jobs/1"
    assert args.jd is None
    assert args.stdin is False


def test_jd_can_still_record_a_source_url():
    args = parse_args(
        ["analyze", "--jd", "jobs/example.txt", "--url", "https://example.com/jobs/1"]
    )

    assert args.jd == "jobs/example.txt"
    assert args.url == "https://example.com/jobs/1"


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_unknown_status_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["list", "--status", "GHOSTED"])


def test_analyze_defaults_to_saved_status():
    args = build_parser().parse_args(["analyze", "--stdin"])

    assert args.status == ApplicationStatus.SAVED.value
    assert args.save is False


# --- list command -----------------------------------------------------------


def test_list_on_an_empty_database(tmp_path, capsys):
    exit_code = main(["list", "--database", str(tmp_path / "tracker.db")])

    assert exit_code == 0
    assert "No jobs tracked yet" in capsys.readouterr().out


def test_list_shows_saved_jobs(tmp_path, capsys):
    database = tmp_path / "tracker.db"

    with JobTracker(database) as tracker:
        tracker.save(make_analysis(), status=ApplicationStatus.APPLIED)

    exit_code = main(["list", "--database", str(database)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Example Pharma" in output
    assert "APPLIED" in output


def test_list_can_filter_by_status(tmp_path, capsys):
    database = tmp_path / "tracker.db"

    with JobTracker(database) as tracker:
        tracker.save(make_analysis(), status=ApplicationStatus.APPLIED)

    exit_code = main(
        ["list", "--database", str(database), "--status", "REJECTED"]
    )

    assert exit_code == 0
    assert "No jobs tracked yet" in capsys.readouterr().out


# --- error handling ---------------------------------------------------------


def test_missing_job_description_file_exits_1(tmp_path):
    exit_code = main(
        [
            "analyze",
            "--jd",
            str(tmp_path / "missing.txt"),
            "--database",
            str(tmp_path / "tracker.db"),
        ]
    )

    assert exit_code == 1


def test_empty_job_description_file_exits_1(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")

    exit_code = main(
        [
            "analyze",
            "--jd",
            str(empty),
            "--database",
            str(tmp_path / "tracker.db"),
        ]
    )

    assert exit_code == 1


# --- full analyze workflow, no network --------------------------------------


def test_analyze_and_save_writes_to_the_tracker(tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.copytree(SHIPPED_CONFIG_DIR, workspace / "config")

    posting = workspace / "posting.txt"
    posting.write_text("Example Pharma is hiring.", encoding="utf-8")

    monkeypatch.setattr(
        "job_search_agent.ai_client.AIClient",
        lambda: FakeAIClient(entry_level_evaluation()),
    )
    monkeypatch.chdir(workspace)

    database = workspace / "tracker.db"

    exit_code = main(
        [
            "analyze",
            "--jd",
            str(posting),
            "--url",
            "https://example.com/jobs/7",
            "--save",
            "--status",
            "APPLIED",
            "--notes",
            "Applied via referral",
            "--database",
            str(database),
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Overall Match: 77.5 / 100" in output
    assert "Recommendation: APPLY" in output

    with JobTracker(database) as tracker:
        jobs = tracker.list_jobs()

    assert len(jobs) == 1
    assert jobs[0].status == ApplicationStatus.APPLIED
    assert jobs[0].notes == "Applied via referral"
    assert jobs[0].source_url == "https://example.com/jobs/7"
    assert jobs[0].match_score == 77.5


def test_analyze_without_save_leaves_the_tracker_empty(
    tmp_path, monkeypatch, capsys
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.copytree(SHIPPED_CONFIG_DIR, workspace / "config")

    posting = workspace / "posting.txt"
    posting.write_text("Example Pharma is hiring.", encoding="utf-8")

    monkeypatch.setattr(
        "job_search_agent.ai_client.AIClient",
        lambda: FakeAIClient(entry_level_evaluation()),
    )
    monkeypatch.chdir(workspace)

    database = workspace / "tracker.db"

    exit_code = main(
        ["analyze", "--jd", str(posting), "--database", str(database)]
    )

    assert exit_code == 0
    assert "Overall Match" in capsys.readouterr().out

    with JobTracker(database) as tracker:
        assert tracker.list_jobs() == []


def test_analyze_url_feeds_extracted_text_into_the_matcher(
    tmp_path, monkeypatch, capsys
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.copytree(SHIPPED_CONFIG_DIR, workspace / "config")

    html = (
        Path(__file__).resolve().parent / "fixtures" / "sample_job_page.html"
    ).read_text(encoding="utf-8")
    captured: list[str] = []

    def fake_fetch(url: str) -> str:
        captured.append(url)
        return html

    monkeypatch.setattr("job_search_agent.page_loader.fetch_html", fake_fetch)
    monkeypatch.setattr(
        "job_search_agent.ai_client.AIClient",
        lambda: FakeAIClient(entry_level_evaluation()),
    )
    monkeypatch.chdir(workspace)

    exit_code = main(
        ["analyze", "--url", "https://example.com/jobs/42"]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert captured == ["https://example.com/jobs/42"]
    assert "Overall Match" in output
    assert "Example Pharma" in output


def test_get_job_description_from_stdin_ignores_url(monkeypatch):
    monkeypatch.setattr(
        "job_search_agent.main.sys.stdin",
        type("Stdin", (), {"read": lambda self: "stdin posting text"})(),
    )

    args = parse_args(
        ["analyze", "--stdin", "--url", "https://example.com/jobs/1"]
    )

    assert get_job_description(args) == "stdin posting text"


def test_get_job_description_from_url_uses_the_loader(monkeypatch):
    monkeypatch.setattr(
        "job_search_agent.main.load_job_description_from_url",
        lambda url: f"extracted from {url}",
    )

    args = parse_args(["analyze", "--url", "https://example.com/jobs/1"])

    assert get_job_description(args) == "extracted from https://example.com/jobs/1"
