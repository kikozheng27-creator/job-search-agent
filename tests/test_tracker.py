from datetime import datetime

import pytest
from conftest import make_analysis, make_requirements

from job_search_agent.models import Recommendation
from job_search_agent.tracker import ApplicationStatus, JobTracker


@pytest.fixture
def tracker():
    with JobTracker(":memory:") as tracker:
        yield tracker


def test_save_then_read_round_trips_every_field(tracker):
    analysis = make_analysis(
        overall_score=88.0,
        recommendation=Recommendation.STRONGLY_APPLY,
        source_url="https://example.com/jobs/1",
        analyzed_at=datetime(2026, 9, 16, 12, 30),
    )

    job_id = tracker.save(analysis, notes="Referral from a classmate")
    saved = tracker.get(job_id)

    assert saved is not None
    assert saved.company == "Example Pharma"
    assert saved.job_title == "Biostatistician"
    assert saved.location == "Boston, MA"
    assert saved.source_url == "https://example.com/jobs/1"
    assert saved.match_score == 88.0
    assert saved.recommendation == Recommendation.STRONGLY_APPLY
    assert saved.analyzed_at == datetime(2026, 9, 16, 12, 30)
    assert saved.status == ApplicationStatus.SAVED
    assert saved.notes == "Referral from a classmate"


def test_save_defaults_to_saved_status_and_no_notes(tracker):
    saved = tracker.get(tracker.save(make_analysis()))

    assert saved.status == ApplicationStatus.SAVED
    assert saved.notes is None


def test_get_returns_none_for_unknown_id(tracker):
    assert tracker.get(999) is None


def test_list_jobs_is_empty_on_a_fresh_database(tracker):
    assert tracker.list_jobs() == []


def test_list_jobs_returns_newest_first(tracker):
    tracker.save(
        make_analysis(
            requirements=make_requirements(company="Older Co"),
            analyzed_at=datetime(2026, 1, 1),
        )
    )
    tracker.save(
        make_analysis(
            requirements=make_requirements(company="Newer Co"),
            analyzed_at=datetime(2026, 6, 1),
        )
    )

    companies = [job.company for job in tracker.list_jobs()]

    assert companies == ["Newer Co", "Older Co"]


def test_list_jobs_can_filter_by_status(tracker):
    tracker.save(make_analysis(), status=ApplicationStatus.APPLIED)
    tracker.save(make_analysis(), status=ApplicationStatus.SAVED)

    applied = tracker.list_jobs(status=ApplicationStatus.APPLIED)

    assert len(applied) == 1
    assert applied[0].status == ApplicationStatus.APPLIED


def test_list_jobs_respects_limit(tracker):
    for _ in range(5):
        tracker.save(make_analysis())

    assert len(tracker.list_jobs(limit=2)) == 2


@pytest.mark.parametrize("status", list(ApplicationStatus))
def test_every_application_status_can_be_stored(tracker, status):
    job_id = tracker.save(make_analysis(), status=status)

    assert tracker.get(job_id).status == status


def test_update_status_changes_status_and_keeps_notes(tracker):
    job_id = tracker.save(make_analysis(), notes="Original note")

    assert tracker.update_status(job_id, ApplicationStatus.INTERVIEW) is True

    saved = tracker.get(job_id)

    assert saved.status == ApplicationStatus.INTERVIEW
    assert saved.notes == "Original note"


def test_update_status_can_replace_notes(tracker):
    job_id = tracker.save(make_analysis(), notes="Original note")

    tracker.update_status(job_id, ApplicationStatus.OFFER, notes="Verbal offer")

    saved = tracker.get(job_id)

    assert saved.status == ApplicationStatus.OFFER
    assert saved.notes == "Verbal offer"


def test_update_status_reports_unknown_id(tracker):
    assert tracker.update_status(999, ApplicationStatus.APPLIED) is False


def test_save_handles_missing_optional_fields(tracker):
    analysis = make_analysis(
        requirements=make_requirements(location=None),
        source_url=None,
    )

    saved = tracker.get(tracker.save(analysis))

    assert saved.location is None
    assert saved.source_url is None


def test_data_persists_across_connections(tmp_path):
    database = tmp_path / "nested" / "tracker.db"

    with JobTracker(database) as tracker:
        tracker.save(make_analysis())

    with JobTracker(database) as reopened:
        assert len(reopened.list_jobs()) == 1
