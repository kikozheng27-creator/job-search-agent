import sqlite3
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import TracebackType

from pydantic import BaseModel

from job_search_agent.models import JobAnalysis, Recommendation


DEFAULT_DATABASE_PATH = "data/tracker.db"

IN_MEMORY = ":memory:"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company TEXT NOT NULL,
    job_title TEXT NOT NULL,
    location TEXT,
    source_url TEXT,
    match_score REAL NOT NULL,
    recommendation TEXT NOT NULL,
    analyzed_at TEXT NOT NULL,
    status TEXT NOT NULL,
    notes TEXT
);
"""


class ApplicationStatus(str, Enum):
    SAVED = "SAVED"
    APPLIED = "APPLIED"
    INTERVIEW = "INTERVIEW"
    REJECTED = "REJECTED"
    OFFER = "OFFER"
    WITHDRAWN = "WITHDRAWN"


class TrackedJob(BaseModel):
    id: int
    company: str
    job_title: str
    location: str | None
    source_url: str | None
    match_score: float
    recommendation: Recommendation
    analyzed_at: datetime
    status: ApplicationStatus
    notes: str | None


class JobTracker:
    """Local SQLite store of analyzed jobs and their application status."""

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        if str(database_path) != IN_MEMORY:
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(database_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)

    def __enter__(self) -> "JobTracker":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def save(
        self,
        analysis: JobAnalysis,
        status: ApplicationStatus = ApplicationStatus.SAVED,
        notes: str | None = None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO jobs (
                company, job_title, location, source_url, match_score,
                recommendation, analyzed_at, status, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis.company,
                analysis.job_title,
                analysis.location,
                analysis.source_url,
                analysis.overall_score,
                analysis.recommendation.value,
                analysis.analyzed_at.isoformat(),
                status.value,
                notes,
            ),
        )

        self.connection.commit()

        return int(cursor.lastrowid)

    def get(self, job_id: int) -> TrackedJob | None:
        row = self.connection.execute(
            "SELECT * FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()

        return TrackedJob.model_validate(dict(row)) if row else None

    def list_jobs(
        self,
        status: ApplicationStatus | None = None,
        limit: int | None = None,
    ) -> list[TrackedJob]:
        query = "SELECT * FROM jobs"
        parameters: list[object] = []

        if status is not None:
            query += " WHERE status = ?"
            parameters.append(status.value)

        query += " ORDER BY analyzed_at DESC, id DESC"

        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)

        rows = self.connection.execute(query, parameters).fetchall()

        return [TrackedJob.model_validate(dict(row)) for row in rows]

    def update_status(
        self,
        job_id: int,
        status: ApplicationStatus,
        notes: str | None = None,
    ) -> bool:
        """Return False when no job has the given id."""
        if notes is None:
            cursor = self.connection.execute(
                "UPDATE jobs SET status = ? WHERE id = ?",
                (status.value, job_id),
            )
        else:
            cursor = self.connection.execute(
                "UPDATE jobs SET status = ?, notes = ? WHERE id = ?",
                (status.value, notes, job_id),
            )

        self.connection.commit()

        return cursor.rowcount > 0
