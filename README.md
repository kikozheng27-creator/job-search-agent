# Job Search Agent

A local Python CLI that ingests job descriptions, extracts structured
requirements with an LLM, grounds important claims in the posting, and
produces an explainable match score and recommendation. It can also write JSON
reports and track selected jobs in a local SQLite database.

## Architecture

1. **Job-description ingestion** — `analyze` accepts a local text file, stdin,
   or a public HTTP(S) job URL. URL ingestion uses `requests` and
   `BeautifulSoup` to remove page chrome and application forms. Pages that are
   empty, not HTML, or require client-side JavaScript are rejected.
2. **Evidence units** — Python splits the cleaned posting into stable,
   numbered source units. These units are the boundary for skill extraction
   and grounding.
3. **Structured LLM extraction** — the OpenAI client requests Pydantic-backed
   structured outputs for job requirements and a dedicated per-unit skill
   inventory. The model also supplies career-relevance scoring and explanatory
   prose.
4. **Grounding and safeguards** — Python rejects unknown evidence-unit IDs,
   skills not present in their cited unit, non-skill categories, and
   contextual tool mentions that are not candidate requirements. Missing
   skill-unit coverage gets one repair pass and remains visible as an
   unresolved concern if repair fails.
5. **Deterministic component scoring** — Python computes skill overlap from
   the grounded inventory, experience from candidate years versus the stated
   minimum, and education from degree-level requirements. Education affects
   the weighted score but is not a hard filter.
6. **Sponsorship classification and hard filters** — Python classifies
   sponsorship/work-authorization language from source sentences and
   overwrites unsupported model classifications. Configurable experience,
   current-work-authorization, and location gates run independently of the
   weighted score and abstain when the posting is ambiguous.
7. **Deterministic decision** — Python calculates the weighted overall score
   from `config/scoring.yaml`. It then applies hard-filter results and score
   thresholds to choose `STRONGLY_APPLY`, `APPLY`, `MAYBE`, or `SKIP`. The LLM
   output schema has no overall-score or recommendation field.
8. **Outputs** — every analysis writes a JSON report under `data/processed/`.
   With `--save`, the CLI also stores summary data and application status in
   SQLite at `data/tracker.db` by default.

Additional hallucination safeguards verify sponsorship quotes against the
source posting and remove unsupported authorization claims from model-written
strengths, missing requirements, and reasoning. If model reasoning is removed,
Python creates fallback reasoning from the deterministic result.

## Requirements

- Python 3.11 or newer
- An OpenAI API key and model name for `analyze`
- No API key for the test suite or the `list` command

## Setup

Create a virtual environment and install the project with development
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Create your private candidate profile from the fictional example:

```powershell
Copy-Item config\candidate_profile.example.yaml config\candidate_profile.yaml
```

Edit every profile value, especially the work-authorization fields, before
using the hard filters. `config/candidate_profile.yaml` is intentionally
gitignored and must stay local.

Create the local environment file:

```powershell
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` and `OPENAI_MODEL` in `.env`. Environment files are
gitignored; `.env.example` contains variable names only.

Optional configuration:

- `config/scoring.yaml` controls component weights, skill aliases, and
  recommendation thresholds.
- `config/filters.yaml` controls the deterministic hard filters.

Configuration is validated at startup, including unknown keys and invalid
weights.

## CLI usage

Analyze a local job description:

```powershell
.\.venv\Scripts\python.exe -m job_search_agent.main analyze --jd jobs/example.txt
```

Fetch and analyze a public job page:

```powershell
.\.venv\Scripts\python.exe -m job_search_agent.main analyze --url "https://example.com/jobs/123"
```

Analyze stdin:

```powershell
Get-Content jobs/example.txt | .\.venv\Scripts\python.exe -m job_search_agent.main analyze --stdin
```

Analyze a file, retain its source URL, and save it to the tracker:

```powershell
.\.venv\Scripts\python.exe -m job_search_agent.main analyze `
    --jd jobs/example.txt `
    --url "https://example.com/jobs/123" `
    --save `
    --status APPLIED `
    --notes "Follow up next week"
```

List tracked jobs:

```powershell
.\.venv\Scripts\python.exe -m job_search_agent.main list
.\.venv\Scripts\python.exe -m job_search_agent.main list --status APPLIED --limit 20
```

Supported statuses are `SAVED`, `APPLIED`, `INTERVIEW`, `REJECTED`, `OFFER`,
and `WITHDRAWN`. Both `analyze` and `list` accept `--database` to override the
default SQLite path.

JSON reports, SQLite databases, and the private candidate profile are ignored
by Git because they can contain personal information.

## Tests

Run the complete suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Tests use fake structured LLM responses and mocked HTTP calls; they do not
require an OpenAI key or external API access. The suite covers configuration,
CLI behavior, ingestion, evidence units, grounding, deterministic component
and overall scoring, sponsorship policy, hard filters, reporting, and SQLite
tracking.

## Project layout

```text
config/                  Private-profile template and scoring/filter settings
src/job_search_agent/    CLI, ingestion, extraction, scoring, and tracking
tests/                   Automated unit and integration tests
evaluation/              Repeatability evaluation tooling
jobs/                    Example postings and authorization-language fixtures
data/processed/          Local generated JSON reports (ignored)
```
