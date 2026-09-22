# Job Search Agent

[![Tests](https://github.com/kikozheng27-creator/job-search-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/kikozheng27-creator/job-search-agent/actions/workflows/tests.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

An evidence-grounded AI job-matching engine that combines LLM-based
requirement extraction with deterministic scoring and hard-requirement
filtering.

The local Python CLI ingests job postings, extracts structured
requirements with an LLM, and grounds important claims against source
evidence. It then evaluates skills, education, experience, career
relevance, sponsorship/work authorization, and hard requirements, and
produces a reproducible match score and recommendation.

## Key Features

- Job-posting ingestion from local files, stdin, and public URLs
- Structured LLM requirement extraction
- Evidence-unit grounding of skill claims
- Deterministic skill, experience, and education scoring
- Deterministic career-relevance scoring from structured alignment evidence
- Python sponsorship and work-authorization classification
- Hard-requirement filtering, independent of the weighted score
- Safeguards against hallucinated or unsupported claims
- Deterministic overall score and recommendation
- JSON reports and optional SQLite application tracking
- Automated tests on GitHub Actions

## Example Output

`analyze` prints a report in this format. The example below is fictional:

```text
============================================================
Company: Northwind Analytics
Role: Data Analyst
Location: Remote
============================================================

Overall Match: 79.4 / 100
Recommendation: APPLY

Scores
  Skills:            74
  Education:         100
  Experience:        70
  Career Relevance:  80

Concerns (not blocking)
  - Posting states it does not offer visa sponsorship. This does not block you now, but it will matter once your current work authorization ends (status: Example work visa).

Strengths
  - Strong overlap with Python and SQL requirements
  - Degree level meets the stated education requirement

Missing Requirements
  - Tableau

Reasoning
The candidate matches most core requirements, with remaining gaps in preferred visualization tools. The recommendation is APPLY.
```

Failed hard filters print `Hard Filters: FAILED` with reasons and force
`SKIP`. Every analysis also writes a JSON report under `data/processed/`.

## Architecture

1. **Ingestion.** `analyze` accepts a local text file, stdin, or a public
   HTTP(S) job URL. URL ingestion uses `requests` and `BeautifulSoup` to
   remove page chrome and application forms. Pages that are empty, not
   HTML, or require client-side JavaScript are rejected.
2. **Evidence units.** Python splits the cleaned posting into stable,
   numbered source units. These units are the boundary for skill
   extraction and grounding.
3. **Structured LLM extraction.** The OpenAI client requests
   Pydantic-backed structured outputs for job requirements and a dedicated
   per-unit skill inventory. The model also supplies structured
   career-alignment evidence and explanatory prose. Python scores career
   relevance from that evidence.
4. **Grounding and safeguards.** Python rejects unknown evidence-unit IDs,
   skills not present in their cited unit, non-skill categories, and
   contextual tool mentions that are not candidate requirements. Missing
   skill-unit coverage gets one repair pass and remains visible as an
   unresolved concern if repair fails.
5. **Deterministic component scoring.** Python computes skill overlap from
   the grounded inventory, experience from candidate years versus the
   stated minimum, education from degree-level requirements, and career
   relevance from quoted target-family and preferred-industry alignment.
   Education affects the weighted score but is not a hard filter.
6. **Sponsorship classification and hard filters.** Python classifies
   sponsorship/work-authorization language from source sentences and
   overwrites unsupported model classifications. Configurable experience,
   current-work-authorization, and location gates run independently of
   the weighted score and abstain when the posting is ambiguous.
7. **Deterministic decision.** Python calculates the weighted overall
   score from `config/scoring.yaml`. It then applies hard-filter results
   and score thresholds to choose `STRONGLY_APPLY`, `APPLY`, `MAYBE`, or
   `SKIP`. The LLM output schema has no overall-score or recommendation
   field.
8. **Outputs.** Every analysis writes a JSON report under
   `data/processed/`. With `--save`, the CLI also stores summary data and
   application status in SQLite at `data/tracker.db` by default.

Additional hallucination safeguards verify sponsorship quotes against the
source posting and remove unsupported authorization claims from
model-written strengths, missing requirements, and reasoning. If model
reasoning is removed, Python creates fallback reasoning from the
deterministic result.

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

## CLI Usage

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

## Project Layout

```text
config/                  Private-profile template and scoring/filter settings
src/job_search_agent/    CLI, ingestion, extraction, scoring, and tracking
tests/                   Automated unit and integration tests
evaluation/              Repeatability evaluation tooling
jobs/                    Example postings and authorization-language fixtures
data/processed/          Local generated JSON reports (ignored)
```
