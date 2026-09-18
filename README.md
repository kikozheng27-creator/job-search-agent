# job-search-agent

A personal job-search assistant. It reads a job posting, extracts the
requirements with an LLM, scores the match **deterministically in Python**,
and tracks what you have applied to in a local SQLite database.

## Design principle

The LLM extracts facts and rates four components. It never decides the
outcome.

| Decision | Made by |
|---|---|
| Extracting requirements from the posting | LLM (structured output) |
| Component scores (skills, education, experience, career relevance) | LLM |
| Overall score | Python, from `config/scoring.yaml` weights |
| Hard filters (pass/fail gates) | Python, from `config/filters.yaml` |
| Recommendation | Python, from hard filters + score thresholds |

`JobEvaluation` — the LLM's structured output type — deliberately has no
`overall_score` and no `recommendation` field, so the model cannot supply
them. A test asserts this.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` and
`OPENAI_MODEL`. `.env` is gitignored and must never be committed.

## Configure your profile

All personal data lives in YAML, never in Python.

- `config/candidate_profile.yaml` — your degrees, skills, projects,
  experience, locations, work authorization, and target job families.
  **Sections marked `REVIEW` must be filled in before the hard filters can be
  trusted**, especially `work_authorization`.
- `config/scoring.yaml` — component weights (must sum to 1.0) and the score
  thresholds for each recommendation.
- `config/filters.yaml` — which hard filters are enabled and how strict they
  are.

Invalid configuration fails at startup with a specific message rather than
part-way through an analysis.

## Usage

Analyze a posting from a file:

```powershell
.\.venv\Scripts\python.exe -m job_search_agent.main analyze --jd jobs/example.txt
```

Analyze from a public job URL:

```powershell
.\.venv\Scripts\python.exe -m job_search_agent.main analyze --url "https://example.com/jobs/123"
```

Analyze from stdin:

```powershell
Get-Content jobs/example.txt | .\.venv\Scripts\python.exe -m job_search_agent.main analyze --stdin
```

Analyze a local file and record a source URL with the tracker:

```powershell
.\.venv\Scripts\python.exe -m job_search_agent.main analyze `
    --jd jobs/example.txt `
    --url "https://example.com/jobs/123" `
    --save --status APPLIED --notes "Referral from a classmate"
```

List what you have tracked:

```powershell
.\.venv\Scripts\python.exe -m job_search_agent.main list
.\.venv\Scripts\python.exe -m job_search_agent.main list --status APPLIED --limit 20
```

Every analysis also writes a full JSON report to `data/processed/`.

Application statuses: `SAVED`, `APPLIED`, `INTERVIEW`, `REJECTED`, `OFFER`,
`WITHDRAWN`. The tracker database defaults to `data/tracker.db` and is
gitignored, since it holds your personal application history.

## Hard filters

Hard filters are pass/fail gates kept entirely separate from weighted
scoring. Failing any one of them forces a `SKIP` recommendation regardless of
score, and the reason appears in `hard_filter_reasons`.

Issues that are worth knowing but should not disqualify a job appear in
`concerns` instead. Concerns never affect the score, the filters, or the
recommendation — they exist so an ambiguity can be surfaced rather than
turned into a false negative.

Each rule abstains when the posting is ambiguous, so silence in a job
description never disqualifies it:

- **Experience** — rejects only when the required years exceed
  `max_required_years + tolerance_years`. With the shipped values (2 + 2), a
  posting asking for 3 or 4 years still reaches weighted scoring; 5+ is
  filtered. A posting that states no number is never filtered.
- **Work authorization** — rejects only when the posting rules out your
  *current* status, never merely because it declines future sponsorship. A
  generic "we are unable to sponsor" or "no H-1B" is recorded as
  `not_offered` and preserved, because it does not stop someone already
  authorized (for example on OPT) from taking the job. Only
  `permanent_authorization_required` ("no F-1 candidates", "permanent
  unrestricted work authorization required") and `citizenship_required`
  disqualify. Most postings say nothing, which is recorded as
  `not_mentioned` and passes. A stance the model cannot quote a sentence for
  is downgraded to `not_mentioned`, so a job is never filtered on invented
  evidence.
- **Location** — rejects only locations you explicitly list under
  `locations.unavailable`. Leave that list empty to disable the filter. A
  remote posting nominally based in an excluded city still passes.

## Layout

```
src/job_search_agent/
  main.py           CLI: argument parsing and command dispatch
  candidate.py      CandidateProfile and its sub-models
  models.py         Job requirements, LLM output, final analysis
  config_models.py  Validated schemas for the YAML config files
  config_loader.py  YAML loading
  prompts.py        Extraction and scoring prompt text
  ai_client.py      OpenAI structured-output client
  protocols.py      JobEvaluator protocol (keeps openai out of unit tests)
  job_matcher.py    Orchestration
  page_loader.py    URL fetch and HTML-to-text adapter
  scoring.py        Overall score and recommendation (pure functions)
  filters.py        Hard filters (pure functions)
  concerns.py       Non-blocking warnings (pure functions)
  tracker.py        SQLite job tracker
  reporting.py      Console output and JSON reports
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Unit tests never call the OpenAI API. `tests/conftest.py` provides a
`FakeAIClient` and model builders; `JobMatcher` depends on the `JobEvaluator`
protocol rather than the concrete client, so the test suite does not even
import `openai` for the core paths.
