from typing import Protocol

from job_search_agent.models import JobEvaluation


class JobEvaluator(Protocol):
    """The only capability JobMatcher needs from an AI client.

    Depending on this instead of the concrete AIClient keeps the openai
    package out of the import graph for unit tests.
    """

    def evaluate_job(self, prompt: str) -> JobEvaluation: ...
