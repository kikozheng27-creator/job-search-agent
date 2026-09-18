import os

import truststore
from dotenv import load_dotenv
from openai import OpenAI

from job_search_agent.models import JobEvaluation


load_dotenv()

# The openai SDK verifies TLS against certifi's bundle, which omits roots that
# only the operating system trusts -- including those added by local
# TLS-inspecting security software. Verification stays fully enabled; this only
# changes which trust store it is checked against.
truststore.inject_into_ssl()


class AIClient:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing")

        if not model:
            raise ValueError("OPENAI_MODEL is missing")

        self.model = model
        self.client = OpenAI(api_key=api_key)

    def evaluate_job(self, prompt: str) -> JobEvaluation:
        response = self.client.responses.parse(
            model=self.model,
            input=prompt,
            text_format=JobEvaluation,
        )

        evaluation = response.output_parsed

        if evaluation is None:
            raise ValueError(
                "The model did not return a parsable job evaluation."
            )

        return evaluation
