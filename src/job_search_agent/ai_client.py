import os

from dotenv import load_dotenv
from openai import OpenAI
from job_search_agent.models import JobAnalysis
from job_search_agent.models import JobEvaluation


load_dotenv()


class AIClient:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing")

        if not model:
            raise ValueError("OPENAI_MODEL is missing")

        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        return response.output_text

    def analyze_job(self, prompt: str) -> JobAnalysis:
        response = self.client.responses.parse(
            model=self.model,
            input=prompt,
            text_format=JobAnalysis,
        )

        return response.output_parsed

    def evaluate_job(self, prompt: str) -> JobEvaluation:
        response = self.client.responses.parse(
            model=self.model,
            input=prompt,
            text_format=JobEvaluation,
        )

        return response.output_parsed