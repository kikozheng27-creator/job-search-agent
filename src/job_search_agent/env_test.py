import os

from dotenv import load_dotenv


load_dotenv()

secret = os.getenv("OPENAI_API_KEY")

if not secret:
    raise ValueError("OPENAI_API_KEY is missing")

print("OPENAI_API_KEY loaded successfully.")