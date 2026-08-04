import os
from openai import OpenAI

api_key_env = None
for key, value in os.environ.items():
    if "API_KEY_groq" in key:
        api_key_env = value
        break

if api_key_env:
    print("API Key successfully charged from environment.")
else:
    print("API Key not found. Please check your .env file or Docker flags.")

client_groq = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key_env,
)