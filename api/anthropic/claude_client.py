import os
from openai import OpenAI


api_key_env = None
for key, value in os.environ.items():
    if "ANTHROPIC_API_KEY" in key:
        api_key_env = value
        break

if api_key_env:
    print("Anthropic API Key successfully charged from environment.")
else:
    print("Anthropic API Key not found. Please check your .env file or Docker flags.")

# Anthropic exposes an OpenAI-compatible endpoint via their Messages API
# Use the anthropic SDK directly for production; this OpenAI-compatible
# client works for tool-calling workflows that already use the OpenAI format.
client_anthropic = OpenAI(
    base_url="https://api.anthropic.com/v1",
    api_key=api_key_env,
)
