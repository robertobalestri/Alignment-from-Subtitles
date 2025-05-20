from dotenv import load_dotenv
from typing import Any, Dict
from pathlib import Path
import os

load_dotenv(override=True)

SUBTITLES_DIRECTORY = Path("subtitles")

LLM_OUTPUT_DIRECTORY = Path("output")

SRT_TO_DIALOGUE_JSON_DIRECTORY = SUBTITLES_DIRECTORY / "json_converted"

# Azure OpenAI Settings
AZURE_OPENAI_SETTINGS: Dict[str, Any] = {
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "api_endpoint": os.getenv("AZURE_OPENAI_API_ENDPOINT"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    "gpt4o_deployment": os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME_GPT4O"),
    "gpt4o_model": os.getenv("AZURE_OPENAI_LLM_MODEL_NAME_GPT4O"),
    "gpt4omini_deployment": os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME_GPT4OMINI"),
    "gpt4omini_model": os.getenv("AZURE_OPENAI_LLM_MODEL_NAME_GPT4OMINI"),
    "temperature": os.getenv("AZURE_OPENAI_TEMPERATURE")
}

OPENROUTER_GEMINI_CONFIG = {
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": os.getenv("OPEN_ROUTER_API_KEY_GEMINI"),
    "model_name": "google/gemini-2.5-pro-exp-03-25:free"
}

GOOGLE_GEMINI_CONFIG = {
    "api_key": os.getenv("GOOGLE_API_KEY"),
    "model_name": "gemini-2.5-pro-preview-05-06"
}

OPENROUTER_DEEPSEEK_V3_CONFIG = {
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": os.getenv("OPEN_ROUTER_API_KEY_DEEPSEEK_V3"),
    "model_name": "deepseek/deepseek-chat-v3-0324:free"
}

NEBIUS_AI_DEEPSEEK_V3_CONFIG = {
    "api_key": os.getenv("NEBIUS_AI_API_KEY_DEEPSEEK_V3"),
    "model_name": "deepseek-ai/DeepSeek-V3-0324",
    "base_url": "https://api.studio.nebius.com/v1/"
}
# GPT-4.1 Settings
GPT4_1_CONFIG = {
    "api_key": os.getenv("GPT4_1_API_KEY"),
    "api_endpoint": os.getenv("GPT4_1_ENDPOINT"), # Changed base_url to api_endpoint
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    "deployment_name": os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME_GPT4_1"), # Added deployment name
    "gpt4_1_model": os.getenv("AZURE_OPENAI_LLM_MODEL_NAME_GPT4_1"),
    "temperature": os.getenv("AZURE_OPENAI_TEMPERATURE")
    # Add model name if required by the endpoint, e.g.:
    # "model_name": "gpt-4.1"
}

TOGETHER_AI_DEEPSEEK_V3_CONFIG = {
    "api_key": os.getenv("TOGETHER_AI_API_KEY"),
    "model_name": "deepseek-ai/DeepSeek-V3",
}