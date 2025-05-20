import logging
import json
import re
import os
from enum import Enum
from langchain_openai import AzureChatOpenAI
from openai import OpenAI, AzureOpenAI
from google import genai
from google.genai import types
import time
import sys
from together import Together
# Dynamic import that works both when called from main and from the file itself
try:
    # Try relative import first (when imported from another module)
    from src.config.settings import (
        AZURE_OPENAI_SETTINGS,
        OPENROUTER_GEMINI_CONFIG,
        OPENROUTER_DEEPSEEK_V3_CONFIG,
        GOOGLE_GEMINI_CONFIG,
        NEBIUS_AI_DEEPSEEK_V3_CONFIG,
        TOGETHER_AI_DEEPSEEK_V3_CONFIG,
        GPT4_1_CONFIG  # Add import for GPT-4.1 config
    )
except ImportError:
    # If that fails, try absolute import (when run as script)
    # Add the project root to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.config.settings import (
        AZURE_OPENAI_SETTINGS,
        OPENROUTER_GEMINI_CONFIG,
        OPENROUTER_DEEPSEEK_V3_CONFIG,
        GOOGLE_GEMINI_CONFIG,
        NEBIUS_AI_DEEPSEEK_V3_CONFIG,
        TOGETHER_AI_DEEPSEEK_V3_CONFIG,
        GPT4_1_CONFIG  # Add import for GPT-4.1 config
    )


logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> dict:
    """
    Robustly extract JSON from a text string, handling various edge cases.
    
    :param text: Input text potentially containing JSON
    :return: Parsed JSON dictionary
    :raises json.JSONDecodeError: If no valid JSON can be extracted
    """
    # List of strategies to extract JSON
    json_extraction_strategies = [
        # Strategy 1: Find the first complete JSON object
        lambda t: re.search(r'\{.*?\}', t, re.DOTALL),
        
        # Strategy 2: Find JSON between first { and last }
        lambda t: re.search(r'\{.*\}', t, re.DOTALL),
        
        # Strategy 3: Extract content between first { and matching }
        lambda t: re.search(r'\{.*\}', t.split('{', 1)[-1].rsplit('}', 1)[0], re.DOTALL)
    ]
    
    # Try each extraction strategy
    for strategy in json_extraction_strategies:
        try:
            # Find JSON match
            match = strategy(text)
            
            if match:
                # Extract the matched JSON string
                json_str = match.group(0)
                
                # Attempt to parse the JSON
                parsed_json = json.loads(json_str)
                
                return parsed_json
        
        except (json.JSONDecodeError, AttributeError):
            # If this strategy fails, continue to next
            continue
    
    # If all strategies fail, raise an exception
    raise json.JSONDecodeError("Could not extract valid JSON", text, 0)


class LLMType(Enum):
    GPT4O = "gpt-4o"

    GPT4OMINI = "gpt-4o-mini"

    GPT4_1 = "gpt-4.1" # Add GPT-4.1 enum member

    GEMINI25_NATIVE = "gemini-2.5"

    NEBIUS_DEEPSEEK_V3 = "deepseek-v3"
    DEEPSEEK_V3 = "deepseek-v3"
    TOGETHER_DEEPSEEK_V3 = "deepseek-v3"


class AIModelsService:
    def __init__(self):
        self._gpt4o = None
        self._gpt4omini = None
        self._GPT4_1_client = None # Add client attribute for GPT-4.1

        self._gemini_native_client = None

        self._deepseek_v3_client = None
        self._deepseek_v3_client_nebius = None
        self._deepseek_v3_client_together = None


    def _initialize_azure_llm(self, llm_type: LLMType) -> AzureChatOpenAI:
        try:
            if llm_type == LLMType.GPT4O:
                return AzureChatOpenAI(
                    deployment_name=AZURE_OPENAI_SETTINGS["gpt4o_deployment"],
                    model=AZURE_OPENAI_SETTINGS["gpt4o_model"],
                    api_key=AZURE_OPENAI_SETTINGS["api_key"],
                    azure_endpoint=AZURE_OPENAI_SETTINGS["api_endpoint"],
                    api_version=AZURE_OPENAI_SETTINGS["api_version"],
                    temperature=float(AZURE_OPENAI_SETTINGS["temperature"]),
                )
            elif llm_type == LLMType.GPT4OMINI:
                return AzureChatOpenAI(
                    deployment_name=AZURE_OPENAI_SETTINGS["gpt4omini_deployment"],
                    model=AZURE_OPENAI_SETTINGS["gpt4omini_model"],
                    api_key=AZURE_OPENAI_SETTINGS["api_key"],
                    azure_endpoint=AZURE_OPENAI_SETTINGS["api_endpoint"],
                    api_version=AZURE_OPENAI_SETTINGS["api_version"],
                    temperature=float(AZURE_OPENAI_SETTINGS["temperature"]),
                )
            else:
                raise ValueError(f"LLM Azure non valido: {llm_type}")
        except Exception as e:
            logger.error(f"Errore inizializzazione Azure LLM ({llm_type}): {e}")
            raise

    def get_llm(self, llm_type: LLMType):
        if llm_type == LLMType.GPT4O:
            if self._gpt4o is None:
                self._gpt4o = self._initialize_azure_llm(LLMType.GPT4O)
            return self._gpt4o
        elif llm_type == LLMType.GPT4OMINI:
            if self._gpt4omini is None:
                self._gpt4omini = self._initialize_azure_llm(LLMType.GPT4OMINI)
            return self._gpt4omini

        elif llm_type == LLMType.GPT4_1: # Add handling for GPT-4.1
            if self._GPT4_1_client is None:
                if not GPT4_1_CONFIG or not all(k in GPT4_1_CONFIG for k in ["api_key", "api_endpoint", "api_version"]):
                    raise ValueError("GPT-4.1 configuration is missing required keys (api_key, api_endpoint, api_version).")
                self._GPT4_1_client = AzureOpenAI(
                    azure_endpoint=GPT4_1_CONFIG["api_endpoint"],
                    api_key=GPT4_1_CONFIG["api_key"],
                    api_version=GPT4_1_CONFIG["api_version"]
                )
            return self._GPT4_1_client

        elif llm_type == LLMType.DEEPSEEK_V3:
            if self._deepseek_v3_client is None:
                if not OPENROUTER_DEEPSEEK_V3_CONFIG:
                    raise ValueError("DeepSeek configuration is missing.")
                self._deepseek_v3_client = OpenAI(
                    base_url=OPENROUTER_DEEPSEEK_V3_CONFIG["base_url"],
                    api_key=OPENROUTER_DEEPSEEK_V3_CONFIG["api_key"]
                )
            return self._deepseek_v3_client
        
        elif llm_type == LLMType.NEBIUS_DEEPSEEK_V3:
            if self._deepseek_v3_client_nebius is None:
                if not NEBIUS_AI_DEEPSEEK_V3_CONFIG:
                    raise ValueError("DeepSeek configuration is missing.")
                self._deepseek_v3_client_nebius = OpenAI(
                    base_url=NEBIUS_AI_DEEPSEEK_V3_CONFIG["base_url"],
                    api_key=NEBIUS_AI_DEEPSEEK_V3_CONFIG["api_key"]
                )
            return self._deepseek_v3_client_nebius
        
        elif llm_type == LLMType.TOGETHER_DEEPSEEK_V3:
            if self._deepseek_v3_client_together is None:
                if not TOGETHER_AI_DEEPSEEK_V3_CONFIG:
                    raise ValueError("DeepSeek configuration is missing.")
                self._deepseek_v3_client_together = Together()
            return self._deepseek_v3_client_together
        
        elif llm_type == LLMType.GEMINI25_NATIVE:
            if self._gemini_native_client is None:
                if not GOOGLE_GEMINI_CONFIG:
                    raise ValueError("Google Gemini configuration is missing.")
                self._gemini_native_client = genai.Client(
                    api_key=GOOGLE_GEMINI_CONFIG["api_key"]
                )
            return self._gemini_native_client

        else:
            raise ValueError(f"Tipo di LLM non supportato: {llm_type}")

    def call_llm(self, prompt: str, llm_type: LLMType, max_attempts: int = 3, delay_seconds: int = 10, with_json_extraction: bool = True) -> str:
        attempt = 1
        last_error = None
        
        while attempt <= max_attempts:
            try:
                logger.debug(f"Attempt {attempt}/{max_attempts} calling {llm_type.value} with prompt:\n{prompt}")

                if llm_type in [LLMType.GPT4O, LLMType.GPT4OMINI]:
                    llm = self.get_llm(llm_type)
                    response = llm.invoke(prompt)
                    content = response.content.strip()       

                elif llm_type == LLMType.GPT4_1: # Add call handling for GPT-4.1
                    client = self.get_llm(llm_type)
                    # Ensure deployment name and temperature are in config
                    if "deployment_name" not in GPT4_1_CONFIG:
                        raise ValueError("GPT-4.1 configuration is missing 'deployment_name'.")
                    deployment_name = GPT4_1_CONFIG["deployment_name"]
                    temperature = float(GPT4_1_CONFIG.get("temperature", 1.0)) # Default temperature if not specified
                    
                    response = client.chat.completions.create(
                        model=deployment_name, # Use deployment name for Azure OpenAI
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature
                    )
                    content = response.choices[0].message.content.strip()

                elif llm_type in [LLMType.DEEPSEEK_V3]:
                     client = self.get_llm(llm_type)
                     model = OPENROUTER_DEEPSEEK_V3_CONFIG["model_name"]
                     response = client.chat.completions.create(
                         model=model,
                         messages=[{"role": "user", "content": prompt}]
                     )
                     content = response.choices[0].message.content.strip()
                
                elif llm_type in [LLMType.NEBIUS_DEEPSEEK_V3]:
                    client = self.get_llm(llm_type)
                    model = NEBIUS_AI_DEEPSEEK_V3_CONFIG["model_name"]
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}]
                    )

                elif llm_type in [LLMType.TOGETHER_DEEPSEEK_V3]:
                    client = self.get_llm(llm_type)
                    model = TOGETHER_AI_DEEPSEEK_V3_CONFIG["model_name"]
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                
                elif llm_type in [LLMType.GEMINI25]:
                    client = self.get_llm(llm_type)
                    model = OPENROUTER_GEMINI_CONFIG["model_name"]
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.choices[0].message.content.strip()
                
                elif llm_type in [LLMType.GEMINI25_NATIVE]:
                    client = self.get_llm(llm_type)
                    model = GOOGLE_GEMINI_CONFIG["model_name"]
                    contents = [
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(text=prompt),
                            ],
                        ),
                    ]
                    generate_content_config = types.GenerateContentConfig(
                        response_mime_type="text/plain",
                    )
                    
                    # Capture the full response from streaming
                    full_response = ""
                    for chunk in client.models.generate_content_stream(
                        model=model,
                        contents=contents,
                        config=generate_content_config,
                    ):
                        chunk_text = chunk.text
                        full_response += chunk_text
                        print(chunk_text, end="")
                    
                    content = full_response.strip()

                else:
                    raise ValueError(f"LLM type {llm_type} not handled.")

                logger.debug(f"Raw LLM response:\n{content}")

                print("Content answer: ", content)

                if not with_json_extraction:
                    return content
                    
                # Pulizia blocchi markdown se presenti
                if "```" in content:
                    content = re.sub(r"```(?:json|markdown|plaintext)?", "", content).strip()
                    logger.debug(f"Cleaned LLM content:\n{content}")

                json_response = extract_json_from_text(content)
                return json_response

            except Exception as e:
                last_error = e
                logger.error(f"Attempt {attempt} failed for {llm_type.value}: {e}")
                if attempt < max_attempts:
                    logger.debug(f"Waiting {delay_seconds} seconds before retry...")
                    time.sleep(delay_seconds)
                attempt += 1
        
        logger.error(f"All {max_attempts} attempts failed for {llm_type.value}")
        raise last_error if last_error else Exception(f"Failed to call {llm_type.value} after {max_attempts} attempts")


def test_azure_4o():
    print("Test Azure 4o")
    #print what's inside AZURE_OPENAI_SETTINGS,
    for key, value in AZURE_OPENAI_SETTINGS.items():
        print(f"{key}: {value}")
    llm_service = AIModelsService()
    llm = llm_service.get_llm(LLMType.GEMINI25)
    response = llm.invoke("Ciao")
    print(response)

if __name__ == "__main__":
    test_azure_4o()