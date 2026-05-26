import os
import logging
from typing import Any, Dict, List
from dotenv import load_dotenv

# Import LangChain LLM classes
from langchain_groq import ChatGroq

logger = logging.getLogger("rag_app.llm")

# Provider model defaults and listings
SUPPORTED_PROVIDERS = {
    "groq": {
        "name": "Groq",
        "env_key": "GROQ_API_KEY",
        "models": {
            "llama-3.1-8b-instant": "Llama 3.1 8B Instant",
            "llama-3.3-70b-versatile": "Llama 3.3 70B",
            "mixtral-8x7b-32768": "Mixtral 8x7B"
        },
        "default_model": "llama-3.1-8b-instant"
    }
}

class LLMService:
    def __init__(self):
        load_dotenv()

    def get_llm(self, provider: str, model: str = None, temperature: float = 0.3) -> Any:
        """
        Dynamically construct and return a LangChain ChatModel instance.
        """
        provider = provider.strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported LLM provider: {provider}. Supported: {list(SUPPORTED_PROVIDERS.keys())}")

        provider_info = SUPPORTED_PROVIDERS[provider]
        env_var_name = provider_info["env_key"]
        
        # Read API key
        api_key = os.getenv(env_var_name)

        if not api_key:
            raise ValueError(f"API key for provider '{provider}' ({env_var_name}) is missing from environment.")

        # Determine model
        if not model or model not in provider_info["models"]:
            model = provider_info["default_model"]
            logger.info(f"Model not specified or invalid for {provider}. Falling back to default: {model}")

        logger.info(f"Initializing LLM: provider={provider}, model={model}, temp={temperature}")

        try:
            if provider == "groq":
                return ChatGroq(
                    model=model,
                    temperature=temperature,
                    groq_api_key=api_key,
                    streaming=True
                )
        except Exception as e:
            logger.error(f"Error creating LLM client for {provider}: {str(e)}")
            raise e

    def get_supported_info(self) -> Dict[str, Any]:
        """
        Returns supported providers, available models, and API key availability.
        """
        status_dict = {}
        for key, info in SUPPORTED_PROVIDERS.items():
            env_key = info["env_key"]
            has_key = bool(os.getenv(env_key))
                
            status_dict[key] = {
                "name": info["name"],
                "models": info["models"],
                "default_model": info["default_model"],
                "available": has_key
            }
        return status_dict

