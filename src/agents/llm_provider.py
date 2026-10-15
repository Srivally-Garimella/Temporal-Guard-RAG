"""
LLM Provider Factory for TemporalGuard-RAG

Supports multiple LLM backends:
- OpenAI (GPT-4, GPT-3.5)
- Ollama (local models: llama3, mistral, etc.)
"""

import os
import logging
from typing import Optional, Literal
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_PROVIDER = "ollama"  # Use local by default
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OPENAI_MODEL = "gpt-4"


def get_llm(
    provider: Literal["openai", "ollama"] = None,
    model_name: str = None,
    temperature: float = 0,
    base_url: str = "http://localhost:11434"
):
    """
    Factory function to get LLM instance.
    
    Args:
        provider: "openai" or "ollama" (defaults to ollama if no API key)
        model_name: Model name (provider-specific)
        temperature: Sampling temperature
        base_url: Ollama server URL (default: localhost:11434)
        
    Returns:
        LangChain chat model instance
    """
    # Auto-detect provider if not specified
    if provider is None:
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
            logger.info("OpenAI API key found, using OpenAI")
        else:
            provider = "ollama"
            logger.info("No OpenAI API key, using Ollama (local)")
    
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI provider requires the 'langchain-openai' package; install project dependencies from requirements.txt."
            ) from exc
        model = model_name or DEFAULT_OPENAI_MODEL
        logger.info(f"Initializing OpenAI LLM: {model}")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "Ollama provider requires the 'langchain-ollama' package; install project dependencies from requirements.txt."
            ) from exc
        model = model_name or DEFAULT_OLLAMA_MODEL
        logger.info(f"Initializing Ollama LLM: {model} at {base_url}")
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'ollama'")


def check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running."""
    import urllib.request
    import urllib.error
    
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, Exception):
        return False


def list_ollama_models(base_url: str = "http://localhost:11434") -> list:
    """List available Ollama models."""
    import urllib.request
    import json
    
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as response:
            data = json.loads(response.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        logger.warning(f"Could not list Ollama models: {e}")
        return []


# Convenience functions
def get_openai_llm(model_name: str = None, temperature: float = 0):
    """Get OpenAI LLM."""
    return get_llm(provider="openai", model_name=model_name, temperature=temperature)


def get_ollama_llm(model_name: str = None, temperature: float = 0, base_url: str = "http://localhost:11434"):
    """Get Ollama LLM."""
    return get_llm(provider="ollama", model_name=model_name, temperature=temperature, base_url=base_url)
