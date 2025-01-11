from typing import List

from openai import Client

from .base_translator import BaseTranslator


class OllamaTranslator(BaseTranslator):
    """Translator implementation using the Ollama API."""

    def chat(self, messages: List[dict], base_url: str = "localhost:11434", api_key_env_var: str = "OLLAMA_API_KEY", model: str = "qwen2.5:7b") -> str:
        """
        Send messages to the Ollama API for translation. 

        If you are running this inside docker, the base_url shall be set to http://host.docker.internal:11434

        Args:
            messages: List of message dictionaries
            base_url: Base URL for the Ollama API
            api_key_env_var: Environment variable name containing API key
            model: Ollama model to use for translation

        Returns:
            Translated text from Ollama
        """
        client = Client(host=base_url)

        response = client.chat(
            model=model,
            messages=messages
        )

        return response['message']['content'].strip()
