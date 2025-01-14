from typing import List

from openai import Client

from .base_translator import BaseTranslator


class OllamaTranslator(BaseTranslator):
    """Translator implementation using the Ollama API."""

    def __init__(self, 
                 matcher = None, 
                 system_prompt_file = "prompt_template/system_prompt_v2.txt", 
                 user_prompt_file = "prompt_template/user_prompt_v2.txt", 
                 base_url = "localhost:11434", 
                 api_key_env_var = "OLLAMA_API_KEY", 
                 model = "qwen2.5:7b"):
        """
        Initialize the translator with optional glossary components.
        If you are running this inside docker, use http://host.docker.internal:11434 for base_url.

        Args:
            matcher: GlossaryMatcher instance for exact match glossary lookup
            system_prompt_file: Path to file containing the system prompt template
            base_url: Base URL for the API. 
            api_key_env_var: Environment variable name containing API key
            model: Model to use for translation
        
        """
        
        super().__init__(matcher, system_prompt_file, user_prompt_file, base_url, api_key_env_var, model)

        self.llm_client = Client(host=base_url)

    def chat(self, messages: List[dict]) -> str:
        """
        Send messages to the Ollama API for translation. 

        Args:
            messages: List of message dictionaries

        Returns:
            Translated text from Ollama
        """

        response = self.llm_client.chat(
            model=self.model,
            messages=messages
        )

        return response['message']['content'].strip()
