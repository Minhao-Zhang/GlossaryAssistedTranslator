import os
from typing import List

from openai import OpenAI

from .base_translator import BaseTranslator


class OpenAITranslator(BaseTranslator):
    """Translator implementation using the OpenAI API."""

    def chat(self, messages: List[dict], base_url: str = "https://api.openai.com/v1", api_key_env_var: str = "OPENAI_API_KEY", model: str = "gpt-4o-mini") -> str:
        """
        Send messages to the OpenAI API for translation.

        Args:
            messages: List of message dictionaries
            base_url: Base URL for the OpenAI API
            api_key_env_var: Environment variable name containing API key
            model: OpenAI model to use for translation

        Returns:
            Translated text from OpenAI
        """
        client = OpenAI(
            api_key=os.environ.get(api_key_env_var),
            base_url=base_url
        )

        response = client.chat.completions.create(
            messages=messages,
            model=model,
        )

        return response.choices[0].message.content.strip()
