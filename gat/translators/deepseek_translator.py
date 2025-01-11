import os
from typing import List

from openai import OpenAI

from .base_translator import BaseTranslator


class DeepSeekTranslator(BaseTranslator):
    """Translator implementation using the DeepSeek API."""

    def chat(self, messages: List[dict], base_url: str = "https://api.deepseek.com/v1", api_key_env_var: str = "DEEPSEEK_API_KEY", model: str = "deepseek-chat") -> str:
        """
        Send messages to the DeepSeek API for translation.

        Args:
            messages: List of message dictionaries
            base_url: Base URL for the DeepSeek API
            api_key_env_var: Environment variable name containing API key
            model: DeepSeek model to use for translation

        Returns:
            Translated text from DeepSeek
        """
        client = OpenAI(
            api_key=os.environ.get(api_key_env_var),
            base_url=base_url
        )

        response = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=1.3  # This is set according to official doc https://api-docs.deepseek.com/quick_start/parameter_settings
        )

        return response.choices[0].message.content.strip()
