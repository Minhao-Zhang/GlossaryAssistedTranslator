import os
from typing import List

from openai import OpenAI

from gat.glossary_matcher import GlossaryMatcher

from .base_translator import BaseTranslator


class OpenRouterTranslator(BaseTranslator):
    """Translator implementation using the OpenAI API."""

    def __init__(self, 
                 matcher: GlossaryMatcher = None, 
                 system_prompt_file: str = "prompt_template/system_prompt_v2.txt", 
                 user_prompt_file: str = "prompt_template/user_prompt_v2.txt", 
                 base_url: str = "https://openrouter.ai/api/v1", 
                 api_key_env_var: str = "OPENROUTER_API_KEY", 
                 model: str = "openai/gpt-4o-mini"
                 ):
        super().__init__(matcher, system_prompt_file, user_prompt_file, base_url, api_key_env_var, model)

        self.llm_client = OpenAI(
            api_key=os.environ.get(api_key_env_var),
            base_url=base_url
        )


    def chat(self, messages: List[dict]) -> str:
        """
        Send messages to the OpenAI API for translation.

        Args:
            messages: List of message dictionaries

        Returns:
            Translated text from DeepSeek
        """

        response = self.llm_client.chat.completions.create(
            messages=messages,
            model=self.model,
        )

        return response.choices[0].message.content.strip()
