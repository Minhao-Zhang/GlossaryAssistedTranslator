import os
from typing import List

from openai import OpenAI

from .base_translator import BaseTranslator
from gat.glossary_matcher import GlossaryMatcher


class DeepSeekTranslator(BaseTranslator):
    """Translator implementation using the DeepSeek API."""

    def __init__(self,
                 matcher: GlossaryMatcher = None,
                 system_prompt_file: str = "prompt_template/system_prompt_v2.txt",
                 user_prompt_file: str = "prompt_template/user_prompt_v2.txt",
                 base_url: str = "https://api.deepseek.com/v1",
                 api_key_env_var: str = "DEEPSEEK_API_KEY",
                 model: str = "deepseek-chat"
                 ):
        super().__init__(matcher, system_prompt_file,
                         user_prompt_file, base_url, api_key_env_var, model)

        self.llm_client = OpenAI(
            api_key=os.environ.get(api_key_env_var),
            base_url=base_url
        )

    def chat(self, messages: List[dict]) -> str:
        """
        Send messages to the DeepSeek API for translation.

        Args:
            messages: List of message dictionaries

        Returns:
            Translated text from DeepSeek
        """

        response = self.llm_client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=1.3  # This is set according to official doc https://api-docs.deepseek.com/quick_start/parameter_settings
        )

        return response.choices[0].message.content.strip()
