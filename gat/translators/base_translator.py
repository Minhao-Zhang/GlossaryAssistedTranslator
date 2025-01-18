"""Module containing the main translator classes for different AI providers.

This module provides a base translator class and implementations for various
AI translation services including Ollama, OpenAI, and DeepSeek.
"""

from collections import deque
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from gat.glossary_matcher import GlossaryMatcher


class BaseTranslator:
    """
    Base class for AI-powered translators. This should not be used directly.

    Provides common functionality for managing translation context, glossary
    integration, and message formatting.
    """

    def __init__(self,
                 matcher: Optional[GlossaryMatcher] = None,
                 system_prompt_file: str = "prompt_template/system_prompt_v2.txt",
                 user_prompt_file: str = "prompt_template/user_prompt_v2.txt",
                 base_url: str = None,
                 api_key_env_var: str = None,
                 model: str = None
                 ):
        """
        Initialize the translator with optional glossary components.

        Args:
            matcher: GlossaryMatcher instance for exact match glossary lookup
            system_prompt_file: Path to file containing the system prompt template
            base_url: Base URL for the API
            api_key_env_var: Environment variable name containing API key
            model: Model to use for translation
        """
        self.matcher = matcher
        self.base_url = base_url
        self.api_key_env_var = api_key_env_var
        self.model = model
        with open(system_prompt_file, "r") as file:
            self.system_prompt = file.read()
        with open(user_prompt_file, "r") as file:
            self.user_prompt = file.read()

    def translate_text(self, text: str, use_glossary=False) -> str:
        """
        Translate a single text string using the configured translator.

        Args:
            text: The text string to translate
            use_glossary: Whether to use glossary integration for the translation.
                         If True, will look up relevant glossary terms and include
                         their definitions and examples in the translation context.

        Returns:
            The translated text as a string
        """

        if use_glossary:
            definitions, examples = self.format_glossary(
                self.get_glossary(text))
        else:
            definitions, examples = "", ""

        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": self.user_prompt.format(definitions=definitions, examples=examples, sentence=text)
            }
        ]

        return self.chat(messages)

    def get_glossary(self, sentence: str) -> pd.DataFrame:
        """
        Retrieve relevant glossary entries for a given sentence.

        Combines results from both RAG and exact match glossary lookups.
        Returns an empty DataFrame with required columns if no glossary components are provided.

        Args:
            sentence: The sentence to find glossary entries for

        Returns:
            DataFrame containing matching glossary entries with columns:
            Term, Translation, Definition, Example
        """
        all_results = pd.DataFrame(
            columns=["Term", "Translation", "Definition", "Example"])
        if self.matcher:
            matcher_results = self.matcher.search_sentence(sentence)
            all_results = pd.concat([all_results, matcher_results])

        return all_results.drop_duplicates()

    def format_glossary(self, glossary: pd.DataFrame) -> tuple[List[str], List[str]]:
        """
        Format glossary entries into definitions and examples.

        Args:
            glossary: DataFrame containing glossary entries

        Returns:
            Tuple of (definitions, examples) as formatted strings
        """
        definitions = "\n".join(glossary["Definition"].tolist())
        examples = "\n".join(glossary["Example"].tolist())

        return definitions, examples

    def build_messages(self, history: deque, definitions: str, examples: str, sentence: str) -> List[dict]:
        """
        Build the messages list for the API call.

        Args:
            history: The history to include
            system_prompt: The formatted system prompt
            definitions: Formatted glossary definitions
            examples: Formatted glossary examples
            sentence: The sentence to translate

        Returns:
            List of message dictionaries for the API call
        """
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
        messages.extend(list(history))
        messages.append({
            "role": "user",
            "content": self.user_prompt.format(definitions=definitions, examples=examples, sentence=sentence)
        })

        return messages

    def chat(self, messages: List[dict],) -> str:
        """
        Abstract method for sending messages to the translation API. 
        You should override this in sub-classes.

        Args:
            messages: List of message dictionaries

        Returns:
            Translated text from the API
        """
        raise NotImplementedError(
            "Chat method must be implemented in subclass")

    def translate_sentences(self, sentences: List[str], n_history=3) -> List[str]:
        """
        Translate a list of sentences using the configured translator.

        Args:
            sentences: List of sentences to translate

        Returns:
            List of translated sentences
        """
        translated = []
        history = deque(maxlen=n_history*2)
        for sentence in tqdm(sentences, desc="Translating sentences"):
            glossary = self.get_glossary(sentence)
            definitions, examples = self.format_glossary(glossary)
            messages = self.build_messages(
                history, definitions, examples, sentence)

            translated_text = self.chat(messages)
            translated.append(translated_text)

            history.append({
                "role": "user",
                "content": sentence,
            })
            history.append({
                "role": "assistant",
                "content": translated_text,
            })

            # print(f"Original  : {sentence}")
            # print(f"Translated: {translated_text}")

        return translated
