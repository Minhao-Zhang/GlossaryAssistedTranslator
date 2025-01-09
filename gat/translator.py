"""Module containing the main translator classes for different AI providers.

This module provides a base translator class and implementations for various
AI translation services including Ollama, OpenAI, and DeepSeek.
"""

from collections import deque
import os
from typing import List, Optional
from tqdm import tqdm

from ollama import Client
from openai import OpenAI
import pandas as pd
from gat.glossary_matcher import GlossaryMatcher
from gat.glossary_rag import GlossaryRAG


class BaseTranslator:
    """
    Base class for AI-powered translators. This should not be used directly.

    Provides common functionality for managing translation context, glossary
    integration, and message formatting.
    """

    def __init__(self,
                 rag: Optional[GlossaryRAG] = None,
                 matcher: Optional[GlossaryMatcher] = None,
                 n_history: int = 5,
                 system_prompt_file: str = "prompt_template/glossary_v3.txt"):
        """
        Initialize the translator with optional glossary components.

        Args:
            rag: GlossaryRAG instance for semantic glossary lookup
            matcher: GlossaryMatcher instance for exact match glossary lookup
            n_history: Number of previous translation pairs to maintain as context
            system_prompt_file: Path to file containing the system prompt template
        """
        self.rag = rag
        self.matcher = matcher
        self.n_history = n_history
        # keep the previous n_history pairs
        self.history = deque(maxlen=n_history*2)
        with open(system_prompt_file, "r") as file:
            self.system_prompt = file.read()

    def update_history(self, user: str, assistant: str):
        """
        Update the translation history with the latest user-assistant pair.

        Args:
            user: The original text from the user
            assistant: The translated text from the assistant
        """
        self.history.append({
            "role": "user",
            "content": user,
        })
        self.history.append({
            "role": "assistant",
            "content": assistant,
        })

    def get_glossary(self, sentence: str) -> pd.DataFrame:
        """
        Retrieve relevant glossary entries for a given sentence.

        Combines results from both RAG and exact match glossary lookups.

        Args:
            sentence: The sentence to find glossary entries for

        Returns:
            DataFrame containing matching glossary entries
        """
        all_results = pd.DataFrame()
        if self.rag:
            rag_results = self.rag.query(sentence)
            all_results = pd.concat([all_results, rag_results])
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

    def build_messages(self, system_prompt: str, definitions: str, examples: str, sentence: str) -> List[dict]:
        """
        Build the messages list for the API call.

        Args:
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
                "content": system_prompt.format(definitions=definitions, examples=examples)
            }
        ]
        messages.extend(list(self.history))
        messages.append({
            "role": "user",
            "content": sentence
        })

        return messages

    def chat(self, messages: List[dict], base_url: str, api_key_env_var: str, model: str) -> str:
        """
        Abstract method for sending messages to the translation API.

        Args:
            messages: List of message dictionaries
            base_url: Base URL for the API
            api_key_env_var: Environment variable name containing API key
            model: Model to use for translation

        Returns:
            Translated text from the API
        """
        raise NotImplementedError(
            "Chat method must be implemented in subclass")

    def translate_sentences(self, sentences: List[str]) -> List[str]:
        """
        Translate a list of sentences using the configured translator.

        Args:
            sentences: List of sentences to translate

        Returns:
            List of translated sentences
        """
        translated = []
        for sentence in tqdm(sentences, desc="Translating sentences"):
            glossary = self.get_glossary(sentence)
            definitions, examples = self.format_glossary(glossary)
            messages = self.build_messages(
                self.system_prompt, definitions, examples, sentence)

            translated_text = self.chat(messages)
            translated.append(translated_text)

            self.update_history(sentence, translated_text)
        return translated


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
            temperature=1.3
        )

        return response.choices[0].message.content.strip()


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

# If you wish to use other providers or packages, you can just override the chat function. 