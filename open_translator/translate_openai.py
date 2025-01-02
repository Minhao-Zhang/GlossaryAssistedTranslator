import os
from openai import OpenAI
from collections import deque
from tqdm import tqdm

from open_translator.glossary_rag import GlossaryRAG

# https://api.deepseek.com/v1/
# DEEPSEEK_API_KEY


def translate_subtitle_v2(
    original: list[str],
    rag: GlossaryRAG = None,
    base_url: str = "https://api.openai.com/v1/",
    api_key_env_var: str = "OPENAI_API_KEY",
    model: str = "gpt-4o-mini",
    subtitle_history_length: int = 5
) -> list[str]:
    """Translate a list of subtitle lines using OpenAI's API.

    Args:
        original (list[str]): List of original subtitle lines to translate
        rag (RAG, optional): RAG instance for glossary lookup. Defaults to None.
        base_url (str, optional): Base URL for OpenAI API. Defaults to "https://api.openai.com/v1/".
        api_key_env_var (str, optional): Environment variable name containing API key. Defaults to "OPENAI_API_KEY".
        model (str, optional): OpenAI model to use. Defaults to "gpt-4o-mini".
        subtitle_history_length (int, optional): Number of previous subtitle pairs to maintain as context. Defaults to 5.

    Returns:
        list[str]: List of translated subtitle lines
    """

    client = OpenAI(
        api_key=os.environ.get(api_key_env_var),
        base_url=base_url
    )

    translated = []
    history = deque(maxlen=subtitle_history_length * 2)

    with open("prompt_template/glossary_v2.txt", "r") as file:
        system_prompt = file.read()

    for i in tqdm(range(len(original))):
        if rag:
            # Query glossary definitions and format results
            glossary_df = rag.query(
                original[i],
                n_results=3
            )

            glossary = ""
            for _, row in glossary_df.iterrows():
                # glossary += f"Term: {row['Term']}\n"
                glossary += f"Definition: {row['Definition']}\n"
                glossary += f"Example: {row['Example']}\n\n"

            formmated_system_prompt = system_prompt.format(
                glossary=glossary.strip()
            )
        else:
            formmated_system_prompt = system_prompt

        # Prepare the messages for the API
        messages = [
            {
                "role": "system",
                "content": formmated_system_prompt,
            },
        ]
        messages.extend(list(history))
        messages.append({
            "role": "user",
            "content": original[i],
        })

        # Call the OpenAI Chat API
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=1.3
        )

        translated_text = response.choices[0].message.content.strip()
        translated.append(translated_text)

        # Update the history
        history.append({
            "role": "user",
            "content": original[i],
        })
        history.append({
            "role": "assistant",
            "content": translated_text,
        })

    return translated
