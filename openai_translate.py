import os
from openai import OpenAI
from collections import deque
from tqdm import tqdm

from rag import RAG

# https://api.deepseek.com/v1/
# DEEPSEEK_API_KEY


def translate_subtitle_v2(
    original: list[str],
    rag: RAG = None,
    base_url: str = "https://api.openai.com/v1/",
    api_key_env_var: str = "OPENAI_API_KEY",
    model: str = "gpt-4o-mini",
    subtitle_history_length: int = 5
) -> list[str]:

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
            # Query glossary definitions
            docs = rag.query(
                "Find possible definition involved in words in the sentence below: " + original[i], n_results=5
            )
            glossary = ""
            for doc in docs:
                glossary += "Definition: " + doc.page_content + \
                    "\nSample Translation: " + \
                    doc.metadata["example_translation"] + "\n"
            formmated_system_prompt = system_prompt.format(
                glossary=glossary
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
