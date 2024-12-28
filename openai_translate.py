import os
from openai import OpenAI
from collections import deque
from tqdm import tqdm

from rag import RAG

# Initialize OpenAI client
# client = OpenAI(
#     api_key=os.environ.get("OPENAI_API_KEY")
# )
client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com")


def translate_subtitle_v2(original: list[str], rag: RAG, model: str = "gpt-4o-mini", subtitle_history_length: int = 5) -> list[str]:
    translated = []
    history = deque(maxlen=subtitle_history_length * 2)

    for i in tqdm(range(len(original))):
        # Query glossary definitions
        docs = rag.query(
            "Find possible definition involved in words in the sentence below: " + original[i], n_results=5
        )
        glossary = ""
        for doc in docs:
            glossary += "Definition: " + doc.page_content + \
                "\nSample Translation: " + \
                doc.metadata["example_translation"] + "\n"

        # Build the system prompt
        system_prompt = f"""
        You are a multilingual subtitle translation assistant. Translate the following English subtitle into Simplified Chinese (Mandarin).
        This is a Valorant analysis video, and you may encounter technical terms related to the game.
        The word "Vitality" should always be translated to "VIT" and "Trace" or "Trace Esports" should always be translated to "TE".
        Refer to the following definition of technical terms and example translation:
        ---
        {glossary}
        ---
        Ensure the translation is natural and culturally localized, avoiding overly direct English phrasing.
        Consider the preceding and following sentences to maintain contextual accuracy and flow.
        The word "Vitality" should always be translated to "VIT" and "Trace" or "Trace Esports" should always be translated to "TE".

        Reply **only** with the translated text and nothing else.
        """

        # Prepare the messages for the API
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]
        messages.extend(list(history))
        messages.append({
            "role": "user",
            "content": "Translate the following sentence while paying attention to the sentence before this: " + original[i],
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
            "content": "Translate the following sentence while paying attention to the sentence before this: " + original[i],
        })
        history.append({
            "role": "assistant",
            "content": translated_text,
        })

    return translated
