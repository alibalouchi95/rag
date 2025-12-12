from dotenv import load_dotenv
import ollama

load_dotenv()

# Choose a free model from OpenRouter (example)
MODEL_NAME = "tngtech/deepseek-r1t2-chimera:free"


def local_translate(text: str) -> str:

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert legal translator specializing in Persian (Farsi) "
                "and English law. Translate all text with strict legal accuracy.\n\n"
                "Requirements:\n"
                "- Preserve all legal meaning and intent exactly.\n"
                "- Maintain structure, definitions, obligations, conditions, and formal tone.\n"
                "- Do NOT interpret, paraphrase, summarize, or simplify any concepts.\n"
                "- Use precise legal terminology appropriate for the target language.\n"
                "- Preserve enumerated items, sections, articles, and formatting.\n"
                "- Do NOT add explanations, examples, comments, or clarifications.\n"
                "- Output ONLY the translated text.\n"
            ),
        },
        {
            "role": "user",
            "content": f"Translate this legal text into English accurately:\n\n{text}",
        },
    ]

    response = ollama.chat(model="qwen2-local", messages=messages)
    translated_text = response["message"]["content"]
    return translated_text.strip()
