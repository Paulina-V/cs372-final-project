"""Shared OpenAI-compatible LLM client utilities."""

from openai import OpenAI

from src.config import MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL


def get_client() -> OpenAI:
    """Create an OpenAI client, optionally pointed at Duke's gateway."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")

    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL

    return OpenAI(**kwargs)


def chat_completion(messages: list[dict], max_tokens: int = 2000, temperature: float = 0.2) -> str:
    """Call an OpenAI-compatible chat completion endpoint and return text."""
    response = get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()
