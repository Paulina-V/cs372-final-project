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


def user_facing_model_error(exc: Exception) -> str:
    """Return a concise model error without exposing raw API payload details."""
    message = str(exc).lower()
    if "incorrect api key" in message or "invalid_api_key" in message or "401" in message:
        return "LLM API key is invalid or still set to the placeholder value. Update `.env` and restart the app."
    if "openai_api_key is not set" in message or "api_key" in message and "not set" in message:
        return "OPENAI_API_KEY is not set. Add your real key to `.env` and restart the app."
    if "base_url" in message or "connection" in message:
        return "Could not reach the configured LLM endpoint. Check OPENAI_BASE_URL, network access, and model settings."
    return f"{type(exc).__name__}: {exc}"
