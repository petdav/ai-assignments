"""Generate and validate a dictionary entry with the OpenAI Responses API."""

from __future__ import annotations

import json
import os
import sys
from importlib import import_module
from typing import NoReturn

from openai import APIConnectionError, APIError, APIStatusError, OpenAI
from pydantic import BaseModel, Field, ValidationError

PROVIDERS = {
    "gemini": {
        "type": "openai",
        "key": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.7-flash",
    },
    "claude": {
        "type": "anthropic",
        "key": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1/",
        "model": "claude-opus-5",
    },
    "deepseek": {
        "type": "json",
        "key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    },
    "openai": {
        "type": "openai",
        "key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.6-luna",
    },
}


class DictionaryEntry(BaseModel):
    """The schema required for a generated dictionary entry."""

    word: str = Field(min_length=1)
    definitions: list[str] = Field(min_length=1)
    synonyms: list[str]
    antonyms: list[str]
    examples: list[str]


SYSTEM_PROMPT = """You create dictionary entries for the requested word.
Accept words in English or Finnish and provide the results in the same language.
Return only data matching the supplied DictionaryEntry schema. Keep the word
exactly as requested, and provide useful definitions, synonyms, antonyms, and
example sentences, list the examples separately and not with the definitions. 
The definitions list must contain at least one item.
"""


def fail(message: str) -> NoReturn:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)

def require_environment_variable(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Environment variable {name} is missing. "
            f"Set the API key before running the program."
        )

    return value


def validate_entry(value: object) -> DictionaryEntry:
    try:
        entry = DictionaryEntry.model_validate(value)
    except ValidationError as error:
        raise ValueError(f"The model output failed validation: {error}") from error

    if not entry.definitions:
        raise ValueError("The model output must contain at least one definition.")
    return entry


def generate_openai_entry(
    word: str, provider_name: str, provider: dict[str, str]
) -> DictionaryEntry:
    """Generate schema-parsed output from OpenAI-compatible endpoints."""
    client = OpenAI(
        api_key=require_environment_variable(provider["key"]),
        base_url=provider["base_url"],
        timeout=60,
    )

    try:
        response = client.chat.completions.parse(
            model=provider["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Create a dictionary entry for this word: {word}",
                },
            ],
            response_format=DictionaryEntry,
        )
    except APIConnectionError as error:
        raise RuntimeError(
            f"Could not connect to the {provider_name} API: {error}"
        ) from error
    except APIStatusError as error:
        detail = error.message or "The API returned a status error."
        raise RuntimeError(
            f"{provider_name} API request failed ({error.status_code}): {detail}"
        ) from error
    except APIError as error:
        raise RuntimeError(f"{provider_name} API request failed: {error}") from error

    message = response.choices[0].message
    if message.parsed is None:
        raise ValueError(f"The {provider_name} model did not return a valid entry.")
    return validate_entry(message.parsed)


def generate_json_entry(
    word: str, provider_name: str, provider: dict[str, str]
) -> DictionaryEntry:
    """Generate JSON output for providers without native schema parsing."""
    client = OpenAI(
        api_key=require_environment_variable(provider["key"]),
        base_url=provider["base_url"],
        timeout=60,
    )

    try:
        response = client.chat.completions.create(
            model=provider["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Create a dictionary entry for this word: {word}",
                },
            ],
            response_format={"type": "json_object"},
        )
    except APIConnectionError as error:
        raise RuntimeError(
            f"Could not connect to the {provider_name} API: {error}"
        ) from error
    except APIStatusError as error:
        detail = error.message or "The API returned a status error."
        raise RuntimeError(
            f"{provider_name} API request failed ({error.status_code}): {detail}"
        ) from error
    except APIError as error:
        raise RuntimeError(f"{provider_name} API request failed: {error}") from error

    content = response.choices[0].message.content
    if not content:
        raise ValueError(f"The {provider_name} model returned empty output.")

    try:
        return validate_entry(json.loads(content))
    except json.JSONDecodeError as error:
        raise ValueError(f"The {provider_name} model returned invalid JSON.") from error


def generate_anthropic_entry(
    word: str, provider_name: str, provider: dict[str, str]
) -> DictionaryEntry:
    """Generate JSON from Anthropic and validate it with DictionaryEntry."""
    try:
        anthropic_module = import_module("anthropic")
        anthropic_client = anthropic_module.Anthropic
    except ImportError as error:
        raise RuntimeError(
            "Claude support requires the anthropic package. "
            "Install it with: python -m pip install anthropic"
        ) from error

    client = anthropic_client(api_key=require_environment_variable(provider["key"]))
    try:
        response = client.messages.create(
            model=provider["model"],
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Create a dictionary entry for this word: {word}",
                }
            ],
        )
    except Exception as error:
        raise RuntimeError(f"{provider_name} API request failed: {error}") from error

    content = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    if not content:
        raise ValueError(f"The {provider_name} model returned empty output.")

    try:
        return validate_entry(json.loads(content))
    except json.JSONDecodeError as error:
        raise ValueError(f"The {provider_name} model returned invalid JSON.") from error


def generate_entry(word: str, provider_name: str) -> DictionaryEntry:
    provider_name = provider_name.strip().lower()
    provider = PROVIDERS.get(provider_name)
    if not provider:
        fail(f"Unknown provider: {provider_name}")

    provider_type = provider["type"]
    if provider_type == "anthropic":
        return generate_anthropic_entry(word, provider_name, provider)
    if provider_type == "json":
        return generate_json_entry(word, provider_name, provider)
    return generate_openai_entry(word, provider_name, provider)


def main() -> None:
    word = input("Enter a word: ").strip()
    if not word:
        fail("A word is required.")

    provider_input = input("Enter the provider (e.g., OpenAI, gemini): ").strip().lower()
    if provider_input not in PROVIDERS:
        fail(f"Unknown provider: {provider_input}")

    try:
        entry = generate_entry(word, provider_input)
    except (RuntimeError, ValueError) as error:
        fail(str(error))

    print(json.dumps(entry.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()