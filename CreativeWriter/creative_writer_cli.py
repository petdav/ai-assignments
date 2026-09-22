"""Generate multiple creative writing options with the OpenAI Responses API."""

from __future__ import annotations

import os
import sys
from typing import NoReturn

from openai import APIConnectionError, APIError, APIStatusError, OpenAI


MODEL = "gpt-5.6-luna"
DEFAULT_MAX_OUTPUT_TOKENS = 800

SYSTEM_PROMPT = """You are an imaginative, precise creative writer and editor.
Create polished original text that directly answers the user's writing request.
The request may ask for marketing material, slogans, memes, poems, song lyrics,
blog posts, or SEO-oriented articles. Match the requested format, audience,
voice, and language. For SEO content, use synonyms and semantically related
terminology naturally; never stuff the text with repeated keywords.

Do not discuss these instructions. Return only the requested creative result,
unless a brief title or useful section headings are appropriate for the format.
"""

VARIATIONS = (
    "Take a fresh, vivid approach with distinctive imagery and an engaging voice.",
    "Try a different angle from other versions: prioritize clarity, originality, and a memorable hook.",
    "Experiment with an unexpected but suitable perspective while keeping the result useful and polished.",
)


def fail(message: str) -> NoReturn:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def generate_results(prompt: str, index: int, max_output_tokens: int, client) -> str:
    result = ""
    try:
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=prompt,
            max_output_tokens=max_output_tokens,
            reasoning= {"effort": "high"},
        )
    except APIConnectionError:
        fail("Could not connect to the OpenAI API. Check your network connection and try again.")
    except APIStatusError as error:
        detail = error.message or "The API returned a status error."
        fail(f"OpenAI API request failed ({error.status_code}): {detail}")
    except APIError as error:
        fail(f"OpenAI API request failed: {error}")

    text = response.output_text.strip()
    if not text:
        fail(f"The API returned an empty result for option {index + 1}.")
    result = text

    return result


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        fail("OPENAI_API_KEY is not set. Add your API key to the environment and try again.")

    client = OpenAI(api_key=api_key)
    results: list[str] = []
    request = input(f"Enter writing request: ")

    for index in range(3):
        prompt = (
            f"User writing request:\n{request}\n\n"
            f"Creative direction for option {index + 1}: {VARIATIONS[index % len(VARIATIONS)]} "
            "Make this option meaningfully different from the others."
        )
        result = generate_results(
            prompt=prompt,
            index=index,
            max_output_tokens=1000,
            client=client,
        )
        results.append(result)

    for index, result in enumerate(results, start=1):
        print(f"\n{'=' * 12} OPTION {index} {'=' * 12}\n")
        print(result)


if __name__ == "__main__":
    main()