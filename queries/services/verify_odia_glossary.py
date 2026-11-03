from __future__ import annotations

import json
import os
import re
from django.conf import settings
from google import genai


MODEL_NAME = "gemini-3.1-flash-lite"

BATCH_SIZE = 30

def get_client():
    api_key = settings.GEMINI_API_KEY

    print(f"api_key is set: {bool(api_key)}")

    return genai.Client(
        api_key=api_key
    )

VERIFY_PROMPT = """
You are an expert Odia agricultural linguist.

You will receive a JSON object mapping English agricultural terms
to Odia translations/transliterations.

For EACH entry:

1. Check whether the Odia text is correct.
2. Correct mistakes.
3. Preserve agricultural terminology.
4. Preserve variety names.
5. Preserve abbreviations like DAP, MOP, NPK, SSP.
6. Use translation when a natural Odia word exists.
7. Use transliteration when no common Odia translation exists.

Return ONLY a JSON object:

{
  "english term": "correct odia"
}

No markdown.
No explanation.
No comments.
"""


def _verify_batch(client: genai.Client, batch: dict[str, str]) -> dict[str, str]:

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            VERIFY_PROMPT,
            json.dumps(batch, ensure_ascii=False)
        ]
    )

    raw = response.text.strip()

    raw = (
        raw.replace("```json", "")
           .replace("```", "")
           .strip()
    )

    try:
        result = json.loads(raw)

        return {
            str(k).lower(): str(v)
            for k, v in result.items()
        }

    except Exception:
        print("[WARN] Failed to parse Gemini response")
        return batch


def verify_glossary(
    draft_path: str,
    final_path: str
) -> None:

    with open(draft_path, "r", encoding="utf-8") as f:
        glossary = json.load(f)

    items = list(glossary.items())

    verified = {}
    client = get_client()

    total_batches = (
        len(items) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    try:
        for i in range(0, len(items), BATCH_SIZE):

            batch_no = (i // BATCH_SIZE) + 1

            print(
                f"[VERIFY] Batch "
                f"{batch_no}/{total_batches}"
            )

            batch = dict(
                items[i:i + BATCH_SIZE]
            )

            verified.update(
                _verify_batch(client, batch)
            )
    finally:
        client.close()

    with open(
        final_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            verified,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True
        )

    print(
        f"Verified glossary saved -> "
        f"{final_path}"
    )
