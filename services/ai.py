from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)


def _schema_to_text(schema: Any) -> str:
    if isinstance(schema, str):
        return schema
    try:
        return json.dumps(schema, indent=2, default=str)
    except Exception:
        return str(schema)


def generate_table_description(
    table_name: str,
    schema: Any,
    user_table_description: str | None = None,
) -> str:
    """
    Generate a concise description for a table.

    The function uses the official Groq SDK for table-description generation.
    """

    schema_text = _schema_to_text(schema)
    prompt = [
        "You are generating a short database table description for retrieval.",
        f"Table name: {table_name}",
        f"Schema: {schema_text}",
    ]

    if user_table_description:
        prompt.append(f"User-provided description/context: {user_table_description}")

    prompt.append(
        "Write 1-2 sentences that describe what this table stores. "
        "If the user provided context exists, incorporate it without contradicting the schema."
    )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Groq table enrichment cannot run.")

    # Disable client-level retries so one table results in one LLM call.
    client = Groq(api_key=api_key, max_retries=0)
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You describe database tables for retrieval systems. "
                    "Return a concise 1-2 sentence description."
                ),
            },
            {
                "role": "user",
                "content": "\n".join(prompt),
            },
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError(f"Groq returned an empty table description for table `{table_name}`.")

    return text
