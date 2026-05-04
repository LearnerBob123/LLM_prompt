# modules/claim_extractor.py — Decomposes a response into atomic factual claims
import json
import ollama
from config import CLAIM_EXTRACTOR_MODEL, OLLAMA_HOST

CLAIM_EXTRACTION_PROMPT = """\
You are an expert fact-checker. Given a text response, extract all atomic factual claims.

Rules:
- Each claim must be a single, self-contained factual statement.
- Do NOT include opinions, hedges, or meta-commentary.
- Do NOT merge two facts into one claim.
- Output a JSON object with a single key "claims" whose value is a list of strings.
- No preamble, no explanation — only the JSON object.

Example input:
"The Eiffel Tower was built in 1889 and is located in Paris. It is 330 meters tall."

Example output:
{{"claims": ["The Eiffel Tower was built in 1889.", "The Eiffel Tower is located in Paris.", "The Eiffel Tower is 330 meters tall."]}}

Now extract claims from this response:
{response}"""


def extract_claims(response: str) -> list:
    """
    Returns a list of atomic factual claim strings extracted from the response.
    Returns [] on any failure.
    """
    prompt = CLAIM_EXTRACTION_PROMPT.format(response=response)
    client = ollama.Client(host=OLLAMA_HOST)

    result = client.chat(
        model=CLAIM_EXTRACTOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
        format="json",
    )

    raw = result.message.content.strip()
    # Strip markdown code fences if the model added them anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [c for c in data if isinstance(c, str) and c.strip()]
        if isinstance(data, dict):
            claims = data.get("claims", data.get("claim", []))
            if isinstance(claims, list):
                return [c for c in claims if isinstance(c, str) and c.strip()]
    except (json.JSONDecodeError, AttributeError):
        pass

    return []
