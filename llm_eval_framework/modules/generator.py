# modules/generator.py — LLM answer generation via local Ollama
import time
import ollama
from config import GENERATOR_MODEL, OLLAMA_HOST


def generate_answer(
    prompt: str,
    context_docs: list,
    with_context: bool = True,
    model: str = None,
) -> dict:
    """
    Generate an answer using Ollama.

    Returns:
        {
            "response": str,
            "token_logprobs": list[float]   # empty list when logprobs unavailable
            "token_data": list[{"token": str, "logprob": float}]
        }
    """
    m = model or GENERATOR_MODEL
    context_str = "\n\n".join(context_docs) if (with_context and context_docs) else ""

    system_msg = (
        "You are a helpful assistant that answers questions based on the provided context. "
        "Be accurate, concise, and stay grounded in the context."
    )
    if with_context and context_str:
        user_msg = f"Context:\n{context_str}\n\nQuestion: {prompt}"
    else:
        user_msg = f"Question: {prompt}"

    # Retry up to 3 times to handle stale connections (common after long NLI GPU runs)
    last_err = None
    for attempt in range(3):
        try:
            client = ollama.Client(host=OLLAMA_HOST)
            response = client.chat(
                model=m,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                options={"temperature": 0.0},
                logprobs=True,
            )
            break  # success
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2 ** attempt)  # 1s, 2s back-off
    else:
        raise last_err

    token_logprobs = []
    token_data = []       # list of {"token": str, "logprob": float} — used for per-claim confidence
    if response.logprobs:
        for lp in response.logprobs:
            token_logprobs.append(lp.logprob)
            token_data.append({"token": lp.token, "logprob": lp.logprob})

    return {
        "response": response.message.content,
        "token_logprobs": token_logprobs,
        "token_data": token_data,   # per-token (text + logprob) for claim-level confidence
    }
