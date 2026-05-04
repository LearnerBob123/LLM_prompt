# modules/generator.py — LLM answer generation via local Ollama
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

    token_logprobs = []
    if response.logprobs:
        token_logprobs = [lp.logprob for lp in response.logprobs]

    return {
        "response": response.message.content,
        "token_logprobs": token_logprobs,
    }
