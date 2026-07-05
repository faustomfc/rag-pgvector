import logging

import ollama

from src.utils.constants import LLM_MODEL, MAX_HISTORY

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """
You are a search query optimizer for a vector database.

Given the conversation history and the current question, rewrite the question
into a concise, self-contained search query that retrieves relevant documents.

Rules:
- Return ONLY the rewritten query, no explanation
- The output must be a statement or question suitable for document retrieval
- Do NOT ask questions back to the user
- If the question is already clear, return it as-is

HISTORY:
{history}

CURRENT QUESTION:
{question}

SEARCH QUERY:
"""


def rewrite_query(question: str, history: list[dict]) -> str:
    if not history:
        return question

    last_turns = history[-MAX_HISTORY:]
    history_text = "".join(
        f"\nUser:\n{turn['question']}\n\nAssistant:\n{turn['answer']}\n"
        for turn in last_turns
    )

    prompt = _REWRITE_PROMPT.format(history=history_text, question=question)

    logger.info("Rewriting query with conversation context.")

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    rewritten = response["message"]["content"].strip()

    if not rewritten or "?" in rewritten and len(rewritten) > len(question) * 2:
        logger.warning("Rewritten query seems invalid, using original.")
        return question

    logger.info(f"Rewritten query: {rewritten}")
    return rewritten
