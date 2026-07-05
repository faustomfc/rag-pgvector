import logging

import ollama

from src.utils.constants import LLM_MODEL, MAX_HISTORY

logger = logging.getLogger(__name__)

_ANSWER_PROMPT = """
You are a technical expert assistant.

Use the retrieved context as your primary source of information.
If the context is insufficient to answer, say so — never fabricate information.

==================================
CONVERSATION HISTORY
==================================

{history}

==================================
RETRIEVED CONTEXT
==================================

{context}

==================================
QUESTION
==================================

{question}

==================================
ANSWER
==================================
"""


def _build_history_text(history: list[dict]) -> str:
    if not history:
        return ""
    return "".join(
        f"\nUser:\n{turn['question']}\n\nAssistant:\n{turn['answer']}\n"
        for turn in history[-MAX_HISTORY:]
    )


def generate_answer(question: str, context_chunks: list[str], history: list[dict]) -> str:
    context = "\n\n".join(context_chunks)
    history_text = _build_history_text(history)

    prompt = _ANSWER_PROMPT.format(
        history=history_text,
        context=context,
        question=question,
    )

    logger.info(f"Generating answer with {len(context_chunks)} context chunks.")

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response["message"]["content"]
    logger.info("Answer generated.")
    return answer
