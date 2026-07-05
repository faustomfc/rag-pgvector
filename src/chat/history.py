import logging

from src.utils.constants import MAX_HISTORY_MEM

logger = logging.getLogger(__name__)


class ConversationHistory:
    def __init__(self, max_size: int = MAX_HISTORY_MEM):
        self._turns: list[dict] = []
        self._max_size = max_size

    def add(self, question: str, answer: str) -> None:
        self._turns.append({"question": question, "answer": answer})
        if len(self._turns) > self._max_size:
            self._turns = self._turns[-self._max_size:]
            logger.debug(f"History trimmed to {self._max_size} turns.")

    def get(self) -> list[dict]:
        return self._turns

    def is_empty(self) -> bool:
        return len(self._turns) == 0
