import logging
from threading import Lock

from src.chat.history import ConversationHistory

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, ConversationHistory] = {}
        self._lock = Lock()

    def get_or_create(self, session_id: str) -> ConversationHistory:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = ConversationHistory()
                logger.info(f"New session created: {session_id}")
            return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Session deleted: {session_id}")

    def active_sessions(self) -> int:
        return len(self._sessions)
