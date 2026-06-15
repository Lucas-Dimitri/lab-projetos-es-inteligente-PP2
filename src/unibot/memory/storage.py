"""Storage factories for session history and long-term user memory."""

from typing import Any

from agno.db.sqlite import SqliteDb
from agno.memory.manager import MemoryManager


def build_session_db(db_path: str) -> SqliteDb:
    """Create a SQLite database for per-session conversation history.

    Agno stores each session's messages keyed by session_id, restoring
    context automatically on subsequent runs within the same session.

    Args:
        db_path: Filesystem path to the SQLite database file.

    Returns:
        A configured SqliteDb instance ready to be passed to Agent(db=...).
    """
    return SqliteDb(db_file=db_path)


def build_memory_manager(model: Any, db: SqliteDb) -> MemoryManager:
    """Create a MemoryManager for persistent long-term user memories.

    Agno's MemoryManager automatically extracts facts about the user
    (name, course, campus, preferences) from conversation turns using
    a lightweight LLM call, then persists them in SQLite for recall
    in future sessions.

    Args:
        model: Any Agno-compatible model instance for memory extraction.
        db: Shared SqliteDb instance for memory persistence.

    Returns:
        A configured MemoryManager ready to be passed to Agent(memory_manager=...).
    """
    return MemoryManager(model=model, db=db)
