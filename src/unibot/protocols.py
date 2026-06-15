"""Abstract interfaces (protocols) following ISP and DIP principles."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class KnowledgeIndexer(Protocol):
    """Interface for indexing documents into a vector store."""

    def load(self, recreate: bool = False) -> None:
        """Load and index documents into the vector store.

        Args:
            recreate: If True, drop and rebuild the entire index.
        """
        ...


@runtime_checkable
class SearchProvider(Protocol):
    """Interface for external web search providers (fallback)."""

    def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        """Run a web search and return a list of result dicts.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of dicts with at least 'title', 'href', and 'body' keys.
        """
        ...


@runtime_checkable
class AgentFactory(Protocol):
    """Interface for building a configured Agno Agent instance."""

    def build(self, session_id: str, user_id: str) -> object:
        """Construct and return an Agno Agent for the given session and user.

        Args:
            session_id: Unique identifier for the current conversation session.
            user_id: Identifier for the end-user (drives long-term memory).

        Returns:
            A configured agno.agent.Agent instance.
        """
        ...
