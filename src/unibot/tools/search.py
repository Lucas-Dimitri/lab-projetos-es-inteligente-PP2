"""DuckDuckGo-based external search tool for the fallback mechanism."""

from agno.tools.duckduckgo import DuckDuckGoTools


def build_search_tools() -> DuckDuckGoTools:
    """Return a DuckDuckGo tool bundle for use in the Agno agent.

    DuckDuckGo requires no API key, making it a zero-cost fallback
    for questions outside the institutional corpus.

    Returns:
        A DuckDuckGoTools instance exposing web-search capabilities
        to the Agno agent's tool-calling loop.
    """
    return DuckDuckGoTools(
        enable_search=True,
        enable_news=False,
        fixed_max_results=5,
    )
