"""Agno agent factory with RAG, memory, and external search."""

from typing import Any

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.knowledge import Knowledge
from agno.memory.manager import MemoryManager

from unibot.config import Settings
from unibot.models import build_model
from unibot.rag.embedder import build_embedder
from unibot.rag.knowledge_base import build_knowledge_base, build_vector_db
from unibot.memory.storage import build_memory_manager, build_session_db
from unibot.tools.search import build_search_tools

_SYSTEM_PROMPT = """\
You are UniBot, a helpful assistant for the Universidade Federal do Pampa (UNIPAMPA).
You answer questions from students, staff, and the general public about UNIPAMPA's
institutional documents (resolutions, academic calendar, course syllabi, etc.).

## How to answer

1. **Always search the knowledge base first** using the search_knowledge tool.
2. If you find relevant content, ground your answer entirely in those documents.
   At the end of your response, always add a "**Fontes:**" section listing:
   - The document file name
   - The section or page used
3. **If the knowledge base returns no relevant results**, use the DuckDuckGo search tool
   and explicitly state:

   "⚠️ Esta informação não foi encontrada nos documentos institucionais da UNIPAMPA.
   A seguir, resultado de busca externa:"

   before presenting external results. Also add a "**Fontes:**" section with the URLs.
4. Never fabricate institutional rules, dates, or resolutions. If uncertain, say so.

## Memory

- Remember key facts the user shares about themselves (name, course, campus, preferences).
- Greet returning users by name and use their context to personalise responses.

## Language

- Respond in Portuguese (pt-BR) unless the user writes in another language.
- Keep responses concise and well-structured.
"""


def _build_model_kwargs(settings: Settings) -> dict[str, Any]:
    """Extract provider-specific keyword arguments from settings.

    Args:
        settings: Application settings.

    Returns:
        Dict of extra kwargs forwarded to the model builder.
    """
    if settings.unibot_llm_provider == "ollama":
        return {"host": settings.unibot_ollama_host}
    return {}


def build_knowledge(settings: Settings) -> Knowledge:
    """Assemble the Knowledge base from settings (DIP: injected settings).

    Args:
        settings: Application settings containing paths and model config.

    Returns:
        A Knowledge instance backed by LanceDB.
    """
    embedder = build_embedder(settings.unibot_embedding_model)
    vector_db = build_vector_db(settings.vector_store_uri, embedder)
    return build_knowledge_base(
        vector_db=vector_db,
        max_results=settings.unibot_num_retrieved_docs,
    )


def build_agent(
    session_id: str,
    user_id: str,
    knowledge: Knowledge,
    db: SqliteDb,
    memory_manager: MemoryManager,
    model: Any,
) -> Agent:
    """Construct a fully configured Agno agent for a session.

    Follows DIP: all dependencies are injected rather than instantiated here.

    Args:
        session_id: Unique ID for this conversation session.
        user_id: Persistent user identifier (drives long-term memory).
        knowledge: Pre-built Knowledge base with institutional documents.
        db: SQLite database for conversation history persistence.
        memory_manager: Long-term memory manager for cross-session preferences.
        model: Any Agno-compatible LLM model instance.

    Returns:
        A ready-to-run Agno Agent.
    """
    return Agent(
        model=model,
        knowledge=knowledge,
        tools=[build_search_tools()],
        db=db,
        memory_manager=memory_manager,
        session_id=session_id,
        user_id=user_id,
        description=_SYSTEM_PROMPT,
        search_knowledge=True,
        add_history_to_context=True,
        num_history_runs=10,
        enable_user_memories=True,
        add_memories_to_context=True,
        update_memory_on_run=False,
        debug_mode=False,
        markdown=True,
    )


def create_agent(session_id: str, user_id: str, settings: Settings) -> Agent:
    """Top-level factory: wire all components and return a ready agent.

    This is the single entry point used by the Streamlit interface.

    Args:
        session_id: Unique ID for this conversation session.
        user_id: Persistent user identifier.
        settings: Application settings.

    Returns:
        A configured, ready-to-run Agno Agent.
    """
    settings.unibot_data_dir.mkdir(parents=True, exist_ok=True)
    extra = _build_model_kwargs(settings)
    main_model = build_model(settings.unibot_llm_provider, settings.unibot_model_id, **extra)
    memory_model = build_model(
        settings.unibot_llm_provider, settings.unibot_memory_model_id, **extra
    )
    knowledge = build_knowledge(settings)
    db = build_session_db(settings.sessions_db_path)
    memory_manager = build_memory_manager(memory_model, db)
    return build_agent(
        session_id=session_id,
        user_id=user_id,
        knowledge=knowledge,
        db=db,
        memory_manager=memory_manager,
        model=main_model,
    )
