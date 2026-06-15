"""Streamlit chat interface for UniBot."""

import uuid

import streamlit as st
from agno.agent import Agent
from agno.run.agent import RunContentEvent, RunCompletedEvent

from unibot.agent import create_agent
from unibot.config import get_settings

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="UniBot – UNIPAMPA",
    page_icon="🎓",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    """Initialise all required st.session_state keys on first load."""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = ""
    if "agent" not in st.session_state:
        st.session_state["agent"] = None
    if "_cached_user_id" not in st.session_state:
        st.session_state["_cached_user_id"] = None


def _get_or_create_agent(user_id: str) -> Agent:
    """Return the cached agent or build a new one if user_id changed.

    Args:
        user_id: Current user identifier from the sidebar.

    Returns:
        A configured Agno Agent instance.
    """
    if (
        st.session_state["agent"] is None
        or st.session_state["_cached_user_id"] != user_id
    ):
        settings = get_settings()
        settings.unibot_docs_dir.mkdir(parents=True, exist_ok=True)
        settings.unibot_data_dir.mkdir(parents=True, exist_ok=True)
        st.session_state["agent"] = create_agent(
            session_id=st.session_state["session_id"],
            user_id=user_id,
            settings=settings,
        )
        st.session_state["_cached_user_id"] = user_id
    return st.session_state["agent"]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> str:
    """Render the sidebar and return the current user identifier.

    Returns:
        The user_id string entered by the user (may be empty).
    """
    with st.sidebar:
        st.title("🎓 UniBot")
        st.caption("Assistente institucional da UNIPAMPA")
        st.divider()

        st.subheader("Perfil do usuário")
        user_id = st.text_input(
            "Seu nome ou matrícula",
            value=st.session_state.get("user_id", ""),
            placeholder="Ex: João Silva / 2300123456",
            help="Usado para personalizar respostas entre sessões.",
        )
        st.session_state["user_id"] = user_id

        st.divider()
        st.subheader("Sobre")
        st.markdown(
            "UniBot usa **RAG** (Retrieval-Augmented Generation) para responder "
            "perguntas com base nos documentos institucionais da UNIPAMPA.\n\n"
            "Quando a base não cobre a pergunta, ele aciona busca externa via "
            "**DuckDuckGo** e sinaliza a origem.\n\n"
            "Suas preferências são salvas entre sessões via **memória de longo prazo**."
        )
        st.divider()

        if st.button("🗑️ Nova conversa", use_container_width=True):
            st.session_state["session_id"] = str(uuid.uuid4())
            st.session_state["messages"] = []
            st.session_state["agent"] = None
            st.session_state["_cached_user_id"] = None
            st.rerun()

    return user_id


# ---------------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------------


def _render_chat_history() -> None:
    """Re-render all messages stored in session state."""
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def _stream_response(agent: Agent, user_message: str) -> str:
    """Send user_message to the agent and stream the response token by token.

    Args:
        agent: The active Agno Agent.
        user_message: The user's latest chat message.

    Returns:
        The complete assistant response as a string.
    """
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for event in agent.run(user_message, stream=True, stream_events=True):
            if isinstance(event, RunContentEvent) and event.content:
                full_response += str(event.content)
                placeholder.markdown(full_response + "▌")
            elif isinstance(event, RunCompletedEvent) and event.content:
                full_response = str(event.content)

        placeholder.markdown(full_response)
    return full_response


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the Streamlit application."""
    _init_session_state()
    user_id = _render_sidebar()

    st.title("UniBot – Assistente UNIPAMPA")
    st.caption(
        "Faça perguntas sobre resoluções, calendário acadêmico, "
        "projetos pedagógicos e mais."
    )
    st.divider()

    effective_user = user_id.strip() or f"anon-{st.session_state['session_id'][:8]}"
    agent = _get_or_create_agent(effective_user)

    _render_chat_history()

    if prompt := st.chat_input("Sua pergunta sobre a UNIPAMPA..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            response = _stream_response(agent, prompt)
            st.session_state["messages"].append(
                {"role": "assistant", "content": response}
            )
        except Exception as exc:  # noqa: BLE001
            error_msg = f"❌ Erro ao processar sua pergunta: {exc}"
            st.error(error_msg)
            st.session_state["messages"].append(
                {"role": "assistant", "content": error_msg}
            )


if __name__ == "__main__":
    main()
