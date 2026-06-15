"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for UniBot, driven by .env or environment variables."""

    # LLM provider and models
    unibot_llm_provider: str = "ollama"
    unibot_model_id: str = "llama3.2"
    unibot_memory_model_id: str = "llama3.2"

    # Ollama server (only used when unibot_llm_provider = "ollama")
    unibot_ollama_host: str = "http://localhost:11434"

    # Embeddings (local FastEmbed, no API key needed)
    unibot_embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # RAG retrieval
    unibot_num_retrieved_docs: int = 5
    unibot_chunk_size: int = 500

    # Paths
    unibot_docs_dir: Path = Path("docs")
    unibot_vector_store_dir: Path = Path("vector_store")
    unibot_data_dir: Path = Path("data")

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def sessions_db_path(self) -> str:
        """Absolute path to the SQLite sessions database file."""
        return str(self.unibot_data_dir / "sessions.db")

    @property
    def memory_db_path(self) -> str:
        """Absolute path to the SQLite long-term memory database file."""
        return str(self.unibot_data_dir / "memory.db")

    @property
    def vector_store_uri(self) -> str:
        """URI for LanceDB vector store directory."""
        return str(self.unibot_vector_store_dir)


def get_settings() -> Settings:
    """Return a fully-initialised Settings instance."""
    return Settings()
