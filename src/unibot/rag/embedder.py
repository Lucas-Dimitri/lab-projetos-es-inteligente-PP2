"""Embedder factory: returns a configured FastEmbed embedder for Agno."""

from agno.knowledge.embedder.fastembed import FastEmbedEmbedder


def build_embedder(model: str) -> FastEmbedEmbedder:
    """Create a FastEmbed embedder for the given model name.

    FastEmbed runs locally without an API key and supports multilingual
    models, making it suitable for Portuguese institutional documents.

    Args:
        model: A model name supported by FastEmbed, e.g.
               'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'.

    Returns:
        A configured FastEmbedEmbedder instance ready for use in LanceDb.
    """
    return FastEmbedEmbedder(id=model)
