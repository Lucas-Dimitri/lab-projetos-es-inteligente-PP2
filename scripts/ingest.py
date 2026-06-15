"""Ingestion pipeline: load PDFs into the LanceDB vector store."""

import argparse
import sys
from pathlib import Path

from unibot.config import get_settings
from unibot.rag.embedder import build_embedder
from unibot.rag.knowledge_base import (
    build_knowledge_base,
    build_pdf_reader,
    build_vector_db,
    index_documents,
)


def ingest(docs_dir: Path, chunk_size: int, recreate: bool) -> None:
    """Index all PDFs in docs_dir into the vector store.

    Chunking strategy:
    - FixedSizeChunking divides each document page into windows of
      `chunk_size` tokens with 20% overlap between adjacent chunks.
    - Default chunk_size: 500 tokens (configurable via UNIBOT_CHUNK_SIZE).
    - Hybrid search (semantic + BM25) is enabled in LanceDB via Tantivy.
    - Fallback: triggered by the agent's LLM judgment when retrieved
      chunks are not relevant; DuckDuckGo is used as the external source.

    Args:
        docs_dir: Directory containing institutional PDF files.
        chunk_size: Token budget for each text chunk.
        recreate: If True, re-index all documents from scratch.
    """
    settings = get_settings()
    pdf_files = list(docs_dir.glob("*.pdf"))

    if not pdf_files:
        print(
            f"[WARN] No PDF files found in '{docs_dir}'. "
            "Run 'python -m scripts.download_docs' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) in '{docs_dir}'.")
    print(f"Embedding model : {settings.unibot_embedding_model}")
    print(f"Vector store    : {settings.vector_store_uri}")
    print(f"Chunk size      : {chunk_size} tokens (overlap: {chunk_size // 5})")
    print(f"Recreate index  : {recreate}\n")

    embedder = build_embedder(settings.unibot_embedding_model)
    vector_db = build_vector_db(settings.vector_store_uri, embedder)
    knowledge = build_knowledge_base(
        vector_db=vector_db,
        max_results=settings.unibot_num_retrieved_docs,
    )
    reader = build_pdf_reader(chunk_size)

    print("Indexing documents (this may take a few minutes on first run)...")
    count = index_documents(
        knowledge=knowledge,
        docs_dir=docs_dir,
        reader=reader,
        recreate=recreate,
    )
    print(f"Ingestion complete. {count} PDF(s) processed. Vector store is ready.")


def main() -> None:
    """CLI entry point: parse arguments and run the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Ingest institutional PDF documents into the UniBot vector store. "
            "Run this after downloading documents with download_docs.py."
        )
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=None,
        help="Directory containing PDFs (default: value of UNIBOT_DOCS_DIR or 'docs/').",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Token size per chunk (default: value of UNIBOT_CHUNK_SIZE or 500).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and rebuild the entire vector index from scratch.",
    )
    args = parser.parse_args()

    settings = get_settings()
    docs_dir: Path = args.docs_dir or settings.unibot_docs_dir
    chunk_size: int = args.chunk_size or settings.unibot_chunk_size
    settings.unibot_vector_store_dir.mkdir(parents=True, exist_ok=True)
    settings.unibot_data_dir.mkdir(parents=True, exist_ok=True)

    ingest(docs_dir=docs_dir, chunk_size=chunk_size, recreate=args.recreate)


if __name__ == "__main__":
    main()
