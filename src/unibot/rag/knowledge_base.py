"""Knowledge-base factory: builds the PDF-backed LanceDB knowledge base."""

from pathlib import Path

from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
from agno.knowledge.chunking.fixed import FixedSizeChunking
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.vectordb.lancedb import LanceDb, SearchType

VECTOR_TABLE_NAME = "unibot_knowledge"


def build_vector_db(uri: str, embedder: FastEmbedEmbedder) -> LanceDb:
    """Instantiate the LanceDB vector store with hybrid search enabled.

    Hybrid search combines dense vector similarity with sparse BM25 (Tantivy),
    improving recall for exact terms like resolution numbers or dates.

    Args:
        uri: Path to the LanceDB directory.
        embedder: Embedder used to produce document vectors.

    Returns:
        A configured LanceDb instance.
    """
    return LanceDb(
        uri=uri,
        table_name=VECTOR_TABLE_NAME,
        embedder=embedder,
        search_type=SearchType.hybrid,
    )


def build_pdf_reader(chunk_size: int) -> PDFReader:
    """Create a PDFReader with fixed-size chunking.

    Chunking strategy: FixedSizeChunking divides each page's text into
    windows of `chunk_size` tokens with `overlap` tokens shared between
    consecutive chunks to preserve sentence continuity.

    Args:
        chunk_size: Approximate number of tokens per chunk.

    Returns:
        A configured PDFReader instance.
    """
    return PDFReader(
        chunking_strategy=FixedSizeChunking(
            chunk_size=chunk_size,
            overlap=chunk_size // 5,  # 20% overlap
        ),
        sanitize_content=True,
        split_on_pages=True,
    )


def build_knowledge_base(vector_db: LanceDb, max_results: int) -> Knowledge:
    """Create a Knowledge instance backed by LanceDB.

    Documents are not loaded here; call `knowledge.insert(path=...)` to index.

    Args:
        vector_db: Pre-built LanceDb instance.
        max_results: Number of relevant chunks to retrieve per query.

    Returns:
        A Knowledge instance ready for document insertion and retrieval.
    """
    return Knowledge(
        name="UNIPAMPA Institutional Documents",
        description="Resoluções, calendário acadêmico, projetos pedagógicos e demais documentos da UNIPAMPA.",
        vector_db=vector_db,
        max_results=max_results,
    )


def index_documents(
    knowledge: Knowledge,
    docs_dir: Path,
    reader: PDFReader,
    recreate: bool = False,
) -> int:
    """Index all PDFs found in docs_dir into the knowledge base.

    Args:
        knowledge: The Knowledge instance to insert documents into.
        docs_dir: Directory containing institutional PDF files.
        reader: Configured PDFReader for chunking.
        recreate: If True, re-index even documents that already exist.

    Returns:
        Number of PDF files processed.
    """
    pdf_files = list(docs_dir.glob("*.pdf"))
    for pdf_path in pdf_files:
        knowledge.insert(
            path=str(pdf_path),
            reader=reader,
            skip_if_exists=(not recreate),
        )
    return len(pdf_files)
