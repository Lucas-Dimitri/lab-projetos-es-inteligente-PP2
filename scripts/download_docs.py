"""Download UNIPAMPA institutional PDF documents from the official website."""

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Known UNIPAMPA pages that contain institutional documents
DEFAULT_SEED_URLS: list[str] = [
    "https://sites.unipampa.edu.br/consuni/resolucoes/",
    "https://sites.unipampa.edu.br/consuni/estatuto-da-unipampa/",
    "https://sites.unipampa.edu.br/prograd/calendarios-academicos/",
    "https://sites.unipampa.edu.br/acessoainformacao/institucional/",
    "https://sites.unipampa.edu.br/acessoainformacao/institucional/administracao-geral/",
]

_PDF_LINK_RE = re.compile(r"\.pdf$", re.IGNORECASE)
_REQUEST_DELAY_SECONDS = 1.0


def _sanitise_filename(url: str) -> str:
    """Derive a safe filename from a PDF URL.

    Args:
        url: Full URL of the PDF file.

    Returns:
        A filesystem-safe filename string ending in '.pdf'.
    """
    path = urlparse(url).path
    name = Path(path).name
    safe = re.sub(r"[^\w\-.]", "_", name)
    return safe if safe.endswith(".pdf") else safe + ".pdf"


def _find_pdf_links(page_url: str, session: requests.Session) -> list[str]:
    """Fetch a page and collect all absolute PDF hyperlinks.

    Args:
        page_url: URL of the HTML page to scrape.
        session: Requests session for HTTP calls.

    Returns:
        Deduplicated list of absolute PDF URLs found on the page.
    """
    try:
        response = session.get(page_url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [WARN] Could not fetch {page_url}: {exc}", file=sys.stderr)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    pdf_urls: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href: str = anchor["href"]
        if _PDF_LINK_RE.search(href):
            pdf_urls.append(urljoin(page_url, href))

    return list(dict.fromkeys(pdf_urls))  # preserve order, remove dupes


def _download_pdf(url: str, dest_dir: Path, session: requests.Session) -> bool:
    """Download a single PDF to dest_dir if not already present.

    Args:
        url: URL of the PDF to download.
        dest_dir: Directory where the file will be saved.
        session: Requests session for HTTP calls.

    Returns:
        True if the file was downloaded, False if it already existed.
    """
    filename = _sanitise_filename(url)
    dest_path = dest_dir / filename

    if dest_path.exists():
        print(f"  [SKIP] {filename} already exists.")
        return False

    try:
        response = session.get(url, timeout=30, stream=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [ERROR] Failed to download {url}: {exc}", file=sys.stderr)
        return False

    with dest_path.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=8192):
            fh.write(chunk)

    print(f"  [OK] Downloaded {filename}")
    return True


def download_documents(seed_urls: list[str], dest_dir: Path) -> int:
    """Crawl each seed URL, find PDF links, and download them.

    Args:
        seed_urls: List of UNIPAMPA page URLs to crawl for PDF links.
        dest_dir: Directory where PDFs will be saved.

    Returns:
        Total number of files newly downloaded.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "UniBot/0.1 (+educational-project)"})

    total_downloaded = 0
    for page_url in seed_urls:
        print(f"\nCrawling: {page_url}")
        pdf_links = _find_pdf_links(page_url, session)
        print(f"  Found {len(pdf_links)} PDF link(s).")
        for pdf_url in pdf_links:
            if _download_pdf(pdf_url, dest_dir, session):
                total_downloaded += 1
            time.sleep(_REQUEST_DELAY_SECONDS)

    return total_downloaded


def main() -> None:
    """CLI entry point: parse arguments and run the downloader."""
    parser = argparse.ArgumentParser(
        description="Download UNIPAMPA institutional PDFs to the docs/ directory."
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Destination directory for downloaded PDFs (default: docs/).",
    )
    parser.add_argument(
        "--urls",
        nargs="*",
        default=DEFAULT_SEED_URLS,
        help="Space-separated list of UNIPAMPA page URLs to crawl (overrides defaults).",
    )
    args = parser.parse_args()

    print("UniBot Document Downloader")
    print(f"Destination: {args.docs_dir.resolve()}")
    print(f"Seed URLs:   {len(args.urls)}\n")

    count = download_documents(seed_urls=args.urls, dest_dir=args.docs_dir)
    print(f"\nDone. {count} new file(s) downloaded.")


if __name__ == "__main__":
    main()
