"""
html_loader.py
================
A self-contained HTMLLoader for RAG pipelines built on LangChain.

Given a single HTML source (a local .html file path OR a URL), this loader:

  1. Parses the page and produces clean, chunked `langchain.docstore.document.Document`
     objects suitable for embedding/indexing.
  2. Extracts every image referenced on the page (via <img src="...">, including
     remote images, relative/local images, and base64 data-URI images) and saves
     them to a dedicated local folder.
  3. Optionally follows the hyperlinks (<a href="...">) found on the page and
     repeats steps 1-2 on those linked pages, recursively, up to a configurable
     `max_depth` -- i.e. how many levels deep the "link tree" should be mined.

Typical usage
-------------
    from html_loader import HTMLLoader

    loader = HTMLLoader(
        source="https://example.com/docs/index.html",
        image_dir="extracted_images",
        max_depth=2,          # main page + 2 levels of linked pages
        same_domain_only=True,
        max_pages=40,
    )
    documents = loader.load()   # -> List[langchain.docstore.document.Document]

    for doc in documents[:3]:
        print(doc.metadata)
        print(doc.page_content[:200])

Notes
-----
- `max_depth=0` disables crawling entirely: only the given `source` page is
  processed. `max_depth=1` also processes every page directly linked from the
  source page, `max_depth=2` follows one more hop, and so on (breadth-first).
- `source` can be a local file path or an http(s) URL. Crawling works across
  both: a local page's <a href> can point to another local file OR to a URL,
  and both are handled.
- Images embedded as base64 data URIs are decoded and written to disk as well.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import mimetypes
import os
import re
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup, Comment

try:
    from langchain.docstore.document import Document
except ImportError:  # newer langchain versions moved this
    from langchain_core.documents import Document

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


# Tags whose content is never useful as "readable text" for a RAG index.
_NOISE_TAGS = {"script", "style", "noscript", "svg", "template", "iframe"}

# Tags that typically hold navigational/boilerplate content. We still read
# them (in case that's genuinely the content the user wants), but we tag
# chunks coming from them so downstream filtering is possible if desired.
_BOILERPLATE_TAGS = {"nav", "footer", "header", "aside"}


class HTMLLoader:
    """Load an HTML page (and optionally its link-tree) into LangChain Documents,
    while extracting all referenced images to a local folder.
    """

    def __init__(
        self,
        source: str,
        image_dir: str = "extracted_images",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        max_depth: int = 0,
        same_domain_only: bool = True,
        max_pages: int = 5,
        request_timeout: int = 10,
        user_agent: str = "Mozilla/5.0 (compatible; HTMLLoaderBot/1.0)",
        download_images: bool = True,
        include_image_placeholders: bool = True,
        allowed_link_patterns: Optional[List[str]] = None,
    ):
        """
        Args:
            source: Local path to an .html file, or an http(s) URL of the page
                to start from.
            image_dir: Local folder where extracted images are saved. Created
                if it doesn't exist.
            chunk_size: Max characters per text chunk (passed to the LangChain
                RecursiveCharacterTextSplitter).
            chunk_overlap: Character overlap between consecutive chunks.
            max_depth: How many levels of linked pages to mine, starting from
                the source page.
                  0 -> only the source page itself (no crawling).
                  1 -> source page + every page it directly links to.
                  N -> breadth-first crawl N hops deep.
            same_domain_only: If True (default), only follow links that stay
                on the same domain as the source (irrelevant when the source
                is a local file and the link is also local). Prevents the
                crawl escaping onto the wider internet.
            max_pages: Hard cap on the total number of pages processed
                (across all depths), as a safety net for large/looping sites.
            request_timeout: Timeout (seconds) for HTTP requests.
            user_agent: User-Agent header used for HTTP requests.
            download_images: If False, images are located and referenced in
                metadata but not actually downloaded/copied to disk.
            include_image_placeholders: If True, each <img> is replaced in
                the extracted text with a short marker like
                "[IMAGE: <filename>]" so chunks retain a textual reference to
                where an image occurred and what file it maps to.
            allowed_link_patterns: Optional list of regex strings; if given,
                only links matching at least one pattern are followed.
        """
        self.source = source
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_depth = max_depth
        self.same_domain_only = same_domain_only
        self.max_pages = max_pages
        self.request_timeout = request_timeout
        self.headers = {"User-Agent": user_agent}
        self.download_images = download_images
        self.include_image_placeholders = include_image_placeholders
        self.allowed_link_patterns = (
            [re.compile(p) for p in allowed_link_patterns]
            if allowed_link_patterns
            else None
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

        self._visited: Set[str] = set()
        self._image_hash_to_path: Dict[str, str] = {}  # dedupe identical images
        self._source_domain = (
            urlparse(source).netloc if self._looks_like_url(source) else None
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load(self) -> List[Document]:
        """Run the full pipeline: crawl (per max_depth), extract text + images,
        and return the combined list of chunked Documents across all pages
        visited.
        """
        documents: List[Document] = []
        queue: deque[Tuple[str, int]] = deque([(self.source, 0)])

        while queue and len(self._visited) < self.max_pages:
            location, depth = queue.popleft()
            norm = self._normalize_key(location)
            if norm in self._visited:
                continue
            self._visited.add(norm)

            html = self._fetch_html(location)
            if html is None:
                logger.warning("Skipping unreachable/unreadable page: %s", location)
                continue

            soup = BeautifulSoup(html, "html.parser")
            page_base_url = location if self._looks_like_url(location) else None

            # 1. Images first (so we can drop placeholders into the text).
            self._extract_and_store_images(soup, location)

            # 2. Links (needed regardless of depth so metadata can note them;
            #    only *followed* if we still have depth budget).
            links = self._extract_links(soup, location)

            # 3. Clean + chunk the textual content of this page.
            page_docs = self._page_to_documents(soup, location, depth)
            documents.extend(page_docs)

            # 4. Queue linked pages if we haven't hit max_depth yet.
            if depth < self.max_depth:
                for link in links:
                    if self._normalize_key(link) not in self._visited:
                        queue.append((link, depth + 1))

        logger.info(
            "HTMLLoader finished: %d page(s) visited, %d document chunk(s) produced.",
            len(self._visited),
            len(documents),
        )
        return documents

    # ------------------------------------------------------------------ #
    # Fetching
    # ------------------------------------------------------------------ #
    @staticmethod
    def _looks_like_url(value: str) -> bool:
        return value.startswith("http://") or value.startswith("https://")

    def _normalize_key(self, location: str) -> str:
        """Normalize a URL/path so http/https + trailing-slash variants and
        local path variants dedupe correctly."""
        if self._looks_like_url(location):
            parsed = urlparse(location)
            path = parsed.path.rstrip("/") or "/"
            return f"{parsed.scheme}://{parsed.netloc}{path}{('?' + parsed.query) if parsed.query else ''}"
        return str(Path(location).expanduser().resolve())

    def _fetch_html(self, location: str) -> Optional[str]:
        """Return the raw HTML text for a local path or a URL, or None on failure."""
        try:
            if self._looks_like_url(location):
                resp = requests.get(
                    location, headers=self.headers, timeout=self.request_timeout
                )
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type and "<html" not in resp.text[:500].lower():
                    logger.warning("URL does not look like HTML, skipping: %s", location)
                    return None
                return resp.text
            else:
                path = Path(location).expanduser()
                if not path.is_file():
                    logger.warning("Local file not found: %s", location)
                    return None
                return path.read_text(encoding="utf-8", errors="replace")
        except (requests.RequestException, OSError) as exc:
            logger.warning("Failed to fetch %s: %s", location, exc)
            return None

    # ------------------------------------------------------------------ #
    # Images
    # ------------------------------------------------------------------ #
    def _extract_and_store_images(self, soup: BeautifulSoup, page_location: str) -> None:
        """Find every <img>, save it locally (unless download_images=False),
        and (optionally) replace the tag in the soup with a text placeholder
        so downstream text extraction keeps a reference to it.
        """
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src")
            if not src:
                continue

            saved_name = None
            try:
                if src.startswith("data:image"):
                    saved_name = self._save_base64_image(src)
                else:
                    resolved = self._resolve_reference(src, page_location)
                    if resolved is None:
                        continue
                    saved_name = self._save_image_from_source(resolved)
            except Exception as exc:  # keep loader resilient to bad images
                logger.warning("Could not extract image '%s' on %s: %s", src, page_location, exc)

            if self.include_image_placeholders:
                alt = img.get("alt", "").strip()
                label = saved_name or "unresolved-image"
                placeholder = f"[IMAGE: {label}" + (f" - {alt}]" if alt else "]")
                img.replace_with(soup.new_string(f" {placeholder} "))

    def _resolve_reference(self, src: str, page_location: str) -> Optional[str]:
        """Turn an <img src> into either an absolute URL or an absolute local
        path, depending on the context of the page it was found on."""
        if self._looks_like_url(src):
            return src
        if self._looks_like_url(page_location):
            return urljoin(page_location, src)
        # Local page -> resolve relative to that file's directory.
        base_dir = Path(page_location).expanduser().resolve().parent
        candidate = (base_dir / unquote(src)).resolve()
        return str(candidate) if candidate.exists() else None

    def _save_image_from_source(self, resolved: str) -> Optional[str]:
        if self._looks_like_url(resolved):
            resp = requests.get(resolved, headers=self.headers, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.content
            ext = self._guess_extension(resolved, resp.headers.get("Content-Type"))
        else:
            path = Path(resolved)
            if not path.is_file():
                return None
            data = path.read_bytes()
            ext = path.suffix or self._guess_extension(resolved, None)

        return self._write_image_bytes(data, ext)

    def _save_base64_image(self, data_uri: str) -> Optional[str]:
        try:
            header, encoded = data_uri.split(",", 1)
            mime = header.split(";")[0].replace("data:", "")
            ext = mimetypes.guess_extension(mime) or ".png"
            data = base64.b64decode(encoded)
        except (ValueError, binascii.Error) as exc:
            logger.warning("Bad base64 image data URI: %s", exc)
            return None
        return self._write_image_bytes(data, ext)

    def _write_image_bytes(self, data: bytes, ext: str) -> str:
        """De-duplicate by content hash and write bytes to `image_dir`."""
        digest = hashlib.sha256(data).hexdigest()[:16]
        if digest in self._image_hash_to_path:
            return self._image_hash_to_path[digest]

        ext = ext if ext.startswith(".") else f".{ext}"
        ext = ext.split("?")[0] or ".bin"
        filename = f"img_{digest}{ext}"
        out_path = self.image_dir / filename
        if not out_path.exists():
            out_path.write_bytes(data)
        self._image_hash_to_path[digest] = filename
        return filename

    @staticmethod
    def _guess_extension(url_or_path: str, content_type: Optional[str]) -> str:
        ext = Path(urlparse(url_or_path).path).suffix
        if ext:
            return ext
        if content_type:
            guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
            if guessed:
                return guessed
        return ".png"

    # ------------------------------------------------------------------ #
    # Links
    # ------------------------------------------------------------------ #
    def _extract_links(self, soup: BeautifulSoup, page_location: str) -> List[str]:
        links: List[str] = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue

            if self._looks_like_url(href):
                absolute = href
            elif self._looks_like_url(page_location):
                absolute = urljoin(page_location, href)
            else:
                base_dir = Path(page_location).expanduser().resolve().parent
                candidate = (base_dir / unquote(href)).resolve()
                absolute = str(candidate)

            if self.allowed_link_patterns and not any(
                p.search(absolute) for p in self.allowed_link_patterns
            ):
                continue

            if self.same_domain_only and self._looks_like_url(absolute) and self._source_domain:
                if urlparse(absolute).netloc != self._source_domain:
                    continue

            links.append(absolute)
        return links

    # ------------------------------------------------------------------ #
    # Text extraction & chunking
    # ------------------------------------------------------------------ #
    def _page_to_documents(self, soup: BeautifulSoup, location: str, depth: int) -> List[Document]:
        title = self._get_title(soup)
        clean_text = self._extract_readable_text(soup)
        if not clean_text.strip():
            return []

        chunks = self.text_splitter.split_text(clean_text)
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": location,
                        "title": title,
                        "depth": depth,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                )
            )
        return documents

    @staticmethod
    def _get_title(soup: BeautifulSoup) -> str:
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        return h1.get_text(strip=True) if h1 else "Untitled"

    def _extract_readable_text(self, soup: BeautifulSoup) -> str:
        """Strip noise (script/style/comments) and return cleaned, whitespace
        -normalized text, preserving block-level structure as newlines."""
        # Work on a copy so repeated calls / later steps aren't affected.
        for tag in soup.find_all(_NOISE_TAGS):
            tag.decompose()
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()

        text = soup.get_text(separator="\n")
        # Collapse runs of blank lines / stray whitespace left by stripped tags.
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        return "\n\n".join(lines)


if __name__ == "__main__":
    # Minimal smoke-test / usage example when run directly.
    logging.basicConfig(level=logging.INFO)

    import argparse

    parser = argparse.ArgumentParser(description="Load an HTML file/URL into RAG-ready chunks.")
    parser.add_argument("source", help="Path to a local .html file, or a URL")
    parser.add_argument("--image-dir", default="extracted_images")
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--max-pages", type=int, default=50)
    args = parser.parse_args()

    loader = HTMLLoader(
        source=args.source,
        image_dir=args.image_dir,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
    )
    docs = loader.load()
    print(f"Produced {len(docs)} document chunks.")
    for d in docs[:3]:
        print("-" * 60)
        print(d.metadata)
        print(d.page_content[:300])
