"""
PyMuPDF-Only Advanced PDF Parser with Section-Aware Chunking
- Heading detection via font-size/boldness heuristics and regex patterns
- Optional TOC alignment to improve heading levels
- Robust section extraction across pages
- Sentence-aware chunk splitting with overlap
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
import numpy as np
from langchain_core.documents import Document


# ----------------------------- Data Models -----------------------------

@dataclass
class PDFSection:
    title: str
    content: str
    page_number: int
    level: int


@dataclass
class PDFChunk:
    text: str
    section_title: str
    section_level: int
    page_number: int
    chunk_index: int
    metadata: Dict


# ----------------------------- Parser -----------------------------

class PyMuPDFAdvancedParser:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_section_length: int = 50,
        toc_assist: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_section_length = min_section_length
        self.toc_assist = toc_assist

    # ------------------------- Public API -------------------------

    def parse(self, pdf_path: str | Path) -> List[PDFChunk]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        with fitz.open(str(pdf_path)) as doc:
            headings = self._detect_headings(doc)
            toc = []
            if self.toc_assist:
                toc = self._read_toc(doc)
                if toc:
                    headings = self._align_headings_with_toc(headings, toc)
            # If still no headings but TOC exists, synthesize from TOC
            if not headings and toc:
                headings = [(pg, title, lvl, 0.0) for (lvl, pg, title) in toc]

            sections = self._extract_sections(doc, headings)

        if not sections:
            # Fallback: single section from entire document
            with fitz.open(str(pdf_path)) as doc:
                full_text = "\n\n".join([page.get_text() for page in doc])
            sections = [PDFSection("Full Document", full_text, 1, 1)]

        return self._create_chunks_from_sections(sections)

    def to_langchain_documents(self, chunks: List[PDFChunk], source_path: str | Path) -> List[Document]:
        documents: List[Document] = []
        for c in chunks:
            metadata: Dict = {
                "source": str(source_path),
                "page_number": c.page_number,
                "section_title": c.section_title,
                "section_level": c.section_level,
                "chunk_index": c.chunk_index,
            }
            # Merge any additional metadata captured during chunking
            metadata.update(c.metadata or {})
            documents.append(Document(page_content=c.text, metadata=metadata))
        return documents

    def parse_to_documents(self, pdf_path: str | Path) -> List[Document]:
        chunks = self.parse(pdf_path)
        return self.to_langchain_documents(chunks, pdf_path)

    def export_chunks_to_json(self, chunks: List[PDFChunk], output_path: str | Path) -> None:
        data = [
            {
                "text": c.text,
                "section_title": c.section_title,
                "section_level": c.section_level,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
            }
            for c in chunks
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ Exported {len(chunks)} chunks to {output_path}")

    def print_summary(self, chunks: List[PDFChunk]) -> None:
        print("\n" + "=" * 60)
        print("PDF PARSING SUMMARY (PyMuPDF)")
        print("=" * 60)
        print(f"Total chunks: {len(chunks)}")
        sections: Dict[str, List[PDFChunk]] = {}
        for c in chunks:
            sections.setdefault(c.section_title, []).append(c)
        print(f"Total sections: {len(sections)}")
        print("\nSections found:")
        for i, (title, sc) in enumerate(sections.items(), 1):
            print(f"  {i}. {title} (Level {sc[0].section_level}) - {len(sc)} chunk(s)")
        print("\n" + "=" * 60)

    # ----------------------- Heading Detection -----------------------

    def _detect_headings(self, doc: fitz.Document) -> List[Tuple[int, str, int, float]]:
        """
        Returns a list of (page_num, heading_text, level, y_top) sorted by page/y.
        Level heuristic is based on font size Z-score banding.
        """
        raw_headings: List[Tuple[int, str, float, bool, float]] = []  # page, text, font_size, is_bold, y

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_width = float(page.rect.width)

            # rawdict preserves spans with size/flags and bounding boxes
            text_dict = page.get_text("rawdict")
            span_sizes: List[float] = []

            for block in text_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        t = span.get("text", "").strip()
                        if not t:
                            continue
                        span_sizes.append(span.get("size", 0.0))

            if not span_sizes:
                continue

            span_sizes_np = np.array(span_sizes)
            median_size = float(np.median(span_sizes_np))
            std_size = float(np.std(span_sizes_np)) if len(span_sizes_np) > 1 else 0.0

            for block in text_dict.get("blocks", []):
                for line in block.get("lines", []):
                    parts: List[str] = []
                    sizes: List[float] = []
                    bold_flags: List[bool] = []
                    center_flags: List[bool] = []
                    y_top = None

                    for span in line.get("spans", []):
                        t = span.get("text", "").strip()
                        if not t:
                            continue
                        parts.append(t)
                        sizes.append(float(span.get("size", 0.0)))
                        font_name = (span.get("font", "") or "").lower()
                        bold_by_flag = (span.get("flags", 0) & 16) != 0
                        bold_by_name = ("bold" in font_name) or ("black" in font_name) or ("heavy" in font_name)
                        bold_flags.append(bold_by_flag or bold_by_name)
                        bbox = span.get("bbox") or [0, 0, 0, 0]
                        if y_top is None:
                            y_top = float(bbox[1])
                        # center detection: midpoint ~ 40%-60% of page width
                        mid_x = (float(bbox[0]) + float(bbox[2])) / 2.0
                        center_flags.append(0.4 * page_width <= mid_x <= 0.6 * page_width)

                    if not parts:
                        continue

                    text = " ".join(parts)
                    if len(text) > 200:  # headings are typically short
                        continue

                    avg_size = float(np.mean(sizes)) if sizes else median_size
                    is_bold = any(bold_flags)
                    is_centered = any(center_flags)

                    # regex/title-shape heuristics
                    looks_like_heading = self._looks_like_heading(text)

                    # font/layout boost heuristic
                    size_boost = avg_size > (median_size + (0.6 * std_size)) if std_size > 0 else avg_size > median_size
                    bold_boost = is_bold and avg_size >= median_size
                    center_boost = is_centered and avg_size >= median_size

                    if looks_like_heading or size_boost or bold_boost or center_boost:
                        raw_headings.append((page_num, text, avg_size, is_bold, y_top or 0.0))

        # Rank levels by font size (bigger => higher level)
        if not raw_headings:
            return []

        sizes = np.array([h[2] for h in raw_headings])
        # Quantile banding into up to 4 levels
        q75 = float(np.quantile(sizes, 0.75))
        q50 = float(np.quantile(sizes, 0.50))
        q25 = float(np.quantile(sizes, 0.25))

        def level_from_size(s: float) -> int:
            if s >= q75:
                return 1
            if s >= q50:
                return 2
            if s >= q25:
                return 3
            return 4

        headings: List[Tuple[int, str, int, float]] = [
            (p, t, level_from_size(s), y) for (p, t, s, _b, y) in raw_headings
        ]
        # Sort by page then by y position
        headings.sort(key=lambda x: (x[0], x[3]))
        return headings

    def _looks_like_heading(self, text: str) -> bool:
        patterns = [
            r"^\d+(?:\.\d+)*[\.\)]\s+\S",   # 1. Title / 1.2.3) Title
            r"^(?:Appendix|Annex|Chapter|Section|Part)\s+[A-Za-z0-9]+\b",
            r"^[IVXLC]+[\.\)]\s+\S",         # I. Title
            r"^[A-Z][A-Z\s\-:&]{2,}$",         # ALL CAPS short lines
            r"^[A-Z][^a-z\n]{2,}$",             # Mostly uppercase
        ]
        t = text.strip()
        if len(t) <= 4:
            return False
        # avoid obvious non-headings like figure/table captions
        if re.match(r"^(Figure|Table)\s+\d+", t):
            return True
        return any(re.match(p, t) for p in patterns)

    def _read_toc(self, doc: fitz.Document) -> List[Tuple[int, int, str]]:
        """Return TOC as list of (level, page, title)."""
        try:
            toc = doc.get_toc(simple=True)  # [(level, title, page), ...]
            # Normalize to (level, page_index, title)
            norm = [(lvl, max(1, pg) - 1, title) for (lvl, title, pg) in toc]
            return norm
        except Exception:
            return []

    def _align_headings_with_toc(
        self,
        headings: List[Tuple[int, str, int, float]],
        toc: List[Tuple[int, int, str]],
    ) -> List[Tuple[int, str, int, float]]:
        """
        Align detected headings with TOC levels when title fuzzy-match is close by page.
        Keeps original order, overrides level when a confident TOC match is found.
        """
        if not headings or not toc:
            return headings if headings else [(pg, title, lvl, 0.0) for (lvl, pg, title) in toc]

        def normalize(s: str) -> str:
            s = re.sub(r"\s+", " ", s.strip().lower())
            s = re.sub(r"[\.:]+$", "", s)
            return s

        toc_by_page: Dict[int, List[Tuple[int, str]]] = {}
        for lvl, pg, title in toc:
            toc_by_page.setdefault(pg, []).append((lvl, normalize(title)))

        adjusted: List[Tuple[int, str, int, float]] = []
        for p, title, level, y in headings:
            nt = normalize(title)
            best_level = level
            # Check same page; if none, check neighbors
            pages_to_check = [p, p - 1, p + 1]
            for cpage in pages_to_check:
                if cpage in toc_by_page:
                    for tlvl, ttitle in toc_by_page[cpage]:
                        # very light fuzzy match: start-with or high overlap of tokens
                        if nt.startswith(ttitle) or ttitle.startswith(nt):
                            best_level = tlvl
                            break
                    if best_level != level:
                        break
            adjusted.append((p, title, int(best_level), y))

        # Merge missing TOC entries that were not detected at all
        detected_titles = {normalize(t) for (_p, t, _l, _y) in adjusted}
        for lvl, pg, title in toc:
            key = normalize(title)
            if key not in detected_titles:
                adjusted.append((pg, title, lvl, 0.0))

        # keep order by page then y
        adjusted.sort(key=lambda x: (x[0], x[3]))
        return adjusted

    # ----------------------- Section Extraction -----------------------

    def _extract_sections(
        self,
        doc: fitz.Document,
        headings: List[Tuple[int, str, int, float]],
    ) -> List[PDFSection]:
        if not headings:
            return []

        # Sort for safety
        headings = sorted(headings, key=lambda x: (x[0], x[3]))

        # Pull page texts once
        page_texts: List[str] = [doc[p].get_text() for p in range(len(doc))]

        sections: List[PDFSection] = []
        for i, (page_num, title, level, _y) in enumerate(headings):
            # Identify the end boundary: next heading of same or higher level
            end_page = min(page_num + 10, len(doc) - 1)  # cap search
            for j in range(i + 1, len(headings)):
                n_page, _ntitle, n_level, _ny = headings[j]
                if n_level <= level:
                    end_page = n_page
                    break

            # Stitch text from start to (exclusive) end_page; include start page, stop before next heading page
            content_parts: List[str] = []
            for p in range(page_num, min(end_page + 1, len(doc))):
                t = page_texts[p]
                if p == page_num:
                    # strip the heading itself if present at start
                    t = re.sub(re.escape(title) + r"\s*\n?", "", t, count=1)
                content_parts.append(t)

            content = "\n\n".join(content_parts).strip()
            if len(content) >= self.min_section_length:
                sections.append(PDFSection(title=title, content=content, page_number=page_num + 1, level=level))

        return sections

    # -------------------------- Chunking --------------------------

    def _create_chunks_from_sections(self, sections: List[PDFSection]) -> List[PDFChunk]:
        chunks: List[PDFChunk] = []
        for section in sections:
            text = section.content
            if len(text) <= self.chunk_size:
                chunks.append(
                    PDFChunk(
                        text=text,
                        section_title=section.title,
                        section_level=section.level,
                        page_number=section.page_number,
                        chunk_index=0,
                        metadata={
                            "chunk_type": "complete_section",
                            "section_title": section.title,
                            "section_level": section.level,
                        },
                    )
                )
                continue

            start = 0
            cidx = 0
            while start < len(text):
                end = start + self.chunk_size
                if end < len(text):
                    # snap to nearest sentence boundary around end
                    search_start = max(start, end - 120)
                    window = text[search_start : min(len(text), end + 120)]
                    # prefer terminal punctuation
                    candidates = [
                        window.rfind(". "),
                        window.rfind(".\n"),
                        window.rfind("! "),
                        window.rfind("? "),
                        window.rfind("; "),
                    ]
                    best = max(candidates)
                    if best > 0:
                        end = search_start + best + 1
                chunk_text = text[start:end].strip()
                if len(chunk_text) >= self.min_section_length:
                    chunks.append(
                        PDFChunk(
                            text=chunk_text,
                            section_title=section.title,
                            section_level=section.level,
                            page_number=section.page_number,
                            chunk_index=cidx,
                            metadata={
                                "chunk_type": "section_part",
                                "section_title": section.title,
                                "section_level": section.level,
                            },
                        )
                    )
                    cidx += 1
                start = max(end - self.chunk_overlap, end)
                if start >= len(text):
                    break
        return chunks


# ----------------------------- CLI -----------------------------

def main() -> None:
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "sample.pdf"

    print(f"Parsing PDF (PyMuPDF): {pdf_path}")
    print("-" * 60)

    parser = PyMuPDFAdvancedParser(
        chunk_size=1000,
        chunk_overlap=200,
        min_section_length=50,
        toc_assist=True,
    )

    try:
        chunks = parser.parse(pdf_path)
        parser.print_summary(chunks)

        out_json = str(pdf_path).replace(".pdf", "_pymupdf_chunks.json")
        parser.export_chunks_to_json(chunks, out_json)

        # Create LangChain Documents
        docs = parser.parse_to_documents(pdf_path)
        print(f"\nCreated {len(docs)} LangChain Document chunks.")
        print(docs)
        print("\n" + "=" * 60)
        print("SAMPLE CHUNKS (first 3)")
        print("=" * 60)
        for i, c in enumerate(chunks[:3], 1):
            print(f"\n--- Chunk {i} ---")
            print(f"Section: {c.section_title} (Level {c.section_level})")
            print(f"Page: {c.page_number}")
            print(f"Length: {len(c.text)} characters")
            print(f"Preview: {c.text[:200]}...")

    except Exception as e:
        print(f"Error parsing PDF: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
