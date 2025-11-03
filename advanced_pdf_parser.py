"""
Advanced PDF Parser with Intelligent Section-Based Chunking
Uses multiple advanced libraries for robust PDF parsing and semantic chunking.
"""

import fitz  # PyMuPDF
import pdfplumber
from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import elements_to_json
import json


@dataclass
class PDFSection:
    """Represents a section in the PDF."""
    title: str
    content: str
    page_number: int
    level: int  # Heading level (1, 2, 3, etc.)
    start_char: int
    end_char: int


@dataclass
class PDFChunk:
    """Represents a chunk of text from the PDF."""
    text: str
    section_title: str
    section_level: int
    page_number: int
    chunk_index: int
    metadata: Dict


class AdvancedPDFParser:
    """
    Advanced PDF parser that intelligently identifies sections and creates chunks.
    Uses multiple parsing strategies for robustness.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_section_length: int = 50,
                 use_unstructured: bool = True):
        """
        Initialize the PDF parser.
        
        Args:
            chunk_size: Target size for text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            min_section_length: Minimum length for a section to be considered valid
            use_unstructured: Whether to use unstructured library for parsing
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_section_length = min_section_length
        self.use_unstructured = use_unstructured
        
    def detect_headings_with_fitz(self, doc: fitz.Document) -> List[Tuple[int, str, int, int]]:
        """
        Detect headings using PyMuPDF's text extraction and font analysis.
        Returns list of (page_num, heading_text, level, y_position).
        """
        headings = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get text blocks with font information
            blocks = page.get_text("dict")
            
            font_sizes = []
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span.get("text", "").strip():
                                font_sizes.append(span.get("size", 0))
            
            if not font_sizes:
                continue
                
            # Calculate font size statistics
            font_sizes = np.array(font_sizes)
            median_size = np.median(font_sizes)
            std_size = np.std(font_sizes) if len(font_sizes) > 1 else 0
            
            # Extract text with detailed information
            text_dict = page.get_text("rawdict")
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        text_parts = []
                        line_font_sizes = []
                        line_is_bold = []
                        
                        for span in line["spans"]:
                            text = span.get("text", "").strip()
                            if text:
                                text_parts.append(text)
                                line_font_sizes.append(span.get("size", 0))
                                line_is_bold.append(span.get("flags", 0) & 16 != 0)
                        
                        if not text_parts:
                            continue
                            
                        line_text = " ".join(text_parts)
                        avg_font_size = np.mean(line_font_sizes) if line_font_sizes else 0
                        is_bold = any(line_is_bold)
                        
                        # Heuristic: headings are usually larger, bold, or match heading patterns
                        is_likely_heading = (
                            avg_font_size > median_size + std_size * 0.5 or
                            (is_bold and avg_font_size >= median_size) or
                            self._is_heading_pattern(line_text)
                        )
                        
                        if is_likely_heading and len(line_text) < 200:  # Headings are typically short
                            # Determine heading level based on font size
                            if avg_font_size > median_size + std_size * 1.5:
                                level = 1
                            elif avg_font_size > median_size + std_size * 0.8:
                                level = 2
                            elif avg_font_size > median_size:
                                level = 3
                            else:
                                level = 4
                                
                            bbox = line.get("bbox", [0, 0, 0, 0])
                            y_position = bbox[1] if bbox else 0
                            
                            headings.append((page_num, line_text, level, y_position))
        
        return headings
    
    def _is_heading_pattern(self, text: str) -> bool:
        """Check if text matches common heading patterns."""
        # Patterns like "1. Title", "Section 2", "Chapter 3", etc.
        heading_patterns = [
            r'^\d+[\.\)]\s+[A-Z]',  # "1. Title" or "1) Title"
            r'^[A-Z][a-z]+(?:\s+\d+)?:\s+',  # "Chapter 1:", "Section 2:"
            r'^[IVX]+[\.\)]\s+[A-Z]',  # "I. Title", "IV) Title"
            r'^[A-Z][A-Z\s]{2,}$',  # ALL CAPS short lines
        ]
        return any(re.match(pattern, text.strip()) for pattern in heading_patterns)
    
    def extract_sections_with_fitz(self, doc: fitz.Document) -> List[PDFSection]:
        """Extract sections from PDF using PyMuPDF."""
        headings = self.detect_headings_with_fitz(doc)
        sections = []
        
        # Get full text for content extraction
        full_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            full_text.append((page_num, text))
        
        # Group headings by page and sort by position
        headings_by_page = {}
        for page_num, heading_text, level, y_pos in headings:
            if page_num not in headings_by_page:
                headings_by_page[page_num] = []
            headings_by_page[page_num].append((heading_text, level, y_pos))
        
        # Sort headings by page and y-position
        sorted_headings = []
        for page_num in sorted(headings_by_page.keys()):
            page_headings = headings_by_page[page_num]
            page_headings.sort(key=lambda x: x[2])  # Sort by y_position
            for heading_text, level, _ in page_headings:
                sorted_headings.append((page_num, heading_text, level))
        
        # Extract sections
        for i, (page_num, title, level) in enumerate(sorted_headings):
            # Find content until next heading
            start_page = page_num
            end_page = start_page + 5  # Limit search to next 5 pages
            
            # Find next heading at same or higher level
            next_heading_idx = i + 1
            while next_heading_idx < len(sorted_headings):
                next_page, next_title, next_level = sorted_headings[next_heading_idx]
                if next_level <= level:
                    end_page = next_page
                    break
                next_heading_idx += 1
            
            # Extract content
            content_parts = []
            for p in range(start_page, min(end_page + 1, len(doc))):
                if p < len(full_text):
                    page_text = full_text[p][1]
                    if p == start_page:
                        # Remove the heading from content
                        page_text = re.sub(re.escape(title), "", page_text, count=1)
                    content_parts.append(page_text)
            
            content = "\n\n".join(content_parts).strip()
            
            if len(content) >= self.min_section_length:
                sections.append(PDFSection(
                    title=title,
                    content=content,
                    page_number=page_num,
                    level=level,
                    start_char=0,
                    end_char=len(content)
                ))
        
        return sections
    
    def extract_sections_with_unstructured(self, pdf_path: str) -> List[PDFSection]:
        """
        Extract sections using the unstructured library with advanced chunking.
        This uses ML models for better section detection.
        """
        try:
            # Partition PDF into structured elements
            elements = partition_pdf(
                filename=pdf_path,
                strategy="hi_res",  # High resolution strategy for better accuracy
                infer_table_structure=True,
                extract_images_in_pdf=False,
            )
            
            # Chunk by title using unstructured's advanced chunking
            chunks = chunk_by_title(
                elements=elements,
                max_characters=self.chunk_size,
                combine_text_under_n_chars=self.chunk_size // 2,
                new_after_n_chars=self.chunk_size,
                overlap=self.chunk_overlap,
            )
            
            sections = []
            current_section_title = "Introduction"
            current_level = 1
            page_num = 1
            
            for chunk in chunks:
                text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                metadata = chunk.metadata.to_dict() if hasattr(chunk, 'metadata') else {}
                
                # Check if this chunk is a heading
                if hasattr(chunk, 'category') and chunk.category in ['Title', 'NarrativeText']:
                    if chunk.category == 'Title' and len(text) < 200:
                        current_section_title = text
                        current_level = metadata.get('level', 1)
                        continue
                
                # Get page number from metadata
                if 'page_number' in metadata:
                    page_num = metadata['page_number']
                
                if len(text.strip()) >= self.min_section_length:
                    sections.append(PDFSection(
                        title=current_section_title,
                        content=text,
                        page_number=page_num,
                        level=current_level,
                        start_char=0,
                        end_char=len(text)
                    ))
            
            return sections
            
        except Exception as e:
            print(f"Warning: Unstructured library failed: {e}. Falling back to PyMuPDF.")
            return []
    
    def extract_sections_with_pdfplumber(self, pdf_path: str) -> List[PDFSection]:
        """Extract sections using pdfplumber for detailed layout analysis."""
        sections = []
        
        with pdfplumber.open(pdf_path) as pdf:
            full_text_by_page = []
            
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text_by_page.append((page.page_number - 1, text))
            
            # Detect headings using text properties
            headings = []
            for i, (page_num, text) in enumerate(full_text_by_page):
                lines = text.split('\n')
                for line in lines:
                    line_stripped = line.strip()
                    if self._is_heading_pattern(line_stripped) and len(line_stripped) < 200:
                        # Estimate level based on line format
                        if re.match(r'^\d+[\.\)]', line_stripped):
                            level = int(re.match(r'^(\d+)', line_stripped).group(1)) if re.match(r'^(\d+)', line_stripped) else 1
                        else:
                            level = 1
                        headings.append((page_num, line_stripped, level))
            
            # Create sections from headings
            current_title = "Introduction"
            current_level = 1
            
            for i, (page_num, title, level) in enumerate(headings):
                # Get content until next heading
                content_parts = []
                start_idx = page_num
                
                if i < len(headings) - 1:
                    end_idx = headings[i + 1][0]
                else:
                    end_idx = len(full_text_by_page) - 1
                
                for p in range(start_idx, end_idx + 1):
                    if p < len(full_text_by_page):
                        page_text = full_text_by_page[p][1]
                        if p == start_idx:
                            page_text = re.sub(re.escape(title), "", page_text, count=1)
                        content_parts.append(page_text)
                
                content = "\n\n".join(content_parts).strip()
                
                if len(content) >= self.min_section_length:
                    sections.append(PDFSection(
                        title=title,
                        content=content,
                        page_number=page_num,
                        level=level,
                        start_char=0,
                        end_char=len(content)
                    ))
                    current_title = title
                    current_level = level
        
        return sections
    
    def create_chunks_from_sections(self, sections: List[PDFSection]) -> List[PDFChunk]:
        """
        Create intelligent chunks from sections with overlap and size management.
        """
        chunks = []
        
        for section in sections:
            # If section is small enough, create a single chunk
            if len(section.content) <= self.chunk_size:
                chunks.append(PDFChunk(
                    text=section.content,
                    section_title=section.title,
                    section_level=section.level,
                    page_number=section.page_number,
                    chunk_index=0,
                    metadata={
                        "section_title": section.title,
                        "section_level": section.level,
                        "chunk_type": "complete_section"
                    }
                ))
            else:
                # Split large sections into multiple chunks with overlap
                text = section.content
                start = 0
                chunk_idx = 0
                
                while start < len(text):
                    # Calculate end position
                    end = start + self.chunk_size
                    
                    # Try to break at sentence boundary
                    if end < len(text):
                        # Look for sentence endings near the chunk boundary
                        search_start = max(start, end - 100)
                        search_text = text[search_start:end + 100]
                        
                        # Find last sentence boundary
                        sentence_end = max(
                            search_text.rfind('. '),
                            search_text.rfind('.\n'),
                            search_text.rfind('! '),
                            search_text.rfind('? ')
                        )
                        
                        if sentence_end > 0:
                            end = search_start + sentence_end + 1
                    
                    chunk_text = text[start:end].strip()
                    
                    if len(chunk_text) >= self.min_section_length:
                        chunks.append(PDFChunk(
                            text=chunk_text,
                            section_title=section.title,
                            section_level=section.level,
                            page_number=section.page_number,
                            chunk_index=chunk_idx,
                            metadata={
                                "section_title": section.title,
                                "section_level": section.level,
                                "chunk_index": chunk_idx,
                                "chunk_type": "section_part",
                                "total_chunks_in_section": len(text) // self.chunk_size + 1
                            }
                        ))
                        chunk_idx += 1
                    
                    # Move start position with overlap
                    start = end - self.chunk_overlap
                    if start >= len(text):
                        break
        
        return chunks
    
    def parse(self, pdf_path: str) -> List[PDFChunk]:
        """
        Main parsing method that combines multiple strategies.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of PDFChunk objects
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        sections = []
        
        # Strategy 1: Try unstructured library (most advanced)
        if self.use_unstructured:
            try:
                sections = self.extract_sections_with_unstructured(str(pdf_path))
                if sections:
                    print(f"✓ Successfully parsed {len(sections)} sections using unstructured library")
                    return self.create_chunks_from_sections(sections)
            except Exception as e:
                print(f"Note: Unstructured library encountered issue: {e}")
        
        # Strategy 2: Use PyMuPDF (robust fallback)
        try:
            doc = fitz.open(str(pdf_path))
            sections = self.extract_sections_with_fitz(doc)
            doc.close()
            if sections:
                print(f"✓ Successfully parsed {len(sections)} sections using PyMuPDF")
        except Exception as e:
            print(f"Warning: PyMuPDF encountered issue: {e}")
        
        # Strategy 3: Fallback to pdfplumber
        if not sections:
            try:
                sections = self.extract_sections_with_pdfplumber(str(pdf_path))
                if sections:
                    print(f"✓ Successfully parsed {len(sections)} sections using pdfplumber")
            except Exception as e:
                print(f"Warning: pdfplumber encountered issue: {e}")
        
        # If no sections found, create chunks from full text
        if not sections:
            print("Note: No sections detected. Creating chunks from full text.")
            doc = fitz.open(str(pdf_path))
            full_text = "\n\n".join([page.get_text() for page in doc])
            doc.close()
            
            sections = [PDFSection(
                title="Full Document",
                content=full_text,
                page_number=1,
                level=1,
                start_char=0,
                end_char=len(full_text)
            )]
        
        return self.create_chunks_from_sections(sections)
    
    def export_chunks_to_json(self, chunks: List[PDFChunk], output_path: str):
        """Export chunks to JSON file."""
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                "text": chunk.text,
                "section_title": chunk.section_title,
                "section_level": chunk.section_level,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Exported {len(chunks)} chunks to {output_path}")
    
    def print_summary(self, chunks: List[PDFChunk]):
        """Print a summary of parsed chunks."""
        print("\n" + "="*60)
        print("PDF PARSING SUMMARY")
        print("="*60)
        print(f"Total chunks: {len(chunks)}")
        
        # Group by section
        sections = {}
        for chunk in chunks:
            title = chunk.section_title
            if title not in sections:
                sections[title] = []
            sections[title].append(chunk)
        
        print(f"Total sections: {len(sections)}")
        print("\nSections found:")
        for i, (title, section_chunks) in enumerate(sections.items(), 1):
            print(f"  {i}. {title} (Level {section_chunks[0].section_level}) - {len(section_chunks)} chunk(s)")
        
        print("\n" + "="*60)


def main():
    """Example usage of the AdvancedPDFParser."""
    import sys
    
    # Get PDF path from command line or use default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "sample.pdf"
    
    print(f"Parsing PDF: {pdf_path}")
    print("-" * 60)
    
    # Initialize parser
    parser = AdvancedPDFParser(
        chunk_size=1000,
        chunk_overlap=200,
        min_section_length=50,
        use_unstructured=True  # Set to False if unstructured library is not available
    )
    
    # Parse PDF
    try:
        chunks = parser.parse(pdf_path)
        
        # Print summary
        parser.print_summary(chunks)
        
        # Export to JSON
        output_json = pdf_path.replace('.pdf', '_chunks.json')
        parser.export_chunks_to_json(chunks, output_json)
        
        # Print first few chunks as examples
        print("\n" + "="*60)
        print("SAMPLE CHUNKS")
        print("="*60)
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n--- Chunk {i} ---")
            print(f"Section: {chunk.section_title} (Level {chunk.section_level})")
            print(f"Page: {chunk.page_number}")
            print(f"Length: {len(chunk.text)} characters")
            print(f"Preview: {chunk.text[:200]}...")
        
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

