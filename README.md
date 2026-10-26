# omniparse-docs (starter)

A lightweight, extensible document parsing starter with a public SDK, CLI and agent skill.

## Install locally

```bash
pip install -e .
# Optional engines
pip install -e '.[mineru]'
pip install -e '.[textract]'
```

## Python

```python
from omniparse import DocumentParser
result = DocumentParser(mode="auto").parse("document.pdf")
print(result.markdown)
print(result.chunks())
```

## CLI

```bash
omniparse parse document.pdf --mode fast --format json
```

## Implementation status

- Implemented: PyMuPDF PDF extraction, MinerU CLI adapter, single-image Textract OCR, Pydantic schema, CLI, basic character chunking, skill instructions.
- Planned: true per-page hybrid routing, Office adapters, asynchronous Textract PDF jobs, structured MinerU blocks and page provenance, table normalization, layout-aware chunking, retries, concurrency, benchmarks, MCP server and REST API.
- Auto mode uses native PDF extraction and optionally falls back to MinerU when most pages lack text; it is not yet page-level routing.
- MinerU adapter is CLI-based, expects `mineru-kit` installed, and returns one Markdown block; verify exact tier support against installed MinerU version.
- This is a starter scaffold, not a production-ready release or a published PyPI package.
