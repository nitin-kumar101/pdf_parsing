import logging
from pathlib import Path

from .backends.pymupdf import parse_pdf
from .models import ParseResult
from .timing import configure_logging, log_step

logger = logging.getLogger("omniparse")


class DocumentParser:
    def __init__(self, mode: str = "auto", *, mineru_tier: str = "basic", aws_region: str | None = None):
        if mode not in {"auto", "fast", "mineru", "textract"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.mineru_tier = mineru_tier
        self.aws_region = aws_region

    def parse(self, path: str | Path, output_dir: str | Path | None = None) -> ParseResult:
        configure_logging()
        path = Path(path).expanduser().resolve(strict=True)
        suffix = path.suffix.lower()
        timings: dict[str, float] = {}
        with log_step("parse_document", timings, path=path, mode=self.mode):
            result = self._parse(path, suffix, output_dir, timings)
        result.metadata["timings"] = {**result.metadata.get("timings", {}), **timings}
        logger.info("Parse timings: %s", result.metadata["timings"])
        return result

    def _parse(
        self,
        path: Path,
        suffix: str,
        output_dir: str | Path | None,
        timings: dict[str, float],
    ) -> ParseResult:
        if suffix == ".pdf":
            if self.mode == "textract":
                raise ValueError("Textract PDF support requires an asynchronous S3 workflow; use image input or another mode")
            images: list[dict] = []
            mineru_output = None
            mineru_timings: dict[str, float] = {}
            if self.mode == "mineru":
                from .backends.mineru import parse_pdf as parse_mineru
                pages, images, mineru_output, mineru_timings = parse_mineru(str(path), self.mineru_tier, output_dir)
            else:
                with log_step("pymupdf_parse", timings, path=path):
                    pages = parse_pdf(str(path))
                if self.mode == "auto" and pages and sum(bool(p.text.strip()) for p in pages) / len(pages) < 0.7:
                    try:
                        from .backends.mineru import parse_pdf as parse_mineru
                        pages, images, mineru_output, mineru_timings = parse_mineru(str(path), self.mineru_tier, output_dir)
                    except (FileNotFoundError, ImportError):
                        pass  # Preserve available native text; expose OCR needs in metadata
            metadata = {
                "pages": len(pages),
                "ocr_may_be_needed": any(not p.text.strip() for p in pages),
                "timings": mineru_timings,
            }
            if mineru_output is not None:
                metadata["mineru_output_dir"] = str(mineru_output)
            return ParseResult(source=str(path), pages=pages, metadata=metadata, images=images)
        if suffix in {".png", ".jpg", ".jpeg"} and self.mode == "textract":
            from .backends.textract import parse_image
            with log_step("textract_parse", timings, path=path):
                pages = parse_image(str(path), self.aws_region)
            return ParseResult(source=str(path), pages=pages, metadata={"timings": {}})
        raise ValueError(f"Unsupported format/mode: {suffix}/{self.mode}. Office adapters are planned, not implemented in this starter.")
