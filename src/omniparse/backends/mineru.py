import base64
import hashlib
import json
import logging
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..models import Page
from ..timing import log_step

logger = logging.getLogger("omniparse.mineru")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
HTML_IMAGE_RE = re.compile(
    r"(<img\b[^>]*\bsrc\s*=\s*)([\"'])([^\"']+)\2",
    re.IGNORECASE,
)
DATA_URI_RE = re.compile(
    r"^data:image/([A-Za-z0-9.+-]+)(?:;charset=[^;]+)?;base64,(.+)$",
    re.IGNORECASE | re.DOTALL,
)
JSON_IMAGE_PATH_KEYS = {"image_path", "img_path", "image_url"}
JSON_IMAGE_DATA_KEYS = {"image_base64"}
MIME_TO_EXT = {
    "jpeg": ".jpg",
    "jpg": ".jpg",
    "png": ".png",
    "gif": ".gif",
    "webp": ".webp",
    "bmp": ".bmp",
    "tiff": ".tiff",
    "tif": ".tif",
    "svg+xml": ".svg",
    "svg": ".svg",
}


class ImageStore:
    def __init__(self, images_dir: Path):
        self.images_dir = images_dir
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._next_index = 1
        self._id_by_hash: dict[str, str] = {}
        self.ref_to_id: dict[str, str] = {}
        self.records: list[dict[str, str]] = []

    def ingest_bytes(self, data: bytes, ext: str, original_ref: str) -> str:
        digest = hashlib.sha256(data).hexdigest()
        if digest in self._id_by_hash:
            image_id = self._id_by_hash[digest]
            self.ref_to_id[original_ref] = image_id
            return image_id

        image_id = f"img_{self._next_index:04d}"
        self._next_index += 1
        suffix = _normalize_ext(ext)
        filename = f"{image_id}{suffix}"
        dest = self.images_dir / filename
        dest.write_bytes(data)

        self._id_by_hash[digest] = image_id
        self.ref_to_id[original_ref] = image_id
        self.records.append(
            {
                "image_id": image_id,
                "path": f"images/{filename}",
                "original_ref": original_ref,
            }
        )
        return image_id

    def ingest_file(self, path: Path, original_ref: str | None = None) -> str | None:
        if not path.is_file():
            return None
        ref = original_ref or path.name
        return self.ingest_bytes(path.read_bytes(), path.suffix, ref)

    def ingest_reference(self, reference: str, search_roots: list[Path]) -> str | None:
        ref = reference.strip()
        if not ref:
            return None
        if ref in self.ref_to_id:
            return self.ref_to_id[ref]
        if _is_remote_url(ref):
            return None

        data_uri = DATA_URI_RE.match(ref)
        if data_uri:
            ext = MIME_TO_EXT.get(data_uri.group(1).lower(), ".png")
            try:
                payload = base64.b64decode(data_uri.group(2), validate=False)
            except Exception:
                return None
            return self.ingest_bytes(payload, ext, ref)

        local_path = _resolve_image_path(ref, search_roots)
        if local_path is None:
            return None
        return self.ingest_file(local_path, ref)


SUPPORTED_TIERS = {"basic", "flash", "standard"}


def parse_pdf(path: str, tier: str = "basic", output_dir: str | Path | None = None):
    if tier not in SUPPORTED_TIERS:
        raise ValueError(f"Unsupported MinerU tier: {tier}")

    pdf_path = Path(path).resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    timings: dict[str, float] = {}
    destination = _resolve_output_dir(pdf_path, tier, output_dir)
    rewritten_markdown = ""
    records: list[dict[str, str]] = []
    logger.info("MinerU parse started path=%s tier=%s output=%s", pdf_path, tier, destination)

    with log_step("mineru_parse_pdf", timings, path=pdf_path, tier=tier):
        with log_step("prepare_output_dir", timings, output=destination):
            destination.mkdir(parents=True, exist_ok=True)
            images_dir = destination / "images"
            if images_dir.exists():
                shutil.rmtree(images_dir)
            images_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            with log_step("mineru_kit_parse", timings, tier=tier):
                _run_mineru(pdf_path, scratch, tier)
            with log_step("extract_mineru_zip", timings):
                extract_root = _materialize_mineru_output(scratch)

            with log_step("read_mineru_artifacts", timings):
                markdown = _read_first_text(extract_root, "*.md")
                if markdown is None:
                    raise RuntimeError("MinerU completed but no Markdown output was found.")
                raw_json = _read_preferred_json(extract_root)
                logger.info(
                    "Loaded MinerU artifacts markdown_chars=%s has_json=%s",
                    len(markdown),
                    raw_json is not None,
                )

            store = ImageStore(images_dir)
            with log_step("extract_images", timings):
                _ingest_extracted_image_files(extract_root, store)
                logger.info("Ingested %s unique image(s)", len(store.records))

            with log_step("rewrite_image_ids", timings):
                rewritten_markdown = rewrite_markup(markdown or "", store, [extract_root])
                rewritten_json = rewrite_json_images(raw_json, store, [extract_root]) if raw_json is not None else None
                if rewritten_json is None:
                    rewritten_json = {
                        "source": str(pdf_path),
                        "engine": f"mineru-{tier}",
                        "markdown": rewritten_markdown,
                        "images": store.records,
                    }
                elif isinstance(rewritten_json, dict):
                    rewritten_json = {
                        **rewritten_json,
                        "images": store.records,
                    }

            with log_step("save_markdown_json", timings, output=destination):
                markdown_path = destination / f"{pdf_path.stem}.md"
                json_path = destination / f"{pdf_path.stem}.json"
                manifest_path = destination / "images.json"
                markdown_path.write_text(rewritten_markdown, encoding="utf-8")
                json_path.write_text(
                    json.dumps(rewritten_json, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                manifest_path.write_text(
                    json.dumps(store.records, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info(
                    "Wrote %s, %s, and %s",
                    markdown_path.name,
                    json_path.name,
                    manifest_path.name,
                )
            records = store.records

    logger.info("MinerU parse timings: %s", timings)
    return [
        Page(
            number=1,
            text=rewritten_markdown,
            engine=f"mineru-{tier}",
        )
    ], records, destination, timings


def rewrite_markup(text: str, store: ImageStore, search_roots: list[Path]) -> str:
    def replace_markdown(match: re.Match[str]) -> str:
        alt, src = match.group(1), match.group(2).strip()
        image_id = store.ingest_reference(src, search_roots)
        if image_id is None:
            return match.group(0)
        return f"![{alt}]({image_id})"

    def replace_html(match: re.Match[str]) -> str:
        prefix, quote, src = match.group(1), match.group(2), match.group(3).strip()
        image_id = store.ingest_reference(unquote(src), search_roots)
        if image_id is None:
            return match.group(0)
        return f"{prefix}{quote}{image_id}{quote}"

    updated = MARKDOWN_IMAGE_RE.sub(replace_markdown, text)
    return HTML_IMAGE_RE.sub(replace_html, updated)


def rewrite_json_images(value: Any, store: ImageStore, search_roots: list[Path]) -> Any:
    if isinstance(value, dict):
        rewritten: dict[str, Any] = {}
        image_id: str | None = None
        for key, item in value.items():
            if key in JSON_IMAGE_PATH_KEYS and isinstance(item, str) and item:
                found = store.ingest_reference(item, search_roots)
                if found:
                    image_id = found
                else:
                    rewritten[key] = item
                continue
            if key in JSON_IMAGE_DATA_KEYS and isinstance(item, str) and item:
                ref = item if item.startswith("data:") else f"data:image/png;base64,{item}"
                found = store.ingest_reference(ref, search_roots)
                if found:
                    image_id = found
                else:
                    rewritten[key] = item
                continue
            if key in {"content", "table_body", "html", "text"} and isinstance(item, str):
                rewritten[key] = rewrite_markup(item, store, search_roots)
            else:
                rewritten[key] = rewrite_json_images(item, store, search_roots)
        if image_id:
            rewritten["image_id"] = image_id
        return rewritten
    if isinstance(value, list):
        return [rewrite_json_images(item, store, search_roots) for item in value]
    return value


def _run_mineru(pdf_path: Path, output_dir: Path, tier: str) -> None:
    command = [
        "mineru-kit",
        "parse",
        str(pdf_path),
        "-o",
        str(output_dir),
        "--tier",
        tier,
        "--format",
        "zip",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            timeout=1800,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise RuntimeError("MinerU executable 'mineru-kit' was not found.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"MinerU failed:\n{exc.stderr}") from exc

    if not list(output_dir.rglob("*")):
        raise RuntimeError(
            "MinerU completed but produced no output.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


def _materialize_mineru_output(scratch: Path) -> Path:
    zip_files = list(scratch.rglob("*.zip"))
    if zip_files:
        extract_root = scratch / "extracted"
        extract_root.mkdir(parents=True, exist_ok=True)
        _safe_extract_zip(zip_files[0], extract_root)
        return extract_root
    return scratch


def _safe_extract_zip(zip_path: Path, dest: Path) -> None:
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            target = (dest / info.filename).resolve()
            if dest not in target.parents and target != dest:
                raise RuntimeError(f"Unsafe zip entry skipped: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)


def _ingest_extracted_image_files(extract_root: Path, store: ImageStore) -> None:
    for image_path in sorted(extract_root.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS or not image_path.is_file():
            continue
        relative = image_path.relative_to(extract_root).as_posix()
        store.ingest_file(image_path, relative)
        store.ingest_file(image_path, image_path.name)


def _read_first_text(root: Path, pattern: str) -> str | None:
    matches = sorted(root.rglob(pattern))
    if not matches:
        return None
    return matches[0].read_text(encoding="utf-8")


def _read_preferred_json(root: Path) -> Any | None:
    preferred_names = ("middle_json.json", "structured_content.json", "model_output.json")
    for name in preferred_names:
        matches = sorted(path for path in root.rglob(name) if path.is_file())
        if matches:
            return json.loads(matches[0].read_text(encoding="utf-8"))
    matches = sorted(
        path
        for path in root.rglob("*.json")
        if path.is_file()
    )
    if not matches:
        return None
    return json.loads(matches[0].read_text(encoding="utf-8"))


def _resolve_output_dir(pdf_path: Path, tier: str, output_dir: str | Path | None) -> Path:
    folder_name = f"{pdf_path.stem}_{tier}"
    if output_dir is None:
        return pdf_path.parent / folder_name
    return Path(output_dir).expanduser().resolve() / folder_name


def _resolve_image_path(reference: str, search_roots: list[Path]) -> Path | None:
    cleaned = unquote(reference.split("?", 1)[0].split("#", 1)[0].strip().replace("\\", "/"))
    raw = Path(cleaned)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        for root in search_roots:
            candidates.append(root / cleaned)
            candidates.append(root / Path(cleaned).name)
            if not cleaned.startswith("images/"):
                candidates.append(root / "images" / Path(cleaned).name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _normalize_ext(ext: str) -> str:
    suffix = ext.lower()
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    if suffix == ".jpeg":
        return ".jpg"
    if suffix in IMAGE_EXTENSIONS or suffix == ".svg":
        return suffix
    return ".png"


def _is_remote_url(value: str) -> bool:
    scheme = urlparse(value).scheme.lower()
    return scheme in {"http", "https"}