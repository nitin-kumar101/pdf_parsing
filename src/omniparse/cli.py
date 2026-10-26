import json
from pathlib import Path
import typer
from .parser import DocumentParser
app = typer.Typer()

@app.command()
def parse(
    path: Path,
    mode: str = "auto",
    output: Path | None = None,
    format: str = "markdown",
    mineru_output: Path | None = None,
    mineru_tier: str = "flash",
):
    result = DocumentParser(mode=mode, mineru_tier=mineru_tier).parse(path, output_dir=mineru_output)
    if format not in {"markdown", "json"}:
        raise typer.BadParameter("format must be markdown or json")
    content = result.markdown if format == "markdown" else json.dumps(result.model_dump(), indent=2)
    if output:
        output.write_text(content, encoding="utf-8")
    else:
        typer.echo(content)