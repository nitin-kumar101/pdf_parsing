import base64
from pathlib import Path

from omniparse.backends.mineru import ImageStore, rewrite_json_images, rewrite_markup

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
)


def test_rewrite_markdown_and_json_replace_images_with_ids(tmp_path: Path):
    extract_root = tmp_path / "extract"
    images_src = extract_root / "images"
    images_src.mkdir(parents=True)
    source_image = images_src / "figure-1.png"
    source_image.write_bytes(PNG_BYTES)

    store = ImageStore(tmp_path / "out" / "images")
    store.ingest_file(source_image, "images/figure-1.png")

    markdown = "Intro\n\n![chart](images/figure-1.png)\n"
    rewritten_md = rewrite_markup(markdown, store, [extract_root])
    assert rewritten_md == "Intro\n\n![chart](img_0001)\n"

    payload = {
        "pdf_info": [
            {
                "type": "image",
                "img_path": "images/figure-1.png",
                "content": '<img src="images/figure-1.png">',
            }
        ]
    }
    rewritten_json = rewrite_json_images(payload, store, [extract_root])
    assert rewritten_json["pdf_info"][0]["image_id"] == "img_0001"
    assert "img_path" not in rewritten_json["pdf_info"][0]
    assert rewritten_json["pdf_info"][0]["content"] == '<img src="img_0001">'
    assert (tmp_path / "out" / "images" / "img_0001.png").is_file()


def test_data_uri_is_extracted_to_image_id(tmp_path: Path):
    store = ImageStore(tmp_path / "images")
    data_uri = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
    rewritten = rewrite_markup(f"![]({data_uri})", store, [tmp_path])
    assert rewritten == "![](img_0001)"
    assert store.records[0]["image_id"] == "img_0001"
