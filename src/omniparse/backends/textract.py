from ..models import Page

def parse_image(path: str, region_name: str | None = None) -> list[Page]:
    import boto3
    client = boto3.client("textract", region_name=region_name)
    with open(path, "rb") as file:
        response = client.detect_document_text(Document={"Bytes": file.read()})
    lines = [block["Text"] for block in response.get("Blocks", []) if block.get("BlockType") == "LINE"]
    return [Page(number=1, text="\n".join(lines), engine="textract")]
