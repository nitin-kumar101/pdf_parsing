from pydantic import BaseModel, Field

class Page(BaseModel):
    number: int
    text: str
    engine: str

class ParseResult(BaseModel):
    source: str
    pages: list[Page] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    tables: list[dict] = Field(default_factory=list)
    images: list[dict] = Field(default_factory=list)
    @property
    def markdown(self) -> str:
        return "\n\n".join(p.text for p in self.pages)
    @property
    def text(self) -> str:
        return self.markdown
    def chunks(self, max_chars: int = 1500) -> list[dict]:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        result = []
        for page in self.pages:
            text = page.text
            for start in range(0, len(text), max_chars):
                result.append({"text": text[start:start+max_chars], "page": page.number, "engine": page.engine})
        return result
