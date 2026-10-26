from omniparse.models import Page, ParseResult

def test_chunks_keep_page():
    result = ParseResult(source="sample.pdf", pages=[Page(number=2, text="abcdef", engine="test")])
    assert result.chunks(3) == [{"text": "abc", "page": 2, "engine": "test"}, {"text": "def", "page": 2, "engine": "test"}]
