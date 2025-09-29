import sys

sys.path.append("..")

from wikipedia_async import WikipediaClient
import pytest


@pytest.mark.asyncio
async def test_wikipedia_page(client: WikipediaClient, data):
    response = await client.get_page(data["page"]["url"])
    assert response.url == data["page"]["url"]
    assert response.title == "Python (programming language)"
    assert data["page"]["content_snippet"] in response.content

    # Check Section Helper
    section = response.helper.get_section_by_title("History")
    assert section is not None

    assert section.title == "History"
    assert "Python was conceived in the late 1980s" in section.content

    table = response.helper.get_table_by_caption("Summary of Python 3's built-in types")
    assert table is not None
    assert "Type" in table.headers
    assert "int" in table["Type"]

    assert response.tables == response.helper.tables
    # print(response.tables)

    # Simple is better than complex. in paragraphs
    assert any(
        "Simple is better than complex." in para.paragraph_text
        for para in response.helper.paragraphs
    )
