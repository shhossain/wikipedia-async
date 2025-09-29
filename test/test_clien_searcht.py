import sys

sys.path.append("..")

from wikipedia_async import (
    WikipediaClient,
    SearchResult,
    SearchResults,
    ValidationError,
)
import pytest



@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["search"])
async def test_wikipedia_search(client: WikipediaClient, data, method: str):
    response = await getattr(client, method)(**data["search"]["args"])

    assert isinstance(response, SearchResults)
    assert all(isinstance(item, SearchResult) for item in response.results)
    assert len(response.results) <= data["search"]["args"]["limit"]
    if data["search"]["args"]["suggestion"]:
        assert isinstance(response.suggestion, str) or response.suggestion is None

    assert any(
        data["search"]["args"]["query"].lower() in item.title.lower()
        for item in response.results
    )


@pytest.mark.asyncio
async def test_wikipedia_search_no_results(client: WikipediaClient):
    response = await client.search("asdkfjaskldfjasdf", limit=5)
    assert isinstance(response, SearchResults)
    assert len(response.results) == 0
    assert response.suggestion is None


@pytest.mark.asyncio
async def test_wikipedia_search_empty_query(client: WikipediaClient):
    try:
        await client.search("", limit=5)
    except Exception as e:
        assert isinstance(e, ValidationError)
