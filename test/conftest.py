import sys
from typing import TypedDict
import pytest_asyncio
import aiohttp

sys.path.append("..")

from wikipedia_async import (
    WikipediaClient,
)
import pytest


@pytest.fixture
def data():
    return {
        "search": {
            "args": {
                "query": "Python",
                "limit": 10,
                "suggestion": True,
                "namespace": 0,
                "lang": "en",
            },
        },
        "page": {
            "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "content_snippet": "Python is a high-level, general-purpose programming language.",
        },
    }


@pytest_asyncio.fixture
async def client():
    client = WikipediaClient()
    yield client
    await client.close()
