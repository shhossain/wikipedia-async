"""
Modern Wikipedia API Client

A production-ready, async Wikipedia API client with advanced features.
"""

from wikipedia_async.client import WikipediaClient
from wikipedia_async.config import ClientConfig
from wikipedia_async.models.wiki_client_model import (
    SearchResult,
    WikiPage,
    PageSummary,
    Coordinates,
    Language,
    GeoSearchResult,
    SearchResults,
)
from wikipedia_async.exceptions import (
    WikipediaException,
    TimeoutError,
    NetworkError,
    RateLimitError,
    PageNotFoundError,
    DisambiguationError,
    RedirectError,
    APIError,
    ValidationError,
)

from wikipedia_async.helpers.section_helpers import (
    SectionHelper,
)
from wikipedia_async.models.section_models import Section, Table, Paragraph, Link

__version__ = "0.6.0"
__all__ = [
    "WikipediaClient",
    "ClientConfig",
    "SearchResult",
    "WikiPage",
    "PageSummary",
    "Coordinates",
    "Language",
    "GeoSearchResult",
    "SearchResults",
    "WikipediaException",
    "PageNotFoundError",
    "DisambiguationError",
    "RedirectError",
    "RateLimitError",
    "TimeoutError",
    "NetworkError",
    "APIError",
    "ValidationError",
    "Section",
    "SectionHelper",
    "Table",
    "Paragraph",
    "Link",
]
