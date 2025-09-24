"""
Main Wikipedia client with async support, rate limiting, caching, and retry logic.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
import time
from decimal import Decimal

import aiohttp
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
)

from .config import ClientConfig
from .models import (
    SearchResult,
    WikiPage,
    PageSummary,
    Coordinates,
    Language,
    GeoSearchResult,
    BatchResult,
    RandomPageResult,
    SuggestionResult,
)
from .exceptions import (
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
from .cache import AsyncTTLCache, BaseCache, cache_key

logger = logging.getLogger(__name__)


class WikipediaClient:
    """
    Modern async Wikipedia API client with production-ready features.

    Features:
    - Async/await support with aiohttp
    - Automatic rate limiting with aiolimiter
    - Exponential backoff retry logic
    - TTL-based caching
    - Batch operations for efficiency
    - Full type safety with Pydantic
    - Comprehensive error handling
    """

    def __init__(
        self,
        config: Optional[ClientConfig] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        """
        Initialize the Wikipedia client.

        Args:
            config: Client configuration (uses defaults if None)
            session: Optional aiohttp session (creates new if None)
        """
        self.config = config or ClientConfig()
        self._session = session
        self._owned_session = session is None

        # Rate limiter
        self._rate_limiter = AsyncLimiter(
            self.config.rate_limit_calls, self.config.rate_limit_period
        )

        # Cache
        self._cache: Optional[BaseCache] = None
        if self.config.enable_cache:
            self._cache = AsyncTTLCache(
                cache_type=self.config.cache_type,
                cache_dir=self.config.cache_dir,
                max_size=self.config.max_cache_size,
                default_ttl=self.config.cache_ttl,
                serializer=self.config.cache_serializer,
            )

        # Semaphore for concurrent requests
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        # Session headers
        self._headers = {"User-Agent": self.config.user_agent, **self.config.headers}

    async def __aenter__(self) -> "WikipediaClient":
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session is initialized."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            connector = aiohttp.TCPConnector(limit=self.config.max_concurrent_requests)
            self._session = aiohttp.ClientSession(
                timeout=timeout, connector=connector, headers=self._headers
            )

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        if self._session and self._owned_session:
            await self._session.close()
            self._session = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=60),
        retry=retry_if_exception_type((NetworkError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _make_request(
        self, params: Dict[str, Any], use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make a request to the Wikipedia API with retries and rate limiting.

        Args:
            params: API parameters
            use_cache: Whether to use caching for this request

        Returns:
            API response data

        Raises:
            NetworkError: On network-related errors
            TimeoutError: On request timeout
            RateLimitError: On rate limit exceeded
            APIError: On API errors
        """
        await self._ensure_session()

        # Add standard parameters
        params.update({"action": "query", "format": "json", "formatversion": "2"})

        # Check cache first
        cache_key_str = (
            cache_key(lang=self.config.language, **params)
            if use_cache and self._cache
            else None
        )
        if cache_key_str and self._cache:
            cached_result = await self._cache.get(cache_key_str)
            if cached_result is not None:
                return cached_result

        # Rate limiting
        async with self._rate_limiter:
            async with self._semaphore:
                try:
                    async with self._session.get(
                        self.config.api_url, params=params
                    ) as response:

                        if response.status == 429:
                            retry_after = response.headers.get("Retry-After")
                            raise RateLimitError(
                                "Rate limit exceeded",
                                retry_after=float(retry_after) if retry_after else None,
                            )

                        if response.status >= 400:
                            raise NetworkError(
                                f"HTTP {response.status}: {response.reason}",
                                status_code=response.status,
                            )

                        data = await response.json()

                        # Check for API errors
                        if "error" in data:
                            error_info = data["error"]
                            raise APIError(
                                code=error_info.get("code", "unknown"),
                                message=error_info.get("info", "Unknown API error"),
                            )

                        # Cache successful results
                        if cache_key_str and self._cache and use_cache:
                            await self._cache.set(cache_key_str, data)

                        return data

                except asyncio.TimeoutError as e:
                    raise TimeoutError("Request timed out") from e
                except aiohttp.ClientError as e:
                    raise NetworkError(f"Network error: {str(e)}") from e

    async def search(
        self, query: str, limit: int = 10, suggestion: bool = False, namespace: int = 0
    ) -> Union[List[SearchResult], SuggestionResult]:
        """
        Search for Wikipedia articles.

        Args:
            query: Search query
            limit: Maximum number of results (1-500)
            suggestion: Return search suggestions
            namespace: Namespace to search (0 = articles)

        Returns:
            List of search results or suggestion result

        Raises:
            ValidationError: On invalid parameters
            WikipediaException: On API errors
        """
        if not query.strip():
            raise ValidationError("query", "Query cannot be empty")

        if not 1 <= limit <= 500:
            raise ValidationError("limit", "Limit must be between 1 and 500")

        params = {
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "srnamespace": namespace,
            "srprop": "snippet|titlesnippet|size|wordcount|timestamp",
        }

        if suggestion:
            params["srinfo"] = "suggestion"

        try:
            response = await self._make_request(params)
            search_data = response["query"]["search"]

            results = [
                SearchResult(
                    title=item["title"],
                    snippet=item.get("snippet", ""),
                    page_id=item.get("pageid"),
                    word_count=item.get("wordcount"),
                    size=item.get("size"),
                    timestamp=item.get("timestamp"),
                )
                for item in search_data
            ]

            if suggestion:
                suggestion_text = (
                    response["query"].get("searchinfo", {}).get("suggestion")
                )
                return SuggestionResult(
                    query=query, suggestion=suggestion_text, results=results
                )

            return results

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}")
            raise

    async def get_page(
        self,
        title: Optional[str] = None,
        page_id: Optional[int] = None,
        include_content: bool = True,
        include_images: bool = False,
        include_references: bool = False,
        include_links: bool = False,
        include_categories: bool = False,
        include_coordinates: bool = False,
    ) -> WikiPage:
        """
        Get a Wikipedia page with optional content.

        Args:
            title: Page title (mutually exclusive with page_id)
            page_id: Page ID (mutually exclusive with title)
            include_content: Include full page content
            include_images: Include page images
            include_references: Include external references
            include_links: Include internal links
            include_categories: Include page categories
            include_coordinates: Include geographic coordinates

        Returns:
            WikiPage object with requested data

        Raises:
            ValidationError: On invalid parameters
            PageNotFoundError: If page doesn't exist
            DisambiguationError: If title is ambiguous
            RedirectError: If page redirects and redirects disabled
        """
        if not title and not page_id:
            raise ValidationError(
                "title/page_id", "Either title or page_id must be provided"
            )

        if title and page_id:
            raise ValidationError(
                "title/page_id", "Cannot specify both title and page_id"
            )

        # Build properties list
        props = ["info", "pageprops"]
        if include_content:
            props.append("extracts")
        if include_images:
            props.append("images")
        if include_references:
            props.append("extlinks")
        if include_links:
            props.append("links")
        if include_categories:
            props.append("categories")
        if include_coordinates:
            props.append("coordinates")

        params = {
            "prop": "|".join(props),
            "inprop": "url",
            "explaintext": "" if include_content else None,
            "exintro": "" if include_content else None,
        }

        if title:
            params["titles"] = title
        else:
            params["pageids"] = page_id

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        try:
            response = await self._make_request(params)
            pages = response["query"]["pages"]

            if not pages:
                raise PageNotFoundError(title or str(page_id))

            page_data = pages[0]

            # Check for missing page
            if "missing" in page_data:
                raise PageNotFoundError(title or str(page_id))

            # Check for disambiguation
            if "pageprops" in page_data and "disambiguation" in page_data["pageprops"]:
                # Get disambiguation options
                options_response = await self._get_disambiguation_options(
                    page_data["title"]
                )
                raise DisambiguationError(page_data["title"], options_response)

            # Build WikiPage object
            wiki_page = WikiPage(
                title=page_data["title"],
                page_id=page_data["pageid"],
                url=page_data["fullurl"],
                extract=page_data.get("extract"),
                namespace=page_data.get("ns", 0),
            )

            # Add optional data
            if include_images and "images" in page_data:
                wiki_page.images = [img["title"] for img in page_data["images"]]

            if include_references and "extlinks" in page_data:
                wiki_page.references = [link["*"] for link in page_data["extlinks"]]

            if include_links and "links" in page_data:
                wiki_page.links = [link["title"] for link in page_data["links"]]

            if include_categories and "categories" in page_data:
                wiki_page.categories = [
                    cat["title"].replace("Category:", "")
                    for cat in page_data["categories"]
                ]

            if include_coordinates and "coordinates" in page_data:
                coords = page_data["coordinates"][0]
                wiki_page.coordinates = Coordinates(
                    latitude=Decimal(str(coords["lat"])),
                    longitude=Decimal(str(coords["lon"])),
                )

            return wiki_page

        except (PageNotFoundError, DisambiguationError, RedirectError):
            raise
        except Exception as e:
            logger.error(f"Failed to get page '{title or page_id}': {str(e)}")
            raise WikipediaException(f"Failed to get page: {str(e)}")

    async def get_pages_batch(
        self, titles: List[str], include_content: bool = True
    ) -> BatchResult:
        """
        Get multiple pages in batches for efficiency.

        Args:
            titles: List of page titles
            include_content: Include page content

        Returns:
            BatchResult with successful and failed pages
        """
        if not titles:
            return BatchResult(total_requested=0)

        if len(titles) > self.config.max_batch_size:
            # Process in chunks
            results = []
            for i in range(0, len(titles), self.config.max_batch_size):
                chunk = titles[i : i + self.config.max_batch_size]
                chunk_result = await self._get_pages_chunk(chunk, include_content)
                results.append(chunk_result)

            # Combine results
            successful = []
            failed = []
            for result in results:
                successful.extend(result.successful)
                failed.extend(result.failed)

            return BatchResult(
                successful=successful, failed=failed, total_requested=len(titles)
            )

        return await self._get_pages_chunk(titles, include_content)

    async def _get_pages_chunk(
        self, titles: List[str], include_content: bool
    ) -> BatchResult:
        """Get a chunk of pages (internal method)."""
        props = ["info", "pageprops"]
        if include_content:
            props.append("extracts")

        params = {
            "prop": "|".join(props),
            "titles": "|".join(titles),
            "inprop": "url",
            "explaintext": "" if include_content else None,
            "exintro": "" if include_content else None,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        try:
            response = await self._make_request(params)
            pages = response["query"]["pages"]

            successful = []
            failed = []

            for page_data in pages:
                try:
                    if "missing" in page_data:
                        failed.append(
                            {
                                "title": page_data.get("title", "unknown"),
                                "error": "Page not found",
                            }
                        )
                        continue

                    wiki_page = WikiPage(
                        title=page_data["title"],
                        page_id=page_data["pageid"],
                        url=page_data["fullurl"],
                        extract=page_data.get("extract"),
                        namespace=page_data.get("ns", 0),
                    )
                    successful.append(wiki_page)

                except Exception as e:
                    failed.append(
                        {"title": page_data.get("title", "unknown"), "error": str(e)}
                    )

            return BatchResult(
                successful=successful, failed=failed, total_requested=len(titles)
            )

        except Exception as e:
            # All pages failed
            failed = [{"title": title, "error": str(e)} for title in titles]
            return BatchResult(
                successful=[], failed=failed, total_requested=len(titles)
            )

    async def get_summary(
        self, title: str, sentences: int = 0, characters: int = 0
    ) -> PageSummary:
        """
        Get a page summary.

        Args:
            title: Page title
            sentences: Number of sentences (0 = auto)
            characters: Number of characters (0 = auto)

        Returns:
            PageSummary object
        """
        params = {
            "prop": "extracts|info",
            "titles": title,
            "inprop": "url",
            "explaintext": "",
            "exintro": "",
        }

        if sentences > 0:
            params["exsentences"] = sentences
        elif characters > 0:
            params["exchars"] = characters

        response = await self._make_request(params)
        pages = response["query"]["pages"]

        if not pages or "missing" in pages[0]:
            raise PageNotFoundError(title)

        page_data = pages[0]

        return PageSummary(
            title=page_data["title"],
            page_id=page_data["pageid"],
            extract=page_data.get("extract", ""),
            url=page_data["fullurl"],
        )

    async def geosearch(
        self, latitude: float, longitude: float, radius: int = 1000, limit: int = 10
    ) -> List[GeoSearchResult]:
        """
        Search for pages by geographic coordinates.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            radius: Search radius in meters (10-10000)
            limit: Maximum results (1-500)

        Returns:
            List of geographic search results
        """
        if not -90 <= latitude <= 90:
            raise ValidationError("latitude", "Latitude must be between -90 and 90")

        if not -180 <= longitude <= 180:
            raise ValidationError("longitude", "Longitude must be between -180 and 180")

        if not 10 <= radius <= 10000:
            raise ValidationError(
                "radius", "Radius must be between 10 and 10000 meters"
            )

        if not 1 <= limit <= 500:
            raise ValidationError("limit", "Limit must be between 1 and 500")

        params = {
            "list": "geosearch",
            "gscoord": f"{latitude}|{longitude}",
            "gsradius": radius,
            "gslimit": limit,
        }

        response = await self._make_request(params)
        results = response["query"]["geosearch"]

        return [
            GeoSearchResult(
                title=item["title"],
                page_id=item["pageid"],
                coordinates=Coordinates(
                    latitude=Decimal(str(item["lat"])),
                    longitude=Decimal(str(item["lon"])),
                ),
                distance=item.get("dist"),
            )
            for item in results
        ]

    async def random(self, count: int = 1, namespace: int = 0) -> RandomPageResult:
        """
        Get random page titles.

        Args:
            count: Number of random pages (1-10)
            namespace: Namespace (0 = articles)

        Returns:
            RandomPageResult with page titles
        """
        if not 1 <= count <= 10:
            raise ValidationError("count", "Count must be between 1 and 10")

        params = {"list": "random", "rnnamespace": namespace, "rnlimit": count}

        response = await self._make_request(params)
        random_pages = response["query"]["random"]

        return RandomPageResult(
            pages=[page["title"] for page in random_pages], namespace=namespace
        )

    async def suggest(self, query: str) -> Optional[str]:
        """
        Get search suggestion for a query.

        Args:
            query: Search query

        Returns:
            Suggested search term or None
        """
        params = {
            "list": "search",
            "srsearch": query,
            "srinfo": "suggestion",
            "srprop": "",
        }

        response = await self._make_request(params)
        search_info = response["query"].get("searchinfo", {})

        return search_info.get("suggestion")

    async def get_languages(self) -> List[Language]:
        """
        Get list of available Wikipedia languages.

        Returns:
            List of Language objects
        """
        params = {"meta": "siteinfo", "siprop": "languages"}

        response = await self._make_request(params)
        languages = response["query"]["languages"]

        return [Language(code=lang["code"], name=lang["*"]) for lang in languages]

    async def set_language(self, language: str) -> None:
        """
        Change the client language.

        Args:
            language: Language code (e.g., 'en', 'fr', 'de')
        """
        self.config = self.config.with_language(language)

    async def _get_disambiguation_options(self, title: str) -> List[str]:
        """Get disambiguation options for a title."""
        params = {
            "prop": "revisions",
            "titles": title,
            "rvprop": "content",
            "rvparse": "",
            "rvlimit": 1,
        }

        try:
            response = await self._make_request(params)
            pages = response["query"]["pages"]

            if not pages:
                return []

            # Parse disambiguation page for options
            # This is a simplified implementation
            return []  # Would need HTML parsing for full implementation

        except Exception:
            return []

    async def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._cache:
            return {"cache_enabled": False}

        stats = await self._cache.stats()
        return {"cache_enabled": True, **stats}

    async def clear_cache(self) -> None:
        """Clear the cache."""
        if self._cache:
            await self._cache.clear()
