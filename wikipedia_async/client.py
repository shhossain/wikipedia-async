"""
Main Wikipedia client with async support, rate limiting, caching, and retry logic.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
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
import traceback

from wikipedia_async.helpers.section_helpers import SectionHelper

from wikipedia_async.config import ClientConfig
from wikipedia_async.models.wiki_client_model import (
    BatchHTMLResult,
    HTMLResult,
    SearchResult,
    WikiPage,
    PageSummary,
    Coordinates,
    Language,
    GeoSearchResult,
    BatchResult,
    RandomPageResult,
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
from wikipedia_async.cache import AsyncTTLCache, BaseCache, cache_key
from urllib.parse import unquote_plus
from wikipedia_async.helpers.logger_helpers import logger

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
        self._session: aiohttp.ClientSession = session  # type: ignore
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
            self._session = None  # type: ignore

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
        lang = params.pop("lang", self.config.language)
        api_url = self.config.api_base_url.format(lang=lang)
        # Check cache first
        cache_key_str = (
            cache_key(lang=lang, **params) if use_cache and self._cache else None
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
                        api_url,
                        params=params,
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

    def _title_to_url(self, title: str) -> str:
        """Convert a page title to its full Wikipedia URL."""
        base_url = f"https://{self.config.language}.wikipedia.org/wiki/"
        return base_url + title.replace(" ", "_")

    def _normalize_title(self, title: str) -> str:
        # title can be url
        res = title.strip().replace("_", " ")
        if title.startswith("http://") or title.startswith("https://"):
            parts = title.split("/wiki/")
            if len(parts) == 2:
                res = parts[1].replace("_", " ")
                url = f"{self.config.language}.wikipedia.org"
                if url not in title:
                    lang = parts[0].split("//")[1].split(".")[0]
                    if self.config.auto_detect_language:
                        logger.once(
                            logging.WARNING,
                            f"Auto changing language to {lang}. If not desired, than set `auto_detect_language` to False in config",
                        )
                        self.config = self.config.with_language(lang)
                    else:
                        raise ValidationError(
                            "title",
                            f"URL language '{lang}' does not match client language '{self.config.language}'",
                        )
            else:
                raise ValidationError("title", "Invalid Wikipedia URL")

        if not res:
            raise ValidationError("title", "Title cannot be empty")
        return unquote_plus(res)

    async def search(
        self,
        query: str,
        limit: int = 10,
        suggestion: bool = False,
        namespace: int = 0,
        lang: Optional[str] = None,
    ) -> SearchResults:
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
        if lang:
            params["lang"] = lang

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
                    url=self._title_to_url(item["title"]),
                )
                for item in search_data
            ]
            suggestion_text = None
            if suggestion:
                suggestion_text = (
                    response["query"].get("searchinfo", {}).get("suggestion")
                )

            return SearchResults(results=results, suggestion=suggestion_text)

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
        include_tables: bool | str = "auto",
        include_html: bool = False,
        lang: Optional[str] = None,
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
            include_tables: Include extracted tables (True, False, or 'auto')
            include_html: Include raw HTML content

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
        }

        if title:
            params["titles"] = self._normalize_title(title)
        else:
            params["pageids"] = page_id

        if lang:
            params["lang"] = lang

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
            helper = SectionHelper.from_content(page_data.get("extract", ""))
            wiki_page = WikiPage(
                title=page_data["title"],
                page_id=page_data["pageid"],
                url=page_data["fullurl"],
                extract=page_data.get("extract"),
                namespace=page_data.get("ns", 0),
                summary=helper.first_content(),
                sections=helper.sections,
                content=page_data.get("extract", ""),
                helper=helper,
                html_content=None,
                revision_id=None,
                coordinates=None,
                last_modified=None,
                parent_id=None,
            )
            html = ""
            should_fetch_html = False
            if include_html:
                should_fetch_html = True

            should_fetch_tables = False
            if include_tables == "auto":
                for sec in helper.sections:
                    name = sec.title.lower().strip()
                    content = sec.content
                    if not content.strip():
                        should_fetch_tables = True
                        break
                    if (
                        name == "table"
                        or "table of" in name
                        or "list of" in name
                        or "list:" in name
                    ):
                        should_fetch_tables = True
                        break

            elif include_tables:
                should_fetch_tables = True

            if should_fetch_tables:
                should_fetch_html = True

            if should_fetch_html:
                html = await self.get_page_html(title=title, page_id=page_id, lang=lang)
                wiki_page.html_content = html
                if html:
                    helper = SectionHelper.from_html(html, url=wiki_page.url)
                    wiki_page.sections = helper.sections
                    wiki_page.tables = [
                        s for sec in helper.sections for s in sec.tables
                    ]
                    wiki_page.helper = helper
                    wiki_page.links = helper.links

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

            # with open(f"debug_{wiki_page.title}.html", "w", encoding="utf-8") as f:
            #     f.write(wiki_page.html_content)

            return wiki_page

        except (PageNotFoundError, DisambiguationError, RedirectError):
            raise
        except Exception as e:
            logger.error(f"Failed to get page '{title or page_id}': {str(e)}")
            raise WikipediaException(f"Failed to get page: {str(e)}")

    async def get_pages_batch(
        self,
        titles: List[str],
        include_tables: bool | str = "auto",
        include_html: bool = False,
        lang: Optional[str] = None,
    ) -> BatchResult:
        """
        Get multiple pages in batches for efficiency.

        Args:
            titles: List of page titles
            include_tables: Include extracted tables (True, False, or 'auto')
            include_html: Include raw HTML content
            lang: Optional language code for the pages
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
                chunk_result = await self._get_pages_chunk(
                    chunk,
                    include_tables=include_tables,
                    include_html=include_html,
                    lang=lang,
                )
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

        return await self._get_pages_chunk(
            titles, include_tables=include_tables, include_html=include_html, lang=lang
        )

    async def _get_pages_chunk(
        self,
        titles: List[str],
        include_tables: bool | str = "auto",
        include_html: bool = False,
        lang: Optional[str] = None,
    ) -> BatchResult:
        """Get a chunk of pages (internal method)."""

        try:
            successful = []
            failed = []

            res = await asyncio.gather(
                *[
                    self.get_page(
                        title=title,
                        include_tables=include_tables,
                        include_html=include_html,
                        lang=lang,
                    )
                    for title in titles
                ],
                return_exceptions=True,
            )
            for title, item in zip(titles, res):
                if isinstance(item, WikiPage):
                    successful.append(item)
                else:
                    failed.append(
                        {
                            "title": title,
                            "error": str(item),
                            "trace": traceback.format_exc(),
                        }
                    )

            return BatchResult(
                successful=successful, failed=failed, total_requested=len(titles)
            )

        except Exception as e:
            # All pages failed
            trace = traceback.format_exc()
            failed = [
                {
                    "title": title,
                    "error": str(e),
                    "trace": trace,
                }
                for title in titles
            ]
            return BatchResult(
                successful=[], failed=failed, total_requested=len(titles)
            )

    async def get_summary(
        self,
        title: str,
        sentences: int = 0,
        characters: int = 0,
        lang: Optional[str] = None,
    ) -> PageSummary:
        """
        Get a page summary.

        Args:
            title: Page title
            sentences: Number of sentences (0 = auto)
            characters: Number of characters (0 = auto)
            lang: Optional language code for the page

        Returns:
            PageSummary object
        """
        params: dict = {
            "prop": "extracts|info",
            "titles": self._normalize_title(title),
            "inprop": "url",
            "explaintext": "",
            "exintro": "",
        }

        if sentences > 0:
            params["exsentences"] = sentences
        elif characters > 0:
            params["exchars"] = characters

        if lang:
            params["lang"] = lang

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

    async def get_summary_batch(
        self,
        titles: List[str],
        sentences: int = 0,
        characters: int = 0,
        lang: Optional[str] = None,
    ) -> List[PageSummary]:
        """
        Get summaries for multiple pages in batch.

        Args:
            titles: List of page titles
            sentences: Number of sentences (0 = auto)
            characters: Number of characters (0 = auto)

        Returns:
            List of PageSummary objects
        """
        if not titles:
            return []

        if len(titles) > self.config.max_batch_size:
            # Process in chunks
            results = []
            for i in range(0, len(titles), self.config.max_batch_size):
                chunk = titles[i : i + self.config.max_batch_size]
                chunk_result = await self.get_summary_batch(
                    chunk, sentences, characters
                )
                results.extend(chunk_result)
            return results

        params: dict = {
            "prop": "extracts|info",
            "titles": "|".join([self._normalize_title(t) for t in titles]),
            "inprop": "url",
            "explaintext": "",
            "exintro": "",
        }

        if sentences > 0:
            params["exsentences"] = sentences
        elif characters > 0:
            params["exchars"] = characters

        if lang:
            params["lang"] = lang

        response = await self._make_request(params)
        pages = response["query"]["pages"]

        summaries = []
        for page_data in pages:
            if "missing" in page_data:
                continue
            summaries.append(
                PageSummary(
                    title=page_data["title"],
                    page_id=page_data["pageid"],
                    extract=page_data.get("extract", ""),
                    url=page_data["fullurl"],
                )
            )

        return summaries

    async def geosearch(
        self,
        latitude: float,
        longitude: float,
        radius: int = 1000,
        limit: int = 10,
        lang: Optional[str] = None,
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
        if lang:
            params["lang"] = lang

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

    async def random(
        self,
        count: int = 1,
        namespace: int = 0,
        lang: Optional[str] = None,
    ) -> RandomPageResult:
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
        if lang:
            params["lang"] = lang

        response = await self._make_request(params)
        random_pages = response["query"]["random"]

        return RandomPageResult(
            pages=[page["title"] for page in random_pages], namespace=namespace
        )

    async def suggest(
        self,
        query: str,
        lang: Optional[str] = None,
    ) -> Optional[str]:
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
        if lang:
            params["lang"] = lang

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

        return [
            Language(code=lang["code"], name=lang["*"], english_name=lang.get("en"))
            for lang in languages
        ]

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

    async def get_page_html_batch(
        self,
        titles: List[str],
        lang: Optional[str] = None,
    ) -> BatchHTMLResult:
        """
        Get HTML content for multiple pages in batch.

        Args:
            titles: List of page titles

        Returns:
            list of HTML content strings
        """
        # params = {
        #     "prop": "revisions",
        #     "rvprop": "content",
        #     "rvparse": "",
        #     "rvlimit": 1,
        #     "titles": "|".join([self._normalize_title(t) for t in titles]),
        # }
        # response = await self._make_request(params)
        # pages = response["query"]["pages"]
        # for page_data in pages:
        #     data = ""
        #     if "revisions" in page_data and page_data["revisions"]:
        #         rev = page_data["revisions"][0]
        #         if "slots" in rev and "main" in rev["slots"]:
        #             rev = rev["slots"]["main"]

        #         if "content" in rev:
        #             data = rev["content"]
        #         elif "*" in rev:
        #             data = rev["*"]
        #     html_contents.append(data)

        res = await asyncio.gather(
            *[self.get_page_html(title=title, lang=lang) for title in titles],
            return_exceptions=True,
        )

        successful = []
        failed = []
        for title, r in zip(titles, res):
            if isinstance(r, Exception):
                failed.append({"error": str(r), "title": title})
            elif isinstance(r, str):
                successful.append(HTMLResult(title=title, html=r))
            else:
                failed.append(
                    {
                        "error": f"Unexpected result type: {type(r)}",
                        "title": title,
                    }
                )

        return BatchHTMLResult(
            successful=successful, failed=failed, total_requested=len(titles)
        )

    async def get_page_html(
        self,
        title: Optional[str] = None,
        page_id: Optional[int] = None,
        lang: Optional[str] = None,
    ) -> str:
        """
        Get the HTML content of a Wikipedia page.

        Args:
            title: Page title (mutually exclusive with page_id)
            page_id: Page ID (mutually exclusive with title)

        Returns:
            Full page HTML content
        """
        if not title and not page_id:
            raise ValidationError(
                "title/page_id", "Either title or page_id must be provided"
            )

        params = {
            "prop": "revisions",
            "rvprop": "content",
            "rvparse": "",
            "rvlimit": 1,
        }

        if title:
            params["titles"] = self._normalize_title(title)
        else:
            params["pageids"] = page_id

        if lang:
            params["lang"] = lang

        response = await self._make_request(params)
        pages = response["query"]["pages"]

        if not pages or "missing" in pages[0]:
            raise PageNotFoundError(title or str(page_id))

        page_data = pages[0]
        data = ""
        if "revisions" in page_data and page_data["revisions"]:
            rev = page_data["revisions"][0]
            if "slots" in rev and "main" in rev["slots"]:
                rev = rev["slots"]["main"]

            if "content" in rev:
                data = rev["content"]
            elif "*" in rev:
                data = rev["*"]

        return data
