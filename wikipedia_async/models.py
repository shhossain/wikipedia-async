"""
Pydantic models for Wikipedia API responses and configuration.
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator, HttpUrl
import re


class Coordinates(BaseModel):
    """Geographic coordinates."""

    latitude: Decimal = Field(..., description="Latitude in decimal degrees")
    longitude: Decimal = Field(..., description="Longitude in decimal degrees")

    @validator("latitude")
    def validate_latitude(cls, v: Decimal) -> Decimal:
        if not -90 <= v <= 90:
            raise ValueError("Latitude must be between -90 and 90 degrees")
        return v

    @validator("longitude")
    def validate_longitude(cls, v: Decimal) -> Decimal:
        if not -180 <= v <= 180:
            raise ValueError("Longitude must be between -180 and 180 degrees")
        return v


class Language(BaseModel):
    """Wikipedia language information."""

    code: str = Field(..., description="Language code (e.g., 'en', 'fr')")
    name: str = Field(..., description="Language name in its native form")
    english_name: Optional[str] = Field(
        None, description="English name of the language"
    )


class SearchResult(BaseModel):
    """Search result from Wikipedia."""

    title: str = Field(..., description="Page title")
    snippet: Optional[str] = Field(None, description="Search result snippet")
    page_id: Optional[int] = Field(None, description="Wikipedia page ID")
    word_count: Optional[int] = Field(None, description="Number of words in the page")
    size: Optional[int] = Field(None, description="Page size in bytes")
    timestamp: Optional[datetime] = Field(None, description="Last modification time")

    # remove all span tags from snippet
    @validator("snippet", pre=True, always=True)
    def clean_snippet(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # Remove HTML tags
        clean = re.sub(r"<[^>]*>", "", v)
        # Replace multiple spaces with a single space
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean


class PageSummary(BaseModel):
    """Brief summary of a Wikipedia page."""

    title: str = Field(..., description="Page title")
    page_id: int = Field(..., description="Wikipedia page ID")
    extract: str = Field(..., description="Page summary text")
    url: HttpUrl = Field(..., description="Full URL to the page")


class WikiPage(BaseModel):
    """Complete Wikipedia page information."""

    title: str = Field(..., description="Page title")
    page_id: int = Field(..., description="Wikipedia page ID")
    url: HttpUrl = Field(..., description="Full URL to the page")
    extract: Optional[str] = Field(None, description="Page summary")
    summary: Optional[str] = Field(None, description="Page summary (alias for extract)")
    content: Optional[str] = Field(None, description="Full page content")

    @validator("summary", always=True)
    def sync_summary_with_extract(cls, v, values):
        """Keep summary and extract fields synchronized."""
        return v if v is not None else values.get("extract")

    # Metadata
    revision_id: Optional[int] = Field(None, description="Current revision ID")
    parent_id: Optional[int] = Field(None, description="Parent revision ID")
    last_modified: Optional[datetime] = Field(
        None, description="Last modification time"
    )

    # Content organization
    sections: List[str] = Field(default_factory=list, description="Section titles")
    categories: List[str] = Field(default_factory=list, description="Page categories")

    # External content
    images: List[HttpUrl] = Field(default_factory=list, description="Image URLs")
    references: List[HttpUrl] = Field(
        default_factory=list, description="External links"
    )
    links: List[str] = Field(
        default_factory=list, description="Internal Wikipedia links"
    )

    # Geographic data
    coordinates: Optional[Coordinates] = Field(
        None, description="Geographic coordinates"
    )

    # Additional metadata
    language: str = Field(default="en", description="Page language")
    namespace: int = Field(default=0, description="Wikipedia namespace")


class GeoSearchResult(BaseModel):
    """Geographic search result."""

    title: str = Field(..., description="Page title")
    page_id: int = Field(..., description="Wikipedia page ID")
    coordinates: Coordinates = Field(..., description="Geographic coordinates")
    distance: Optional[float] = Field(
        None, description="Distance from search point in meters"
    )


class BatchResult(BaseModel):
    """Result from batch operations."""

    successful: List[WikiPage] = Field(
        default_factory=list, description="Successfully retrieved pages"
    )
    failed: List[Dict[str, Any]] = Field(
        default_factory=list, description="Failed requests with errors"
    )
    total_requested: int = Field(..., description="Total number of pages requested")

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_requested == 0:
            return 0.0
        return (len(self.successful) / self.total_requested) * 100

    # add a iter
    def __iter__(self):
        """Iterate over all pages, yielding successful pages first, then failed."""
        for page in self.successful:
            yield page

    def __len__(self):
        """Total number of results (successful + failed)."""
        return len(self.successful)

    # slice
    def __getitem__(self, index):
        """Support indexing and slicing."""
        combined = self.successful
        return combined[index]


class RandomPageResult(BaseModel):
    """Result from random page request."""

    pages: List[str] = Field(..., description="Random page titles")
    namespace: int = Field(default=0, description="Namespace of the pages")


class SuggestionResult(BaseModel):
    """Search suggestion result."""

    query: str = Field(..., description="Original search query")
    suggestion: Optional[str] = Field(None, description="Suggested search term")
    results: List[SearchResult] = Field(
        default_factory=list, description="Search results"
    )

    def __iter__(self):
        """Iterate over search results."""
        for result in self.results:
            yield result

    def __len__(self):
        """Number of search results."""
        return len(self.results)

    def __getitem__(self, index):
        """Support indexing and slicing of results."""
        return self.results[index]

    def __contains__(self, item):
        """Check if a SearchResult with the given title exists in results."""
        return any(
            (
                result.title == item
                if isinstance(item, str)
                else (
                    item.title
                    if isinstance(item, SearchResult)
                    else False and result == item
                )
            )
            for result in self.results
        )


class APIResponse(BaseModel):
    """Generic API response wrapper."""

    status: str = Field(..., description="Response status")
    data: Dict[str, Any] = Field(..., description="Response data")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
    request_id: Optional[str] = Field(None, description="Request tracking ID")
