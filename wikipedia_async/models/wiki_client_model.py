"""
Pydantic models for Wikipedia API responses and configuration.
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import re

from wikipedia_async.helpers.section_helpers import Section, SectionHelper
from wikipedia_async.models.section_models import Table


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
    url: str = Field(..., description="Full URL to the page")
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


class SearchResults(BaseModel):
    """Search results from Wikipedia."""

    results: List[SearchResult] = Field(
        default_factory=list, description="List of search results"
    )
    suggestion: Optional[str] = Field(None, description="Search suggestion")


class PageSummary(BaseModel):
    """Brief summary of a Wikipedia page."""

    title: str = Field(..., description="Page title")
    page_id: int = Field(..., description="Wikipedia page ID")
    extract: str = Field(..., description="Page summary text")
    url: str = Field(..., description="Full URL to the page")


class WikiPage(BaseModel):
    """Complete Wikipedia page information."""

    title: str = Field(..., description="Page title")
    page_id: int = Field(..., description="Wikipedia page ID")
    summary: str = Field(..., description="First paragraph summary")
    url: str = Field(..., description="Full URL to the page")
    extract: Optional[str] = Field(None, description="Page summary")
    content: Optional[str] = Field(None, description="Full page content")

    # Field to store HTML content (excluded from serialization)
    html_content: Optional[str] = Field(
        None, description="Raw HTML content", exclude=True
    )

    def html(self) -> Optional[str]:
        """
        Get the HTML content of the page.

        Returns:
            HTML content if available, None otherwise
        """
        return self.html_content

    # Metadata
    revision_id: Optional[int] = Field(None, description="Current revision ID")
    parent_id: Optional[int] = Field(None, description="Parent revision ID")
    last_modified: Optional[datetime] = Field(
        None, description="Last modification time"
    )

    # Content organization
    sections: list[Section] = Field(
        default_factory=list, description="Section titles with content"
    )
    categories: List[str] = Field(default_factory=list, description="Page categories")

    # External content
    images: List[str] = Field(default_factory=list, description="Image URLs")
    references: List[str] = Field(default_factory=list, description="External links")
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
    tables: list[Table] = Field(
        default_factory=list,
        description="Extracted tables from the page",
    )
    """A list of tables extracted from the page."""

    helper: SectionHelper = Field(..., description="Helper for section operations")
    """Helper instance for section operations."""

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

class HTMLResult(BaseModel):
    title: str = Field(..., description="Page title")
    html: str = Field(..., description="HTML content of the page")


class BatchHTMLResult(BaseModel):
    """Result from batch HTML retrieval."""

    successful: list[HTMLResult] = Field(
        default_factory=list, description="Successfully retrieved HTML contents"
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


class APIResponse(BaseModel):
    """Generic API response wrapper."""

    status: str = Field(..., description="Response status")
    data: Dict[str, Any] = Field(..., description="Response data")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
    request_id: Optional[str] = Field(None, description="Request tracking ID")
