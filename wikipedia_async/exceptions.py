"""
Exception classes for the modern Wikipedia client.
"""

from typing import List, Optional


class WikipediaException(Exception):
    """Base exception for all Wikipedia-related errors."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NetworkError(WikipediaException):
    """Raised when a network-related error occurs."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class TimeoutError(WikipediaException):
    """Raised when a request times out."""

    def __init__(self, message: str = "Request timed out") -> None:
        super().__init__(message)


class RateLimitError(WikipediaException):
    """Raised when rate limit is exceeded."""

    def __init__(
        self, message: str = "Rate limit exceeded", retry_after: Optional[float] = None
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PageNotFoundError(WikipediaException):
    """Raised when a requested page is not found."""

    def __init__(self, title: str) -> None:
        super().__init__(f"Page not found: {title}")
        self.title = title


class DisambiguationError(WikipediaException):
    """Raised when a page title refers to multiple pages."""

    def __init__(self, title: str, options: List[str]) -> None:
        super().__init__(f"Page '{title}' is ambiguous. Options: {', '.join(options[:5])}")
        self.title = title
        self.options = options


class RedirectError(WikipediaException):
    """Raised when a page redirects and redirects are disabled."""

    def __init__(self, from_title: str, to_title: str) -> None:
        super().__init__(f"Page '{from_title}' redirects to '{to_title}'")
        self.from_title = from_title
        self.to_title = to_title


class InvalidLanguageError(WikipediaException):
    """Raised when an invalid language code is provided."""

    def __init__(self, language: str) -> None:
        super().__init__(f"Invalid language code: {language}")
        self.language = language


class APIError(WikipediaException):
    """Raised when the Wikipedia API returns an error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"API Error {code}: {message}")
        self.code = code


class ValidationError(WikipediaException):
    """Raised when input validation fails."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"Validation error for {field}: {message}")
        self.field = field