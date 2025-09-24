"""
Configuration classes for the Wikipedia client.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ClientConfig:
    """Configuration for the Wikipedia client."""

    # Language settings
    language: str = "en"

    # Rate limiting
    rate_limit_calls: int = 10  # calls per period
    rate_limit_period: float = 1.0  # seconds

    # Retry configuration
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    retry_max_wait: float = 60.0
    retry_on_status: tuple = (429, 500, 502, 503, 504)

    # Caching
    enable_cache: bool = True
    cache_ttl: int = 300  # seconds (5 minutes)
    max_cache_size: int = 1000  # number of entries
    cache_type: str = "memory"  # "memory" or "file"
    cache_dir: str = "./wiki_cache"  # used if cache_type is "file"
    cache_serializer: str = "pickle"  # "pickle" or "json"

    # Request settings
    timeout: float = 30.0  # seconds
    max_concurrent_requests: int = 10

    # User agent
    user_agent: str = (
        "modern-wikipedia/1.0.0 (https://github.com/example/modern-wikipedia)"
    )

    # Custom headers
    headers: Dict[str, str] = field(default_factory=dict)

    # API settings
    api_base_url: str = "https://{lang}.wikipedia.org/w/api.php"

    # Content settings
    auto_suggest: bool = True
    follow_redirects: bool = True

    # Batch settings
    max_batch_size: int = 50

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.rate_limit_calls <= 0:
            raise ValueError("rate_limit_calls must be positive")

        if self.rate_limit_period <= 0:
            raise ValueError("rate_limit_period must be positive")

        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

        if self.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")

        if self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive")

        if self.max_cache_size <= 0:
            raise ValueError("max_cache_size must be positive")

        if self.max_batch_size <= 0 or self.max_batch_size > 500:
            raise ValueError("max_batch_size must be between 1 and 500")

    @property
    def api_url(self) -> str:
        """Get the API URL for the current language."""
        return self.api_base_url.format(lang=self.language)

    def with_language(self, language: str) -> "ClientConfig":
        """Create a new config with a different language."""
        new_config = ClientConfig(**self.__dict__)
        new_config.language = language
        return new_config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "language": self.language,
            "rate_limit_calls": self.rate_limit_calls,
            "rate_limit_period": self.rate_limit_period,
            "max_retries": self.max_retries,
            "retry_backoff_factor": self.retry_backoff_factor,
            "retry_max_wait": self.retry_max_wait,
            "retry_on_status": self.retry_on_status,
            "enable_cache": self.enable_cache,
            "cache_ttl": self.cache_ttl,
            "max_cache_size": self.max_cache_size,
            "timeout": self.timeout,
            "max_concurrent_requests": self.max_concurrent_requests,
            "user_agent": self.user_agent,
            "headers": self.headers.copy(),
            "api_base_url": self.api_base_url,
            "auto_suggest": self.auto_suggest,
            "follow_redirects": self.follow_redirects,
            "max_batch_size": self.max_batch_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClientConfig":
        """Create config from dictionary."""
        return cls(**data)
