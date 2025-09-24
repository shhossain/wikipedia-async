# Modern Wikipedia API Client

A async Wikipedia API client with advanced features for professional use.

## Features

- **Async/Await Support**: Built with `aiohttp` for high-performance async operations
- **Automatic Rate Limiting**: Configurable rate limiting with smart backoff
- **Retry Logic**: Exponential backoff with jitter for robust error handling
- **Caching**: Time-based caching with TTL support
- **Multi-language Support**: Easy language switching
- **Batch Operations**: Optimized batch requests for multiple pages
- **Type Safety**: Full type annotations with Pydantic models
- **Error Handling**: Comprehensive error hierarchy
- **Production Ready**: Optimal defaults for production use

## Quick Start

### Installation

```bash
pip install wikipedia-async
```

```python
import asyncio
from wikipedia_async import WikipediaClient

async def main():
    # Initialize client with optimal defaults
    client = WikipediaClient()

    # Search for articles
    results = await client.search("Python programming")

    # Get page content
    page = await client.get_page("Python (programming language)")
    print(f"Title: {page.title}")
    print(f"Summary: {page.summary[:200]}...")

    # Batch operations
    pages = await client.get_pages_batch(["Python", "JavaScript", "Rust"])

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

```python
from wikipedia_async import WikipediaClient, ClientConfig

config = ClientConfig(
    language="en",
    rate_limit_calls=10,
    rate_limit_period=1.0,
    max_retries=3,
    cache_ttl=300,  # 5 minutes
    timeout=30.0,
    max_concurrent_requests=10
)

client = WikipediaClient(config=config)
```

See Docs [Here](docs)

See Examples [Here](examples)



