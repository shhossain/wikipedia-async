"""
Example usage of the modern Wikipedia client.
"""

import asyncio
import logging
from wikipedia_async import WikipediaClient, ClientConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_usage_example():
    """Basic usage with default configuration."""
    print("=== Basic Usage Example ===")

    async with WikipediaClient() as client:
        # Simple search
        res = await client.search("Python programming", limit=5)
        print(f"Found {len(res.results)} search results:")
        for i, result in enumerate(res.results, 1):
            print(f"{i}. {result.title}")

        # Get a page
        page = await client.get_page("Python (programming language)")
        print(f"\nPage: {page.title}")
        print(f"Summary: {page.summary[:200]}...")

        # Get page sections
        print("\nSections:")
        for section in page.sections:
            print(f"Title: {section.title}: {section.content[:100]}...")
        print(f"Total sections: {len(page.sections)}")

        # Convert to markdown
        for section in page.sections:
            print(f"\nMarkdown for section '{section.title}':")
            print(section.to_string(markdown=True)[:300] + "...")


async def advanced_usage_example():
    """Advanced usage with custom configuration."""
    print("\n=== Advanced Usage Example ===")

    # Custom configuration for production use
    config = ClientConfig(
        language="en",
        rate_limit_calls=15,  # Higher rate limit
        rate_limit_period=1.0,
        max_retries=5,  # More retries for reliability
        cache_ttl=600,  # 10 minute cache
        timeout=45.0,  # Longer timeout
        max_concurrent_requests=20,  # More concurrent requests
        user_agent="MyApp/1.0 (https://myapp.com)",
    )

    async with WikipediaClient(config=config) as client:
        # Batch operations
        titles = ["Python", "JavaScript", "Rust", "Go", "TypeScript"]
        batch_result = await client.get_pages_batch(titles)

        print("Batch operation results:")
        print(f"  Success rate: {batch_result.success_rate:.1f}%")
        print(f"  Successful: {len(batch_result.successful)}")
        print(f"  Failed: {len(batch_result.failed)}")

        # Get detailed page with all content
        detailed_page = await client.get_page(
            "Artificial intelligence",
            include_content=True,
            include_images=True,
            include_categories=True,
            include_coordinates=True,
        )

        print(f"\nDetailed page: {detailed_page.title}")
        print(f"Categories: {detailed_page.categories[:5]}")  # First 5 categories
        print(f"Images: {len(detailed_page.images)}")
        if detailed_page.coordinates:
            print(
                f"Coordinates: {detailed_page.coordinates.latitude}, {detailed_page.coordinates.longitude}"
            )


async def geographic_search_example():
    """Geographic search example."""
    print("\n=== Geographic Search Example ===")

    async with WikipediaClient() as client:
        # Search near Paris coordinates
        paris_lat, paris_lon = 48.8566, 2.3522
        geo_results = await client.geosearch(
            latitude=paris_lat, longitude=paris_lon, radius=5000, limit=10  # 5km radius
        )

        print(f"Found {len(geo_results)} pages near Paris:")
        for result in geo_results:
            print(f"  {result.title} (distance: {result.distance}m)")


async def language_switching_example():
    """Example of switching languages."""
    print("\n=== Language Switching Example ===")

    async with WikipediaClient() as client:
        # Search in English
        res = await client.search("Artificial Intelligence", limit=3)
        en_results = res.results
        print("English results:")
        for result in en_results:
            print(f"  {result.title}")

        # Switch to French
        await client.set_language("fr")
        res = await client.search("Intelligence artificielle", limit=3)
        fr_results = res.results
        print("\nFrench results:")
        for result in fr_results:
            print(f"  {result.title}")


async def error_handling_example():
    """Example of error handling."""
    print("\n=== Error Handling Example ===")

    async with WikipediaClient() as client:
        try:
            # Try to get a non-existent page
            await client.get_page("ThisPageDoesNotExist123456")
        except Exception as e:
            print(f"Expected error: {type(e).__name__}: {e}")

        try:
            # Try disambiguation page
            await client.get_page("Mercury")
        except Exception as e:
            print(f"Disambiguation error: {type(e).__name__}: {e}")


async def caching_example():
    """Example showing caching benefits."""
    print("\n=== Caching Example ===")

    config = ClientConfig(cache_ttl=60)  # 1 minute cache

    async with WikipediaClient(config=config) as client:
        import time

        # First request (will hit API)
        start = time.time()
        _ = await client.get_page("Python (programming language)")
        first_time = time.time() - start

        # Second request (will hit cache)
        start = time.time()
        _ = await client.get_page("Python (programming language)")
        second_time = time.time() - start

        print(f"First request time: {first_time:.3f}s")
        print(f"Second request time (cached): {second_time:.3f}s")
        print(f"Speed improvement: {first_time/second_time:.1f}x faster")

        # Cache statistics
        stats = await client.cache_stats()
        print(f"Cache stats: {stats}")


async def random_and_suggestions_example():
    """Example of random pages and suggestions."""
    print("\n=== Random Pages and Suggestions Example ===")

    async with WikipediaClient() as client:
        # Get random pages
        random_result = await client.random(count=5)
        print("Random pages:")
        for title in random_result.pages:
            print(f"  {title}")

        # Get suggestion for misspelled query
        suggestion = await client.suggest("artifical inteligence")  # Misspelled
        if suggestion:
            print(f"\nSuggestion for 'artifical inteligence': {suggestion}")


async def production_monitoring_example():
    """Example for production monitoring."""
    print("\n=== Production Monitoring Example ===")

    config = ClientConfig(
        rate_limit_calls=50,
        rate_limit_period=1.0,
        max_retries=3,
        cache_ttl=300,
        max_concurrent_requests=25,
    )

    async with WikipediaClient(config=config) as client:
        # Simulate high-load operations
        tasks = []
        search_queries = [
            "Machine learning",
            "Data science",
            "Computer vision",
            "Natural language processing",
            "Deep learning",
        ]

        for query in search_queries:
            task = client.search(query, limit=10)
            tasks.append(task)

        # Execute all searches concurrently
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = asyncio.get_event_loop().time()

        successful_searches = sum(1 for r in results if not isinstance(r, Exception))
        failed_searches = len(results) - successful_searches

        print(f"Concurrent operations completed in {end_time - start_time:.2f}s")
        print(f"Successful: {successful_searches}/{len(results)}")
        print(f"Failed: {failed_searches}/{len(results)}")

        # Cache performance
        cache_stats = await client.cache_stats()
        if cache_stats["cache_enabled"]:
            print(f"Cache utilization: {cache_stats['utilization']:.1%}")


async def main():
    """Run all examples."""
    examples = [
        basic_usage_example,
        advanced_usage_example,
        geographic_search_example,
        language_switching_example,
        error_handling_example,
        caching_example,
        random_and_suggestions_example,
        production_monitoring_example,
    ]

    for example in examples:
        try:
            await example()
        except Exception as e:
            logger.error(f"Example {example.__name__} failed: {e}")

        # Small delay between examples
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(main())
