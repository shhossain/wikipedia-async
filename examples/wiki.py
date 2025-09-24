"""
Modern Wikipedia API Client - Main Usage Example

A production-ready async Wikipedia client with advanced features.
"""

import asyncio
from wikipedia_async import WikipediaClient, ClientConfig


async def main():
    """Main example showing key features."""
    
    # Create client with optimal production settings
    config = ClientConfig(
        language="en",
        rate_limit_calls=15,     # Reasonable rate limiting
        rate_limit_period=1.0,
        max_retries=3,           # Retry failed requests
        cache_ttl=300,           # 5-minute cache
        timeout=30.0,
        max_concurrent_requests=10,
        user_agent="MyApp/1.0 (contact@myapp.com)"
    )
    
    async with WikipediaClient(config=config) as client:
        print("=== Modern Wikipedia Client Demo ===\n")
        
        # 1. Search for articles
        print("1. Searching for 'Python programming'...")
        results = await client.search("Python programming", limit=5)
        
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.title}")
            if result.snippet:
                print(f"      {result.snippet[:100]}...")
        
        # 2. Get detailed page information
        print("\n2. Getting detailed page information...")
        page = await client.get_page(
            "Python (programming language)",
            include_content=True,
            include_categories=True,
            include_images=True
        )
        
        print(f"   Title: {page.title}")
        print(f"   URL: {page.url}")
        print(f"   Page ID: {page.page_id}")
        print(f"   Categories: {len(page.categories)}")
        print(f"   Images: {len(page.images)}")
        if page.extract:
            print(f"   Summary: {page.extract[:200]}...")
        
        # 3. Batch operations for efficiency
        print("\n3. Batch operations...")
        programming_languages = [
            "Python", "JavaScript", "Java", "C++", "Rust"
        ]
        
        batch_result = await client.get_pages_batch(programming_languages)
        print(f"   Requested: {batch_result.total_requested} pages")
        print(f"   Success: {len(batch_result.successful)} pages")
        print(f"   Failed: {len(batch_result.failed)} pages")
        print(f"   Success rate: {batch_result.success_rate:.1f}%")
        
        # 4. Geographic search
        print("\n4. Geographic search near San Francisco...")
        geo_results = await client.geosearch(
            latitude=37.7749,   # San Francisco coordinates
            longitude=-122.4194,
            radius=2000,        # 2km radius
            limit=5
        )
        
        for result in geo_results:
            distance_km = result.distance / 1000 if result.distance else 0
            print(f"   {result.title} ({distance_km:.1f}km away)")
        
        # 5. Language switching
        print("\n5. Language switching to French...")
        await client.set_language("fr")
        
        fr_results = await client.search("Intelligence artificielle", limit=3)
        for result in fr_results:
            print(f"   {result.title}")
        
        # Switch back to English
        await client.set_language("en")
        
        # 6. Random pages
        print("\n6. Random pages...")
        random_result = await client.random(count=5)
        
        for title in random_result.pages:
            print(f"   {title}")
        
        # 7. Cache performance
        print("\n7. Cache performance test...")
        import time
        
        # First request (hits API)
        start = time.time()
        await client.search("Machine learning")
        first_time = time.time() - start
        
        # Second request (hits cache)
        start = time.time()
        await client.search("Machine learning")
        second_time = time.time() - start
        
        print(f"   First request: {first_time:.3f}s")
        print(f"   Cached request: {second_time:.3f}s")
        print(f"   Speedup: {first_time/second_time:.1f}x faster")
        
        # 8. Cache statistics
        stats = await client.cache_stats()
        if stats["cache_enabled"]:
            print("\n8. Cache statistics:")
            print(f"   Total entries: {stats['total_entries']}")
            print(f"   Active entries: {stats['active_entries']}")
            print(f"   Cache utilization: {stats['utilization']:.1%}")
        
        print("\n=== Demo Complete ===")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())