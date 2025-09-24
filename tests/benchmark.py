"""
Performance benchmarks for the modern Wikipedia client.
"""

import asyncio
import time
import statistics
from typing import List
from wikipedia_async import WikipediaClient, ClientConfig


async def benchmark_search_performance():
    """Benchmark search performance."""
    print("=== Search Performance Benchmark ===")
    
    config = ClientConfig(
        rate_limit_calls=50,
        max_concurrent_requests=20,
        cache_ttl=300
    )
    
    async with WikipediaClient(config=config) as client:
        queries = [
            "Python programming",
            "Machine learning", 
            "Artificial intelligence",
            "Data science",
            "Computer vision",
            "Natural language processing",
            "Deep learning",
            "Web development",
            "Software engineering",
            "Database systems"
        ]
        
        # Sequential execution
        print("Sequential execution...")
        start_time = time.time()
        for query in queries:
            await client.search(query, limit=5)
        sequential_time = time.time() - start_time
        
        # Clear cache for fair comparison
        await client.clear_cache()
        
        # Concurrent execution
        print("Concurrent execution...")
        start_time = time.time()
        tasks = [client.search(query, limit=5) for query in queries]
        await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        print(f"Sequential time: {sequential_time:.2f}s")
        print(f"Concurrent time: {concurrent_time:.2f}s")
        print(f"Speed improvement: {sequential_time/concurrent_time:.1f}x")


async def benchmark_cache_performance():
    """Benchmark cache performance."""
    print("\n=== Cache Performance Benchmark ===")
    
    config = ClientConfig(cache_ttl=60)
    
    async with WikipediaClient(config=config) as client:
        query = "Python programming"
        times = []
        
        # First request (cold cache)
        start = time.time()
        await client.search(query)
        first_time = time.time() - start
        
        # Subsequent requests (warm cache)
        for _ in range(10):
            start = time.time()
            await client.search(query)
            times.append(time.time() - start)
        
        avg_cached_time = statistics.mean(times)
        
        print(f"First request (cold): {first_time:.3f}s")
        print(f"Cached requests (avg): {avg_cached_time:.3f}s")
        print(f"Cache speedup: {first_time/avg_cached_time:.1f}x")


async def benchmark_batch_vs_individual():
    """Benchmark batch vs individual requests."""
    print("\n=== Batch vs Individual Requests Benchmark ===")
    
    async with WikipediaClient() as client:
        titles = [
            "Python", "JavaScript", "Java", "C++", "C#",
            "Ruby", "Go", "Rust", "Swift", "Kotlin"
        ]
        
        # Individual requests
        start_time = time.time()
        individual_results = []
        for title in titles:
            try:
                page = await client.get_page(title)
                individual_results.append(page)
            except Exception:
                pass
        individual_time = time.time() - start_time
        
        # Clear cache
        await client.clear_cache()
        
        # Batch request
        start_time = time.time()
        batch_result = await client.get_pages_batch(titles)
        batch_time = time.time() - start_time
        
        print(f"Individual requests: {individual_time:.2f}s ({len(individual_results)} pages)")
        print(f"Batch request: {batch_time:.2f}s ({len(batch_result.successful)} pages)")
        print(f"Batch speedup: {individual_time/batch_time:.1f}x")


async def benchmark_rate_limiting():
    """Benchmark rate limiting behavior."""
    print("\n=== Rate Limiting Benchmark ===")
    
    # High rate limit config
    fast_config = ClientConfig(rate_limit_calls=100, rate_limit_period=1.0)
    
    # Low rate limit config  
    slow_config = ClientConfig(rate_limit_calls=5, rate_limit_period=1.0)
    
    queries = ["test query " + str(i) for i in range(20)]
    
    # Fast client
    async with WikipediaClient(config=fast_config) as fast_client:
        start_time = time.time()
        tasks = [fast_client.search(query, limit=1) for query in queries]
        await asyncio.gather(*tasks, return_exceptions=True)
        fast_time = time.time() - start_time
    
    # Slow client
    async with WikipediaClient(config=slow_config) as slow_client:
        start_time = time.time()
        tasks = [slow_client.search(query, limit=1) for query in queries]
        await asyncio.gather(*tasks, return_exceptions=True)
        slow_time = time.time() - start_time
    
    print(f"High rate limit (100/s): {fast_time:.2f}s")
    print(f"Low rate limit (5/s): {slow_time:.2f}s")
    print(f"Rate limiting overhead: {slow_time/fast_time:.1f}x slower")


async def benchmark_memory_usage():
    """Simple memory usage benchmark."""
    print("\n=== Memory Usage Benchmark ===")
    
    import tracemalloc
    
    tracemalloc.start()
    
    config = ClientConfig(
        max_cache_size=1000,
        cache_ttl=300
    )
    
    async with WikipediaClient(config=config) as client:
        # Perform many operations to fill cache
        queries = [f"query {i}" for i in range(100)]
        
        for query in queries:
            try:
                await client.search(query, limit=5)
            except Exception:
                pass
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
        print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
        
        # Cache stats
        cache_stats = await client.cache_stats()
        print(f"Cache entries: {cache_stats['total_entries']}")
        print(f"Cache utilization: {cache_stats['utilization']:.1%}")


async def run_all_benchmarks():
    """Run all benchmarks."""
    benchmarks = [
        benchmark_search_performance,
        benchmark_cache_performance,
        benchmark_batch_vs_individual,
        benchmark_rate_limiting,
        benchmark_memory_usage,
    ]
    
    for benchmark in benchmarks:
        try:
            await benchmark()
            await asyncio.sleep(1)  # Brief pause between benchmarks
        except Exception as e:
            print(f"Benchmark {benchmark.__name__} failed: {e}")


if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())