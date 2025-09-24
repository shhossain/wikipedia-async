"""
Examples demonstrating the different cache implementations.
"""

import asyncio
from wikipedia_async.cache import BaseCache, MemoryCache, FileCache, cache_key


async def demo_memory_cache():
    """Demonstrate memory cache usage."""
    print("=== Memory Cache Demo ===")
    
    # Create a memory cache with 100 max entries and 60 second TTL
    cache = MemoryCache[str](max_size=100, default_ttl=60)
    
    # Set some values
    await cache.set("user:123", "John Doe")
    await cache.set("user:456", "Jane Smith", ttl=30)  # Custom TTL
    
    # Get values
    user1 = await cache.get("user:123")
    user2 = await cache.get("user:456")
    print(f"Retrieved: {user1}, {user2}")
    
    # Check cache stats
    stats = await cache.stats()
    print(f"Cache stats: {stats}")
    
    # Cache size
    size = await cache.size()
    print(f"Cache size: {size}")
    
    print()


async def demo_file_cache():
    """Demonstrate file cache usage."""
    print("=== File Cache Demo ===")
    
    # Create a file cache with pickle serialization
    cache = FileCache[dict](
        cache_dir="./cache_data",
        max_size=50,
        default_ttl=300,
        serializer="pickle"
    )
    
    # Set some complex data
    user_data = {
        "id": 123,
        "name": "John Doe",
        "email": "john@example.com",
        "settings": {"theme": "dark", "notifications": True}
    }
    
    await cache.set("user_data:123", user_data)
    
    # Create a JSON file cache for simple data
    json_cache = FileCache[str](
        cache_dir="./json_cache",
        max_size=100,
        default_ttl=600,
        serializer="json"
    )
    
    await json_cache.set("message", "Hello, World!")
    
    # Retrieve data
    retrieved_user = await cache.get("user_data:123")
    retrieved_message = await json_cache.get("message")
    
    print(f"Retrieved user: {retrieved_user}")
    print(f"Retrieved message: {retrieved_message}")
    
    # Check file cache stats
    stats = await cache.stats()
    print(f"File cache stats: {stats}")
    
    print()


async def demo_cache_key_generation():
    """Demonstrate cache key generation utility."""
    print("=== Cache Key Generation Demo ===")
    
    # Generate keys for function calls
    key1 = cache_key("search", "python programming", page=1, limit=10)
    key2 = cache_key("search", "python programming", limit=10, page=1)  # Same data, different order
    key3 = cache_key("search", "javascript", page=1, limit=10)
    
    print(f"Key 1: {key1}")
    print(f"Key 2: {key2}")
    print(f"Key 3: {key3}")
    print(f"Key 1 == Key 2: {key1 == key2}")  # Should be True (same args, different order)
    
    print()


async def demo_cache_operations():
    """Demonstrate common cache operations."""
    print("=== Cache Operations Demo ===")
    
    cache = MemoryCache[int](max_size=5, default_ttl=2)  # Short TTL for demo
    
    # Add some data
    for i in range(10):
        await cache.set(f"item_{i}", i * 10)
    
    print(f"Cache size after adding 10 items (max 5): {await cache.size()}")
    
    # Check what's in cache
    for i in range(10):
        value = await cache.get(f"item_{i}")
        if value is not None:
            print(f"item_{i}: {value}")
    
    # Wait for expiration
    print("\nWaiting 3 seconds for expiration...")
    await asyncio.sleep(3)
    
    # Check expired entries
    expired_count = await cache.cleanup_expired()
    print(f"Cleaned up {expired_count} expired entries")
    print(f"Cache size after cleanup: {await cache.size()}")
    
    # Clear all
    await cache.clear()
    print(f"Cache size after clear: {await cache.size()}")
    
    print()


async def demo_polymorphic_usage():
    """Demonstrate using caches polymorphically via base class."""
    print("=== Polymorphic Cache Usage Demo ===")
    
    caches: list[BaseCache[str]] = [
        MemoryCache[str](max_size=10, default_ttl=60),
        FileCache[str](cache_dir="./poly_cache", max_size=10, default_ttl=60)
    ]
    
    for i, cache in enumerate(caches):
        cache_type = "Memory" if isinstance(cache, MemoryCache) else "File"
        print(f"\n--- {cache_type} Cache ---")
        
        # Same operations on different cache types
        await cache.set("greeting", f"Hello from {cache_type} cache!")
        greeting = await cache.get("greeting")
        print(f"Retrieved: {greeting}")
        
        stats = await cache.stats()
        print(f"Stats: {stats}")
    
    print()


async def main():
    """Run all demonstrations."""
    print("Cache System Demonstrations\n")
    
    await demo_memory_cache()
    await demo_file_cache()
    await demo_cache_key_generation()
    await demo_cache_operations()
    await demo_polymorphic_usage()
    
    print("All demonstrations completed!")


if __name__ == "__main__":
    asyncio.run(main())