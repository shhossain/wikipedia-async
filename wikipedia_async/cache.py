"""
Caching utilities for the Wikipedia client.
"""

import asyncio
import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Literal, Optional, TypeVar, Generic, Hashable
from dataclasses import dataclass


T = TypeVar("T")


def _stable_hash(value: str, length: int = 32) -> str:
    """Generate a stable hash using MD5."""
    return hashlib.md5(value.encode("utf-8")).hexdigest()[:length]


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with TTL support."""

    value: T
    timestamp: float
    ttl: float

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.timestamp > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """Convert cache entry to dictionary for serialization."""
        return {"value": self.value, "timestamp": self.timestamp, "ttl": self.ttl}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry[T]":
        """Create cache entry from dictionary."""
        return cls(value=data["value"], timestamp=data["timestamp"], ttl=data["ttl"])


class BaseCache(ABC, Generic[T]):
    """Abstract base class for all cache implementations."""

    def __init__(self, max_size: int = 1000, default_ttl: float = 300) -> None:
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of entries to store
            default_ttl: Default TTL in seconds
        """
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()

    @abstractmethod
    async def get(self, key: Hashable) -> Optional[T]:
        """Get a value from the cache."""
        pass

    @abstractmethod
    async def set(self, key: Hashable, value: T, ttl: Optional[float] = None) -> None:
        """Set a value in the cache."""
        pass

    @abstractmethod
    async def delete(self, key: Hashable) -> bool:
        """Delete a key from the cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    async def size(self) -> int:
        """Get the current cache size."""
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Remove all expired entries and return the number removed."""
        pass

    @abstractmethod
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCache(BaseCache[T]):
    """Thread-safe async in-memory cache with TTL (Time To Live) support."""

    def __init__(self, max_size: int = 1000, default_ttl: float = 300, **kw) -> None:
        """
        Initialize the memory cache.

        Args:
            max_size: Maximum number of entries to store
            default_ttl: Default TTL in seconds
        """
        super().__init__(max_size, default_ttl)
        self._cache: Dict[Hashable, CacheEntry[T]] = {}

    async def get(self, key: Hashable) -> Optional[T]:
        """Get a value from the cache."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            if entry.is_expired:
                del self._cache[key]
                return None

            return entry.value

    async def set(self, key: Hashable, value: T, ttl: Optional[float] = None) -> None:
        """Set a value in the cache."""
        async with self._lock:
            # Remove oldest entries if cache is full
            if len(self._cache) >= self._max_size:
                await self._evict_oldest()

            ttl = ttl or self._default_ttl
            self._cache[key] = CacheEntry(value=value, timestamp=time.time(), ttl=ttl)

    async def delete(self, key: Hashable) -> bool:
        """Delete a key from the cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()

    async def size(self) -> int:
        """Get the current cache size."""
        async with self._lock:
            return len(self._cache)

    async def cleanup_expired(self) -> int:
        """Remove all expired entries and return the number removed."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired
            ]

            for key in expired_keys:
                del self._cache[key]

            return len(expired_keys)

    async def _evict_oldest(self) -> None:
        """Evict the oldest entry from the cache."""
        if not self._cache:
            return

        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
        del self._cache[oldest_key]

    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_entries = len(self._cache)
            expired_entries = sum(
                1 for entry in self._cache.values() if entry.is_expired
            )

            return {
                "total_entries": total_entries,
                "expired_entries": expired_entries,
                "active_entries": total_entries - expired_entries,
                "max_size": self._max_size,
                "utilization": (
                    total_entries / self._max_size if self._max_size > 0 else 0
                ),
            }


class FileCache(BaseCache[T]):
    """Thread-safe async file-based cache with TTL (Time To Live) support."""

    def __init__(
        self,
        cache_dir: str = ".cache",
        max_size: int = 1000,
        default_ttl: float = 300,
        serializer: str = "pickle",
        **kw,
    ) -> None:
        """
        Initialize the file cache.

        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of entries to store
            default_ttl: Default TTL in seconds
            serializer: Serialization method ('pickle' or 'json')
        """
        super().__init__(max_size, default_ttl)
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(exist_ok=True)
        self._serializer = serializer
        self._index_file = self._cache_dir / "cache_index.json"
        self._cache_index: Dict[str, Dict[str, Any]] = {}

    async def _load_index(self) -> None:
        """Load cache index from file."""
        if self._index_file.exists():
            try:
                with open(self._index_file, "r") as f:
                    self._cache_index = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache_index = {}

    async def _save_index(self) -> None:
        """Save cache index to file."""
        try:
            with open(self._index_file, "w") as f:
                json.dump(self._cache_index, f)
        except IOError:
            pass  # Ignore write errors

    def _get_cache_file_path(self, key: Hashable) -> Path:
        """Get file path for a cache key."""
        key_hash = _stable_hash(str(key))
        return self._cache_dir / f"cache_{key_hash}.{self._serializer}"

    async def _serialize_value(self, value: T) -> bytes:
        """Serialize a value for storage."""
        if self._serializer == "json":
            return json.dumps(value, default=str).encode("utf-8")
        else:  # pickle
            return pickle.dumps(value)

    async def _deserialize_value(self, data: bytes) -> T:
        """Deserialize a value from storage."""
        if self._serializer == "json":
            return json.loads(data.decode("utf-8"))
        else:  # pickle
            return pickle.loads(data)

    async def get(self, key: Hashable) -> Optional[T]:
        """Get a value from the cache."""
        async with self._lock:
            await self._load_index()

            key_str = str(key)
            if key_str not in self._cache_index:
                return None

            entry_info = self._cache_index[key_str]

            # Check if expired
            if time.time() - entry_info["timestamp"] > entry_info["ttl"]:
                await self._delete_entry(key_str)
                return None

            # Load from file
            cache_file = self._get_cache_file_path(key)
            if not cache_file.exists():
                # File was deleted externally, clean up index
                await self._delete_entry(key_str)
                return None

            try:
                with open(cache_file, "rb") as f:
                    data = f.read()
                return await self._deserialize_value(data)
            except (IOError, pickle.PickleError, json.JSONDecodeError):
                await self._delete_entry(key_str)
                return None

    async def set(self, key: Hashable, value: T, ttl: Optional[float] = None) -> None:
        """Set a value in the cache."""
        async with self._lock:
            await self._load_index()

            # Remove oldest entries if cache is full
            if len(self._cache_index) >= self._max_size:
                await self._evict_oldest()

            ttl = ttl or self._default_ttl
            key_str = str(key)

            # Serialize and save to file
            cache_file = self._get_cache_file_path(key)
            try:
                serialized_data = await self._serialize_value(value)
                with open(cache_file, "wb") as f:
                    f.write(serialized_data)

                # Update index
                self._cache_index[key_str] = {
                    "timestamp": time.time(),
                    "ttl": ttl,
                    "file": str(cache_file),
                }

                await self._save_index()
            except (IOError, pickle.PickleError, TypeError):
                pass  # Ignore write errors

    async def delete(self, key: Hashable) -> bool:
        """Delete a key from the cache."""
        async with self._lock:
            await self._load_index()
            key_str = str(key)

            if key_str in self._cache_index:
                await self._delete_entry(key_str)
                await self._save_index()
                return True
            return False

    async def _delete_entry(self, key_str: str) -> None:
        """Delete a cache entry (file and index entry)."""
        if key_str in self._cache_index:
            cache_file = Path(self._cache_index[key_str]["file"])
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except OSError:
                    pass  # Ignore file deletion errors

            del self._cache_index[key_str]

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            await self._load_index()

            # Delete all cache files
            for key_str in list(self._cache_index.keys()):
                await self._delete_entry(key_str)

            self._cache_index.clear()
            await self._save_index()

    async def size(self) -> int:
        """Get the current cache size."""
        async with self._lock:
            await self._load_index()
            return len(self._cache_index)

    async def cleanup_expired(self) -> int:
        """Remove all expired entries and return the number removed."""
        async with self._lock:
            await self._load_index()

            current_time = time.time()
            expired_keys = [
                key_str
                for key_str, entry_info in self._cache_index.items()
                if current_time - entry_info["timestamp"] > entry_info["ttl"]
            ]

            for key_str in expired_keys:
                await self._delete_entry(key_str)

            if expired_keys:
                await self._save_index()

            return len(expired_keys)

    async def _evict_oldest(self) -> None:
        """Evict the oldest entry from the cache."""
        if not self._cache_index:
            return

        oldest_key = min(
            self._cache_index.keys(), key=lambda k: self._cache_index[k]["timestamp"]
        )
        await self._delete_entry(oldest_key)

    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            await self._load_index()

            total_entries = len(self._cache_index)
            current_time = time.time()
            expired_entries = sum(
                1
                for entry_info in self._cache_index.values()
                if current_time - entry_info["timestamp"] > entry_info["ttl"]
            )

            # Calculate total cache directory size
            total_size = 0
            try:
                for file_path in self._cache_dir.glob("cache_*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
            except OSError:
                total_size = -1  # Error calculating size

            return {
                "total_entries": total_entries,
                "expired_entries": expired_entries,
                "active_entries": total_entries - expired_entries,
                "max_size": self._max_size,
                "utilization": (
                    total_entries / self._max_size if self._max_size > 0 else 0
                ),
                "cache_dir": str(self._cache_dir),
                "total_size_bytes": total_size,
                "serializer": self._serializer,
            }


def AsyncTTLCache(
    cache_type: Literal["memory", "file"] = "memory",
    cache_dir: str = ".cache",
    max_size: int = 1000,
    default_ttl: float = 300,
    serializer: str = "pickle",
) -> BaseCache[Any]:
    """Factory function to create cache instances."""
    kwargs = {
        "max_size": max_size,
        "default_ttl": default_ttl,
        "serializer": serializer,
        "cache_dir": cache_dir,
    }
    if cache_type == "memory":
        return MemoryCache(**kwargs)
    elif cache_type == "file":
        return FileCache(**kwargs)
    else:
        raise ValueError("Unsupported cache_type. Use 'memory' or 'file'.")


def cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate a cache key from function arguments."""
    key_parts = []

    # Add positional arguments
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            arg_hash = _stable_hash(str(arg), 8)
            key_parts.append(f"hash_{arg_hash}")

    # Add keyword arguments (sorted for consistency)
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(f"{k}={v}")
        else:
            v_hash = _stable_hash(str(v), 8)
            key_parts.append(f"{k}=hash_{v_hash}")

    return "|".join(key_parts)
