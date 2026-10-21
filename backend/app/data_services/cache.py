from typing import Any, Optional, Tuple
import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

_GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash(lat: float, lon: float, precision: int = 7) -> str:
    """Standard geohash encoding (precision 7 ≈ 150 m cells, 6 ≈ 1.2 km, 5 ≈ 5 km)."""
    lat_rng, lon_rng = [-90.0, 90.0], [-180.0, 180.0]
    bits, bit_count, even, out = 0, 0, True, []
    while len(out) < precision:
        rng, val = (lon_rng, lon) if even else (lat_rng, lat)
        mid = (rng[0] + rng[1]) / 2
        if val >= mid:
            bits = (bits << 1) | 1
            rng[0] = mid
        else:
            bits <<= 1
            rng[1] = mid
        even = not even
        bit_count += 1
        if bit_count == 5:
            out.append(_GEOHASH_ALPHABET[bits])
            bits, bit_count = 0, 0
    return "".join(out)


class Cache(ABC):
    """Abstract cache interface."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    async def clear(self) -> bool:
        pass


class InMemoryCache(Cache):
    """In-memory TTL cache with a size bound (oldest entries evicted first)."""

    def __init__(self, max_entries: int = 5000):
        self.store: dict[str, Tuple[float, Any]] = {}
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Optional[Any]:
        item = self.store.get(key)
        if item is None:
            self.misses += 1
            return None
        expiry, value = item
        if time.monotonic() > expiry:
            del self.store[key]
            self.misses += 1
            return None
        self.hits += 1
        return value

    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        if len(self.store) >= self.max_entries:
            oldest = min(self.store, key=lambda k: self.store[k][0])
            del self.store[oldest]
        self.store[key] = (time.monotonic() + ttl, value)
        return True

    async def delete(self, key: str) -> bool:
        return self.store.pop(key, None) is not None

    async def clear(self) -> bool:
        self.store.clear()
        return True
