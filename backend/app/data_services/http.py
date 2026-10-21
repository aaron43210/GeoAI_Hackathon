"""Shared HTTP gateway: one client per process, per-host rate limits, exponential backoff with jitter.

Every external call goes through `HttpGateway.request` so rate limits and retries are enforced in one place,
and every call is recorded (URL, status, latency) for the evidence trail.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urlparse

import httpx

logger = logging.getLogger(__name__)


class UpstreamError(Exception):
    """Raised when an upstream source cannot provide data after retries."""

    def __init__(self, source: str, message: str):
        super().__init__(f"{source}: {message}")
        self.source = source


@dataclass
class HostPolicy:
    min_interval_s: float = 0.0     # minimum spacing between request starts
    max_concurrent: int = 4


class HttpGateway:
    def __init__(self, user_agent: str, policies: Dict[str, HostPolicy], timeout_s: float = 20.0,
                 max_retries: int = 2, backoff_base_s: float = 0.6, backoff_max_s: float = 6.0):
        self.client = httpx.AsyncClient(timeout=timeout_s, headers={"User-Agent": user_agent},
                                        follow_redirects=True)
        self.policies = policies
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last_start: Dict[str, float] = {}

    def _policy(self, host: str) -> HostPolicy:
        return self.policies.get(host, HostPolicy())

    async def _throttle(self, host: str) -> None:
        policy = self._policy(host)
        if policy.min_interval_s <= 0:
            return
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = self._last_start.get(host, 0.0) + policy.min_interval_s - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_start[host] = time.monotonic()

    async def request(self, method: str, url: str, *, source: str, params: Optional[Dict[str, Any]] = None,
                      data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None,
                      timeout_s: Optional[float] = None, retries: Optional[int] = None) -> Any:
        """Return parsed JSON. Retries 429/5xx/timeouts with full-jitter exponential backoff."""
        host = urlparse(url).netloc
        sem = self._semaphores.setdefault(host, asyncio.Semaphore(self._policy(host).max_concurrent))
        attempts = (self.max_retries if retries is None else retries) + 1
        last_err = "unknown error"
        for attempt in range(1, attempts + 1):
            async with sem:
                await self._throttle(host)
                started = time.monotonic()
                try:
                    resp = await self.client.request(method, url, params=params, data=data, headers=headers,
                                                     timeout=timeout_s or httpx.USE_CLIENT_DEFAULT)
                    elapsed = round((time.monotonic() - started) * 1000)
                    logger.info("%s %s %s -> %s in %sms", source, method, host, resp.status_code, elapsed)
                    if resp.status_code == 429 or resp.status_code >= 500:
                        last_err = f"HTTP {resp.status_code}"
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit() and attempt < attempts:
                            await asyncio.sleep(min(float(retry_after), self.backoff_max_s))
                            continue
                    elif resp.status_code >= 400:
                        raise UpstreamError(source, f"HTTP {resp.status_code}: {resp.text[:200]}")
                    else:
                        return resp.json()
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_err = f"{type(exc).__name__}: {exc}"
                    logger.warning("%s %s failed (attempt %s/%s): %s", source, host, attempt, attempts, last_err)
                except ValueError as exc:
                    raise UpstreamError(source, f"invalid JSON: {exc}") from exc
            if attempt < attempts:
                delay = random.uniform(0, min(self.backoff_max_s, self.backoff_base_s * 2 ** (attempt - 1)))
                await asyncio.sleep(delay)
        raise UpstreamError(source, last_err)

    @staticmethod
    def url_with_params(url: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Full URL used as provenance in evidence records."""
        return f"{url}?{urlencode(params, doseq=True)}" if params else url

    async def close(self) -> None:
        await self.client.aclose()
