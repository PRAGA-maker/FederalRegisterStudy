"""
Multi-key rate limiter for Regulations.gov API.

This module provides thread-safe rate limiting across multiple API keys,
with per-key tracking of RPM limits, hourly quotas, and cooldown periods.

Example:
    >>> from stratification_scripts.config import get_regs_api_keys
    >>> from stratification_scripts.regulations_gov.rate_limiter import MultiKeyLimiter
    >>> 
    >>> keys = get_regs_api_keys()
    >>> limiter = MultiKeyLimiter(keys)
    >>> 
    >>> # Get next available key (blocks if all are rate-limited)
    >>> key = limiter.acquire()
    >>> # ... make API call ...
    >>> 
    >>> # If we get a 429, notify the limiter
    >>> limiter.on_429(key, retry_after=60)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from stratification_scripts.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class KeyState:
    """
    Track state for a single API key.
    
    Attributes:
        key: The API key string
        rpm: Requests per minute limit
        hourly_limit: Requests per hour limit
        used_this_window: Requests made in current hourly window
        window_start: Timestamp when current hourly window started
        next_ready_at: Earliest timestamp when this key can be used
        disabled: Whether this key has been disabled (auth failure)
    """
    key: str
    rpm: int = 16
    hourly_limit: int = 1000
    used_this_window: int = 0
    window_start: float = field(default_factory=time.time)
    next_ready_at: float = 0.0
    disabled: bool = False

    @property
    def min_interval(self) -> float:
        """Minimum seconds between requests for this key's RPM limit."""
        return 60.0 / self.rpm if self.rpm > 0 else 0.0

    def tick(self) -> None:
        """Reset hourly window if 3600 seconds have elapsed."""
        now = time.time()
        if now - self.window_start >= 3600:
            self.window_start = now
            self.used_this_window = 0

    def is_available(self, now: float) -> bool:
        """Check if this key is available for use."""
        if self.disabled:
            return False
        self.tick()
        if self.used_this_window >= self.hourly_limit:
            return False
        return now >= self.next_ready_at

    def get_ready_at(self, now: float) -> float:
        """Get the timestamp when this key will next be available."""
        self.tick()
        if self.disabled:
            return float("inf")
        if self.used_this_window >= self.hourly_limit:
            # Wait until hourly window resets
            return self.window_start + 3600
        return max(self.next_ready_at, now)


class MultiKeyLimiter:
    """
    Fair scheduler over multiple api.data.gov keys.
    
    This class manages rate limiting across multiple API keys, providing:
    - Per-key RPM (requests per minute) limits
    - Per-key hourly quota tracking
    - Retry-after handling for 429 responses
    - Automatic key disabling on auth failures
    
    Thread-safe for use from multiple concurrent workers.
    
    Attributes:
        keys: List of KeyState objects being managed
    
    Example:
        >>> keys = [("key1", 5000), ("key2", 1000)]
        >>> limiter = MultiKeyLimiter(keys)
        >>> 
        >>> # In worker thread:
        >>> key = limiter.acquire()  # Blocks until key available
        >>> try:
        ...     response = requests.get(url, headers={"X-Api-Key": key})
        ...     if response.status_code == 429:
        ...         limiter.on_429(key, int(response.headers.get("Retry-After", 60)))
        ... except AuthError:
        ...     limiter.on_auth_fail(key)
    """
    
    def __init__(self, keys_with_limits: List[Tuple[str, int]]) -> None:
        """
        Initialize with list of (key, requests_per_hour) tuples.
        
        Args:
            keys_with_limits: List of (api_key, rph_limit) tuples.
        
        Raises:
            ValueError: If no API keys provided.
        """
        if not keys_with_limits:
            raise ValueError("No API keys provided")
        
        self._lock = threading.Lock()
        self._keys: List[KeyState] = []
        
        for key, rph in keys_with_limits:
            # Convert requests per hour to requests per minute
            rpm = max(1, int(rph / 60.0))
            self._keys.append(KeyState(
                key=key.strip(),
                rpm=rpm,
                hourly_limit=rph,
            ))
        
        logger.debug(f"Initialized MultiKeyLimiter with {len(self._keys)} keys")

    @property
    def num_active_keys(self) -> int:
        """Number of keys that haven't been disabled."""
        with self._lock:
            return sum(1 for k in self._keys if not k.disabled)

    @property
    def total_rpm(self) -> int:
        """Total requests per minute across all active keys."""
        with self._lock:
            return sum(k.rpm for k in self._keys if not k.disabled)

    def acquire(self) -> str:
        """
        Block until a key is available and return it.
        
        This method will block if all keys are rate-limited, waiting until
        at least one key becomes available.
        
        Returns:
            API key string ready for use.
        
        Note:
            The key's usage counters are updated when acquired.
            If you encounter a 429 response, call on_429() to push
            the key's next_ready_at forward.
        """
        while True:
            with self._lock:
                now = time.time()
                best: Optional[KeyState] = None
                best_ready = float("inf")
                
                for ks in self._keys:
                    ks.tick()
                    if ks.disabled:
                        continue
                    
                    # If hourly budget exhausted, set next_ready to end of window
                    if ks.used_this_window >= ks.hourly_limit:
                        wait = ks.window_start + 3600 - now
                        ks.next_ready_at = max(ks.next_ready_at, now + max(wait, 0.0))
                    
                    ready_at = max(ks.next_ready_at, now)
                    if ready_at < best_ready:
                        best_ready, best = ready_at, ks

                if best is None:
                    # All disabled - log warning and wait
                    active_count = sum(1 for k in self._keys if not k.disabled)
                    if active_count == 0:
                        logger.warning(
                            f"All {len(self._keys)} API keys have been disabled "
                            "due to auth failures"
                        )
                    sleep_for = 5.0
                else:
                    sleep_for = max(0.0, best_ready - now)

                if sleep_for <= 0.0 and best is not None:
                    # We can use this key now
                    best.used_this_window += 1
                    # Advance RPM gate
                    best.next_ready_at = now + best.min_interval
                    return best.key

            # Sleep outside the lock
            time.sleep(min(sleep_for, 60.0))

    def on_429(self, key: str, retry_after_seconds: Optional[int] = None) -> None:
        """
        Handle 429 response by pushing key's next_ready_at forward.
        
        Args:
            key: The API key that received the 429.
            retry_after_seconds: Value from Retry-After header, if present.
        """
        with self._lock:
            for ks in self._keys:
                if ks.key == key:
                    wait = retry_after_seconds or 60
                    wait = min(max(wait, 1), 120)  # Clamp to 1-120 seconds
                    ks.next_ready_at = max(ks.next_ready_at, time.time() + wait)
                    # Back off RPM a bit after a 429 to be gentle
                    ks.rpm = max(1, int(ks.rpm * 0.9))
                    logger.debug(
                        f"Key {key[:8]}... rate limited, "
                        f"wait {wait}s, new RPM {ks.rpm}"
                    )
                    return

    def on_auth_fail(self, key: str) -> None:
        """
        Permanently disable a key that failed authentication.
        
        Args:
            key: The API key that failed auth (401/403).
        """
        with self._lock:
            for ks in self._keys:
                if ks.key == key:
                    ks.disabled = True
                    logger.warning(f"API key {key[:8]}... disabled due to auth failure")
                    return

    def get_stats(self) -> dict:
        """
        Get current statistics about key usage.
        
        Returns:
            Dict with keys: num_keys, active_keys, total_rpm, keys_stats
        """
        with self._lock:
            stats = {
                "num_keys": len(self._keys),
                "active_keys": sum(1 for k in self._keys if not k.disabled),
                "total_rpm": sum(k.rpm for k in self._keys if not k.disabled),
                "keys_stats": [],
            }
            
            for i, ks in enumerate(self._keys):
                stats["keys_stats"].append({
                    "index": i,
                    "rpm": ks.rpm,
                    "hourly_limit": ks.hourly_limit,
                    "used_this_window": ks.used_this_window,
                    "disabled": ks.disabled,
                })
            
            return stats

