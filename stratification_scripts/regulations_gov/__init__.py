"""
Regulations.gov API client with multi-key rate limiting.

This subpackage provides a robust client for interacting with the regulations.gov
API, including:
- Multi-key rotation for higher throughput
- Per-key rate limiting and cooldown handling
- Thread-safe caching for document and comment data

Example:
    >>> from stratification_scripts.regulations_gov import RegsGovClient
    >>> from stratification_scripts.config import get_regs_api_keys
    >>> 
    >>> keys = get_regs_api_keys()
    >>> with RegsGovClient(keys, retries=5) as client:
    ...     doc = client.get_document_detail("EPA-HQ-2024-0001-0001")
"""

from stratification_scripts.regulations_gov.client import RegsGovClient
from stratification_scripts.regulations_gov.rate_limiter import MultiKeyLimiter

__all__ = [
    "RegsGovClient",
    "MultiKeyLimiter",
]

