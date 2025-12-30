"""
Custom exceptions for the Federal Register Study pipeline.

This module defines exception classes for different error conditions
that can occur during pipeline execution.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base exception for all pipeline errors."""
    pass


class ConfigurationError(PipelineError):
    """Raised when required configuration is missing or invalid."""
    pass


class APIError(PipelineError):
    """Base exception for API-related errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class AuthenticationError(APIError):
    """Raised when API authentication fails."""
    pass


class DocumentNotFoundError(APIError):
    """Raised when a requested document is not found."""
    pass


class DataError(PipelineError):
    """Raised when there's an issue with data processing."""
    pass


class ValidationError(DataError):
    """Raised when data validation fails."""
    pass

