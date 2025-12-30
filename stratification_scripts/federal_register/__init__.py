"""
Federal Register API client.

This subpackage provides a client for interacting with the Federal Register API
to fetch document metadata and comment information.

Example:
    >>> from stratification_scripts.federal_register import FederalRegisterClient
    >>> client = FederalRegisterClient()
    >>> doc = client.fetch_document_details("2024-12345")
"""

from stratification_scripts.federal_register.client import FederalRegisterClient

__all__ = [
    "FederalRegisterClient",
]

