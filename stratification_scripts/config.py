"""
Centralized configuration management for the Federal Register Study pipeline.

This module provides:
- Environment variable parsing and validation
- Path construction for input/output files
- Runtime configuration via PipelineConfig dataclass

Environment Variables:
    REGS_API_KEYS: Comma-separated API keys with optional RPH limits
                   Format: "KEY1:5000,KEY2:1000" or "KEY1,KEY2" (default 1000 RPH)
    REGS_API_KEY: Legacy single API key (fallback if REGS_API_KEYS not set)
    OPENAI_API_KEY: OpenAI API key for classification

Example:
    >>> from stratification_scripts.config import PipelineConfig, get_regs_api_keys
    >>> 
    >>> # Get API keys from environment
    >>> keys = get_regs_api_keys()
    >>> print(f"Using {len(keys)} API keys")
    >>> 
    >>> # Create pipeline config with custom settings
    >>> config = PipelineConfig(year=2024, concurrent_workers=10)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""
    pass


def get_regs_api_keys(
    default_rph: int = 1000,
    required: bool = False,
) -> List[Tuple[str, int]]:
    """
    Parse Regulations.gov API keys from environment variables.
    
    Reads from REGS_API_KEYS (preferred) or REGS_API_KEY (legacy fallback).
    
    Args:
        default_rph: Default requests per hour for keys without explicit RPH.
        required: If True, raise ConfigurationError when no keys found.
    
    Returns:
        List of (api_key, requests_per_hour) tuples.
    
    Raises:
        ConfigurationError: If required=True and no keys are configured.
    
    Example:
        >>> os.environ["REGS_API_KEYS"] = "key1:5000,key2:1000"
        >>> keys = get_regs_api_keys()
        >>> keys
        [('key1', 5000), ('key2', 1000)]
    """
    keys_with_limits: List[Tuple[str, int]] = []
    
    # Try REGS_API_KEYS first (preferred format)
    api_keys_str = os.environ.get("REGS_API_KEYS", "")
    
    if api_keys_str:
        for pair in api_keys_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            
            if ":" in pair:
                key, rph_str = pair.split(":", 1)
                try:
                    rph = int(rph_str.strip())
                except ValueError:
                    logger.warning(f"Invalid RPH {rph_str!r} for API key, using default {default_rph}")
                    rph = default_rph
                keys_with_limits.append((key.strip(), rph))
            else:
                keys_with_limits.append((pair.strip(), default_rph))
    
    # Fall back to legacy REGS_API_KEY
    if not keys_with_limits:
        legacy_key = os.environ.get("REGS_API_KEY", "")
        if legacy_key:
            keys_with_limits.append((legacy_key.strip(), default_rph))
    
    if required and not keys_with_limits:
        raise ConfigurationError(
            "No Regulations.gov API keys configured. "
            "Set REGS_API_KEYS='key1:5000,key2:1000' or REGS_API_KEY='your-key'"
        )
    
    return keys_with_limits


def get_openai_api_key(required: bool = False) -> Optional[str]:
    """
    Get OpenAI API key from environment.
    
    Args:
        required: If True, raise ConfigurationError when key not found.
    
    Returns:
        API key string or None if not set.
    
    Raises:
        ConfigurationError: If required=True and key is not configured.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if required and not api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it with: export OPENAI_API_KEY='your-key-here'"
        )
    
    return api_key if api_key else None


def get_gemini_api_key(required: bool = False) -> Optional[str]:
    """
    Get Gemini API key from environment.

    Args:
        required: If True, raise ConfigurationError when key not found.

    Returns:
        API key string or None if not set.

    Raises:
        ConfigurationError: If required=True and key is not configured.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")

    if required and not api_key:
        raise ConfigurationError(
            "GEMINI_API_KEY environment variable not set. "
            "Set it with: export GEMINI_API_KEY='your-key-here'"
        )

    return api_key if api_key else None


def get_xai_api_key(required: bool = False) -> Optional[str]:
    """
    Get xAI API key from environment.

    Args:
        required: If True, raise ConfigurationError when key not found.

    Returns:
        API key string or None if not set.

    Raises:
        ConfigurationError: If required=True and key is not configured.
    """
    api_key = os.environ.get("XAI_API_KEY", "")

    if required and not api_key:
        raise ConfigurationError(
            "XAI_API_KEY environment variable not set. "
            "Set it with: export XAI_API_KEY='your-key-here'"
        )

    return api_key if api_key else None


def get_package_root() -> Path:
    """
    Get the root directory of the stratification_scripts package.
    
    Returns:
        Path to stratification_scripts/ directory.
    """
    return Path(__file__).parent


def get_project_root() -> Path:
    """
    Get the root directory of the project (parent of stratification_scripts).
    
    Returns:
        Path to project root directory.
    """
    return Path(__file__).parent.parent


def get_output_dir(year: Optional[int] = None) -> Path:
    """
    Get the output directory path, optionally with year subdirectory.
    
    Args:
        year: If provided, returns path to year-specific subdirectory.
    
    Returns:
        Path to output directory.
    
    Side Effects:
        Creates the directory if it doesn't exist.
    """
    base = get_package_root() / "output"
    if year:
        path = base / str(year)
    else:
        path = base
    
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir(year: Optional[int] = None) -> Path:
    """
    Get the data directory path for raw/intermediate data.
    
    Args:
        year: If provided, used for constructing year-specific paths.
    
    Returns:
        Path to data directory.
    
    Side Effects:
        Creates the directory if it doesn't exist.
    """
    path = get_package_root() / "makeup" / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_plots_dir(year: int) -> Path:
    """
    Get the plots directory path for a specific year.
    
    Args:
        year: Year for which to get plots directory.
    
    Returns:
        Path to plots directory.
    
    Side Effects:
        Creates the directory if it doesn't exist.
    """
    path = get_data_dir() / "plots" / str(year)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_fr_csv_path(year: int) -> Path:
    """
    Get path to Federal Register documents CSV for a year.
    
    Args:
        year: Year to get CSV path for.
    
    Returns:
        Path to federal_register_{year}_comments.csv
    """
    return get_output_dir() / f"federal_register_{year}_comments.csv"


def get_comments_raw_path(year: int) -> Path:
    """
    Get path to raw comments CSV for a year.
    
    Args:
        year: Year to get CSV path for.
    
    Returns:
        Path to comments_raw_{year}.csv
    """
    return get_data_dir() / f"comments_raw_{year}.csv"


def get_makeup_results_path(year: int) -> Path:
    """
    Get path to makeup classification results CSV for a year.
    
    Args:
        year: Year to get CSV path for.
    
    Returns:
        Path to makeup_results_{year}.csv
    """
    return get_data_dir() / f"makeup_results_{year}.csv"


def get_makeup_data_path(year: int) -> Path:
    """
    Get path to final makeup data CSV for a year.
    
    Args:
        year: Year to get CSV path for.
    
    Returns:
        Path to makeup_data_{year}.csv
    """
    return get_output_dir() / f"makeup_data_{year}.csv"


def get_agency_responses_path(year: int) -> Path:
    """
    Get path to agency responses CSV for a year.

    Args:
        year: Year to get CSV path for.

    Returns:
        Path to agency_responses_{year}.csv
    """
    return get_data_dir() / f"agency_responses_{year}.csv"


def get_lifecycle_csv_path(year: int) -> Path:
    """
    Get path to RIN lifecycle summary CSV for a year.

    This CSV contains aggregated lifecycle data keyed by RIN.

    Args:
        year: Year to get CSV path for.

    Returns:
        Path to rin_lifecycle_{year}.csv
    """
    return get_data_dir() / f"rin_lifecycle_{year}.csv"


def get_frozen_dir() -> Path:
    """
    Get the repo-root frozen/ directory holding immutable CSV snapshots.

    Deliberately at the project root (one level above the package data dirs)
    for extra separation from the re-runnable pipeline path.

    Returns:
        Path to frozen/ directory.

    Side Effects:
        Creates the directory if it doesn't exist.
    """
    path = get_project_root() / "frozen"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_frozen_snapshot_path(snapshot_id: str) -> Path:
    """
    Get the directory for a specific frozen snapshot.

    Args:
        snapshot_id: Snapshot identifier, e.g. "2026-07-15-17958e6".

    Returns:
        Path to frozen/<snapshot_id>/.
    """
    return get_frozen_dir() / snapshot_id


@dataclass
class PipelineConfig:
    """
    Configuration for the Federal Register analysis pipeline.
    
    This dataclass holds all runtime configuration options that control
    pipeline behavior. Use this to customize execution without modifying
    environment variables.
    
    Attributes:
        year: Year to process (default: 2024)
        limit_docs: Max documents to process (None = all)
        concurrent_workers: Number of concurrent API workers
        retries: Max retry attempts for API calls
        fetch_strategy: Comment sampling strategy ("all" or "stratified")
        max_comments_per_doc: Skip documents with more comments than this
        openai_model: OpenAI model for classification
        max_concurrency: Max concurrent OpenAI requests
        sample_threshold: Use sampling for docs with more comments
        deduplication_threshold: Cosine similarity threshold for duplicate detection
        deduplication_model: Sentence transformer model for embeddings
        enable_deduplication: Enable comment deduplication feature
        fr_sleep: Sleep between FR page requests
        fr_detail_sleep: Sleep between FR detail requests
        verbose: Enable verbose logging
        quiet: Suppress non-error output
    
    Example:
        >>> config = PipelineConfig(
        ...     year=2024,
        ...     concurrent_workers=10,
        ...     verbose=True,
        ... )
    """
    # General settings
    year: int = 2024
    limit_docs: Optional[int] = None
    verbose: bool = False
    quiet: bool = False
    
    # API settings
    concurrent_workers: Optional[int] = None
    retries: int = 10
    
    # Comment mining settings
    fetch_strategy: str = "all"
    max_comments_per_doc: Optional[int] = None
    
    # Classification settings
    openai_model: str = "gpt-4o-mini"
    max_concurrency: int = 100
    sample_threshold: int = 1000
    sampling_seed: Optional[int] = None
    
    # Deduplication settings
    deduplication_threshold: float = 0.95
    deduplication_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    enable_deduplication: bool = True
    
    # Federal Register API settings
    fr_sleep: float = 0.2
    fr_detail_sleep: float = 0.5
    per_key_hourly: int = 1000
    
    # Response tracking settings
    response_provider: str = "xai"  # "xai", "openai", or "gemini"

    # Primary-source grounding (Final Rule preamble) for Tier 1 (xai provider only).
    # When True, comments whose document links to a Final Rule are classified from the
    # isolated preamble "Response to Comments" section (no web search); others fall back
    # to web-search Tier 1.
    enable_primary_source_grounding: bool = True
    grounded_max_chars: int = 100_000

    # xAI API settings
    xai_model: str = "grok-4-1-fast-reasoning"
    xai_max_concurrency: int = 50

    # Gemini API settings
    gemini_model: str = "gemini-3-flash-preview"
    gemini_max_concurrency: int = 150
    enable_search_grounding: bool = True
    max_comment_pages: int = 30
    # Thinking level for Gemini 3 models. Valid SDK enum values: "LOW", "MEDIUM", "HIGH",
    # "MINIMAL" (case-insensitive). Default None because thinking + Google Search tools +
    # structured JSON output is an unsupported combination in the Gemini API
    # (see googleapis/python-genai#867). Set to a value only if search grounding
    # is disabled or structured output is not used.
    gemini_thinking_level: Optional[str] = None

    # Lifecycle tracking settings
    enable_lifecycle_tracking: bool = True
    unified_agenda_timeout: float = 30.0
    unified_agenda_sleep: float = 1.0
    lifecycle_cache_days: int = 7  # Cache Unified Agenda (updates twice yearly)

    # Derived paths (computed on access)
    _output_dir: Optional[Path] = field(default=None, repr=False)
    _data_dir: Optional[Path] = field(default=None, repr=False)
    
    @property
    def output_dir(self) -> Path:
        """Get output directory, creating if needed."""
        if self._output_dir is None:
            return get_output_dir()
        return self._output_dir
    
    @output_dir.setter
    def output_dir(self, value: Path) -> None:
        self._output_dir = value
    
    @property
    def data_dir(self) -> Path:
        """Get data directory, creating if needed."""
        if self._data_dir is None:
            return get_data_dir()
        return self._data_dir
    
    @data_dir.setter
    def data_dir(self, value: Path) -> None:
        self._data_dir = value
    
    @property
    def fr_csv_path(self) -> Path:
        """Get Federal Register CSV path for configured year."""
        return get_fr_csv_path(self.year)
    
    @property
    def comments_raw_path(self) -> Path:
        """Get raw comments CSV path for configured year."""
        return get_comments_raw_path(self.year)
    
    @property
    def makeup_results_path(self) -> Path:
        """Get makeup results CSV path for configured year."""
        return get_makeup_results_path(self.year)
    
    @property
    def makeup_data_path(self) -> Path:
        """Get final makeup data CSV path for configured year."""
        return get_makeup_data_path(self.year)
    
    @property
    def plots_dir(self) -> Path:
        """Get plots directory for configured year."""
        return get_plots_dir(self.year)
    
    def get_worker_count(self, num_keys: int) -> int:
        """
        Calculate number of workers based on API keys available.
        
        Args:
            num_keys: Number of API keys configured.
        
        Returns:
            Recommended worker count.
        """
        if self.concurrent_workers is not None:
            return self.concurrent_workers
        
        if num_keys > 1:
            return min(8, num_keys * 2)
        elif num_keys == 1:
            return 4
        else:
            return 1

