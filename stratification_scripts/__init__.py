"""
Federal Register Study - Analysis Pipeline for Public Comments

This package provides tools for:
- Fetching Federal Register documents and their metadata
- Mining public comments from regulations.gov
- Classifying comment authors using LLM-based analysis
- Generating policy-brief visualizations

Main entry points:
    - `python -m stratification_scripts` - CLI interface
    - `stratification_scripts.pipeline.run_year(year)` - Programmatic API

Example:
    >>> from stratification_scripts.pipeline import run_year
    >>> from stratification_scripts.config import PipelineConfig
    >>> config = PipelineConfig()
    >>> run_year(2024, config)
"""

__version__ = "0.1.0"
__author__ = "Federal Register Study Team"

__all__ = [
    "__version__",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "PipelineConfig":
        from stratification_scripts.config import PipelineConfig
        return PipelineConfig
    elif name == "get_regs_api_keys":
        from stratification_scripts.config import get_regs_api_keys
        return get_regs_api_keys
    elif name == "get_openai_api_key":
        from stratification_scripts.config import get_openai_api_key
        return get_openai_api_key
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

