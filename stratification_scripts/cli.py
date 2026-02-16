"""
Command-line interface for the Federal Register Study pipeline.

This module provides the main CLI entry point, accessible via:
    python -m stratification_scripts [options]
    
Or if installed via pip/uv:
    federal-register [options]

Example:
    # Run full pipeline for 2024
    python -m stratification_scripts --years 2024
    
    # Run specific steps for multiple years
    python -m stratification_scripts --years 2020-2024 --steps fetch,mine
    
    # Run with verbose logging
    python -m stratification_scripts --years 2024 --verbose
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from stratification_scripts.config import PipelineConfig
from stratification_scripts.logging_utils import setup_logging, get_logger
from stratification_scripts.pipeline import run_year, run_many, validate_environment

logger = get_logger(__name__)


def parse_years(years_str: str) -> List[int]:
    """
    Parse a year specification string into a list of years.
    
    Supports:
    - Single year: "2024"
    - Comma-separated: "2020,2021,2022"
    - Range: "2020-2024"
    - Mixed: "2015,2018-2020,2024"
    
    Args:
        years_str: Year specification string
    
    Returns:
        Sorted list of unique years.
    """
    years = set()
    
    for part in years_str.split(","):
        part = part.strip()
        if "-" in part:
            # Range
            start, end = part.split("-", 1)
            try:
                start_year = int(start.strip())
                end_year = int(end.strip())
                years.update(range(start_year, end_year + 1))
            except ValueError:
                pass
        else:
            # Single year
            try:
                years.add(int(part))
            except ValueError:
                pass
    
    return sorted(years)


def parse_steps(steps_str: str) -> List[str]:
    """
    Parse a steps specification string.
    
    Args:
        steps_str: Comma-separated step names
    
    Returns:
        List of valid step names.
    """
    valid_steps = {"fetch", "mine", "deduplicate", "classify", "track_responses", "plot"}
    steps = []
    
    for step in steps_str.split(","):
        step = step.strip().lower()
        if step in valid_steps:
            steps.append(step)
    
    return steps if steps else None


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        prog="federal-register",
        description="Federal Register Study - Analysis pipeline for public comments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run full pipeline for 2024:
    %(prog)s --years 2024
    
  Run for multiple years:
    %(prog)s --years 2020-2024
    
  Run only fetch and mine steps:
    %(prog)s --years 2024 --steps fetch,mine
    
  Validate environment:
    %(prog)s --validate
        """,
    )
    
    # Year selection
    parser.add_argument(
        "--years",
        type=str,
        default="2024",
        help="Years to process. Supports ranges (2020-2024) and lists (2020,2022,2024). Default: 2024",
    )
    
    # Step selection
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated steps to run: fetch,mine,classify,plot. Default: all steps",
    )
    
    # Configuration options
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process (for testing)",
    )
    parser.add_argument(
        "--concurrent-workers",
        type=int,
        default=None,
        help="Number of concurrent API workers",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for classification. Default: gpt-4o-mini",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: limit to 50 documents, same business logic",
    )
    parser.add_argument(
        "--fetch-strategy",
        type=str,
        default="all",
        choices=["all", "stratified"],
        help="Comment sampling strategy. 'all' processes all docs (samples comments only). "
             "'stratified' samples both docs and comments. Default: all",
    )
    
    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output",
    )
    
    # Utility commands
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate environment and exit",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    
    # Validate environment if requested
    if args.validate:
        if validate_environment():
            logger.info("Environment validation passed")
            return 0
        else:
            logger.error("Environment validation failed")
            return 1
    
    # Parse years and steps
    years = parse_years(args.years)
    if not years:
        logger.error(f"No valid years in: {args.years}")
        return 1
    
    steps = parse_steps(args.steps) if args.steps else None
    
    # --test sets default limit of 50; --limit can override
    if args.test and args.limit is None:
        limit_docs = 50
    else:
        limit_docs = args.limit

    # Create base config
    config = PipelineConfig(
        limit_docs=limit_docs,
        test_mode=args.test,
        concurrent_workers=args.concurrent_workers,
        openai_model=args.model,
        fetch_strategy=args.fetch_strategy,
        verbose=args.verbose,
        quiet=args.quiet,
    )
    
    # Run pipeline
    if len(years) == 1:
        config.year = years[0]
        success = run_year(config, steps)
    else:
        success = run_many(years, config, steps)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

