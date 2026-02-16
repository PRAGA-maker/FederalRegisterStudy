"""
Pipeline orchestration for the Federal Register Study.

This module provides functions to run the full analysis pipeline for
one or more years, coordinating all steps from data fetching to visualization.

Example:
    >>> from stratification_scripts.pipeline import run_year, run_many
    >>> from stratification_scripts.config import PipelineConfig
    >>> 
    >>> # Run for a single year
    >>> config = PipelineConfig(year=2024)
    >>> run_year(config)
    >>> 
    >>> # Run for multiple years
    >>> run_many([2020, 2021, 2022, 2023, 2024])
"""

from __future__ import annotations

import sys
from typing import List, Optional

from stratification_scripts.config import PipelineConfig, get_regs_api_keys, get_openai_api_key
from stratification_scripts.logging_utils import (
    get_logger,
    log_banner,
    log_step,
    log_completion,
    setup_logging,
    set_year_context,
)

logger = get_logger(__name__)


def run_year(
    config: PipelineConfig,
    steps: Optional[List[str]] = None,
) -> bool:
    """
    Run the full pipeline for a single year.
    
    The pipeline consists of 6 steps:
    1. fetch: Fetch and enrich Federal Register documents
    2. mine: Mine comments from regulations.gov
    3. deduplicate: Deduplicate comments using embedding similarity
    4. classify: Classify comment authors using OpenAI
    5. track_responses: Track agency responses using Gemini
    6. plot: Generate visualizations
    
    Args:
        config: Pipeline configuration with year and settings
        steps: Optional list of steps to run. If None, runs all steps.
               Valid values: "fetch", "mine", "deduplicate", "classify", "track_responses", "plot"
    
    Returns:
        True if pipeline completed successfully, False otherwise.
    
    Side Effects:
        Makes HTTP requests to Federal Register and Regulations.gov APIs.
        Makes API calls to OpenAI for classification.
        Makes API calls to Gemini for response tracking.
        Writes CSV files and plot images to disk.
        Logs progress to configured logger.
    """
    year = config.year
    set_year_context(year)
    
    if steps is None:
        steps = ["fetch", "mine", "deduplicate", "classify", "track_responses", "plot"]
    
    total_steps = len(steps)
    current_step = 0
    
    log_banner(logger, f"FEDERAL REGISTER STUDY PIPELINE - YEAR {year}")
    logger.info(f"Steps to run: {', '.join(steps)}")
    
    try:
        # Step 1: Fetch documents
        if "fetch" in steps:
            current_step += 1
            log_step(logger, current_step, total_steps, "Fetching and enriching documents")
            
            from stratification_scripts.distribution import fetch_and_enrich_documents, save_documents
            
            df = fetch_and_enrich_documents(config)
            if len(df) == 0:
                logger.error("No documents fetched")
                return False
            
            save_documents(df, config)
            log_completion(logger, f"Fetched {len(df)} documents")
        
        # Step 2: Mine comments
        if "mine" in steps:
            current_step += 1
            log_step(logger, current_step, total_steps, "Mining comments")
            
            # Verify API keys are available
            api_keys = get_regs_api_keys(required=True)
            logger.info(f"Using {len(api_keys)} Regulations.gov API key(s)")
            
            from stratification_scripts.makeup.mine_comments import mine_comments_for_year
            
            mine_comments_for_year(config)
            log_completion(logger, "Comment mining complete")
        
        # Step 3: Deduplicate comments
        if "deduplicate" in steps:
            current_step += 1
            log_step(logger, current_step, total_steps, "Deduplicating comments")
            
            from stratification_scripts.makeup.deduplicate_comments import deduplicate_comments_for_year
            
            deduplicate_comments_for_year(config)
            log_completion(logger, "Deduplication complete")
        
        # Step 4: Classify comments
        if "classify" in steps:
            current_step += 1
            log_step(logger, current_step, total_steps, "Classifying comment makeup")
            
            # Verify OpenAI key is available
            get_openai_api_key(required=True)
            
            from stratification_scripts.makeup.classify_makeup import classify_comments_for_year
            
            classify_comments_for_year(config)
            log_completion(logger, "Classification complete")
        
        # Step 5: Track agency responses
        if "track_responses" in steps:
            current_step += 1
            log_step(logger, current_step, total_steps, "Tracking agency responses")
            
            # Verify Gemini key is available
            from stratification_scripts.config import get_gemini_api_key
            get_gemini_api_key(required=True)
            
            from stratification_scripts.makeup.track_responses import track_responses_for_year
            
            track_responses_for_year(config)
            log_completion(logger, "Response tracking complete")
        
        # Step 6: Generate plots
        if "plot" in steps:
            current_step += 1
            log_step(logger, current_step, total_steps, "Generating visualizations")
            
            from stratification_scripts.output.makeup_plots import generate_plots_for_year
            
            generate_plots_for_year(config)
            log_completion(logger, "Visualizations complete")
        
        log_completion(logger, f"Year {year} complete!")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed for year {year}: {e}")
        return False


def run_many(
    years: List[int],
    config: Optional[PipelineConfig] = None,
    steps: Optional[List[str]] = None,
) -> bool:
    """
    Run the pipeline for multiple years sequentially.
    
    Args:
        years: List of years to process
        config: Base configuration (year will be overridden for each year).
                If None, uses default PipelineConfig.
        steps: Optional list of steps to run for each year.
    
    Returns:
        True if all years completed successfully, False if any failed.
    
    Side Effects:
        Same as run_year, but for each year in the list.
    """
    if config is None:
        config = PipelineConfig()
    
    log_banner(logger, "FEDERAL REGISTER STUDY - MULTI-YEAR PIPELINE")
    logger.info(f"Years to process: {years}")
    
    success_count = 0
    failed_years = []
    
    for year in years:
        # Create year-specific config
        year_config = PipelineConfig(
            year=year,
            limit_docs=config.limit_docs,
            test_mode=config.test_mode,
            concurrent_workers=config.concurrent_workers,
            retries=config.retries,
            fetch_strategy=config.fetch_strategy,
            max_comments_per_doc=config.max_comments_per_doc,
            openai_model=config.openai_model,
            max_concurrency=config.max_concurrency,
            sample_threshold=config.sample_threshold,
            sampling_seed=config.sampling_seed,
            deduplication_threshold=config.deduplication_threshold,
            deduplication_model=config.deduplication_model,
            enable_deduplication=config.enable_deduplication,
            fr_sleep=config.fr_sleep,
            fr_detail_sleep=config.fr_detail_sleep,
            per_key_hourly=config.per_key_hourly,
            verbose=config.verbose,
            quiet=config.quiet,
        )
        
        if run_year(year_config, steps):
            success_count += 1
        else:
            failed_years.append(year)
    
    # Summary
    log_banner(logger, "PIPELINE COMPLETE")
    logger.info(f"Successful: {success_count}/{len(years)} years")
    
    if failed_years:
        logger.error(f"Failed years: {failed_years}")
        return False
    
    logger.info("All years processed successfully!")
    return True


def validate_environment() -> bool:
    """
    Validate that required environment variables are set.
    
    Returns:
        True if all required variables are set, False otherwise.
    """
    from stratification_scripts.exceptions import ConfigurationError
    
    errors = []
    
    try:
        get_regs_api_keys(required=True)
    except ConfigurationError as e:
        errors.append(str(e))
    
    try:
        get_openai_api_key(required=True)
    except ConfigurationError as e:
        errors.append(str(e))
    
    if errors:
        for error in errors:
            logger.error(error)
        return False
    
    return True

