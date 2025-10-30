#!/usr/bin/env python3
"""
Multi-year Federal Register comment distribution runner.

This script runs the 2024distribution.py script for multiple years between
a start and end year, collecting comment distribution data for each year.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


def run_distribution_script(
    year: int,
    script_path: str,
    output_dir: str,
    additional_args: List[str]
) -> bool:
    """
    Run the distribution script for a specific year.
    
    Args:
        year: The year to process
        script_path: Path to the 2024distribution.py script
        output_dir: Output directory for results
        additional_args: Additional command line arguments to pass through
        
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        sys.executable,
        script_path,
        "--year", str(year),
        "--output-dir", output_dir
    ] + additional_args
    
    print(f"\n{'='*60}")
    print(f"Processing year {year}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        # Ensure the child process receives the exact year via environment as well
        child_env = os.environ.copy()
        child_env["FR_YEAR"] = str(year)
        result = subprocess.run(cmd, check=True, capture_output=False, env=child_env)
        elapsed = time.time() - start_time
        print(f"\n✓ Year {year} completed successfully in {elapsed:.1f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Year {year} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ Year {year} interrupted by user")
        return False
    except Exception as e:
        print(f"\n✗ Year {year} failed with error: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Federal Register comment distribution analysis for multiple years",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run for years 2020-2024 (from repo root)
  python stratification_scripts/output/run_multi_year.py --start 2020 --end 2024
  
  # Run for 2022 only with custom parameters
  python stratification_scripts/output/run_multi_year.py --start 2022 --end 2022 --limit 100
  
  # Run with regulations.gov API key
  python stratification_scripts/output/run_multi_year.py --start 2020 --end 2024 --regs-api-key YOUR_KEY
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--start", 
        type=int, 
        required=True,
        help="Start year (inclusive)"
    )
    parser.add_argument(
        "--end", 
        type=int, 
        required=True,
        help="End year (inclusive)"
    )
    
    # Optional arguments that get passed through to the distribution script
    parser.add_argument("--limit", type=int, help="Limit number of documents per year")
    parser.add_argument("--regs-api-key", type=str, help="Regulations.gov API key")
    parser.add_argument("--regs-rpm", type=int, default=16, help="Regulations.gov requests per minute")
    parser.add_argument("--fr-sleep", type=float, default=0.2, help="Sleep between FR page fetches")
    parser.add_argument("--retries", type=int, default=3, help="Max retries for failed requests")
    
    # Control arguments
    parser.add_argument("--continue-on-error", action="store_true", 
                       help="Continue processing remaining years if one fails")
    parser.add_argument("--delay", type=float, default=1.0, 
                       help="Delay between years (seconds)")
    
    args = parser.parse_args()
    
    # Validate year range
    if args.start > args.end:
        print("Error: Start year must be <= end year")
        sys.exit(1)
    
    if args.start < 2000 or args.end > 2030:
        print("Warning: Year range seems unusual. Proceeding anyway...")
    
    # Find the distribution script (when run from repo root)
    script_path = Path("stratification_scripts/2024distribution.py")
    
    if not script_path.exists():
        print(f"Error: Distribution script not found at {script_path}")
        print("Make sure you're running this script from the repo root directory")
        sys.exit(1)
    
    # Prepare output directory
    output_dir = Path("stratification_scripts/output")
    output_dir.mkdir(exist_ok=True)
    
    # Build additional arguments to pass through
    additional_args = []
    if args.limit is not None:
        additional_args.extend(["--limit", str(args.limit)])
    if args.regs_api_key:
        additional_args.extend(["--regs-api-key", args.regs_api_key])
    additional_args.extend(["--regs-rpm", str(args.regs_rpm)])
    additional_args.extend(["--fr-sleep", str(args.fr_sleep)])
    additional_args.extend(["--retries", str(args.retries)])
    
    # Process each year
    years = list(range(args.start, args.end + 1))
    successful_years = []
    failed_years = []
    
    print(f"Starting multi-year analysis for years {args.start}-{args.end}")
    print(f"Total years to process: {len(years)}")
    print(f"Output directory: {output_dir}")
    print(f"Distribution script: {script_path}")
    
    start_time = time.time()
    
    for i, year in enumerate(years, 1):
        print(f"\n[{i}/{len(years)}] Processing year {year}...")
        
        success = run_distribution_script(
            year=year,
            script_path=str(script_path),
            output_dir=str(output_dir),
            additional_args=additional_args
        )
        
        if success:
            successful_years.append(year)
        else:
            failed_years.append(year)
            if not args.continue_on_error:
                print(f"\nStopping due to failure in year {year} (use --continue-on-error to continue)")
                break
        
        # Add delay between years (except for the last one)
        if i < len(years) and args.delay > 0:
            print(f"Waiting {args.delay} seconds before next year...")
            time.sleep(args.delay)
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("MULTI-YEAR ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Successful years: {len(successful_years)} - {successful_years}")
    print(f"Failed years: {len(failed_years)} - {failed_years}")
    
    if successful_years:
        print(f"\nOutput files created:")
        for year in successful_years:
            csv_file = output_dir / f"federal_register_{year}_comments.csv"
            png_file = output_dir / f"comments_distribution_{year}.png"
            if csv_file.exists():
                print(f"  - {csv_file}")
            if png_file.exists():
                print(f"  - {png_file}")
    
    if failed_years:
        print(f"\n⚠ {len(failed_years)} year(s) failed. Check the output above for details.")
        sys.exit(1)
    else:
        print(f"\n✓ All {len(successful_years)} years completed successfully!")


if __name__ == "__main__":
    main()
