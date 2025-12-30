#!/bin/bash
# ============================================================================
# Federal Register Study - Single Year Pipeline Runner
# ============================================================================
#
# This script runs the full analysis pipeline for a single year.
# It's a thin wrapper around the Python CLI.
#
# Prerequisites:
#   1. Set REGS_API_KEYS: export REGS_API_KEYS='key1:5000,key2:1000'
#   2. Set OPENAI_API_KEY: export OPENAI_API_KEY='your-key-here'
#   3. Install dependencies: uv sync (or pip install -e .)
#
# Usage:
#   ./run_single_year.sh 2024              # Run full pipeline for 2024
#   ./run_single_year.sh 2024 --verbose    # Run with verbose logging
#   ./run_single_year.sh 2024 --steps fetch,mine  # Run only specific steps
#
# ============================================================================

set -e  # Exit on error

# Check if year argument provided
if [ -z "$1" ]; then
    echo "Usage: ./run_single_year.sh <year> [options]"
    echo ""
    echo "Examples:"
    echo "  ./run_single_year.sh 2024"
    echo "  ./run_single_year.sh 2024 --verbose"
    echo "  ./run_single_year.sh 2024 --steps fetch,mine"
    echo "  ./run_single_year.sh 2024 --limit 100  # Process only 100 docs"
    echo ""
    echo "Options:"
    echo "  --steps <steps>     Comma-separated steps: fetch,mine,classify,plot"
    echo "  --limit <n>         Limit number of documents (for testing)"
    echo "  --verbose           Enable verbose logging"
    echo "  --quiet             Suppress non-error output"
    exit 1
fi

YEAR=$1
shift

# Validate environment
if [ -z "$REGS_API_KEYS" ]; then
    echo "ERROR: REGS_API_KEYS environment variable not set!"
    echo "Set it with: export REGS_API_KEYS='key1:5000,key2:1000,...'"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable not set!"
    echo "Set it with: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

echo "=========================================="
echo "Federal Register Study Pipeline"
echo "Year: $YEAR"
echo "=========================================="
echo ""

# Run the Python pipeline
python -m stratification_scripts --years "$YEAR" "$@"

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo ""
    echo "ERROR: Pipeline failed with exit code $exit_code"
    exit $exit_code
fi

echo ""
echo "=========================================="
echo "Year $YEAR complete!"
echo "=========================================="
