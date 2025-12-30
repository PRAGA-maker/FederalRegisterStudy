#!/bin/bash
# ============================================================================
# Federal Register Study - Multi-Year Pipeline Runner
# ============================================================================
#
# This script runs the full analysis pipeline for multiple years.
# It's a thin wrapper around the Python CLI.
#
# Prerequisites:
#   1. Set REGS_API_KEYS: export REGS_API_KEYS='key1:5000,key2:1000'
#   2. Set OPENAI_API_KEY: export OPENAI_API_KEY='your-key-here'
#   3. Install dependencies: uv sync (or pip install -e .)
#
# Usage:
#   ./run_all_years.sh                    # Run years 2017-2024 (default)
#   ./run_all_years.sh 2020-2024          # Run specific range
#   ./run_all_years.sh --steps fetch,mine # Run only specific steps
#
# ============================================================================

set -e  # Exit on error

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

# Default years if not specified
YEARS="${1:-2017-2024}"
shift 2>/dev/null || true

echo "=========================================="
echo "Federal Register Study Pipeline"
echo "Years: $YEARS"
echo "=========================================="
echo ""

# Run the Python pipeline
python -m stratification_scripts --years "$YEARS" "$@"

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo ""
    echo "ERROR: Pipeline failed with exit code $exit_code"
    exit $exit_code
fi

echo ""
echo "=========================================="
echo "All years processed successfully!"
echo "=========================================="
