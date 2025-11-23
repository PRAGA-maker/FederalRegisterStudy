#!/bin/bash

# Set API keys (set these before running or pass as environment variables)
if [ -z "$REGS_API_KEYS" ]; then
    echo "ERROR: REGS_API_KEYS environment variable not set!"
    echo "Set it with: export REGS_API_KEYS='key1:rpm1,key2:rpm2,...'"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable not set!"
    echo "Set it with: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

# Check if year argument provided
if [ -z "$1" ]; then
    echo "Usage: ./run_single_year.sh <year>"
    echo "Example: ./run_single_year.sh 2015"
    exit 1
fi

YEAR=$1

echo "=========================================="
echo "Processing Year: $YEAR"
echo "=========================================="
echo ""

echo "Step 1/4: Fetching and enriching documents..."
python stratification_scripts/2024distribution.py --year $YEAR

echo ""
echo "Step 2/4: Mining comments..."
python stratification_scripts/makeup/mine_comments.py --year $YEAR --concurrent-workers 10

echo ""
echo "Step 3/4: Classifying comment makeup..."
python stratification_scripts/makeup/classify_makeup.py --year $YEAR

echo ""
echo "Step 4/4: Generating plots..."
python stratification_scripts/output/makeup_plots.py --year $YEAR

echo ""
echo "=========================================="
echo "Year $YEAR complete!"
echo "=========================================="

