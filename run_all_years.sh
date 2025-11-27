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

echo "=========================================="
echo "Starting Federal Register Study Pipeline"
echo "Years: 2015-2025"
echo "=========================================="
echo ""

# Array of years to process
YEARS=(2017 2018 2019 2020 2021 2022 2023 2024 2025)

# Process each year
for YEAR in "${YEARS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing Year: $YEAR"
    echo "=========================================="
    echo ""
    
    echo "Step 1/4: Fetching and enriching documents..."
    python stratification_scripts/2024distribution.py --year $YEAR
    if [ $? -ne 0 ]; then
        echo "ERROR: 2024distribution.py failed for year $YEAR"
        exit 1
    fi
    
    echo ""
    echo "Step 2/4: Mining comments..."
    python stratification_scripts/makeup/mine_comments.py --year $YEAR --concurrent-workers 10
    if [ $? -ne 0 ]; then
        echo "ERROR: mine_comments.py failed for year $YEAR"
        exit 1
    fi
    
    echo ""
    echo "Step 3/4: Classifying comment makeup..."
    python stratification_scripts/makeup/classify_makeup.py --year $YEAR
    if [ $? -ne 0 ]; then
        echo "ERROR: classify_makeup.py failed for year $YEAR"
        exit 1
    fi
    
    echo ""
    echo "Step 4/4: Generating plots..."
    python stratification_scripts/output/makeup_plots.py --year $YEAR
    if [ $? -ne 0 ]; then
        echo "ERROR: makeup_plots.py failed for year $YEAR"
        exit 1
    fi
    
    echo ""
    echo "✓ Year $YEAR complete!"
done

echo ""
echo "=========================================="
echo "All years processed successfully!"
echo "=========================================="

