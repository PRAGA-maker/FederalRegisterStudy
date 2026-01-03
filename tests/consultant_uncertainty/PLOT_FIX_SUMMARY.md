# Plot Fix Summary

## Issue
The `decision_by_type_2025.png` plot was showing ~88% uncertain for Political Consultants, but analysis showed it should be ~20% uncertain (when filtering to only comments with `response_found == "yes"`).

## Root Cause
1. **Join type**: The code was using `left` join, which could include comments without makeup classifications
2. **Missing validation**: The filtering didn't explicitly check for valid categories before calculating percentages
3. **No validation**: No sanity checks to ensure percentages sum correctly

## Fixes Applied

### 1. Changed Join Type (line 2473)
- Changed from `how="left"` to `how="inner"`
- Ensures only comments with BOTH response data AND makeup classification are included
- Prevents counting comments without category assignments

### 2. Added Explicit Filtering (lines 2792-2804)
- Filters to `response_found == "yes"` AND valid category AND valid decision
- Ensures we only count comments that meet all criteria
- Added validation to filter out invalid `agency_decision` values

### 3. Added Diagnostic Logging (lines 2810-2827)
- Logs the number of rows after filtering
- Logs category breakdown for consultants specifically
- Helps verify the fix is working

### 4. Added Sanity Checks (lines 2836-2841)
- Validates that percentages sum to ~100% for each category
- Catches any logic errors early

## Expected Results

After regenerating the plot, consultants should show:
- **Accept**: ~35%
- **Reject**: ~44%
- **Uncertain**: ~20-23% (NOT 88%)

Others should show:
- **Accept**: ~38%
- **Reject**: ~60%
- **Uncertain**: ~2%

## How to Regenerate the Plot

Run the plotting script:
```bash
python -m stratification_scripts.output.makeup_plots --year 2025
```

Or use the pipeline:
```bash
python -m stratification_scripts.pipeline --year 2025 --steps plot
```

## Verification

The test script `test_plot_logic.py` confirms the logic is correct:
- Consultants: 22.9% uncertain (when response_found == "yes")
- Others: 1.7% uncertain (when response_found == "yes")

This matches the expected ~20% vs ~2% difference, confirming the signal is real, not a method artifact.

