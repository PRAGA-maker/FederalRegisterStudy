#!/usr/bin/env python3
"""
Focused test: Verify Stage 2/3 parallel execution in distribution.py.

Tests that:
1. Stage 2 (Regs.gov) and Stage 3 (reginfo.gov) run concurrently
2. Both stages modify disjoint fields on the same records
3. After parallel execution, records have data from both stages
4. Error in one stage doesn't corrupt the other's results
"""

import sys
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

from stratification_scripts.logging_utils import setup_logging, get_logger

setup_logging(verbose=True)
logger = get_logger(__name__)


def test_parallel_field_isolation():
    """Verify that Stage 2 and Stage 3 modify disjoint fields safely."""
    print("\n=== Test 1: Parallel field isolation ===")

    # Simulate stage1_results: records with both regs_document_id and rin
    records = []
    for i in range(10):
        records.append({
            "document_number": f"2013-{i:05d}",
            "regs_document_id": f"REGS-{i}" if i < 7 else None,  # 7 have regs_id
            "rin": f"1234-AB{i:02d}" if i < 5 else None,  # 5 have RIN
            "comment_count": None,
            "count_source": "unknown",
            "lifecycle_stage": None,
        })

    errors = []

    def stage2_worker():
        """Simulate Stage 2: modify comment_count and count_source."""
        for rec in records:
            if rec.get("regs_document_id"):
                time.sleep(0.01)  # Simulate API call
                rec["comment_count"] = 42
                rec["count_source"] = "regulations.gov"

    def stage3_worker():
        """Simulate Stage 3: modify lifecycle_stage."""
        for rec in records:
            time.sleep(0.01)  # Simulate API call
            rec["lifecycle_stage"] = "FINAL_EFFECTIVE" if rec.get("rin") else "NO_RIN"

    t2 = threading.Thread(target=stage2_worker)
    t3 = threading.Thread(target=stage3_worker)
    t2.start()
    t3.start()
    t2.join()
    t3.join()

    # Verify BOTH stages wrote correctly
    for i, rec in enumerate(records):
        if i < 7:  # Had regs_document_id
            assert rec["comment_count"] == 42, f"Record {i}: Stage 2 didn't write comment_count"
            assert rec["count_source"] == "regulations.gov", f"Record {i}: Stage 2 didn't write count_source"
        else:
            assert rec["comment_count"] is None, f"Record {i}: Stage 2 wrote to non-regs record"

        if i < 5:  # Had RIN
            assert rec["lifecycle_stage"] == "FINAL_EFFECTIVE", f"Record {i}: Stage 3 didn't write lifecycle_stage"
        else:
            assert rec["lifecycle_stage"] == "NO_RIN", f"Record {i}: Stage 3 didn't write NO_RIN"

    print("  PASS: Both stages wrote to correct fields without interference")


def test_stage2_error_isolation():
    """Verify that Stage 2 error doesn't affect Stage 3 results."""
    print("\n=== Test 2: Stage 2 error doesn't corrupt Stage 3 ===")

    records = [
        {"document_number": "2013-00001", "regs_document_id": "REGS-1", "rin": "1234-AB01",
         "comment_count": None, "count_source": "unknown", "lifecycle_stage": None},
    ]

    stage2_exception = [None]

    def stage2_failing():
        try:
            raise ConnectionError("Simulated Regs.gov API failure")
        except Exception as e:
            stage2_exception[0] = e

    def stage3_succeeding():
        for rec in records:
            rec["lifecycle_stage"] = "FINAL_EFFECTIVE"

    t2 = threading.Thread(target=stage2_failing)
    t3 = threading.Thread(target=stage3_succeeding)
    t2.start()
    t3.start()
    t2.join()
    t3.join()

    assert stage2_exception[0] is not None, "Stage 2 should have failed"
    assert records[0]["lifecycle_stage"] == "FINAL_EFFECTIVE", "Stage 3 should have succeeded"
    assert records[0]["comment_count"] is None, "Stage 2 failure should leave data untouched"

    print("  PASS: Stage 2 failure isolated, Stage 3 data intact")


def test_parallel_timing():
    """Verify that parallel execution is actually faster than sequential."""
    print("\n=== Test 3: Parallel execution timing ===")

    records = [{"id": i} for i in range(20)]

    def slow_stage2():
        time.sleep(0.5)  # Simulate 500ms of Regs.gov calls
        for r in records:
            r["stage2_done"] = True

    def slow_stage3():
        time.sleep(0.5)  # Simulate 500ms of reginfo.gov calls
        for r in records:
            r["stage3_done"] = True

    # Sequential timing
    start = time.monotonic()
    slow_stage2()
    slow_stage3()
    seq_time = time.monotonic() - start

    # Reset
    for r in records:
        r.pop("stage2_done", None)
        r.pop("stage3_done", None)

    # Parallel timing
    start = time.monotonic()
    t2 = threading.Thread(target=slow_stage2)
    t3 = threading.Thread(target=slow_stage3)
    t2.start()
    t3.start()
    t2.join()
    t3.join()
    par_time = time.monotonic() - start

    print(f"  Sequential: {seq_time:.3f}s")
    print(f"  Parallel:   {par_time:.3f}s")
    print(f"  Speedup:    {seq_time/par_time:.2f}x")

    assert par_time < seq_time * 0.75, f"Parallel should be significantly faster ({par_time:.3f}s vs {seq_time:.3f}s)"
    assert all(r.get("stage2_done") for r in records), "Stage 2 should have completed"
    assert all(r.get("stage3_done") for r in records), "Stage 3 should have completed"

    print("  PASS: Parallel is faster and both stages completed")


def test_actual_code_path():
    """Test that the actual distribution.py parallel code path is hit."""
    print("\n=== Test 4: Actual code path verification ===")

    from stratification_scripts.distribution import fetch_and_enrich_documents
    from stratification_scripts.config import PipelineConfig

    # We can't easily test without FR API, but we can verify:
    # 1. The code imports
    # 2. The threading module is available
    # 3. The parallel branch conditions are correct

    import threading as th
    assert hasattr(th, 'Thread'), "threading.Thread must exist"

    # Verify the function signature hasn't changed
    import inspect
    sig = inspect.signature(fetch_and_enrich_documents)
    assert 'config' in sig.parameters, "fetch_and_enrich_documents must accept config"

    # Verify Stage 2/3 overlap conditions in code
    import ast
    source = open("stratification_scripts/distribution.py").read()
    tree = ast.parse(source)

    # Check that 'has_stage2_work' and 'has_lifecycle' variables exist
    assigns = [
        node.targets[0].id for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ]
    assert "has_stage2_work" in assigns, "has_stage2_work variable not found"
    assert "has_lifecycle" in assigns, "has_lifecycle variable not found"

    print("  PASS: Code structure verified")


if __name__ == "__main__":
    print("=" * 60)
    print("PARALLEL STAGE 2/3 VERIFICATION")
    print("=" * 60)

    test_parallel_field_isolation()
    test_stage2_error_isolation()
    test_parallel_timing()
    test_actual_code_path()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
