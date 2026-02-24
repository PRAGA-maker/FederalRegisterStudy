#!/usr/bin/env python3
"""
Enrich 2025 response data with lifecycle information.

This script:
1. Copies 2025 data files from the dev branch (via git show + LFS)
2. Fetches FR API detail for the ~756 unique docs in the response data
3. Fetches reginfo.gov lifecycle data for docs with RINs
4. Produces an enriched agency_responses_2025.csv with lifecycle columns
5. Produces an enriched mini FR CSV for plotting

Run on the VM (hetzner-prod branch):
    python3 enrich_2025_responses.py
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 0: Extract 2025 data from dev branch via git
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 0: Extract 2025 data from dev branch")
print("=" * 60)

BASE = Path(__file__).parent
DATA_DIR = BASE / "stratification_scripts" / "makeup" / "data"
OUTPUT_DIR = BASE / "stratification_scripts" / "output"

files_to_extract = {
    "agency_responses_2025.csv": DATA_DIR / "agency_responses_2025.csv",
    "makeup_data_2025.csv": OUTPUT_DIR / "makeup_data_2025.csv",
    "federal_register_2025_comments.csv": OUTPUT_DIR / "federal_register_2025_comments.csv",
}

git_paths = {
    "agency_responses_2025.csv": "stratification_scripts/makeup/data/agency_responses_2025.csv",
    "makeup_data_2025.csv": "stratification_scripts/output/makeup_data_2025.csv",
    "federal_register_2025_comments.csv": "stratification_scripts/output/federal_register_2025_comments.csv",
}

for name, dest in files_to_extract.items():
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  {name}: already exists ({dest.stat().st_size:,} bytes), skipping")
        continue

    git_path = git_paths[name]
    print(f"  Extracting {name} from dev branch...")

    # First get the LFS pointer info
    result = subprocess.run(
        ["git", "show", f"dev:{git_path}"],
        capture_output=True, text=True, cwd=str(BASE)
    )

    if result.returncode != 0:
        print(f"  ERROR: git show failed for {git_path}: {result.stderr}")
        sys.exit(1)

    content = result.stdout
    if content.startswith("version https://git-lfs"):
        # It's an LFS pointer — need to smudge it
        print(f"  LFS file detected, smudging...")
        smudge = subprocess.run(
            ["git", "lfs", "smudge"],
            input=content.encode(), capture_output=True, cwd=str(BASE)
        )
        if smudge.returncode != 0:
            print(f"  ERROR: git lfs smudge failed: {smudge.stderr.decode()}")
            sys.exit(1)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(smudge.stdout)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    print(f"  Wrote {dest.stat().st_size:,} bytes to {dest}")

print()

# ---------------------------------------------------------------------------
# Step 1: Load response data, identify unique docs
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Load data and identify docs to enrich")
print("=" * 60)

import polars as pl

resp_path = DATA_DIR / "agency_responses_2025.csv"
makeup_path = OUTPUT_DIR / "makeup_data_2025.csv"

resp = pl.read_csv(str(resp_path), infer_schema_length=None)
makeup = pl.read_csv(str(makeup_path), infer_schema_length=None)

# Filter to real responses (not errors)
real_resp = resp.filter(
    ~pl.col("reasoning").str.starts_with("API error:")
    & ~pl.col("reasoning").str.starts_with("API retries exhausted")
    & ~pl.col("reasoning").str.starts_with("Empty model response")
    & ~pl.col("reasoning").str.starts_with("VALIDATION_ERROR:")
)

unique_docs = sorted(set(real_resp["document_number"].unique().to_list()) - {None, "N/A", ""})
print(f"Total responses: {len(resp):,}")
print(f"Real responses: {len(real_resp):,}")
print(f"Unique document numbers to enrich: {len(unique_docs)}")
print()

# ---------------------------------------------------------------------------
# Step 2: Fetch FR API details for each document
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 2: Fetch FR API details (keyless, ~756 docs)")
print("=" * 60)

import urllib.request
import urllib.error

FR_API_BASE = "https://www.federalregister.gov/api/v1/documents"
FR_FIELDS = [
    "document_number", "type", "title", "agency_names",
    "regulation_id_number_info", "abstract", "topics", "action",
    "significant", "cfr_references",
]

doc_details = {}  # document_number -> dict of enrichment fields
errors = []

# Batch fetch: FR API supports up to 20 doc numbers per request
batch_size = 20
for i in range(0, len(unique_docs), batch_size):
    batch = unique_docs[i:i + batch_size]
    doc_nums_param = ",".join(batch)
    url = f"{FR_API_BASE}/{doc_nums_param}.json?fields[]={'&fields[]='.join(FR_FIELDS)}"

    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "FederalRegisterStudy/1.0")
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except Exception as e:
            if attempt == 2:
                print(f"  FAILED batch {i//batch_size}: {e}")
                errors.extend(batch)
                data = {"results": []} if len(batch) > 1 else None
            else:
                time.sleep(1)
                continue

    # Handle single doc vs multiple doc response
    if isinstance(data, dict) and "results" in data:
        results = data["results"]
    elif isinstance(data, dict) and "document_number" in data:
        results = [data]
    elif isinstance(data, list):
        results = data
    else:
        results = []

    for doc in results:
        doc_num = doc.get("document_number")
        if not doc_num:
            continue

        # Extract RIN from regulation_id_number_info
        rin_info = doc.get("regulation_id_number_info") or {}
        rins = list(rin_info.keys()) if isinstance(rin_info, dict) else []
        rin = rins[0] if rins else None

        # Extract doc_type
        doc_type_raw = doc.get("type", "")
        doc_type_map = {
            "Proposed Rule": "PROPRULE",
            "Rule": "RULE",
            "Notice": "NOTICE",
            "Presidential Document": "PRESDOC",
        }
        doc_type = doc_type_map.get(doc_type_raw, doc_type_raw)

        # CFR references
        cfr_refs = doc.get("cfr_references") or []
        cfr_titles = "; ".join(
            str(r.get("title", "")) for r in cfr_refs if isinstance(r, dict) and r.get("title")
        ) if cfr_refs else None

        doc_details[doc_num] = {
            "rin": rin,
            "doc_type": doc_type,
            "abstract": (doc.get("abstract") or "")[:500] if doc.get("abstract") else None,
            "topics": "; ".join(doc.get("topics") or []) if doc.get("topics") else None,
            "action": doc.get("action"),
            "significant": doc.get("significant"),
            "cfr_titles": cfr_titles,
        }

    if (i // batch_size) % 5 == 0:
        print(f"  Fetched {min(i + batch_size, len(unique_docs))}/{len(unique_docs)} docs...")
    time.sleep(0.3)  # Rate limit

print(f"Successfully enriched: {len(doc_details)} docs")
print(f"Failed: {len(errors)} docs")

# Count RINs found
rins_found = [d["rin"] for d in doc_details.values() if d["rin"]]
print(f"RINs found: {len(rins_found)} ({len(set(rins_found))} unique)")
print()

# ---------------------------------------------------------------------------
# Step 3: Fetch reginfo.gov lifecycle for docs with RINs
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 3: Fetch reginfo.gov lifecycle data (keyless)")
print("=" * 60)

unique_rins = sorted(set(rins_found))
print(f"Unique RINs to look up: {len(unique_rins)}")

rin_lifecycle = {}  # rin -> lifecycle dict

for idx, rin in enumerate(unique_rins):
    url = f"https://www.reginfo.gov/public/do/eAgendaViewRule?pubId=202404&RIN={rin}"

    for attempt in range(2):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0 (compatible; FederalRegisterStudy/1.0)")
            with urllib.request.urlopen(req, timeout=20) as response:
                html = response.read().decode("utf-8", errors="replace")

            # Parse timetable from HTML (simplified — just get stage and action count)
            # Look for the timetable section
            lifecycle_stage = "UNKNOWN"
            timetable_actions = 0

            # Find action rows in the timetable
            action_pattern = re.findall(
                r'<td[^>]*>\s*(NPRM|Final Rule|Final Action|Interim Final Rule|'
                r'Advance Notice of Proposed Rulemaking|Withdrawn|'
                r'Notice of Proposed Rulemaking)\s*</td>',
                html, re.IGNORECASE
            )
            timetable_actions = len(action_pattern)

            # Determine lifecycle stage from last action
            if action_pattern:
                last_action = action_pattern[-1].lower()
                if "withdrawn" in last_action:
                    lifecycle_stage = "WITHDRAWN"
                elif "final" in last_action:
                    lifecycle_stage = "FINAL_EFFECTIVE"
                elif "interim" in last_action:
                    lifecycle_stage = "INTERIM_FINAL"
                elif "nprm" in last_action or "notice of proposed" in last_action:
                    lifecycle_stage = "NPRM_PUBLISHED"
                elif "advance" in last_action:
                    lifecycle_stage = "ANPRM"
                else:
                    lifecycle_stage = "IN_PROGRESS"

            # Check for "not found" pages
            if "No results found" in html or "Unable to find" in html:
                lifecycle_stage = "AGENDA_NOT_FOUND"

            rin_lifecycle[rin] = {
                "unified_agenda_stage": lifecycle_stage,
                "timetable_action_count": timetable_actions,
            }
            break

        except Exception as e:
            if attempt == 1:
                rin_lifecycle[rin] = {
                    "unified_agenda_stage": "FETCH_ERROR",
                    "timetable_action_count": 0,
                }
            time.sleep(1)

    if (idx + 1) % 20 == 0:
        print(f"  Processed {idx + 1}/{len(unique_rins)} RINs...")
    time.sleep(0.5)  # Rate limit reginfo.gov

stage_counts = {}
for data in rin_lifecycle.values():
    s = data["unified_agenda_stage"]
    stage_counts[s] = stage_counts.get(s, 0) + 1
print(f"\nLifecycle stage distribution ({len(rin_lifecycle)} RINs):")
for stage, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
    print(f"  {stage}: {count}")
print()

# ---------------------------------------------------------------------------
# Step 4: Build lifecycle_stage for each document
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 4: Assign lifecycle_stage to documents")
print("=" * 60)

for doc_num, detail in doc_details.items():
    rin = detail.get("rin")
    if rin and rin in rin_lifecycle:
        lc = rin_lifecycle[rin]
        detail["lifecycle_stage"] = lc["unified_agenda_stage"]
        detail["timetable_action_count"] = lc["timetable_action_count"]
    elif rin:
        detail["lifecycle_stage"] = "RIN_NO_LIFECYCLE"
        detail["timetable_action_count"] = 0
    else:
        detail["lifecycle_stage"] = "NO_RIN"
        detail["timetable_action_count"] = 0

# Count
lc_counts = {}
for d in doc_details.values():
    s = d["lifecycle_stage"]
    lc_counts[s] = lc_counts.get(s, 0) + 1
print("Document lifecycle_stage distribution:")
for stage, count in sorted(lc_counts.items(), key=lambda x: -x[1]):
    print(f"  {stage}: {count}")
print()

# ---------------------------------------------------------------------------
# Step 5: Enrich response CSV and save
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 5: Enrich response data and save")
print("=" * 60)

# Build enrichment DataFrame
enrich_rows = []
for doc_num, detail in doc_details.items():
    enrich_rows.append({
        "document_number": doc_num,
        "rin": detail.get("rin") or "N/A",
        "doc_type": detail.get("doc_type") or "UNKNOWN",
        "lifecycle_stage": detail.get("lifecycle_stage", "UNKNOWN"),
        "timetable_action_count": detail.get("timetable_action_count", 0),
        "abstract": detail.get("abstract") or "",
        "action": detail.get("action") or "",
        "significant": str(detail.get("significant") or ""),
    })

df_enrich = pl.DataFrame(enrich_rows, infer_schema_length=None)

# Join with response data
resp_enriched = real_resp.join(df_enrich, on="document_number", how="left")

# Fill nulls for docs that weren't enriched
for col in ["rin", "doc_type", "lifecycle_stage"]:
    if col in resp_enriched.columns:
        resp_enriched = resp_enriched.with_columns(
            pl.col(col).fill_null("UNKNOWN")
        )

# Also join with makeup for commenter type
resp_enriched = resp_enriched.join(
    makeup.select(["comment_id", "category"]),
    on="comment_id",
    how="left",
)

# Save enriched responses
enriched_path = DATA_DIR / "agency_responses_2025_enriched.csv"
resp_enriched.write_csv(str(enriched_path))
print(f"Saved enriched responses: {enriched_path}")
print(f"  Rows: {len(resp_enriched):,}")
print(f"  Columns: {resp_enriched.columns}")

# Also save the mini enriched FR CSV (for plot compatibility)
fr_enriched_path = OUTPUT_DIR / "federal_register_2025_enriched.csv"
df_enrich.write_csv(str(fr_enriched_path))
print(f"Saved enriched FR CSV: {fr_enriched_path}")
print(f"  Rows: {len(df_enrich)}")
print()

# ---------------------------------------------------------------------------
# Step 6: Print summary analysis
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 6: Summary analysis")
print("=" * 60)

found = resp_enriched.filter(pl.col("response_found") == "yes")
print(f"Total real responses: {len(resp_enriched):,}")
print(f"Response found: {len(found):,} ({100*len(found)/len(resp_enriched):.1f}%)")
print()

# By lifecycle stage
print("=== RESPONSE RATE BY LIFECYCLE STAGE ===")
for stage in sorted(resp_enriched["lifecycle_stage"].unique().to_list()):
    if stage is None:
        continue
    stage_data = resp_enriched.filter(pl.col("lifecycle_stage") == stage)
    n = len(stage_data)
    stage_found = stage_data.filter(pl.col("response_found") == "yes")
    n_found = len(stage_found)
    if n == 0:
        continue

    accept = stage_found.filter(pl.col("agency_decision") == "accept").shape[0]
    reject = stage_found.filter(pl.col("agency_decision") == "reject").shape[0]

    print(f"  {stage}:")
    print(f"    N={n:,}, Found={n_found} ({100*n_found/n:.1f}%)", end="")
    if n_found > 0:
        print(f", Accept={accept} ({100*accept/n_found:.1f}%), Reject={reject} ({100*reject/n_found:.1f}%)")
    else:
        print()
print()

# By commenter type + lifecycle
print("=== ACCEPTANCE BY COMMENTER TYPE x LIFECYCLE (found=yes only) ===")
for cat in sorted([c for c in found["category"].unique().to_list() if c is not None]):
    cat_data = found.filter(pl.col("category") == cat)
    n = len(cat_data)
    accept = cat_data.filter(pl.col("agency_decision") == "accept").shape[0]
    reject = cat_data.filter(pl.col("agency_decision") == "reject").shape[0]
    print(f"  {cat}: N={n}, Accept={100*accept/n:.1f}%, Reject={100*reject/n:.1f}%")

    # Break down by lifecycle
    for stage in sorted(cat_data["lifecycle_stage"].unique().to_list()):
        if stage is None or stage == "UNKNOWN":
            continue
        sub = cat_data.filter(pl.col("lifecycle_stage") == stage)
        if len(sub) < 5:
            continue
        sub_accept = sub.filter(pl.col("agency_decision") == "accept").shape[0]
        sub_reject = sub.filter(pl.col("agency_decision") == "reject").shape[0]
        print(f"    {stage}: N={len(sub)}, Accept={100*sub_accept/len(sub):.1f}%, Reject={100*sub_reject/len(sub):.1f}%")
    print()

# Agency coverage
print(f"=== COVERAGE ===")
print(f"Unique docs: {resp_enriched['document_number'].n_unique()}")
print(f"Unique agencies: {resp_enriched['agency'].n_unique()}")
print(f"Unique RINs: {len([r for r in resp_enriched['rin'].unique().to_list() if r not in (None, 'N/A', 'UNKNOWN')])}")
print(f"Docs with lifecycle data: {resp_enriched.filter(~pl.col('lifecycle_stage').is_in(['NO_RIN', 'UNKNOWN'])).shape[0]:,}")
print()
print("DONE. Enriched data saved.")
