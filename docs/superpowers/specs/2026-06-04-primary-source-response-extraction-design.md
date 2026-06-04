# Design: Primary-source agency-response extraction (structured FR preamble + RTC docs)

**Date:** 2026-06-04 · **Status:** approved design (pending spec review) · **Scope:** Step 5 (`track_responses`) only.

## 1. Problem & goal

The pipeline measures whether federal agencies accept/reject public comments. Today **Step 5** does this two ways, both weak:
- **Tier 1** — Grok web search per comment ("did the agency respond? accept/reject?"). Hallucination-prone: fabricated document numbers, FR start pages ~9pp off.
- **Tier 2** — Gemini compares NPRM vs Final Rule **flat** `raw_text_url` (truncated 50K blob) for the narrow subset with both doc numbers and no Tier-1 hit.

The authoritative agency response lives in the **Final Rule preamble** (`SUPPLEMENTARY INFORMATION`, APA-required reasoned response to significant comments). The FR API already exposes a **structured `full_text_xml_url`** (`<SUPLINF>`/`<HD>` markup) that the pipeline ignores in favor of flat text.

**Goal:** ground response classification on the **primary-source preamble response discussion** (and, where available, standalone "Response to Comments" docs in the regulations.gov docket), using Grok **without web search** when a linked Final Rule exists; keep web search only as a fallback. This is the higher-quality response data identified in `prd/govinfo_probe/FINDINGS_GO_NOGO.md`.

> Investigation outcome (see FINDINGS doc): **GovInfo = NOGO** — its content is byte-identical OFR/GPO XML to what the FR API already serves. The real win is exploiting the FR API's structured XML we already can fetch.

## 2. Approach (user-approved: "A — structured-primary, web-search fallback")

For each sampled comment:
1. Resolve the comment's doc → its linked **Final Rule** doc number (FR-CSV `final_rule_document_number`; or the comment's own doc if it is itself a Rule).
2. **Grounded path** (no web search): if a Final-Rule response section / RTC doc is available, feed Grok `comment_text + grounded_text (+ RTC excerpt)` and classify accept/reject/partial against that primary source, with a citation. `response_source ∈ {fr_preamble, rtc_doc}`.
3. **Fallback path**: no linked Final Rule or extraction failed → existing Grok web-search Tier 1 unchanged. `response_source = web_search`.
4. **Tier 2** (NPRM↔Final) stays as the final backstop; its Final-Rule side reuses the structured extractor instead of the flat 50K blob.

## 2.5 Comment-level granularity — HARD REQUIREMENT

The output **must remain per-comment**, exactly as today: every sampled comment gets its own `agency_responses` row with its own `response_found` / `agency_decision` / `response_text` / `response_citation`. We do **not** emit an aggregate "the agency responded to comments" verdict and stamp it across a docket.

This is the crux of the whole feature. Agencies respond to comment **themes in aggregate** (e.g. Medicare: *"we are not able to acknowledge or respond to them individually"*; they group ~3,500 comments by topic). So the per-comment signal is **inferred at classification time**, not read off the document:

- The extractor's `grounded_text` is **evidence** (the preamble's full comment-response discussion), NOT the answer.
- For each comment, Grok is asked to **match THIS comment's specific concern to the relevant theme-response** in `grounded_text` and classify **the agency's disposition of THIS comment's request** (accept/reject/partial).
- If the preamble does **not** address this comment's specific concern, the correct per-comment output is `response_found = "no"` (or `"uncertain"`) — we must **never** borrow the agency's disposition of a *different* theme and apply it to this comment.

Design consequences:
- `grounded_text` must be **comprehensive** (the whole comment-discussion, via the density-selection / whole-preamble paths) so that the matching theme-response is present for *any* sampled comment — not just an intro or one section.
- `response_text` / `response_citation` must quote the **specific passage** that responds to this comment's concern, not a generic summary.
- Per-comment fan-out is preserved: one Grok call per sampled comment (grounded on the shared, cached preamble). This keeps granularity even though the source is shared.

## 3. Components

### 3.1 NEW `stratification_scripts/makeup/fr_response_extractor.py`
- `extract_response_section(full_text_xml: str) -> ResponseExtract` — the core (validated; see §5). Returns `grounded_text`, `method`, `matched_header`, `citation`, `extraction_method`.
- `find_docket_rtc_documents(docket_id, regs_client) -> list[RTCDoc]` — list docket "Supporting & Related Material", title-filter for RTC patterns (`response to comments`, `RTC`, `summary of comments`), download text via existing `extract_pdf_text`. Bounded + cached. (Sparse — ~1/8 EPA dockets — so optional enrichment, not required for a verdict.)
- Per-(doc) and per-(docket) in-memory caches.

### 3.2 EDIT `federal_register/client.py`
- `fetch_document_full_text_xml(document_number) -> Optional[str]` — construct/lookup `full_text_xml_url` (pattern `…/full_text/xml/{Y}/{M}/{D}/{docnum}.xml`, or via document details), fetch with the existing retry/backoff. Used by the extractor and by Tier 2.

### 3.3 EDIT `xai_response_client.py`
- `track_response_grounded(comment_text, grounding_text, metadata)` + `track_grounded_batch(...)`: same `responses.parse(..., text_format=AgencyResponse)` but `tools=[]` (no web search) and grounded instructions. Reuses the existing retry/error handling and the `AgencyResponse` schema unchanged.

### 3.4 EDIT `gemini_client.py`
- `GROUNDED_RESPONSE_PROMPT` — variant of `RESPONSE_TRACKING_PROMPT`, **per-comment** (see §2.5): replaces "Search the web" with "Below is the agency's official Response-to-Comments discussion from the Final Rule preamble[ and a docket Response-to-Comments document]. Find where (if anywhere) the agency addressed the specific concern raised in THIS comment, and classify the agency's disposition of THIS comment's request (accept/reject/partial). The agency responds to comment themes in aggregate — match this comment to the relevant theme. If the provided text does not address this comment's concern, set `response_found="no"` — do NOT apply the agency's disposition of a different topic to this comment. Quote the specific passage in `response_text`. Use ONLY the provided text." Same `CLASSIFICATION RULES` and output schema (`AgencyResponse`, unchanged).

### 3.5 EDIT `makeup/track_responses.py`
- Pre-pass: from the existing FR-CSV join, collect unique linked Final-Rule doc numbers + dockets; build the extractor/RTC caches.
- Route each sampled comment to grounded vs web-search fallback.
- Tier 2: swap flat fetch → `fetch_document_full_text_xml` + `extract_response_section`.
- New CSV columns (below).

## 4. Schema changes (`agency_responses_{year}.csv`)
Add: `response_source` (`fr_preamble` | `rtc_doc` | `web_search` | `tier2`), `response_citation` (authoritative `vol FR page` + matched head), `rtc_document_id` (nullable). All existing columns unchanged. `save_responses_incremental` already tolerates schema evolution (adds/casts columns), so old CSVs merge cleanly.

## 5. Extraction algorithm (validated on 37-rule heterogeneity sample)

`extract_response_section(xml)`:
1. **Container** = `<SUPLINF>` if present, else `<PREAMB>` (start at the `SUPPLEMENTARY INFORMATION` head). → the primary-source preamble, excluding regulatory/amendatory text. **100% coverage** on sample (incl. DOE/FHWA which lack `<SUPLINF>`).
2. Parse `<HD SOURCE="HDn">` heads + levels. Find the first non-boilerplate head matching a broad response-to-comments pattern; slice to the next **same-or-higher-level head that is not comment-related** (captures comment-by-topic blocks like BIS "Response to Comments" → many same-level "Comments Related to X").
3. **Assemble `grounded_text`:**
   - matched section, if substantial (`≥ POINTER_MIN=2500` chars, guards against pointer heads like Medicare's "Public Comments Received… responses appear elsewhere") and `≤ GROUND_CAP=100,000` → use it (tight, cheap);
   - else if whole preamble `≤ GROUND_CAP` → whole preamble;
   - else (giant rules, e.g. DOE 615K–648K) → **comment-density selection**: split preamble into HD-chunks, score each by comment/response keyword + agency-disposition-verb density, keep the densest chunks up to budget in reading order. This concentrates the budget on the actual adjudication text wherever it lives (deep Section IV/J.3/P in DOE rules).
4. Citation from MODS/`citation` (`{vol} FR {page}`) + matched head.

**Heterogeneity reality** (why this design, not naive slicing): header conventions vary widely — `VII. Response to Comments`, `Summary of Comments and Responses`, `Analysis of Comments`, `Discussion of Comments`, or **woven into** `Discussion of the Final Rule`/section-by-section with no comment header (CFPB/EPA). Some rules legitimately have **no** comment response (corrections, list updates, direct-final, comment-soliciting IFRs) — Grok correctly returns `response_found=no`.

## 6. Cost posture (relevant to the $15 Grok cap on reruns)
- Grounded path runs Grok **without web search** → cheaper per call than current Tier 1; far fewer web searches overall.
- XML + RTC fetches are HTTP-only (no LLM cost), cached per doc/docket.
- `GROUND_CAP=100K chars ≈ 25K input tokens` per grounded call (~$0.005 at Grok-fast input pricing); bounded by response sampling.
- The shared per-rule `grounded_text` is **fetched/extracted once and cached**, then reused across that rule's comments — so the HTTP/extraction cost is per-rule, only the (cheap, no-search) Grok classification is per-comment.
- **NOT pursuing** a "one Grok call per rule that emits all dispositions" batching: it risks collapsing to theme/aggregate output and would violate the per-comment requirement (§2.5). Per-comment fan-out stays.

## 7. Error handling / fallback ladder
`full_text_xml` fetch fail → try `raw_text_url` (stripped) → web-search Tier 1. No `<SUPLINF>`/`<PREAMB>` → raw_text → web search. Every fetch retried (existing backoff); failures logged + cached as null; never crash the run. Idempotent re-runs preserved (existing error-row stripping).

## 8. Testing (TDD)
- Unit (`tests/`): `extract_response_section` on small XML fixtures derived from real samples — response-HD detection, same-level comment-by-topic slicing, pointer-head guard, whole-preamble path, comment-density path on a synthetic giant, PREAMB fallback, no-SUPLINF/no-response. Use the 5 prior **problem docs** (Medicare OPPS pointer, 2×DOE truncation, State intro-only, FRA) as regression fixtures.
- Unit: RTC title-filter; `GROUNDED_RESPONSE_PROMPT` formats.
- Integration: run Step 5 on a mid-N slice; assert `response_source` distribution is sane, grounded rows carry `fr_preamble` citations, no crashes, idempotent re-run.

## 9. Verified evidence
- `prd/govinfo_probe/FINDINGS_GO_NOGO.md` — GovInfo NOGO; structured-XML GO.
- `prd/govinfo_probe/extract_proto.py` + `extract_samples/` — 37-rule heterogeneity run; 100% preamble coverage.
- Adversarial verification workflow (37 docs, escalation ladder, run twice):
  - **Pre-fix:** 13/18 present-response rules captured well; failure modes = truncation mid-response (8, incl. both giant DOE rules) + pointer/wrong-section (1, Medicare OPPS).
  - **Post-fix** (density-selection + `GROUND_CAP=100K` + pointer guard): **19/19 present-response rules captured well, all 19 classifiable, 0 problem docs** (1 residual `truncated_mid_response` flag but still captured adequately). Medicare-OPPS pointer case resolved on 3-vote escalation → captures:yes. 18 of 37 rules correctly judged to have **no** comment-response (corrections/list-updates/IFRs) — not extraction misses.
  - Net: the extractor reliably delivers the primary-source comment-response discussion as Grok evidence across a 2014+2024, ~20-agency heterogeneous sample.
- **Live mid-N e2e (year 2024, --limit 40, real xAI Grok):** 0 API/validation errors. Grounded path hit rate 3/5 (60%) vs web-search 11/35 (31%). Decisions varied (accept 6 / partial 4 / reject 4). **Comment-level granularity confirmed:** doc 2024-00553 produced different dispositions for two different comments (one `no/uncertain` — concern not in the text; one `yes/partial`), proving no aggregate bleed. Grounded rows carry real citations; web-search fallback correctly returns `no` for RFIs/ANPRMs/just-closed NPRMs; Tier 2 (Gemini) processed its eligible pair. **Implementation verified.**

## 10. Files touched
NEW `makeup/fr_response_extractor.py`; EDIT `federal_register/client.py`, `xai_response_client.py`, `gemini_client.py`, `makeup/track_responses.py`. No config flags required (behavior is automatic; a `enable_primary_source_grounding=True` toggle may be added for safety).
