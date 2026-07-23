# Document Resolution Layer — Design

**Date:** 2026-07-23
**Status:** Approved, pre-implementation
**Owner:** Jonathan (eng seat)
**Depends on:** frozen snapshot `2026-07-15-ce44ac5`; `goldset` harness (`2026-07-16-goldset-seed-design.md`)

## Problem

`track_responses` puts sophisticated machinery on the **judgment** step (multi-provider clients, Tier 1/Tier 2, grounding) and takes **resolution** — *"which document should I even be reading?"* — on faith from a single field, `final_doc_number`, produced by an upstream join. Hand-labeling 6 gold-set rows showed the errors are almost entirely in resolution, not judgment: given the right document, the answer was usually self-evident.

Three concrete defects, all verified against data:

**1. The frame's premise is false for ~half the population.** The gold-set frame (`lifecycle_stage == FINAL_EFFECTIVE`) was chosen on the premise "a final rule provably exists → a response section exists → the pipeline's 'no' is checkable." In frozen 2024, of 750 `FINAL_EFFECTIVE` rows: **611 (81%) are comments on documents whose FR type is "Proposed Rule"**, and **382 (51%) have no `final_rule_document_number` at all**.

**2. `lifecycle_stage` records agency *intent*, not published fact, and it is time-varying.** It is derived from the Unified Agenda at run time. Traced end-to-end for `EPA-HQ-OAR-2022-0491-0022` (RIN 2060-AV81): frozen data says `FINAL_EFFECTIVE`; the *same code* against today's agenda returns `LONG_TERM_STALLED`. The agenda re-classifies rules each cycle, so a stamped value can be wrong later. Contributing paths in `determine_lifecycle_stage()`: `stage == "FINAL"` → `FINAL_EFFECTIVE` (agenda "Final Rule Stage" means the agency *plans* a final rule, not that one exists); `stage == "COMPLETED"` → `FINAL_EFFECTIVE`; and an `UNKNOWN` fallback that infers `FINAL_EFFECTIVE` from doc_type alone.

**3. The disconfirming evidence is systematically unparseable.** The timetable row that would falsify "a final rule exists" — `Final Rule | To Be Determined` — cannot survive `client.py:411`, whose regex requires a literal `MM/DD/YYYY` in the second cell. The `LONG_TERM` synthetic-entry workaround only fires when the timetable is *entirely empty*, so a rule with a real NPRM row plus a TBD final-rule row silently loses the TBD signal.

**Underneath all three: the pipeline's ontology of where a response can live models exactly one topology** — comment → its document → that document's final rule → response in that rule's preamble. Four real topologies were observed by hand.

## Goal

A standalone resolver that answers **"where could a response to this comment live?"** and returns every candidate it found, with provenance and a three-valued confidence. It does **not** decide whether the agency responded.

**Acceptance:** for each of the six hand-traced fixture rows, the resolver returns the documented expected candidate set, rule classification, and status — including recovering the CMS cross-RIN response and correctly reporting confident absence for EPA Method 320.

## Non-goals

- **No judgment.** Whether a response *addresses* a comment stays downstream (LLM or human).
- **Not wired into `cli.py`.** Standalone, like `freeze` and `goldset`. Wiring into `track_responses` is a later, separately-measured change.
- **No Ruler B.** regulations.gov docket RTC PDFs are out of the declared envelope (`find_docket_rtc_documents()` exists if that changes).
- **No rewrite of the LLM judgment path.** It is not what is failing.

## Observed topologies (the fixture set)

| Fixture row | Topology | Expected resolver outcome |
|---|---|---|
| `NOAA-NMFS-2023-0125-0016` | normal: final rule under same RIN, has responses section | `FOUND`, 1 FINAL candidate |
| `BLM-2024-0001-0003` | direct final rule; no response section possible | `CONFIDENTLY_ABSENT` / `NO_VENUE_POSSIBLE` — a candidate *is* returned but non-qualifying (`rule_class=DIRECT_FINAL`, `has_response_section=False`), which is what explains the absence |
| `FBI-2024-0002-0006` | NPRM-only, no final rule | `CONFIDENTLY_ABSENT` / `RESPONSE_NOT_YET_PUBLISHED` |
| `EPA-HQ-OAR-2022-0491-0022` | no final rule; agenda says "To Be Determined" / Long-Term | `CONFIDENTLY_ABSENT` / `NO_FINAL_RULE_PLANNED`, w/ agenda corroboration |
| `DOT-OST-2024-0090-0049` | packet link points at an **unrelated agency's** rule (FCC); **and** the true final rule has no matching response header | link rejected by relevance check (`AGENCY_MISMATCH`); correct final rule `2024-29990` found via `RIN_SEARCH`; resolves **`FOUND`** despite `response_header_matched == False` (`method=suplinf_full`). **This row is the regression test for "header flag is not a gate."** |
| `CMS-2024-0131-6043` | response in a **later rule under a different RIN** | `FOUND` via `FULLTEXT_SEARCH` on `"1808-IFC"` → `2025-14681` |

## Core principle: absence is not provable

A finite search cannot prove "no response exists anywhere." The design therefore **bounds and calibrates instead of guaranteeing**:

- **Declared envelope.** The five channels below *are* the universe. Results are reported relative to that stated envelope, making them reproducible and falsifiable rather than absolute.
- **Three-valued output**, mirroring the annotator schema (`yes`/`no`/`uncertain`). A boolean would assert something unprovable and reintroduce the silent-failure mode.
- **Calibration over guarantee.** The gold set measures recovery rate; each channel's marginal yield gives an empirical stopping rule.
- **Misses are structured, not random.** Agencies *cite what they respond to*. We cover *linking mechanisms*, a small enumerable space — not "all documents," an unbounded one.

## Discovery channels (ordered precise → wide)

| # | Channel | Notes |
|---|---|---|
| 1 | `PACKET_LINK` | the existing `final_rule_document_number`. **Must pass the relevance check** — DOT's pointed at an FCC rule. |
| 2 | `RIN_SEARCH` | all FR docs under each RIN from `extract_all_rins()`. |
| 3 | `DOCKET_SEARCH` | all FR docs under the docket id. |
| 4 | `AGENDA` | reginfo status; also yields further identifiers (docket #, FR cites). Corroboration only. |
| 5 | `FULLTEXT_SEARCH` | the wide net. **Query identifiers only, never topic words.** |

**Hard constraint on channel 5.** Full-text search must query *identifiers* (docket id, e.g. `"1808-IFC"`, RIN string), never subject terms. Measured: `"Method 320"` returned **83** unrelated rules that merely cite the method; `"1808-IFC"` returned **3**, one exactly right. Identifiers give precision; topics give noise.

## Data contract

```
ResolutionResult
  comment_id, comment_date
  source_document          # the document the comment was filed on
  status                   # FOUND | CONFIDENTLY_ABSENT | UNKNOWN
  absence_reason           # populated ONLY when status == CONFIDENTLY_ABSENT; else None
                           # NO_VENUE_POSSIBLE | RESPONSE_NOT_YET_PUBLISHED | NO_FINAL_RULE_PLANNED
  candidates: [CandidateDocument]
  agenda: AgendaStatus     # stage, timetable (incl. TBD rows), fetched_at
  channels_run: [Channel]  # which ran, which failed/skipped — the envelope, per row
  resolved_at

CandidateDocument
  document_number, publication_date, type, action, title, agency
  rule_class               # FINAL | DIRECT_FINAL | INTERIM_FINAL | CORRECTION
                           # | CONFIRMATION_OF_EFFECTIVE_DATE | PROPOSED | OTHER
  rins[], docket_id
  discovered_by            # Channel
  postdates_comment        # bool
  relevance                # MATCH | AGENCY_MISMATCH | LINEAGE_MISMATCH
  response_evidence        # NONE | WEAK | STRONG  — see below; evidence, NOT a gate
  response_header_matched  # bool | None — extract_response_section().found_response_hd
  response_section_ref
```

### Response evidence is evidence, not a gate

`extract_response_section()` sets `found_response_hd = True` **only** when its `RESP_HD` header regex matches. Its `suplinf_full` and `comment_density` fallbacks return *real* response text with `found_response_hd = False`. Measured against the three rules we hand-verified as genuine responses:

| Rule | `found_response_hd` | method |
|---|---|---|
| NOAA `2024-15931` | `True` | `response_hd` |
| **DOT `2024-29990`** | **`False`** | `suplinf_full` — 68k chars of real response text |
| CMS `2025-14681` | `True` | but matched a *different* section than the one that matters |

DOT is a hand-confirmed genuine response. Gating on the header flag would disqualify it and — with agenda corroboration — manufacture a false `CONFIDENTLY_ABSENT`: **precisely the silent-failure mode this layer exists to eliminate.** The flag is also unreliable in the other direction (CMS matched the wrong section). So:

`response_evidence` is derived from the extract, not from the header flag alone:
- `STRONG` — header matched, **or** grounded text carries substantial comment/response density (reuse `DENSITY_KW`).
- `WEAK` — grounded text exists but density is low.
- `NONE` — no preamble / empty grounded text.

**Status semantics:**
- **Qualifying candidate** = `rule_class == FINAL ∧ postdates_comment ∧ relevance == MATCH`. Response evidence is *not* part of qualification.
- `FOUND` — ≥1 qualifying candidate with `response_evidence != NONE`. **`FOUND` needs only its own evidence** — it does not require every channel to have run, so an early exit after a hit is legal.
- `CONFIDENTLY_ABSENT` — no qualifying candidate, **all five channels ran clean**, and corroboration holds *for the specific `absence_reason`* (below).
- `UNKNOWN` — any channel failed or was skipped, **or** a qualifying candidate exists whose `response_evidence` is `NONE`/`WEAK` (we found the venue but can't read it), **or** a *non-FINAL* candidate postdates the comment with `response_evidence == STRONG` (an interim final rule or "final rule with request for comments" can answer earlier-stage comments in its own preamble — that possibility blocks an absence claim). A failed or missing agenda fetch (`AGENDA_NOT_FOUND`) yields `UNKNOWN`. **Never collapse `UNKNOWN` into absence.**

The all-channels-clean requirement applies **only to absence claims**. Absence is the expensive assertion; presence is not.

**`absence_reason`** (set only when `CONFIDENTLY_ABSENT`) distinguishes absences that mean different things downstream, without multiplying the state machine:

| Value | Meaning | Re-checkable later? |
|---|---|---|
| `NO_VENUE_POSSIBLE` | A rule exists but structurally cannot contain a response — un-withdrawn direct final rule, confirmation-of-effective-date. | No — permanent |
| `RESPONSE_NOT_YET_PUBLISHED` | The responding document is expected but unpublished — an NPRM not yet finalized, or an IFC awaiting a finalizing rule. | **Yes** |
| `NO_FINAL_RULE_PLANNED` | The agenda corroborates that no final rule is scheduled — "To Be Determined" / Long-Term Actions. | Unlikely |

Downstream consumers need this split: `NO_VENUE_POSSIBLE` is a structural true that should arguably leave the FN denominator entirely, while `RESPONSE_NOT_YET_PUBLISHED` is a row worth re-resolving on a later run.

**Corroboration is per-`absence_reason`, not one global agenda clause.** A single "the agenda says TBD/long-term" requirement is wrong: for `NO_VENUE_POSSIBLE` the agenda will show a *completed final action* — the opposite of TBD — so the BLM fixture could never reach its expected status.

| `absence_reason` | What corroborates it |
|---|---|
| `NO_VENUE_POSSIBLE` | The candidate's own structure: `rule_class ∈ {DIRECT_FINAL, CONFIRMATION_OF_EFFECTIVE_DATE}`, not withdrawn, and no later qualifying document under any channel. |
| `RESPONSE_NOT_YET_PUBLISHED` | An NPRM or IFC exists with **no** qualifying successor postdating the comment. |
| `NO_FINAL_RULE_PLANNED` | The **agenda** — final-rule row absent or "To Be Determined", or stage is Long-Term. Requires the `client.py:411` prerequisite fix. |

**Rule classification** derives from the FR `action` field, not `type` (both a plain final rule and a direct final rule have `type == "Rule"`; only `action` distinguishes them). `CONFIRMATION_OF_EFFECTIVE_DATE` and `"Final rule with request for comments"` classify as deferred-response variants, not `FINAL`.

**Chronology rule:** a candidate may only be a response if it **postdates the comment**. Observed failure: the CMS RIN's three Rule-type documents all predate the comment and are therefore all disqualified despite looking like final rules.

**Relevance check (channel 1):** reject a packet link whose agency or docket/RIN lineage does not match the comment's. This is what would have caught the DOT→FCC link.

## Consumers

- **`track_responses`** — replaces the faith-based `final_doc_number` lookup. On `CONFIDENTLY_ABSENT`/`UNKNOWN` it emits a **typed outcome**, never a silent degrade to web-search-with-a-confident-"no."
  **Chosen-candidate policy:** pass **all** qualifying candidates to judgment, ordered earliest-publication-first — do not pre-select one. Picking a single candidate reintroduces a resolution decision that can be wrong (a final rule plus its correction, or two finals under different RINs, are all plausibly the venue), and the judgment step is already reading text. If a caller needs exactly one, the documented default is the earliest qualifying `FINAL`.
- **Gold-set frame** — the corrected frame predicate is a one-liner over the result: `status == FOUND`, replacing `lifecycle_stage == FINAL_EFFECTIVE`, whose premise is false for ~half the population.
  **Planned cost, not a surprise:** the existing seed was drawn from the `FINAL_EFFECTIVE` frame with Horvitz–Thompson weights (`response_sample_weight`, `frame_weight_mass`, `projected_missed`). Changing the frame predicate changes frame *membership*, so the 378-row frame, the 30-row draw, and the weighted projection **do not carry over — a redraw is required.**
  **What does carry over: the labels.** Human labels are per-comment ground truth and remain valid as labeled examples; they simply stop being a *probability sample* of the new frame. So annotation in flight is not wasted and this is **not** a reason to pause the Arvind handoff — only the estimator needs rebuilding.
- **Annotation packet (v2)** — emits **all** candidates with `rule_class`, date, `response_evidence`, and `discovered_by`. **Constraint: the packet must show a superset of what the pipeline used**, or the gold set inherits the pipeline's blind spots and goes blind to the failures it exists to measure.

**Fetch policy (cost control).** `response_evidence` requires `full_text_xml` per candidate, and `RIN_SEARCH`/`DOCKET_SEARCH` can return many documents (one observed umbrella RIN returned 100). XML is therefore fetched **only for candidates surviving chronology + relevance + rule-class filters**, and a **cross-row document cache** is required — many comments on the same rulemaking share the same final rule, so the same document would otherwise be fetched once per comment.

## Reuse vs. build

**Reuse:** `reginfo.client.fetch_unified_agenda()` / `extract_structured_timeline()`; `fr_response_extractor.extract_response_section()`; `federal_register.client` (`extract_all_rins`, `extract_docket_id`, `normalize_docket_id`, fetch by number/citation).

**Build:** FR search by RIN; FR search by docket; FR full-text search; chronology filter; rule classification from `action`; relevance check; the composition + three-valued status.

**Prerequisite bug fix (small, separable):** `reginfo/client.py:411` must capture timetable rows whose date cell is non-numeric (`To Be Determined`), and the `LONG_TERM` synthetic-entry workaround (line ~490) must not be gated on a fully-empty timetable. Without this, `NO_FINAL_RULE_PLANNED` has no corroboration source and is dead.

Two things in that function assume a parsed date and must be handled in the same change: the dedup key `(raw_action.upper(), date_iso)` — still unique with a raw-string date, but verify no collisions when several rows share "To Be Determined" — and the citation-extraction path, which reads the *next* `<td>` after the date cell and must not be skipped for non-date rows. Ships as its own commit ahead of the layer.

## Testing

Two tiers, mirroring the `rtc` parser's approach:

1. **Unit** — pure functions (rule classification from `action`, chronology filter, relevance check, status derivation) against table-driven cases.
2. **Fixture/golden** — the six topology rows above, with recorded API responses committed as fixtures so tests are deterministic and offline. Each pins a distinct topology; together they are the regression suite for the whole ontology.

Live-network tests are `skipif`-gated, never required for CI.

## Open questions (deferred, not blocking)

- Whether `lifecycle_stage` should be **retired** from the pipeline outputs or kept as an advisory field with a "time-varying, not a fact" caveat. Its reproducibility hazard (same comments → different values on re-run) may deserve its own `AUDIT_FINDINGS` entry.
- Whether packet v2 ships resolver candidates in the same change as `attachment_text` (already banked). Packet v2 is where the **goldset-harness walkthrough debt gates**, since it stacks on unwalked packet design decisions.
- Marginal-yield thresholds for the stopping rule — measurable only once gold-set labels exist.
