# EPA RTC Crosswalk Parser (Tier-0) — Design

**Date:** 2026-07-18
**Status:** Approved (dispatched eng auto session), pre-implementation
**Owner:** Jonathan (eng seat)
**Depends on:** nothing in the pipeline. Standalone module, mirrors the `freeze`/`goldset` pattern.

## Problem

The pipeline attributes agency responses to comments by reading Federal Register preambles and
web search. For a class of rulemakings the agency instead publishes a dedicated
**Response-to-Comments (RTC) document**: a coded crosswalk that says, authoritatively, *which
comment got which disposition*. Where an RTC exists, it is ground truth for the exact question the
gold set is trying to measure — but it is locked inside a prose PDF built around cross-reference
codes, not a table anyone can join.

The target document is EPA's **Draft CCL 5 Response to Comments** (EPA 815-R-22-001, October 2022,
docket EPA-HQ-OW-2018-0594) — 159 pages, 114 comment excerpts, 22 topics. We want its crosswalk as
structured data: for each comment, its regulations.gov Document ID, the topic(s) EPA filed it
under, and the agency's disposition text for those topics.

This is a **Tier-0** instrument: it reads the agency's own coded record. It is a *new coded-table
parser*, deliberately NOT an extension of the prose response extractor (`makeup/track_responses.py`,
`makeup/fr_response_extractor.py`), which does a fundamentally different thing (LLM reads free
prose). The two must not be conflated.

## Goal

A standalone module that parses the CCL5 RTC PDF end-to-end into a structured per-comment
crosswalk: **Document ID → topic(s) → disposition text**, plus the supporting Exhibit 2 commenter
table and the per-topic Agency Topic Discussions.

**Acceptance:** the CCL5 PDF parses end-to-end into the structured crosswalk with tests green.

## Non-goals

- **No pipeline wiring.** Never imported by `stratification_scripts/cli.py`. Mirrors `goldset`/`freeze`:
  its own `__main__`, invoked only as `python -m stratification_scripts.rtc_parser`.
- **No validation claims.** This session does not assert the parse is *correct* against ground
  truth. The gold-set labels that would grade it are being produced separately today. We assert the
  parser *runs end-to-end and its unit behavior is tested* — not that its topic resolution is right.
- **No generalization beyond CCL5.** The docket prefix is parameterized (default
  `EPA-HQ-OW-2018-0594`) so the code isn't gratuitously hardcoded, but the structural anchors are
  tuned to *this* document's layout. A second RTC doc is a future spec, not a YAGNI abstraction now.
- **No LLM.** The document is a coded table with deterministic textual anchors. Parsing is
  rule-based and deterministic. No model calls.
- **No OCR / scanned-page handling.** The PDF has an embedded text layer (verified).

## Document anatomy (from inspecting the real PDF)

Text extraction (PyMuPDF) is clean and linear. The structural anchors are all robust literal
strings. The document has three parseable structures plus repeating page noise.

**Page noise** (strip before parsing): every page carries a 5-line header
(`EPA-OGWDW` / `Draft CCL 5 Response to Comments` / `EPA 815-R-22-001` / `October 2022` /
`Page N of 159`) and topic sections repeat a running header (`Agency Discussion on <Topic>`,
`Comments Received on <Topic>`).

**(a) Exhibit 2 — commenter table** (pages 11–13). One row per commenter, extracted as sequential
lines: `Commenter Number`, `Document ID` (`EPA-HQ-OW-2018-0594-####`), `First Name`, `Last Name`,
`Organization Name`. Anonymous rows have blank first/last and org `Anonymous`/`Private Citizen`.
Org names may wrap across multiple lines. Parse via a state machine anchored on the Document-ID
regex, with the immediately-preceding integer line as the commenter number.

**(b) Individual Responses — per-comment topic cross-references.** Section 2 presents each comment
as `Comment Excerpt from Commenter <N>` + verbatim excerpt, followed by
`Individual Response: Please see Discussion[s] on <topic list>.` and optional comment-specific
supplemental text. The `<topic list>` is the crosswalk edge: comment N → topic(s).

**Records are per-excerpt, not per-commenter.** There are 114 excerpts but only 52 distinct
commenter numbers (range 51→104): a single commenter (one Document ID) is excerpted separately
under each distinct topic they addressed. So the crosswalk grain is the *excerpt*; `document_id`
(joined from Exhibit 2 by commenter number) repeats across a commenter's excerpts. This is correct —
it is exactly the comment→topic edge list the RTC encodes.

**(c) Agency Topic Discussions — disposition text.** Each of the 22 topics opens with
`Agency Topic Discussion:` followed by EPA's collective disposition prose for that topic. Keyed by
topic name (from the `Agency Discussion on <Topic>` running header that precedes it).

**The 22 topics:** General Comments; Length of CCL 5; Contaminant Groups; Comments on Individual
Chemical Contaminants; Chemical Data/Data Sources; Chemical Technical Support Documents; Comments
Related to Process – Chemicals; Contaminants Not on the Draft CCL 5; Suggestions to Improve the
Process for Future CCLs; Comment Outside the Scope of CCL; Other Drinking Water Programs; Other EPA
Programs; PFAS; DBPs; EDCs and PPCPs; Perchlorate; Pesticides; Cyanotoxins; Draft CCL 5-Microbes;
Microbial Screening Process/Criteria; Legionella pneumophila; Mycobacterium. (Canonical names are
the running-header strings.)

### Two wrinkles that shape the design

1. **Cross-references are NOT verbatim topic names.** They use short forms — `PFAS` for
   "Per- and Polyfluoroalkyl substances (PFAS)", `DBPs` for "Disinfection Byproducts (DBPs)" —
   span line/page breaks mid-list, use Oxford-comma + "and" separators, and sometimes name things
   that are not among the 22 topics at all (e.g. `1,4-Dioxane`, a sub-chemical). Resolving a
   mention to a canonical topic is inherently fuzzy.

2. **114 comment excerpts vs 113 `Individual Response:` blocks** — an off-by-one. At least one
   excerpt has no following Individual Response (or the count is skewed by a near-match). The parser
   must tolerate an excerpt with no response without misaligning every subsequent record.

## Architecture

Four small units, each independently testable. Following the brainstorming isolation principle: the
one io-bound, non-deterministic unit (PDF→text) is thin and isolated; all the real logic is pure
functions over strings.

```
rtc_parser/
  __init__.py
  __main__.py            # python -m stratification_scripts.rtc_parser
  cli.py                 # argparse: `parse`  (mirrors goldset/cli.py shape)
  extract.py             # PDF bytes -> list[str] per-page text (PyMuPDF); the ONLY io/fitz unit
  clean.py               # strip page headers + running headers; join wrapped lines
  exhibit2.py            # cleaned text -> commenter table (number, doc_id, name, org)
  responses.py           # cleaned text -> per-comment excerpts + Individual Response + raw cross-ref
  topics.py              # (i) Agency Topic Discussion blocks by topic; (ii) resolve cross-ref
                         #     mentions -> canonical topics via an explicit alias map
  crosswalk.py           # assemble per-comment records; join responses x exhibit2 x topics
  models.py              # dataclasses: Commenter, CommentRecord, TopicRef, TopicDiscussion
```

**Data flow:** `extract` (PDF→pages) → `clean` (de-noise) → in parallel `exhibit2` (commenters),
`responses` (comment excerpts + cross-ref strings), `topics` (discussion blocks) → `crosswalk`
joins them into per-comment records → `cli` writes outputs.

### Topic resolution (the crux)

`topics.resolve()` takes a raw cross-reference clause (`"General Comments, Length of CCL 5, and
PFAS"`), splits it on `,`/` and `, normalizes each mention, and maps it to a canonical topic via an
**explicit alias table** (`PFAS` → "Per- and Polyfluoroalkyl substances (PFAS)"; `DBPs` →
"Disinfection Byproducts (DBPs)"; etc.) plus case-insensitive exact + substring matching against
the 22 canonical names.

**Fidelity rule (no false precision, no silent drops):** every `TopicRef` carries
`raw` (the verbatim mention), `canonical` (the resolved topic or `None`), and `resolved` (bool).
An unresolvable mention (`1,4-Dioxane`) is **kept** with `resolved=False`, never dropped and never
force-matched. This is deliberate for a Tier-0 prototype with no validation claims: the parser
reports what the document says and flags what it could not confidently map, rather than fabricating
a clean crosswalk. Silent drops or greedy fuzzy-matching would corrupt exactly the signal the gold
set will later grade.

### Off-by-one handling

`responses.py` segments Section 2 into comment blocks anchored on `Comment Excerpt from Commenter
<N>`. Each block's Individual Response is whatever `Individual Response:` clause falls *inside that
block* (before the next `Comment Excerpt`/topic boundary). A block with none yields a record with
`topic_refs=[]` and `has_individual_response=False` — surfaced, not silently aligned to a
neighbor's response.

## Output

Written to `rtc/<doc_slug>/` (top-level sibling of `goldset/`/`frozen/`, via a new
`config.get_rtc_output_path()` helper mirroring `get_goldset_seed_path`):

- **`crosswalk.jsonl`** — the primary artifact, one rich JSON record per comment:
  `{commenter_number, document_id, first_name, last_name, organization, comment_excerpt,
    has_individual_response, topic_refs:[{raw, canonical, resolved}],
    individual_response_supplemental, topic_discussions:{canonical: text}}`.
- **`crosswalk.csv`** — flat, one row per comment with topics joined (`"PFAS; DBPs"`) and an
  `unresolved_topic_refs` column, for spreadsheet eyeballing.
- **`commenters.csv`** — Exhibit 2 as-is (number, document_id, first, last, organization).
- **`topic_discussions.json`** — `{canonical_topic: disposition_text}` for all 22 topics.
- **`parse_manifest.json`** — source PDF sha256 + source URL + page_count + counts
  (commenters, comments, topics, unresolved_refs) — the honest self-report of what the parse found.

## Input storage

The 1.6 MB source PDF is **not committed** (mirrors the freeze pattern: external bytes stay local,
manifest is committed; also avoids LFS friction — git-lfs is not on PATH here). It lives gitignored
under `rtc/inputs/` and is recorded by sha256 + source URL in `parse_manifest.json`. A committed
`.gitignore` entry keeps the bytes out of history.

## Testing

Two tiers, so the committed suite is deterministic and network-free while acceptance is proven
against the real document.

- **Unit tests (committed, deterministic):** synthetic text fixtures — small hand-authored strings
  that reproduce each structure and its edge cases: a 3-row Exhibit 2 (incl. an Anonymous row and a
  wrapped org name); two comment blocks incl. one multi-topic cross-ref spanning a line break and
  one *missing* Individual Response (the off-by-one); a topic-resolution table incl. an
  unresolvable mention. These test `clean`, `exhibit2`, `responses`, `topics`, `crosswalk` as pure
  functions — no PDF, no fitz.
- **Real-slice test (committed):** a small extracted-text fixture taken from the actual PDF (a
  handful of pages spanning Exhibit 2 + one full topic incl. its discussion and several comments),
  committed as text. The parser runs against it and asserts stable counts/joins on real data
  without a binary in the repo.
- **End-to-end acceptance (skip-if-absent):** a test that runs `extract` on the full local PDF and
  asserts the parse yields the expected structure (~54 Exhibit-2 commenters; 114 excerpt records
  across 52 distinct commenter numbers; 22 topics), `@pytest.mark.skipif` when the gitignored PDF
  isn't present. Run live this session to demonstrate acceptance (record counts reported), and green
  in CI via skip.

## Decision Ledger

Adds to the same ledger review banked for the goldset harness. Each entry: choice / why /
rejected+why / what reopens it.

1. **Text state-machine over layout/color detection.** *Why:* extraction is clean and linear;
   robust literal anchors exist (Document-ID regex; `Comment Excerpt from Commenter N`;
   `Individual Response:`; `Agency Topic Discussion:`; `Agency Discussion on <Topic>`; `Please see
   Discussion[s] on …`). *Rejected:* detecting the green/blue header colors via fitz block
   coordinates — brittle, couples to render internals, buys nothing the text anchors don't. *Reopen
   if:* a target RTC lacks these textual anchors and only color distinguishes sections.

2. **Split `extract` (io/fitz) from pure `parse` units.** *Why:* isolation + testability — the
   real logic is pure functions over strings, tested on fixtures; fitz is a thin adapter. *Rejected:*
   monolithic parse-from-bytes — couples the logic to a binary, untestable without PDFs in the repo.
   *Reopen if:* parsing genuinely needs layout coordinates (would force coords through the seam).

3. **Preserve raw cross-ref + explicit alias resolution + `resolved` flag; never drop, never
   force-match.** *Why:* cross-refs use short forms and out-of-list mentions; resolution is fuzzy;
   a Tier-0 prototype with no validation claims must report faithfully and flag uncertainty.
   *Rejected:* verbatim-only matching (drops most cross-refs); greedy fuzzy best-guess (fabricates
   precision into the very signal the gold set will grade). *Reopen if:* gold-set labels show the
   alias map is systematically off — then it's a data-informed fix, not a guess.

4. **PDF gitignored + manifest committed; no binary in git.** *Why:* mirrors freeze's bytes-local
   convention; avoids LFS friction (git-lfs absent). Deterministic tests use committed text
   fixtures. *Rejected:* committing the 1.6 MB PDF (LFS hook friction; binary in history). *Reopen
   if:* CI must run full e2e without a fetch step — then commit an extracted-text snapshot, still
   not the binary.

5. **Four output artifacts (jsonl rich + csv flat + commenters + topic_discussions) + manifest.**
   *Why:* the jsonl is the joinable primary; the csv is for human eyeballing; separating Exhibit 2
   and discussions keeps each unit's output inspectable. *Rejected:* one mega-JSON (hard to diff /
   spot-check). *Reopen if:* a downstream consumer wants a single shape.

6. **Tolerate the 114/113 off-by-one explicitly.** *Why:* a comment excerpt with no Individual
   Response must not shift every later record. *Rejected:* zipping excerpts to responses positionally
   (silent misalignment). *Reopen if:* the true cause is a false-positive anchor match — then fix the
   anchor, keep the tolerance.

7. **Docket prefix parameterized, default `EPA-HQ-OW-2018-0594`.** *Why:* avoids gratuitous
   hardcoding of the Document-ID shape without pretending to generalize the layout. *Reopen if:* a
   second RTC needs different structural anchors — that's a new spec.
