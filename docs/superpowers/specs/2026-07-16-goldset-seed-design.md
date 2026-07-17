# Gold-Set Seed — Design

**Date:** 2026-07-16
**Status:** Approved, pre-implementation
**Owner:** Jonathan (eng seat)
**Depends on:** frozen snapshot `2026-07-15-ce44ac5` (see `2026-07-15-frozen-csv-snapshots-design.md`)

## Problem

Nobody has measured whether the response-attribution labels the pipeline produces are *correct*.
Section *coverage* was validated upstream; per-comment attribution *accuracy* never was. The
headline number we owe is the **web-search false-negative rate**: how often the pipeline says
"no response" when the agency in fact responded. False "no"s bias the published response rates
downward, and web-search-sourced rows carry disproportionate population weight.

To measure it we need a **gold set**: human ground-truth labels on a reproducible sample, made
**blind** to the pipeline's own answer, against a **fixed** artifact (the frozen snapshot).

The seed is the first ~30 labels, produced by Jonathan. It is deliberately small, because its
job is not a publishable rate — it is to (a) prove the labeling schema and instructions survive
contact with real data, (b) give a first directional signal, and (c) hand the annotator a
validated instrument for the scale-up that produces the precise number.

## Goal

A reproducible stratified sample from the frozen 2024 data, exported as a **blind labeling
packet**, plus a **grader** that joins returned labels back to predictions and reports
false-negative rates per source with honest uncertainty.

**Acceptance:** first per-comment accuracy numbers exist, including a first web-search
false-negative estimate, computed from a fixed snapshot by a rerunnable command.

## Non-goals

- **No annotator instructions doc / packet** — that is the separate annotator-package item, and it
  depends on a coordination answer (does the annotator's own research share this schema?).
  This spec only makes the schema *handoff-ready*.
- **No `NO_RIN` frame** — see Frame below; that measurement waits on the docket-first linkage work.
- **No HTML/CLI labeling UI** — the instrument is a spreadsheet, on purpose (see Instrument).
- **No inter-rater agreement stats** — no second labeler exists yet. The overlap flag is recorded
  now; the statistic comes later.
- **Not a pipeline step** — like `freeze`, this is a standalone research/eval tool, never wired
  into `cli.py` or `run_*.sh`.

## Frame

Restricted to rows where the claim "no response" is **checkable**:

```
lifecycle_stage == "FINAL_EFFECTIVE"
  AND response_found == "no"
  AND response_source IN ("web_search", "fr_preamble")
```
→ **378 rows** (web_search 150, fr_preamble 228).

**Why FINAL_EFFECTIVE only.** A final rule provably exists, so a response section exists, so
"was this comment responded to?" is a question a human can definitively answer by reading the
primary source. Rows in other lifecycle stages are either structurally-trivial (`WITHDRAWN`,
`NPRM_CLOSED`, `NPRM_ACTIVE` — no final rule can exist, so "no" is correct by construction) or
require establishing whether a final rule exists at all before labeling can even begin.

**The `NO_RIN` exclusion is a known, deliberate narrowing.** `NO_RIN` is 727 of the 1,014
web_search×no rows (72%) — the largest false-negative mass, and precisely where the Unified
Agenda dependency is suspected of driving underestimates. But each such label requires manual
docket resolution first (does a final rule exist?), which is the docket-first linkage work
parked for a later week. **Consequence to state in every result:** the seed's FN rate describes
the *"final rule provably exists"* subpopulation, **not** the full population, and very likely
not the worst-affected part of it.

## Strata & sampling

- **Strata:** `response_source` — `web_search` (150) and `fr_preamble` (228).
- **Allocation:** n = 15 per stratum (configurable) → ~30 labels.
- **Why both strata:** the research claim is a *contrast* — is the grounded (`fr_preamble`)
  extractor's FN rate actually lower than web-search's? Labeling only web_search would produce a
  number with nothing to compare it to. This contrast is also what the "beat the grounded
  baseline" target is scored on.
- **Seeded RNG.** The seed and the `snapshot_id` are recorded in `sample_manifest.json`. Same
  seed + same snapshot ⇒ identical sample, forever. (The pipeline's unseeded mining is exactly
  the irreproducibility the frozen snapshot exists to escape; the sampler must not reintroduce it.)
- Reads the snapshot via `config.get_frozen_snapshot_path(snapshot_id)` with an **explicit,
  pinned id** — never a "latest" pointer.
- **Overlap flag:** `overlap_candidate` marks rows (default: 10) intended for double-labeling by
  the future annotator, so agreement becomes measurable. Recorded now, used later.

### Weights

Every frame row carries `response_sample_weight` (Horvitz–Thompson). Across the whole 2024 file
these weights sum to 26,159 — the canonical estimand. Within the frame the cell weight mass is:

| cell | rows | weight mass |
|---|---|---|
| web_search × no | 150 | **2,615** (avg ≈ 17.4/row) |
| fr_preamble × no | 228 | 1,719 (avg ≈ 7.5/row) |

web-search "no"s are individually ~2.3× heavier, so their false negatives hit the headline rate
harder than raw counts suggest. The sampler records each sampled row's weight; the grader uses it.

## Instrument: a spreadsheet, deliberately

The annotator scale-up must be packaged as "here's a dataset + plain instructions" — no code. If
Jonathan labels via a bespoke CLI/UI and the annotator labels in a spreadsheet, the schema gets
validated on an instrument nobody else will use, and the learning doesn't transfer. So the seed
uses **the same artifact the annotator will get**: a CSV, labeled in a spreadsheet, re-imported.
The seed *is* the pilot run of the annotator's package.

## The blind packet

`labeling_packet.csv` — one row per sampled comment, containing **only inputs a human needs**:

| column | source | notes |
|---|---|---|
| `label_row_id` | generated | **opaque**; must not encode stratum or position |
| `comment_id` | agency_responses | |
| `comment_text` | comments_raw_2024 | what the commenter actually said |
| `organization`, `submitter_type` | comments_raw_2024 | context |
| `agency`, `title` | agency_responses / FR | which rule |
| `rin` | agency_responses | 100% populated |
| `document_number` | agency_responses | the NPRM |
| `final_rule_document_number` | FR csv | 43% populated |
| `final_action_citation` | FR csv | 52% populated, e.g. `89 FR 102448` |
| `docket_id` | FR csv | **text hint only — never a URL** (see below) |
| `nprm_url`, `rin_url`, `final_rule_url`, `comment_url` | constructed | see Links |
| *(empty, to fill)* `true_response_found` | labeler | `yes` / `no` / `uncertain` |
| *(empty)* `evidence_quote` | labeler | required when `yes` |
| *(empty)* `evidence_citation` | labeler | e.g. `89 FR 102448` / section |
| *(empty)* `true_agency_decision` | labeler | `accept`/`partial`/`reject`/`uncertain` |
| *(empty)* `labeler_notes`, `minutes_spent`, `labeler_id` | labeler | |

### Hidden — never in the packet

`response_found`, `agency_decision`, `reasoning`, `response_text`, `response_location`,
`response_citation`, `rtc_document_id`, `tier2_*`, **and `response_source` itself** (it leaks the
answer path). These live in `prediction_key.csv`, joined only at grade time.

**Anti-anchoring requirements (these protect the ruler's validity):**
1. Model verdicts are absent from the packet entirely — not greyed out, not in a far column: absent.
2. Packet rows are **shuffled so strata interleave** — position must not hint at source.
3. `label_row_id` is **opaque** — it must not encode stratum, index within stratum, or source.

A labeler who can infer the pipeline's answer will unconsciously ratify it, and the gold set
stops being an independent ruler.

### Links (verified against the frame, not assumed)

Coverage was measured on all 378 frame rows before this design was fixed:

- `rin` — **378/378**, all matching the standard `####-A###` RIN pattern.
- `regs_document_id`, `docket_id`, `title` — 378/378.
- `final_rule_document_number` — 162/378 (43%).
- `final_action_citation` — 195/378 (52%).
- `comment_url` (from the FR csv) — **61/378 (16%)** — unusable as-is.
- `comment_id` — 376/378 look like regulations.gov document ids.

Therefore the packet builds links in **layers**, so every row has a working path to the primary
source:

| column | construction | coverage |
|---|---|---|
| `rin_url` | `https://www.federalregister.gov/r/<rin>` | **100%** — the universal fallback; lists the RIN's documents incl. the final rule |
| `nprm_url` | `https://www.federalregister.gov/d/<document_number>` | 100% |
| `final_rule_url` | `https://www.federalregister.gov/d/<final_rule_document_number>` | 43% — the direct shortcut when known, blank otherwise |
| `comment_url` | `https://www.regulations.gov/comment/<comment_id>` | ~99% — blank for the 2 non-conforming ids |

**`docket_id` must NOT be turned into a URL.** Its real values are prose — `Docket No. FAA-2024-1931`,
`FAR Case 2019-015`, `Investigation Nos. 701-TA-684 and 731-TA-1597 (Final)` — not regulations.gov
docket ids. Constructing links from it would produce dead links and waste labeler time. It ships
as a text hint only.

The prefilled links are the friction-killer: they are the difference between ~10 minutes and
~40 minutes per label, and at 30 labels that is the difference between a session and a weekend.

## Grader

`grade` joins `labels_returned.csv` → `prediction_key.csv` on `label_row_id`.

**Validation (fail loud, never compute on bad input):**
- every packet row accounted for; no unknown `label_row_id`s
- `true_response_found` ∈ {yes, no, uncertain}; `true_agency_decision` ∈ {accept, partial, reject, uncertain} or blank
- `true_response_found == "yes"` ⇒ `evidence_quote` non-empty (a "yes" without evidence is not a label, it's a guess)

**Statistics, per `response_source` stratum:**
- **FN rate** = P(true = yes | pred = no) — the estimand, since every frame row has `pred = no`.
- **Unweighted** proportion + **Wilson score 95% CI**.
- **HT-weighted** point estimate: Σ(weight | true=yes) / Σ(weight) within stratum.
- **The contrast:** web_search FN vs fr_preamble FN — the claim under test.
- **Projection ("so what"):** FN_weighted(web_search) × 2,615 = comments in the
  web_search×no×FINAL_EFFECTIVE cell whose responses the pipeline missed.

**Honesty requirements, emitted in the output itself (not just documentation):**
- n = 15/cell ⇒ CI half-width ≈ ±25pp. Results are **directional, not publishable**.
- The weighted point estimate ships **without** a weighted CI — a design-based variance estimator
  for a weighted proportion on n=15 is not credible at this size. Report it as a point estimate
  and say so, rather than manufacturing a false interval.
- Every result restates the frame caveat: this is the *"final rule provably exists"*
  subpopulation, not the 26,159 population.

Outputs `results.json` (machine) + `results.md` (a short table for humans).

## Structure & artifacts

Follows the existing `makeup/` subpackage convention:

```
stratification_scripts/goldset/
  __init__.py
  sample.py    # frame filter, seeded stratified draw, sample manifest
  packet.py    # blind packet + prediction key construction, link building
  grade.py     # label validation, FN rates, CIs, projection, report
  cli.py       # argparse: sample | grade

goldset/<seed-id>/
  sample_manifest.json    # snapshot_id, rng seed, frame, strata, allocation, row ids + weights
  labeling_packet.csv     # blind → spreadsheet
  prediction_key.csv      # hidden verdicts, joined at grade time
  labels_returned.csv     # filled by the labeler, committed
  results.json / results.md
```

`<seed-id>`: `<YYYY-MM-DD>-<snapshot-short>` (e.g. `2026-07-16-ce44ac5`), mirroring the freeze
ID convention.

**CLI** (standalone; `python -m stratification_scripts.goldset`):
- `sample --snapshot <id> [--n 15] [--seed N] [--overlap 10]` → writes manifest + packet + key.
  Refuses to overwrite an existing seed dir.
- `grade <seed-id> --labels <path>` → validates, computes, writes results.

### Artifacts are committed as plain blobs — a `.gitattributes` fix

The labels are the single most valuable artifact this project produces: they are the ruler
everything else is graded by, they cost human hours, and they are irreplaceable. They get
committed.

But `.gitattributes` currently has a blanket `*.csv filter=lfs`, which would push these tiny,
precious, human-edited files through Git-LFS — costing quota against the parent repository, making
them non-diffable, and adding a materialization step to read 30 rows. So this spec adds a
`.gitattributes` override for `goldset/**/*.csv` restoring plain-blob behaviour, **verified with
`git check-attr`** during implementation rather than assumed to work.

Note the contrast with the frozen snapshot: there, CSV bytes are huge and derived, so they are
gitignored and reconstructible from history. Here they are tiny and irreproducible, so they are
committed as text. Different data, opposite call, same reasoning — cost vs. irreplaceability.

## Testing

Fixtures: a small synthetic `agency_responses` / `comments_raw` / FR triple in `tmp_path`;
injected snapshot dir.

1. **Frame filter** selects exactly the FINAL_EFFECTIVE ∧ no ∧ {web_search, fr_preamble} rows.
2. **Seeded determinism:** same seed ⇒ identical sampled `comment_id` set; different seed ⇒ different.
3. **Strata allocation:** n per stratum honoured; requesting n > stratum size fails loudly.
4. **The packet contains none of the forbidden columns** — asserted against an explicit denylist
   incl. `response_source`. *This is the guard that protects the ruler; it is not optional.*
5. **Packet rows are not stratum-ordered** — strata interleave.
6. `label_row_id` is opaque, unique, and stable for a given (seed, snapshot).
7. **Link layering:** `rin_url` present for every row; `final_rule_url` blank exactly when
   `final_rule_document_number` is absent; **no URL is ever constructed from `docket_id`**.
8. **Key/packet join** is total: every `label_row_id` in the packet appears exactly once in the key.
9. **Grader math:** on a fixture with a known planted FN count, unweighted rate is exact; weighted
   rate differs from unweighted when weights are skewed and equals it when weights are uniform.
10. **Grader validation** rejects: unknown ids, missing rows, bad enum values, `yes` without evidence.
11. Wilson CI: known-input sanity (e.g. 0/15 and 15/15 produce bounded, non-degenerate intervals).
12. `sample` refuses to overwrite an existing seed dir.

## Risks / notes

- **Frame narrowness is the main threat to interpretation**, not to correctness. Guarded by
  restating the caveat in every emitted result.
- **Blindness is trust-based within one repo** — `prediction_key.csv` sits next to the packet and
  the labeler could open it. Acceptable for a self-labeled seed; the annotator handoff ships only
  the packet, so blindness is enforced by distribution there.
- **n=15/cell is small by design.** The seed buys a validated instrument and a direction, not a
  number to publish. Stated in the output so it cannot be quoted out of context.
