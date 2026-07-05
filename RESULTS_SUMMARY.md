# Federal Register Comment-Makeup & Agency-Response Study — Results Summary

Four fully-processed years (2014, 2015, 2024, 2025): commenter-type composition + agency-response
tracking, weighted to population via Horvitz–Thompson expansion. Plots per year under
`stratification_scripts/makeup/data/plots/<year>/`; per-RIN lifecycle summaries in
`rin_lifecycle_<year>.csv`; response verdicts in `agency_responses_<year>.csv`.

## Funnel (per year)

| Stage | 2014 | 2015 | 2024 | 2025 |
|---|---|---|---|---|
| FR documents | 10,374 | 10,756 | 24,433 | 18,631 |
| comments mined (incl. dups) | 18,356 | 32,340 | 40,498 | 39,872 |
| → canonical (post-dedup) | 15,528 | 26,117 | 26,159 | 28,378 |
| → classified (makeup, incl. dup-prop) | 20,312 | 32,460 | 40,498 | 40,068 |
| → **sampled** for response-tracking | 2,934 | 2,407 | 2,159 | 2,301 |
| regs.gov comment_count universe | 240,451 | 555,249 | 1,240,918 | 1,964,693 |

Classification runs on **all** canonical comments (no document sampling); **response tracking is
sampled** (stratified by commenter category, Cochran + FPC, census ≤30) and weight-expanded back to
the canonical universe.

## Weighted agency-response rates

| Year | overall | grounded (primary-source) | web-search fallback | weighted universe |
|---|---|---|---|---|
| 2014 | 53.1% | 39.6% | 60.5% | 15,528 |
| 2015 | 36.5% | 17.1% | 50.8% | 26,117 |
| 2024 | 35.6% | 39.6% | 34.9% | 26,159 |
| 2025 | 32.5% | 33.4% | 32.4% | 28,378 |

Web-search response rates fall for recent years (2024/2025 rules are newer — fewer finalized
responses published yet). Unique RINs with lifecycle reconstruction: 2014=1,011, 2015=1,028,
2024=1,447, 2025=1,192.

## Methodology notes

- **Response tiers.** Tier 1 = grounded on the Final-Rule preamble (`response_source=fr_preamble`)
  when the comment's document links to a Final Rule; else a web-search fallback
  (`response_source=web_search`). Tier 2 = NPRM-vs-Final text comparison for no-response comments.
- **Weighting (Horvitz–Thompson).** Plots multiply `weight_doc × response_sample_weight`. **Always
  weight** — `agency_responses` is a *sample*; raw row counts are not population estimates.
- **Weight calibration fix.** The response sampler's per-document force-include previously
  double-counted (weighted response N overshot the universe, e.g. 2014 was 1.6×). Post-stratification
  calibration now scales each category's weights to sum to its true size N_cat, so `sum(weights) ==
  canonical universe` exactly (2014=15,528 / 2015=26,117 / 2024=26,159 / 2025=28,378).
- **Cost-aware web-search sampling.** Web-search response calls (the fallback stratum) dominated cost,
  so they are sub-sampled by agency (census small agencies, sample large) and weighted up — unbiased,
  widens the CI on web-derived stats only.
