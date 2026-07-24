# The Comment Agent & Its Fuel Supply Chain — Design

**Date:** 2026-07-24
**Status:** Approved in design session (Jonathan), pre-implementation
**Owner:** Jonathan (design); build ownership split per phase (§8)
**Companion specs:** `2026-07-23-document-resolution-layer-design.md` (built 2026-07-24, 13/13), `2026-07-16-goldset-seed-design.md`, `2026-07-18-epa-rtc-crosswalk-parser-design.md`
**Supersedes:** the audit's R3 web-search compromise (`AUDIT_FINDINGS.md`) — see §6 row 5.

## 1. Thesis and the honest value claim

The product is an AI agent that helps org staff write public comments agencies are legally obligated to engage with. Its value claim, verbatim from the design session:

> **The agent doesn't "improve" your comment. It maps your draft against the standard agencies are legally held to, and shows you where it falls short.**

That claim is verifiable (the rubric is inspectable, every dimension cites doctrine), requires no ML validation to make truthfully, and survives the worst-case research outcome. Claims discipline: doctrine licenses *"the agency is legally obligated to engage a comment like this"* — never *"your comment will get a response."* Nothing partner-facing is phrased as "predictive" until measured with error bars.

**Primary user (v1):** org government-relations staff (1–2 person teams, capacity-starved, already submit member-representing comments). Ordinary-citizen mode is a later variant.

## 2. The generation firewall (named invariant)

**The agent never produces comment prose.** Motivations: (a) gen-AI backlash protection; (b) efficacy — agencies dedupe and discount template text, so AI-written comments self-defeat; (c) the product must stay on the right side of its own thesis (the mass-fake-comment literature, §5 sources). Enforced at the interface: the agent's response schema has no draft-text field. Its output vocabulary is exactly four types:

1. **Questions** (elicitation, evidence prompts)
2. **Structure** (outline scaffolds: headings + per-section guidance, never filled in)
3. **Critique** (dimension-scored review of the user's text)
4. **Retrieved excerpts with attribution** (real statutes, real rule sections, real comments — shown, never paraphrased into the draft)

Boundary case, decided: **proposed regulatory text** stays behind the firewall too. The agent shows real amendment examples and points at the section to amend; the operative words are the user's.

Partner/press sentence: *"No AI ever writes a word of your comment."*

## 3. The agent loop

1. **Ingest** — rulemaking context (NPRM full text, deadline, docket comments) + org profile.
2. **Elicit** — staffer states position; agent asks clarifying questions only.
3. **Plan** — maps concerns to argument types with doctrine leverage (statutory objection, empirical critique, procedural challenge, concrete alternative); proposes an outline (headings + guidance, no prose). Structure is guidance, never a locked form. Rationale for headings: agencies respond by theme; sectioned comments pre-sort arguments into the buckets the agency's response process uses (cf. RTC crosswalk structure).
4. **Write (human)** — section by section; per-section prompts + one attributed real example. The agent never fills a section.
5. **Review** — the critic scores the draft per dimension with actionable feedback; iterate.
6. **Deliver** — the user's comment + submission metadata. (Submit/track: sketched, §9.)

## 4. The critic

Scores a draft *absent / weak / strong* per dimension, justifies by quoting the user's text, fixes via question or retrieved example (firewall applies to feedback).

| Dimension | Checks | Backing |
|---|---|---|
| Section anchoring | Each argument names the provision it targets | Doctrine (specificity = significance) |
| Evidence grounding | Data/citations/member experience, not assertion | Doctrine; *magnitude* is hypothesis |
| Statutory hook | Objection tied to authorizing statute | Doctrine, sharpened post-*Loper Bright* |
| Procedural validity | In scope of the NPRM, timely | Doctrine |
| Concrete alternative | Specific modification, ideally proposed reg text | Doctrine (*State Farm*) |
| Materiality | Would adopting it change the rule? | Doctrine-adjacent (the "cheerleading" failure mode) |
| Originality | Distance from corroborated campaign clusters | **Own-data-backed** (Fig-6 direction), pending calibration |
| Structure | Headed, numbered, theme-sortable | Hypothesis (H2), flagged as such in UI |

**Mechanics:** LLM judge, structured output, pinned model, rubric as versioned data (`rubric.yaml`), every scorecard logs judge+rubric versions.

**Judge-evaluation gate (non-negotiable):** a small human-scored set measures judge–human agreement per dimension before any partner sees scores. Dimensions that fail demote to "advisory."

**Two-number display (v2):** doctrine score (stable, explainable) + calibrated signal (measured engagement, with CIs) — never collapsed; the calibrated number appears only for dimensions that cleared the data bar.

## 5. Evidence status & validation plan

| Claim | Status |
|---|---|
| Agencies must engage significant comments; ignored alternatives risk vacatur | Real doctrine (APA §553/§706, *State Farm*, *Ohio v. EPA*) |
| Agencies respond by theme | Solid (hand-tracing, RTC structure, admin-law literature) |
| Original (non-template) comments do better | Measured-but-uncalibrated (team Fig 6; rides the broken pipeline) |
| Specificity/evidence/hooks predict engagement | Not measured by us; literature confounded with identity |
| Structured comments get engaged more | Anecdote; testable |
| Proposed reg text is strongest | Doctrine-supported; adoption-lifts-wording untested |

**The central open risk:** the doctrine→practice inference. The team's own paper hypothesizes agencies respond to *litigation-capable actors* — if identity swamps content, the critic measures the wrong variable. This is testable, and we must be the ones who test it.

**Hypotheses:** H1 — content features predict engagement within docket after identity controls. H2 — structure predicts engagement. H3 — proposed-text comments see higher adoption. Tests: **within-docket, within-theme contrasts** (content, identity held fixed) and **duplicate-group contrasts** (identity, text held perfectly fixed). Confounder-control is a Phase-5 need, not a v1 blocker.

**Banked exploratory question:** do suspected LLM-generated comments (2024–25 corpora) receive less engagement than matched human comments on the same docket? Observational only; needs heavy uncertainty framing (detection is unreliable — see Weiss below).

**Sources (verified 2026-07-24):** Broder et al. 1997 (shingling); Manku, Jain & Das Sarma, WWW 2007 (SimHash); Lee et al. 2021, arXiv:2107.06499 (substring/MinHash dedup at scale); Kao 2017, HackerNoon (1.3M+ synonym-spun FCC comments — the existence proof for the semantic tier); Hitlin, Olmstead & Toor, Pew 2017 (21.7M FCC comments, 6% unique, 90,458 in one second); Balla, Dooling, Herz & Livermore, ACUS 2021 + Recommendation 2021-1; Weiss 2019, Technology Science (1,001 GPT-generated comments, human detection at 49% — fully generated text defeats all text-similarity tiers; only metadata forensics and platform auth remain); Comment Integrity and Management Act of 2024 (House-passed).

## 6. The fuel map

Every asset the agent consumes; who supplies it; what the revamp owes it.

**Sampling policy (governs rows 1–5):** census the metadata → census text + embeddings for any docket we analyze (one-time historical scrape; past comments are immutable; incremental front edge only) → sample the expensive LLM judgments across dockets, with weights → **demand-driven labeling**: the agent's retrieval triggers resolution/judgment for its own neighborhoods, so the labeling budget follows real usage. Rationale: sampling error is dwarfed by measurement error; budget goes to calibration, not census. Exception: dedup and within-docket contrasts structurally require census-within-docket.

| # | Asset | Consumed by | Supplier | Deliverable |
|---|---|---|---|---|
| 1 | Rulemaking corpus | Ingest; match/brief | fetch | Versioned artifacts (append-mostly library: current front edge, every past state preserved); incremental refresh |
| 2 | Comment corpus | Exemplar retrieval; research | mine | **Two modes:** survey (seeded, π persisted, weighted) + harvest (targeted complete dockets); versioned |
| 3 | Campaign index | Critic originality; identity contrasts | dedup | Tiered detector (below); stable hash IDs; own artifact; census-within-docket input |
| 4 | Identity labels | Exemplar filtering; confounder control | classify | Honest sampling (π persisted or full coverage); ~100-item validation set; versioned prompts |
| 5 | Outcome labels | Exemplar boost; critic calibration; accountability | track | Cascade + envelope semantics (below) |
| 6 | Calibration | Gates every outward claim | goldset + estimator | The house pattern (below) |
| 7 | Doctrine corpus | Plan; critic citations | hand curation (Ashlie ask) | `doctrine/` in-repo, versioned; entries carry source, date-checked, dimensions backed |
| 8 | Exemplar library | Write step; critic fixes | **join of 2+4+5** | **Contract first, store later:** `{text, author_type, outcome, response_location, provenance, embedding}`; v1 = query-time filter over seed corpus (RTC pairs + gold-set rows); materialize only when latency forces it. The contract is the research↔product boundary; exemplars are born attribution-complete |

**Row 3 — tiered campaign detection** (replaces cosine-only, which conflated template blasts with convergent sincere opinion): Tier 0 exact hash → Tier 1 shingle/MinHash Jaccard (the tier that earns the word "duplicate") → Tier 2 embeddings with centroid-capped clustering (labeled "semantically convergent," promoted to "campaign" only with metadata corroboration: timing bursts, formatting fingerprints). Two views: *mimicry* (≈ agency filters; current method acceptable) and *forensic* (tiered, corroborated, gold-set-thresholded). Critic uses forensic only; originality flags fire only against large corroborated clusters and always show the matched text. Dedup is annotation, never deletion.

**Row 5 — cascade and envelope semantics:** crosswalk (deterministic) → resolver (five channels) → grounded judgment → **envelope-relative `not_found`**. **The LLM web-search tier is killed entirely** (Jonathan's decision, reversing R3): "no" means *not found in the declared, versioned envelope* — never "doesn't exist." Envelope version lives in the run manifest; rows inherit it; per-row `channels_run` is finer-grained truth (partial failures and demand-driven mixed-version states stay visible). Two negatives stay distinct: envelope-clean `not_found` vs `UNKNOWN` (search incomplete) — never collapsed. Envelope v2 (docket RTC bodies / Ruler B; GovInfo) is deliberately deferred: added later as declared channels, valued by re-resolving v1 negatives (the recovery rate = measured marginal yield). Coverage grows by widening the envelope, never by unbounded search.

**Row 6 — the calibration house pattern:** *every machine judgment carries a measured error rate before it's published; the measurement is a small labeled sample built just-in-time by whoever's cheapest to spend.* Four instances: response gold set (30 rows, Arvind, in flight — now also samples `UNKNOWN`s to measure what the envelope's blind spot hides); judge-agreement (~30, Jonathan+Arvind, the only v1-blocking set); cluster set (~50, fast); identity set (~100, delegable). Total ≈ 10–15 person-hours across months. Deliverable: the goldset harness generalized into a reusable label-sample→agreement/error→attach kit.

## 7. Architecture invariants (the supply chain)

Strangler rebuild of the six stages in the proven house pattern (freeze/goldset/rtc/resolution): standalone packages, pure decision core vs I/O shell, DI, fixture tests. Old pipeline runs until each replacement validates against it.

1. **Run-versioned immutable artifacts** — every stage writes `runs/{year}/{run_id}/` (parquet + manifest: git SHA, config hash, seed, input hashes, row counts, fetch dates, LLM cost). Nothing overwrites. Kills F23/F18/F22-class failures; freeze becomes what the pipeline *is*.
2. **Weights are data** — every sampling decision emits `(unit, stage, inclusion_probability)`; one estimator composes them; `Σw ≈ N` asserted per level. Estimand: the full regs.gov universe (Q4 decision).
3. **Determinism** — one seed in the manifest; no wall-clock in data rows; content-hash IDs.
4. **Schema-validated boundaries** — key uniqueness enforced at every join (F22 impossible by construction).
5. **No speculative machinery** — no DB, no envelope registry, no migration tooling until a measured need forces it.

## 8. Build order

- **Phase 0 (done/in flight):** resolver (13/13, 2026-07-24), RTC parser, goldset harness, frozen snapshots. Next in lane: `track_responses` rewire; gold-set frame swap/redraw.
- **Phase 1 — v1 agent, zero pipeline dependency:** doctrine corpus; `rubric.yaml` + judge; exemplar contract over seed corpus; judge-agreement set. **Done = working coach/reviewer behind the firewall, every claim doctrine-cited.** Product-side build against this spec; Jonathan reviews.
- **Phase 2 — outcome labels complete (Jonathan's lane):** cascade wiring, envelope semantics, no web search; frame swap + redraw; FN calibration incl. UNKNOWN composition. **Done = every label carries tier+confidence; every aggregate carries error bars.**
- **Phase 3 — corpus stages (mine→dedup→classify):** survey/harvest modes, census-text policy, tiered detector + cluster set, identity set. **Done = versioned artifacts; audit F-findings closed by construction.**
- **Phase 4 — estimator + rendering split:** universe weights, tidy stats tables, thin renderer; fetch versioning folded in. **Done = paper figures regenerate from a pinned snapshot, universe-weighted.**
- **Phase 5 — calibrated critic + envelope v2:** the two contrast analyses; rubric-weight fitting; two-number display; Ruler B/GovInfo as v2. **Done = the calibrated signal exists with CIs — or we've honestly measured that identity swamps content and claims retreat to doctrine.**

Phases 1 and 2 parallelize across different owners.

## 9. Sketched capabilities (mapped, not specified)

- **Match / brief / monitor** — fuel: row 1 + embeddings; nearest to the existing site.
- **Submit / track** — email + physical-mail out (regs.gov API ban); own-comment tracking is a *separate* resolution problem (matching a mailed comment to its eventual docket entry), reusing the scrape machinery.
- **Accountability** — the ignored-comments DB for legal pursuit. **Hard-gated on measured FN rates** (row 6): never flag an agency from a pipeline whose miss rate is unknown.

## 10. Non-goals

No AI-generated comment prose, ever, in any mode. No open-web search anywhere in the measurement path. No database, envelope registry, or migration tooling in v1. No accountability flags before FN calibration. No citizen-mode UX in v1.

## 11. Banked (no owner until claimed)

- LLM-generated-comment engagement study (§5) — exploratory.
- Packet v2 (resolver-powered annotation packets; carries `posted_date`) — separately specced.
- Authorship conversation: the calibration instrument is a paper-grade contribution.
- Envelope v2 channels: docket RTC bodies (parser exists), GovInfo.
