# Plain Politics — System Architecture, Explained for the Whole Team

*Written 2026-07-24 (Jonathan). Companion to the formal design spec (`docs/superpowers/specs/2026-07-24-comment-agent-and-fuel-architecture-design.md`). Everything here is accurate to that spec, with terms defined as they appear. Read time ~15 minutes.*

---

## 0. The one-paragraph version

We're building two connected things. **The coach**: an AI assistant that helps organizations write public comments that agencies are legally required to engage with — it never writes for you, it questions, structures, grades, and shows real examples. **The measurement engine**: a data system that traces what happened to millions of past comments (who wrote them, did the agency respond) so that our advice is eventually backed by measured evidence, not folklore. The coach can ship soon because its first version runs on law, not data. The engine matures behind it and upgrades the coach's claims from "legally sound" to "measurably effective."

---

## 1. The legal foundation (why this product can exist)

- **Notice-and-comment**: when a federal agency wants to make a rule, the **Administrative Procedure Act (APA)** requires it to publish the proposal (a **NPRM** — Notice of Proposed Rulemaking) and let anyone submit written comments before finalizing.
- **The duty to respond**: the APA's §553 plus decades of court decisions (**doctrine** — statute as interpreted by binding case law) require agencies to respond to **significant comments**: ones that are specific, evidence-backed, legally grounded, or propose concrete alternatives. Ignoring one can get the final rule **vacated** (struck down by a court) as "arbitrary and capricious" — *State Farm* (1983) is the landmark case; *Ohio v. EPA* (2024) is a recent example.
- **The gap we exploit**: most people's comments are opinions, which agencies may lawfully set aside. Very few commenters know what makes a comment one the agency *must* engage. That knowledge is our product.

---

## 2. The coach (the agent)

### What it does — the six-step loop

1. **Ingest** — user picks a rulemaking; we pull the proposal's full text, deadline, and existing docket comments. (A **docket** is the public folder holding everything about one rulemaking.)
2. **Elicit** — the user states their position; the agent asks clarifying questions ("what evidence does your org hold?").
3. **Plan** — the agent maps their concerns to the argument types with legal teeth (statutory objection, empirical critique, procedural challenge, concrete alternative) and proposes an outline: headings plus guidance per section. Why headings: agencies respond to comments *by theme*; a sectioned comment pre-sorts its arguments into the buckets the agency's own response process uses.
4. **Write** — the human writes, section by section. The agent prompts and shows one real, attributed example per section.
5. **Review** — the **critic** (below) scores the draft and gives actionable feedback. Iterate.
6. **Deliver** — the finished comment, in the user's own words, ready to submit.

### The generation firewall (our most important product rule)

**The AI never produces comment prose.** Its outputs are only: questions, outline structure, critique, and retrieved real excerpts with attribution. This is enforced in the software interface itself (the response format has no "draft text" field), not just in instructions.

Three reasons, all load-bearing:
1. **Backlash**: "AI wrote my public comment" is exactly the story we never want written about us.
2. **Efficacy**: agencies run deduplication software that collapses similar/template comments into one. AI-drafted text converges across users → gets collapsed → your comment counts less. Original words literally work better.
3. **Thesis-consistency**: the research literature (§6) documents machines flooding this channel with fake voice. We are the countermeasure, not a contributor.

Even **proposed regulatory text** — suggesting the actual replacement wording for a rule section, the single strongest move a comment can make — stays behind the firewall: we show real examples of amendments that worked and point at the section to change; the user writes the words.

### The critic

The critic grades a draft against a **rubric** — a fixed checklist of dimensions, each scored absent/weak/strong, each backed by a citation:

| Dimension | Question it asks |
|---|---|
| Section anchoring | Does each argument name the exact provision it targets? |
| Evidence grounding | Data, citations, documented experience — or just assertion? |
| Statutory hook | Is any objection tied to the agency's authorizing statute? |
| Procedural validity | In scope of what was actually proposed, and on time? |
| Concrete alternative | Does it propose a specific change (ideally regulatory text)? |
| Materiality | Would adopting the point change the rule, or is it cheerleading? |
| Originality | How far is this from known mass-campaign text? |
| Structure | Headed, numbered, theme-sortable? |

Mechanically, the critic is an **LLM judge**: a language model given the draft plus the rubric, required to return structured output (scores + justifications quoting the user's text + a fix suggestion per weak dimension). The rubric lives as a versioned file in the repo (`rubric.yaml`), so changes to it are reviewed like code.

Two honesty rules:
- **The judge itself gets graded.** Before any partner sees scores, humans score a sample of comments on the same rubric and we measure human–AI agreement per dimension (**judge-agreement**). Dimensions where the AI can't match humans get labeled "advisory."
- **Two kinds of claims, never blended.** The **doctrine score** ("the law says an agency can set this aside") is available on day one and is always true. The **calibrated signal** ("drafts like this measurably got engaged X% of the time, ± error") appears later, only where our data has actually cleared the bar. We never say "predictive" about anything we haven't measured.

---

## 3. The measurement engine (the fuel)

Everything the coach shows or claims eventually traces to data assets produced by a six-stage pipeline. Terms first:

- **API** — a website's machine-readable interface. The Federal Register and Regulations.gov both have official APIs we pull from.
- **Scraping/mining** — programmatically downloading records (documents, comments) at scale.
- **Artifact** — a saved, dated output file from a pipeline run. Our rule: artifacts are **immutable** (never overwritten); every run writes a new versioned folder with a **manifest** (a metadata file recording what code, settings, and inputs produced it). This is what makes results reproducible and auditable.

### Stage 1 — Fetch: the rulemaking library
Downloads every Federal Register document (proposals, final rules, notices) with metadata: agency, deadline, docket ID. Feeds the coach's Ingest step and the site's discovery features. Kept fresh by small **incremental** pulls (only new documents — published FR documents never change once out).

### Stage 2 — Mine: the comment corpus
Downloads the comments themselves. Two modes:
- **Survey mode** (research): a **stratified sample** — instead of downloading all ~1.24M comments/year, we sample within groups (by agency × comment-volume) and record each comment's **inclusion probability** (its chance of being picked). Keeping those probabilities as data is what lets a sample honestly represent the whole population (**weighting**).
- **Harvest mode** (product): complete downloads of specific dockets we care about — because you can't show "similar past comments" from data you never collected.

Policy in one line: *census the cheap layers (metadata, raw text), sample the expensive ones (AI judgments), and let real product usage decide where to spend labeling effort (demand-driven).*

### Stage 3 — Deduplicate: the campaign index
Detects mass campaigns — thousands of near-identical submissions. Three tiers, because "duplicate" hides two different things:
- **Tier 0/1 — literal templates**: exact text hashing plus **MinHash/shingling** (comparing documents by their overlapping word-sequences — the standard web-scale near-duplicate technique). This catches form letters even when people add a personal paragraph.
- **Tier 2 — spun campaigns**: **embeddings** (turning text into numerical vectors so similar meanings land near each other) catch campaigns that swap synonyms per copy to *look* unique — famously, 1.3M+ such comments in the FCC net-neutrality docket.
- **Corroboration**: metadata signals (thousands of comments in the same second, identical formatting) decide what gets called a "campaign" vs merely "similar."

Nothing is deleted — dedup adds labels; analyses choose whether to view all comments or unique texts. The coach's Originality dimension checks the user's draft against this index (and always shows the matched text, so users can judge for themselves).

### Stage 4 — Classify: who's commenting
Labels each comment's author type: ordinary citizen / organization / academic-expert / lobbyist / undecided — cheap rules on metadata first, an LLM for the rest. Used to filter examples (org users see org exemplars) and, on the research side, to separate *what was written* from *who wrote it* (see §5).

### Stage 5 — Track: did the agency respond? (the hard one)
For each comment, determines whether the agency engaged with it, in a fixed order of reliability:
1. **Crosswalk** — some agencies publish explicit response-to-comments tables mapping comment IDs to answers. Deterministic; we parse these directly (our EPA parser does this today).
2. **Resolver** — built and tested as of 7/24: given a comment, it finds every document where a response *could legally live* (the linked final rule, everything under the same **RIN** — the government's tracking number for one rulemaking — the docket, the regulatory agenda, and an identifier-based full-text search). It returns candidates with provenance, or a structured absence.
3. **Grounded judgment** — an LLM reads the *right* document (found by the resolver) and judges whether this comment's substance was addressed.
4. **That's it. There is no web-search fallback.** We removed it deliberately (see §4).

### Stage 6 — Estimate & render
Composes all the recorded inclusion probabilities into population **weights**, produces the statistics tables, and only then draws figures. Numbers are reviewable as tables before they become charts.

---

## 4. What "no response" means here (envelope semantics)

A search can never prove a response doesn't exist *anywhere*. So we don't claim that. Our vocabulary:

- **`not_found`** means: *searched every bin in our declared, versioned search envelope — cleanly — and it isn't there.* The **envelope** is the explicit list of places we look (currently: packet link, RIN, docket, agenda, identifier full-text). Each result inherits the envelope version from its run manifest.
- **`UNKNOWN`** means: *we couldn't complete the search* (a source was down, a fetch failed). Never merged with `not_found` — collapsing "couldn't look" into "isn't there" was precisely the old pipeline's worst bug.
- Structured absences: sometimes "no response" is the *correct* answer with a reason — the rule type can't carry responses, the final rule isn't published yet, or the agency's agenda says no final rule is planned. We record which.
- **Growing coverage = widening the envelope** (adding docket response-PDFs, the GovInfo source) — each addition is a new envelope version, and we measure its value by re-running old negatives and counting recoveries. Coverage never grows by unbounded googling.

Why we killed the old web-search tier: it instructed the model to "make a call" and mapped "searched, found nothing" to a confident **no** — quietly deflating every response rate we published. Honest uncertainty beats manufactured certainty.

---

## 5. The honesty machinery (calibration)

House rule: **every machine judgment carries a measured error rate before it's published.** The instrument is a **gold set** — a small human-labeled sample used to measure the machine against people:

| Gold set | Size | Measures | Who |
|---|---|---|---|
| Response tracking | 30 rows | **False-negative rate** — how often we say "no response" when one exists | Arvind (in flight) |
| Judge agreement | ~30 | Whether the critic's grades match human graders, per dimension | Jonathan + Arvind |
| Cluster validation | ~50 | Whether our "same campaign" calls are right; sets thresholds empirically | fast, anyone careful |
| Identity labels | ~100 | Precision/recall of citizen/org/lobbyist labels | delegable |

Total effort ≈ 10–15 person-hours spread over months, built just-in-time (a gold set is only owed when its layer is about to make a public claim). Published numbers carry **confidence intervals** (error bars: the honest range, not just the point estimate).

**Confounding — the research problem in one example**: suppose comments with proposed regulatory text get engaged 3× more. Is it the *text*, or that *industry lawyers* (whom agencies fear in court) happen to write such text? A **confounder** is a hidden third variable driving both sides of a correlation. Our two controls: compare comments *on the same docket, same argument, same author-type* that differ in writing (isolates content), and compare *identical template texts sent by different author-types* (isolates identity — the campaign index gives us this for free). Until those analyses are done, the coach claims doctrine, not effectiveness.

**The risk we're honest about internally**: it may turn out *who you are* matters more than *what you write* (agencies may respond mainly to actors who can sue). If our data shows that, the calibrated layer will say so — and the product's claims stand on doctrine, which remains true regardless.

---

## 6. Why the firewall is also strategy (the literature in four lines)

- Pew (2017): of 21.7M FCC net-neutrality comments, **6% were unique**; 90,458 identical comments arrived in one second.
- Kao (2017): 1.3M+ of those were synonym-swapped fakes designed to look unique.
- Weiss (2019): a student's GPT bot submitted 1,001 comments to a federal site; trained humans detected them at **49% — a coin flip**.
- ACUS (2021) + a 2024 House bill: regulators and Congress are actively working this problem.

Every one of these is machines drowning authentic voice. Our product is the inverse: helping real voices meet the legal bar. That's the positioning, and it's also why "no AI ever writes a word of your comment" is a strategic asset, not a limitation.

---

## 7. Where things stand + the next two months

**Built and verified (as of 7/24):**
- The **resolver** (stage 5's core): 13 tasks, 203 passing tests, validated against six hand-traced real-world cases including the hardest one (a response published under a *different* RIN than the comment's — found via identifier search).
- The **EPA crosswalk parser** (deterministic comment→response pairs).
- The **gold-set harness** (sampling + blind labeling packets + grading with error bars); the 30-row response set is with Arvind now.
- **Frozen snapshots** (versioned, hash-verified copies of the datasets our analyses cite).

**Website today** (Praneel/Sophia's side — placeholder, to be confirmed by them): pulls real regulations; federal + CA + NY coverage; submission currently manual; comment-tracking backend not yet built.

**Build phases ahead** (dependency order, not dates):
1. **v1 coach** — doctrine corpus (curated law library — a well-shaped ask for Ashlie), rubric + judge, seed exemplar library from crosswalk + gold-set rows. *No dependency on the pipeline revamp.* Product-side build; Jonathan reviews.
2. **Outcome labels complete** (Jonathan) — wire response-tracking to the resolver, envelope semantics, no web search, error rates measured.
3. **Corpus stages revamped** — the two-mode miner, tiered campaign detection, validated identity labels.
4. **Estimator** — population weights done right; the paper's figures regenerate reproducibly.
5. **Calibrated coach** — the confounder-controlled analyses feed measured weights back into the critic; envelope v2.

Phases 1 and 2 run in parallel with different owners.

---

## 8. FAQ

**Why not have AI just write the comment?** See §2 — it would get the user's comment collapsed by the agency's own dedup filters, and it's the one headline we can't survive.

**Why not analyze all 1.24M comments instead of sampling?** Downloading text: fine, we do (once — old comments never change). But every AI judgment costs money per comment; a weighted sample plus hand-checked error rates gives the same answers at ~1% of the cost. Precision without calibration is a precisely wrong answer.

**What does "the agency didn't respond" mean on our platform?** "Not found in any of the places a response can legally appear, per our published list of places, searched cleanly" — with our miss-rate measured and disclosed. Never "we googled and didn't see it."

**When can we say 'comments written with our tool work better'?** When Phase 5's controlled analyses exist, with error bars. Until then we say: "your comment meets the standard agencies are legally required to engage" — which is true now.

**Who do I ask about what?** Coach/product behavior: this doc §2. Data/measurement: §3–5 (Jonathan). Legal framing: §1 + the doctrine corpus (Ashlie, once curated). Annotation: Arvind.
