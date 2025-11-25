#!/usr/bin/env python3
"""Classify comment authors using OpenAI GPT with Word2Vec sampling for large datasets.

For documents with >1000 comments, uses Word2Vec embeddings + density-aware sampling
to select a representative subset with 99.9% confidence interval.
"""
import argparse
import asyncio
import math
import os
import re
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
from gensim.models import Word2Vec
from openai import AsyncOpenAI
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# Category mapping
LABEL_MAP = {
    "undecided": "Undecided/Anonymous",
    "citizen": "Ordinary Citizen",
    "org": "Organization/Corporation",
    "expert": "Academic/Industry/Expert (incl. small/local business)",
    "lobbyist": "Political Consultant/Lobbyist",
}


def classify_from_metadata(row: Dict) -> Optional[str]:
    """Classify comment author from metadata fields before resorting to OpenAI.
    
    Returns category label if confident, None if metadata insufficient.
    """
    organization = row.get("organization")
    first_name = row.get("first_name")
    last_name = row.get("last_name")
    submitter_type = row.get("submitter_type")
    gov_agency = row.get("gov_agency")
    
    # Organization present without personal name → Organization/Corporation
    if organization and not first_name and not last_name:
        org_lower = str(organization).lower()
        # Check for corporate/org indicators
        if any(term in org_lower for term in ["inc", "llc", "corp", "company", "association", "coalition", "group", "council"]):
            return LABEL_MAP["org"]
    
    # Government agency → Academic/Industry/Expert
    if gov_agency:
        return LABEL_MAP["expert"]
    
    # Submitter type hints
    if submitter_type:
        sub_lower = str(submitter_type).lower()
        if "organization" in sub_lower or "company" in sub_lower or "business" in sub_lower:
            return LABEL_MAP["org"]
        if "individual" in sub_lower and not organization:
            return LABEL_MAP["citizen"]
    
    # Organization + name suggests small business or expert affiliation
    if organization and (first_name or last_name):
        org_lower = str(organization).lower()
        if any(term in org_lower for term in ["university", "college", "institute", "research", "lab", "dept", "department"]):
            return LABEL_MAP["expert"]
        # Could be small business owner - lean expert
        if any(term in org_lower for term in ["consulting", "consultancy", "partners", "solutions", "services"]):
            return LABEL_MAP["expert"]
    
    # Insufficient metadata
    return None

# Enhanced prompt with decision guidance - categories ordered to minimize bias
# IMPORTANT: Comment text goes FIRST for better LLM attention
PROMPT_TEMPLATE = """Comment to classify:
{comment_text}

Metadata:
- Organization: {organization}
- Submitter Type: {submitter_type}
- Name: {first_name} {last_name}

---

Classify the AUTHOR of the above public comment as exactly ONE of these categories:

**org** - Large corporations, industry associations, trade groups, or organizational entities
**citizen** - Individual ordinary citizens, residents, or community members without professional affiliation
**expert** - Academics, researchers, small/local businesses, industry experts, technical specialists
**lobbyist** - Political consultants, registered lobbyists, advocacy groups, or professional campaigners
**undecided** - ONLY if truly anonymous with no identifying info or completely unclear authorship

Decision guidance:
- Use ALL available context clues (tone, language, content, structure, signatures, letterheads)
- Look for: organizational letterhead, academic titles, business names, lobbying language, personal stories, signatures
- Large company or trade association or group → org
- Generic "concerned citizen" with personal story/opinion → citizen (NOT undecided)
- Individual small business owners → expert (not org)
- Professional advocates or political operatives → lobbyist
- AVOID undecided unless genuinely impossible to determine - make your best inference from the text

Be decisive. Most comments CAN be classified if you look at the language, tone, and context. Only use "undecided" as a last resort.

Note: Text may be from inline comment or extracted from PDF attachment (first 2 pages).

Reply with ONLY ONE WORD from: org, citizen, expert, lobbyist, undecided"""


def tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase, split, remove short words."""
    tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
    # Basic stopword removal
    stopwords = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'has', 'are', 'was', 'were', 'been', 'will'}
    return [t for t in tokens if t not in stopwords]


def vectorize_comments(comments: List[str]) -> Tuple[np.ndarray, Optional[Word2Vec]]:
    """Train Word2Vec and return normalized comment vectors."""
    tokenized_all = [tokenize(c) for c in comments]
    corpus = [tokens for tokens in tokenized_all if tokens]

    if len(corpus) < 10:
        return np.zeros((len(comments), 300)), None

    model = Word2Vec(
        sentences=corpus,
        vector_size=300,
        window=5,
        min_count=2,
        workers=4,
        seed=42,
    )

    vectors = []
    for tokens in tokenized_all:
        if not tokens:
            vectors.append(np.zeros(300))
            continue

        vecs = [model.wv[t] for t in tokens if t in model.wv]
        if not vecs:
            vectors.append(np.zeros(300))
            continue

        vec = np.mean(vecs, axis=0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append(vec)

    return np.array(vectors), model


def compute_sample_size(
    vectors: np.ndarray,
    pilot_size: int = 500,
    Z: float = 3.291,
    E: float = 0.05,
    random_state: Optional[int] = None,
) -> int:
    """Calculate required sample size for 99.9% CI."""
    N = len(vectors)
    if N == 0:
        return 0

    pilot_count = min(pilot_size, N)
    rng = np.random.default_rng(random_state)
    pilot_indices = rng.choice(N, pilot_count, replace=False)
    pilot_vecs = vectors[pilot_indices]

    norms = np.linalg.norm(pilot_vecs, axis=1)
    sigma = np.std(norms) if norms.size else 0.0

    if sigma == 0:
        return min(1, N)

    n = (Z**2 * sigma**2) / (E**2)
    n_corrected = n / (1 + (n - 1) / N)

    return max(1, min(N, int(np.ceil(n_corrected))))


def density_aware_sampling(
    vectors: np.ndarray,
    n_samples: int,
    k_neighbors: int = 50,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Sample inversely proportional to density with cluster coverage."""
    N = len(vectors)
    if N == 0:
        return np.array([], dtype=int)

    n_samples = min(max(1, n_samples), N)
    if N == 1 or n_samples == N:
        return np.arange(n_samples)

    rng = np.random.default_rng(random_state)

    k_clusters = int(math.sqrt(max(N / 2, 1)))
    k_clusters = max(1, min(max(10, k_clusters), N))

    kmeans = MiniBatchKMeans(n_clusters=k_clusters, random_state=42, batch_size=1000)
    clusters = kmeans.fit_predict(vectors)

    k_neighbors = min(k_neighbors, N - 1)
    if k_neighbors <= 0:
        return rng.choice(N, size=n_samples, replace=False)

    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="cosine")
    nn.fit(vectors)
    distances, _ = nn.kneighbors(vectors)

    mean_dist = distances[:, 1:].mean(axis=1)
    density = 1.0 / (mean_dist + 1e-6)

    weights = 1.0 / (density + 1e-6)
    weights = weights / weights.sum()

    sampled_indices = rng.choice(N, size=n_samples, replace=False, p=weights)

    missing_clusters = []
    for cluster_id in range(k_clusters):
        cluster_mask = clusters == cluster_id
        if cluster_mask.any() and not np.any(clusters[sampled_indices] == cluster_id):
            missing_clusters.append(cluster_id)

    if missing_clusters:
        replace_order = list(np.argsort(weights[sampled_indices])[::-1])
        for idx, cluster_id in enumerate(missing_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if cluster_indices.size == 0:
                continue
            target_idx = replace_order[idx] if idx < len(replace_order) else replace_order[-1]
            sampled_indices[target_idx] = rng.choice(cluster_indices)

    return sampled_indices


async def classify_comment(client: AsyncOpenAI, comment_text: str, metadata: Dict[str, str], semaphore: asyncio.Semaphore, model: str) -> Tuple[Optional[str], str, str]:
    """Classify a single comment, return (category, prompt_used, model_response)."""
    prompt = PROMPT_TEMPLATE.format(
        comment_text=comment_text[:3000],
        organization=metadata.get("organization", "N/A"),
        submitter_type=metadata.get("submitter_type", "N/A"),
        first_name=metadata.get("first_name", ""),
        last_name=metadata.get("last_name", "")
    )
    
    async with semaphore:
        backoff = 2.0  # Start at 2 seconds
        max_attempts = 8  # Increase attempts from 5 to 8
        
        for attempt in range(max_attempts):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=20,
                )
                
                # Extract response
                if response.choices and response.choices[0].message.content:
                    raw_result = response.choices[0].message.content.strip()
                else:
                    raw_result = ""
                
                if not raw_result:
                    tqdm.write("  WARN: Empty LLM response")
                    return LABEL_MAP["undecided"], prompt, "EMPTY_RESPONSE"
                
                result = raw_result.lower()
                
                # Validate response
                if result in LABEL_MAP:
                    return LABEL_MAP[result], prompt, raw_result
                
                for token in LABEL_MAP.keys():
                    if token in result:
                        return LABEL_MAP[token], prompt, raw_result
                
                tqdm.write(f"  WARN: Unexpected LLM response: {raw_result}")
                return LABEL_MAP["undecided"], prompt, raw_result
                
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in error_str or "rate limit" in error_str
                
                if attempt < max_attempts - 1:
                    # Calculate sleep time with Jitter
                    # Jitter prevents all threads from hitting the API again at the exact same moment
                    jitter = random.uniform(0.5, 2.0)
                    sleep_time = backoff + jitter
                    
                    # If it's a 429, impose a stiff penalty immediately
                    if is_rate_limit:
                        sleep_time += 20.0  # Force a minimum 20s wait for rate limits
                        tqdm.write(f"  RATE LIMIT (429): Retrying in {sleep_time:.1f}s...")
                    else:
                        tqdm.write(f"  ERROR: API failed (attempt {attempt+1}/{max_attempts}): {e}. Retrying in {sleep_time:.1f}s...")
                    
                    await asyncio.sleep(sleep_time)
                    
                    # Exponential backoff, but cap at 60 seconds (window reset time)
                    backoff = min(backoff * 2, 60.0)
                else:
                    tqdm.write(f"  ERROR: API call failed after {max_attempts} attempts: {e}")
        
        return LABEL_MAP["undecided"], prompt, "ERROR: retries_exhausted"


async def classify_batch(
    client: AsyncOpenAI,
    comments: List[Tuple[str, str, Dict[str, str]]],  # (comment_id, text, metadata)
    model: str,
    max_concurrency: int,
) -> List[Tuple[str, Optional[str], str, str]]:
    """Classify a batch of comments. Returns [(comment_id, category, prompt, model_response)]."""
    if not comments:
        return []

    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_single(comment_id: str, text: str, metadata: Dict[str, str]) -> Tuple[str, Optional[str], str, str]:
        category, prompt, model_response = await classify_comment(client, text, metadata, semaphore, model)
        return comment_id, category, prompt, model_response

    tasks = [asyncio.create_task(run_single(comment_id, text, metadata)) for comment_id, text, metadata in comments]

    results: List[Tuple[str, Optional[str], str, str]] = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Classifying"):
        results.append(await task)

    return results


async def run_classification_batches(
    client: AsyncOpenAI,
    batches: List[Tuple[str, List[Tuple[str, str, Dict[str, str]]]]],
    model: str,
    max_concurrency: int,
    results_csv: Path,
    chunk_size: int = 100,
) -> None:
    """Run classify_batch sequentially for each document using one event loop."""
    pending: List[Tuple[str, Optional[str], str, str]] = []

    for doc_number, comments in batches:
        if not comments:
            continue
        doc_results = await classify_batch(client, comments, model, max_concurrency)
        pending.extend(doc_results)

        if len(pending) >= chunk_size:
            save_results(results_csv, pending, model)
            pending = []

    if pending:
        save_results(results_csv, pending, model)


def classify_comments(
    comments_csv: Path,
    results_csv: Path,
    fr_csv: Path,
    output_csv: Path,
    api_key: str,
    model: str,
    max_concurrency: int,
    sample_threshold: int,
    sampling_seed: Optional[int],
) -> None:
    """Main classification pipeline with metadata-first classification."""
    
    # Read raw comments
    if not comments_csv.exists():
        print(f"ERROR: {comments_csv} not found. Run mine_comments.py first.")
        return
    
    df_comments = pl.read_csv(str(comments_csv))
    
    # Load existing results
    classified_ids: Set[str] = set()
    if results_csv.exists():
        df_results = pl.read_csv(str(results_csv))
        classified_ids = set(df_results["comment_id"].to_list())
        print(f"Loaded {len(classified_ids)} already-classified comments")
    
    # Filter to unclassified
    df_unclassified = df_comments.filter(~pl.col("comment_id").is_in(list(classified_ids)))
    
    if len(df_unclassified) == 0:
        print("All comments already classified")
        # Join and write output
        if results_csv.exists():
            join_and_write_output(comments_csv, results_csv, fr_csv, output_csv)
        return
    
    print(f"Classifying {len(df_unclassified)} comments...")
    
    # Phase 1: Metadata-based classification and empty comment handling
    print("\nPhase 1: Metadata-based classification and empty comment filtering...")
    metadata_results = []
    needs_llm = []
    excluded_count = 0
    
    for row in df_unclassified.iter_rows(named=True):
        comment_id = row.get("comment_id")
        comment_text = row.get("comment_text", "")
        attachment_text = row.get("attachment_text", "")
        
        # Check if we have any text data at all (gate before metadata or LLM work)
        has_comment_text = bool(comment_text and comment_text.strip())
        has_attachment_text = bool(attachment_text and attachment_text.strip())
        if not has_comment_text and not has_attachment_text:
            excluded_count += 1
            continue
        
        # Try metadata classification first
        category = classify_from_metadata(row)
        
        if category:
            # Classified via metadata
            metadata_results.append((comment_id, category, "metadata", "metadata"))
        elif has_comment_text or has_attachment_text:
            # Has text data - needs LLM classification
            needs_llm.append(row)
    
    print(f"  Classified via metadata: {len(metadata_results)}/{len(df_unclassified)} ({100*len(metadata_results)/len(df_unclassified):.1f}%)")
    print(f"  Excluded (no data): {excluded_count}/{len(df_unclassified)} ({100*excluded_count/len(df_unclassified):.1f}%)")
    print(f"  Needs LLM classification: {len(needs_llm)}/{len(df_unclassified)} ({100*len(needs_llm)/len(df_unclassified):.1f}%)")
    
    # Save metadata results
    if metadata_results:
        save_results(results_csv, metadata_results, "metadata")
        print(f"\nSaved metadata classifications to: {results_csv.absolute()}")
    
    # Phase 2: LLM classification for remaining
    if not needs_llm:
        print("\nAll comments classified via metadata or excluded!")
        join_and_write_output(comments_csv, results_csv, fr_csv, output_csv)
        return
    
    print(f"\nPhase 2: LLM classification for {len(needs_llm)} remaining comments...")
    
    by_doc: Dict[str, List[Tuple[str, str, Dict[str, str]]]] = {}
    for row in needs_llm:
        doc_num = row.get("document_number")
        comment_id = row.get("comment_id")
        comment_text = row.get("comment_text")
        attachment_text = row.get("attachment_text")
        
        # Combine texts so we don't lose the PDF if there is a short inline comment
        # Handle None values explicitly (Polars can return None for missing values)
        parts = []
        if comment_text is not None and str(comment_text).strip() and str(comment_text).strip().lower() != "none":
            parts.append(str(comment_text).strip())
        if attachment_text is not None and str(attachment_text).strip() and str(attachment_text).strip().lower() != "none":
            parts.append(str(attachment_text).strip())
            
        text = "\n\n".join(parts)
        
        if not doc_num or not comment_id or not text:
            continue
            
        # Prepare metadata for prompt
        metadata = {
            "organization": str(row.get("organization", "")),
            "submitter_type": str(row.get("submitter_type", "")),
            "first_name": str(row.get("first_name", "")),
            "last_name": str(row.get("last_name", "")),
        }
        
        by_doc.setdefault(doc_num, []).append((comment_id, text, metadata))
    
    if not by_doc:
        print("No documents with usable text after filtering.")
        join_and_write_output(comments_csv, results_csv, fr_csv, output_csv)
        return
    
    client = AsyncOpenAI(api_key=api_key)
    doc_batches: List[Tuple[str, List[Tuple[str, str, Dict[str, str]]]]] = []
    
    for doc_number, comments in by_doc.items():
        n_comments = len(comments)
        print(f"\nDocument {doc_number}: {n_comments} comments")
        
        if n_comments <= sample_threshold:
            to_classify = comments
        else:
            print(f"  Vectorizing {n_comments} comments...")
            texts = [c[1] for c in comments]
            vectors, w2v_model = vectorize_comments(texts)
            
            if vectors.shape[0] == 0 or w2v_model is None:
                print("  Skipping (insufficient data for vectorization)")
                continue
            
            print("  Computing sample size...")
            n_samples = compute_sample_size(vectors, random_state=sampling_seed)
            if n_samples <= 0:
                n_samples = min(1, len(comments))
            print(f"  Required sample size: {n_samples} (from {n_comments} total)")
            
            print("  Density-aware sampling...")
            sampled_indices = density_aware_sampling(
                vectors,
                n_samples,
                random_state=sampling_seed,
            )
            
            to_classify = [comments[i] for i in sampled_indices]
            print(f"  Selected {len(to_classify)} comments to classify")
        
        if to_classify:
            doc_batches.append((doc_number, to_classify))
    
    if not doc_batches:
        print("No documents ready for LLM classification after sampling.")
        join_and_write_output(comments_csv, results_csv, fr_csv, output_csv)
        return
    
    asyncio.run(
        run_classification_batches(
            client,
            doc_batches,
            model,
            max_concurrency,
            results_csv,
        )
    )
    
    join_and_write_output(comments_csv, results_csv, fr_csv, output_csv)


def save_results(results_csv: Path, results: List[Tuple[str, Optional[str], str, str]], model: str) -> None:
    """Append results to CSV, including model response for debugging."""
    if not results:
        return
    
    df_new = pl.DataFrame({
        "comment_id": [r[0] for r in results],
        "category": [r[1] or "Undecided/Anonymous" for r in results],
        "model_response": [r[3] for r in results],  # NEW: Include raw model response for debugging
        "prompt_used": [r[2] for r in results],
        "model": [model] * len(results),
    })
    
    if results_csv.exists():
        df_existing = pl.read_csv(str(results_csv))
        # Ensure old data has model_response column
        if "model_response" not in df_existing.columns:
            df_existing = df_existing.with_columns(pl.lit(None).alias("model_response"))
        df_combined = pl.concat([df_existing, df_new])
    else:
        df_combined = df_new
    
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    df_combined.write_csv(str(results_csv))


def join_and_write_output(comments_csv: Path, results_csv: Path, fr_csv: Path, output_csv: Path) -> None:
    """Join comments + results + FR data and write makeup_data.csv for plots."""
    print("\nJoining results...")
    
    df_comments = pl.read_csv(str(comments_csv))
    df_results = pl.read_csv(str(results_csv))
    
    # Join
    df_joined = df_comments.join(df_results, on="comment_id", how="left")
    
    # Add agency from FR CSV
    if fr_csv.exists():
        df_fr = pl.read_csv(str(fr_csv))
        if "agency" in df_fr.columns and "document_number" in df_fr.columns:
            df_joined = df_joined.join(
                df_fr.select(["document_number", "agency"]),
                on="document_number",
                how="left",
            )
    
    # Select final columns for plots
    cols = ["document_number", "comment_id", "category"]
    if "agency" in df_joined.columns:
        cols.append("agency")
    
    df_output = df_joined.select(cols).filter(pl.col("category").is_not_null())
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_output.write_csv(str(output_csv))
    
    print(f"\n{'='*60}")
    print(f"SAVED CLASSIFIED COMMENTS")
    print(f"{'='*60}")
    print(f"Output file: {output_csv.absolute()}")
    print(f"Total classified: {len(df_output)}")
    print(f"{'='*60}")
    
    # Summary stats
    if len(df_output) > 0:
        print("\nCategory breakdown:")
        for cat in LABEL_MAP.values():
            count = df_output.filter(pl.col("category") == cat).shape[0]
            pct = 100.0 * count / len(df_output)
            print(f"  {cat}: {count} ({pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify comment authors with OpenAI")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--max-concurrency", type=int, default=100)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--sample-threshold", type=int, default=1000, help="Use sampling for documents with more than this many comments")
    parser.add_argument("--sampling-seed", type=int, default=None, help="Optional RNG seed for sampling reproducibility")
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    comments_csv = script_dir / "data" / f"comments_raw_{args.year}.csv"
    results_csv = script_dir / "data" / f"makeup_results_{args.year}.csv"
    fr_csv = script_dir.parent / "output" / f"federal_register_{args.year}_comments.csv"
    output_csv = script_dir.parent / "output" / f"makeup_data_{args.year}.csv"
    
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return
    
    classify_comments(
        comments_csv,
        results_csv,
        fr_csv,
        output_csv,
        api_key,
        args.model,
        args.max_concurrency,
        args.sample_threshold,
        args.sampling_seed,
    )


if __name__ == "__main__":
    main()


# learnings from building the classification pipeline:
#
# metadata-first classification:
# - metadata-first classification is a game changer - can classify 30-50% of comments without
#   touching the LLM based on organization, submitter_type, gov_agency fields from the detail
#   endpoint
# - this saves massive API costs and is actually more accurate for clear-cut cases (big corp
#   with no personal name = org, gov agency = expert)
# - key insight: organization + no name = corporation, but organization + name = small
#   business/expert since individuals often submit on behalf of their small businesses
# - empty submissions (no text, no attachments, no useful metadata) should be completely
#   excluded rather than marked as "undecided" - this creates a high-quality sample instead of
#   population with noise
#
# attachment text handling:
# - comments with empty inline text but pdf attachments can still be classified if we extract
#   the pdf text first (first 2 pages only to keep token counts low)
# - fallback chain: try metadata → try comment_text → try attachment_text → exclude if all empty
# - attachment_text field comes from mine_comments.py and contains concatenated text from all
#   pdf attachments (separated by \n\n---\n\n)
# - update prompt to mention text may be from pdf attachment so the model isn't confused by
#   formatting artifacts
#
# model selection:
# - max_completion_tokens=20 is sufficient for gpt-4o-mini (single word response)
# - model default: gpt-4o-mini is ~2x cheaper than gpt-5-nano despite same per-token pricing:
#   * gpt-5-nano needs 500+ output tokens (~$0.0003 output cost) but only uses 1 word
#   * gpt-4o-mini only needs 20 output tokens (~$0.000012 output cost) for same result
#   * gpt-5-nano input cost savings ($0.075 vs $0.150/1M) don't offset wasted output tokens
# - cost per classification: gpt-4o-mini ~$0.00018, gpt-5-nano ~$0.00039 (2x more expensive)
# - if you want to use gpt-5-nano anyway, set max_completion_tokens=500+ manually
#
# sampling for large dockets:
# - word2vec + density-aware sampling is essential for large dockets - naive random sampling
#   misses low-density clusters in embedding space
# - the 99.9% confidence interval formula (Z=3.291, E=0.05) from pilot sample gives rigorous
#   statistical backing for sample size calculation
# - cluster coverage ensures at least 1 comment from each semantic cluster is sampled
# - only sample for documents with >1000 comments (configurable via --sample-threshold)
#
# async/concurrency:
# - async/await with semaphore for rate limiting is the right pattern - maxes out openai
#   concurrency without hitting rate limits
# - asyncio.run() works fine for batch classification, don't need persistent event loop
# - tqdm works with async tasks if you iterate through them properly
# - --max-concurrency should be set based on your openai tier (20 is safe for most)
#
# data management:
# - incremental saving every 100 results prevents data loss on crashes during long runs
# - always join back to FR data to get agency info for plots
# - results are stored separately from raw comments (comments_raw_YYYY.csv vs
#   makeup_results_YYYY.csv) to allow re-classification without re-mining
# - output file (makeup_data.csv) joins comments + results + FR data for downstream plotting
#
# prompt engineering:
# - be decisive in the prompt - tell the model to avoid "undecided" unless absolutely necessary
# - provide clear decision guidance with examples for edge cases
# - order categories neutrally to minimize bias (alphabetically or by category type)
# - truncate long comments to 2000 chars to avoid token bloat - first 2000 chars usually
#   contain the key identifying information
#
# error handling:
# - always validate llm responses - check if output is in LABEL_MAP before accepting
# - try to extract valid tokens from malformed responses (e.g. "i think it's citizen" → "citizen")
# - return None for failed classifications and log them for review
# - don't crash on individual failures - continue processing remaining comments
#
# cost optimization:
# - use metadata classification first to reduce llm api calls by 30-50%
# - use sampling on large dockets (>1000 comments) to get statistical confidence without
#   classifying everything
# - set --max-comments-per-doc in mine_comments.py to skip extremely large dockets entirely
# - use cheap models (gpt-5-nano) for simple classification tasks
# - truncate prompts to 2000 chars to reduce input token costs
# - only extract first 2 pages of pdf attachments to keep token counts manageable
#
# year-specific outputs (nov 2024):
# - output_csv now includes year suffix: makeup_data_2015.csv, makeup_data_2020.csv, etc.
# - prevents overwriting data from different years when running multi-year analyses
# - each year's classification results are preserved independently for comparison
# - downstream plotting scripts (makeup_plots.py) should also use year-specific paths
#
# enhanced output messages (nov 2024):
# - added prominent "SAVED CLASSIFIED COMMENTS" banner with file path
# - prints absolute path to make it easy to find output files
# - metadata classification phase now prints when results are saved
# - helps track pipeline progress and locate intermediate outputs
#
# responses api migration + bug fix (nov 2024):
# - PROBLEM: Massive "'NoneType' object is not subscriptable" errors flooding output
# - ROOT CAUSE: classify_comment was doing response.output[0].content[0].text without checking
#   if response.output was None or empty. When API calls failed or returned partial responses,
#   this line threw TypeError which was caught and printed as "API call failed", but the None
#   return value broke downstream code expecting (category, prompt, response) tuples.
# - FIX #1: Use response.output_text helper (doesn't explode on None) as primary extraction path
# - FIX #2: Add fallback with safe attribute access using getattr(..., None) or [] guards
# - FIX #3: Added retry logic (3 attempts with exponential backoff) for transient failures
# - FIX #4: NEVER return None - always return (LABEL_MAP["undecided"], prompt, error_msg) to keep
#   pipeline stable even after all retries exhausted
# - FIX #5: Fixed API parameters for gpt-5-nano compatibility:
#   * Removed text={"verbosity": "low"} which was interfering with response format
#   * Removed reasoning.effort (not supported by gpt-5-nano, causes 400 errors)
# - FIX #6: Changed input format from [{"role": "user", "content": prompt}] to just prompt
#   (simpler + recommended for Responses API)
# - METADATA IN PROMPT: Now includes organization, submitter_type, first_name, last_name in prompt
#   so LLM has full context to classify (user request: "make sure we're getting both the 
#   organization name and the first/last name into the LLM context")
# - RESULT: Classification runs without errors, all comments get classified (no None gaps)
# - REFERENCE: https://platform.openai.com/docs/api-reference/responses (output_text helper)
#              https://platform.openai.com/docs/guides/migrate-to-responses (best practices)
#

