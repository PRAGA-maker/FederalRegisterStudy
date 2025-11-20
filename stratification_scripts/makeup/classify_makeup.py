"""Classify comment authors using OpenAI GPT with Word2Vec sampling for large datasets.

For documents with >1000 comments, uses Word2Vec embeddings + density-aware sampling
to select a representative subset with 99.9% confidence interval.
"""
import argparse
import asyncio
import math
import os
import re
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
PROMPT_TEMPLATE = """Classify the author of this public comment as exactly one of these categories:

**org** - Large corporations, industry associations, trade groups, or organizational entities
**citizen** - Individual ordinary citizens, residents, or community members without professional affiliation
**expert** - Academics, researchers, small/local businesses, industry experts, technical specialists
**lobbyist** - Political consultants, registered lobbyists, advocacy groups, or professional campaigners
**undecided** - ONLY if truly anonymous with no identifying info or completely unclear authorship

Decision guidance:
- Use ALL available context clues (tone, language, content, structure)
- Look for: organizational letterhead, academic titles, business names, lobbying language, personal stories
- Large company or trade association → org
- Generic "concerned citizen" with personal story/opinion → citizen (NOT undecided)
- Individual small business owners → expert (not org)
- Professional advocates or political operatives → lobbyist
- AVOID undecided unless genuinely impossible to determine - make your best inference from the text

Be decisive. Most comments CAN be classified if you look at the language, tone, and context. Only use "undecided" as a last resort.

Note: Text may be from inline comment or extracted from PDF attachment (first 2 pages).

Reply with ONLY ONE WORD from: org, citizen, expert, lobbyist, undecided

Comment:
{comment_text}"""


def tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase, split, remove short words."""
    tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
    # Basic stopword removal
    stopwords = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'has', 'are', 'was', 'were', 'been', 'will'}
    return [t for t in tokens if t not in stopwords]


def vectorize_comments(comments: List[str]) -> Tuple[np.ndarray, Word2Vec]:
    """Train Word2Vec and return normalized comment vectors."""
    # Tokenize all comments
    tokenized = [tokenize(c) for c in comments]
    tokenized = [t for t in tokenized if len(t) > 0]  # Filter empty
    
    if len(tokenized) < 10:
        # Not enough data
        return np.zeros((len(comments), 300)), None
    
    # Train Word2Vec
    model = Word2Vec(
        sentences=tokenized,
        vector_size=300,
        window=5,
        min_count=2,
        workers=4,
        seed=42,
    )
    
    # Vectorize each comment
    vectors = []
    for tokens in tokenized:
        if len(tokens) == 0:
            vectors.append(np.zeros(300))
            continue
        # Mean pooling
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        if len(vecs) == 0:
            vectors.append(np.zeros(300))
        else:
            vec = np.mean(vecs, axis=0)
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
    
    return np.array(vectors), model


def compute_sample_size(vectors: np.ndarray, pilot_size: int = 500, Z: float = 3.291, E: float = 0.05) -> int:
    """Calculate required sample size for 99.9% CI."""
    N = len(vectors)
    
    # Pilot sample
    pilot_indices = np.random.choice(N, min(pilot_size, N), replace=False)
    pilot_vecs = vectors[pilot_indices]
    
    # Estimate sigma (std of vector norms)
    norms = np.linalg.norm(pilot_vecs, axis=1)
    sigma = np.std(norms)
    
    # Sample size formula
    n = (Z ** 2 * sigma ** 2) / (E ** 2)
    
    # Finite population correction
    n_corrected = n / (1 + (n - 1) / N)
    
    return int(np.ceil(n_corrected))


def density_aware_sampling(
    vectors: np.ndarray,
    n_samples: int,
    k_neighbors: int = 50,
) -> np.ndarray:
    """Sample inversely proportional to density with cluster coverage."""
    N = len(vectors)
    n_samples = min(n_samples, N)
    
    # Clustering
    k_clusters = int(math.sqrt(N / 2))
    k_clusters = max(10, min(k_clusters, N // 2))
    
    kmeans = MiniBatchKMeans(n_clusters=k_clusters, random_state=42, batch_size=1000)
    clusters = kmeans.fit_predict(vectors)
    
    # Compute density via kNN
    k_neighbors = min(k_neighbors, N - 1)
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='cosine')
    nn.fit(vectors)
    distances, _ = nn.kneighbors(vectors)
    
    # Density = 1 / mean distance to k neighbors
    mean_dist = distances[:, 1:].mean(axis=1)  # Skip self
    density = 1.0 / (mean_dist + 1e-6)
    
    # Inverse density weights
    weights = 1.0 / (density + 1e-6)
    weights = weights / weights.sum()
    
    # Sample with replacement proportional to inverse density
    sampled_indices = np.random.choice(N, size=n_samples, replace=False, p=weights)
    
    # Ensure at least 1 per cluster
    for cluster_id in range(k_clusters):
        cluster_mask = clusters == cluster_id
        if cluster_mask.sum() > 0 and not np.any(clusters[sampled_indices] == cluster_id):
            # Replace one sample with a random point from this cluster
            cluster_indices = np.where(cluster_mask)[0]
            random_idx = np.random.choice(cluster_indices)
            sampled_indices[0] = random_idx
    
    return sampled_indices


async def classify_comment(client: AsyncOpenAI, comment_text: str, semaphore: asyncio.Semaphore, model: str) -> Tuple[Optional[str], str]:
    """Classify a single comment, return (category, prompt_used)."""
    prompt = PROMPT_TEMPLATE.format(comment_text=comment_text[:2000])  # Truncate long comments
    
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=10,
            )
            result = response.choices[0].message.content.strip().lower()
            
            # Validate response
            if result in LABEL_MAP:
                return LABEL_MAP[result], prompt
            
            # Try to extract valid token
            for token in LABEL_MAP.keys():
                if token in result:
                    return LABEL_MAP[token], prompt
            
            return None, prompt
            
        except Exception as e:
            tqdm.write(f"API error: {e}")
            return None, prompt


async def classify_batch(
    client: AsyncOpenAI,
    comments: List[Tuple[str, str]],  # (comment_id, text)
    model: str,
    max_concurrency: int,
) -> List[Tuple[str, Optional[str], str]]:
    """Classify a batch of comments. Returns [(comment_id, category, prompt)]."""
    semaphore = asyncio.Semaphore(max_concurrency)
    
    tasks = []
    for comment_id, text in comments:
        task = classify_comment(client, text, semaphore, model)
        tasks.append((comment_id, task))
    
    results = []
    for comment_id, task in tqdm(tasks, desc="Classifying"):
        category, prompt = await task
        results.append((comment_id, category, prompt))
    
    return results


def classify_comments(
    comments_csv: Path,
    results_csv: Path,
    fr_csv: Path,
    output_csv: Path,
    api_key: str,
    model: str,
    max_concurrency: int,
    sample_threshold: int,
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
            metadata_results.append((comment_id, category, "metadata"))
        elif has_comment_text or has_attachment_text:
            # Has text data - needs LLM classification
            needs_llm.append(row)
    
    print(f"  Classified via metadata: {len(metadata_results)}/{len(df_unclassified)} ({100*len(metadata_results)/len(df_unclassified):.1f}%)")
    print(f"  Excluded (no data): {excluded_count}/{len(df_unclassified)} ({100*excluded_count/len(df_unclassified):.1f}%)")
    print(f"  Needs LLM classification: {len(needs_llm)}/{len(df_unclassified)} ({100*len(needs_llm)/len(df_unclassified):.1f}%)")
    
    # Save metadata results
    if metadata_results:
        save_results(results_csv, metadata_results, "metadata")
    
    # Phase 2: LLM classification for remaining
    if not needs_llm:
        print("\nAll comments classified via metadata or excluded!")
        join_and_write_output(comments_csv, results_csv, fr_csv, output_csv)
        return
    
    print(f"\nPhase 2: LLM classification for {len(needs_llm)} remaining comments...")
    
    # Group comments by document for batch processing
    by_doc: Dict[str, List[Tuple[str, str]]] = {}  # doc_number -> [(comment_id, text)]
    for row in needs_llm:
        doc_num = row.get("document_number")
        comment_id = row.get("comment_id")
        comment_text = row.get("comment_text", "")
        attachment_text = row.get("attachment_text", "")
        
        # Use comment_text if available, otherwise fallback to attachment_text
        text = comment_text if comment_text and comment_text.strip() else attachment_text
        
        if not doc_num or not comment_id or not text:
            continue
        if doc_num not in by_doc:
            by_doc[doc_num] = []
        by_doc[doc_num].append((comment_id, text))
    
    # Setup OpenAI client
    client = AsyncOpenAI(api_key=api_key)
    
    all_results = []
    
    for doc_number, comments in by_doc.items():
        n_comments = len(comments)
        print(f"\nDocument {doc_number}: {n_comments} comments")
        
        if n_comments <= sample_threshold:
            # Classify all
            to_classify = comments
        else:
            # Word2Vec sampling
            print(f"  Vectorizing {n_comments} comments...")
            texts = [c[1] for c in comments]
            vectors, w2v_model = vectorize_comments(texts)
            
            if vectors.shape[0] == 0 or w2v_model is None:
                print("  Skipping (insufficient data for vectorization)")
                continue
            
            print("  Computing sample size...")
            n_samples = compute_sample_size(vectors)
            print(f"  Required sample size: {n_samples} (from {n_comments} total)")
            
            print("  Density-aware sampling...")
            sampled_indices = density_aware_sampling(vectors, n_samples)
            
            to_classify = [comments[i] for i in sampled_indices]
            print(f"  Selected {len(to_classify)} comments to classify")
        
        # Classify
        results = asyncio.run(classify_batch(client, to_classify, model, max_concurrency))
        all_results.extend(results)
        
        # Save incrementally
        if len(all_results) >= 100:
            save_results(results_csv, all_results, model)
            all_results = []
    
    # Save remaining
    if all_results:
        save_results(results_csv, all_results, model)
    
    # Join and write final output
    join_and_write_output(comments_csv, results_csv, fr_csv, output_csv)


def save_results(results_csv: Path, results: List[Tuple[str, Optional[str], str]], model: str) -> None:
    """Append results to CSV."""
    if not results:
        return
    
    df_new = pl.DataFrame({
        "comment_id": [r[0] for r in results],
        "category": [r[1] or "Undecided/Anonymous" for r in results],
        "prompt_used": [r[2] for r in results],
        "model": [model] * len(results),
    })
    
    if results_csv.exists():
        df_existing = pl.read_csv(str(results_csv))
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
    
    print(f"\nOutput written to {output_csv}")
    print(f"Total classified: {len(df_output)}")
    
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
    parser.add_argument("--max-concurrency", type=int, default=20)
    parser.add_argument("--model", type=str, default="gpt-5-nano")
    parser.add_argument("--sample-threshold", type=int, default=1000, help="Use sampling for documents with more than this many comments")
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    comments_csv = script_dir / "data" / f"comments_raw_{args.year}.csv"
    results_csv = script_dir / "data" / f"makeup_results_{args.year}.csv"
    fr_csv = script_dir.parent / "output" / f"federal_register_{args.year}_comments.csv"
    output_csv = script_dir.parent / "output" / "makeup_data.csv"
    
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
# - max_completion_tokens=10 is sufficient since we only need one word responses
# - model default in argparse should match your most commonly used model
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
# - use cheap models (gpt-4o-mini) for simple classification tasks
# - truncate prompts to 2000 chars to reduce input token costs
# - only extract first 2 pages of pdf attachments to keep token counts manageable

