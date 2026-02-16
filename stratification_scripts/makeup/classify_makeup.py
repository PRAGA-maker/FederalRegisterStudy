#!/usr/bin/env python3
"""
Classify comment authors using OpenAI GPT with Word2Vec sampling.

For documents with >1000 comments, uses Word2Vec embeddings + density-aware
sampling to select a representative subset with 99.9% confidence interval.

Example:
    # CLI usage
    $ python -m stratification_scripts.makeup.classify_makeup --year 2024
    
    # Programmatic usage
    >>> from stratification_scripts.makeup.classify_makeup import classify_comments_for_year
    >>> from stratification_scripts.config import PipelineConfig
    >>> 
    >>> config = PipelineConfig(year=2024, openai_model="gpt-4o-mini")
    >>> classify_comments_for_year(config)
"""

from __future__ import annotations

import argparse
import asyncio
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
from gensim.models import Word2Vec
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from stratification_scripts.config import (
    PipelineConfig,
    get_openai_api_key,
    get_comments_raw_path,
    get_makeup_results_path,
    get_makeup_data_path,
    get_fr_csv_path,
)
from stratification_scripts.logging_utils import (
    get_logger,
    log_banner,
    setup_logging,
)
from stratification_scripts.openai_client import (
    OpenAIClassifier,
    classify_from_metadata,
    LABEL_MAP,
)

logger = get_logger(__name__)


def tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase, split, remove short words."""
    tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
    stopwords = {
        'the', 'and', 'for', 'with', 'this', 'that', 'from',
        'have', 'has', 'are', 'was', 'were', 'been', 'will'
    }
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

    # Ensure cluster coverage
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


def propagate_results_to_duplicates(
    results: List[Tuple[str, Optional[str], str, str]],
    df_comments: pl.DataFrame,
) -> List[Tuple[str, Optional[str], str, str]]:
    """
    Propagate classification results to duplicate comments.
    
    For each canonical comment result, find all duplicates in the same group
    and create result entries for them with the same classification.
    
    Args:
        results: List of (comment_id, category, prompt, model_response) tuples
        df_comments: DataFrame with duplicate_group_id and canonical_comment_id columns
    
    Returns:
        Expanded list of results including duplicates
    """
    if "duplicate_group_id" not in df_comments.columns:
        # No deduplication columns, return as-is
        return results
    
    # Build mapping from canonical_comment_id to result
    canonical_to_result = {r[0]: r for r in results}
    
    # Find all duplicates that need results propagated
    expanded_results = list(results)  # Start with canonical results
    
    # Group by duplicate_group_id to find duplicates
    duplicate_groups = df_comments.filter(
        pl.col("duplicate_group_id").is_not_null() & 
        (pl.col("is_canonical") == False)
    )
    
    if len(duplicate_groups) > 0:
        for row in duplicate_groups.iter_rows(named=True):
            duplicate_id = row["comment_id"]
            canonical_id = row["canonical_comment_id"]
            
            # If canonical was classified, propagate to duplicate
            if canonical_id in canonical_to_result:
                canonical_result = canonical_to_result[canonical_id]
                # Create duplicate result with same classification
                duplicate_result = (
                    duplicate_id,
                    canonical_result[1],  # category
                    canonical_result[2],  # prompt
                    canonical_result[3],  # model_response
                )
                expanded_results.append(duplicate_result)
    
    return expanded_results


def save_results(
    results_csv: Path,
    results: List[Tuple[str, Optional[str], str, str]],
    model: str,
    df_comments: Optional[pl.DataFrame] = None,
) -> None:
    """
    Append results to CSV, optionally propagating to duplicates.
    
    Args:
        results_csv: Path to results CSV
        results: List of (comment_id, category, prompt, model_response) tuples
        model: Model name
        df_comments: Optional DataFrame with comments (for duplicate propagation)
    """
    if not results:
        return
    
    # Propagate results to duplicates if comments DataFrame provided
    if df_comments is not None:
        results = propagate_results_to_duplicates(results, df_comments)
    
    df_new = pl.DataFrame({
        "comment_id": [r[0] for r in results],
        "category": [r[1] or "Undecided/Anonymous" for r in results],
        "model_response": [r[3] for r in results],
        "prompt_used": [r[2] for r in results],
        "model": [model] * len(results),
    })
    
    if results_csv.exists():
        df_existing = pl.read_csv(str(results_csv))
        if "model_response" not in df_existing.columns:
            df_existing = df_existing.with_columns(pl.lit(None).alias("model_response"))
        df_combined = pl.concat([df_existing, df_new])
    else:
        df_combined = df_new
    
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    df_combined.write_csv(str(results_csv))


def join_and_write_output(
    comments_csv: Path,
    results_csv: Path,
    fr_csv: Path,
    output_csv: Path,
) -> None:
    """Join comments + results + FR data and write makeup_data.csv."""
    logger.info("Joining results...")
    
    df_comments = pl.read_csv(str(comments_csv))
    df_results = pl.read_csv(str(results_csv))
    
    df_joined = df_comments.join(df_results, on="comment_id", how="left")
    
    if fr_csv.exists():
        df_fr = pl.read_csv(str(fr_csv))
        if "agency" in df_fr.columns and "document_number" in df_fr.columns:
            df_joined = df_joined.join(
                df_fr.select(["document_number", "agency"]),
                on="document_number",
                how="left",
            )
    
    cols = ["document_number", "comment_id", "category"]
    if "agency" in df_joined.columns:
        cols.append("agency")
    if "lifecycle_stage" in df_joined.columns:
        cols.append("lifecycle_stage")
    if "rin" in df_joined.columns:
        cols.append("rin")

    df_output = df_joined.select(cols).filter(pl.col("category").is_not_null())
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_output.write_csv(str(output_csv))
    
    log_banner(logger, "SAVED CLASSIFIED COMMENTS")
    logger.info(f"Output file: {output_csv.absolute()}")
    logger.info(f"Total classified: {len(df_output)}")
    
    if len(df_output) > 0:
        logger.info("\nCategory breakdown:")
        for cat in LABEL_MAP.values():
            count = df_output.filter(pl.col("category") == cat).shape[0]
            pct = 100.0 * count / len(df_output)
            logger.info(f"  {cat}: {count} ({pct:.1f}%)")


async def process_classification_async(
    classifier: OpenAIClassifier,
    by_doc: Dict[str, List[Tuple[str, str, Dict[str, str]]]],
    results_csv: Path,
    model: str,
    max_concurrency: int,
    sample_threshold: int,
    sampling_seed: Optional[int],
    df_comments: Optional[pl.DataFrame] = None,
) -> None:
    """Process all documents in async context."""
    execution_queue: List[Tuple[str, str, Dict[str, str]]] = []
    buffer_size_threshold = 10000
    
    for doc_number, comments in by_doc.items():
        n_comments = len(comments)
        logger.info(f"\nDocument {doc_number}: {n_comments} comments")
        
        if n_comments <= sample_threshold:
            to_classify = comments
        else:
            logger.info(f"  Vectorizing {n_comments} comments...")
            texts = [c[1] for c in comments]
            vectors, w2v_model = vectorize_comments(texts)
            
            if vectors.shape[0] == 0 or w2v_model is None:
                logger.info("  Skipping (insufficient data)")
                continue
            
            n_samples = compute_sample_size(vectors, random_state=sampling_seed)
            if n_samples <= 0:
                n_samples = min(1, len(comments))
            logger.info(f"  Required sample size: {n_samples}")
            
            sampled_indices = density_aware_sampling(
                vectors, n_samples, random_state=sampling_seed
            )
            to_classify = [comments[i] for i in sampled_indices]
            logger.info(f"  Selected {len(to_classify)} comments")
        
        if to_classify:
            execution_queue.extend(to_classify)
            
            if len(execution_queue) >= buffer_size_threshold:
                batch = execution_queue[:]
                execution_queue.clear()
                logger.info(f"\nProcessing batch of {len(batch)} comments...")
                results = await classifier.classify_batch(batch, max_concurrency)
                save_results(results_csv, results, model, df_comments)
    
    if execution_queue:
        logger.info(f"\nProcessing final {len(execution_queue)} comments...")
        results = await classifier.classify_batch(execution_queue, max_concurrency)
        save_results(results_csv, results, model, df_comments)


def classify_comments_for_year(config: PipelineConfig) -> None:
    """
    Classify comments for a single year.
    
    This is the main entry point for programmatic use.
    
    Args:
        config: Pipeline configuration
    
    Side Effects:
        Makes API calls to OpenAI
        Writes classification results to CSV
    """
    comments_csv = get_comments_raw_path(config.year)
    results_csv = get_makeup_results_path(config.year)
    fr_csv = get_fr_csv_path(config.year)
    output_csv = get_makeup_data_path(config.year)
    
    api_key = get_openai_api_key(required=True)
    
    if not comments_csv.exists():
        logger.error(f"{comments_csv} not found. Run mine_comments.py first.")
        return
    
    df_comments = pl.read_csv(str(comments_csv))
    
    # Check if deduplication was performed
    has_deduplication = "is_canonical" in df_comments.columns
    
    if has_deduplication:
        canonical_count = df_comments.filter(pl.col("is_canonical") == True).shape[0]
        total_count = len(df_comments)
        duplicate_count = total_count - canonical_count
        logger.info(f"Deduplication detected: {canonical_count:,} canonical, {duplicate_count:,} duplicates")
    
    # Load existing results
    classified_ids: Set[str] = set()
    if results_csv.exists():
        df_results = pl.read_csv(str(results_csv))
        classified_ids = set(df_results["comment_id"].to_list())
        logger.info(f"Loaded {len(classified_ids)} already-classified comments")
    
    # Filter unclassified comments
    # If deduplication enabled, only classify canonical comments
    if has_deduplication:
        df_unclassified = df_comments.filter(
            (pl.col("is_canonical") == True) &
            (~pl.col("comment_id").is_in(list(classified_ids)))
        )
        logger.info(f"Filtering to canonical comments only: {len(df_unclassified):,} to classify")
    else:
        df_unclassified = df_comments.filter(~pl.col("comment_id").is_in(list(classified_ids)))
    
    if len(df_unclassified) == 0:
        logger.info("All comments already classified")
        if results_csv.exists():
            join_and_write_output(comments_csv, results_csv, fr_csv, output_csv)
        return
    
    logger.info(f"Classifying {len(df_unclassified)} comments...")
    
    # Phase 1: Metadata-based classification
    log_banner(logger, "PHASE 1: METADATA CLASSIFICATION")
    
    metadata_results = []
    needs_llm = []
    excluded_count = 0
    
    for row in df_unclassified.iter_rows(named=True):
        comment_id = row.get("comment_id")
        comment_text = row.get("comment_text", "")
        attachment_text = row.get("attachment_text", "")
        
        has_comment_text = bool(comment_text and str(comment_text).strip())
        has_attachment_text = bool(attachment_text and str(attachment_text).strip())
        
        if not has_comment_text and not has_attachment_text:
            excluded_count += 1
            continue
        
        category = classify_from_metadata(row)
        
        if category:
            metadata_results.append((comment_id, category, "metadata", "metadata"))
        elif has_comment_text or has_attachment_text:
            needs_llm.append(row)
    
    pct_metadata = 100 * len(metadata_results) / len(df_unclassified) if len(df_unclassified) > 0 else 0
    logger.info(f"Classified via metadata: {len(metadata_results)} ({pct_metadata:.1f}%)")
    logger.info(f"Excluded (no data): {excluded_count}")
    logger.info(f"Needs LLM: {len(needs_llm)}")
    
    if metadata_results:
        save_results(results_csv, metadata_results, "metadata", df_comments if has_deduplication else None)
    
    # Phase 2: LLM classification
    if not needs_llm:
        logger.info("All comments classified via metadata!")
        join_and_write_output(comments_csv, results_csv, fr_csv, output_csv)
        return
    
    log_banner(logger, "PHASE 2: LLM CLASSIFICATION")
    
    by_doc: Dict[str, List[Tuple[str, str, Dict[str, str]]]] = {}
    
    for row in needs_llm:
        doc_num = row.get("document_number")
        comment_id = row.get("comment_id")
        comment_text = row.get("comment_text")
        attachment_text = row.get("attachment_text")
        
        parts = []
        if comment_text and str(comment_text).strip() and str(comment_text).strip().lower() != "none":
            parts.append(str(comment_text).strip())
        if attachment_text and str(attachment_text).strip() and str(attachment_text).strip().lower() != "none":
            parts.append(str(attachment_text).strip())
            
        text = "\n\n".join(parts)
        
        if not doc_num or not comment_id or not text:
            continue
        
        metadata = {
            "organization": str(row.get("organization", "")),
            "submitter_type": str(row.get("submitter_type", "")),
            "first_name": str(row.get("first_name", "")),
            "last_name": str(row.get("last_name", "")),
        }
        
        by_doc.setdefault(doc_num, []).append((comment_id, text, metadata))
    
    if not by_doc:
        logger.info("No documents with usable text")
        join_and_write_output(comments_csv, results_csv, fr_csv, output_csv)
        return

    classifier = OpenAIClassifier(
        api_key=api_key,
        model=config.openai_model,
        max_retries=8,
    )
    
    # Pass df_comments for duplicate propagation
    asyncio.run(process_classification_async(
        classifier,
        by_doc,
        results_csv,
        config.openai_model,
        config.max_concurrency,
        config.sample_threshold,
        config.sampling_seed,
        df_comments if has_deduplication else None,
    ))
    
    join_and_write_output(comments_csv, results_csv, fr_csv, output_csv)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Classify comment authors with OpenAI"
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--max-concurrency", type=int, default=100)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--sample-threshold", type=int, default=1000,
                        help="Use sampling for docs with more than this many comments")
    parser.add_argument("--sampling-seed", type=int, default=None,
                        help="RNG seed for sampling reproducibility")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    
    setup_logging(verbose=args.verbose, quiet=args.quiet, year=args.year)
    
    config = PipelineConfig(
        year=args.year,
        max_concurrency=args.max_concurrency,
        openai_model=args.model,
        sample_threshold=args.sample_threshold,
        sampling_seed=args.sampling_seed,
        verbose=args.verbose,
        quiet=args.quiet,
    )
    
    try:
        classify_comments_for_year(config)
        return 0
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
