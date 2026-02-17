#!/usr/bin/env python3
"""
Deduplicate comments within each document using embedding-based similarity.

Identifies near-duplicate comments (0.95+ cosine similarity) using sentence-transformers
and groups them together, marking one as canonical for efficient classification.

Example:
    # CLI usage
    $ python -m stratification_scripts.makeup.deduplicate_comments --year 2024
    
    # Programmatic usage
    >>> from stratification_scripts.makeup.deduplicate_comments import deduplicate_comments_for_year
    >>> from stratification_scripts.config import PipelineConfig
    >>> 
    >>> config = PipelineConfig(year=2024)
    >>> deduplicate_comments_for_year(config)
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

from stratification_scripts.config import (
    PipelineConfig,
    get_comments_raw_path,
)
from stratification_scripts.logging_utils import (
    get_logger,
    log_banner,
    setup_logging,
)

logger = get_logger(__name__)

# Threshold for switching from numpy (O(N^2) memory) to FAISS (constant memory).
# Below this, the full similarity matrix fits comfortably in RAM.
# 5000 comments -> 5000x5000 float32 matrix -> ~100MB, well within limits.
_FAISS_THRESHOLD_N = 5000


def compute_comment_embeddings(
    comments: List[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Generate embeddings for comments using sentence-transformers.
    
    Args:
        comments: List of comment text strings
        model: Loaded SentenceTransformer model
    
    Returns:
        numpy array of shape (n_comments, embedding_dim) with normalized embeddings
    """
    if not comments:
        return np.array([]).reshape(0, model.get_sentence_embedding_dimension())
    
    # Generate embeddings in batch
    embeddings = model.encode(
        comments,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    
    return embeddings


class _UnionFind:
    """Disjoint-set / union-find for building connected components from edges."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # union by rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self) -> Dict[int, List[int]]:
        """Return groups that contain more than one element."""
        from collections import defaultdict
        components: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            components[self.find(i)].append(i)
        # Filter to multi-element groups and renumber
        result: Dict[int, List[int]] = {}
        gid = 0
        for members in components.values():
            if len(members) > 1:
                result[gid] = members
                gid += 1
        return result


def _find_duplicate_groups_numpy(
    embeddings: np.ndarray,
    threshold: float,
) -> Dict[int, List[int]]:
    """
    Find duplicate groups using a full NxN cosine similarity matrix (numpy).

    Suitable for small N (below _FAISS_THRESHOLD_N) where the full matrix
    fits in RAM.  Embeddings must already be L2-normalized.
    """
    n = len(embeddings)
    # Compute full similarity matrix -- O(N^2) memory
    similarity_matrix = np.dot(embeddings, embeddings.T)
    adjacency = similarity_matrix >= threshold

    # Connected components via BFS
    visited: Set[int] = set()
    groups: Dict[int, List[int]] = {}
    group_id = 0

    for i in range(n):
        if i in visited:
            continue

        group: List[int] = []
        stack = [i]

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            group.append(node)

            neighbors = np.where(adjacency[node])[0]
            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)

        if len(group) > 1:
            groups[group_id] = group
            group_id += 1

    return groups


def _find_duplicate_groups_faiss(
    embeddings: np.ndarray,
    threshold: float,
) -> Dict[int, List[int]]:
    """
    Find duplicate groups using FAISS range_search -- constant memory per query.

    Uses IndexFlatIP (inner product on L2-normalized vectors = cosine similarity)
    with range_search to find all pairs above the threshold without materializing
    the full NxN similarity matrix.  Connected components are built with union-find.

    Embeddings must already be L2-normalized and contiguous float32.
    """
    n, d = embeddings.shape

    # FAISS requires contiguous float32
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)

    # IndexFlatIP: exact inner-product search (= cosine sim for unit vectors)
    index = faiss.IndexFlatIP(d)
    index.add(emb)

    # range_search returns all (query, result) pairs with score >= threshold.
    # lims is an (n+1,) array: results for query i are I[lims[i]:lims[i+1]].
    lims, D, I = index.range_search(emb, threshold)

    # Build connected components via union-find from the sparse edge list
    uf = _UnionFind(n)
    for i in range(n):
        start, end = int(lims[i]), int(lims[i + 1])
        for idx in range(start, end):
            j = int(I[idx])
            if j != i:
                uf.union(i, j)

    return uf.groups()


def find_duplicate_groups(
    embeddings: np.ndarray,
    threshold: float = 0.95,
) -> Dict[int, List[int]]:
    """
    Find duplicate groups using cosine similarity.

    Uses transitive closure: if A~=B and B~=C, then A~=B~=C are in the same group.

    For small inputs (N <= 5000), uses an in-memory numpy dot product.
    For larger inputs, uses FAISS IndexFlatIP.range_search to avoid
    allocating the full NxN similarity matrix (which causes OOM at ~50K).

    Args:
        embeddings: Normalized embeddings array (n_comments, embedding_dim)
        threshold: Cosine similarity threshold (default: 0.95)

    Returns:
        Dict mapping group_id to list of comment indices in that group
    """
    n = len(embeddings)
    if n <= 1:
        return {}

    if n <= _FAISS_THRESHOLD_N:
        logger.debug(f"Using numpy path for {n} embeddings (below threshold {_FAISS_THRESHOLD_N})")
        return _find_duplicate_groups_numpy(embeddings, threshold)

    # Large dataset -- require FAISS
    if not _FAISS_AVAILABLE:
        raise ImportError(
            f"FAISS is required for deduplication of large comment sets "
            f"(N={n:,} > {_FAISS_THRESHOLD_N:,}). The full NxN similarity matrix "
            f"would require ~{n * n * 4 / 1e9:.1f} GB of RAM. "
            f"Install FAISS: pip install faiss-cpu"
        )

    logger.debug(f"Using FAISS path for {n} embeddings (above threshold {_FAISS_THRESHOLD_N})")
    return _find_duplicate_groups_faiss(embeddings, threshold)


def deduplicate_document_comments(
    doc_df: pl.DataFrame,
    model: SentenceTransformer,
    threshold: float,
) -> pl.DataFrame:
    """
    Deduplicate comments for a single document.
    
    Args:
        doc_df: DataFrame with comments for one document
        model: SentenceTransformer model
        threshold: Similarity threshold
    
    Returns:
        DataFrame with duplicate_group_id, canonical_comment_id, is_canonical columns added
    """
    n_comments = len(doc_df)
    
    # Handle edge cases
    if n_comments <= 1:
        # No duplicates possible
        return doc_df.with_columns([
            pl.lit(None).cast(pl.Utf8).alias("duplicate_group_id"),
            pl.col("comment_id").alias("canonical_comment_id"),
            pl.lit(True).alias("is_canonical"),
        ])
    
    # Extract comment text (combine comment_text + attachment_text)
    comment_texts = []
    for row in doc_df.iter_rows(named=True):
        text_parts = []
        
        comment_text = row.get("comment_text")
        if comment_text and str(comment_text).strip() and str(comment_text).strip().lower() != "none":
            text_parts.append(str(comment_text).strip())
        
        attachment_text = row.get("attachment_text")
        if attachment_text and str(attachment_text).strip() and str(attachment_text).strip().lower() != "none":
            text_parts.append(str(attachment_text).strip())
        
        combined_text = "\n\n".join(text_parts) if text_parts else ""
        comment_texts.append(combined_text)
    
    # Filter out empty comments
    valid_indices = [i for i, text in enumerate(comment_texts) if text.strip()]
    
    if len(valid_indices) <= 1:
        # No duplicates possible (all empty or only one valid)
        return doc_df.with_columns([
            pl.lit(None).cast(pl.Utf8).alias("duplicate_group_id"),
            pl.col("comment_id").alias("canonical_comment_id"),
            pl.lit(True).alias("is_canonical"),
        ])
    
    # Generate embeddings only for valid comments
    valid_texts = [comment_texts[i] for i in valid_indices]
    valid_embeddings = compute_comment_embeddings(valid_texts, model)
    
    # Find duplicate groups (returns indices relative to valid_indices)
    duplicate_groups = find_duplicate_groups(valid_embeddings, threshold)
    
    # Initialize columns
    duplicate_group_ids = [None] * n_comments
    canonical_comment_ids = [None] * n_comments
    is_canonical_flags = [True] * n_comments
    
    # Get comment_ids as list for easy access
    comment_ids = doc_df["comment_id"].to_list()
    
    # Process duplicate groups
    for group_id, group_indices in duplicate_groups.items():
        # Map back to original indices
        original_indices = [valid_indices[i] for i in group_indices]
        
        # Choose first comment as canonical
        canonical_idx = original_indices[0]
        canonical_comment_id = comment_ids[canonical_idx]
        
        # Generate UUID for duplicate group
        group_uuid = str(uuid.uuid4())
        
        # Mark all comments in group
        for idx in original_indices:
            duplicate_group_ids[idx] = group_uuid
            canonical_comment_ids[idx] = canonical_comment_id
            is_canonical_flags[idx] = (idx == canonical_idx)
    
    # Fill in non-duplicate comments (they are their own canonical)
    for i in range(n_comments):
        if duplicate_group_ids[i] is None:
            canonical_comment_ids[i] = comment_ids[i]
            is_canonical_flags[i] = True
    
    # Add columns to DataFrame
    # Explicitly set dtype to avoid Null vs String type mismatch during concat
    return doc_df.with_columns([
        pl.Series("duplicate_group_id", duplicate_group_ids, dtype=pl.Utf8),
        pl.Series("canonical_comment_id", canonical_comment_ids, dtype=pl.Utf8),
        pl.Series("is_canonical", is_canonical_flags),
    ])


def deduplicate_comments_for_year(config: PipelineConfig) -> None:
    """
    Deduplicate comments for a single year.
    
    This is the main entry point for programmatic use.
    
    Args:
        config: Pipeline configuration
    
    Side Effects:
        Modifies comments_raw_{year}.csv in place, adding duplicate_group_id,
        canonical_comment_id, and is_canonical columns
    """
    if not config.enable_deduplication:
        logger.info("Deduplication disabled, skipping")
        return
    
    comments_csv = get_comments_raw_path(config.year)
    
    if not comments_csv.exists():
        logger.error(f"{comments_csv} not found. Run mine_comments.py first.")
        return
    
    log_banner(logger, f"DEDUPLICATING COMMENTS - YEAR {config.year}")
    logger.info(f"Loading comments from: {comments_csv}")
    
    # Load comments
    df = pl.read_csv(str(comments_csv))
    
    if len(df) == 0:
        logger.warning("No comments to deduplicate")
        return
    
    logger.info(f"Loaded {len(df):,} comments")
    
    # Check if deduplication already done
    if "duplicate_group_id" in df.columns:
        logger.info("Deduplication already completed, skipping")
        return
    
    # Load model
    logger.info(f"Loading embedding model: {config.deduplication_model}")
    try:
        model = SentenceTransformer(config.deduplication_model)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load deduplication model '{config.deduplication_model}': {e}. "
            f"Deduplication cannot proceed — downstream steps depend on this."
        ) from e
    
    logger.info(f"Using similarity threshold: {config.deduplication_threshold}")
    
    # Group by document_number
    grouped = df.group_by("document_number")
    n_documents = grouped.len().height
    
    logger.info(f"Processing {n_documents} documents...")
    
    # Process each document
    deduplicated_dfs = []
    total_groups = 0
    total_duplicates = 0
    
    for doc_number, doc_df in tqdm(grouped, total=n_documents, desc="Deduplicating"):
        deduplicated_doc = deduplicate_document_comments(
            doc_df,
            model,
            config.deduplication_threshold,
        )
        deduplicated_dfs.append(deduplicated_doc)
        
        # Count groups and duplicates
        if "duplicate_group_id" in deduplicated_doc.columns:
            groups = deduplicated_doc.filter(
                pl.col("duplicate_group_id").is_not_null()
            )
            if len(groups) > 0:
                n_groups = groups["duplicate_group_id"].n_unique()
                n_duplicates = len(groups) - n_groups  # Total in groups - canonical comments
                total_groups += n_groups
                total_duplicates += n_duplicates
    
    # Combine all documents
    df_deduplicated = pl.concat(deduplicated_dfs)
    
    # Save back to CSV
    comments_csv.parent.mkdir(parents=True, exist_ok=True)
    df_deduplicated.write_csv(str(comments_csv))
    
    log_banner(logger, "DEDUPLICATION COMPLETE")
    logger.info(f"Total comments: {len(df_deduplicated):,}")
    logger.info(f"Duplicate groups found: {total_groups:,}")
    logger.info(f"Duplicate comments: {total_duplicates:,}")
    logger.info(f"Canonical comments: {len(df_deduplicated) - total_duplicates:,}")
    logger.info(f"Output file: {comments_csv.absolute()}")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Deduplicate comments using embedding-based similarity"
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Cosine similarity threshold (default: 0.95)")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence transformer model to use")
    parser.add_argument("--disable", action="store_true",
                        help="Disable deduplication (no-op)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    
    setup_logging(verbose=args.verbose, quiet=args.quiet, year=args.year)
    
    config = PipelineConfig(
        year=args.year,
        deduplication_threshold=args.threshold,
        deduplication_model=args.model,
        enable_deduplication=not args.disable,
        verbose=args.verbose,
        quiet=args.quiet,
    )
    
    try:
        deduplicate_comments_for_year(config)
        return 0
    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

