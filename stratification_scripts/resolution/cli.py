"""Standalone document-resolution command line interface."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import polars as pl

from .. import config
from ..federal_register.client import FederalRegisterClient
from ..logging_utils import get_logger
from ..reginfo.client import RegInfoClient
from .inputs import refs_from_goldset_packet, refs_from_snapshot
from .models import CommentRef, ResolutionResult
from .resolver import DocumentResolver, qualifying_candidates

logger = get_logger(__name__)


def write_run(
    *, results: List[ResolutionResult], manifest: dict, out_dir: Path
) -> Path:
    """Write JSONL, CSV summary, and manifest without overwriting a run."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty run dir: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "resolutions.jsonl").open("w") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict()) + "\n")
    rows = []
    for result in results:
        qualifying = qualifying_candidates(result)
        rows.append({
            "comment_id": result.comment_id,
            "status": result.status.value,
            "absence_reason": (
                result.absence_reason.value if result.absence_reason else ""
            ),
            "n_candidates": len(result.candidates),
            "n_qualifying": len(qualifying),
            "first_qualifying": (
                qualifying[0].document_number if qualifying else ""
            ),
            "channels_failed": ";".join(
                channel.value
                for channel, state in result.channels_run.items()
                if state != "ok"
            ),
        })
    pl.DataFrame(rows).write_csv(out_dir / "summary.csv")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return out_dir


def _resolve_refs(
    refs: List[CommentRef],
) -> tuple[List[ResolutionResult], dict]:
    fr = FederalRegisterClient(max_retries=6, sleep_between=0.4)
    reginfo = RegInfoClient()
    resolver = DocumentResolver(fr_client=fr, reginfo_client=reginfo)
    results: List[ResolutionResult] = []
    try:
        for index, ref in enumerate(refs, start=1):
            results.append(resolver.resolve(ref))
            if index % 10 == 0:
                logger.info(f"resolved {index}/{len(refs)}")
    finally:
        fr.close()
        reginfo.close()
    return results, resolver.cache.stats


def cmd_resolve(args) -> int:
    if args.seed_id:
        seed_refs = refs_from_goldset_packet(args.seed_id)
        dated = {
            ref.comment_id: ref
            for ref in refs_from_snapshot(
                args.snapshot,
                year=args.year,
                comment_ids=[ref.comment_id for ref in seed_refs],
            )
        }
        refs = [dated.get(ref.comment_id, ref) for ref in seed_refs]
        undated = [ref.comment_id for ref in refs if not ref.comment_date]
        if undated:
            logger.warning(
                f"{len(undated)} packet row(s) have no comment_date "
                f"(absent from the snapshot) and will resolve UNKNOWN: "
                f"{', '.join(undated)}"
            )
    else:
        refs = refs_from_snapshot(
            args.snapshot,
            year=args.year,
            comment_ids=args.comment_id or None,
            limit=args.limit,
        )
    if not refs:
        logger.error("no comments selected")
        return 2
    started_at = datetime.now().isoformat()
    results, cache_stats = _resolve_refs(refs)
    run_id = args.run_id or f"{datetime.now():%Y-%m-%d}-{args.snapshot}"
    out_dir = write_run(
        results=results,
        manifest={
            "snapshot": args.snapshot,
            "year": args.year,
            "seed_id": args.seed_id,
            "rows": len(results),
            "cache_stats": cache_stats,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "note": "agenda data is time-varying; re-runs can differ",
        },
        out_dir=config.get_resolution_run_path(run_id),
    )
    logger.info(f"wrote {len(results)} resolutions to {out_dir}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stratification_scripts.resolution"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    resolve = sub.add_parser(
        "resolve", help="Resolve response venues for comments."
    )
    resolve.add_argument("--snapshot", required=True, help="Frozen snapshot id.")
    resolve.add_argument("--year", type=int, default=2024)
    resolve.add_argument("--seed-id", default=None)
    resolve.add_argument("--comment-id", action="append", default=[])
    resolve.add_argument("--limit", type=int, default=None)
    resolve.add_argument("--run-id", default=None)
    resolve.set_defaults(func=cmd_resolve)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if argv is not None:
            return int(exc.code)
        raise
    return args.func(args)
