"""RTC parser CLI — standalone, never wired into the pipeline cli.py.

    python -m stratification_scripts.rtc_parser parse --pdf <path> --slug ccl5 \
        [--docket-prefix EPA-HQ-OW-2018-0594] [--source-url <url>] [--out <dir>]

Orchestrates extract -> clean -> (exhibit2 | responses | topics) -> crosswalk
and writes the crosswalk + supporting artifacts + a self-reporting manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from stratification_scripts import config
from stratification_scripts.rtc_parser import clean, crosswalk, exhibit2, extract, responses, topics
from stratification_scripts.rtc_parser.models import Commenter, CommentRecord

_EXHIBIT2_HEADER = "Exhibit 2: List of Public Commenters"
_SECTION2_MARKER = "2. Comments and EPA Responses by Topic"


def parse_pdf(
    pdf_path: Path | str, *, docket_prefix: str = "EPA-HQ-OW-2018-0594"
) -> tuple[list[Commenter], list[CommentRecord], dict[str, str], int]:
    """Full parse: returns (commenters, records, discussions, page_count)."""
    pages = extract.extract_pages(pdf_path)
    text = clean.strip_page_headers(pages)

    commenters = exhibit2.parse_commenters(text, docket_prefix)

    # Section 2 body: the marker also appears in the TOC, so search after Exhibit 2.
    lo = text.find(_EXHIBIT2_HEADER)
    sec2 = text.find(_SECTION2_MARKER, lo if lo != -1 else 0)
    body = text[sec2:] if sec2 != -1 else text

    blocks = responses.split_comment_blocks(body)
    discussions = topics.split_topic_discussions(body)
    records = crosswalk.assemble(commenters, blocks, discussions)
    return commenters, records, discussions, len(pages)


def _record_to_dict(r: CommentRecord) -> dict:
    return {
        "commenter_number": r.commenter_number,
        "document_id": r.document_id,
        "first_name": r.first_name,
        "last_name": r.last_name,
        "organization": r.organization,
        "comment_excerpt": r.comment_excerpt,
        "has_individual_response": r.has_individual_response,
        "topic_refs": [
            {"raw": t.raw, "canonical": t.canonical, "resolved": t.resolved}
            for t in r.topic_refs
        ],
        "individual_response_supplemental": r.individual_response_supplemental,
        "topic_discussions": r.topic_discussions,
    }


def _flat_row(r: CommentRecord) -> dict:
    return {
        "commenter_number": r.commenter_number,
        "document_id": r.document_id or "",
        "first_name": r.first_name,
        "last_name": r.last_name,
        "organization": r.organization,
        "has_individual_response": r.has_individual_response,
        "topics": "; ".join(t.canonical for t in r.topic_refs if t.resolved),
        "unresolved_topic_refs": "; ".join(t.raw for t in r.topic_refs if not t.resolved),
        "individual_response_supplemental": r.individual_response_supplemental,
        "comment_excerpt": r.comment_excerpt,
    }


def write_outputs(
    *,
    commenters: list[Commenter],
    records: list[CommentRecord],
    discussions: dict[str, str],
    manifest: dict,
    out_dir: Path,
) -> Path:
    """Write the five artifacts into out_dir; refuse to overwrite an existing run."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"rtc output already exists: {out_dir}")
    out_dir.mkdir(parents=True)

    pl.DataFrame(
        [
            {
                "number": c.number,
                "document_id": c.document_id,
                "first_name": c.first_name,
                "last_name": c.last_name,
                "organization": c.organization,
            }
            for c in commenters
        ]
    ).write_csv(out_dir / "commenters.csv")

    with (out_dir / "crosswalk.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(_record_to_dict(r)) + "\n")

    pl.DataFrame([_flat_row(r) for r in records]).write_csv(out_dir / "crosswalk.csv")

    (out_dir / "topic_discussions.json").write_text(json.dumps(discussions, indent=2) + "\n")
    (out_dir / "parse_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir


def cmd_parse(args) -> int:
    pdf_path = Path(args.pdf)
    commenters, records, discussions, page_count = parse_pdf(
        pdf_path, docket_prefix=args.docket_prefix
    )
    unresolved = sum(1 for r in records for t in r.topic_refs if not t.resolved)
    manifest = {
        "source_pdf_sha256": extract.pdf_sha256(pdf_path),
        "source_url": args.source_url or "",
        "docket_prefix": args.docket_prefix,
        "page_count": page_count,
        "counts": {
            "commenters": len(commenters),
            "comments": len(records),
            "topics": len(discussions),
            "unresolved_refs": unresolved,
        },
    }
    out_dir = Path(args.out) if args.out else config.get_rtc_output_path(args.slug)
    written = write_outputs(
        commenters=commenters, records=records, discussions=discussions,
        manifest=manifest, out_dir=out_dir,
    )
    print(
        f"parsed {pdf_path.name}: {len(commenters)} commenters, {len(records)} comments, "
        f"{len(discussions)} topics, {unresolved} unresolved refs -> {written}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtc_parser")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("parse", help="parse an RTC PDF into a structured crosswalk")
    p.add_argument("--pdf", required=True, help="path to the RTC PDF")
    p.add_argument("--slug", default="ccl5", help="output subdir under rtc/ (default: ccl5)")
    p.add_argument("--docket-prefix", default="EPA-HQ-OW-2018-0594")
    p.add_argument("--source-url", default="", help="recorded in the manifest")
    p.add_argument("--out", default="", help="explicit output dir (overrides rtc/<slug>)")
    p.set_defaults(func=cmd_parse)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
