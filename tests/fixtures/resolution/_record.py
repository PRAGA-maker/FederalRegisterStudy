"""Record live API responses for the six-topology acceptance fixtures."""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stratification_scripts.federal_register.client import FederalRegisterClient
from stratification_scripts.reginfo.client import RegInfoClient
from stratification_scripts.resolution.channels import fulltext_identifiers
from stratification_scripts.resolution.models import CommentRef

HERE = Path(__file__).parent
MAX_XML_BYTES = 600_000

ROWS = [
    ("NOAA-NMFS-2023-0125-0016", "2024-03-22", "2024-01120",
     "Commerce Department, National Oceanic and Atmospheric Administration",
     ("0648-BM40",), "NOAA-NMFS-2023-0125", "2024-15931"),
    ("BLM-2024-0001-0003", "2024-12-26", "2024-27333",
     "Interior Department, Land Management Bureau",
     ("1004-AF01",), "BLM_HQ_FRN_MO4500181705", "2024-27333"),
    ("FBI-2024-0002-0006", "2025-02-11", "2024-28712", "Justice Department",
     ("1110-AA36",), "Docket No. FBI-158", None),
    ("EPA-HQ-OAR-2022-0491-0022", "2024-05-15", "2024-04359",
     "Environmental Protection Agency",
     ("2060-AV81",), "EPA-HQ-OAR-2022-0491", None),
    ("DOT-OST-2024-0090-0049", "2024-09-23", "2024-18496",
     "Transportation Department",
     ("2105-AF05",), "Docket No. DOT-OST-2024-0090", "2025-02747"),
    ("CMS-2024-0131-6043", "2024-12-03", "2024-22765",
     "Health and Human Services Department",
     ("0938-AV34",), "CMS-1808-IFC", None),
]


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def trim_xml(xml: str) -> str:
    """Keep the SUPLINF container for oversized documents."""
    if len(xml.encode()) <= MAX_XML_BYTES:
        return xml
    match = re.search(r"<SUPLINF\b.*?</SUPLINF>", xml, re.S | re.I)
    if not match:
        return f"<RULE>{xml[:MAX_XML_BYTES]}</RULE>"
    body = match.group(0)
    if len(body.encode()) > MAX_XML_BYTES:
        opening = re.match(r"<SUPLINF\b[^>]*>", body, re.I)
        prefix = opening.group(0) if opening else "<SUPLINF>"
        body = f"{prefix}{body[len(prefix):MAX_XML_BYTES]}</SUPLINF>"
    return f"<RULE>{body}</RULE>"


def main() -> None:
    fr = FederalRegisterClient(max_retries=4, sleep_between=0.5)
    reginfo = RegInfoClient()
    for cid, cdate, source, agency, rins, docket, packet in ROWS:
        out = HERE / cid
        ref = CommentRef(
            comment_id=cid, comment_date=cdate, source_document=source,
            agency=agency, rins=rins, docket_id=docket,
            packet_final_document=packet,
        )
        write_json(out / "input.json", {
            "comment_id": cid, "comment_date": cdate,
            "source_document": source, "agency": agency, "rins": list(rins),
            "docket_id": docket, "packet_final_document": packet,
        })
        doc_numbers = set()
        if packet:
            details = fr.fetch_document_details(packet, enrich_identifiers=True)
            write_json(out / f"fr_doc_{slug(packet)}.json", details)
            doc_numbers.add(packet)
        for rin in rins:
            docs = fr.search_by_rin(rin)
            write_json(out / f"fr_rin_{slug(rin)}.json", docs)
            doc_numbers.update(doc["document_number"] for doc in docs or [])
        docs = fr.search_by_docket(docket)
        write_json(out / f"fr_docket_{slug(docket or '')}.json", docs)
        doc_numbers.update(doc["document_number"] for doc in docs or [])
        for identifier in fulltext_identifiers(ref):
            docs = fr.search_full_text(identifier)
            write_json(out / f"fr_term_{slug(identifier)}.json", docs)
            doc_numbers.update(doc["document_number"] for doc in docs or [])
        for rin in rins:
            write_json(
                out / f"agenda_{slug(rin)}.json",
                reginfo.fetch_unified_agenda(rin),
            )
        for number in sorted(doc_numbers):
            xml = fr.fetch_document_full_text_xml(number)
            if not xml:
                continue
            path = out / f"xml_{slug(number)}.xml.gz"
            path.write_bytes(gzip.compress(trim_xml(xml).encode()))
        print(f"recorded {cid}: {len(doc_numbers)} documents")
    fr.close()
    reginfo.close()


if __name__ == "__main__":
    main()
