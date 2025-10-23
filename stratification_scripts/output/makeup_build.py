import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Dict, Iterable, List, Optional, Tuple, Any

import pandas as pd
import requests
from openai import AsyncOpenAI, APIError, APIStatusError
from tqdm import tqdm
import random

try:
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.console import Console
    HAS_RICH = True
except Exception:
    HAS_RICH = False


# -----------------------------
# Constants and configuration
# -----------------------------

# Default locations relative to repo
DEFAULT_INPUT = Path(__file__).with_name("federal_register_2024_comments.csv")
DEFAULT_OUTPUT = Path(__file__).with_name("makeup_data.csv")

# Regulations.gov v4
REGS_DOC_URL = "https://api.regulations.gov/v4/documents/{documentId}"
REGS_COMMENTS_URL = "https://api.regulations.gov/v4/comments"

# Federal Register detail (fallback to discover regs.gov mapping)
FR_DOC_DETAIL_URL = "https://www.federalregister.gov/api/v1/documents/{document_number}.json"

# Categories for classification
CATEGORIES: List[str] = [
    "Undecided/Anonymous",
    "Ordinary Citizen",
    "Organization/Corporation",
    "Academic/Industry/Expert (incl. small/local business)",
    "Political Consultant/Lobbyist",
]

JSON_SCHEMA = {
    "name": "comment_makeup",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "category": {"type": "string", "enum": CATEGORIES},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["category"],
    },
}


def truncate_text(text: str, max_chars: int) -> str:
    s = (text or "").replace("\n", " ")
    if len(s) <= max_chars:
        return s
    head = max_chars // 2
    tail = max_chars - head - 1
    return f"{s[:head]}…{s[-tail:]}"


class LiveUI:
    def __init__(self, enable_rich: bool, enable_trace: bool, trace_max_chars: int) -> None:
        self.enable_rich = enable_rich and HAS_RICH
        self.enable_trace = enable_trace
        self.trace_max_chars = trace_max_chars
        self.console: Optional[Console] = Console() if self.enable_rich else None
        self.live: Optional[Live] = None
        self.last_sent: str = ""
        self.last_reviewed: str = ""
        self.docs_progress: str = ""
        self.comments_progress: str = ""

    def start(self) -> None:
        if self.enable_rich:
            table = self._build_table()
            self.live = Live(table, console=self.console, refresh_per_second=4)
            self.live.start()

    def stop(self) -> None:
        if self.live is not None:
            self.live.stop()
            self.live = None

    def _build_table(self):
        table = Table.grid(padding=(0, 1))
        table.add_row(Panel(self.docs_progress or "Docs: 0/0", title="Docs"), Panel(self.comments_progress or "Comments: 0", title="Comments"))
        table.add_row(Panel(truncate_text(self.last_sent, self.trace_max_chars) or "(awaiting)", title="Content Sent", border_style="cyan"))
        table.add_row(Panel(truncate_text(self.last_reviewed, self.trace_max_chars) or "(awaiting)", title="Content Reviewed", border_style="green"))
        return table

    def update_progress(self, doc_idx: int, docs_total: int, comments_done: int) -> None:
        self.docs_progress = f"Docs: {doc_idx}/{docs_total}"
        self.comments_progress = f"Comments processed (current doc): {comments_done}"
        if self.enable_rich and self.live is not None:
            self.live.update(self._build_table())

    def update_trace(self, content_sent: Optional[str], content_reviewed: Optional[str]) -> None:
        if content_sent is not None:
            self.last_sent = content_sent
        if content_reviewed is not None:
            self.last_reviewed = content_reviewed
        if self.enable_rich and self.live is not None:
            self.live.update(self._build_table())

    def print_trace_if_no_rich(self) -> None:
        if not self.enable_rich and self.enable_trace:
            if self.last_sent:
                tqdm.write(f"Content Sent: {truncate_text(self.last_sent, self.trace_max_chars)}")
            if self.last_reviewed:
                tqdm.write(f"Content Reviewed: {truncate_text(self.last_reviewed, self.trace_max_chars)}")


@dataclass
class RegsThrottle:
    rpm: int = 45
    last_call_ts: float = 0.0

    @property
    def min_interval(self) -> float:
        return 60.0 / self.rpm if self.rpm and self.rpm > 0 else 0.0

    def sleep_if_needed(self) -> None:
        if self.min_interval <= 0:
            return
        jitter = random.uniform(-0.05, 0.05) * self.min_interval
        wait = (self.min_interval + jitter) - (time.time() - self.last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self.last_call_ts = time.time()


def load_docs_map(input_csv: Path, limit_docs: Optional[int]) -> List[dict]:
    df = pd.read_csv(input_csv)
    # normalize column names we need
    needed = ["document_number", "agency", "title", "regs_document_id", "comment_count"]
    for col in needed:
        if col not in df.columns:
            df[col] = None
    if limit_docs is not None and limit_docs > 0:
        df = df.head(limit_docs)
    return df[needed].to_dict(orient="records")


def get_object_id_for_regs_document(
    regs_document_id: Optional[str],
    headers: Dict[str, str],
    throttle: RegsThrottle,
    retries: int,
) -> Optional[str]:
    if not regs_document_id:
        return None
    backoff = 1.0
    for _ in range(max(1, retries)):
        try:
            throttle.sleep_if_needed()
            r = requests.get(
                REGS_DOC_URL.format(documentId=regs_document_id),
                headers=headers,
                timeout=20,
            )
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        if r.status_code == 200:
            data = r.json() or {}
            attrs = (data.get("data") or {}).get("attributes") or {}
            obj_id = attrs.get("objectId") or attrs.get("objectID")
            if obj_id:
                return obj_id
            return None
        if r.status_code in (401, 403):
            # Likely missing/invalid API key
            try:
                tqdm.write("ERROR: 401/403 from Regulations.gov /documents; check REGS_API_KEY")
            except Exception:
                pass
            return None
        if r.status_code == 429:
            # Honor Retry-After header if present
            try:
                retry_after = int(r.headers.get("Retry-After", "0"))
                if retry_after > 0:
                    time.sleep(min(retry_after, 60))
            except Exception:
                pass
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        if r.status_code in (500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        return None
    return None


def get_regs_document_id_via_fr(document_number: Optional[str], retries: int) -> Optional[str]:
    if not document_number:
        return None
    backoff = 1.0
    for _ in range(max(1, retries)):
        try:
            r = requests.get(FR_DOC_DETAIL_URL.format(document_number=document_number), timeout=20)
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        if r.status_code == 200:
            j = r.json() or {}
            fr_info = j.get("regulations_dot_gov_info") or {}
            if isinstance(fr_info, dict):
                doc_id = fr_info.get("document_id") or fr_info.get("documentId")
                if doc_id:
                    return doc_id
            return None
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        return None
    return None


def iter_comments_for_object(
    object_id: str,
    headers: Dict[str, str],
    throttle: RegsThrottle,
    retries: int,
    limit_comments: Optional[int] = None,
    diag: Optional[Dict[str, Any]] = None,
) -> Iterable[dict]:
    """Yield comment records with windowed paging to bypass 5k cap.

    - Always sort by lastModifiedDate,documentId
    - Page 1..20, then advance window with filter[lastModifiedDate][ge]=last_seen
    """
    if not object_id:
        return

    fetched = 0
    cursor_ge: Optional[str] = None
    windows = 0
    while True:
        page = 1
        last_seen_ts: Optional[str] = None
        while page <= 20:
            params = {
                "filter[commentOnId]": object_id,
                "page[size]": 250,
                "page[number]": page,
                "sort": "lastModifiedDate,documentId",
            }
            if cursor_ge:
                params["filter[lastModifiedDate][ge]"] = cursor_ge
            backoff = 1.0
            while True:
                try:
                    throttle.sleep_if_needed()
                    r = requests.get(REGS_COMMENTS_URL, headers=headers, params=params, timeout=30)
                except requests.RequestException:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)
                    continue
                if r.status_code == 200:
                    break
                if r.status_code in (401, 403):
                    try:
                        tqdm.write("ERROR: 401/403 from Regulations.gov /comments; check REGS_API_KEY")
                    except Exception:
                        pass
                    return
                if r.status_code == 429:
                    try:
                        retry_after = int(r.headers.get("Retry-After", "0"))
                        if retry_after > 0:
                            time.sleep(min(retry_after, 60))
                    except Exception:
                        pass
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)
                    continue
                if r.status_code in (500, 502, 503, 504):
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)
                    continue
                # non-retriable
                try:
                    _ = r.json()
                except Exception:
                    pass
                return

            data = r.json() or {}
            items: List[dict] = data.get("data") or []
            if page == 1 and diag is not None:
                try:
                    diag["first_page_items"] = len(items)
                    diag["total_elements"] = (data.get("meta") or {}).get("totalElements")
                except Exception:
                    pass
            if not items:
                break

            last_item = items[-1]
            attrs_last = last_item.get("attributes") or {}
            last_seen_ts = attrs_last.get("lastModifiedDate") or attrs_last.get("modifyDate")

            for item in items:
                yield item
                fetched += 1
                if limit_comments is not None and fetched >= limit_comments:
                    return

            page += 1

        if not last_seen_ts or cursor_ge == last_seen_ts:
            return
        cursor_ge = last_seen_ts
        windows += 1
        # optional safety window cap if provided via CLI (handled at caller if needed)


def extract_comment_fields(item: dict) -> Tuple[Optional[str], str, Dict[str, Optional[str]]]:
    """Extract (comment_id, comment_text, author_meta) from a Regs.gov comment record.

    We defensively check a few common attribute names to accommodate schema variations.
    """
    if not isinstance(item, dict):
        return None, "", {}

    comment_id = None
    try:
        comment_id = item.get("id") or item.get("commentId")
    except Exception:
        pass

    attrs = item.get("attributes") or {}
    # text candidates
    text = (
        attrs.get("comment")
        or attrs.get("commentText")
        or attrs.get("comment_text")
        or ""
    )

    first = attrs.get("firstName") or attrs.get("first_name")
    last = attrs.get("lastName") or attrs.get("last_name")
    org = attrs.get("organization") or attrs.get("organizationName") or attrs.get("organization_name")
    submitter_type = attrs.get("submitterType") or attrs.get("submitter_type")
    email = attrs.get("email") or attrs.get("emailAddress") or attrs.get("email_address")

    author = None
    if first or last:
        author = f"{first or ''} {last or ''}".strip()

    email_domain = None
    if isinstance(email, str) and "@" in email:
        try:
            email_domain = email.split("@", 1)[1].lower()
        except Exception:
            email_domain = None

    meta = {
        "author": author,
        "organization": org,
        "submitter_type": submitter_type,
        "email_domain": email_domain,
    }
    return comment_id, str(text or ""), meta


def fetch_comment_detail(
    comment_id: str,
    headers: Dict[str, str],
    throttle: RegsThrottle,
    retries: int,
) -> Optional[dict]:
    if not comment_id:
        return None
    url = f"https://api.regulations.gov/v4/comments/{comment_id}"
    backoff = 1.0
    for _ in range(max(1, retries)):
        try:
            throttle.sleep_if_needed()
            r = requests.get(url, headers=headers, timeout=20)
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                return None
        if r.status_code in (401, 403):
            try:
                tqdm.write("ERROR: 401/403 from Regulations.gov /comments/{id}; check REGS_API_KEY")
            except Exception:
                pass
            return None
        if r.status_code == 429:
            try:
                retry_after = int(r.headers.get("Retry-After", "0"))
                if retry_after > 0:
                    time.sleep(min(retry_after, 60))
            except Exception:
                pass
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        if r.status_code in (500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        return None
    return None


def build_user_message(comment_text: str, author_meta: Dict[str, Optional[str]]) -> str:
    snippet = (comment_text or "")[:4000]
    return (
        "Classify the author type of this public comment into exactly one category.\n"
        f"Categories: {CATEGORIES}\n"
        "If unclear, choose 'Undecided/Anonymous'.\n"
        f"Author meta (JSON): {json.dumps(author_meta, ensure_ascii=False)}\n"
        f"Comment excerpt:\n{snippet}"
    )


async def classify_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    user_msg: str,
) -> Dict[str, Optional[str]]:
    async with semaphore:
        # First attempt with json_schema enforcement
        try:
            resp = await client.responses.create(
                model="gpt-5-nano",
                temperature=0,
                response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
                input=[{"role": "user", "content": user_msg}],
                max_output_tokens=120,
            )
            data = json.loads(resp.output_text)
            category = data.get("category")
            confidence = data.get("confidence")
            return {"category": category, "confidence": confidence}
        except (APIError, APIStatusError) as e:
            return {"category": "Undecided/Anonymous", "confidence": None, "error": str(e)[:300]}
        except Exception:
            # Fallback to json_object
            try:
                resp = await client.responses.create(
                    model="gpt-5-nano",
                    temperature=0,
                    response_format={"type": "json_object"},
                    input=[{"role": "user", "content": user_msg}],
                    max_output_tokens=120,
                )
                data = json.loads(resp.output_text)
                category = data.get("category")
                if category not in CATEGORIES:
                    category = "Undecided/Anonymous"
                confidence = data.get("confidence")
                return {"category": category, "confidence": confidence}
            except Exception as e2:
                return {"category": "Undecided/Anonymous", "confidence": None, "error": str(e2)[:300]}


def load_existing_makeup(output_csv: Path) -> Dict[str, dict]:
    if not output_csv.exists():
        return {}
    try:
        df = pd.read_csv(output_csv)
    except Exception:
        return {}
    if "comment_id" not in df.columns:
        return {}
    existing: Dict[str, dict] = {}
    for _, r in df.iterrows():
        cid = r.get("comment_id")
        if isinstance(cid, str) and cid:
            existing[cid] = r.to_dict()
    return existing


async def process_document(
    doc_row: dict,
    openai_client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    headers: Dict[str, str],
    throttle: RegsThrottle,
    retries: int,
    existing_by_comment_id: Dict[str, dict],
    limit_comments: Optional[int],
    ui: Optional[LiveUI],
    doc_idx: int,
    docs_total: int,
) -> List[dict]:
    document_number = doc_row.get("document_number")
    agency = doc_row.get("agency")
    regs_document_id = doc_row.get("regs_document_id")
    comment_count = doc_row.get("comment_count")
    if ui is not None:
        ui.update_progress(doc_idx, docs_total, 0)
    # Fast skip empty docs unless explicitly included
    try:
        if (comment_count is not None) and (not str(comment_count).strip() == ""):
            if int(comment_count) == 0 and not args_include_empty_docs:
                if ui is not None and ui.enable_trace:
                    ui.update_trace(content_sent=f"doc={document_number} (comment_count=0, skipped)", content_reviewed=None)
                return []
    except Exception:
        pass
    if not regs_document_id:
        # Try FR fallback to discover regs.gov mapping
        regs_document_id = get_regs_document_id_via_fr(document_number, retries)

    object_id = get_object_id_for_regs_document(regs_document_id, headers, throttle, retries)
    if not object_id:
        # If we still don't have objectId, we can't fetch comments for this doc
        if ui is not None and ui.enable_trace:
            ui.update_progress(doc_idx, docs_total, 0)
            ui.update_trace(content_sent=f"doc={document_number} regs_document_id={regs_document_id} (no regs.gov objectId)", content_reviewed="")
        return []

    results: List[dict] = []
    tasks: List[asyncio.Task] = []
    task_info: Dict[asyncio.Task, Tuple[str, Dict[str, Optional[str]], str]] = {}
    comments_done = 0

    pbar_comments = tqdm(total=None, desc=f"Comments {document_number}", unit="cmt", position=1, leave=False)

    # diagnostics container for first page
    first_page_diag: Dict[str, Any] = {}

    # Iterate comments synchronously (HTTP via requests) but classify concurrently
    for item in iter_comments_for_object(object_id, headers, throttle, retries, limit_comments, diag=first_page_diag):
        comment_id, text, meta = extract_comment_fields(item)
        if not comment_id:
            continue
        if (comment_id in existing_by_comment_id) and (not args_reclassify_existing):
            # already classified
            continue
        # If list payload lacks comment text, optionally fetch detail
        if text.strip() == "" and args_fetch_detail_if_empty:
            detail = fetch_comment_detail(comment_id, headers, throttle, retries)
            if isinstance(detail, dict):
                attrs_d = (detail.get("data") or {}).get("attributes") or {}
                text = attrs_d.get("comment") or text
        user_msg = build_user_message(text, meta)
        t = asyncio.create_task(classify_async(openai_client, sem, user_msg))
        tasks.append(t)
        task_info[t] = (comment_id, meta, user_msg)

    if not tasks:
        if ui is not None:
            if ui.enable_trace:
                fp = first_page_diag.get("first_page_items")
                te = first_page_diag.get("total_elements")
                ui.update_trace(
                    content_sent=f"doc={document_number} objectId={object_id} page1_items={fp} totalElements={te}",
                    content_reviewed="",
                )
                if not ui.enable_rich:
                    ui.print_trace_if_no_rich()
        pbar_comments.close()
        return []

    for completed in asyncio.as_completed(tasks):
        cls = await completed
        comment_id, meta, user_msg = task_info.get(completed, (None, {}, ""))
        if comment_id:
            out = {
                "document_number": document_number,
                "comment_id": comment_id,
                "category": cls.get("category"),
                "confidence": cls.get("confidence"),
                "agency": agency,
            }
            if "error" in cls:
                out["error"] = cls["error"]
            results.append(out)

        comments_done += 1
        pbar_comments.update(1)
        if ui is not None:
            reviewed_text = f"category={cls.get('category')} confidence={cls.get('confidence')}"
            ui.update_progress(doc_idx, docs_total, comments_done)
            ui.update_trace(content_sent=user_msg, content_reviewed=reviewed_text)
            if not ui.enable_rich and ui.enable_trace:
                ui.print_trace_if_no_rich()

    pbar_comments.close()
    return results


async def main_async(args: argparse.Namespace) -> None:
    input_csv = Path(args.input)
    output_csv = Path(args.output)

    # Load docs
    docs = load_docs_map(input_csv, args.limit_docs)
    if not docs:
        print(f"No documents found in {input_csv}")
        return

    # Load existing classifications to resume
    existing_by_comment_id = load_existing_makeup(output_csv)

    # Prepare clients
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set in environment")
        return
    client = AsyncOpenAI(api_key=openai_key, timeout=args.timeout, max_retries=args.max_retries)

    regs_key = os.environ.get("REGS_API_KEY")
    headers = {"X-Api-Key": regs_key} if regs_key else {}
    throttle = RegsThrottle(rpm=args.regs_rpm)

    sem = asyncio.Semaphore(max(1, args.concurrency))

    ui = LiveUI(enable_rich=args.use_rich, enable_trace=args.trace, trace_max_chars=args.trace_max_chars)
    ui.start()

    all_results: List[dict] = []
    pbar_docs = tqdm(total=len(docs), desc="Docs", unit="doc", position=0)
    for idx, doc in enumerate(docs, start=1):
        doc_results = await process_document(
            doc,
            client,
            sem,
            headers,
            throttle,
            args.max_retries,
            existing_by_comment_id,
            args.limit_comments,
            ui,
            idx,
            len(docs),
        )
        if doc_results:
            all_results.extend(doc_results)

        # Write incremental chunks to avoid large memory usage
        if len(all_results) >= 1000:
            write_results(output_csv, all_results)
            all_results.clear()

        pbar_docs.update(1)

    # Final write remaining
    if all_results:
        write_results(output_csv, all_results)

    pbar_docs.close()
    ui.stop()


def write_results(output_csv: Path, new_rows: List[dict]) -> None:
    out_df_new = pd.DataFrame(new_rows)
    if output_csv.exists():
        try:
            out_df_old = pd.read_csv(output_csv)
        except Exception:
            out_df_old = pd.DataFrame(columns=["document_number", "comment_id", "category", "confidence", "agency", "error"])
        out_df = (
            pd.concat([out_df_old, out_df_new], ignore_index=True)
            .drop_duplicates(subset=["comment_id"], keep="first")
        )
    else:
        out_df = out_df_new
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote {len(new_rows)} new rows to {output_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build makeup_data.csv by classifying comment authors with OpenAI (async)")
    p.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--regs-rpm", type=int, default=45, help="Throttle for Regulations.gov requests per minute (<=50)")
    p.add_argument("--limit-docs", type=int, default=None)
    p.add_argument("--limit-comments", type=int, default=None)
    p.add_argument("--trace", action="store_true", help="Show latest Content Sent/Reviewed for explainability")
    p.add_argument("--trace-max-chars", type=int, default=600, help="Max characters to display in trace panes")
    p.add_argument("--use-rich", action="store_true", help="Use rich live dashboard if available")
    p.add_argument("--fetch-detail-if-empty", dest="fetch_detail_if_empty", action="store_true", help="Fetch /v4/comments/{id} if list has empty text")
    p.add_argument("--no-fetch-detail-if-empty", dest="fetch_detail_if_empty", action="store_false")
    p.set_defaults(fetch_detail_if_empty=True)
    p.add_argument("--max-windows", type=int, default=None, help="Max time windows to page across (>5k handling)")
    p.add_argument("--include-empty-docs", action="store_true", help="Include docs with comment_count == 0")
    p.add_argument("--reclassify-existing", action="store_true", help="Reclassify even if comment_id exists in output cache")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        # expose selected args to inner helpers that aren't refactored to carry args
        global args_fetch_detail_if_empty, args_include_empty_docs, args_reclassify_existing
        args_fetch_detail_if_empty = getattr(args, "fetch_detail_if_empty", True)
        args_include_empty_docs = getattr(args, "include_empty_docs", False)
        args_reclassify_existing = getattr(args, "reclassify_existing", False)
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted")


if __name__ == "__main__":
    main()


