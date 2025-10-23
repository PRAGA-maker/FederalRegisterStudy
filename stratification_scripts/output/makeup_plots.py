import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests


DEFAULT_MAKEUP = Path(__file__).with_name("makeup_data.csv")
DEFAULT_DOCS = Path(__file__).with_name("federal_register_2024_comments.csv")
DEFAULT_OUTDIR = Path(__file__).parent

FR_DOC_DETAIL_URL = "https://www.federalregister.gov/api/v1/documents/{document_number}.json"

CATEGORIES = [
    "Undecided/Anonymous",
    "Ordinary Citizen",
    "Organization/Corporation",
    "Academic/Industry/Expert (incl. small/local business)",
    "Political Consultant/Lobbyist",
]


def ensure_agency(df_makeup: pd.DataFrame, df_docs: pd.DataFrame) -> pd.DataFrame:
    # Join by document_number where possible
    out = df_makeup.copy()
    if "agency" not in out.columns:
        out["agency"] = None
    if "document_number" in out.columns and "document_number" in df_docs.columns:
        joined = out.merge(
            df_docs[["document_number", "agency"]],
            on="document_number",
            how="left",
            suffixes=("", "_doc"),
        )
        # Prefer existing agency then docs agency
        joined["agency"] = joined["agency"].fillna(joined["agency_doc"])
        out = joined.drop(columns=[c for c in joined.columns if c.endswith("_doc")])

    # Fill any remaining missing agencies via FR API
    if "document_number" in out.columns:
        missing_idx = out[out["agency"].isna()].index
        for idx in missing_idx:
            doc = out.at[idx, "document_number"]
            if not isinstance(doc, str) or not doc:
                continue
            try:
                r = requests.get(FR_DOC_DETAIL_URL.format(document_number=doc), timeout=20)
                if r.status_code == 200:
                    j = r.json() or {}
                    agencies = j.get("agencies") or []
                    names = ", ".join([a.get("name") for a in agencies if isinstance(a, dict) and a.get("name")])
                    if names:
                        out.at[idx, "agency"] = names
            except Exception:
                continue
    return out


def plot_pie(df: pd.DataFrame, outdir: Path) -> None:
    counts = df["category"].value_counts().reindex(CATEGORIES, fill_value=0)
    if counts.sum() == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
    ax.set_title("2024 Comment Author Makeup")
    fig.tight_layout()
    fig.savefig(outdir / "makeup_pie_2024.png", dpi=150)
    plt.close(fig)


def compute_agency_shares(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["agency", "category"])['comment_id'].count().rename("n").reset_index()
    totals = grp.groupby("agency")["n"].sum().rename("total")
    merged = grp.merge(totals, on="agency", how="left")
    merged["share"] = merged["n"] / merged["total"].where(merged["total"] > 0, 1)
    # pivot to wide
    wide = merged.pivot(index="agency", columns="category", values="share").fillna(0.0)
    # Ensure all categories exist
    for c in CATEGORIES:
        if c not in wide.columns:
            wide[c] = 0.0
    return wide.reset_index()


def plot_agency_xy(df: pd.DataFrame, outdir: Path, raw_share: bool = True) -> None:
    wide = compute_agency_shares(df)
    if wide.empty:
        return
    x = (
        wide["Ordinary Citizen"]
        if raw_share
        else (wide["Ordinary Citizen"] - wide["Organization/Corporation"]).clip(-1, 1)
    )
    y = wide["Academic/Industry/Expert (incl. small/local business)"] + wide["Political Consultant/Lobbyist"]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x, y, s=30, alpha=0.8)
    for _, r in wide.iterrows():
        ax.annotate(str(r["agency"])[:40], (r["Ordinary Citizen"], r["Academic/Industry/Expert (incl. small/local business)"] + r["Political Consultant/Lobbyist"]), fontsize=8, alpha=0.8)

    ax.axhline(0.5, color="#cccccc", lw=1)
    ax.axvline(0.5, color="#cccccc", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Share Ordinary Citizen" if raw_share else "Share OC minus Org/Corp")
    ax.set_ylabel("Share Expert+Lobbyist")
    ax.set_title("Agency makeup positioning (2024)")
    fig.tight_layout()
    fig.savefig(outdir / ("agency_xy_raw_2024.png" if raw_share else "agency_xy_diff_2024.png"), dpi=150)
    plt.close(fig)


def plot_workload_vs_citizen(df: pd.DataFrame, outdir: Path, by: str = "agency") -> None:
    if by not in ("agency", "document_number"):
        by = "agency"
    grp_total = df.groupby(by)["comment_id"].count().rename("total")
    oc = df[df["category"] == "Ordinary Citizen"].groupby(by)["comment_id"].count().rename("oc")
    merged = pd.concat([grp_total, oc], axis=1).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(merged["total"], merged["oc"], s=30, alpha=0.8)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel(f"Total comments by {by}")
    ax.set_ylabel("Ordinary Citizen comments")
    ax.set_title(f"Workload vs citizen input ({by}, 2024)")
    fig.tight_layout()
    fig.savefig(outdir / f"workload_vs_citizen_by_{by}_2024.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot makeup figures from makeup_data.csv")
    ap.add_argument("--makeup", type=str, default=str(DEFAULT_MAKEUP))
    ap.add_argument("--docs", type=str, default=str(DEFAULT_DOCS))
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUTDIR))
    args = ap.parse_args()

    makeup_path = Path(args.makeup)
    docs_path = Path(args.docs)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not makeup_path.exists():
        print(f"ERROR: {makeup_path} not found")
        return

    df_makeup = pd.read_csv(makeup_path)
    # Ensure required columns
    for c in ("document_number", "comment_id", "category"):
        if c not in df_makeup.columns:
            print(f"ERROR: makeup csv missing column: {c}")
            return

    df_docs = pd.read_csv(docs_path) if docs_path.exists() else pd.DataFrame(columns=["document_number", "agency"])
    df = ensure_agency(df_makeup, df_docs)

    plot_pie(df, outdir)
    plot_agency_xy(df, outdir, raw_share=True)
    plot_agency_xy(df, outdir, raw_share=False)
    plot_workload_vs_citizen(df, outdir, by="agency")
    plot_workload_vs_citizen(df, outdir, by="document_number")


if __name__ == "__main__":
    main()


