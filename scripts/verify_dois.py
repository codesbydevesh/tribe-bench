"""Verify every NeuroCheck claim DOI against CrossRef.

CPU-only, no GPU. Run this to make the claims database bulletproof before the
NeuroCheck resource paper. It:

  1. Checks for duplicate DOIs and duplicate claim IDs.
  2. Resolves each DOI via the CrossRef REST API.
  3. Cross-checks the resolved first-author surname and publication year against
     the claim's citation string, flagging mismatches for human review.

A mismatch is a REVIEW FLAG, not an automatic failure: a resolvable DOI can still
point at the wrong paper (the failure mode behind the historical 37% error rate),
so anything flagged should be checked by a human against the claim.

Usage:
    python scripts/verify_dois.py
    python scripts/verify_dois.py --claims path/to/claims.yaml --out report.md

Set CROSSREF_MAILTO to your email to join CrossRef's faster "polite pool":
    CROSSREF_MAILTO=you@example.com python scripts/verify_dois.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import yaml

CROSSREF_URL = "https://api.crossref.org/works/"
REQUEST_TIMEOUT = 20  # seconds
POLITE_DELAY = 1.5  # seconds between requests, to stay well under CrossRef limits


def _default_claims_path() -> Path:
    return Path(__file__).resolve().parent.parent / "neurocheck" / "claims_db" / "claims.yaml"


def load_raw_claims(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Claims file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of claims, got {type(raw).__name__}")
    return raw


def crossref_lookup(doi: str) -> dict | None:
    """Return the CrossRef 'message' for a DOI, or None if it does not resolve."""
    mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
    url = CROSSREF_URL + urllib.parse.quote(doi)
    if mailto:
        url += "?mailto=" + urllib.parse.quote(mailto)
    ua = "tribe-bench-doi-verifier/1.0"
    if mailto:
        ua += f" (mailto:{mailto})"
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            payload = json.load(resp)
        return payload.get("message")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except urllib.error.URLError:
        raise


def _first_author_surname(message: dict) -> str:
    authors = message.get("author") or []
    if authors:
        return (authors[0].get("family") or "").strip()
    return ""


def _published_year(message: dict) -> str:
    for key in ("published-print", "published-online", "published", "issued"):
        parts = (message.get(key) or {}).get("date-parts") or []
        if parts and parts[0]:
            return str(parts[0][0])
    return ""


def _citation_matches(citation: str, surname: str, year: str) -> tuple[bool, bool]:
    """Return (surname_ok, year_ok) — heuristic containment checks."""
    cite = citation.lower()
    surname_ok = bool(surname) and surname.lower() in cite
    year_ok = bool(year) and year in cite
    # Some citations give a year off by one from the DOI's print/online split.
    if year and not year_ok and year.isdigit():
        near = {str(int(year) + d) for d in (-1, 1)}
        year_ok = any(y in cite for y in near)
    return surname_ok, year_ok


def verify(claims: list[dict]) -> tuple[list[dict], list[str]]:
    """Return (per-claim results, structural warnings)."""
    warnings: list[str] = []

    ids = [c.get("id", "") for c in claims]
    for cid, n in Counter(ids).items():
        if n > 1:
            warnings.append(f"Duplicate claim ID: {cid} appears {n} times")

    dois = [c.get("doi", "") for c in claims if c.get("doi")]
    for doi, n in Counter(dois).items():
        if n > 1:
            offenders = [c.get("id") for c in claims if c.get("doi") == doi]
            warnings.append(f"Duplicate DOI: {doi} shared by {', '.join(offenders)}")

    results = []
    for i, c in enumerate(claims):
        cid = c.get("id", f"#{i}")
        doi = (c.get("doi") or "").strip()
        citation = c.get("citation", "")
        row = {
            "id": cid,
            "doi": doi,
            "citation": citation,
            "resolved": False,
            "title": "",
            "surname": "",
            "year": "",
            "surname_ok": False,
            "year_ok": False,
            "status": "",
        }
        if not doi:
            row["status"] = "MISSING_DOI"
            results.append(row)
            continue
        try:
            msg = crossref_lookup(doi)
        except Exception as e:  # network error, rate limit, etc.
            row["status"] = f"LOOKUP_ERROR: {e}"
            results.append(row)
            time.sleep(POLITE_DELAY)
            continue
        if msg is None:
            row["status"] = "NOT_FOUND"
            results.append(row)
            time.sleep(POLITE_DELAY)
            continue
        title_list = msg.get("title") or [""]
        row["resolved"] = True
        row["title"] = title_list[0] if title_list else ""
        row["surname"] = _first_author_surname(msg)
        row["year"] = _published_year(msg)
        row["surname_ok"], row["year_ok"] = _citation_matches(
            citation, row["surname"], row["year"]
        )
        if row["surname_ok"] and row["year_ok"]:
            row["status"] = "OK"
        else:
            row["status"] = "REVIEW"  # resolves, but citation doesn't match cleanly
        results.append(row)
        time.sleep(POLITE_DELAY)

    return results, warnings


def write_report(results: list[dict], warnings: list[str], out: Path) -> None:
    lines = ["# NeuroCheck DOI Verification Report", ""]
    counts = Counter(r["status"] if r["status"] in ("OK", "REVIEW", "NOT_FOUND", "MISSING_DOI") else "LOOKUP_ERROR" for r in results)
    lines.append(f"- Total claims: {len(results)}")
    lines.append(f"- OK: {counts.get('OK', 0)}")
    lines.append(f"- Needs review (resolves, citation mismatch): {counts.get('REVIEW', 0)}")
    lines.append(f"- Not found on CrossRef: {counts.get('NOT_FOUND', 0)}")
    lines.append(f"- Missing DOI: {counts.get('MISSING_DOI', 0)}")
    lines.append(f"- Lookup errors (rerun): {counts.get('LOOKUP_ERROR', 0)}")
    lines.append("")
    if warnings:
        lines.append("## Structural warnings")
        lines.extend(f"- {w}" for w in warnings)
        lines.append("")
    lines.append("## Per-claim")
    lines.append("")
    lines.append("| ID | Status | DOI | Resolved first author / year | Citation |")
    lines.append("|----|--------|-----|------------------------------|----------|")
    for r in results:
        resolved = f"{r['surname']} {r['year']}".strip() if r["resolved"] else "—"
        lines.append(
            f"| {r['id']} | {r['status']} | `{r['doi']}` | {resolved} | {r['citation']} |"
        )
    out.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--claims", type=Path, default=_default_claims_path())
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "doi_verification_report.md",
    )
    args = ap.parse_args()

    claims = load_raw_claims(args.claims)
    print(f"Loaded {len(claims)} claims from {args.claims}")
    print("Querying CrossRef (this takes ~1 minute for 50 DOIs)...\n")

    results, warnings = verify(claims)

    for w in warnings:
        print(f"  WARNING: {w}")
    if warnings:
        print()

    problems = [r for r in results if r["status"] not in ("OK",)]
    for r in problems:
        print(f"  {r['id']}: {r['status']}  {r['doi']}")
        if r["status"] == "REVIEW":
            print(f"      resolved -> {r['surname']} {r['year']} | citation: {r['citation']}")

    write_report(results, warnings, args.out)
    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\n{ok}/{len(results)} DOIs resolve and match their citation.")
    if warnings:
        print("(Shared DOIs above are informational — two claims may legitimately cite "
              "the same landmark paper.)")
    print(f"Report written to {args.out}")
    # Non-zero exit only if a DOI fails to resolve or mismatches its citation.
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
