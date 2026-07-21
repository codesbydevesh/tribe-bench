"""Resolve correct DOIs for flagged NeuroCheck claims via CrossRef search.

CPU-only, network. This does NOT invent DOIs — it queries CrossRef's
bibliographic search for each claim's citation + topic keywords and prints
ranked real candidates (title / first author / year / journal / DOI) so a human
can pick the true match. This is the antidote to LLM-hallucinated DOIs.

Usage:
    python scripts/resolve_dois.py                 # all flagged claims
    python scripts/resolve_dois.py NC004 NC016     # specific ones
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

SEARCH_URL = "https://api.crossref.org/works"
TIMEOUT = 25
DELAY = 3.0

# Claims whose DOI failed verification (NOT_FOUND / REVIEW / duplicate).
FLAGGED = [
    "NC004", "NC006", "NC007", "NC008", "NC010", "NC013", "NC015", "NC016",
    "NC018", "NC020", "NC024", "NC027", "NC028", "NC029", "NC031", "NC033",
    "NC043", "NC045",
]

# A few topic keywords per claim to sharpen the bibliographic search beyond the
# bare citation. Keeps the query anchored on the actual finding.
KEYWORDS = {
    "NC004": "primary visual cortex contrast retinotopy fMRI",
    "NC006": "intelligible speech auditory cortex fMRI",
    "NC007": "natural sounds auditory belt tones fMRI",
    "NC008": "syntactic complexity Broca area sentences fMRI",
    "NC010": "sentence comprehension semantic pseudoword prefrontal",
    "NC013": "frontal eye fields saccades fMRI",
    "NC015": "human V4 color fMRI",
    "NC016": "audiovisual speech superior temporal sulcus integration",
    "NC018": "optic flow MST motion area fMRI",
    "NC020": "emotional prosody voice superior temporal sulcus",
    "NC024": "intelligible speech temporal lobe PET",
    "NC027": "tonotopy primary auditory cortex frequency tuning fMRI",
    "NC028": "sentence semantic angular gyrus word list fMRI",
    "NC029": "intelligibility speech anterior temporal PET",
    "NC031": "phonemic speech perception superior temporal sulcus non-speech",
    "NC033": "multisensory audiovisual intraparietal ventral premotor macaque",
    "NC043": "grasping anterior intraparietal 3D objects fMRI",
    "NC045": "tool use action observation inferior parietal fMRI",
}


def _default_claims_path() -> Path:
    return Path(__file__).resolve().parent.parent / "neurocheck" / "claims_db" / "claims.yaml"


def crossref_search(query: str, rows: int = 5) -> list[dict]:
    params = {"query.bibliographic": query, "rows": str(rows), "select":
              "DOI,title,author,container-title,issued"}
    url = SEARCH_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent":
        "tribe-bench-doi-resolver/1.0 (mailto:research@example.com)"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        payload = json.load(resp)
    return (payload.get("message") or {}).get("items", [])


def fmt(item: dict) -> str:
    doi = item.get("DOI", "")
    title = (item.get("title") or [""])[0]
    authors = item.get("author") or []
    first = authors[0].get("family", "") if authors else ""
    journal = (item.get("container-title") or [""])[0]
    year = ""
    parts = (item.get("issued") or {}).get("date-parts") or []
    if parts and parts[0]:
        year = str(parts[0][0])
    return f"    {first} {year} | {journal}\n      {title}\n      DOI: {doi}"


def main() -> int:
    which = [a for a in sys.argv[1:]] or FLAGGED
    raw = yaml.safe_load(open(_default_claims_path()))
    by = {c["id"]: c for c in raw}

    for cid in which:
        c = by.get(cid)
        if not c:
            print(f"{cid}: not in claims file")
            continue
        citation = c["citation"]
        kw = KEYWORDS.get(cid, "")
        query = f"{citation} {kw}".strip()
        print(f"=== {cid} === current DOI: {c.get('doi','')}")
        print(f"  CLAIM: {c['claim']}")
        print(f"  CITE : {citation}")
        try:
            items = crossref_search(query)
        except Exception as e:
            print(f"  SEARCH ERROR: {e}\n")
            time.sleep(DELAY)
            continue
        if not items:
            print("  (no candidates)\n")
        for it in items:
            print(fmt(it))
        print()
        time.sleep(DELAY)
    return 0


if __name__ == "__main__":
    sys.exit(main())
