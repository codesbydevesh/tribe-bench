"""Apply CrossRef-verified DOI (and where needed citation) corrections to claims.yaml.

Line-anchored by claim ID so the hand-formatted YAML and its comments are preserved
— only the `doi:` and (when given) `citation:` lines of the named claims change.

Every value here was confirmed by a direct CrossRef lookup (see scripts/resolve_dois.py
and the 2026-07-21 verification pass). Run `python scripts/verify_dois.py` afterwards.
"""

from __future__ import annotations

import re
from pathlib import Path

CLAIMS = Path(__file__).resolve().parent.parent / "neurocheck" / "claims_db" / "claims.yaml"

# id -> {doi, [citation]}. citation is only set where the ORIGINAL citation named the
# wrong journal/year (a mis-attribution, not just a bad DOI).
FIXES: dict[str, dict[str, str]] = {
    # --- clean DOI corrections (paper + citation were right, DOI suffix was wrong) ---
    "NC007": {"doi": "10.1162/089892901564108"},                 # Wessinger 2001 JOCN
    "NC008": {"doi": "10.1093/cercor/bhj106"},                    # Friederici 2006 Cereb Cortex
    "NC010": {"doi": "10.1016/S0896-6273(00)80855-7"},           # Dapretto & Bookheimer 1999 Neuron
    "NC013": {"doi": "10.1016/0028-3932(95)00134-4"},            # Paus 1996 Neuropsychologia
    "NC015": {"doi": "10.1046/j.1460-9568.2000.00905.x"},        # Bartels & Zeki 2000 EJN
    "NC018": {"doi": "10.1038/81860"},                           # Morrone 2000 Nat Neurosci
    "NC031": {"doi": "10.1093/cercor/bhi040"},                   # Liebenthal 2005 Cereb Cortex
    "NC043": {"doi": "10.1007/s00221-003-1591-5"},               # Culham 2003 Exp Brain Res
    "NC045": {"doi": "10.1523/JNEUROSCI.2040-09.2009"},          # Peeters 2009 J Neurosci
    # --- DOI + citation journal correction (citation named the wrong journal) ---
    "NC006": {"doi": "10.1093/cercor/10.5.512",
              "citation": "Binder et al., 2000, Cerebral Cortex"},          # was J Neurosci
    "NC016": {"doi": "10.1016/S0960-9822(00)00513-3",
              "citation": "Calvert et al., 2000, Current Biology"},         # was Science
    "NC020": {"doi": "10.1038/nn1392",
              "citation": "Grandjean et al., 2005, Nature Neuroscience"},   # was Neuropsychologia
    "NC028": {"doi": "10.1093/cercor/bhp055",
              "citation": "Binder et al., 2009, Cerebral Cortex"},          # was J Neurophysiol
    "NC033": {"doi": "10.1016/S0896-6273(01)00198-2",
              "citation": "Bremmer et al., 2001, Neuron"},                  # was J Neurophysiol
    # --- curation changes (original citation was a mis-attributed / duplicated paper) ---
    "NC004": {"doi": "10.1523/JNEUROSCI.16-13-04207.1996",
              "citation": "Boynton et al., 1996, Journal of Neuroscience"},  # was Tootell 1998 (mis-attributed)
    "NC027": {"doi": "10.1016/j.neuroimage.2010.01.046",
              "citation": "Humphries et al., 2010, NeuroImage"},             # A1 tonotopy; was Wessinger 2001 (dup w/ NC007); Formisano is NC023's
    "NC024": {"doi": "10.1162/089892901753165890",
              "citation": "Vouloumanos et al., 2001, Journal of Cognitive Neuroscience"},  # posterior-belt speech vs nonspeech; Scott 2000 kept for NC029
    # NC029 keeps 10.1093/brain/123.12.2400 (Scott 2000 — correct for this claim).
}


def apply_fixes(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines(keepends=True)
    changed: list[str] = []
    # Find the line index where each claim block starts.
    starts: dict[str, int] = {}
    order: list[str] = []
    for i, ln in enumerate(lines):
        m = re.match(r'\s*-\s*id:\s*"?(NC\d+)"?', ln)
        if m:
            starts[m.group(1)] = i
            order.append(m.group(1))

    def block_range(cid: str) -> tuple[int, int]:
        s = starts[cid]
        after = [starts[o] for o in order if starts[o] > s]
        return s, (min(after) if after else len(lines))

    for cid, fix in FIXES.items():
        if cid not in starts:
            raise KeyError(f"{cid} not found in claims file")
        s, e = block_range(cid)
        for field, value in fix.items():
            pat = re.compile(rf'^(\s*{field}:\s*).*$')
            for j in range(s, e):
                m = pat.match(lines[j].rstrip("\n"))
                if m:
                    old = lines[j].rstrip("\n")
                    lines[j] = f'{m.group(1)}"{value}"\n'
                    changed.append(f"{cid} {field}: {old.strip()}  ->  {value}")
                    break
            else:
                raise ValueError(f"{cid}: no '{field}:' line found in its block")

    return "".join(lines), changed


def main() -> int:
    text = CLAIMS.read_text()
    new_text, changed = apply_fixes(text)
    CLAIMS.write_text(new_text)
    for c in changed:
        print(c)
    print(f"\nApplied {len(changed)} line changes across {len(FIXES)} claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
