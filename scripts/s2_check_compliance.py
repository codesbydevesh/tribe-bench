"""Verify an S2 report complies with the FROZEN design, not merely that it parsed.

A well-formed report can still describe a different experiment. This re-derives
every commitment from `neurocheck.s2_design` and checks the report against it.

    python3 scripts/s2_check_compliance.py [data/s2_report.json]

Exit 0 = compliant.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurocheck.s2_design import ALL_PARCELS, S2, stop_eligible_parcels  # noqa: E402


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/s2_report.json")
    if not path.exists():
        print(f"no report at {path}")
        return 2
    r = json.loads(path.read_text())
    out: list[tuple[str, bool, str]] = []

    def c(name: str, ok: bool, detail: str = "") -> None:
        out.append((name, bool(ok), detail))

    c("design fingerprint matches the frozen config",
      r.get("design_fingerprint") == S2.fingerprint(), r.get("design_fingerprint", ""))
    prov = r.get("provenance", {})
    c("pinned model revision recorded",
      (prov.get("checkpoint") or {}).get("revision") == S2.model_revision)
    c("stimulus sha256 recorded", bool((prov.get("video") or {}).get("sha256")))
    c("real images, not placeholders", (prov.get("video") or {}).get("placeholders") is False)
    c("image hashes carried",
      (prov.get("images") or {}).get("n") == S2.n_events, "one per scheduled event")

    res = r.get("results", {})
    c("every parcel scored", set(res) == {p.name for p in ALL_PARCELS}, f"{len(res)}")
    c("results keyed by (parcel, lag)", all("by_lag" in v for v in res.values()))
    both = {str(S2.primary_lag_trs), str(S2.alternative_lag_trs)}
    c("BOTH lags scored for every parcel",
      bool(res) and all(both <= set(map(str, v.get("by_lag", {}))) for v in res.values()),
      f"lags {sorted(both)}")
    c("measured peak lag reported",
      all(v.get("peak_lag_trs") is not None for v in res.values()))
    floors = {v["by_lag"][str(S2.primary_lag_trs)]["floor"] for v in res.values()} \
        if res else set()
    c("floor computed from each parcel's OWN noise", len(floors) > 1)

    v = r.get("verdict", {})
    elig = {p.name for p in stop_eligible_parcels()}
    c("only surviving parcels may gate", set(v.get("stop_eligible", [])) == elig, str(sorted(elig)))
    pp = v.get("per_parcel", {})
    c("ambiguous parcels are report-only",
      all(pp.get(n, {}).get("stop_eligible") is False for n in ("PPA", "VWFA")))
    c("secondary parcels are report-only",
      all(pp.get(n, {}).get("stop_eligible") is False
          for n in ("PPA_literature", "EBA_gate0_union", "V1_control")))
    c("verdict records which lags it evaluated",
      v.get("lags_evaluated") == [S2.primary_lag_trs, S2.alternative_lag_trs])
    c("incomplete-evidence field present", "incomplete" in v)
    # the stop rule must not have fired on one lag alone
    if v.get("stop"):
        ok = all(
            all(pp[n]["by_lag"][str(l)]["status"] == "not_recovered" for l in
                (S2.primary_lag_trs, S2.alternative_lag_trs))
            for n in elig)
        c("stop fired only after failure at BOTH lags", ok)
    else:
        c("stop did not fire (nothing to check)", True)
    c("decision rules travel with the report", bool(r.get("decision_rules")))
    c("lag conflict recorded verbatim", "t=5" in r.get("lag_conflict", ""))
    c("run environment captured", bool((prov.get("run_environment") or {}).get("python")))

    for name, ok, detail in out:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    bad = [n for n, ok, _ in out if not ok]
    print(f"\n{len(out) - len(bad)}/{len(out)} compliance checks PASS")
    if r.get("stub"):
        print("NOTE: this report is a STUB (no GPU). Structure only, never a result.")
    if bad:
        print("NON-COMPLIANT:")
        for n in bad:
            print(f"  - {n}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
