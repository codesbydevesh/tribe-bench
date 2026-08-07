"""Print a compact digest of the completed asset-hunt lane reports.

The workflow journal holds each agent's full structured return value, which is far too large to
read into an agent's context whole. This pulls only the decision-relevant fields.

Usage:  python3 scripts/algonauts/extract_lanes.py [--full]
"""

import json
import sys
from pathlib import Path

JOURNAL = Path(
    "/home/deveshb/.claude/projects/-home-deveshb-workspace/"
    "cfb821b7-b3da-4cc8-a5be-d8a970598e0f/subagents/workflows/wf_4df476d8-70e/journal.jsonl"
)
FULL = "--full" in sys.argv
CAP = 100000 if FULL else 700


def clip(text, limit=CAP):
    text = str(text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + " …"


def main() -> None:
    if not JOURNAL.is_file():
        raise SystemExit(f"journal not found: {JOURNAL}")

    scans, verifies = [], []
    for line in JOURNAL.open():
        event = json.loads(line)
        if event.get("type") != "result":
            continue
        result = event.get("result")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except Exception:
                continue
        if not isinstance(result, dict):
            continue
        if "verdict_for_vision" in result:
            scans.append(result)
        elif "checks" in result:
            verifies.append(result)

    print(f"COMPLETED LANE REPORTS: {len(scans)} scans, {len(verifies)} verifications\n")

    for r in scans:
        print("=" * 78)
        print("LANE:", clip(r.get("lane"), 140))
        print("\nVERDICT vs VISION:", clip(r.get("verdict_for_vision")))
        cands = r.get("candidate_ideas") or []
        if cands:
            print(f"\nCANDIDATES ({len(cands)}):")
            for c in cands:
                print("  * IDEA     :", clip(c.get("idea")))
                print("    STANDOUT :", clip(c.get("why_standout")))
                print("    OCCUPANCY:", clip(c.get("occupancy")))
                print("    ASSETS   :", clip(c.get("uses_which_assets"), 300))
                print("    PAPER    :", clip(c.get("paper_shape"), 400))
                print("    RISK     :", clip(c.get("killer_risk")))
        deads = r.get("dead_ends") or []
        if deads:
            print("\nDEAD ENDS:")
            for d in deads:
                print("  x", clip(d))
        if FULL:
            for f in (r.get("findings") or []):
                print(f"  - [{f.get('confidence')}] {clip(f.get('claim'))}  <{f.get('evidence_url')}>")
        print()

    for v in verifies:
        print("=" * 78)
        print("VERIFICATION:", clip(v.get("lane"), 140))
        print("SURVIVES?:", clip(v.get("verdict_change")))
        for c in (v.get("checks") or []):
            print(f"  {c.get('verdict'):<13} {clip(c.get('claim'), 200)}")
            if c.get("verdict") in ("REFUTED", "PARTLY_TRUE"):
                print("       ->", clip(c.get("note"), 400))
        print()


if __name__ == "__main__":
    main()
