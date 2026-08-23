"""Phase A: the detection-floor RATIO as a sensitivity surface, not a point estimate.

The published claim "spatial_z needs a 6.7x larger effect than the simplest
alternative" is 0.1416/0.0210 at ONE corner of a two-parameter space:
D_AUD = 0.24 (an argmin over 25 single noisy draws, on the last point of its own
grid) and rho = 0 (spatially independent prediction noise, which TRIBE's
rank-<=2048 head makes structurally impossible).

This runs the floor table across rho at the properly-selected D_AUD and reports the
ratio as a range with its parameters stated. It shells out to
detection_floor_table.py so there is exactly one implementation of the floor.

Usage:  python3 scripts/floor_surface.py [n_sim]
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RHOS = (0.0, 0.3, 0.6, 0.9)


def run_point(rho, n_sim, d_aud=None):
    env = dict(os.environ, FLOOR_RHO=str(rho), FLOOR_N_SIM=str(n_sim))
    if d_aud is not None:
        env["FLOOR_D_AUD"] = str(d_aud)
    out = subprocess.run([sys.executable, "-u", "scripts/detection_floor_table.py"],
                         cwd=REPO, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"rho={rho} failed:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    floors = {}
    for line in out.stdout.splitlines():
        m = re.match(r"\s+(\w+):\s+([0-9.]+)\s*$", line)
        if m:
            floors[m.group(1)] = float(m.group(2))
        elif "never reached" in line:
            m2 = re.match(r"\s+(\w+):", line)
            if m2:
                floors[m2.group(1)] = None
    return floors


def main():
    n_sim = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    sel = json.loads((REPO / "data/sensitivity_selection.json").read_text())["selection"]
    d_aud = sel["d_best"]
    print(f"D_AUD = {d_aud} (selected on {sel['curve'][0]['n']} draws/point, "
          f"boundary={sel['on_boundary']}), n_sim={n_sim}\n")

    rows, ratios = [], []
    for rho in RHOS:
        f = run_point(rho, n_sim, d_aud)
        best_alt = min(v for k, v in f.items() if k != "spatial_z" and v)
        ratio = f["spatial_z"] / best_alt if f.get("spatial_z") else float("nan")
        ratios.append(ratio)
        rows.append((rho, f, best_alt, ratio))
        print(f"rho={rho}  spatial_z={f.get('spatial_z')}  best_alt={best_alt:.4f}  "
              f"ratio={ratio:.2f}x")

    lo, hi = min(ratios), max(ratios)
    print(f"\nRATIO RANGE across rho in {RHOS}: {lo:.1f}x .. {hi:.1f}x   (published claim: 6.7x)")

    md = ["# Detection-floor ratio — sensitivity surface", "",
          f"`spatial_z` floor divided by the best non-compositional floor, at "
          f"D_AUD={d_aud}, n_sim={n_sim}, n=15/group, alpha=0.025.", "",
          "The single number 6.7x previously published is the value at "
          "`(D_AUD=0.24, rho=0)`. D_AUD=0.24 was an argmin over 25 single noisy draws "
          "and the last point of its own grid; rho=0 asserts spatially independent "
          "prediction noise, which is structurally impossible for a model whose head is "
          "`nn.Linear(hidden, low_rank_head=2048)` over 20,484 vertices.", "",
          "| rho | spatial_z | raw_roi_mean | roi_minus_reference | glm_contrast_z | ratio |",
          "|---|---|---|---|---|---|"]
    for rho, f, _, ratio in rows:
        md.append(f"| {rho} | {f.get('spatial_z')} | {f.get('raw_roi_mean')} | "
                  f"{f.get('roi_minus_reference')} | {f.get('glm_contrast_z')} | **{ratio:.2f}x** |")
    md += ["", f"**Reported range: {lo:.1f}x – {hi:.1f}x** across rho in {list(RHOS)}.", "",
           "This is a sensitivity range over a stipulated parameter, not a confidence "
           "interval and not min/max sampling noise: each row is a separate simulation "
           f"at n_sim={n_sim} with the same seed policy, so the spread between rows is "
           "the effect of rho, not of resampling.", "",
           "Absolute floors remain in synthetic units and are computed at 15v15; S2 would "
           "run a different n. The ratio is the transferable quantity, and even it is "
           "conditional on the noise model."]
    (REPO / "data/floor_surface.md").write_text("\n".join(md) + "\n")
    (REPO / "data/floor_surface.json").write_text(json.dumps(
        {"d_aud": d_aud, "n_sim": n_sim,
         "rows": [{"rho": r, "floors": f, "ratio": q} for r, f, _, q in rows],
         "ratio_range": [lo, hi]}, indent=2))
    print("written: data/floor_surface.md + .json")


if __name__ == "__main__":
    main()
