"""Variance decomposition of the Algonauts 2025 post-challenge OOD leaderboard.

Inputs (both transcribed and cross-verified, see the source files):
  data/algonauts/ood_leaderboard_2026-08-04.md   21 entries x 4 subjects
  data/algonauts/noise_floor_877771.md           24 subject x movie nulls from a
                                                 pure-noise submission (id 877771)

Question being settled by arithmetic rather than opinion: does this leaderboard
rank models, or does it mostly report which brain is easy to predict?

Everything here is closed-form / exact. No sampling, no fitting beyond two
rank-1 decompositions. CPU only, numpy + scipy.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# Data. Transcribed from ood_leaderboard_2026-08-04.md.
# Column order is the leaderboard's own: sub-01, sub-02, sub-03, sub-05.
# --------------------------------------------------------------------------

SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05"]

# name, average-as-printed, [sub-01, sub-02, sub-03, sub-05]
BOARD = [
    ("zpf666",          0.23465, [0.25567, 0.22890, 0.24930, 0.20475]),
    ("cchen847",        0.22769, [0.25564, 0.22069, 0.24698, 0.18745]),
    ("akgokce",         0.22720, [0.25317, 0.22054, 0.24644, 0.18864]),
    ("NCG-2025",        0.22386, [0.24767, 0.21645, 0.24263, 0.18869]),
    ("mansou",          0.21343, [0.23585, 0.20387, 0.23232, 0.18169]),
    ("NCG-2026",        0.21216, [0.23919, 0.20797, 0.22940, 0.17209]),
    ("MedARC-0726",     0.21172, [0.23332, 0.20425, 0.23208, 0.17722]),
    ("MedARC-0717",     0.21046, [0.23407, 0.20066, 0.23165, 0.17545]),
    ("sabertoaster",    0.20543, [0.22823, 0.19577, 0.22751, 0.17022]),
    ("elsun_nabatov",   0.18235, [0.20577, 0.18111, 0.19356, 0.14895]),
    ("killz",           0.18155, [0.19931, 0.17623, 0.19762, 0.15303]),
    ("alit",            0.15736, [0.17747, 0.15377, 0.17345, 0.12476]),
    ("cji724",          0.14768, [0.16130, 0.14352, 0.16276, 0.12316]),
    ("aditisaxena",     0.12874, [0.14450, 0.12540, 0.14223, 0.10285]),
    ("lio",             0.11710, [0.12817, 0.11164, 0.13148, 0.09711]),
    ("lovableaspargus", 0.11532, [0.11427, 0.12140, 0.13077, 0.09483]),
    ("neko",            0.10592, [0.11827, 0.10678, 0.11896, 0.07968]),
    ("mainak09",        0.09276, [0.10051, 0.08852, 0.10363, 0.07838]),
    ("baseline",        0.08952, [0.09855, 0.08594, 0.10209, 0.07150]),
    ("papayalore",      0.01521, [0.01729, 0.01479, 0.01626, 0.01249]),
]

# The noise submission (877771) is excluded from the model matrix -- it is the null,
# not a model.  Its per-subject-x-movie whole-brain means, transcribed exactly:
# rows sub-01/02/03/05, cols chaplin mononoke passepartout planetearth pulpfiction wot
NOISE_CELLS = np.array([
    [+0.0009, -0.0017, -0.0021, -0.0000, +0.0002, -0.0012],
    [-0.0002, +0.0012, -0.0008, +0.0001, -0.0015, -0.0007],
    [-0.0013, +0.0000, +0.0013, +0.0002, +0.0020, +0.0001],
    [+0.0007, -0.0014, -0.0008, -0.0009, -0.0007, -0.0001],
])
NOISE_PER_SUBJECT = np.array([-0.00065, -0.00033, +0.00037, -0.00053])
NOISE_OVERALL = -0.00028

NAMES = [r[0] for r in BOARD]
PRINTED_AVG = np.array([r[1] for r in BOARD])
X = np.array([r[2] for r in BOARD], dtype=float)  # (20, 4)


def rule(title: str = "", ch: str = "=") -> None:
    print()
    if title:
        print(ch * 78)
        print(title)
        print(ch * 78)
    else:
        print(ch * 78)


# --------------------------------------------------------------------------
# S0. Sanity: does the printed average equal the mean of the four columns?
# --------------------------------------------------------------------------

def s0_transcription_check() -> None:
    rule("S0  TRANSCRIPTION / SCORER CHECK")
    recomputed = X.mean(axis=1)
    resid = recomputed - PRINTED_AVG
    print(f"max |mean(4 subjects) - printed average| = {np.abs(resid).max():.6f}")
    print(f"mean signed error                        = {resid.mean():+.6f}")
    worst = int(np.argmax(np.abs(resid)))
    print(f"worst row: {NAMES[worst]:16s} recomputed {recomputed[worst]:.5f} "
          f"printed {PRINTED_AVG[worst]:.5f}")
    print()
    print("Interpretation: the headline is the unweighted mean of the four subject")
    print("columns, to within 5-decimal rounding. No weighting, no shrinkage.")
    print("Consistent with the noise submission's finding of a bare Pearson mean.")


# --------------------------------------------------------------------------
# S1. Variance decomposition, two-way with no replication.
#     SS_total = SS_model + SS_subject + SS_interaction(=residual)
# --------------------------------------------------------------------------

def decompose(M: np.ndarray, label: str) -> dict:
    n, k = M.shape
    grand = M.mean()
    row = M.mean(axis=1)
    col = M.mean(axis=0)

    ss_total = ((M - grand) ** 2).sum()
    ss_model = k * ((row - grand) ** 2).sum()
    ss_subj = n * ((col - grand) ** 2).sum()
    ss_int = ss_total - ss_model - ss_subj

    df_model, df_subj, df_int = n - 1, k - 1, (n - 1) * (k - 1)

    out = dict(
        label=label, n=n, k=k,
        ss_total=ss_total, ss_model=ss_model, ss_subj=ss_subj, ss_int=ss_int,
        f_model=ss_model / ss_total, f_subj=ss_subj / ss_total, f_int=ss_int / ss_total,
        ms_model=ss_model / df_model, ms_subj=ss_subj / df_subj, ms_int=ss_int / df_int,
    )
    out["F_model"] = out["ms_model"] / out["ms_int"]
    out["F_subj"] = out["ms_subj"] / out["ms_int"]
    out["p_model"] = float(stats.f.sf(out["F_model"], df_model, df_int))
    out["p_subj"] = float(stats.f.sf(out["F_subj"], df_subj, df_int))
    out["sd_int"] = np.sqrt(out["ms_int"])
    return out


def report_decomp(d: dict) -> None:
    print(f"\n--- {d['label']}  ({d['n']} entries x {d['k']} subjects) ---")
    print(f"  SS total        {d['ss_total']:.6e}")
    print(f"  between-MODEL   {d['ss_model']:.6e}   {100*d['f_model']:6.2f} %"
          f"   F={d['F_model']:9.2f}  p={d['p_model']:.3e}")
    print(f"  between-SUBJECT {d['ss_subj']:.6e}   {100*d['f_subj']:6.2f} %"
          f"   F={d['F_subj']:9.2f}  p={d['p_subj']:.3e}")
    print(f"  INTERACTION     {d['ss_int']:.6e}   {100*d['f_int']:6.2f} %"
          f"   residual SD = {d['sd_int']:.5f}")


def s1_variance() -> dict:
    rule("S1  VARIANCE DECOMPOSITION  -- ranking models, or ranking brains?")
    slices = {}
    slices["ALL 20 non-noise entries"] = decompose(X, "ALL 20 non-noise entries")
    slices["19, papayalore dropped"] = decompose(
        X[:-1], "19 entries (papayalore 0.015 dropped -- near-null outlier)")
    slices["top 9 (>= 0.205)"] = decompose(X[:9], "TOP 9 (all >= 0.205)")
    slices["top 5"] = decompose(X[:5], "TOP 5")
    slices["top 3"] = decompose(X[:3], "TOP 3 (0.23465 / 0.22769 / 0.22720)")
    for d in slices.values():
        report_decomp(d)

    print()
    print("Subject main effects (deviation from grand mean), all 20:")
    grand = X.mean()
    for j, s in enumerate(SUBJECTS):
        print(f"  {s}: mean {X[:, j].mean():.5f}   effect {X[:, j].mean()-grand:+.5f}")
    print(f"  best-minus-worst subject = {X.mean(axis=0).max()-X.mean(axis=0).min():.5f}")
    print(f"  best-minus-worst model   = {X.mean(axis=1).max()-X.mean(axis=1).min():.5f}")
    print(f"  top9: best-minus-worst model = "
          f"{X[:9].mean(axis=1).max()-X[:9].mean(axis=1).min():.5f}")
    return slices


# --------------------------------------------------------------------------
# S2. Friedman + Kendall's W: is the subject ordering the same for every entry?
# --------------------------------------------------------------------------

def kendall_w(M: np.ndarray) -> tuple[float, float, float, int]:
    """M is (n_raters=entries, k_items=subjects). Ranks within each row."""
    n, k = M.shape
    R = np.apply_along_axis(stats.rankdata, 1, M)  # 1 = lowest score
    Rsum = R.sum(axis=0)
    S = ((Rsum - Rsum.mean()) ** 2).sum()
    W = 12 * S / (n ** 2 * (k ** 3 - k))
    chi2 = n * (k - 1) * W
    df = k - 1
    p = float(stats.chi2.sf(chi2, df))
    return float(W), chi2, p, df


def s2_concordance() -> None:
    rule("S2  FRIEDMAN / KENDALL W  -- is the per-subject ordering identical?")
    fr = stats.friedmanchisquare(*[X[:, j] for j in range(4)])
    W, chi2, p, df = kendall_w(X)
    print(f"Friedman chi2 = {fr.statistic:.4f}  p = {fr.pvalue:.3e}  (n=20, k=4)")
    print(f"Kendall W     = {W:.4f}  chi2 = {chi2:.4f}  df = {df}  p = {p:.3e}")
    print(f"  (W = 1.0 would mean every entry ranks the four subjects identically)")

    R = np.apply_along_axis(stats.rankdata, 1, X)
    patterns: dict[tuple, list[str]] = {}
    for name, r in zip(NAMES, R):
        patterns.setdefault(tuple(r.astype(int)), []).append(name)
    print()
    print("Distinct within-entry subject orderings (best -> worst):")
    for pat, members in sorted(patterns.items(), key=lambda kv: -len(kv[1])):
        order = [SUBJECTS[i] for i in np.argsort(-np.array(pat))]
        print(f"  {' > '.join(order)}   n={len(members):2d}   {', '.join(members)}")

    print()
    print("*** CORRECTION TO ood_leaderboard_2026-08-04.md ***")
    print("That file states: 'Every one of the 20 non-noise entries has the same")
    print("ordering: sub-01 highest, sub-03 just below it, sub-02 middle, sub-05")
    print("lowest.'  That is FALSE.")
    top1 = np.array([SUBJECTS[int(np.argmax(r))] for r in X])
    n01 = int((top1 == "sub-01").sum())
    print(f"  sub-01 is the top subject in {n01}/20 entries; "
          f"sub-03 in {int((top1=='sub-03').sum())}/20.")
    viol = [NAMES[i] for i in range(len(X)) if np.argmax(X[i]) != 0]
    print(f"  entries where sub-01 is NOT highest: {', '.join(viol)}")
    print("  lovableaspargus is a 3-way violation: sub-03 > sub-02 > sub-01 > sub-05.")
    print("  The 'free quality check' in that file must be weakened to:")
    print("  sub-05 lowest (20/20) and sub-01/sub-03 as the top pair (19/20).")
    lowest = np.array([SUBJECTS[int(np.argmin(r))] for r in X])
    print(f"  sub-05 lowest in {int((lowest=='sub-05').sum())}/20 -- that part holds.")
    toppair = sum(1 for r in X if set(np.argsort(-r)[:2].tolist()) == {0, 2})
    print(f"  {{sub-01, sub-03}} = top two in {toppair}/20 -- that part holds.")


# --------------------------------------------------------------------------
# S3. Would choosing a different subject change who wins?
# --------------------------------------------------------------------------

def s3_column_rankings() -> None:
    rule("S3  DOES ANY SUBJECT COLUMN DISAGREE WITH THE OVERALL RANKING?")
    avg = X.mean(axis=1)
    print("Spearman rho of each subject column against the 4-subject average:")
    for j, s in enumerate(SUBJECTS):
        rho, p = stats.spearmanr(X[:, j], avg)
        tau, ptau = stats.kendalltau(X[:, j], avg)
        # Kendall tau distance = number of discordant pairs
        disc = sum(1 for a, b in itertools.combinations(range(len(X)), 2)
                   if np.sign(X[a, j] - X[b, j]) != np.sign(avg[a] - avg[b]))
        print(f"  {s}: rho = {rho:.5f} (p={p:.2e})  tau = {tau:.5f}  "
              f"discordant pairs = {disc}/190")

    print()
    print("Winner under each single-subject criterion:")
    for j, s in enumerate(SUBJECTS):
        order = np.argsort(-X[:, j])
        top3 = ", ".join(f"{NAMES[i]} {X[i, j]:.5f}" for i in order[:3])
        flag = "" if NAMES[order[0]] == "zpf666" else "   <-- WINNER CHANGES"
        print(f"  {s}: {top3}{flag}")

    print()
    print("Rank-2 vs rank-3 under each subject (cchen847 vs akgokce):")
    for j, s in enumerate(SUBJECTS):
        d = X[1, j] - X[2, j]
        print(f"  {s}: cchen847 - akgokce = {d:+.5f}"
              f"   {'cchen847' if d > 0 else 'AKGOKCE'} ahead")
    print("  -> the sign flips. Ranks 2 and 3 are decided by sub-01 alone.")

    print()
    print("Every place-swap induced by switching to a single-subject criterion:")
    base_order = list(np.argsort(-avg))
    for j, s in enumerate(SUBJECTS):
        col_order = list(np.argsort(-X[:, j]))
        moved = [(NAMES[i], base_order.index(i) + 1, col_order.index(i) + 1)
                 for i in range(len(X)) if base_order.index(i) != col_order.index(i)]
        print(f"  {s}: {len(moved)} entries move rank -> "
              + ("; ".join(f"{m}:{a}->{b}" for m, a, b in moved) if moved else "none"))


# --------------------------------------------------------------------------
# S4. Additive vs multiplicative rank-1 models. The residual is the finding.
# --------------------------------------------------------------------------

def additive_fit(M: np.ndarray) -> tuple[np.ndarray, float]:
    grand = M.mean()
    fit = M.mean(axis=1, keepdims=True) + M.mean(axis=0, keepdims=True) - grand
    resid = M - fit
    ss_tot = ((M - grand) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss_tot
    return resid, float(r2)


def multiplicative_fit(M: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """score_ij = a_i * s_j, fitted in log space (all entries are positive)."""
    L = np.log(M)
    grand = L.mean()
    fitL = L.mean(axis=1, keepdims=True) + L.mean(axis=0, keepdims=True) - grand
    fit = np.exp(fitL)
    resid = M - fit
    ss_tot = ((M - M.mean()) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss_tot
    # subject multipliers, normalised to mean 1
    s = np.exp(L.mean(axis=0) - grand)
    s = s / s.mean()
    return resid, float(r2), s


def s4_models() -> None:
    rule("S4  ADDITIVE vs MULTIPLICATIVE  -- where the non-additivity lives")

    res_a, r2_a = additive_fit(X)
    res_m, r2_m, mult = multiplicative_fit(X)
    print(f"ADDITIVE       score_ij = mu + model_i + subject_j : R^2 = {r2_a:.6f}")
    print(f"MULTIPLICATIVE score_ij = model_i * subject_j      : R^2 = {r2_m:.6f}")
    print(f"  additive       residual RMS = {np.sqrt((res_a**2).mean()):.6f}")
    print(f"  multiplicative residual RMS = {np.sqrt((res_m**2).mean()):.6f}")
    print()
    print("Fitted per-subject multipliers (mean-1 normalised) -- the 'fixed SNR profile':")
    for s, m in zip(SUBJECTS, mult):
        print(f"  {s}: x{m:.4f}")

    print()
    print("Per-entry realised ratio subject/entry-mean. If the trivial-averaging story")
    print("held, each column below would be a constant.")
    ratio = X / X.mean(axis=1, keepdims=True)
    print(f"  {'entry':17s} " + "  ".join(f"{s:>8s}" for s in SUBJECTS))
    for name, r in zip(NAMES, ratio):
        print(f"  {name:17s} " + "  ".join(f"{v:8.4f}" for v in r))
    print(f"  {'CV across entries':17s} " +
          "  ".join(f"{v:8.4f}" for v in ratio.std(axis=0, ddof=1) / ratio.mean(axis=0)))

    for tag, res in (("ADDITIVE", res_a), ("MULTIPLICATIVE", res_m)):
        print()
        print(f"Largest |residual| under the {tag} model "
              f"(model unusually good/bad at one brain):")
        flat = [(abs(res[i, j]), res[i, j], NAMES[i], SUBJECTS[j])
                for i in range(res.shape[0]) for j in range(res.shape[1])]
        for a, v, nm, sb in sorted(flat, reverse=True)[:8]:
            print(f"  {nm:17s} {sb}  {v:+.5f}")


# --------------------------------------------------------------------------
# S4b. The structured non-additivity: the sub-01 / sub-03 crossover.
# --------------------------------------------------------------------------

def s4b_crossover() -> None:
    rule("S4b  THE sub-01 / sub-03 CROSSOVER  -- structure, not noise", ch="-")
    avg = X.mean(axis=1)
    contrast = X[:, 0] - X[:, 2]          # sub-01 minus sub-03
    print(f"  {'entry':17s} {'avg':>8s} {'sub01-sub03':>12s}")
    for name, a, c in zip(NAMES, avg, contrast):
        print(f"  {name:17s} {a:8.5f} {c:+12.5f}")

    sl, ic, r, p, se = stats.linregress(avg, contrast)
    rho, prho = stats.spearmanr(avg, contrast)
    print()
    print(f"OLS  (sub01-sub03) = {ic:+.5f} + {sl:+.5f} * avg")
    print(f"  Pearson r = {r:.4f}  R^2 = {r**2:.4f}  p = {p:.3e}  slope SE = {se:.5f}")
    print(f"  Spearman rho = {rho:.4f}  p = {prho:.3e}")
    print(f"  zero-crossing at avg = {-ic/sl:.5f}")
    print(f"  below that score sub-03 wins; above it sub-01 wins.")

    lo = contrast[avg < -ic / sl]
    hi = contrast[avg >= -ic / sl]
    print(f"  below crossing: n={len(lo)}  mean contrast {lo.mean():+.5f}  "
          f"{int((lo<0).sum())}/{len(lo)} negative")
    print(f"  above crossing: n={len(hi)}  mean contrast {hi.mean():+.5f}  "
          f"{int((hi>0).sum())}/{len(hi)} positive")
    u = stats.mannwhitneyu(lo, hi, alternative="less")
    print(f"  Mann-Whitney (below < above): U = {u.statistic:.1f}  p = {u.pvalue:.3e}")

    print()
    print("Why this cannot be a pure averaging artefact:")
    print("  * pure MULTIPLICATIVE subject SNR  => contrast = avg * (s1 - s3), so the")
    print("    contrast scales with score and NEVER changes sign.")
    print("  * pure ADDITIVE subject offset     => contrast is CONSTANT in score.")
    print("  Observed: the contrast changes sign, monotonically in score. Neither")
    print("  rank-1 model can produce that. It is genuine model x subject interaction.")
    print()
    print("Is the sign flip bigger than the measured detection floor?")
    sd_cell = NOISE_CELLS.std(ddof=1)
    se_subj = sd_cell / np.sqrt(NOISE_CELLS.shape[1])
    se_contrast = se_subj * np.sqrt(2)
    print(f"  null SD of one subject x movie whole-brain mean = {sd_cell:.6f}")
    print(f"  => null SE of a per-subject score (6 movies)    = {se_subj:.6f}")
    print(f"  => null SE of a sub01-sub03 contrast            = {se_contrast:.6f}")
    worst = contrast[np.argmin(contrast)]
    print(f"  largest negative contrast (lovableaspargus)     = {worst:+.5f}"
          f"  = {abs(worst)/se_contrast:.1f} sigma")
    print(f"  largest positive contrast                       = "
          f"{contrast.max():+.5f}  = {contrast.max()/se_contrast:.1f} sigma")
    print("  Both are orders of magnitude above the floor. The flip is real.")


def s4c_robustness() -> None:
    rule("S4c  ROBUSTNESS OF THE CROSSOVER  -- the tests that decide it", ch="-")
    avg = X.mean(axis=1)

    print("(1) The CORRECT null test. Under a fixed multiplicative SNR profile the")
    print("    LOG ratio log(sub01/sub03) is a constant, independent of score.")
    lr = np.log(X[:, 0] / X[:, 2])
    sl, ic, r, p, se = stats.linregress(avg, lr)
    rho, prho = stats.spearmanr(avg, lr)
    print(f"    log-ratio: mean {lr.mean():+.5f}  SD {lr.std(ddof=1):.5f}  "
          f"range {lr.min():+.5f}..{lr.max():+.5f}")
    print(f"    vs score: slope {sl:+.5f} (SE {se:.5f})  r = {r:.4f}  p = {p:.3e}")
    print(f"    Spearman rho = {rho:.4f}  p = {prho:.3e}   -> NOT constant")

    print()
    print("(2) Leave-one-out on the two entries that could be driving it.")
    for drop in ("papayalore", "lovableaspargus", "elsun_nabatov"):
        keep = [i for i, n in enumerate(NAMES) if n != drop]
        a2, l2 = avg[keep], lr[keep]
        c2 = (X[:, 0] - X[:, 2])[keep]
        r_l, p_l = stats.pearsonr(a2, l2)
        rho_l, prho_l = stats.spearmanr(a2, c2)
        print(f"    drop {drop:17s}  log-ratio r = {r_l:+.4f} p = {p_l:.4f}   "
              f"raw-contrast Spearman rho = {rho_l:+.4f} p = {prho_l:.4f}")
    keep = [i for i, n in enumerate(NAMES)
            if n not in ("papayalore", "lovableaspargus")]
    r_l, p_l = stats.pearsonr(avg[keep], lr[keep])
    rho_l, prho_l = stats.spearmanr(avg[keep], (X[:, 0] - X[:, 2])[keep])
    print(f"    drop BOTH                     log-ratio r = {r_l:+.4f} p = {p_l:.4f}   "
          f"raw-contrast Spearman rho = {rho_l:+.4f} p = {prho_l:.4f}")

    print()
    print("(3) Sign test on the raw contrast, top half vs bottom half of the board.")
    top10, bot10 = (X[:10, 0] - X[:10, 2]), (X[10:, 0] - X[10:, 2])
    print(f"    ranks  1-10: {int((top10>0).sum())}/10 positive  "
          f"mean {top10.mean():+.5f}")
    print(f"    ranks 11-20: {int((bot10>0).sum())}/10 positive  "
          f"mean {bot10.mean():+.5f}")
    bt = stats.binomtest(int((bot10 < 0).sum()), 10, 0.5)
    u = stats.mannwhitneyu(bot10, top10, alternative="less")
    print(f"    binomial on bottom half being negative: p = {bt.pvalue:.4f}")
    print(f"    Mann-Whitney bottom < top: U = {u.statistic:.1f}  p = {u.pvalue:.4f}")

    print()
    print("(3b) *** THE TRAP, AND THE ONLY TEST THAT ESCAPES IT ***")
    print("    A fixed MULTIPLICATIVE profile predicts raw contrast = avg * (s1-s3),")
    print("    so a positive Spearman(avg, raw contrast) is the TRIVIAL NULL")
    print("    PREDICTION, not evidence against it. The robust rho ~ 0.77 above is")
    print("    therefore worthless as evidence. Only the SIGN of the contrast is")
    print("    scale-free, because sign(raw) = sign(log-ratio) for positive scores.")
    mult = np.exp(np.log(X).mean(axis=0) - np.log(X).mean())
    mult = mult / mult.mean()
    print(f"    Fitted multipliers: sub-01 x{mult[0]:.4f}  sub-03 x{mult[2]:.4f}"
          f"   ratio {mult[0]/mult[2]:.4f}")
    print(f"    sub-01 and sub-03 are within {100*abs(mult[0]/mult[2]-1):.2f}% of each")
    print("    other -- a near-tie. So which of the two 'wins' is nearly uninformative")
    print("    and flips easily. That is why the ordering claim in the notes broke.")
    sign_pos = (X[:, 0] - X[:, 2]) > 0
    tab = [[int(sign_pos[:10].sum()), int((~sign_pos[:10]).sum())],
           [int(sign_pos[10:].sum()), int((~sign_pos[10:]).sum())]]
    odds, pf = stats.fisher_exact(tab, alternative="two-sided")
    print(f"    2x2  [top10: {tab[0][0]}+ {tab[0][1]}-]  [bottom10: "
          f"{tab[1][0]}+ {tab[1][1]}-]")
    print(f"    Fisher exact (scale-free): p = {pf:.4f}   odds ratio = {odds:.3f}")
    rb, pb = stats.pointbiserialr(sign_pos.astype(int), np.arange(len(X)))
    print(f"    sign vs board rank, point-biserial r = {rb:+.4f}  p = {pb:.4f}")
    print("    VERDICT: the SIGN is score-dependent at p ~ 0.01, which a fixed")
    print("    profile cannot produce. But it rests on ONE 2x2, 20 entries, 6 flips,")
    print("    a clear counterexample (papayalore, rank 20, sign POSITIVE), and no")
    print("    mechanism. Suggestive. NOT a finding. Do not build a paper on it.")

    print()
    print("(4) Is the INTERACTION term real, or is it scorer/rounding noise?")
    sd_cell = NOISE_CELLS.std(ddof=1)
    se_subj = sd_cell / np.sqrt(NOISE_CELLS.shape[1])
    for M, lab in ((X, "all 20"), (X[:9], "top 9")):
        d = decompose(M, lab)
        print(f"    {lab:7s} interaction SD = {d['sd_int']:.5f}  "
              f"= {d['sd_int']/se_subj:5.1f}x the measured per-subject null SE "
              f"({se_subj:.6f})")
    print(f"    5-decimal printing contributes SD ~ {1e-5/np.sqrt(12):.2e} -- negligible.")
    print("    So the model x subject interaction is real signal, not measurement")
    print("    noise. It is SMALL (~1% of variance) but it is not zero.")


# --------------------------------------------------------------------------
# S5. Smallest meaningful gap. Are the top three distinguishable?
# --------------------------------------------------------------------------

def s5_resolution() -> None:
    rule("S5  SMALLEST MEANINGFUL GAP  -- are the top three separable at all?")
    sd_cell = NOISE_CELLS.std(ddof=1)
    n_subj, n_mov = NOISE_CELLS.shape
    se_subj = sd_cell / np.sqrt(n_mov)
    se_overall = sd_cell / np.sqrt(n_subj * n_mov)
    print("(A) MEASUREMENT FLOOR, from submission 877771 (pure noise, real scorer)")
    print(f"  24 subject x movie whole-brain means: mean {NOISE_CELLS.mean():+.6f}  "
          f"SD {sd_cell:.6f}  range {NOISE_CELLS.min():+.4f}..{NOISE_CELLS.max():+.4f}")
    print(f"  observed per-subject nulls SD        = "
          f"{NOISE_PER_SUBJECT.std(ddof=1):.6f}   (predicted {se_subj:.6f})")
    print(f"  observed overall null               = {NOISE_OVERALL:+.6f}   "
          f"(predicted SE {se_overall:.6f})")
    mdd_ind = 1.96 * np.sqrt(2) * se_overall
    print(f"  => two INDEPENDENT submissions differ significantly at 5% only if the")
    print(f"     gap exceeds 1.96*sqrt(2)*{se_overall:.6f} = {mdd_ind:.6f}")
    print("  NOTE this is the floor for UNRELATED predictors. Two real models share")
    print("  most of their signal, so their difference has SMALLER variance than this")
    print("  and the floor is anti-conservative as a separability test. It bounds")
    print("  'can the scorer resolve anything at all', not 'is model A better'.")

    print()
    print("(B) GENERALISATION-TO-NEW-SUBJECTS TEST: paired over the 4 subjects.")
    print("    This is the test a reviewer will actually demand, because a claim of")
    print("    'model A is better' is a claim about brains, not about these four.")
    print()
    print(f"  {'pair':34s} {'d_avg':>9s} {'t(3)':>8s} {'p':>9s} {'signs':>7s}")
    top = [0, 1, 2, 3]
    for a, b in itertools.combinations(top, 2):
        d = X[a] - X[b]
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        p = float(2 * stats.t.sf(abs(t), len(d) - 1))
        sgn = f"{int((d>0).sum())}+/{int((d<0).sum())}-"
        print(f"  {NAMES[a]+' vs '+NAMES[b]:34s} {d.mean():+9.5f} "
              f"{t:8.3f} {p:9.4f} {sgn:>7s}")

    print()
    print("  Per-subject differences for the two contested gaps:")
    for a, b in ((0, 1), (1, 2)):
        d = X[a] - X[b]
        print(f"    {NAMES[a]} - {NAMES[b]}: " +
              "  ".join(f"{s}={v:+.5f}" for s, v in zip(SUBJECTS, d)))

    print()
    print("(C) HOW MUCH OF THE BOARD IS SEPARABLE AT ALL (paired t, 4 subjects)?")
    n = len(X)
    sep = notsep = 0
    adjacent = []
    for a, b in itertools.combinations(range(n), 2):
        d = X[a] - X[b]
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        p = float(2 * stats.t.sf(abs(t), len(d) - 1))
        if p < 0.05:
            sep += 1
        else:
            notsep += 1
        if b == a + 1:
            adjacent.append((NAMES[a], NAMES[b], d.mean(), p))
    tot = sep + notsep
    print(f"  pairs total {tot}   separable at nominal p<0.05: {sep} "
          f"({100*sep/tot:.1f}%)   NOT separable: {notsep} ({100*notsep/tot:.1f}%)")
    bonf = 0.05 / tot
    sep_b = sum(1 for a, b in itertools.combinations(range(n), 2)
                if 2 * stats.t.sf(abs((X[a]-X[b]).mean() /
                   ((X[a]-X[b]).std(ddof=1)/2)), 3) < bonf)
    print(f"  Bonferroni (alpha={bonf:.2e}): separable {sep_b} ({100*sep_b/tot:.1f}%)")
    print()
    print("  Adjacent-rank gaps (the ones the board's ordering actually asserts):")
    nsig = 0
    for a, b, dm, p in adjacent:
        mark = "" if p < 0.05 else "   NOT SIGNIFICANT"
        if p >= 0.05:
            nsig += 1
        print(f"    {a:17s} > {b:17s}  d={dm:+.5f}  p={p:.4f}{mark}")
    print(f"  {nsig}/{len(adjacent)} adjacent rank assertions are not supported.")

    print()
    print("(D) THE HEADLINE ON RESOLUTION")
    g12 = PRINTED_AVG[0] - PRINTED_AVG[1]
    g23 = PRINTED_AVG[1] - PRINTED_AVG[2]
    spread = X[:3].mean(axis=0).max() - X[:3].mean(axis=0).min()
    print(f"  gap rank1-rank2 = {g12:.5f}")
    print(f"  gap rank2-rank3 = {g23:.5f}")
    print(f"  between-subject spread inside the same three entries = {spread:.5f}")
    print(f"  ratio (subject spread) / (rank2-rank3 gap) = {spread/g23:.1f}x")
    print(f"  ratio (subject spread) / (rank1-rank3 gap) = "
          f"{spread/(PRINTED_AVG[0]-PRINTED_AVG[2]):.1f}x")


# --------------------------------------------------------------------------
# S6. The honest counter. What can and cannot be settled with aggregate data.
# --------------------------------------------------------------------------

def s6_counter() -> None:
    rule("S6  THE HONEST COUNTER -- trivial averaging artefact vs real finding")
    res_a, r2_a = additive_fit(X)
    res_m, r2_m, _ = multiplicative_fit(X)
    d_all = decompose(X, "all")
    d_top9 = decompose(X[:9], "top9")

    print("The reviewer's objection, stated at full strength:")
    print("  'Each subject has a fixed data SNR. Averaging four fixed multipliers")
    print("   preserves order by construction. Your invariance is arithmetic, and")
    print("   TRIBE v1 Table 2 already shows the ordering DOES flip per movie.'")
    print()
    print("MOSTLY, THE OBJECTION WINS. Scored honestly:")
    print(f"  1. The fixed-MULTIPLICATIVE profile fits almost perfectly:")
    print(f"     R^2 = {r2_m:.6f} (additive {r2_a:.6f}). Residual RMS "
          f"{np.sqrt((res_m**2).mean()):.5f} on scores of order 0.2.")
    print(f"     Per-subject ratios are stable to CV = "
          f"{(X/X.mean(axis=1,keepdims=True)).std(axis=0,ddof=1).mean()/1:.4f}-ish across a")
    print(f"     15x range of scores. This IS a fixed per-subject SNR profile.")
    print(f"     The reviewer is right about the headline invariance.")
    print(f"  2. My first pass claimed the sub-01/sub-03 sign reversal refutes that.")
    print(f"     On re-test it mostly does not. Spearman(avg, RAW contrast) = 0.77 is")
    print(f"     the null's OWN prediction (raw contrast = avg x const). The scale-free")
    print(f"     version -- log-ratio vs score -- is r = "
          f"{stats.pearsonr(X.mean(axis=1), np.log(X[:,0]/X[:,2]))[0]:.3f}, "
          f"p = {stats.pearsonr(X.mean(axis=1), np.log(X[:,0]/X[:,2]))[1]:.3f}: NOT")
    print(f"     significant, and it collapses when lovableaspargus is dropped.")
    print(f"     Only the sign-vs-tier 2x2 survives (Fisher p ~ 0.011), on 6 flips.")
    print(f"     That is one contingency table. It is not a paper.")
    print()
    print("WHAT SURVIVES AND IS GENUINELY NOT A TRIVIAL ARTEFACT:")
    print(f"  A. RESOLUTION. 10/19 adjacent rank assertions are unsupported, and the")
    print(f"     top 3 are mutually inseparable. This is not an averaging artefact --")
    print(f"     it is a statement about how much the board can resolve, and it")
    print(f"     follows from the gaps and the 4-subject spread alone.")
    print(f"  B. COMPOSITION AT THE TOP. Among entries anyone competes with (top 9),")
    print(f"     model identity is {100*d_top9['f_model']:.1f}% of variance vs subject "
          f"{100*d_top9['f_subj']:.1f}%.")
    print(f"     For the top 3 it is {100*decompose(X[:3],'t3')['f_model']:.1f}% "
          f"model vs {100*decompose(X[:3],'t3')['f_subj']:.1f}% subject. The number")
    print(f"     being competed over is ~2% of what the number contains. TRUE, and")
    print(f"     true regardless of whether the profile is fixed -- in fact a fixed")
    print(f"     profile makes it WORSE, not better.")
    print(f"  C. The interaction is above the measured floor "
          f"({decompose(X,'a')['sd_int']/(NOISE_CELLS.std(ddof=1)/np.sqrt(6)):.0f}x), so a")
    print(f"     model x subject term exists. But at ~1% of variance it is a")
    print(f"     footnote, not a headline.")
    print()
    print("  Note B does NOT need the profile to be non-trivial. That is the one")
    print("  claim here that the reviewer's objection strengthens instead of killing.")
    print()
    print("WHAT THE AGGREGATE DATA CANNOT SETTLE -- state this, do not bluff:")
    print("  a. WHY the crossover happens. Subject-specific fine-tuning, training-set")
    print("     size per subject, head capacity, and per-subject noise ceiling all")
    print("     predict the same aggregate pattern. 20x4 numbers cannot separate them.")
    print("  b. Whether per-MOVIE ordering flips. The board publishes no movie")
    print("     breakdown for other entries; only our own noise submission exposed")
    print("     per-movie panels. So TRIBE v1 Table 2's flip cannot be confirmed or")
    print("     denied here. This is the objection's strongest point and it stands.")
    print("  c. A proper significance test on any gap. n=4 subjects is the only")
    print("     resampling axis available; per-parcel or per-TR predictions from other")
    print("     entrants would be needed, and they are not published.")
    print("  d. Noise-ceiling normalisation. The scorer applies none (proved by the")
    print("     noise submission), so subject differences confound SNR with")
    print("     predictability and the aggregate cannot decompose them.")
    print()
    print("MINIMUM DATA THAT WOULD CONVERT THIS INTO A DEFENSIBLE FINDING:")
    print("  * per-movie x per-subject scores for >=2 entries other than ours.")
    print("    Obtainable only by re-submitting someone else's public predictions")
    print("    (MIRAGE released theirs) and reading OUR OWN detailed-results page.")
    print("    That is one free submission, and it is the decisive experiment.")
    print("  * a per-subject noise ceiling from the public Friends s1-6 fMRI")
    print("    (split-half over repeats). Free, local, no leaderboard needed.")
    print("    Dividing by it tests whether sub-05 is hard or merely noisy.")
    print("  * our own average-subject TRIBE v2 submission gives ONE weak")
    print("    pre-registerable bet from S4c: an average-subject model should land")
    print("    on the sub-03 >= sub-01 side. Prior is only ~p=0.01-from-chance and")
    print("    the outcome is nearly a coin flip (the two multipliers differ by")
    print("    0.56%), so a hit is worth little and a miss costs little. Log it as a")
    print("    prediction; do NOT make it load-bearing for anything.")


def main() -> None:
    print("Algonauts OOD leaderboard -- variance decomposition and resolution limits")
    print("data: ood_leaderboard_2026-08-04.md + noise_floor_877771.md")
    s0_transcription_check()
    s1_variance()
    s2_concordance()
    s3_column_rankings()
    s4_models()
    s4b_crossover()
    s4c_robustness()
    s5_resolution()
    s6_counter()
    rule()
    print("done")


if __name__ == "__main__":
    main()
