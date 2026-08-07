# Algonauts OOD leaderboard, captured 2026-08-04

Phase 2: Model selection (OOD movies) — `codabench.org/competitions/9483`.
Copied from the authenticated page; the board is not readable without a login.

| # | participant | date | id | average | sub-01 | sub-02 | sub-03 | sub-05 |
|---|---|---|---|---|---|---|---|---|
| 1 | zpf666 | 2026-07-22 | 858738 | **0.23465** | 0.25567 | 0.22890 | 0.24930 | 0.20475 |
| 2 | cchen847 | 2026-07-22 | 857403 | 0.22769 | 0.25564 | 0.22069 | 0.24698 | 0.18745 |
| 3 | akgokce | 2026-05-06 | 712349 | 0.22720 | 0.25317 | 0.22054 | 0.24644 | 0.18864 |
| 4 | NCG | 2025-08-14 | 353401 | 0.22386 | 0.24767 | 0.21645 | 0.24263 | 0.18869 |
| 5 | mansou | 2026-06-14 | 796247 | 0.21343 | 0.23585 | 0.20387 | 0.23232 | 0.18169 |
| 6 | NCG | 2026-02-02 | 519386 | 0.21216 | 0.23919 | 0.20797 | 0.22940 | 0.17209 |
| 7 | MedARC | 2025-07-26 | 338408 | 0.21172 | 0.23332 | 0.20425 | 0.23208 | 0.17722 |
| 8 | MedARC | 2025-07-17 | 332996 | 0.21046 | 0.23407 | 0.20066 | 0.23165 | 0.17545 |
| 9 | sabertoaster | 2026-05-26 | 755764 | 0.20543 | 0.22823 | 0.19577 | 0.22751 | 0.17022 |
| 10 | elsun_nabatov | 2026-06-19 | 803100 | 0.18235 | 0.20577 | 0.18111 | 0.19356 | 0.14895 |
| 11 | killz | 2025-11-10 | 418509 | 0.18155 | 0.19931 | 0.17623 | 0.19762 | 0.15303 |
| 12 | alit | 2025-07-25 | 338261 | 0.15736 | 0.17747 | 0.15377 | 0.17345 | 0.12476 |
| 13 | cji724 | 2026-07-11 | 839608 | 0.14768 | 0.16130 | 0.14352 | 0.16276 | 0.12316 |
| 14 | aditisaxena | 2025-07-13 | 331731 | 0.12874 | 0.14450 | 0.12540 | 0.14223 | 0.10285 |
| 15 | lio | 2025-08-06 | 349716 | 0.11710 | 0.12817 | 0.11164 | 0.13148 | 0.09711 |
| 16 | lovableaspargus | 2026-07-08 | 834898 | 0.11532 | 0.11427 | 0.12140 | 0.13077 | 0.09483 |
| 17 | neko | 2026-01-21 | 497762 | 0.10592 | 0.11827 | 0.10678 | 0.11896 | 0.07968 |
| 18 | mainak09 | 2025-07-23 | 335983 | 0.09276 | 0.10051 | 0.08852 | 0.10363 | 0.07838 |
| 19 | **baseline** | 2025-07-07 | 328777 | **0.08952** | 0.09855 | 0.08594 | 0.10209 | 0.07150 |
| 20 | papayalore | 2026-04-04 | 663803 | 0.01521 | 0.01729 | 0.01479 | 0.01626 | 0.01249 |
| 21 | devborugadda (ours, NOISE) | 2026-08-04 | 877771 | −0.00028 | −0.00065 | −0.00033 | 0.00037 | −0.00053 |

## Corrections to earlier notes

**The "2nd place 0.2125 / 3rd 0.2094" figures cited earlier are wrong for this board** and do
not appear on it. They came from secondary reporting about the original challenge. Use the
table above.

**The meaningful reference is the challenge's own baseline at 0.08952** (rank 19). That is the
threshold for "our pipeline produced something real." An earlier note put the
something-is-broken line at 0.05; the honest line is **0.09**.

**TRIBE v1, the challenge winner, is not on this board.** Its weights were never released. The
top three entries are all 2026 submissions from people who kept training after the challenge
closed, and `akgokce` is very likely Abdulkadir Gokce (MIRAGE, arXiv 2605.29850, Brain-Score
lab). The board is actively contested — the top entry is two weeks old.

## Independent confirmation of the screenshot transcription

Our noise row's per-subject values match, to the digit, the per-subject means transcribed from
the 33 detailed-results screenshots (see `noise_floor_877771.md`):

| | leaderboard | screenshots |
|---|---|---|
| average | −0.00028 | −0.0003 |
| sub-01 | −0.00065 | −0.0006 |
| sub-02 | −0.00033 | −0.0003 |
| sub-03 | **+0.00037** | **+0.0004** |
| sub-05 | −0.00053 | −0.0005 |

Including the sign, with sub-03 the only positive. The leaderboard carries 5 decimals and the
maps rounded to 4. This is a stronger check than the three agent verifications, because it is
a completely independent rendering of the same underlying numbers.

## A free quality check for the real run

> ⚠️ **CORRECTION, 2026-08-07 — the claim that used to be here was FALSE.** It read: *"Subject
> difficulty is a property of the data, not of the model. Every one of the 20 non-noise entries
> has the same ordering: sub-01 > sub-03 > sub-02 > sub-05."* That is wrong, and it was wrong on
> the numbers printed beside it — the baseline row reads 0.0986 / 0.0859 / **0.1021** / 0.0715,
> in which sub-03 beats sub-01. The claim was falsified by the table it was written next to, and
> it went on to become the basis of a candidate flagship for two days.

Verified by counting every row (`scripts/algonauts/leaderboard_variance.py`):

**6 of 20 entries flip sub-01/sub-03** — rank 13 cji724 (+0.00146), 15 lio (+0.00331),
16 lovableaspargus (+0.01650), 17 neko (+0.00069), 18 mainak09 (+0.00312), and 19 **the challenge
baseline** (+0.00354). Rank 16 also puts sub-02 above sub-01.

**Every flip sits in the bottom half of the board.** Ranks 1-12 all put sub-01 first; five of the
bottom seven put sub-03 first.

### What it actually means — more interesting than the false claim

The per-subject profile is **score-dependent, not invariant**. Strong models favour sub-01, weak
models favour sub-03. If subject difficulty were purely a property of the data — SNR, motion,
scan quality — the ordering could not depend on how good the model is.

So sub-01 appears to carry structure that only a stronger model can exploit, while sub-03 is the
easier subject for a weak one. That **kills the "this leaderboard is just measuring subject SNR"
reading**, which was the entire basis of the S1 candidate. The honest question is not "which
subject is noisy" but "what does sub-01 have that a model must be good to capture" — and that is
not answerable from aggregate scores alone.

### The check that does survive

Weaker, but free and still worth running: **sub-05 is last in all 20 entries**, and sub-02 is
third in 19 of 20. So a real submission should come back with sub-05 lowest; if it doesn't,
something is wrong regardless of the headline number. Do **not** expect a fixed sub-01/sub-03
order — for an average-subject model that is close to a coin flip.

## What our number will mean

The released TRIBE v2 checkpoint predicts an **average subject**, whereas the entries above it
are subject-specific. So we are structurally capped below them and should say so first, not in
a footnote.

- **≥ 0.09** — beats the challenge baseline, so the pipeline demonstrably works.
- **0.15–0.20** — a strong result for off-the-shelf released weights, in the range of ranks
  10–12 which are trained submissions.
- **< 0.09** — below baseline. Debug rather than publish.

**Nobody on this board appears to be running the publicly released TRIBE v2 checkpoint as-is.**
That makes a reproducible, fully-published reference measurement an un-taken contribution — a
reference point rather than a competitive entry, which is the more defensible thing to own from
this position.
