# TRIBE v2's learned absolute position embedding is untrained noise

**Found 2026-08-10. Zero GPU, no torch, ~10 minutes.** Reproduce with
`python3 scripts/algonauts/probe_position_embed.py --ckpt <best.ckpt>`.

## The claim

`model.time_pos_embed`, the (1, 1024, 1152) learned absolute position embedding added to the
encoder input in the released `facebook/tribev2` checkpoint, is **statistically indistinguishable
from its random initialization** on three independent structural measures. The trained model does
not use it. Absolute-position information plays no role at inference; position enters only through
rotary (relative) embeddings.

## Why magnitude alone was not enough

The parameter is created as `torch.randn(1, max_seq_len, hidden)` (`tribev2/model.py:151-152`) —
std exactly 1.0 at init. So "std ≈ 1.0 after training" is ambiguous. Structure is not:

| measure | learned position code | random init | **observed** |
|---|---|---|---|
| per-slot L2 norm | departs from √dim | ≈ √1152 = 33.94 | **33.93, flat across all 1024 slots** |
| adjacent-slot \|cos\|, used region | ≫ 1/√dim | ≈ 1/√dim = 0.0295 | **0.0286** |
| cosine vs slot-distance | decays with distance | ~0 at every distance | **−0.009 … +0.005, no decay** |

The norm is flat at √dim even across slots 0–100, which a 100-TR window (`duration_trs: 100`)
actually uses — so it is not that only the unused tail is random; the **used** region is random
too. On all three measures the trained tensor sits on the random-init reference.

## Corroborating architecture facts (from the released config and source)

- `config.yaml: rotary_pos_emb: true`, `rel_pos_bias: false`, `alibi_pos_bias: false`. RoPE is the
  live position mechanism, and it is **relative**, which is exactly why the additive absolute
  embedding was free to decay to noise — it was redundant.
- The checkpoint contains only two positional tensors: `time_pos_embed` (dead) and
  `rotary_pos_emb.inv_freq` (RoPE's fixed, non-learned buffer). No other learned absolute-position
  parameter exists.

## What this kills, and what it does not

**Kills:** the proposed "temporal-aperture" flagship that would sweep a knob built on this
embedding. The knob operates through a dead parameter; it would move nothing. This is a genuine
save — the failure would otherwise have surfaced only after a GPU run and a leaderboard submission.

**Does not kill** the research question underneath it. The load-bearing fact for "the prediction
at within-window position p uses future stimulus" is that the encoder is **maskless /
bidirectional**, verified independently of this embedding:

- `config.yaml: causal: false`
- `neuraltrain/models/transformer.py:69-72`: `if self.causal: return Decoder(...) else: return
  Encoder(...)` → returns the bidirectional `Encoder`.
- `tribev2/model.py:233`: `x = self.encoder(x)` — no mask, no `is_causal`.

Maskless attention means the representation at output position p is a function of all window
positions, including everything after p. The lookahead experiment is therefore done by
manipulating **which stimulus is in the window**, not by twiddling this embedding — see
`lookahead_scope.md`.

## Provenance note

This is the fourth claimed asset to collapse against the primary artifact (after: the
"reverse-engineered" mask that was a config key; the "novel" ablation that was v1's headline; the
"identical" per-subject ordering that 6 of 20 entries break). It is also the first such collapse
that *produced* a usable result rather than only removing one, and it did so before any GPU spend.
Standing rule, now paid for five times over: verify the premise against the artifact first.
