# Corticall

An agent-callable instrument for in-silico experiments on [TRIBE v2](https://github.com/facebookresearch/tribev2),
Meta FAIR's trimodal brain-encoding model — and the benchmark that maps where it generalizes
and where it breaks.

> Repo dir is still `tribe-bench` and import paths are still `tribe_tools`/`brainlens`/`neurocheck`
> — the rename is docs-level for now (see `.notes/plans/corticall/IDENTITY.md`).

TRIBE v2 predicts whole-brain fMRI responses from video, audio, and text (Algonauts 2025 winner,
~54% explainable variance). Corticall does two things with it, off one shared engine:

- **NeuroCheck** — a CrossRef-DOI-verified benchmark of landmark neuroscience claims as testable
  brain-activation contrasts. A runnable scoring harness reports which contrasts a movie-trained
  encoder reproduces in-silico and which it misses (the honest generalization story).
- **A read-only brain-response MCP** — the same forward-pass + modality-ablation path, exposed as
  a tool any AI agent can call: give it a contrast or stimulus, get back structured per-region
  response + which modality (video / audio / language) drives it.

Built solo, no lab, no funding — the whole pipeline runs on a single free Kaggle T4 (peak ~11 GB)
via sequential encoder load/cache/free.

Positioning note: this is a scientific readout instrument, **not** an "engagement/attention
predictor" (that framing is falsified in the literature and license-incompatible).

## Install

```bash
pip install -e .
```

For GPU inference (requires TRIBE v2 and PyTorch):

```bash
pip install -e ".[gpu]"
```

## Test

```bash
pytest        # CPU-only suite; no GPU/torch required
```

## Quick Start

```python
from tribe_tools.model import load_model, predict_single
from tribe_tools.atlas import get_topk_rois
from pathlib import Path

model = load_model(device="cuda")            # requires GPU + tribev2 installed
preds, segments = predict_single(model, Path("video.mp4"))
print(get_topk_rois(preds.mean(axis=0), k=10))
```

## Project Structure

```
tribe_tools/   Shared library (model wrapper, atlas, visualization, caching)
brainlens/     Modality-attribution (being rebuilt on exact Shapley)
neurocheck/    NeuroCheck benchmark — 50 DOI-verified claims + scoring harness
notebooks/     Kaggle/Colab notebooks
tests/         CPU-only test suite
.notes/        Living state — start at .notes/BRIEF_ME.md
ops/           Durable reference (facts, contracts, decisions)
```

## Direction

The current plan is in [`.notes/plans/corticall/ROADMAP.md`](.notes/plans/corticall/ROADMAP.md);
the strategic review behind it is in `ops/principal-review-2026-07-23.pdf`.

## License

Code: MIT. Claims database: CC BY 4.0. TRIBE v2 model weights: CC BY-NC 4.0 (Meta).

## Citation

```bibtex
@software{corticall,
  title={Corticall: an agent-callable instrument and benchmark for the TRIBE v2 brain encoder},
  url={https://github.com/codesbydevesh/tribe-bench},
  year={2026}
}
```
