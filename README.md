# tribe-bench

Research toolkit and benchmark suite for [TRIBE v2](https://github.com/facebookresearch/tribev2), Meta FAIR's trimodal brain encoding model.

TRIBE v2 predicts whole-brain fMRI responses from video, audio, and text. It won Algonauts 2025 (1st / 263 teams) and explains ~54% of brain variance. Despite this, the ecosystem around it is empty: no benchmarks, no interpretation tools, no community tooling.

tribe-bench fills that gap with four research builds:

- **BrainLens** -- Modality ablation explorer. Feed a video, get RGB brain maps showing which regions respond to visual, auditory, or language content.
- **NeuroGenre** -- Genre clustering analysis. Process clips across genres, show that brain predictions cluster by content type.
- **ScaleLaw** -- Scaling law replication. Compare TRIBE v2 predictions against real fMRI from StudyForrest.
- **NeuroCheck** -- Curated database of 50 neuroscience claims structured as testable contrasts. A benchmark for brain encoding models.

## Install

```bash
pip install -e .
```

For GPU inference (requires TRIBE v2 and PyTorch):

```bash
pip install -e ".[gpu]"
```

## Quick Start

```python
from tribe_tools.model import load_model, predict_single
from tribe_tools.atlas import get_topk_rois
from pathlib import Path

# Load model (requires GPU + tribev2 installed)
model = load_model(device="cuda")

# Run prediction
preds, segments = predict_single(model, Path("video.mp4"))

# Find most activated brain regions
top_regions = get_topk_rois(preds.mean(axis=0), k=10)
print(top_regions)
```

## Project Structure

```
tribe_tools/     Shared library (model wrapper, atlas, visualization, caching)
brainlens/       Modality ablation explorer
neurogenre/      Genre clustering analysis
scalelaw/        Scaling law replication
neurocheck/      Neuroscience benchmark (50 claims)
notebooks/       Kaggle/Colab notebooks
ops/             Project operations (internal)
```

## License

Code: MIT. Claims database: CC BY 4.0. TRIBE v2 model weights: CC BY-NC 4.0 (Meta).

## Citation

If you use tribe-bench in your research, please cite:

```bibtex
@software{tribe_bench,
  title={tribe-bench: Research Toolkit and Benchmark for TRIBE v2},
  url={https://github.com/deveshb/tribe-bench},
  year={2026}
}
```
