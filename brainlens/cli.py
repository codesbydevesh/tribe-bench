"""BrainLens CLI entry point.

Usage:
    python -m brainlens.cli --video path/to/video.mp4
    python -m brainlens.cli --video path/to/video.mp4 --output brain_map.png
"""

import argparse
import logging
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="brainlens",
        description="BrainLens: Modality ablation explorer for TRIBE v2",
    )
    parser.add_argument(
        "--video", type=Path, required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("brainlens_output.png"),
        help="Output PNG path (default: brainlens_output.png)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="HDF5 cache directory for resuming",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    # Import here to give fast --help response
    from tribe_tools.model import load_model
    from brainlens.inference import run_ablation
    from brainlens.attribution import compute_attribution
    from brainlens.visualization import create_brain_map

    print(f"Loading TRIBE v2 model on {args.device}...")
    model = load_model(device=args.device)

    print(f"Running 4-pass ablation on {args.video}...")
    results = run_ablation(model, args.video, cache_dir=args.cache_dir)

    print("Computing modality attribution...")
    preds_dict = {mod: r[0] for mod, r in results.items()}
    visual, auditory, language = compute_attribution(preds_dict)

    print(f"Generating brain map -> {args.output}")
    create_brain_map(
        visual, auditory, language,
        title=f"BrainLens: {args.video.name}",
        output_path=args.output,
    )

    # Print summary stats
    from tribe_tools.atlas import get_topk_rois
    full_preds = preds_dict["full"]
    top_regions = get_topk_rois(full_preds.mean(axis=0), k=5)
    print(f"\nTop 5 activated regions: {', '.join(top_regions)}")
    print(f"Output shape: {full_preds.shape}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
