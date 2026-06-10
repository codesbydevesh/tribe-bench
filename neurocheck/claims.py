"""NeuroCheck claims loader and data structures."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Stimulus:
    description: str
    source: str = ""  # Dataset or origin, e.g., "CelebA" or "custom recording"


@dataclass
class Contrast:
    stimulus_a: Stimulus
    stimulus_b: Stimulus
    direction: str  # "a_greater_than_b" or "b_greater_than_a"
    expected_effect_size: float


@dataclass
class ROI:
    region: str
    hemisphere: str  # "left", "right", or "both"
    atlas: str = "glasser"  # "glasser" (HCP-MMP1) is the only supported atlas


@dataclass
class Claim:
    id: str
    claim: str
    citation: str
    doi: str
    category: str
    roi: ROI
    contrast: Contrast
    difficulty: str
    notes: str = ""


def load_claims(path: Optional[Path] = None) -> list[Claim]:
    """Load claims from YAML file.

    Args:
        path: Path to claims.yaml. Defaults to the bundled database.

    Returns:
        List of Claim objects.
    """
    if path is None:
        path = Path(__file__).parent / "claims_db" / "claims.yaml"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Claims file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    claims = []
    for entry in raw:
        claim = Claim(
            id=entry["id"],
            claim=entry["claim"],
            citation=entry["citation"],
            doi=entry.get("doi", ""),
            category=entry["category"],
            roi=ROI(
                region=entry["roi"]["region"],
                hemisphere=entry["roi"]["hemisphere"],
                atlas=entry["roi"].get("atlas", "glasser"),
            ),
            contrast=Contrast(
                stimulus_a=Stimulus(
                    description=entry["contrast"]["stimulus_a"]["description"],
                    source=entry["contrast"]["stimulus_a"].get("source", ""),
                ),
                stimulus_b=Stimulus(
                    description=entry["contrast"]["stimulus_b"]["description"],
                    source=entry["contrast"]["stimulus_b"].get("source", ""),
                ),
                direction=entry["contrast"]["direction"],
                expected_effect_size=entry["contrast"]["expected_effect_size"],
            ),
            difficulty=entry["difficulty"],
            notes=entry.get("notes", ""),
        )
        claims.append(claim)

    return claims


def validate_claims(path: Optional[Path] = None) -> list[str]:
    """Validate claims YAML and return list of errors (empty = valid).

    Checks:
    - Required fields present
    - IDs are unique
    - Direction is valid
    - Effect size is positive
    - Difficulty is valid
    """
    errors = []
    try:
        claims = load_claims(path)
    except Exception as e:
        return [f"Failed to load: {e}"]

    ids = set()
    valid_directions = {"a_greater_than_b", "b_greater_than_a"}
    valid_difficulties = {"easy", "medium", "hard"}
    valid_categories = {
        "visual_selectivity", "auditory_processing", "language",
        "multimodal", "emotion", "motor_perception",
        "memory_attention", "high_level_cognition",
    }

    for claim in claims:
        prefix = f"[{claim.id}]"

        if claim.id in ids:
            errors.append(f"{prefix} Duplicate ID")
        ids.add(claim.id)

        if not claim.claim:
            errors.append(f"{prefix} Empty claim text")

        if not claim.citation:
            errors.append(f"{prefix} Missing citation")

        if not claim.roi.region:
            errors.append(f"{prefix} Missing ROI region")

        if claim.roi.hemisphere not in ("left", "right", "both"):
            errors.append(f"{prefix} Invalid hemisphere: {claim.roi.hemisphere}")

        if claim.contrast.direction not in valid_directions:
            errors.append(f"{prefix} Invalid direction: {claim.contrast.direction}")

        if claim.contrast.expected_effect_size <= 0:
            errors.append(f"{prefix} Effect size must be positive")

        if claim.difficulty not in valid_difficulties:
            errors.append(f"{prefix} Invalid difficulty: {claim.difficulty}")

        if claim.category not in valid_categories:
            errors.append(f"{prefix} Invalid category: {claim.category}")

    return errors
