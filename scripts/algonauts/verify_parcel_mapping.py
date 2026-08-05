"""Prove our surface parcellation indexes the same 1,000 parcels the scorer does. No GPU.

Assumption A1 in `prepare_submission.py`: the challenge extracted its 1,000 parcels from
subject-specific MNI *volumetric* Schaefer atlases, while we map TRIBE's fsaverage5
*surface* output through the Schaefer fsaverage5 annotation. If the index conventions
disagree -- most obviously an LH/RH swap -- every parcel is compared against the wrong
region and the leaderboard score collapses to ~0 with no indication of why.

This settles it geometrically instead of empirically: download the challenge's own atlas,
take each parcel's centroid in MNI millimetres, take each of our surface parcels' centroid
on the fsaverage5 pial surface, and check that parcel i lands in the same place in both.

Result when this was first run (2026-08-04):

    centroid correlation      x +0.9988   y +0.9985   z +0.9972
    median distance i <-> i     3.9 mm
    median distance i <-> random 81.8 mm
    their parcels 1-500   100% x < 0 (left)
    their parcels 501-1000 100% x > 0 (right)

A 3.9 mm median offset across 1,000 parcels between a volumetric centroid and a surface
centroid is registration-level disagreement, not an indexing error. A1 confirmed.

Still untested after this passes: A2/A3, the temporal origin and HRF offset. Geometry says
nothing about time. Validate that separately against the PUBLIC Friends responses -- never
by sweeping offsets against the leaderboard, which would be tuning on the withheld set.
"""

import re
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
DATA = REPO / "data" / "algonauts"
ATLAS_DIR = REPO / "data" / "atlas"
ANNOT = "{hemi}.Schaefer2018_1000Parcels_7Networks_order.annot"

RAW = "https://raw.githubusercontent.com/courtois-neuromod/algonauts_2025.competitors/main"
RIA = (
    "https://sftp.conp.ca/users/cneuromod/ria-conp"
    "/5d2/0a1ae-3571-4de8-94c8-8ddb416cd3b0/annex/objects"
)
ATLAS_REL = (
    "fmri/sub-01/atlas/sub-01_space-MNI152NLin2009cAsym"
    "_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"
)
N_PARCELS = 1000


def fetch_annexed(rel_path: str, dest: Path) -> Path:
    """Download git-annex content over plain HTTPS -- no git-annex, no datalad, no sudo.

    The dataset registers a public RIA store in its `git-annex` branch, and every annexed
    file in the GitHub tree is a pointer whose text contains the object's hash directory
    and key. So the content is at <RIA>/<hashdir>/<key>/<key>.
    """
    if dest.is_file() and dest.stat().st_size > 1000:
        return dest
    pointer = urllib.request.urlopen(f"{RAW}/{rel_path}", timeout=60).read().decode()
    key = re.search(r"(MD5E-[^/\s]+)", pointer)
    hashdir = re.search(r"objects/([A-Za-z0-9]{2}/[A-Za-z0-9]{2})/", pointer)
    if not (key and hashdir):
        raise RuntimeError(f"{rel_path} is not an annex pointer: {pointer[:80]!r}")
    blob = urllib.request.urlopen(
        f"{RIA}/{hashdir.group(1)}/{key.group(1)}/{key.group(1)}", timeout=300
    ).read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    return dest


def their_centroids() -> np.ndarray:
    """(1001, 3) MNI-mm centroid per parcel id from the challenge's own atlas."""
    path = fetch_annexed(ATLAS_REL, DATA / "sub-01_schaefer1000_dseg.nii.gz")
    img = nib.load(path)
    vol = np.asarray(img.dataobj)
    out = np.full((N_PARCELS + 1, 3), np.nan)
    for pid in np.unique(vol[vol > 0]).astype(int):
        ijk = np.argwhere(vol == pid).mean(0)
        out[pid] = nib.affines.apply_affine(img.affine, ijk)
    return out


def our_centroids() -> np.ndarray:
    """(1001, 3) fsaverage5 pial centroid per parcel, LH 1..500 then RH 501..1000."""
    from nilearn import datasets

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    out = np.full((N_PARCELS + 1, 3), np.nan)
    for offset, hemi, key in ((0, "lh", "pial_left"), (500, "rh", "pial_right")):
        coords = nib.load(fs[key]).darrays[0].data
        labels, _, _ = nib.freesurfer.read_annot(str(ATLAS_DIR / ANNOT.format(hemi=hemi)))
        for pid in range(1, 501):
            mask = labels == pid
            if mask.any():
                out[pid + offset] = coords[mask].mean(0)
    return out


def main() -> None:
    theirs, ours = their_centroids(), our_centroids()

    lh, rh = theirs[1:501, 0], theirs[501:1001, 0]
    print("their volumetric atlas, hemisphere convention (MNI x < 0 is left):")
    print(f"  parcels    1-500 : {np.mean(lh < 0) * 100:5.1f}% left,  mean x {np.nanmean(lh):+6.1f} mm")
    print(f"  parcels 501-1000 : {np.mean(rh > 0) * 100:5.1f}% right, mean x {np.nanmean(rh):+6.1f} mm")

    ok = ~np.isnan(ours[:, 0]) & ~np.isnan(theirs[:, 0])
    ok[0] = False
    print(f"\ncomparable parcels: {ok.sum()}")
    print("centroid correlation, their volume vs our surface:")
    for i, axis in enumerate("xyz"):
        print(f"  {axis}: r = {np.corrcoef(ours[ok, i], theirs[ok, i])[0, 1]:+.4f}")

    d_true = np.linalg.norm(ours[ok] - theirs[ok], axis=1)
    idx = np.where(ok)[0]
    shuffled = np.random.default_rng(0).permutation(idx)
    d_null = np.linalg.norm(ours[idx] - theirs[shuffled], axis=1)
    print(f"\nmedian distance, parcel i to parcel i        : {np.median(d_true):6.1f} mm")
    print(f"median distance, parcel i to a random parcel : {np.median(d_null):6.1f} mm")

    passed = np.median(d_true) < np.median(d_null) / 3 and np.mean(lh < 0) > 0.95
    print(f"\nA1: {'CONFIRMED - index conventions agree' if passed else 'FAILED - do not submit'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
