"""Brain surface visualization using nilearn."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_1d(data: np.ndarray) -> np.ndarray:
    """Collapse to 1D by averaging over segments if needed."""
    if data.ndim == 2:
        return data.mean(axis=0)
    return data


def plot_activation(
    data: np.ndarray,
    title: str = "",
    cmap: str = "cold_hot",
    hemisphere: str = "both",
    output_path: Optional[Path] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Plot activation map on fsaverage5 brain surface.

    Args:
        data: shape (n_vertices,) or (n_segments, n_vertices).
        title: Plot title.
        cmap: Colormap name.
        hemisphere: "left", "right", or "both".
        output_path: If provided, saves PNG.
        vmin/vmax: Colorbar range.

    Returns:
        matplotlib figure.
    """
    from nilearn import datasets, plotting

    data = _ensure_1d(data)
    n_vertices = len(data)
    hemi_size = n_vertices // 2

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    views = ["lateral", "medial"]
    hemis_to_plot = []
    if hemisphere in ["left", "both"]:
        hemis_to_plot.append(("left", data[:hemi_size]))
    if hemisphere in ["right", "both"]:
        hemis_to_plot.append(("right", data[hemi_size:]))

    fig, axes = None, None
    import matplotlib.pyplot as plt
    n_plots = len(hemis_to_plot) * len(views)
    fig, axes = plt.subplots(
        len(hemis_to_plot), len(views),
        figsize=(6 * len(views), 5 * len(hemis_to_plot)),
        subplot_kw={"projection": "3d"},
    )
    if n_plots == 1:
        axes = np.array([[axes]])
    elif len(hemis_to_plot) == 1:
        axes = axes[np.newaxis, :]
    elif len(views) == 1:
        axes = axes[:, np.newaxis]

    for i, (hemi, hemi_data) in enumerate(hemis_to_plot):
        mesh_key = f"pial_{hemi}"
        for j, view in enumerate(views):
            plotting.plot_surf_stat_map(
                fsaverage[mesh_key],
                hemi_data,
                hemi=hemi,
                view=view,
                colorbar=True,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                title=f"{title} ({hemi}, {view})" if title else f"{hemi} {view}",
                axes=axes[i, j],
                figure=fig,
            )

    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Saved to %s", output_path)

    return fig


def plot_contrast(
    data_a: np.ndarray,
    data_b: np.ndarray,
    title: str = "A vs B",
    output_path: Optional[Path] = None,
):
    """Plot the difference (A - B) on brain surface.

    Positive values (A > B) in warm colors, negative in cool colors.
    """
    data_a = _ensure_1d(data_a)
    data_b = _ensure_1d(data_b)
    diff = data_a - data_b
    vmax = max(abs(diff.min()), abs(diff.max()))
    return plot_activation(
        diff, title=title, cmap="cold_hot",
        vmin=-vmax, vmax=vmax, output_path=output_path,
    )


def plot_rgb_attribution(
    visual: np.ndarray,
    auditory: np.ndarray,
    language: np.ndarray,
    title: str = "Modality Attribution",
    output_path: Optional[Path] = None,
):
    """BrainLens RGB brain map.

    Red = visual, Green = language, Blue = auditory.
    Each array shape (n_vertices,). Normalized to [0,1].
    """
    import matplotlib.pyplot as plt
    from nilearn import datasets, surface

    visual = _ensure_1d(visual)
    auditory = _ensure_1d(auditory)
    language = _ensure_1d(language)

    # Normalize each channel to [0, 1]
    def _norm(x):
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-10:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    r = _norm(visual)
    g = _norm(language)
    b = _norm(auditory)

    n_vertices = len(r)
    hemi_size = n_vertices // 2
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={"projection": "3d"})

    for i, hemi in enumerate(["left", "right"]):
        sl = slice(0, hemi_size) if hemi == "left" else slice(hemi_size, None)
        rgb = np.stack([r[sl], g[sl], b[sl]], axis=-1)

        mesh_key = f"pial_{hemi}"
        coords, faces = surface.load_surf_mesh(fsaverage[mesh_key])

        for j, view in enumerate(["lateral", "medial"]):
            ax = axes[i, j]
            ax.plot_trisurf(
                coords[:, 0], coords[:, 1], coords[:, 2],
                triangles=faces,
                antialiased=False,
                linewidth=0,
            )
            # Color each face by averaging vertex colors
            face_colors = rgb[faces].mean(axis=1)
            ax.collections[0].set_facecolors(face_colors)
            ax.set_title(f"{hemi} {view}")
            ax.axis("off")
            if view == "medial":
                ax.view_init(elev=0, azim=180 if hemi == "left" else 0)
            else:
                ax.view_init(elev=0, azim=180 if hemi == "right" else 0)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Saved to %s", output_path)

    return fig


def export_html(
    data: np.ndarray,
    title: str = "",
    output_path: Path = Path("brain_map.html"),
    hemisphere: str = "both",
) -> Path:
    """Export interactive 3D brain map as HTML using nilearn.

    Args:
        data: shape (n_vertices,) or (n_segments, n_vertices).
        title: Plot title.
        output_path: Where to save the HTML file.
        hemisphere: "left", "right", or "both". When "both", saves two files
            (appends _left.html and _right.html) and returns the left path.

    Returns:
        Path to the saved HTML file.
    """
    from nilearn import datasets, plotting

    data = _ensure_1d(data)
    n_vertices = len(data)
    hemi_size = n_vertices // 2
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    output_path = Path(output_path)

    hemis = []
    if hemisphere in ("left", "both"):
        hemis.append(("left", data[:hemi_size], fsaverage["pial_left"]))
    if hemisphere in ("right", "both"):
        hemis.append(("right", data[hemi_size:], fsaverage["pial_right"]))

    saved_path = output_path
    for hemi_name, hemi_data, mesh in hemis:
        if hemisphere == "both":
            stem = output_path.stem
            suffix = output_path.suffix or ".html"
            hemi_path = output_path.with_name(f"{stem}_{hemi_name}{suffix}")
        else:
            hemi_path = output_path

        label = f"{title} ({hemi_name})" if title else f"Brain Map ({hemi_name})"
        view = plotting.view_surf(mesh, hemi_data, title=label, colorbar=True)
        view.save_as_html(str(hemi_path))
        logger.info("Saved interactive map to %s", hemi_path)
        if hemi_name == "left" or hemisphere != "both":
            saved_path = hemi_path

    return saved_path
