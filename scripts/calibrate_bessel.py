"""Calibrate regime crossover thresholds for ``numerax.special.ive``.

Each of the three regimes (power series, Hankel asymptotic, Olver UAE)
is evaluated on a fine ``(v, z)`` grid and compared to
``scipy.special.ive``. The script produces:

- per-regime relative-error heatmaps,
- a *regime-validity overlay* showing which cells each regime meets a
  fixed accuracy threshold (default 1e-7),
- a *best-regime* map (which regime has the smallest error at each
  cell, ignoring those that don't meet the threshold),
- a combined-``ive`` plot with the current production thresholds
  overlaid so the parametric boundary can be visually compared against
  the data-derived regime-validity regions.

The committed PNGs in ``scripts/calibration_plots/`` are the artifact
that documents the chosen thresholds; re-run this script after any
algorithmic change to ``_bessel.py`` and re-commit both the PNGs and
any threshold updates.

Run:
    conda run -n jax-ml python scripts/calibrate_bessel.py
"""

import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import (  # noqa: E402
    BoundaryNorm,
    ListedColormap,
    LogNorm,
)
from matplotlib.patches import Patch  # noqa: E402
from scipy.special import ive as scipy_ive  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from numerax.special import ive  # noqa: E402
from numerax.special._bessel import (  # noqa: E402
    _OLVER_V_THRESHOLD,
    _SERIES_Z_THRESHOLD,
    _ive_hankel,
    _ive_olver,
    _ive_series,
)

PLOT_DIR = REPO_ROOT / "scripts" / "calibration_plots"
PLOT_DIR.mkdir(exist_ok=True)

# Per-cell "valid" threshold. One decade below the 1e-6 final-product
# target gives us margin against accumulated error.
ACCURACY_THRESHOLD = 1e-7

REGIME_NAMES = ["series", "hankel", "olver"]
REGIME_COLORS = ["#1f77b4", "#2ca02c", "#d62728"]  # blue, green, red


def make_grid():
    """Build a (v, z) grid covering the union of regime domains.

    v starts at 0.1 to keep log-scale plots readable. The plan
    explicitly targets ``v >= 0``; v=0 behaviour is exercised in the
    test suite, not the calibration plots.
    """
    v_grid = np.logspace(np.log10(0.1), np.log10(500.0), 80)
    z_grid = np.logspace(-2, 4, 100)
    return v_grid, z_grid


def relative_error(numerax_vals, scipy_vals):
    """Elementwise |a-b|/|b|; NaN where scipy underflowed to zero."""
    abs_scipy = np.abs(scipy_vals)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        err = np.abs(numerax_vals - scipy_vals) / abs_scipy
    return np.where(abs_scipy > 0, err, np.nan)


def evaluate_grid(fn, v_mesh, z_mesh):
    return np.asarray(fn(jnp.asarray(v_mesh), jnp.asarray(z_mesh)))


def plot_heatmap(v_grid, z_grid, rel_err, title, outpath, boundary_lines=None):
    """2D log-log relative-error heatmap on the (v, z) grid."""
    fig, ax = plt.subplots(figsize=(8, 6))
    clipped = np.clip(rel_err, 1e-16, 1.0)
    mesh = ax.pcolormesh(
        v_grid,
        z_grid,
        clipped.T,
        norm=LogNorm(vmin=1e-15, vmax=1.0),
        cmap="viridis",
        shading="auto",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("v (order)")
    ax.set_ylabel("z (argument)")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="relative error vs scipy.special.ive")
    if boundary_lines is not None:
        for line in boundary_lines:
            ax.plot(line["x"], line["y"], **line["style"])
        ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"  wrote {outpath.relative_to(REPO_ROOT)}")


def plot_validity_overlay(
    v_grid, z_grid, valid_masks, title, outpath, boundary_lines=None
):
    """Show shaded regions where each regime meets the threshold.

    ``valid_masks`` is a list of (v, z)-shaped boolean arrays, one per
    regime in REGIME_NAMES order. Each is rendered as a transparent
    overlay with the regime's color, so overlapping (valid in multiple
    regimes) regions blend.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, color, mask in zip(
        REGIME_NAMES, REGIME_COLORS, valid_masks, strict=False
    ):
        rgba = np.zeros(mask.shape + (4,))
        rgba[mask] = (*_hex_to_rgb(color), 0.35)
        ax.pcolormesh(
            v_grid, z_grid, np.transpose(rgba, (1, 0, 2)), shading="auto"
        )
        ax.plot([], [], color=color, alpha=0.6, linewidth=8, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("v (order)")
    ax.set_ylabel("z (argument)")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9, title="regime is valid")
    if boundary_lines is not None:
        for line in boundary_lines:
            ax.plot(line["x"], line["y"], **line["style"])
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"  wrote {outpath.relative_to(REPO_ROOT)}")


def plot_best_regime(
    v_grid, z_grid, errs_stack, title, outpath, boundary_lines=None
):
    """Categorical map: which regime has the smallest error per cell.

    Cells where no regime achieves the accuracy threshold are shown in
    grey; cells where scipy underflowed are left white.
    """
    errs_filled = np.where(np.isnan(errs_stack), np.inf, errs_stack)
    best_idx = np.argmin(errs_filled, axis=0)
    best_err = np.take_along_axis(errs_filled, best_idx[None, :, :], axis=0)[0]

    # 0..2 = regimes; 3 = no regime good enough; 4 = scipy underflow.
    categorical = np.where(best_err <= ACCURACY_THRESHOLD, best_idx, 3).astype(
        float
    )
    categorical = np.where(
        np.all(np.isnan(errs_stack), axis=0), 4, categorical
    )

    cmap = ListedColormap([*REGIME_COLORS, "#888888", "#ffffff"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(
        v_grid, z_grid, categorical.T, cmap=cmap, norm=norm, shading="auto"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("v (order)")
    ax.set_ylabel("z (argument)")
    ax.set_title(title)
    handles = [
        Patch(color=REGIME_COLORS[0], label="series best"),
        Patch(color=REGIME_COLORS[1], label="hankel best"),
        Patch(color=REGIME_COLORS[2], label="olver best"),
        Patch(color="#888888", label=f"none < {ACCURACY_THRESHOLD:.0e}"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9)
    if boundary_lines is not None:
        for line in boundary_lines:
            ax.plot(line["x"], line["y"], **line["style"])
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"  wrote {outpath.relative_to(REPO_ROOT)}")


def _hex_to_rgb(hex_str):
    h = hex_str.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def production_boundary_lines(v_grid, z_grid):
    """Boundary curves implied by the current ``_bessel.py`` thresholds."""
    return [
        {
            "x": [v_grid.min(), v_grid.max()],
            "y": [_SERIES_Z_THRESHOLD, _SERIES_Z_THRESHOLD],
            "style": {
                "color": "black",
                "linestyle": "--",
                "linewidth": 1.5,
                "label": f"series cut: z = {_SERIES_Z_THRESHOLD:g}",
            },
        },
        {
            "x": [_OLVER_V_THRESHOLD, _OLVER_V_THRESHOLD],
            "y": [_SERIES_Z_THRESHOLD, z_grid.max()],
            "style": {
                "color": "black",
                "linestyle": ":",
                "linewidth": 1.5,
                "label": f"Hankel/Olver cut: v = {_OLVER_V_THRESHOLD:g}",
            },
        },
    ]


def summarise(name, rel_err, mask_desc, mask):
    masked = np.where(mask, rel_err, np.nan)
    finite = masked[np.isfinite(masked)]
    if finite.size == 0:
        print(f"  {name:8s}  [{mask_desc:40s}]  (no finite points)")
        return
    p_max = float(np.max(finite))
    p_99 = float(np.quantile(finite, 0.99))
    print(
        f"  {name:8s}  [{mask_desc:40s}]  "
        f"max={p_max:.2e}  99%={p_99:.2e}  N={finite.size}"
    )


def main():
    v_grid, z_grid = make_grid()
    v_mesh, z_mesh = np.meshgrid(v_grid, z_grid, indexing="ij")
    v_mesh_j, z_mesh_j = jnp.asarray(v_mesh), jnp.asarray(z_mesh)

    print(f"Grid: {v_mesh.shape[0]} v x {v_mesh.shape[1]} z = {v_mesh.size} points")
    print(f"  v in [{v_grid.min():.3f}, {v_grid.max():.1f}]")
    print(f"  z in [{z_grid.min():.3f}, {z_grid.max():.1f}]")
    print(f"  validity threshold: {ACCURACY_THRESHOLD:.0e}")
    print()

    scipy_vals = scipy_ive(v_mesh, z_mesh)

    print("Evaluating regimes...")
    series_vals = evaluate_grid(_ive_series, v_mesh_j, z_mesh_j)
    hankel_vals = evaluate_grid(_ive_hankel, v_mesh_j, z_mesh_j)
    olver_vals = evaluate_grid(_ive_olver, v_mesh_j, z_mesh_j)
    combined_vals = evaluate_grid(ive, v_mesh_j, z_mesh_j)

    err_series = relative_error(series_vals, scipy_vals)
    err_hankel = relative_error(hankel_vals, scipy_vals)
    err_olver = relative_error(olver_vals, scipy_vals)
    err_combined = relative_error(combined_vals, scipy_vals)
    errs_stack = np.stack([err_series, err_hankel, err_olver])

    valid_series = err_series <= ACCURACY_THRESHOLD
    valid_hankel = err_hankel <= ACCURACY_THRESHOLD
    valid_olver = err_olver <= ACCURACY_THRESHOLD
    valid_combined = err_combined <= ACCURACY_THRESHOLD
    print()
    print(f"Cells where each regime achieves err <= {ACCURACY_THRESHOLD:.0e}:")
    nfinite = np.sum(~np.isnan(err_series))
    for name, m in zip(
        [*REGIME_NAMES, "combined"],
        [valid_series, valid_hankel, valid_olver, valid_combined],
        strict=False,
    ):
        n_valid = int(np.sum(m & ~np.isnan(err_series)))
        print(f"  {name:8s}: {n_valid:5d} / {nfinite} valid cells")

    # Combined max error.
    print()
    print("Combined ive error (over all cells where scipy is finite):")
    summarise(
        "combined",
        err_combined,
        "all (v, z) with finite scipy",
        ~np.isnan(err_combined),
    )

    print()
    print("Writing plots...")
    boundaries = production_boundary_lines(v_grid, z_grid)
    plot_heatmap(
        v_grid,
        z_grid,
        err_series,
        "Power series — relative error vs scipy",
        PLOT_DIR / "series.png",
    )
    plot_heatmap(
        v_grid,
        z_grid,
        err_hankel,
        "Hankel asymptotic — relative error vs scipy",
        PLOT_DIR / "hankel.png",
    )
    plot_heatmap(
        v_grid,
        z_grid,
        err_olver,
        "Olver uniform asymptotic — relative error vs scipy",
        PLOT_DIR / "olver.png",
    )
    plot_validity_overlay(
        v_grid,
        z_grid,
        [valid_series, valid_hankel, valid_olver],
        f"Regime validity overlay (err <= {ACCURACY_THRESHOLD:.0e})",
        PLOT_DIR / "regime_validity.png",
        boundary_lines=boundaries,
    )
    plot_best_regime(
        v_grid,
        z_grid,
        errs_stack,
        "Best regime per cell (data-derived) — current thresholds overlaid",
        PLOT_DIR / "best_regime.png",
        boundary_lines=boundaries,
    )
    plot_heatmap(
        v_grid,
        z_grid,
        err_combined,
        "Combined ive — relative error vs scipy (current thresholds)",
        PLOT_DIR / "combined.png",
        boundary_lines=boundaries,
    )

    print()
    print("Current thresholds:")
    print(f"  _SERIES_Z_THRESHOLD = {_SERIES_Z_THRESHOLD}")
    print(f"  _OLVER_V_THRESHOLD = {_OLVER_V_THRESHOLD}")


if __name__ == "__main__":
    main()
