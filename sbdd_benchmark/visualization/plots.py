"""
Visualization module — reads pre-computed CSV files only.

STRICT RULE: No RDKit calls, no fingerprint computation, no model loading.
All functions raise FileNotFoundError with a helpful message if the required
CSV is missing, directing the user to run the corresponding evaluation stage.

Public API
----------
plot_stage1(results_dir)   → reads results/stage1/*.csv
plot_stage2(results_dir)   → reads results/stage2/*.csv
plot_stage3(results_dir)   → reads results/stage3/*.csv
plot_all(results_dir)      → calls all three
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; overridden if caller sets one
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_FIGURES_DIR = REPO_ROOT / "results" / "figures"

# ---------------------------------------------------------------------------
# Style constants (match paper)
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "PocketXMol": "#80B1D3",
    "MolCRAFT":   "#8DD1C6",
    "MolCraft":   "#8DD1C6",
    "TamGen":     "#BDBADB",
    "PocketFlow": "#F47F72",
    "Pocket2Mol": "#fd8d3c",
    "ResGen":     "#fed976",
}
DISPLAY_NAMES = {"MolCraft": "MolCRAFT"}
MODEL_ORDER = ["PocketXMol", "MolCRAFT", "TamGen", "PocketFlow", "Pocket2Mol", "ResGen"]
COLOR_PATENT = "#737373"
COLOR_CHEMBL = "#bdbdbd"

FAMILY_COLORS = {
    "Kinase":             "#237B9F",
    "Non-kinase enzymes": "#71BFB2",
    "GPCRs":              "#AD0B08",
    "Nuclear receptors":  "#EC817E",
    "PPI targets":        "#FEE066",
}
TARGET_FAMILY = {
    "ROCK2": "Kinase", "CDK9": "Kinase", "JAK1": "Kinase", "ACVR1": "Kinase", "AKT1": "Kinase",
    "EZH2": "Non-kinase enzymes", "PRMT5": "Non-kinase enzymes",
    "MMP8": "Non-kinase enzymes", "WRN": "Non-kinase enzymes",
    "GCGR": "GPCRs", "5HT2A": "GPCRs", "DRD2": "GPCRs", "AGTR1": "GPCRs",
    "LXRB": "Nuclear receptors", "FXR": "Nuclear receptors", "AR": "Nuclear receptors",
    "BCL2": "PPI targets", "BRD4": "PPI targets", "Keap1": "PPI targets", "EED": "PPI targets",
}
FAMILY_ORDER = ["Kinase", "Non-kinase enzymes", "GPCRs", "Nuclear receptors", "PPI targets"]


def _require_csv(path: Path, stage: str) -> pd.DataFrame:
    """Load CSV or raise a helpful error pointing to the correct CLI command."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"Run evaluation first:\n"
            f"  python cli.py evaluate --stage {stage} --models all --targets all"
        )
    return pd.read_csv(path)


def _model_color(name: str) -> str:
    return MODEL_COLORS.get(name, MODEL_COLORS.get(DISPLAY_NAMES.get(name, ""), "#888888"))


def _display(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def _ordered_models(models: list[str]) -> list[str]:
    order = MODEL_ORDER + [m for m in models if m not in MODEL_ORDER]
    return [m for m in order if m in models]


def _save(fig: plt.Figure, output_dir: Path, stem: str, dpi: int = 300) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved %s (.png/.svg)", stem)


# ---------------------------------------------------------------------------
# Stage 1 plots
# ---------------------------------------------------------------------------

def plot_stage1(
    results_dir: str | Path = None,
    output_dir:  str | Path = None,
) -> None:
    """Generate Stage 1 figures from pre-computed CSVs.

    Produces:
        Figure_Stage1A_MaxTanimoto_Raincloud.png/svg
        Figure_Stage1B_Circles_Coverage.png/svg
    """
    results_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    output_dir  = Path(output_dir)  if output_dir  else results_dir / "figures"
    stage_dir   = results_dir / "stage1"

    sim_df  = _require_csv(stage_dir / "similarity_summary.csv", "1")
    circ_df = _require_csv(stage_dir / "circles_coverage.csv",   "1")

    models = _ordered_models(sim_df["model"].unique().tolist())

    # ---- Figure A: Raincloud (simplified — distributions need per-molecule CSV) ----
    # Load per-molecule scores if available
    score_path = stage_dir / "similarity_scores.csv"
    if score_path.exists():
        score_df = pd.read_csv(score_path)
        _plot_tanimoto_raincloud(score_df, sim_df, models, output_dir)
    else:
        log.warning("similarity_scores.csv not found — skipping raincloud; summary lollipop only")
        _plot_tanimoto_lollipop(sim_df, models, output_dir)

    # ---- Figure B: #Circles ----
    _plot_circles(circ_df, output_dir)


def _plot_tanimoto_raincloud(
    score_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    models: list[str],
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(20, 6))
    gs  = fig.add_gridspec(1, 4, width_ratios=[3, 0.7, 0.7, 0.7], wspace=0.15)
    ax_rain = fig.add_subplot(gs[0])
    axes_lp = [fig.add_subplot(gs[i]) for i in range(1, 4)]

    thresholds = [0.2, 0.3, 0.5]
    fracs = {t: {} for t in thresholds}
    for _, row in summary_df.iterrows():
        for t in thresholds:
            col = f"frac_above_{t}"
            if col in summary_df.columns:
                fracs[t][row["model"]] = row[col]

    for i, model in enumerate(models):
        data  = score_df[score_df["model"] == model]["max_tanimoto_train"].dropna().values
        data  = data[data >= 0]
        color = _model_color(model)
        if len(data) == 0:
            continue

        cloud_h = 0.35
        kde = gaussian_kde(data, bw_method=0.15)
        x_k = np.linspace(0, 1, 300)
        dens = kde(x_k) / kde(x_k).max() * cloud_h
        ax_rain.fill_between(x_k, i, i + dens, alpha=0.6, color=color, edgecolor="none")
        ax_rain.plot(x_k, i + dens, color=color, linewidth=1.2, alpha=0.8)

        ax_rain.boxplot([data], positions=[i - 0.12], vert=False, widths=0.12,
                        patch_artist=True, manage_ticks=False,
                        boxprops=dict(facecolor=color, edgecolor="black", linewidth=1.2, alpha=0.7),
                        medianprops=dict(color="black", linewidth=1.5),
                        whiskerprops=dict(color="black", linewidth=1),
                        capprops=dict(color="black", linewidth=1),
                        flierprops=dict(marker="none"))

        sample = data[np.random.default_rng(42).choice(len(data), min(500, len(data)), replace=False)]
        jitter = np.random.default_rng(42).uniform(-0.12, 0, len(sample))
        ax_rain.scatter(sample, i - 0.26 + jitter, c=color, s=10, alpha=0.4,
                        edgecolors="none", zorder=2)

    ax_rain.set_yticks(range(len(models)))
    ax_rain.set_yticklabels([_display(m) for m in models], fontsize=15)
    ax_rain.set_xlabel("Max Tanimoto Similarity to Training Set", fontsize=18, fontweight="bold")
    ax_rain.set_xlim(-0.02, 1.02)
    ax_rain.set_ylim(-0.45, len(models) - 0.3)
    ax_rain.spines["top"].set_visible(False)
    ax_rain.spines["right"].set_visible(False)
    ax_rain.tick_params(axis="x", labelsize=16)
    ax_rain.grid(axis="x", alpha=0.3, linewidth=0.5)

    for ax, t in zip(axes_lp, thresholds):
        for i, model in enumerate(models):
            frac = fracs[t].get(model, 0)
            color = _model_color(model)
            ax.hlines(i, 0, frac, color=color, linewidth=3, alpha=0.8)
            ax.scatter(frac, i, s=100, color=color, edgecolors="white", linewidths=1.5, zorder=3)
            ax.text(frac, i + 0.2, f"{frac:.0%}", fontsize=14, va="bottom", ha="center", fontweight="bold")
        ax.set_xlim(-0.05, 1.15)
        ax.set_ylim(-0.45, len(models) - 0.3)
        ax.set_xlabel(f"Similarity > {t}", fontsize=16, fontweight="bold")
        ax.set_yticks([])
        ax.set_xticks([0, 0.5])
        ax.set_xticklabels(["0%", "50%"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=16)

    plt.tight_layout()
    _save(fig, output_dir, "Figure_Stage1A_MaxTanimoto_Raincloud")


def _plot_tanimoto_lollipop(summary_df: pd.DataFrame, models: list[str], output_dir: Path) -> None:
    """Fallback: lollipop-only chart when per-molecule scores are unavailable."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    thresholds = [0.2, 0.3, 0.5]
    for ax, t in zip(axes, thresholds):
        col = f"frac_above_{t}"
        if col not in summary_df.columns:
            continue
        for i, model in enumerate(models):
            row = summary_df[summary_df["model"] == model]
            if row.empty:
                continue
            frac  = float(row[col].values[0])
            color = _model_color(model)
            ax.hlines(i, 0, frac, color=color, linewidth=3, alpha=0.8)
            ax.scatter(frac, i, s=120, color=color, edgecolors="white", linewidths=1.5, zorder=3)
            ax.text(frac + 0.02, i, f"{frac:.0%}", fontsize=13, va="center")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([_display(m) for m in models], fontsize=13)
        ax.set_xlabel(f"Fraction with Tanimoto > {t}", fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(fig, output_dir, "Figure_Stage1A_MaxTanimoto_Lollipop")


def _plot_circles(circ_df: pd.DataFrame, output_dir: Path) -> None:
    models = _ordered_models(circ_df["model"].unique().tolist())
    SCALE  = 1_000.0

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(20, 6))

    for i, model in enumerate(models):
        row   = circ_df[circ_df["model"] == model].iloc[0]
        color = _model_color(model)
        val_k = row["circles_mean"] / SCALE
        eff   = row.get("efficiency", row["circles_mean"] / max(row["n_sampled"], 1) * 1000)

        ax_l.barh(i, val_k, height=0.6, color=color, alpha=0.85, edgecolor="black", linewidth=1)
        ax_l.text(val_k + 0.5, i, f"{row['circles_mean']:.0f} ± {row['circles_std']:.0f}",
                  fontsize=16, va="center", ha="left", fontweight="bold")

        ax_r.hlines(i, 0, eff, color=color, linewidth=4, alpha=0.85)
        ax_r.scatter(eff, i, s=150, color=color, edgecolors="white", linewidths=2, zorder=3)
        ax_r.text(eff + 2, i, f"{eff:.1f}", fontsize=16, va="center", ha="left", fontweight="bold")

    for ax in (ax_l, ax_r):
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([_display(m) for m in models], fontsize=18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=18)
        ax.grid(axis="x", alpha=0.3, linewidth=0.5)

    ax_l.set_xlabel("#Circles Coverage Score (×10³)", fontsize=20, fontweight="bold")
    ax_r.set_xlabel("Coverage Efficiency (per 1k molecules)", fontsize=20, fontweight="bold")

    plt.tight_layout()
    _save(fig, output_dir, "Figure_Stage1B_Circles_Coverage")


# ---------------------------------------------------------------------------
# Stage 2 plots
# ---------------------------------------------------------------------------

def plot_stage2(
    results_dir: str | Path = None,
    output_dir:  str | Path = None,
) -> None:
    """Generate Stage 2 figures from pre-computed CSVs.

    Produces:
        Figure_Stage2A_REOS_PassRate.png/svg
        Figure_Stage2B_Wasserstein_Heatmap.png/svg
        Figure_Stage2C_Ridgeline_*.png/svg  (if per-molecule property CSVs exist)
    """
    results_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    output_dir  = Path(output_dir)  if output_dir  else results_dir / "figures"
    stage_dir   = results_dir / "stage2"

    reos_df = _require_csv(stage_dir / "reos_summary.csv", "2")
    _plot_reos_passrate(reos_df, output_dir)

    heatmap_path = stage_dir / "wasserstein_heatmap.csv"
    if heatmap_path.exists():
        heatmap_df = pd.read_csv(heatmap_path, index_col=0)
        _plot_wasserstein_heatmap(heatmap_df, output_dir)
    else:
        log.warning("wasserstein_heatmap.csv not found — skipping heatmap")

    # Per-molecule ridgeline (optional — needs property CSVs)
    _plot_ridgelines_if_available(stage_dir, output_dir)


def _plot_reos_passrate(reos_df: pd.DataFrame, output_dir: Path) -> None:
    models = _ordered_models(reos_df["model"].unique().tolist())
    rates  = {row["model"]: row["reos_pass_pct"] for _, row in reos_df.iterrows()}

    fig, ax = plt.subplots(figsize=(8, 6))
    all_names = models
    for i, name in enumerate(all_names):
        color = _model_color(name)
        rate  = rates.get(name, 0)
        ax.bar(i, rate, width=0.7, color=color, edgecolor="black", linewidth=1.0)
        ax.text(i, rate + 1.5, f"{rate:.1f}%", ha="center", va="bottom",
                fontsize=15, fontweight="bold")

    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels([_display(m) for m in all_names], fontsize=16, rotation=30, ha="right")
    ax.set_ylabel("REOS Pass Rate (%)", fontsize=18)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="both", labelsize=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    _save(fig, output_dir, "Figure_Stage2A_REOS_PassRate")


def _plot_wasserstein_heatmap(heatmap_df: pd.DataFrame, output_dir: Path) -> None:
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns

    # Order models
    ordered = _ordered_models(heatmap_df.index.tolist())
    heatmap_df = heatmap_df.loc[[m for m in ordered if m in heatmap_df.index]]
    heatmap_df.index = [_display(m) for m in heatmap_df.index]

    cmap = LinearSegmentedColormap.from_list("white_darkred", ["#ffffff", "#b2182b"], N=256)
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.heatmap(
        heatmap_df.astype(float), cmap=cmap, vmin=0, vmax=1,
        linewidths=0.8, linecolor="white",
        annot=True, fmt=".2f",
        annot_kws={"fontsize": 10, "weight": "bold"},
        cbar_kws={"label": "Normalized Wasserstein Distance\n(relative to Patent)",
                  "shrink": 0.85},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    _save(fig, output_dir, "Figure_Stage2B_Wasserstein_Heatmap")


def _plot_ridgelines_if_available(stage_dir: Path, output_dir: Path) -> None:
    """Plot ridgeline distributions if per-molecule property CSVs are present."""
    from scipy.stats import gaussian_kde

    prop_files = list(stage_dir.glob("*_properties.csv"))
    if not prop_files:
        log.info("No property CSVs found — skipping ridgeline plots")
        return

    dfs = []
    for f in prop_files:
        df = pd.read_csv(f)
        if "model" not in df.columns:
            df["model"] = f.stem.replace("_properties", "")
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    reos_mask = combined["REOS_Pass"].astype(str).str.lower().isin(["true", "1"])
    combined = combined[reos_mask]

    models = _ordered_models(combined["model"].unique().tolist())
    PROPS_CONT = [("MW", (100, 900), "Molecular Weight (Da)"),
                  ("Fsp3", (0, 1), "Fsp³"),
                  ("N_RotB", (0, 25), "Rotatable Bonds")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    overlap = 0.6
    for ax, (col, xlim, xlabel) in zip(axes, PROPS_CONT):
        n = len(models)
        x_range = np.linspace(*xlim, 500)
        for i, model in enumerate(models):
            data = combined[combined["model"] == model][col].dropna().values
            if len(data) < 10:
                continue
            color  = _model_color(model)
            kde    = gaussian_kde(data, bw_method=0.3)
            y_raw  = kde(x_range)
            y_norm = y_raw / y_raw.max() * 0.8
            offset = (n - 1 - i) * overlap
            ax.fill_between(x_range, offset, y_norm + offset, color=color, alpha=0.6)
            ax.plot(x_range, y_norm + offset, color=color, linewidth=1.5)
            ax.text(x_range[0] - (x_range[-1] - x_range[0]) * 0.02,
                    offset + 0.08, _display(model), ha="right", va="bottom", fontsize=12)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_xlim(*xlim)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    plt.tight_layout()
    _save(fig, output_dir, "Figure_Stage2C_Ridgeline_Continuous")


# ---------------------------------------------------------------------------
# Stage 3 plots
# ---------------------------------------------------------------------------

def plot_stage3(
    results_dir: str | Path = None,
    output_dir:  str | Path = None,
) -> None:
    """Generate Stage 3 figures from pre-computed CSVs.

    Produces:
        Figure_Stage3A_Recovery_Barplot.png/svg
        Figure_Stage3B_Recovery_Heatmap.png/svg
        Figure_Stage3C_Radar.png/svg
    """
    results_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    output_dir  = Path(output_dir)  if output_dir  else results_dir / "figures"
    stage_dir   = results_dir / "stage3"

    summary_df = _require_csv(stage_dir / "recovery_summary.csv", "3")
    _plot_recovery_barplot(summary_df, output_dir)

    # Heatmap needs per-target recovery files
    per_target_dfs = []
    for f in sorted(stage_dir.glob("*_recovery.csv")):
        df = pd.read_csv(f)
        if "model" not in df.columns:
            df["model"] = f.stem.replace("_recovery", "")
        per_target_dfs.append(df)
    if per_target_dfs:
        combined = pd.concat(per_target_dfs, ignore_index=True)
        _plot_recovery_heatmap(combined, output_dir)
        _plot_radar(combined, summary_df, output_dir)
    else:
        log.warning("No *_recovery.csv files found — skipping heatmap and radar")


def _plot_recovery_barplot(summary_df: pd.DataFrame, output_dir: Path) -> None:
    models = _ordered_models(summary_df["model"].unique().tolist())
    models_sorted = sorted(
        models,
        key=lambda m: summary_df[summary_df["model"] == m]["generic_mean_%"].values[0]
        if len(summary_df[summary_df["model"] == m]) else 0
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    def _bar(ax, col_mean, col_sem, xlabel, xlim):
        for i, m in enumerate(models_sorted):
            row   = summary_df[summary_df["model"] == m]
            if row.empty:
                continue
            mean_ = float(row[col_mean].values[0])
            sem_  = float(row[col_sem].values[0])  if col_sem in row.columns else 0
            color = _model_color(m)
            ax.barh(i, mean_, height=0.5, xerr=sem_, color=color, edgecolor="black",
                    linewidth=1.0, capsize=3,
                    error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "black"})
            ax.text(mean_ + sem_ + xlim * 0.01, i, f"{mean_:.2f}%",
                    ha="left", va="center", fontsize=11, fontweight="bold")
        ax.set_yticks(range(len(models_sorted)))
        ax.set_yticklabels([_display(m) for m in models_sorted], fontsize=13)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_xlim(0, xlim)
        ax.set_ylim(-0.5, len(models_sorted) - 0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    _bar(axes[0], "bm_mean_%",      "bm_sem_%",      "BM Scaffold Recovery Rate (%)",      3.0)
    _bar(axes[1], "generic_mean_%", "generic_sem_%",  "Generic Scaffold Recovery Rate (%)", 20.0)
    axes[0].set_title("BM Scaffold: Mean Recovery Rate",      fontsize=15, fontweight="bold")
    axes[1].set_title("Generic Scaffold: Mean Recovery Rate", fontsize=15, fontweight="bold")

    plt.tight_layout()
    _save(fig, output_dir, "Figure_Stage3A_Recovery_Barplot")


def _plot_recovery_heatmap(combined: pd.DataFrame, output_dir: Path) -> None:
    """Diagonal split heatmap: BM (upper-left) / Generic (lower-right) per cell."""
    import matplotlib.patches as patches

    if "bm_scaffold_recovery_ratio" not in combined.columns:
        return

    targets_ordered: list[str] = []
    for fam in FAMILY_ORDER:
        targets_ordered.extend(
            sorted(t for t in combined["target"].unique() if TARGET_FAMILY.get(t) == fam)
        )
    targets_ordered = [t for t in targets_ordered if t in combined["target"].unique()]

    models_all   = _ordered_models(combined["model"].unique().tolist())
    pivot_bm     = combined.pivot(index="model", columns="target", values="bm_scaffold_recovery_ratio").reindex(index=models_all, columns=targets_ordered) * 100
    pivot_gen    = combined.pivot(index="model", columns="target", values="generic_scaffold_recovery_ratio").reindex(index=models_all, columns=targets_ordered) * 100

    import matplotlib.colors as mcolors
    norm_bm  = mcolors.Normalize(vmin=0, vmax=10)
    norm_gen = mcolors.Normalize(vmin=0, vmax=50)

    fam_cmaps = {f: LinearSegmentedColormap.from_list("c", ["#FFFFFF", c], N=256)
                 for f, c in FAMILY_COLORS.items()}

    n_m, n_t = len(models_all), len(targets_ordered)
    fig = plt.figure(figsize=(max(14, n_t * 0.9), max(4, n_m * 0.9)))
    ax  = fig.add_subplot(111)

    for i, model in enumerate(models_all):
        for j, target in enumerate(targets_ordered):
            bv  = pivot_bm.loc[model, target]  if model in pivot_bm.index  else 0
            gv  = pivot_gen.loc[model, target] if model in pivot_gen.index else 0
            fam = TARGET_FAMILY.get(target, "Kinase")
            cmap = fam_cmaps[fam]
            ax.add_patch(patches.Polygon([[j, n_m-i], [j+1, n_m-i], [j, n_m-i-1]], closed=True,
                                          facecolor=cmap(norm_bm(bv)), edgecolor="white", linewidth=0.6))
            ax.add_patch(patches.Polygon([[j+1, n_m-i], [j+1, n_m-i-1], [j, n_m-i-1]], closed=True,
                                          facecolor=cmap(norm_gen(gv)), edgecolor="white", linewidth=0.6))
            ax.add_patch(patches.Rectangle((j, n_m-i-1), 1, 1, fill=False,
                                            edgecolor="white", linewidth=0.6))
            bv_c = "white" if norm_bm(bv)  > 0.5 else "black"
            gv_c = "white" if norm_gen(gv) > 0.5 else "black"
            ax.text(j+0.28, n_m-i-0.25, f"{bv:.1f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=bv_c)
            ax.text(j+0.72, n_m-i-0.75, f"{gv:.1f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=gv_c)

    ax.set_xlim(0, n_t); ax.set_ylim(0, n_m)
    ax.set_xticks(np.arange(n_t) + 0.5)
    ax.set_xticklabels(targets_ordered, rotation=0, ha="center", fontsize=11)
    ax.set_yticks(np.arange(n_m) + 0.5)
    ax.set_yticklabels(models_all[::-1], fontsize=13)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.2); sp.set_color("black")
    ax.tick_params(axis="both", length=0)

    plt.tight_layout()
    _save(fig, output_dir, "Figure_Stage3B_Recovery_Heatmap")


def _plot_radar(combined: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> None:
    from math import pi

    metrics = ["exact_recovery_ratio", "bm_scaffold_recovery_ratio", "generic_scaffold_recovery_ratio"]
    labels  = ["Exact\nRecovery", "BM\nScaffold", "Generic\nScaffold"]
    models  = _ordered_models(summary_df["model"].unique().tolist())

    model_means = combined.groupby("model")[metrics].mean()
    model_means_norm = (model_means - model_means.min()) / (model_means.max() - model_means.min() + 1e-10)

    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))] + [0]

    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, projection="polar")

    for model in models:
        if model not in model_means_norm.index:
            continue
        vals   = model_means_norm.loc[model].tolist() + [model_means_norm.loc[model].tolist()[0]]
        color  = _model_color(model)
        ax.plot(angles, vals, "o-", linewidth=2.5, label=_display(model), color=color, markersize=6)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_title("Scaffold Recovery: All Models", size=15, fontweight="bold", pad=16)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.1), fontsize=11)
    ax.spines["polar"].set_color("#9E9E9E")
    ax.spines["polar"].set_linewidth(1.4)
    ax.grid(True, color="#B0B0B0", alpha=0.6, linewidth=0.9)

    plt.tight_layout()
    _save(fig, output_dir, "Figure_Stage3C_Radar")


# ---------------------------------------------------------------------------
# Convenience: plot all
# ---------------------------------------------------------------------------

def plot_all(
    results_dir: str | Path = None,
    output_dir:  str | Path = None,
) -> None:
    """Run all three stage plot functions."""
    for fn, stage in [(plot_stage1, 1), (plot_stage2, 2), (plot_stage3, 3)]:
        try:
            fn(results_dir=results_dir, output_dir=output_dir)
            print(f"[plot_all] Stage {stage} figures saved.")
        except FileNotFoundError as exc:
            print(f"[plot_all] Stage {stage} skipped — {exc}")
