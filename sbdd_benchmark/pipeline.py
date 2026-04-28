"""
High-level pipeline orchestration.

run_stage1(models, targets, data_dir, results_dir, config)
run_stage2(models, targets, data_dir, results_dir, config)
run_stage3(models, targets, data_dir, results_dir, config)
run_all(models, targets, data_dir, results_dir, config)

Architecture (strict computation / visualization separation)
------------------------------------------------------------
Each run_stageN() function:
  1. Loads pre-computed unified parquets from data/generated/unified/
  2. Runs computation
  3. Saves tidy CSV files to results/stageN/
  4. Returns nothing — visualization reads the CSVs separately

Plotting is handled by sbdd_benchmark.visualization.plots, which only
reads from results/stageN/ and never calls RDKit or computes fingerprints.
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR    = REPO_ROOT / "data"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

ALL_MODELS = ["Pocket2Mol", "MolCRAFT", "TamGen", "PocketFlow", "PocketXMol", "ResGen"]
ALL_TARGETS = [
    "ROCK2", "CDK9", "JAK1", "ACVR1", "AKT1",
    "EZH2", "PRMT5", "MMP8", "WRN",
    "GCGR", "5HT2A", "DRD2", "AGTR1",
    "LXRB", "FXR", "AR",
    "BCL2", "BRD4", "Keap1", "EED",
]

TARGET_FAMILIES = {
    "kinase":              ["ROCK2", "CDK9", "JAK1", "ACVR1", "AKT1"],
    "non_kinase_enzyme":   ["EZH2", "PRMT5", "MMP8", "WRN"],
    "gpcr":                ["GCGR", "5HT2A", "DRD2", "AGTR1"],
    "nuclear_receptor":    ["LXRB", "FXR", "AR"],
    "ppi":                 ["BCL2", "BRD4", "Keap1", "EED"],
}


def _resolve_targets(targets_arg) -> list[str]:
    """Resolve target list from string keywords or explicit list."""
    if targets_arg is None or targets_arg == "all":
        return ALL_TARGETS
    if isinstance(targets_arg, str):
        targets_arg = [targets_arg]
    resolved = []
    for t in targets_arg:
        if t in TARGET_FAMILIES:
            resolved.extend(TARGET_FAMILIES[t])
        elif t == "all":
            resolved.extend(ALL_TARGETS)
        else:
            resolved.append(t)
    return list(dict.fromkeys(resolved))  # deduplicate, preserve order


def _resolve_models(models_arg) -> list[str]:
    if models_arg is None or models_arg == "all":
        return ALL_MODELS
    if isinstance(models_arg, str):
        models_arg = [models_arg]
    return list(models_arg)


def _load_parquet(model: str, data_dir: Path) -> Optional[pd.DataFrame]:
    """Load unified parquet for a model, returning None with warning on missing."""
    from sbdd_benchmark.io.loaders import load_generated
    try:
        return load_generated(model, data_dir=data_dir / "generated" / "unified", valid_only=True)
    except FileNotFoundError as exc:
        log.warning(str(exc))
        return None


def _load_benchmark_for_target(target: str, data_dir: Path) -> Optional[pd.DataFrame]:
    from sbdd_benchmark.io.loaders import load_benchmark
    try:
        return load_benchmark(target, data_dir=data_dir / "benchmark")
    except FileNotFoundError as exc:
        log.warning(str(exc))
        return None


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def run_stage1(
    models=None,
    targets=None,
    data_dir: str | Path = None,
    results_dir: str | Path = None,
    fp_cache_dir: str | Path = None,
    seed: int = 42,
    n_circles_runs: int = 3,
    circles_threshold: float = 0.75,
    compare_baselines: bool = False,
) -> Path:
    """Run Stage 1: Chemical Space Exploration.

    Outputs (saved to results/stage1/):
        similarity_scores.csv    — per-molecule MaxTanimoto to training set
        similarity_summary.csv   — per-model mean, median, frac>0.2/0.3/0.5
        circles_coverage.csv     — #Circles mean±std, efficiency per model

    Parameters
    ----------
    compare_baselines:
        If True, load pre-computed baseline results from results/baselines/stage1/
        and merge them with newly computed model results.

    Returns
    -------
    Path to the stage1 results directory.
    """
    from sbdd_benchmark.stage1.exploration import (
        build_fp_cache, training_similarity, circles_coverage
    )
    from sbdd_benchmark.io.loaders import load_training_set

    data_dir    = Path(data_dir)    if data_dir    else DEFAULT_DATA_DIR
    results_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    stage_dir   = results_dir / "stage1"
    stage_dir.mkdir(parents=True, exist_ok=True)

    if fp_cache_dir is None:
        fp_cache_dir = stage_dir / "fp_cache"
    fp_cache_dir = Path(fp_cache_dir)
    fp_cache_dir.mkdir(parents=True, exist_ok=True)

    models  = _resolve_models(models)
    targets = _resolve_targets(targets)

    print(f"\n[Stage 1] Models: {models}")
    print(f"[Stage 1] Targets: {targets}")

    # Pre-load training set fingerprints (cached)
    log.info("Loading CrossDocked2020 training set fingerprints...")
    train_df_cd = load_training_set("Pocket2Mol", data_dir=data_dir / "training_set")
    crossdock_fps = build_fp_cache(
        train_df_cd["smiles"].dropna().tolist(),
        cache_path=fp_cache_dir / "crossdocked_fps.npz",
        desc="CrossDocked FP",
    )
    log.info("CrossDocked: %d fingerprints", len(crossdock_fps))

    pocketxmol_fps = None
    if "PocketXMol" in models:
        log.info("Loading PocketXMol training set fingerprints...")
        train_df_pxm = load_training_set("PocketXMol", data_dir=data_dir / "training_set")
        pocketxmol_fps = build_fp_cache(
            train_df_pxm["smiles"].dropna().tolist(),
            cache_path=fp_cache_dir / "pocketxmol_fps.npz",
            desc="PocketXMol FP",
        )
        log.info("PocketXMol train: %d fingerprints", len(pocketxmol_fps))

    sim_rows = []
    circles_rows = []

    for model in models:
        print(f"\n  >> Stage 1: {model}")
        df = _load_parquet(model, data_dir)
        if df is None:
            continue

        # Filter to requested targets
        df = df[df["target"].isin(targets)]
        if df.empty:
            log.warning("%s: no molecules for requested targets", model)
            continue

        smiles_list = df["smiles"].tolist()
        ref_fps = pocketxmol_fps if (model == "PocketXMol" and pocketxmol_fps) else crossdock_fps

        # --- Max Tanimoto ---
        t0 = time.time()
        sim_result = training_similarity(smiles_list, ref_fps, model_name=model)
        log.info("%s: Tanimoto done in %.1fs", model, time.time() - t0)

        for i, (smi, tgt) in enumerate(zip(df["smiles"], df["target"])):
            sim_rows.append({
                "model":  model,
                "target": tgt,
                "smiles": smi,
                "max_tanimoto_train": sim_result["similarities"][i],
            })

        # --- #Circles ---
        t0 = time.time()
        circ_result = circles_coverage(
            smiles_list, threshold=circles_threshold,
            n_runs=n_circles_runs, seed=seed, desc=model,
        )
        log.info("%s: #Circles done in %.1fs", model, time.time() - t0)
        circles_rows.append({
            "model":       model,
            "circles_mean":   circ_result["mean"],
            "circles_std":    circ_result["std"],
            "circles_runs":   str(circ_result["run_counts"]),
            "n_sampled":      circ_result["n_sampled"],
            "efficiency":     circ_result["efficiency"],
            "threshold":      circles_threshold,
        })

    # Save outputs
    sim_df = pd.DataFrame(sim_rows)
    sim_df.to_csv(stage_dir / "similarity_scores.csv", index=False)

    # Summary per model
    if not sim_df.empty:
        rows = []
        for model, grp in sim_df.groupby("model"):
            vals = grp["max_tanimoto_train"]
            valid = vals[vals >= 0]
            rows.append({
                "model": model,
                "n_molecules": len(grp),
                "mean": valid.mean(),
                "median": valid.median(),
                "frac_above_0.2": (valid > 0.2).mean(),
                "frac_above_0.3": (valid > 0.3).mean(),
                "frac_above_0.5": (valid > 0.5).mean(),
            })
        pd.DataFrame(rows).to_csv(stage_dir / "similarity_summary.csv", index=False)

    circles_df = pd.DataFrame(circles_rows)
    circles_df.to_csv(stage_dir / "circles_coverage.csv", index=False)

    # Merge baselines if requested
    if compare_baselines:
        _merge_baselines(stage_dir, results_dir / "baselines" / "stage1", "similarity_summary.csv")
        _merge_baselines(stage_dir, results_dir / "baselines" / "stage1", "circles_coverage.csv")

    print(f"\n[Stage 1] Saved results to {stage_dir}")
    return stage_dir


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def run_stage2(
    models=None,
    targets=None,
    data_dir: str | Path = None,
    results_dir: str | Path = None,
    rascore_model_path: str | Path = None,
    compute_rascore: bool = False,
    compare_baselines: bool = False,
) -> Path:
    """Run Stage 2: Chemical Admissibility.

    Outputs (saved to results/stage2/):
        {model}_properties.csv   — per-molecule descriptors + REOS + SA
        reos_summary.csv         — per-model REOS pass rate
        wasserstein_heatmap.csv  — normalized Wasserstein distances vs patent

    Parameters
    ----------
    compute_rascore:
        If False (default), RAscore column is skipped (requires special env).
        If True, requires 'rascore' conda environment.
    """
    from sbdd_benchmark.stage2.filters import reos_filter
    from sbdd_benchmark.stage2.synthesizability import sa_score
    from sbdd_benchmark.stage2.properties import compute_descriptors, wasserstein_profile
    from sbdd_benchmark.io.loaders import load_benchmark

    data_dir    = Path(data_dir)    if data_dir    else DEFAULT_DATA_DIR
    results_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    stage_dir   = results_dir / "stage2"
    stage_dir.mkdir(parents=True, exist_ok=True)

    models  = _resolve_models(models)
    targets = _resolve_targets(targets)

    print(f"\n[Stage 2] Models: {models}")

    # Load patent reference for Wasserstein and compute its descriptors
    patent_smiles_frames = []
    for tgt in targets:
        bm_df = _load_benchmark_for_target(tgt, data_dir)
        if bm_df is not None:
            patent_smiles_frames.append(bm_df)
    if patent_smiles_frames:
        patent_raw = pd.concat(patent_smiles_frames, ignore_index=True)
        log.info("Computing descriptors for %d patent molecules...", len(patent_raw))
        patent_df = compute_descriptors(patent_raw["smiles"].dropna().tolist())
    else:
        patent_df = pd.DataFrame()

    model_prop_dfs = {}
    reos_rows = []

    for model in models:
        print(f"\n  >> Stage 2: {model}")
        prop_cache = stage_dir / f"{model}_properties.csv"

        if prop_cache.exists():
            log.info("%s: loading cached properties from %s", model, prop_cache)
            model_df = pd.read_csv(prop_cache)
        else:
            df = _load_parquet(model, data_dir)
            if df is None:
                continue
            df = df[df["target"].isin(targets)]
            if df.empty:
                log.warning("%s: no molecules for requested targets", model)
                continue

            smiles_list = df["smiles"].tolist()

            # Descriptors
            desc_df = compute_descriptors(smiles_list)
            desc_df["model"]  = model
            desc_df["target"] = df["target"].values

            # SA Score
            sa_results = sa_score(smiles_list)
            desc_df["SA_Score"] = [r["sa_score"] for r in sa_results]

            # REOS
            reos_results = reos_filter(smiles_list)
            desc_df["REOS_Pass"]      = [r["reos_pass"]      for r in reos_results]
            desc_df["REOS_Violation"] = [r["reos_violation"]  for r in reos_results]

            # RAscore (optional)
            if compute_rascore:
                from sbdd_benchmark.stage2.synthesizability import rascore as compute_rascore_fn
                ra_results = compute_rascore_fn(smiles_list, model_path=rascore_model_path)
                desc_df["RAscore"] = [r["rascore"] for r in ra_results]

            model_df = desc_df
            model_df.to_csv(prop_cache, index=False)
            log.info("%s: saved %d molecules to %s", model, len(model_df), prop_cache)

        model_prop_dfs[model] = model_df

        # REOS summary row
        pass_mask = model_df["REOS_Pass"].astype(str).str.lower().isin(["true", "1"])
        reos_rows.append({
            "model":        model,
            "n_total":      len(model_df),
            "n_reos_pass":  int(pass_mask.sum()),
            "reos_pass_pct": float(pass_mask.mean() * 100),
        })

    # REOS summary
    reos_df = pd.DataFrame(reos_rows)
    reos_df.to_csv(stage_dir / "reos_summary.csv", index=False)

    # Wasserstein heatmap
    if model_prop_dfs and not patent_df.empty:
        raw_wd, norm_wd = wasserstein_profile(model_prop_dfs, patent_df)
        raw_wd.to_csv(stage_dir / "wasserstein_raw.csv")
        norm_wd.to_csv(stage_dir / "wasserstein_heatmap.csv")
        log.info("Wasserstein heatmap saved")

    if compare_baselines:
        _merge_baselines(stage_dir, results_dir / "baselines" / "stage2", "reos_summary.csv")
        _merge_baselines(stage_dir, results_dir / "baselines" / "stage2", "wasserstein_heatmap.csv")

    print(f"\n[Stage 2] Saved results to {stage_dir}")
    return stage_dir


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------

def run_stage3(
    models=None,
    targets=None,
    data_dir: str | Path = None,
    results_dir: str | Path = None,
    compare_baselines: bool = False,
) -> Path:
    """Run Stage 3: Patent Molecule Recovery.

    Outputs (saved to results/stage3/):
        {model}_recovery.csv     — per-target recovery metrics
        recovery_summary.csv     — cross-target mean±SEM, target coverage
    """
    from sbdd_benchmark.stage3.recovery import evaluate_target, recovery_summary

    data_dir    = Path(data_dir)    if data_dir    else DEFAULT_DATA_DIR
    results_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    stage_dir   = results_dir / "stage3"
    stage_dir.mkdir(parents=True, exist_ok=True)

    models  = _resolve_models(models)
    targets = _resolve_targets(targets)

    print(f"\n[Stage 3] Models: {models}")
    print(f"[Stage 3] Targets: {targets}")

    summary_rows = []

    for model in models:
        print(f"\n  >> Stage 3: {model}")
        df = _load_parquet(model, data_dir)
        if df is None:
            continue

        per_target = []
        for target in tqdm(targets, desc=f"  {model}", leave=False):
            bm_df = _load_benchmark_for_target(target, data_dir)
            if bm_df is None:
                continue
            true_smiles = bm_df["smiles"].dropna().tolist()
            gen_smiles = df[df["target"] == target]["smiles"].tolist()
            if not gen_smiles:
                log.warning("%s / %s: no generated molecules", model, target)
                continue
            result = evaluate_target(target, gen_smiles, true_smiles)
            per_target.append(result)

        if per_target:
            tgt_df = pd.DataFrame(per_target)
            tgt_df["model"] = model
            tgt_df.to_csv(stage_dir / f"{model}_recovery.csv", index=False)

            summ = recovery_summary(per_target, model_name=model)
            summary_rows.append(summ)
            log.info(
                "%s: generic scaffold %.2f%% (mean, %d targets)",
                model, summ.get("generic_mean_%", 0), len(per_target)
            )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(stage_dir / "recovery_summary.csv", index=False)

    if compare_baselines:
        _merge_baselines(stage_dir, results_dir / "baselines" / "stage3", "recovery_summary.csv")

    print(f"\n[Stage 3] Saved results to {stage_dir}")
    return stage_dir


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_all(
    models=None,
    targets=None,
    data_dir: str | Path = None,
    results_dir: str | Path = None,
    compute_rascore: bool = False,
    compare_baselines: bool = False,
    seed: int = 42,
) -> dict[str, Path]:
    """Run all three stages sequentially.

    Returns a dict mapping stage name to output directory.
    """
    t0 = time.time()
    dirs = {
        "stage1": run_stage1(models, targets, data_dir, results_dir,
                             seed=seed, compare_baselines=compare_baselines),
        "stage2": run_stage2(models, targets, data_dir, results_dir,
                             compute_rascore=compute_rascore,
                             compare_baselines=compare_baselines),
        "stage3": run_stage3(models, targets, data_dir, results_dir,
                             compare_baselines=compare_baselines),
    }
    print(f"\n[run_all] Completed in {time.time()-t0:.1f}s")
    return dirs


# ---------------------------------------------------------------------------
# Baseline merge helper
# ---------------------------------------------------------------------------

def _merge_baselines(current_dir: Path, baseline_dir: Path, filename: str) -> None:
    """Append baseline rows to a current results CSV (deduplicating by model)."""
    current_path  = current_dir  / filename
    baseline_path = baseline_dir / filename
    if not baseline_path.exists():
        return
    if not current_path.exists():
        return

    curr_df = pd.read_csv(current_path)
    base_df = pd.read_csv(baseline_path)

    # Only keep baseline rows for models NOT in current results
    if "model" in curr_df.columns and "model" in base_df.columns:
        existing_models = set(curr_df["model"].unique())
        base_df = base_df[~base_df["model"].isin(existing_models)]

    merged = pd.concat([curr_df, base_df], ignore_index=True)
    merged.to_csv(current_path, index=False)
    log.info("Merged baseline rows into %s", current_path)
