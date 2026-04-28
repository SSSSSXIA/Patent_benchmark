"""
Stage 3 — Patent Molecule Recovery.

Functions
---------
exact_recovery(generated, true_molecules)
    Canonical SMILES match (stereochemistry removed).

bm_recovery(generated, true_molecules)
    Bemis-Murcko scaffold match.

generic_recovery(generated, true_molecules)
    Generic scaffold match (all heteroatoms→C, all bonds→single).

recovery_summary(results_per_target)
    Aggregate per-target recovery results into cross-target statistics:
    mean ± SEM, target coverage (% targets with >0%), good coverage (>5%).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scaffold utilities
# ---------------------------------------------------------------------------

def _get_rdkit():
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    return Chem, MurckoScaffold


def canonicalize(smiles: str, remove_stereo: bool = True) -> Optional[str]:
    """Return canonical RDKit SMILES, optionally stripping stereochemistry."""
    Chem, _ = _get_rdkit()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if remove_stereo:
            Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def get_bm_scaffold(smiles: str) -> Optional[str]:
    """Bemis-Murcko scaffold SMILES, or None on failure."""
    Chem, MurckoScaffold = _get_rdkit()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return None


def get_generic_scaffold(smiles: str) -> Optional[str]:
    """Generic (topology-only) scaffold SMILES.

    Applies MurckoScaffold.MakeScaffoldGeneric() which replaces all
    heteroatoms with C and sets all bonds to single — the most permissive
    scaffold comparison used in the paper.
    """
    Chem, MurckoScaffold = _get_rdkit()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        bm = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(bm)
        return Chem.MolToSmiles(generic, canonical=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-target evaluation
# ---------------------------------------------------------------------------

def evaluate_target(
    target_name: str,
    generated_smiles: list[str],
    true_smiles: list[str],
) -> dict:
    """Evaluate all three recovery metrics for one model × target pair.

    Parameters
    ----------
    target_name:
        Target identifier (for output labelling only).
    generated_smiles:
        All generated SMILES for this target from this model.
    true_smiles:
        Patent active molecule SMILES for this target.

    Returns
    -------
    dict with keys:
        target
        num_true_molecules
        num_true_scaffolds_bm
        num_true_scaffolds_generic
        num_generated_molecules
        num_unique_generated
        exact_recovery_count
        exact_recovery_ratio
        bm_scaffold_recovery_count
        bm_scaffold_recovery_ratio
        bm_unique_scaffolds_recovered
        generic_scaffold_recovery_count
        generic_scaffold_recovery_ratio
        generic_unique_scaffolds_recovered
    """
    from tqdm import tqdm

    # Canonicalize true molecules (remove stereo for matching)
    true_canon = [c for s in true_smiles if (c := canonicalize(s, remove_stereo=True)) is not None]
    true_set = set(true_canon)

    # True scaffold sets
    true_bm: dict[str, list[str]] = {}
    true_generic: dict[str, list[str]] = {}
    for smi in true_canon:
        bm = get_bm_scaffold(smi)
        if bm:
            true_bm.setdefault(bm, []).append(smi)
        gen = get_generic_scaffold(smi)
        if gen:
            true_generic.setdefault(gen, []).append(smi)

    # Canonicalize generated molecules (remove stereo for matching)
    gen_canon_list = [c for s in generated_smiles if (c := canonicalize(s, remove_stereo=True)) is not None]
    gen_set = set(gen_canon_list)

    # Generated scaffold sets
    gen_bm: dict[str, list[str]] = {}
    gen_generic: dict[str, list[str]] = {}
    for smi in gen_canon_list:
        bm = get_bm_scaffold(smi)
        if bm:
            gen_bm.setdefault(bm, []).append(smi)
        gen = get_generic_scaffold(smi)
        if gen:
            gen_generic.setdefault(gen, []).append(smi)

    n_true = len(true_canon)

    # ---- Exact recovery ----
    exact_recovered = true_set & gen_set
    exact_count = len(exact_recovered)
    exact_ratio = exact_count / n_true if n_true else 0.0

    # ---- BM scaffold recovery ----
    recovered_bm_scaffolds = set(true_bm) & set(gen_bm)
    true_mols_with_bm = sum(len(true_bm[s]) for s in recovered_bm_scaffolds)
    bm_count = true_mols_with_bm
    bm_ratio = bm_count / n_true if n_true else 0.0

    # ---- Generic scaffold recovery ----
    recovered_gen_scaffolds = set(true_generic) & set(gen_generic)
    true_mols_with_gen = sum(len(true_generic[s]) for s in recovered_gen_scaffolds)
    gen_count = true_mols_with_gen
    gen_ratio = gen_count / n_true if n_true else 0.0

    return {
        "target":                          target_name,
        "num_true_molecules":              n_true,
        "num_true_scaffolds_bm":           len(true_bm),
        "num_true_scaffolds_generic":      len(true_generic),
        "num_generated_molecules":         len(gen_canon_list),
        "num_unique_generated":            len(gen_set),
        "exact_recovery_count":            exact_count,
        "exact_recovery_ratio":            exact_ratio,
        "bm_scaffold_recovery_count":      bm_count,
        "bm_scaffold_recovery_ratio":      bm_ratio,
        "bm_unique_scaffolds_recovered":   len(recovered_bm_scaffolds),
        "generic_scaffold_recovery_count": gen_count,
        "generic_scaffold_recovery_ratio": gen_ratio,
        "generic_unique_scaffolds_recovered": len(recovered_gen_scaffolds),
    }


# ---------------------------------------------------------------------------
# Convenience wrappers (single-metric)
# ---------------------------------------------------------------------------

def exact_recovery(generated: list[str], true_molecules: list[str]) -> dict:
    """Return exact recovery stats (no stereo). Wrapper around evaluate_target()."""
    result = evaluate_target("", generated, true_molecules)
    return {
        "count": result["exact_recovery_count"],
        "ratio": result["exact_recovery_ratio"],
        "n_true": result["num_true_molecules"],
    }


def bm_recovery(generated: list[str], true_molecules: list[str]) -> dict:
    """Return BM scaffold recovery stats. Wrapper around evaluate_target()."""
    result = evaluate_target("", generated, true_molecules)
    return {
        "count":              result["bm_scaffold_recovery_count"],
        "ratio":              result["bm_scaffold_recovery_ratio"],
        "unique_scaffolds":   result["bm_unique_scaffolds_recovered"],
        "n_true":             result["num_true_molecules"],
    }


def generic_recovery(generated: list[str], true_molecules: list[str]) -> dict:
    """Return generic scaffold recovery stats. Wrapper around evaluate_target()."""
    result = evaluate_target("", generated, true_molecules)
    return {
        "count":              result["generic_scaffold_recovery_count"],
        "ratio":              result["generic_scaffold_recovery_ratio"],
        "unique_scaffolds":   result["generic_unique_scaffolds_recovered"],
        "n_true":             result["num_true_molecules"],
    }


# ---------------------------------------------------------------------------
# Cross-target summary
# ---------------------------------------------------------------------------

def recovery_summary(
    per_target_results: list[dict],
    model_name: str = "",
) -> dict:
    """Aggregate per-target recovery dicts into cross-target statistics.

    Parameters
    ----------
    per_target_results:
        List of dicts returned by evaluate_target() for each target.
    model_name:
        For labelling the summary row.

    Returns
    -------
    dict with:
        model
        n_targets
        exact_mean / exact_sem
        bm_mean / bm_sem
        generic_mean / generic_sem
        exact_target_coverage    (% targets with >0% exact recovery)
        bm_target_coverage       (% with >0% BM)
        generic_target_coverage  (% with >0% generic)
        bm_good_coverage         (% with >5% BM)
        generic_good_coverage    (% with >5% generic)
    """
    if not per_target_results:
        return {}

    df = pd.DataFrame(per_target_results)
    n = len(df)

    def _mean_sem(col: str) -> tuple[float, float]:
        vals = df[col].values * 100  # convert to %
        return float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals)))

    def _coverage(col: str, threshold: float = 0.0) -> float:
        return float((df[col] > threshold).sum() / n * 100)

    exact_m, exact_s = _mean_sem("exact_recovery_ratio")
    bm_m, bm_s = _mean_sem("bm_scaffold_recovery_ratio")
    gen_m, gen_s = _mean_sem("generic_scaffold_recovery_ratio")

    return {
        "model":                    model_name,
        "n_targets":                n,
        "exact_mean_%":             round(exact_m, 4),
        "exact_sem_%":              round(exact_s, 4),
        "bm_mean_%":                round(bm_m, 4),
        "bm_sem_%":                 round(bm_s, 4),
        "generic_mean_%":           round(gen_m, 4),
        "generic_sem_%":            round(gen_s, 4),
        "exact_target_coverage_%":  round(_coverage("exact_recovery_ratio",   0.0), 1),
        "bm_target_coverage_%":     round(_coverage("bm_scaffold_recovery_ratio", 0.0), 1),
        "generic_target_coverage_%": round(_coverage("generic_scaffold_recovery_ratio", 0.0), 1),
        "bm_good_coverage_%":       round(_coverage("bm_scaffold_recovery_ratio", 0.05), 1),
        "generic_good_coverage_%":  round(_coverage("generic_scaffold_recovery_ratio", 0.05), 1),
    }
