"""
Stage 1 — Chemical Space Exploration.

Metrics
-------
training_similarity(smiles_list, train_fps)
    Max Tanimoto similarity of each generated molecule to the training set.
    Returns a dict with the similarity array and fraction above thresholds.

circles_coverage(smiles_list, threshold, n_runs, seed)
    #Circles chemical-space coverage at threshold t.
    Greedy algorithm averaged over n_runs random permutations.
    Returns mean ± std as specified in the manuscript.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------

FP_RADIUS = 2
FP_NBITS = 2048


def _get_rdkit():
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    return Chem, DataStructs, AllChem, rdFingerprintGenerator


def mol_to_fp_bitvect(smiles: str):
    """Return an RDKit ExplicitBitVect ECFP4, or None on failure."""
    Chem, _, AllChem, _ = _get_rdkit()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
    except Exception:
        return None


def mol_to_fp_numpy(smiles: str) -> Optional[np.ndarray]:
    """Return an ECFP4 as uint8 numpy array (for fast matrix ops), or None."""
    Chem, _, _, rdFPGen = _get_rdkit()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        gen = rdFPGen.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)
        return gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
    except Exception:
        return None


def build_fp_cache(
    smiles_list: list[str],
    cache_path: Optional[str | Path] = None,
    desc: str = "FP",
) -> list:
    """Compute ECFP4 BitVect fingerprints with optional npz disk cache.

    Parameters
    ----------
    smiles_list:
        List of SMILES strings.
    cache_path:
        If provided, load from .npz if it exists; otherwise compute and save.
    desc:
        Label for tqdm progress bar.

    Returns
    -------
    List of RDKit ExplicitBitVect fingerprints (invalid SMILES are skipped).
    """
    from tqdm import tqdm
    from rdkit import DataStructs

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            log.info("Loading fingerprint cache from %s", cache_path)
            npz = np.load(cache_path)
            matrix = npz["fps"]
            fps = []
            for row in matrix:
                bv = DataStructs.ExplicitBitVect(FP_NBITS)
                bv.SetBitsFromList(np.where(row)[0].tolist())
                fps.append(bv)
            return fps

    fps = []
    for smi in tqdm(smiles_list, desc=desc, leave=False):
        fp = mol_to_fp_bitvect(smi)
        if fp is not None:
            fps.append(fp)

    if cache_path is not None and fps:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.zeros((len(fps), FP_NBITS), dtype=np.uint8)
        for i, fp in enumerate(fps):
            for bit in fp.GetOnBits():
                matrix[i, bit] = 1
        np.savez_compressed(cache_path, fps=matrix)
        log.info("Saved fingerprint cache → %s (%d fps)", cache_path, len(fps))

    return fps


# ---------------------------------------------------------------------------
# Max Tanimoto similarity to training set
# ---------------------------------------------------------------------------

def training_similarity(
    smiles_list: list[str],
    train_fps: list,
    model_name: str = "",
) -> dict:
    """Compute max Tanimoto similarity of each molecule to the training set.

    Parameters
    ----------
    smiles_list:
        Generated (valid) SMILES to evaluate.
    train_fps:
        List of RDKit BitVect fingerprints for the training set.
    model_name:
        Used only for logging.

    Returns
    -------
    dict with keys:
        similarities : np.ndarray of max Tanimoto values (-1 = invalid SMILES)
        n_invalid    : int
        frac_above   : dict {0.2: float, 0.3: float, 0.5: float}
        mean         : float
        median       : float
    """
    from rdkit import DataStructs
    from tqdm import tqdm

    sims = []
    n_invalid = 0
    for smi in tqdm(smiles_list, desc=f"Tanimoto [{model_name}]", leave=False):
        fp = mol_to_fp_bitvect(smi)
        if fp is None:
            sims.append(-1.0)
            n_invalid += 1
        else:
            max_sim = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
            sims.append(max_sim)

    arr = np.array(sims)
    valid = arr[arr >= 0]
    frac_above = {
        t: float(np.mean(valid > t)) for t in (0.2, 0.3, 0.5)
    }
    return {
        "similarities": arr,
        "n_invalid":    n_invalid,
        "frac_above":   frac_above,
        "mean":         float(np.mean(valid)) if len(valid) else float("nan"),
        "median":       float(np.median(valid)) if len(valid) else float("nan"),
    }


# ---------------------------------------------------------------------------
# #Circles coverage
# ---------------------------------------------------------------------------

def _greedy_circles_numpy(fp_matrix: np.ndarray, threshold: float) -> int:
    """Greedy #Circles on a uint8 fingerprint matrix (rows = molecules).

    Uses numpy bitwise ops for speed. Assumes rows are already in a random
    order — call with np.random.permutation applied beforehand.

    Returns the number of selected 'circle centres'.
    """
    n, n_bits = fp_matrix.shape
    sel_fps = np.zeros((max(n // 10, 100), n_bits), dtype=np.uint8)
    n_sel = 0

    for i in range(n):
        fp = fp_matrix[i]
        if n_sel == 0:
            sel_fps[0] = fp
            n_sel = 1
        else:
            batch = sel_fps[:n_sel]
            intersection = np.sum(fp & batch, axis=1)
            union = np.sum(fp | batch, axis=1)
            distances = 1.0 - intersection / (union + 1e-10)  # d = 1 - Tanimoto
            if np.all(distances > threshold):  # t=0.75 is a distance threshold
                if n_sel >= len(sel_fps):
                    sel_fps = np.vstack([sel_fps, np.zeros_like(sel_fps)])
                sel_fps[n_sel] = fp
                n_sel += 1

    return n_sel


def circles_coverage(
    smiles_list: list[str],
    threshold: float = 0.75,
    n_runs: int = 3,
    seed: int = 42,
    max_pocketxmol_sample: int = 160_000,
    desc: str = "",
) -> dict:
    """Compute #Circles chemical-space coverage.

    Algorithm
    ---------
    For each of *n_runs* random permutations of the input molecules:
        1. Compute ECFP4 fingerprints (uint8 numpy arrays).
        2. Run greedy selection: add a molecule to the set iff its Tanimoto
           similarity to ALL already-selected molecules is ≤ threshold.
    Report mean ± std of the circle count across runs.

    Coverage efficiency = mean_circles / n_valid_molecules * 1000.

    Parameters
    ----------
    smiles_list:
        Valid SMILES (already canonicalized).
    threshold:
        Tanimoto distance threshold t (paper uses 0.75; d = 1 - Tanimoto).
        A molecule is added to the circle set only if its distance to every
        existing circle centre exceeds t.
    n_runs:
        Number of random permutations (paper uses 3).
    seed:
        Base random seed; each run uses seed + run_index.
    max_pocketxmol_sample:
        If len(smiles_list) > this, subsample (matching notebook behaviour
        for PocketXMol which has unlimited generated molecules).
    desc:
        Label for progress display.

    Returns
    -------
    dict with keys:
        mean          : float — mean #Circles over n_runs
        std           : float — std dev over n_runs
        run_counts    : list[int] — per-run circle counts
        n_sampled     : int — molecules used (after subsampling)
        efficiency    : float — mean / n_sampled * 1000
    """
    from tqdm import tqdm

    rng = np.random.default_rng(seed)

    # Build fingerprint matrix
    fps_list = []
    for smi in tqdm(smiles_list, desc=f"FP [{desc}]", leave=False):
        fp = mol_to_fp_numpy(smi)
        if fp is not None:
            fps_list.append(fp)

    if not fps_list:
        return {"mean": 0.0, "std": 0.0, "run_counts": [], "n_sampled": 0, "efficiency": 0.0}

    # Subsample if needed (mirrors notebook behaviour for PocketXMol)
    if len(fps_list) > max_pocketxmol_sample:
        idx = rng.choice(len(fps_list), size=max_pocketxmol_sample, replace=False)
        fps_list = [fps_list[i] for i in idx]
        log.info("#Circles: subsampled to %d molecules", max_pocketxmol_sample)

    fp_matrix = np.vstack(fps_list)
    n_sampled = len(fp_matrix)

    run_counts = []
    for run in range(n_runs):
        run_seed = seed + run
        perm = np.random.default_rng(run_seed).permutation(n_sampled)
        shuffled = fp_matrix[perm]
        count = _greedy_circles_numpy(shuffled, threshold)
        run_counts.append(count)
        log.debug("  Run %d: #Circles = %d", run + 1, count)

    mean_c = float(np.mean(run_counts))
    std_c = float(np.std(run_counts))
    efficiency = mean_c / n_sampled * 1000 if n_sampled else 0.0

    return {
        "mean":       mean_c,
        "std":        std_c,
        "run_counts": run_counts,
        "n_sampled":  n_sampled,
        "efficiency": efficiency,
    }
