"""
Stage 2 — Chemical Admissibility: synthesizability scores.

sa_score(smiles_list)
    Synthetic Accessibility score (1=easy, 10=hard) via RDKit sascorer.

rascore(smiles_list, model_path)
    Retrosynthetic accessibility score (0–1, higher=more tractable)
    via the RAscore DNN model. Requires the RAscore package and model weights.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Default model path relative to repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RASCORE_MODEL = (
    _REPO_ROOT / "RAscore" / "RAscore" / "models" / "models"
    / "DNN_chembl_fcfp_counts" / "model.h5"
)
# Also check _backup/
_BACKUP_RASCORE_MODEL = (
    _REPO_ROOT / "_backup" / "RAscore" / "RAscore" / "models" / "models"
    / "DNN_chembl_fcfp_counts" / "model.h5"
)


def sa_score(smiles_list: list[str]) -> list[dict]:
    """Compute SA Score for each SMILES.

    Parameters
    ----------
    smiles_list:
        Canonical SMILES strings.

    Returns
    -------
    List of dicts:
        smiles   : str
        sa_score : float (1–10) or None if SMILES is invalid
    """
    from rdkit import Chem, RDLogger
    from rdkit.Contrib.SA_Score import sascorer
    RDLogger.DisableLog("rdApp.*")

    results = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results.append({"smiles": smi, "sa_score": None})
            else:
                score = sascorer.calculateScore(mol)
                results.append({"smiles": smi, "sa_score": float(score)})
        except Exception as exc:
            log.debug("SA Score error on '%s': %s", smi[:40], exc)
            results.append({"smiles": smi, "sa_score": None})
    return results


def rascore(
    smiles_list: list[str],
    model_path: Optional[str | Path] = None,
    batch_size: int = 256,
) -> list[dict]:
    """Compute RAscore (DNN model) for each SMILES.

    Replicates the exact calling pattern from RAscore/rascore_usage.ipynb:
        from RAscore import RAscore_NN
        scorer = RAscore_NN.RAScorerNN(model_path=...)
        score = scorer.predict(smiles)

    Parameters
    ----------
    smiles_list:
        Canonical SMILES strings.
    model_path:
        Path to the DNN .h5 model weights. If None, uses the default path
        relative to the repo root (or _backup/ fallback).
    batch_size:
        Unused here (RAscore processes one at a time), kept for API symmetry.

    Returns
    -------
    List of dicts:
        smiles   : str
        rascore  : float (0–1) or -1 for invalid SMILES
    """
    from tqdm import tqdm

    # Resolve model path
    if model_path is None:
        for candidate in (_DEFAULT_RASCORE_MODEL, _BACKUP_RASCORE_MODEL):
            if candidate.exists():
                model_path = candidate
                break
        if model_path is None:
            raise FileNotFoundError(
                "RAscore model not found. Expected at:\n"
                f"  {_DEFAULT_RASCORE_MODEL}\n"
                "or after restructuring:\n"
                f"  {_BACKUP_RASCORE_MODEL}\n"
                "Provide model_path= explicitly if stored elsewhere."
            )

    try:
        from RAscore import RAscore_NN
    except ImportError:
        raise ImportError(
            "RAscore package not installed in current environment.\n"
            "Activate the 'rascore' conda env: conda activate rascore"
        )

    log.info("Loading RAscore DNN model from %s", model_path)
    scorer = RAscore_NN.RAScorerNN(model_path=str(model_path))

    results = []
    for smi in tqdm(smiles_list, desc="RAscore", leave=False):
        try:
            if not smi or not isinstance(smi, str):
                results.append({"smiles": smi, "rascore": -1})
                continue
            score = scorer.predict(smi)
            results.append({"smiles": smi, "rascore": float(score)})
        except Exception as exc:
            log.debug("RAscore error on '%s': %s", smi[:40], exc)
            results.append({"smiles": smi, "rascore": -1})
    return results
