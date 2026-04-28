"""
Stage 2 — Chemical Admissibility: physicochemical descriptors.

compute_descriptors(smiles_list)
    Compute the 9 descriptors used in the paper:
    MW, Fsp3, logP, N_AliR, N_AroR, N_ChiA, N_HetA, N_RotB, N_BriA

wasserstein_profile(model_desc_df, patent_desc_df)
    Compute Wasserstein distance of each descriptor distribution vs patent,
    then normalize per descriptor across models (exact paper figure logic).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Descriptors in the order used by the paper heatmap
DESCRIPTOR_COLS = ["MW", "logP", "Fsp3", "N_AliR", "N_AroR", "N_ChiA", "N_HetA", "N_RotB", "N_BriA"]


def _get_rdkit():
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    return Chem, Descriptors, rdMolDescriptors


def compute_descriptors(smiles_list: list[str]) -> pd.DataFrame:
    """Compute 9 physicochemical descriptors for each SMILES.

    Replicates `calculate_all_properties()` from evaluation_level2.ipynb exactly.
    Column names match all_models_properties.csv: N_AliR, N_AroR, N_ChiA,
    N_HetA, N_RotB, N_BriA.

    Parameters
    ----------
    smiles_list:
        Canonical SMILES strings (should already be valid).

    Returns
    -------
    DataFrame with columns:
        smiles, MW, logP, Fsp3, N_AliR, N_AroR, N_ChiA, N_HetA, N_RotB, N_BriA
    Invalid SMILES → all descriptor columns = NaN (row still present for index alignment).
    """
    from tqdm import tqdm

    Chem, Descriptors, rdMolDescriptors = _get_rdkit()

    records = []
    for smi in tqdm(smiles_list, desc="Descriptors", leave=False):
        row: dict = {"smiles": smi}
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("invalid SMILES")

            ring_info = mol.GetRingInfo()
            n_rings = ring_info.NumRings()
            n_aromatic = len([
                r for r in ring_info.AtomRings()
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
            ])
            n_aliphatic = n_rings - n_aromatic

            row.update({
                "MW":     Descriptors.MolWt(mol),
                "logP":   Descriptors.MolLogP(mol),
                "Fsp3":   rdMolDescriptors.CalcFractionCSP3(mol),
                "N_AliR": n_aliphatic,
                "N_AroR": n_aromatic,
                "N_ChiA": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
                "N_HetA": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6)),
                "N_RotB": Descriptors.NumRotatableBonds(mol),
                "N_BriA": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            })
        except Exception:
            row.update({col: np.nan for col in DESCRIPTOR_COLS})
        records.append(row)

    return pd.DataFrame(records)


def wasserstein_profile(
    model_results: dict[str, pd.DataFrame],
    patent_df: pd.DataFrame,
    desc_cols: list[str] = DESCRIPTOR_COLS,
    reos_pass_col: Optional[str] = "REOS_Pass",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Wasserstein distances and produce the normalized heatmap matrix.

    Replicates the exact normalization from evaluation_level2.ipynb Figure 4D:
        1. For each model, compute WD(model_dist, patent_dist) per descriptor.
           Only REOS-passing molecules are used (if reos_pass_col is present).
        2. Patent row = 0.0 by definition.
        3. Normalize per descriptor: divide by max WD across all non-Patent models.

    Parameters
    ----------
    model_results:
        Dict mapping model_name → DataFrame with at least the descriptor columns
        (and optionally a 'REOS_Pass' boolean column).
    patent_df:
        Patent reference DataFrame with the same descriptor columns.
    desc_cols:
        Descriptor columns to evaluate (default: 9 paper descriptors).
    reos_pass_col:
        If present in a model DataFrame, use only passing molecules.
        Set to None to disable filtering.

    Returns
    -------
    raw_wd_df : DataFrame (models × descriptors) of raw Wasserstein distances
    norm_wd_df : DataFrame (models × descriptors) of normalized distances [0–1]
    """
    from scipy.stats import wasserstein_distance

    rows_raw = {}
    for model_name, df in model_results.items():
        if reos_pass_col and reos_pass_col in df.columns:
            df = df[df[reos_pass_col].astype(bool)]
        row = {}
        for col in desc_cols:
            if col not in df.columns or col not in patent_df.columns:
                row[col] = np.nan
                continue
            mod_vals = df[col].dropna().values
            pat_vals = patent_df[col].dropna().values
            if len(mod_vals) > 10 and len(pat_vals) > 10:
                row[col] = wasserstein_distance(mod_vals, pat_vals)
            else:
                row[col] = np.nan
        rows_raw[model_name] = row

    raw_wd_df = pd.DataFrame(rows_raw).T  # shape: models × descriptors
    raw_wd_df.columns = desc_cols

    # Normalize per descriptor: divide by max across non-Patent models
    norm_wd_df = raw_wd_df.copy()
    for col in desc_cols:
        max_val = raw_wd_df[col].max()
        if pd.notna(max_val) and max_val > 0:
            norm_wd_df[col] = raw_wd_df[col] / max_val
        else:
            norm_wd_df[col] = 0.0

    return raw_wd_df, norm_wd_df
