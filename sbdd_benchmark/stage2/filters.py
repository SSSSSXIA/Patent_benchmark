"""
Stage 2 — Chemical Admissibility: molecule filters.

reos_filter(smiles_list)
    Apply the Dundee REOS filter (via useful_rdkit_utils).
    Returns per-molecule pass/fail and violation names.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


def reos_filter(smiles_list: list[str], rule_set: str = "Dundee") -> list[dict]:
    """Apply the REOS filter to a list of SMILES.

    Parameters
    ----------
    smiles_list:
        SMILES strings to evaluate.
    rule_set:
        REOS rule set name. Paper uses "Dundee" (105 rules).

    Returns
    -------
    List of dicts, one per input SMILES:
        smiles          : str
        reos_pass       : bool
        reos_violation  : str or None — name of violated rule if failed
    """
    try:
        from useful_rdkit_utils import REOS
    except ImportError:
        raise ImportError(
            "useful_rdkit_utils is required for REOS filtering.\n"
            "Install: pip install useful-rdkit-utils"
        )

    reos = REOS(active_rules=[rule_set])
    results = []
    for smi in smiles_list:
        try:
            result = reos.process_smiles(smi)
            passed = result == ("ok", "ok")
            violation = None if passed else str(result)
            results.append({"smiles": smi, "reos_pass": passed, "reos_violation": violation})
        except Exception as exc:
            log.debug("REOS error on '%s': %s", smi[:40], exc)
            results.append({"smiles": smi, "reos_pass": False, "reos_violation": f"error: {exc}"})
    return results


def pains_filter(smiles_list: list[str]) -> list[dict]:
    """Apply the PAINS filter (via RDKit FilterCatalog).

    Returns
    -------
    List of dicts:
        smiles       : str
        pains_pass   : bool   (True = no PAINS alert)
        pains_alerts : list[str] — matched PAINS alert names
    """
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    from rdkit import Chem

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)

    results = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results.append({"smiles": smi, "pains_pass": False, "pains_alerts": ["invalid_smiles"]})
                continue
            entries = catalog.GetMatches(mol)
            alerts = [e.GetDescription() for e in entries]
            results.append({"smiles": smi, "pains_pass": len(alerts) == 0, "pains_alerts": alerts})
        except Exception as exc:
            log.debug("PAINS error on '%s': %s", smi[:40], exc)
            results.append({"smiles": smi, "pains_pass": False, "pains_alerts": [f"error: {exc}"]})
    return results
