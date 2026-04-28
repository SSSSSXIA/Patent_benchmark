"""
Per-model molecule readers and unified parquet I/O.

Unified parquet schema (data/generated/unified/{model}.parquet):
    smiles      : str   — canonical RDKit SMILES (sanitized)
    target      : str   — target name, e.g. "JAK1"
    model       : str   — model name, e.g. "PocketXMol"
    is_valid    : bool  — passed RDKit sanitization
    source_file : str   — original file path for traceability
"""

from __future__ import annotations

import glob
import logging
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_MOLECULES_PER_TARGET = 8_000  # cap per target for all models

BASE_DIR = Path(__file__).resolve().parents[2]  # repo root

MODEL_RAW_ROOTS = {
    "Pocket2Mol": BASE_DIR / "Pocket2Mol" / "outputs_0",
    "MolCRAFT":   BASE_DIR / "MolCRAFT"   / "benchmark",
    "TamGen":     BASE_DIR / "TamGen"      / "customized_example",
    "PocketFlow": BASE_DIR / "PocketFlow"  / "gen_results_0",
    "PocketXMol": BASE_DIR / "PocketXMol"  / "outputs_0",
    "ResGen":     BASE_DIR / "ResGen"      / "examples_0",
}

# ---------------------------------------------------------------------------
# Low-level SMILES extraction helpers
# ---------------------------------------------------------------------------

def _try_import_rdkit():
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        return Chem
    except ImportError:
        raise ImportError("RDKit is required: conda install -c conda-forge rdkit")


def _sdf_smiles(sdf_path: str, max_count: Optional[int] = None) -> Iterator[tuple[str, str]]:
    """Yield (canonical_smiles, source_file) from an SDF file.

    Stops after *max_count* unique SMILES have been yielded (set None for unlimited).
    Logs but does not raise on per-molecule sanitization failures.
    """
    Chem = _try_import_rdkit()
    seen: set[str] = set()
    try:
        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=True, strictParsing=False)
        for mol in supplier:
            if mol is None:
                continue
            try:
                smi = Chem.MolToSmiles(mol, canonical=True)
                if smi and smi not in seen:
                    seen.add(smi)
                    yield smi, str(sdf_path)
                    if max_count and len(seen) >= max_count:
                        return
            except Exception:
                pass
    except Exception as exc:
        log.warning("Failed reading SDF %s: %s", sdf_path, exc)


def _smi_smiles(smi_path: str, max_count: Optional[int] = None) -> Iterator[tuple[str, str]]:
    """Yield (smiles, source_file) from a .smi file (first whitespace token per line)."""
    seen: set[str] = set()
    try:
        with open(smi_path) as fh:
            for line in fh:
                parts = line.strip().split()
                if parts:
                    smi = parts[0]
                    if smi not in seen:
                        seen.add(smi)
                        yield smi, str(smi_path)
                        if max_count and len(seen) >= max_count:
                            return
    except Exception as exc:
        log.warning("Failed reading SMI %s: %s", smi_path, exc)


def _tsv_smiles(tsv_path: str, max_count: Optional[int] = None) -> Iterator[tuple[str, str]]:
    """Yield (smiles, source_file) from a TSV file (first tab-separated field per line)."""
    seen: set[str] = set()
    try:
        with open(tsv_path) as fh:
            for line in fh:
                smi = line.strip().split("\t")[0] if line.strip() else None
                if smi and smi not in seen:
                    seen.add(smi)
                    yield smi, str(tsv_path)
                    if max_count and len(seen) >= max_count:
                        return
    except Exception as exc:
        log.warning("Failed reading TSV %s: %s", tsv_path, exc)


def _group_sdf_round_robin(sdf_files: list[str]) -> list[str]:
    """Group SDF files by leading numeric prefix and interleave round-robin.

    Pocket2Mol and ResGen generate multiple conformers per pocket seed.
    Round-robin over groups ensures an even molecule spread when capping.
    """
    groups: dict[int | str, list[str]] = defaultdict(list)
    for sdf in sdf_files:
        m = re.match(r"(\d+)", os.path.basename(sdf))
        key: int | str = int(m.group(1)) if m else os.path.basename(sdf)
        groups[key].append(sdf)
    for k in groups:
        groups[k].sort()
    result: list[str] = []
    max_len = max(len(v) for v in groups.values()) if groups else 0
    for i in range(max_len):
        for k in sorted(groups):
            if i < len(groups[k]):
                result.append(groups[k][i])
    return result


# ---------------------------------------------------------------------------
# Per-model readers
# Each reader yields (smiles, target, source_file) tuples.
# ---------------------------------------------------------------------------

def _read_pocket2mol(root: Path, target: str, limit: Optional[int]) -> Iterator[tuple[str, str, str]]:
    """SDF files under outputs_0/{TARGET}/sample_*/SDF/*.sdf."""
    target_path = root / target
    seen: set[str] = set()
    done = False
    subdirs = sorted(
        d for d in target_path.iterdir()
        if d.is_dir()
    )
    for subdir in subdirs:
        sdf_folder = subdir / "SDF"
        if not sdf_folder.exists():
            continue
        sdf_files = _group_sdf_round_robin(sorted(glob.glob(str(sdf_folder / "*.sdf"))))
        for sdf_path in sdf_files:
            for smi, src in _sdf_smiles(sdf_path):
                if smi not in seen:
                    seen.add(smi)
                    yield smi, target, src
                    if limit and len(seen) >= limit:
                        done = True
                        break
            if done:
                break
        if done:
            break


def _read_molcraft(root: Path, target: str, limit: Optional[int]) -> Iterator[tuple[str, str, str]]:
    """SDF files flat under benchmark/{TARGET}/*.sdf."""
    target_path = root / target
    seen: set[str] = set()
    for sdf_path in sorted(glob.glob(str(target_path / "*.sdf"))):
        for smi, src in _sdf_smiles(sdf_path):
            if smi not in seen:
                seen.add(smi)
                yield smi, target, src
                if limit and len(seen) >= limit:
                    return


def _read_tamgen(root: Path, target: str, limit: Optional[int]) -> Iterator[tuple[str, str, str]]:
    """TSV files under customized_example/{TARGET}/*_nonvae_flatten.tsv."""
    target_path = root / target
    seen: set[str] = set()
    for tsv_path in sorted(glob.glob(str(target_path / "*_nonvae_flatten.tsv"))):
        for smi, src in _tsv_smiles(tsv_path):
            if smi not in seen:
                seen.add(smi)
                yield smi, target, src
                if limit and len(seen) >= limit:
                    return


def _read_pocketflow(root: Path, target: str, limit: Optional[int]) -> Iterator[tuple[str, str, str]]:
    """SMI files under gen_results_0/{TARGET}/{TIMESTAMP}/generated.smi."""
    target_path = root / target
    seen: set[str] = set()
    subdirs = sorted(
        d for d in target_path.iterdir()
        if d.is_dir()
    )
    for subdir in subdirs:
        smi_file = subdir / "generated.smi"
        if not smi_file.exists():
            # SDF fallback
            sdf_file = subdir / "generated.sdf"
            if not sdf_file.exists():
                continue
            for smi, src in _sdf_smiles(str(sdf_file)):
                if smi not in seen:
                    seen.add(smi)
                    yield smi, target, src
                    if limit and len(seen) >= limit:
                        return
        else:
            for smi, src in _smi_smiles(str(smi_file)):
                if smi not in seen:
                    seen.add(smi)
                    yield smi, target, src
                    if limit and len(seen) >= limit:
                        return


def _read_pocketxmol(root: Path, target: str, limit: Optional[int]) -> Iterator[tuple[str, str, str]]:
    """SDF files under outputs_0/{TARGET}/tmp{ID}_pxm_{TS}/tmp{ID}_pxm_{TS}_SDF/*.sdf.

    Groups SDF files by outer-dir numeric ID (pocket seed) and interleaves
    round-robin — same strategy as Pocket2Mol and ResGen — so the 8 000-molecule
    cap draws evenly from all pocket seeds rather than exhausting early ones first.
    """
    target_path = root / target
    seen: set[str] = set()

    # Collect SDF files grouped by outer-dir numeric ID
    groups: dict[int | str, list[str]] = defaultdict(list)
    for outer in sorted(target_path.iterdir()):
        if not outer.is_dir():
            continue
        m = re.match(r"tmp(\d+)", outer.name)
        key: int | str = int(m.group(1)) if m else outer.name
        for inner in sorted(outer.iterdir()):
            if inner.is_dir() and inner.name.startswith("tmp"):
                groups[key].extend(sorted(glob.glob(str(inner / "*.sdf"))))

    if not groups:
        return

    # Round-robin across pocket seeds
    max_len = max(len(v) for v in groups.values())
    done = False
    for i in range(max_len):
        for key in sorted(groups):
            if i >= len(groups[key]):
                continue
            for smi, src in _sdf_smiles(groups[key][i]):
                if smi not in seen:
                    seen.add(smi)
                    yield smi, target, src
                    if limit and len(seen) >= limit:
                        done = True
                        break
            if done:
                break
        if done:
            break


def _read_resgen(root: Path, target: str, limit: Optional[int]) -> Iterator[tuple[str, str, str]]:
    """SDF files under examples_0/{TARGET}/{TARGET}_ligand/SDF/*.sdf."""
    target_path = root / target
    sdf_folder = target_path / f"{target}_ligand" / "SDF"
    seen: set[str] = set()
    if not sdf_folder.exists():
        return
    sdf_files = _group_sdf_round_robin(sorted(glob.glob(str(sdf_folder / "*.sdf"))))
    for sdf_path in sdf_files:
        for smi, src in _sdf_smiles(sdf_path):
            if smi not in seen:
                seen.add(smi)
                yield smi, target, src
                if limit and len(seen) >= limit:
                    return


# ---------------------------------------------------------------------------
# Model dispatcher
# ---------------------------------------------------------------------------

_READERS = {
    "Pocket2Mol": ("pocket2mol", _read_pocket2mol),
    "MolCRAFT":   ("molcraft",   _read_molcraft),
    "TamGen":     ("tamgen",     _read_tamgen),
    "PocketFlow": ("pocketflow", _read_pocketflow),
    "PocketXMol": ("pocketxmol", _read_pocketxmol),
    "ResGen":     ("resgen",     _read_resgen),
}


def read_model_raw(
    model_name: str,
    raw_root: Optional[str | Path] = None,
    max_per_target: int = MAX_MOLECULES_PER_TARGET,
) -> pd.DataFrame:
    """Read all molecules for *model_name* from its raw output directory.

    Returns a DataFrame with columns: smiles, target, model, source_file.
    Molecules are NOT yet validated or canonicalized here (done in unify()).

    Parameters
    ----------
    model_name:
        One of the 6 supported model names.
    raw_root:
        Override the default raw data root (useful when raw data is in _backup/).
    max_per_target:
        Maximum molecules per target.
    """
    if model_name not in _READERS:
        raise ValueError(f"Unknown model '{model_name}'. Supported: {list(_READERS)}")

    _, reader_fn = _READERS[model_name]
    root = Path(raw_root) if raw_root else MODEL_RAW_ROOTS[model_name]

    if not root.exists():
        log.warning("Model root not found: %s — skipping %s", root, model_name)
        return pd.DataFrame(columns=["smiles", "target", "model", "source_file"])

    records = []
    limit = max_per_target

    target_dirs = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not target_dirs:
        log.warning("No target subdirectories found under %s", root)
        return pd.DataFrame(columns=["smiles", "target", "model", "source_file"])

    for target in target_dirs:
        try:
            it = reader_fn(root, target, limit)
            for smi, tgt, src in it:
                records.append({"smiles": smi, "target": tgt, "model": model_name, "source_file": src})
        except Exception as exc:
            log.warning("Error reading %s / %s: %s", model_name, target, exc)

    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["smiles", "target", "model", "source_file"]
    )
    log.info("%s: %d raw molecules across %d targets", model_name, len(df), df["target"].nunique() if len(df) else 0)
    return df


# ---------------------------------------------------------------------------
# Canonicalization + validation
# ---------------------------------------------------------------------------

def _canonicalize_smiles(smiles: str) -> Optional[str]:
    """Return canonical RDKit SMILES, or None if sanitization fails."""
    Chem = _try_import_rdkit()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Unify: raw → canonical parquet
# ---------------------------------------------------------------------------

def unify(
    model_name: str,
    output_path: str | Path,
    raw_root: Optional[str | Path] = None,
    max_per_target: int = MAX_MOLECULES_PER_TARGET,
) -> pd.DataFrame:
    """Read raw model outputs, canonicalize SMILES, and save as parquet.

    Parameters
    ----------
    model_name:
        Model to process.
    output_path:
        Destination .parquet file path.
    raw_root:
        Override raw data root (e.g. when model dirs are in _backup/).
    max_per_target:
        Per-target molecule cap applied to all models.

    Returns
    -------
    Unified DataFrame with columns:
        smiles, target, model, is_valid, source_file
    """
    from tqdm import tqdm

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_df = read_model_raw(model_name, raw_root=raw_root, max_per_target=max_per_target)
    if raw_df.empty:
        log.warning("No molecules found for %s", model_name)
        unified = pd.DataFrame(columns=["smiles", "target", "model", "is_valid", "source_file"])
        unified.to_parquet(output_path, index=False)
        return unified

    canonical, is_valid_list = [], []
    for smi in tqdm(raw_df["smiles"], desc=f"Canonicalizing {model_name}", leave=False):
        canon = _canonicalize_smiles(smi)
        if canon is not None:
            canonical.append(canon)
            is_valid_list.append(True)
        else:
            canonical.append(smi)  # keep original for traceability
            is_valid_list.append(False)

    unified = pd.DataFrame({
        "smiles":      canonical,
        "target":      raw_df["target"].values,
        "model":       raw_df["model"].values,
        "is_valid":    is_valid_list,
        "source_file": raw_df["source_file"].values,
    })

    # Drop duplicate canonical SMILES within same (target, model) pair
    before = len(unified)
    unified = unified.drop_duplicates(subset=["smiles", "target", "model"])
    after = len(unified)
    if before != after:
        log.info("%s: removed %d duplicates (canonical dedup)", model_name, before - after)

    unified.to_parquet(output_path, index=False)

    n_valid = unified["is_valid"].sum()
    log.info(
        "%s → %s  [%d total | %d valid (%.1f%%) | %d targets]",
        model_name, output_path.name, len(unified), n_valid,
        100 * n_valid / max(len(unified), 1),
        unified["target"].nunique(),
    )
    return unified


# ---------------------------------------------------------------------------
# Load functions (post-unify)
# ---------------------------------------------------------------------------

def load_generated(
    model_name: str,
    data_dir: str | Path = None,
    valid_only: bool = True,
) -> pd.DataFrame:
    """Load the unified parquet for *model_name*.

    Parameters
    ----------
    model_name:
        Model whose parquet to load.
    data_dir:
        Root data directory (default: repo_root/data/generated/unified/).
    valid_only:
        If True, return only rows where is_valid=True.
    """
    if data_dir is None:
        data_dir = BASE_DIR / "data" / "generated" / "unified"
    parquet_path = Path(data_dir) / f"{model_name}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Unified parquet not found: {parquet_path}\n"
            f"Run: python cli.py unify --models {model_name}"
        )
    df = pd.read_parquet(parquet_path)
    if valid_only:
        df = df[df["is_valid"]].reset_index(drop=True)
    return df


def load_benchmark(
    target: str,
    data_dir: str | Path = None,
) -> pd.DataFrame:
    """Load patent benchmark molecules for a given target.

    Looks in data/benchmark/{target}.csv which has columns:
        smiles, target, activity, grade  (plus optional: pdbcode, patent_id, lig_id)

    Falls back to the original targets/{target}/{target}_final.csv if the
    curated benchmark CSV does not exist.
    """
    if data_dir is None:
        data_dir = BASE_DIR / "data" / "benchmark"
    benchmark_path = Path(data_dir) / f"{target}.csv"
    if benchmark_path.exists():
        df = pd.read_csv(benchmark_path)
        col_map = {"SMILES": "smiles", "Target": "target", "Activity": "activity"}
        df = df.rename(columns=col_map)
        return df

    # Fallback: original targets directory (may be in _backup/)
    original = BASE_DIR / "targets" / target / f"{target}_final.csv"
    backup = BASE_DIR / "_backup" / "targets" / target / f"{target}_final.csv"
    for path in (original, backup):
        if path.exists():
            df = pd.read_csv(path)
            # Normalise column names to unified schema
            col_map = {
                "SMILES": "smiles", "Target": "target",
                "Activity": "activity",
            }
            df = df.rename(columns=col_map)
            if "smiles" not in df.columns:
                raise KeyError(f"No SMILES column in {path}")
            return df
    raise FileNotFoundError(
        f"Benchmark data not found for '{target}'.\n"
        f"Expected: {benchmark_path}\n"
        f"Run: python cli.py prepare-benchmark to copy from targets/ directory."
    )


def load_training_set(
    model_name: str,
    data_dir: str | Path = None,
) -> pd.DataFrame:
    """Load the training set SMILES for a given model.

    PocketXMol uses its own training set (all_smiles_merged.csv);
    all other models use CrossDocked2020 (crossdock_train_smi_with_rascore.csv).

    Returns DataFrame with at minimum a 'smiles' column.
    """
    if data_dir is None:
        data_dir = BASE_DIR / "data" / "training_set"
    data_dir = Path(data_dir)

    if model_name == "PocketXMol":
        fname = "pocketxmol_train_smiles.csv"
        fallbacks = [
            data_dir / fname,
            BASE_DIR / "PocketXMol" / "all_smiles_merged.csv",
            BASE_DIR / "_backup" / "PocketXMol" / "all_smiles_merged.csv",
        ]
    else:
        fname = "crossdocked_smiles.csv"
        fallbacks = [
            data_dir / fname,
            BASE_DIR / "crossdock_train_smi_with_rascore.csv",
            BASE_DIR / "_backup" / "crossdock_train_smi_with_rascore.csv",
        ]

    for path in fallbacks:
        if Path(path).exists():
            df = pd.read_csv(path)
            # Normalise SMILES column to lowercase 'smiles'
            for col in ("SMILES", "Smiles", "smiles"):
                if col in df.columns:
                    df = df.rename(columns={col: "smiles"})
                    break
            return df

    raise FileNotFoundError(
        f"Training set not found for '{model_name}'.\n"
        f"Checked: {[str(p) for p in fallbacks]}"
    )


# ---------------------------------------------------------------------------
# Ingest: external user models
# ---------------------------------------------------------------------------

def ingest_external(
    model_name: str,
    input_dir: str | Path,
    output_path: str | Path,
    input_format: str = "sdf",
    smiles_field: Optional[str] = None,
    target_field: Optional[str] = None,
    max_per_target: Optional[int] = None,
) -> pd.DataFrame:
    """Convert an external model's output into the unified parquet schema.

    Supported input_format values:
        "sdf"    — directory tree or flat SDF files; target from subdirectory name
        "smiles" — single .smi file with per-molecule SMILES
        "tsv"    — TSV file; first column = SMILES unless smiles_field overrides
        "csv"    — CSV with explicit smiles_field and target_field columns

    Returns the unified DataFrame and writes it to *output_path*.
    """
    from tqdm import tqdm

    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    fmt = input_format.lower()

    if fmt == "sdf":
        # Directory tree: subdirectory name = target
        sdf_files = sorted(input_dir.glob("**/*.sdf"))
        for sdf_path in tqdm(sdf_files, desc=f"Reading {model_name} SDFs"):
            # Determine target: parent dir name (if not same as input_dir)
            rel_parent = sdf_path.parent.relative_to(input_dir)
            target = rel_parent.parts[0] if rel_parent.parts else sdf_path.stem
            lim = max_per_target  # None = unlimited
            for smi, src in _sdf_smiles(str(sdf_path), lim):
                records.append({"smiles": smi, "target": target,
                                 "model": model_name, "source_file": src})

    elif fmt in ("smiles", "smi"):
        fname = smiles_field or "smiles"
        for smi_file in sorted(input_dir.glob("**/*.smi")):
            target = smi_file.parent.name if smi_file.parent != input_dir else smi_file.stem
            for smi, src in _smi_smiles(str(smi_file), max_per_target):
                records.append({"smiles": smi, "target": target,
                                 "model": model_name, "source_file": src})

    elif fmt == "tsv":
        for tsv_file in sorted(input_dir.glob("**/*.tsv")):
            target = tsv_file.parent.name if tsv_file.parent != input_dir else tsv_file.stem
            for smi, src in _tsv_smiles(str(tsv_file), max_per_target):
                records.append({"smiles": smi, "target": target,
                                 "model": model_name, "source_file": src})

    elif fmt == "csv":
        smiles_col = smiles_field or "smiles"
        target_col = target_field or "target"
        for csv_file in sorted(input_dir.glob("**/*.csv")):
            df_raw = pd.read_csv(csv_file)
            if smiles_col not in df_raw.columns:
                log.warning("Column '%s' not found in %s — skipping", smiles_col, csv_file)
                continue
            for _, row in df_raw.iterrows():
                tgt = str(row[target_col]) if target_col in df_raw.columns else csv_file.stem
                records.append({"smiles": str(row[smiles_col]), "target": tgt,
                                 "model": model_name, "source_file": str(csv_file)})
    else:
        raise ValueError(f"Unsupported input_format='{fmt}'. Use: sdf, smiles, tsv, csv")

    if not records:
        log.warning("No molecules found in %s", input_dir)
        unified = pd.DataFrame(columns=["smiles", "target", "model", "is_valid", "source_file"])
        unified.to_parquet(output_path, index=False)
        return unified

    raw_df = pd.DataFrame(records)

    # Canonicalize
    canonical, is_valid_list = [], []
    for smi in tqdm(raw_df["smiles"], desc=f"Canonicalizing {model_name}", leave=False):
        canon = _canonicalize_smiles(smi)
        if canon is not None:
            canonical.append(canon)
            is_valid_list.append(True)
        else:
            canonical.append(smi)
            is_valid_list.append(False)

    unified = pd.DataFrame({
        "smiles":      canonical,
        "target":      raw_df["target"].values,
        "model":       raw_df["model"].values,
        "is_valid":    is_valid_list,
        "source_file": raw_df["source_file"].values,
    }).drop_duplicates(subset=["smiles", "target", "model"])

    unified.to_parquet(output_path, index=False)

    # Validation report
    n_total = len(unified)
    n_valid = int(unified["is_valid"].sum())
    print(f"\n=== Ingest Report: {model_name} ===")
    print(f"Total molecules : {n_total}")
    print(f"Valid (RDKit)   : {n_valid} ({100*n_valid/max(n_total,1):.1f}%)")
    print(f"Invalid         : {n_total - n_valid}")
    print("\nPer-target counts:")
    for tgt, grp in unified.groupby("target"):
        n_v = int(grp["is_valid"].sum())
        print(f"  {tgt:<20s} {len(grp):>6d} total | {n_v:>6d} valid")
    print(f"\nOutput: {output_path}")

    return unified
