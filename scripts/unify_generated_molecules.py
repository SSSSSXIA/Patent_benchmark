#!/usr/bin/env python3
"""
One-time conversion script: raw model outputs → unified parquet files.

Usage
-----
# All 6 models (default):
    python scripts/unify_generated_molecules.py

# Specific models:
    python scripts/unify_generated_molecules.py --models MolCRAFT PocketXMol

# Override raw data root per model (useful if model dirs are in _backup/):
    python scripts/unify_generated_molecules.py --raw-root /path/to/model/parent

Output
------
    data/generated/unified/{model_name}.parquet

Each parquet has columns:
    smiles       : canonical RDKit SMILES
    target       : target name (e.g. "JAK1")
    model        : model name (e.g. "PocketXMol")
    is_valid     : True if RDKit sanitization succeeded
    source_file  : original file path for traceability
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure the repo root is on sys.path when run directly
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sbdd_benchmark.io.loaders import unify, MODEL_RAW_ROOTS

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

ALL_MODELS = list(MODEL_RAW_ROOTS.keys())
UNIFIED_DIR = REPO_ROOT / "data" / "generated" / "unified"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert raw model outputs to unified parquet format."
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        choices=ALL_MODELS,
        metavar="MODEL",
        help=f"Models to unify. Default: all. Choices: {ALL_MODELS}",
    )
    p.add_argument(
        "--output-dir",
        default=str(UNIFIED_DIR),
        help=f"Directory for output parquets. Default: {UNIFIED_DIR}",
    )
    p.add_argument(
        "--raw-root",
        default=None,
        help=(
            "Override raw model root. If set, each model is read from "
            "<raw-root>/<ModelName>/<model_subpath>. "
            "Useful when model dirs have been moved to _backup/."
        ),
    )
    p.add_argument(
        "--max-per-target",
        type=int,
        default=8000,
        help="Max molecules per target (PocketXMol is always unlimited). Default: 8000",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-generate even if parquet already exists.",
    )
    return p.parse_args()


def _raw_root_for(model_name: str, global_override: str | None) -> str | None:
    """Return the raw root for this model, accounting for _backup/ fallback."""
    if global_override:
        # User explicitly passed --raw-root
        return global_override

    default = MODEL_RAW_ROOTS[model_name]
    if default.exists():
        return None  # use built-in default

    # Try _backup/
    backup_candidate = REPO_ROOT / "_backup" / model_name
    # Reconstruct model subpath relative to MODEL_RAW_ROOTS
    # e.g. default = .../Pocket2Mol/outputs_0  → model folder = Pocket2Mol
    model_folder = default.parent.name  # e.g. "Pocket2Mol"
    backup_root = REPO_ROOT / "_backup" / model_folder
    if backup_root.exists():
        # Return the _backup parent so unify() can construct the right subpath
        # Actually we need to pass the *same relative structure*
        # MODEL_RAW_ROOTS[model_name] = REPO_ROOT / model_folder / subpath
        # In _backup: REPO_ROOT / "_backup" / model_folder / subpath
        return str(REPO_ROOT / "_backup" / model_folder / default.relative_to(default.parent))
    return None


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SBDD Benchmark — Unify Generated Molecules")
    print(f"{'='*60}")
    print(f"Models    : {args.models}")
    print(f"Output dir: {output_dir}")
    print(f"Cap/target: {args.max_per_target} (PocketXMol: unlimited)")
    print(f"{'='*60}\n")

    summary = []
    total_t0 = time.time()

    for model_name in args.models:
        parquet_path = output_dir / f"{model_name}.parquet"

        if parquet_path.exists() and not args.force:
            import pandas as pd
            existing = pd.read_parquet(parquet_path)
            print(f"[SKIP] {model_name}: parquet already exists "
                  f"({len(existing)} rows, {existing['target'].nunique()} targets). "
                  f"Use --force to regenerate.\n")
            summary.append({
                "model": model_name, "status": "skipped",
                "total": len(existing), "valid": int(existing["is_valid"].sum()),
                "targets": existing["target"].nunique(),
            })
            continue

        print(f">>> Processing {model_name} ...")
        t0 = time.time()

        # Determine raw root: default or _backup/ fallback
        raw_root = None
        default_root = MODEL_RAW_ROOTS[model_name]
        if not default_root.exists():
            # try _backup
            backup = REPO_ROOT / "_backup" / default_root.relative_to(REPO_ROOT)
            if backup.exists():
                raw_root = str(backup)
                print(f"    [INFO] Using _backup path: {raw_root}")
            else:
                print(f"    [WARN] Raw root not found: {default_root} — skipping.")
                summary.append({"model": model_name, "status": "missing_data",
                                 "total": 0, "valid": 0, "targets": 0})
                continue

        if args.raw_root:
            raw_root = args.raw_root

        try:
            df = unify(
                model_name=model_name,
                output_path=parquet_path,
                raw_root=raw_root,
                max_per_target=args.max_per_target,
            )
            elapsed = time.time() - t0
            n_valid = int(df["is_valid"].sum())
            n_targets = df["target"].nunique() if len(df) else 0
            print(
                f"    Done: {len(df)} molecules | {n_valid} valid "
                f"({100*n_valid/max(len(df),1):.1f}%) | "
                f"{n_targets} targets | {elapsed:.1f}s\n"
            )
            summary.append({
                "model": model_name, "status": "ok",
                "total": len(df), "valid": n_valid, "targets": n_targets,
            })
        except Exception as exc:
            log.error("Failed to unify %s: %s", model_name, exc, exc_info=True)
            summary.append({"model": model_name, "status": "error",
                             "total": 0, "valid": 0, "targets": 0})

    # Final summary table
    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"SUMMARY  (total time: {total_elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Status':<12} {'Total':>8} {'Valid':>8} {'Targets':>8}")
    print(f"{'-'*15} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    for s in summary:
        print(
            f"{s['model']:<15} {s['status']:<12} "
            f"{s['total']:>8} {s['valid']:>8} {s['targets']:>8}"
        )
    print(f"{'='*60}\n")

    failed = [s["model"] for s in summary if s["status"] == "error"]
    if failed:
        print(f"[ERROR] Failed models: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
