#!/usr/bin/env python3
"""
SBDD Benchmark CLI — command-line interface for the evaluation framework.

Usage examples
--------------
# Convert raw model outputs to unified parquet format:
    python cli.py unify --models all
    python cli.py unify --models MolCRAFT PocketXMol --raw-root _backup/

# Ingest a new model (external user):
    python cli.py ingest --model MyModel --input-dir /path/to/outputs \
        --input-format sdf --target-field dirname

# Evaluate one or more stages:
    python cli.py evaluate --stage 1 --models all --targets all
    python cli.py evaluate --stage 2 --models MolCRAFT PocketXMol --targets JAK1 DRD2
    python cli.py evaluate --stage 3 --models all --targets kinase gpcr
    python cli.py evaluate --stage all --models all --targets all --output-dir results/

# Compare a new model against pre-computed baselines:
    python cli.py evaluate --stage all --models MyModel --targets all \
        --compare-baselines

# Generate figures (reads CSVs — no RDKit):
    python cli.py plot --stage 1 --output-dir results/figures/
    python cli.py plot --stage all --output-dir results/figures/

# Show available targets and target families:
    python cli.py list-targets
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# ── Target/model constants (duplicated here to avoid import at parse time) ──
ALL_MODELS  = ["Pocket2Mol", "MolCRAFT", "TamGen", "PocketFlow", "PocketXMol", "ResGen"]
ALL_TARGETS = [
    "ROCK2", "CDK9", "JAK1", "ACVR1", "AKT1",
    "EZH2", "PRMT5", "MMP8", "WRN",
    "GCGR", "5HT2A", "DRD2", "AGTR1",
    "LXRB", "FXR", "AR",
    "BCL2", "BRD4", "Keap1", "EED",
]
TARGET_FAMILIES = {
    "kinase":            ["ROCK2", "CDK9", "JAK1", "ACVR1", "AKT1"],
    "non_kinase_enzyme": ["EZH2", "PRMT5", "MMP8", "WRN"],
    "gpcr":              ["GCGR", "5HT2A", "DRD2", "AGTR1"],
    "nuclear_receptor":  ["LXRB", "FXR", "AR"],
    "ppi":               ["BCL2", "BRD4", "Keap1", "EED"],
}
FAMILY_KEYWORDS = list(TARGET_FAMILIES.keys()) + ["all"]


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="cli.py",
        description="SBDD Benchmark — decision-centric evaluation framework for SBDD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = root.add_subparsers(dest="command", required=True)

    # ── unify ────────────────────────────────────────────────────────────────
    p_unify = sub.add_parser(
        "unify",
        help="Convert raw model outputs to unified parquet format",
    )
    p_unify.add_argument(
        "--models", nargs="+", default=["all"],
        help=f"Models to process. Use 'all' for all 6. Choices: {ALL_MODELS}",
    )
    p_unify.add_argument(
        "--raw-root", default=None,
        help="Override root for raw model directories (useful when dirs are in _backup/)",
    )
    p_unify.add_argument(
        "--output-dir", default=str(REPO_ROOT / "data" / "generated" / "unified"),
        help="Destination directory for parquet files",
    )
    p_unify.add_argument("--max-per-target", type=int, default=8000)
    p_unify.add_argument("--force", action="store_true",
                         help="Regenerate even if parquet already exists")

    # ── ingest ───────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser(
        "ingest",
        help="Import an external/new model into the unified parquet format",
    )
    p_ingest.add_argument("--model", required=True, help="Name for the new model")
    p_ingest.add_argument("--input-dir", required=True, help="Directory containing model outputs")
    p_ingest.add_argument(
        "--input-format", choices=["sdf", "smiles", "tsv", "csv"], default="sdf",
        help="Input file format",
    )
    p_ingest.add_argument("--smiles-field", default=None,
                          help="Column/property name for SMILES (CSV/SDF)")
    p_ingest.add_argument("--target-field", default=None,
                          help="Column name for target ID (CSV) or 'dirname' to use parent dir")
    p_ingest.add_argument(
        "--output-dir", default=str(REPO_ROOT / "data" / "generated" / "unified"),
        help="Output directory for the unified parquet",
    )

    # ── evaluate ─────────────────────────────────────────────────────────────
    p_eval = sub.add_parser(
        "evaluate",
        help="Run evaluation stages (1=similarity, 2=properties, 3=recovery)",
    )
    p_eval.add_argument(
        "--stage", required=True,
        choices=["1", "2", "3", "all"],
        help="Evaluation stage to run",
    )
    p_eval.add_argument(
        "--models", nargs="+", default=["all"],
        help=f"Models to evaluate. 'all' = all 6. Choices: {ALL_MODELS}",
    )
    p_eval.add_argument(
        "--targets", nargs="+", default=["all"],
        help=(
            "Targets to evaluate. 'all' = all 20. "
            f"Family keywords: {FAMILY_KEYWORDS}. "
            "Or explicit names e.g.: JAK1 DRD2 BCL2"
        ),
    )
    p_eval.add_argument(
        "--output-dir", default=str(REPO_ROOT / "results"),
        help="Root directory for results CSVs",
    )
    p_eval.add_argument(
        "--data-dir", default=str(REPO_ROOT / "data"),
        help="Root data directory containing benchmark/, training_set/, generated/",
    )
    p_eval.add_argument(
        "--compare-baselines", action="store_true",
        help=(
            "Merge pre-computed baseline results from results/baselines/ "
            "into output CSVs for cross-model comparison"
        ),
    )
    p_eval.add_argument(
        "--compute-rascore", action="store_true",
        help="Compute RAscore (Stage 2 only; requires 'rascore' conda env)",
    )
    p_eval.add_argument("--seed", type=int, default=42)

    # ── plot ─────────────────────────────────────────────────────────────────
    p_plot = sub.add_parser(
        "plot",
        help="Generate figures from pre-computed result CSVs (no RDKit required)",
    )
    p_plot.add_argument(
        "--stage", required=True, choices=["1", "2", "3", "all"],
        help="Which stage figures to generate",
    )
    p_plot.add_argument(
        "--results-dir", default=str(REPO_ROOT / "results"),
        help="Root directory where stage1/stage2/stage3 result CSVs live",
    )
    p_plot.add_argument(
        "--output", default=str(REPO_ROOT / "results" / "figures"),
        help="Output directory for PNG/SVG figures",
    )

    # ── list-targets ─────────────────────────────────────────────────────────
    sub.add_parser("list-targets", help="Print all targets and family groupings")

    return root


# ── Helper: resolve model/target lists ──────────────────────────────────────

def _resolve_models(models_arg: list[str]) -> list[str]:
    if "all" in models_arg:
        return ALL_MODELS
    return models_arg


def _resolve_targets(targets_arg: list[str]) -> list[str]:
    if "all" in targets_arg:
        return ALL_TARGETS
    resolved = []
    for t in targets_arg:
        if t in TARGET_FAMILIES:
            resolved.extend(TARGET_FAMILIES[t])
        else:
            resolved.append(t)
    return list(dict.fromkeys(resolved))


# ── Command handlers ──────────────────────────────────────────────────────────

def cmd_unify(args: argparse.Namespace) -> None:
    from sbdd_benchmark.io.loaders import unify, MODEL_RAW_ROOTS

    models = _resolve_models(args.models)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Unifying {len(models)} model(s) → {output_dir}")
    failed = []
    for model in models:
        parquet_path = output_dir / f"{model}.parquet"
        if parquet_path.exists() and not args.force:
            import pandas as pd
            n = len(pd.read_parquet(parquet_path))
            print(f"  [SKIP] {model}: {n} rows already. Use --force to regenerate.")
            continue

        raw_root = args.raw_root
        # Auto-fallback to _backup/ if default root is missing
        if raw_root is None:
            default_root = MODEL_RAW_ROOTS.get(model)
            if default_root and not default_root.exists():
                backup = REPO_ROOT / "_backup" / default_root.relative_to(REPO_ROOT)
                if backup.exists():
                    raw_root = str(backup)
                    print(f"  [INFO] {model}: using _backup path")

        try:
            df = unify(model, parquet_path, raw_root=raw_root,
                       max_per_target=args.max_per_target)
            print(f"  [OK]   {model}: {len(df)} molecules → {parquet_path.name}")
        except Exception as exc:
            print(f"  [ERR]  {model}: {exc}", file=sys.stderr)
            failed.append(model)

    if failed:
        print(f"\n[WARN] Failed: {failed}", file=sys.stderr)
        sys.exit(1)


def cmd_ingest(args: argparse.Namespace) -> None:
    from sbdd_benchmark.io.loaders import ingest_external

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{args.model}.parquet"

    ingest_external(
        model_name=args.model,
        input_dir=args.input_dir,
        output_path=parquet_path,
        input_format=args.input_format,
        smiles_field=args.smiles_field,
        target_field=args.target_field,
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    from sbdd_benchmark.pipeline import run_stage1, run_stage2, run_stage3, run_all

    models  = _resolve_models(args.models)
    targets = _resolve_targets(args.targets)
    data_dir    = args.data_dir
    results_dir = args.output_dir
    kw = dict(models=models, targets=targets,
              data_dir=data_dir, results_dir=results_dir,
              compare_baselines=args.compare_baselines)

    if args.stage == "1":
        run_stage1(**kw, seed=args.seed)
    elif args.stage == "2":
        run_stage2(**kw, compute_rascore=args.compute_rascore)
    elif args.stage == "3":
        run_stage3(**kw)
    elif args.stage == "all":
        run_all(**kw, seed=args.seed, compute_rascore=args.compute_rascore)


def cmd_plot(args: argparse.Namespace) -> None:
    from sbdd_benchmark.visualization.plots import plot_stage1, plot_stage2, plot_stage3, plot_all

    kw = dict(results_dir=args.results_dir, output_dir=args.output)
    if args.stage == "1":
        plot_stage1(**kw)
    elif args.stage == "2":
        plot_stage2(**kw)
    elif args.stage == "3":
        plot_stage3(**kw)
    elif args.stage == "all":
        plot_all(**kw)
    print(f"\nFigures saved to: {args.output}")


def cmd_list_targets(_args: argparse.Namespace) -> None:
    print(f"\nAll 20 targets ({len(ALL_TARGETS)}):")
    for t in ALL_TARGETS:
        print(f"  {t}")
    print("\nTarget families:")
    for fam, tgts in TARGET_FAMILIES.items():
        print(f"  --targets {fam:<20s}  →  {', '.join(tgts)}")
    print(
        "\nUsage examples:\n"
        "  python cli.py evaluate --stage 3 --targets kinase\n"
        "  python cli.py evaluate --stage 1 --targets JAK1 DRD2 BCL2\n"
        "  python cli.py evaluate --stage all --targets all\n"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    dispatch = {
        "unify":        cmd_unify,
        "ingest":       cmd_ingest,
        "evaluate":     cmd_evaluate,
        "plot":         cmd_plot,
        "list-targets": cmd_list_targets,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
