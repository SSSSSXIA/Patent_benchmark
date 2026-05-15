#  decision-centric framework for benchmarking generative AI in structure-based drug design


This package implements the three-stage evaluation framework described in: *"A decision-centric framework for benchmarking generative AI for structure-based drug design"*.

Each stage mirrors a real medicinal chemistry decision point:

| Stage | Decision | Key Metrics |
|-------|----------|-------------|
| **1 — Chemical Space Exploration** | Are generated molecules novel vs training data, and do they broadly cover chemical space? | Max Tanimoto similarity to training set; #Circles coverage (t = 0.75) |
| **2 — Chemical Admissibility** | Do generated molecules meet drug-likeness criteria? | REOS Dundee pass rate; SA Score; 9 physicochemical descriptors vs patent reference (Wasserstein distance) |
| **3 — Patent Molecule Recovery** | Can the model generate scaffold-level analogues of known actives? | Exact / Bemis-Murcko / Generic scaffold recovery rate across 20 targets |

---

## Installation

RDKit must be installed via conda:

```bash
conda create -n sbdd python=3.9
conda activate sbdd
conda install -c conda-forge rdkit
pip install -e .
pip install useful-rdkit-utils    # required for REOS filtering
```

---

## Evaluating a new model

### Step 1 — Prepare your generated molecules

Organize your model's output as a directory tree where **each subdirectory is named after a target**:

```
my_model_outputs/
├── JAK1/
│   ├── mol_001.sdf
│   ├── mol_002.sdf
│   └── ...
├── DRD2/
│   └── ...
└── BCL2/
    └── ...
```

Supported formats: `sdf` (recommended), `smiles` (`.smi` files), `tsv`, `csv`.

For CSV/TSV, make sure there is a SMILES column and a target column (pass column names via `--smiles-field` and `--target-field`).

### Step 2 — Ingest into unified format

```bash
python cli.py ingest \
    --model MyModel \
    --input-dir /path/to/my_model_outputs \
    --input-format sdf \
    --output-dir data/generated/unified/
```

This reads all SDF files, canonicalizes SMILES via RDKit, removes duplicates, and saves a single parquet file at `data/generated/unified/MyModel.parquet`.

**For CSV input** (one file with smiles + target columns):

```bash
python cli.py ingest \
    --model MyModel \
    --input-dir /path/to/outputs \
    --input-format csv \
    --smiles-field SMILES \
    --target-field target_name \
    --output-dir data/generated/unified/
```

### Step 3 — Run all evaluation stages

```bash
python cli.py evaluate \
    --stage all \
    --models MyModel \
    --targets all \
    --output-dir results/MyModel/
```

This runs Stages 1, 2, and 3 sequentially and saves result CSVs to `results/MyModel/`.

### Step 4 — Compare against the 6 paper baselines

Add `--compare-baselines` to merge pre-computed baseline results into the output CSVs so all models appear side by side in figures:

```bash
python cli.py evaluate \
    --stage all \
    --models MyModel \
    --targets all \
    --compare-baselines \
    --output-dir results/MyModel/
```

### Step 5 — Generate figures

```bash
python cli.py plot \
    --stage all \
    --results-dir results/MyModel/ \
    --output results/MyModel/figures/
```

Figures are saved as both PNG (300 dpi) and SVG.

---

## Running individual stages

```bash
# Stage 1 only (training similarity + #Circles)
python cli.py evaluate --stage 1 --models MyModel --targets all --output-dir results/MyModel/

# Stage 2 only (descriptors, REOS, Wasserstein)
python cli.py evaluate --stage 2 --models MyModel --targets all --output-dir results/MyModel/

# Stage 3 only (scaffold recovery)
python cli.py evaluate --stage 3 --models MyModel --targets all --output-dir results/MyModel/
```

> **Note:** Stages 1 and 2 compute global metrics over all generated molecules.
> Stage 3 is evaluated per target.

---

## Evaluating a subset of targets

All 20 targets can be filtered by name or by target family:

```bash
# Specific targets
python cli.py evaluate --stage 3 --models MyModel --targets JAK1 DRD2 BCL2

# By family keyword
python cli.py evaluate --stage 3 --models MyModel --targets kinase
python cli.py evaluate --stage 3 --models MyModel --targets kinase gpcr ppi
```

Available family keywords:

```
kinase            →  ROCK2  CDK9  JAK1  ACVR1  AKT1
non_kinase_enzyme →  EZH2   PRMT5 MMP8  WRN
gpcr              →  GCGR   5HT2A DRD2  AGTR1
nuclear_receptor  →  LXRB   FXR   AR
ppi               →  BCL2   BRD4  Keap1 EED
```

To list all 20 targets and families:

```bash
python cli.py list-targets
```

---

## Output files

Results are written as tidy CSVs with columns `model`, `target`, and metric columns.

```
results/
├── stage1/
│   ├── similarity_scores.csv      # per-molecule MaxTanimoto to training set
│   ├── similarity_summary.csv     # per-model mean, median, frac > 0.2 / 0.3 / 0.5
│   └── circles_coverage.csv       # #Circles mean ± std, efficiency per model
├── stage2/
│   ├── {model}_properties.csv     # per-molecule descriptors + REOS pass/fail + SA Score
│   ├── reos_summary.csv           # per-model REOS pass rate
│   ├── wasserstein_raw.csv        # raw Wasserstein distances vs patent reference
│   └── wasserstein_heatmap.csv    # normalized Wasserstein distances [0–1]
├── stage3/
│   ├── {model}_recovery.csv       # per-target exact / BM / generic scaffold recovery
│   └── recovery_summary.csv       # cross-target mean ± SEM, target coverage
└── figures/
    ├── *.png                      # 300 dpi raster figures
    └── *.svg                      # vector figures
```

Pre-computed results for the 6 paper models are stored in `results/baselines/` and are used automatically when `--compare-baselines` is specified.

---

## Data directory layout

```
data/
├── benchmark/
│   └── {TARGET}.csv          # patent active molecules per target
│                             # columns: smiles, target, activity
├── training_set/
│   ├── crossdocked_smiles.csv          # CrossDocked2020 (used by all models except PocketXMol)
│   └── pocketxmol_train_smiles.csv     # PocketXMol training set
└── generated/
    └── unified/
        └── {model}.parquet   # one file per model (created by ingest or unify)
```

`data/benchmark/` and `data/training_set/` are included in this repository.

---

## Reproducibility notes

- **Fingerprints**: ECFP4, Morgan radius = 2, 2048 bits throughout.
- **#Circles**: greedy algorithm, distance threshold t = 0.75 (d = 1 − Tanimoto), mean ± std over 3 random permutations.
- **Exact recovery**: stereochemistry stripped before SMILES comparison.
- **Generic scaffold**: `MurckoScaffold.MakeScaffoldGeneric()` — all heteroatoms → C, all bonds → single.
- **Wasserstein normalization**: per-descriptor, divided by max WD across models; patent reference = 0.
- **Molecule cap**: 8,000 molecules per target for all models.
- **Random seed**: 42 throughout; configurable via `--seed`.

---

## Python API

```python
from sbdd_benchmark.io.loaders import load_generated, load_benchmark
from sbdd_benchmark.stage1.exploration import training_similarity, circles_coverage
from sbdd_benchmark.stage2.filters import reos_filter
from sbdd_benchmark.stage2.properties import compute_descriptors
from sbdd_benchmark.stage3.recovery import evaluate_target, recovery_summary
from sbdd_benchmark.pipeline import run_all

# Load data
df = load_generated("MolCRAFT")      # unified parquet → DataFrame
bm = load_benchmark("JAK1")          # patent benchmark molecules

# Stage 3: recovery for a single target
result = evaluate_target(
    target_name="JAK1",
    generated_smiles=df[df["target"] == "JAK1"]["smiles"].tolist(),
    true_smiles=bm["smiles"].tolist(),
)
print(f"Generic scaffold recovery: {result['generic_scaffold_recovery_ratio']:.1%}")

# Full pipeline (all stages, selected models and targets)
run_all(
    models=["MolCRAFT", "MyModel"],
    targets=["JAK1", "DRD2", "BCL2"],
    results_dir="results/comparison/",
)
```

---

## Citation

```bibtex
@article{sbdd_benchmark_2026,
  title   = {A decision-centric framework for benchmarking generative AI
             for structure-based drug design},
  author  = {},
  journal = {},
  year    = {},
}
```
