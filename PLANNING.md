# PLANNING.md — Generation_Benchmark Inventory Report

> Auto-generated during Step 0 exploration prior to package refactoring.  
> Do NOT delete; this is the authoritative record for the refactor.

---

## 1. Per-Model File Formats & Quirks

| Model | Output root | File format | SMILES extraction | Target identifier | Cap |
|-------|------------|------------|-------------------|-------------------|-----|
| **Pocket2Mol** | `Pocket2Mol/outputs_0/{TARGET}/sample_for_pdb_{TARGET}_protein.pdb_{TIMESTAMP}/SDF/` | SDF | `Chem.MolToSmiles(mol)` | Folder name | 8 000 |
| **MolCRAFT** | `MolCRAFT/benchmark/{TARGET}/*.sdf` | SDF (flat) | `Chem.MolToSmiles(mol)` | Folder name | 8 000 |
| **TamGen** | `TamGen/customized_example/{TARGET}/{PDB}_nonvae_flatten.tsv` | TSV | First tab-delimited field | Folder name | 8 000 |
| **PocketFlow** | `PocketFlow/gen_results_0/{TARGET}/{TIMESTAMP}/generated.smi` | SMI (primary) or SDF (fallback) | First space-separated token | Folder name | 8 000 |
| **PocketXMol** | `PocketXMol/outputs_0/{TARGET}/tmp{ID}_{TIMESTAMP}/*.sdf` | SDF (nested temp dirs) | `Chem.MolToSmiles(mol)` | Folder name | **None** (all molecules) |
| **ResGen** | `ResGen/examples_0/{TARGET}/{TARGET}_ligand/SDF/*.sdf` | SDF (deepest nesting) | `Chem.MolToSmiles(mol)` | Folder prefix | 8 000 |
| **RAScore** | External package — not a generator; provides `rascore()` per SMILES | — | — | — | — |

### Model-specific quirks
- **PocketXMol** is the only model with no molecule cap (`limit=None`). It was also trained on PDBBind + CrossDocked2020 + Binding MOAD (all others: CrossDocked2020 only), so a *separate training set* (`all_smiles_merged.csv`) must be used for its Tanimoto baseline.
- **MolCRAFT** folder on disk is named `MolCRAFT/` but code aliases it to `"MolCraft"` in some places. Display name is `"MolCRAFT"`.
- **TamGen** is the only model that stores outputs as TSV rather than SDF or SMI.
- **PocketFlow** uses round-robin across multiple timestamped run directories.
- **Pocket2Mol & ResGen** use round-robin sub-directory sampling to reach the 8 k cap.
- All SDF readers use `sanitize=True, removeHs=True, strictParsing=False`.

---

## 2. Reference Data

### targets/ directory (20 targets)
- **Path**: `targets/{TARGET}/{TARGET}_final.csv`
- **Columns**: `Target, PDBcode, crystal ligand, Patent id, lig_id, SMILES, Activity`
- **Activity grades**: A (< 0.1 µM), B (0.1–1 µM), C (1–5 µM), D (5–10 µM)
- **Consolidated patent file**: `targets/merged_patent.csv` (83k+ molecules, all targets)
- **ChEMBL reference**: `targets/merged_chembl.csv`
- **3-D SDF**: `targets/ligpreppatent.sdf` (191 MB), `targets/ligprepchembl.sdf` (224 MB)

### 20 protein targets
```
Kinases (5):           ROCK2, CDK9, JAK1, ACVR1, AKT1
Non-kinase Enzymes (4): EZH2, PRMT5, MMP8, WRN
GPCRs (4):             GCGR, 5HT2A, DRD2, AGTR1
Nuclear Receptors (3): LXRB, FXR, AR
PPI Targets (4):       BCL2, BRD4, Keap1, EED
```

### Training set
- `crossdock_train_smi_with_rascore.csv` — CrossDocked2020 training SMILES + pre-computed RAScore column
- Used as the Tanimoto baseline for all models *except* PocketXMol

---

## 3. Notebook Purposes (one sentence each)

| Notebook | Purpose |
|----------|---------|
| `merge_molecules.ipynb` | Loads all 6 model SDF/TSV/SMI outputs across 20 targets, applies 8k/target cap (unlimited for PocketXMol), and writes `all_models_generation.csv`. |
| `process_smiles.ipynb` | Computes Morgan-FP Tanimoto similarity to training set, #Circles coverage (t=0.75, 3 runs), t-SNE/PCA projections, and physicochemical properties (REOS, PAINS, NPR, PBF). |
| `evaluation_level1.ipynb` | Evaluates Stage 1 (chemical space): per-model MaxTanimoto distributions and #Circles coverage scores, outputs raincloud + lollipop figures. |
| `evaluation_level2.ipynb` | Evaluates Stage 2 (chemical admissibility): computes 14 molecular descriptors (MW, logP, Fsp3, rings, SA, REOS, RAscore, MaxTanimoto) for ~968 k molecules, generates ridgeline plots and heatmap comparisons. |
| `evaluation_allmodel_level3.ipynb` | Evaluates Stage 3 (recovery) across all models: BM scaffold + generic scaffold recovery per target, heatmap and radar charts. |
| `evaluation_singlemodel_level3.ipynb` | Single-model version of Stage 3: exact/BM/generic recovery per target, outputs per-target CSV with recovered molecule details. |
| `chembl_patent_comparison.ipynb` | Reference analysis: compares ChEMBL vs patent chemical space (t-SNE, NPR/PBF shape triangle, property distributions, REOS/PAINS pass rates). |

---

## 4. Data Flow

```
MODEL RAW OUTPUTS (SDF / TSV / SMI)
         │
         ▼
 merge_molecules.ipynb
         │ all_models_generation.csv
         │ (SMILES, Model, Target)
         ▼
 evaluation_level2.ipynb
         │ all_models_properties.csv
         │ (968k rows, 14 descriptor columns)
         ▼
 evaluation_level1.ipynb   ──→  comparison_results/  (Stage 1 figures)
 evaluation_level2.ipynb   ──→  results_evaluation/comparison_results_level2/ (Stage 2)
 evaluation_allmodel_level3.ipynb ──→ results_evaluation/comparison_results_level3/ (Stage 3)

REFERENCE:
 targets/{T}/{T}_final.csv  ──→  Stage 3 recovery ground truth
 crossdock_train_smi_with_rascore.csv  ──→  Stage 1 Tanimoto baseline (all except PocketXMol)
```

---

## 5. Key Algorithm Parameters (verified in notebooks)

| Parameter | Value | Source |
|-----------|-------|--------|
| Morgan fingerprint radius | 2 | `AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)` |
| Fingerprint bit length | 2048 | same |
| #Circles Tanimoto threshold | 0.75 | `process_smiles.ipynb` |
| #Circles runs | 3 | averaged over 3 random permutations |
| Max molecules per target | 8 000 (PocketXMol: unlimited) | `merge_molecules.ipynb` |
| REOS rule set | Dundee | `useful_rdkit_utils.REOS(active_rules=['Dundee'])` |
| SA Score range | 1–10 (lower = easier) | `sascorer.calculateScore()` |
| Stereochemistry for exact recovery | **stripped** before comparison | `Chem.RemoveStereochemistry()` |
| Generic scaffold | BM scaffold → all heteroatoms→C, all bonds→single | `MurckoScaffold.MakeScaffoldGeneric()` |
| KDE bandwidth (ridge plots) | 0.15 | `bw_method=0.15` |
| t-SNE perplexity | 30 | sklearn default |

---

## 6. Python Dependencies (from all notebooks)

```
# Core scientific stack
pandas
numpy
scipy
matplotlib
seaborn
tqdm

# Cheminformatics
rdkit                  # Chem, AllChem, Descriptors, rdMolDescriptors,
                       # Scaffolds.MurckoScaffold, DataStructs
useful-rdkit-utils     # REOS filter (Dundee rule set)
sascorer               # SA Score (included with RDKit examples)

# Machine learning / dimensionality reduction
scikit-learn           # TSNE, PCA, StandardScaler
umap-learn             # UMAP projection (used in some figures)

# I/O and serialization
pyarrow                # parquet I/O (new dependency for unified format)
pyyaml                 # config files (new dependency)
openpyxl               # optional: Excel export

# Retrosynthesis scoring
rascore                # RAscore package (external; DO NOT reimplement)

# Additional
joblib                 # Parallel processing (part of sklearn)
```

---

## 7. Existing Outputs (do NOT delete)

| Path | Description |
|------|-------------|
| `results_evaluation/comparison_results_level2/all_models_properties.csv` | 968k-row master properties table |
| `results_evaluation/comparison_results_level2/single_model_results/` | Per-model cached property CSVs |
| `results_evaluation/comparison_results_level3/{MODEL}/evaluation_summary.csv` | Per-model Stage 3 recovery summaries |
| `comparison_results/` | Stage 1 figure data |

---

## 8. Target Package Structure (planned)

```
Generation_Benchmark/
├── README.md
├── pyproject.toml
├── PLANNING.md                          ← this file
├── data/
│   ├── benchmark/{target}.csv           ← from targets/{T}/{T}_final.csv
│   ├── training_set/crossdocked_smiles.csv
│   └── generated/unified/{model}.parquet
├── sbdd_benchmark/
│   ├── __init__.py
│   ├── io/loaders.py                    ← per-model readers + unify()
│   ├── stage1/exploration.py            ← training_similarity(), circles_coverage()
│   ├── stage2/
│   │   ├── filters.py                   ← reos_filter()
│   │   ├── synthesizability.py          ← sa_score(), rascore()
│   │   └── properties.py               ← compute_descriptors(), wasserstein_profile()
│   ├── stage3/recovery.py               ← exact/BM/generic recovery
│   ├── visualization/plots.py
│   └── pipeline.py
├── cli.py
├── configs/default_config.yaml
├── notebooks/                           ← cleaned copies of originals
└── scripts/unify_generated_molecules.py
```

---

---
## PLANNING.md v2 — Corrections & Resolved Questions (2026-04-23)

### Corrected: PocketXMol path structure (verified on disk)
Actual nested path:
```
PocketXMol/outputs_0/{TARGET}/tmp{ID}_pxm_{TIMESTAMP}/tmp{ID}_pxm_{TIMESTAMP}_SDF/*.sdf
```
There is also a plain `SDF/` dir at the same level (contains -all/-in/-out/-raw files — NOT output SDFs).
The `tmp{ID}_pxm_{TIMESTAMP}_SDF/` dir also has a `0_inputs/` subdir (pocket/input, not outputs).
Loader must glob only the `*_SDF/` sub-directory.

### Resolved: RAscore API
```python
from RAscore import RAscore_NN
scorer = RAscore_NN.RAScorerNN(
    model_path='<repo>/RAscore/RAscore/models/models/DNN_chembl_fcfp_counts/model.h5'
)
score = scorer.predict(smiles)  # returns float in [0,1]
# Invalid SMILES → returns -1 (convention used in existing CSV)
```

### Resolved: PocketXMol training set
File: `PocketXMol/all_smiles_merged.csv`, single column `smiles`, 85 434 molecules.
Used as Tanimoto baseline **only** for PocketXMol.

### Resolved: #Circles aggregation
Notebook `run_circles()` tracks all 3 run counts in `run_results` list.
**Package implements: mean ± std over 3 runs** (as requested by user for manuscript).
`circles_coverage()` returns `(mean, std, run_results)`.

### Resolved: Wasserstein normalization
From `evaluation_level2.ipynb` heatmap cell — exact logic:
1. Compute raw WD(model, patent) per descriptor per model.
2. Patent row = 0.0 (reference).
3. Per descriptor: `max_val = max WD across all non-Patent models`.
4. Normalize: `normalized_WD = raw_WD / max_val` (Patent excluded from normalization).
5. Heatmap shows models × descriptors, color encodes normalized WD.

### Resolved: Coverage efficiency denominator
From `process_smiles.ipynb` cell 10: `efficiency = circles_avg / num_sampled * 1000`
where `num_sampled` = number of **valid** molecules (those that yielded a fingerprint), NOT total generated.

### Resolved: Training set SMILES column name
- CrossDocked CSV: column `smiles` (lowercase)
- PocketXMol CSV: column `smiles` (lowercase)

### Resolved: Property column names (confirmed from all_models_properties.csv header)
`N_AliR, N_AroR, N_ChiA, N_HetA, N_RotB, N_BriA`

---
## 9. Open Questions / Decisions Needed Before Coding

1. **RAscore interface**: The notebooks load pre-computed RAscore from `crossdock_train_smi_with_rascore.csv` and `all_models_properties.csv`. The live `rascore()` function call pattern should be confirmed — does the package expose a simple `predict(smiles)` API?
2. **PocketXMol training set path**: `all_smiles_merged.csv` — where exactly is this file? It is referenced in Level 1 but needs to be located on disk.
3. **#Circles: best-of-3 vs mean-of-3**: The manuscript says "mean ± std over 3 seeds". The notebook code takes the *best* (max) of 3 runs. Need to confirm which is correct for the paper.
4. **Wasserstein normalization**: The heatmap normalizes per-descriptor across models (min-max). This logic needs to be captured exactly from `evaluation_level2.ipynb`.
5. **Coverage efficiency formula**: "#Circles per 1000 valid molecules" — confirm denominator is valid molecules only (not all generated).
