# BLS + Prototype-Based Residual Learning for Geopolymer Performance Prediction

An experimental machine-learning pipeline combining gradient-boosted trees, target-informed RBF prototype features, residual correction, and broad learning, applied to geopolymer and soil-bentonite material performance prediction.

> "Concept Space (CCS)" is a project-specific working name, not a claim of a new memory architecture. See below.

---

## Method at a glance

1. Gradient-boosted trees provide the baseline prediction.
2. Training-fold samples are used to construct target-informed prototypes (25 prototypes, quantile-stratified with remainder distribution).
3. RBF features encode similarity to those prototypes (prototype similarities implicitly define a sample–prototype bipartite graph).
4. A broad-learning model predicts/corrects residual structure (prototype selection based on residual, shrinkage selected via inner CV, RBF scale fixed to the main RBF-BLS selected value).
5. A record-backed prototype store keeps associated sample information and supports record-level deletion with node statistic reconstruction.
6. Nested cross-validation is used for model evaluation (5-fold × 3 random partition seeds).

Physical interaction features (5 dimensionless ratios: water-binder, bentonite-binder, solid-liquid, slag-binder, cement-binder) are included as additional inputs.

## Concept Space (working name)

In this project, "Concept Space" (CCS) is a working name for a **record-backed prototype representation** used to organize samples in embedding space.

The term is **not intended to claim a new class of memory architecture**. Related ideas exist across prototype learning, clustering, metric learning, memory-augmented models, and online statistics.

What it actually does:
- Maintains prototype nodes with cosine-similarity-based retrieval
- Merges new samples into existing nodes when similarity exceeds a threshold
- Stores complete record metadata per node (source, group, row_id, embedding)
- Supports record-level deletion with node statistic reconstruction (standard Welford online variance)
- Serves as a provenance/retention layer complementary to the regression model

What it does not do:
- It is not a continual-learning benchmark for catastrophic forgetting
- It does not prevent the predictive model's parameters from drifting
- The "100% retention" result measures record identity retrievability, not prediction stability

## Existing components vs. project-specific combination

This project does **not** claim to invent the individual components used here. The implementation combines established techniques including:

- Gradient-boosted decision trees (GBDT)
- RBF/kernel feature mappings
- Prototype-based representations
- Ridge regression (closed-form)
- Broad learning systems (BLS)
- Nested cross-validation
- NSGA-II multi-objective optimization

The experimental contribution is primarily in **how these components are combined and evaluated** for the target geopolymer prediction task, including:
- Target-informed prototype selection integrated with residual correction
- Record-level provenance tracking in the prototype store
- Honest evaluation under random CV vs. leave-one-source-out (LOSO)
- Sparse-feature ablation with pre-defined threshold

## Key results

**5-fold CV repeated under 3 random partition seeds** (BLS enhancement-node seed fixed at 42; the three seeds change KFold splits, not BLS internal initialization).

| Target | GBDT baseline | This method | ΔR² | paired t p | Interpretation |
|--------|--------------|-------------|-----|-----------|----------------|
| UCS (kPa) | 0.822 | 0.825 | +0.002 | 0.598 | Positive trend, not statistically significant |
| Permeability −log₁₀(k) | 0.819 | 0.830 | +0.011 | 0.039 | Fold-level paired test supports improvement |

**Statistical note**: The 15 fold scores share training observations and are therefore **not independent replicates**. Reported p-values are exploratory rather than evidence from 15 independent experiments. The primary evidence is the ΔR² with fold-level confidence intervals.

**Stacking reference**: A 3-model stacking baseline achieves 0.825 (UCS) and 0.831 (permeability). The proposed method is competitive but does not uniformly outperform all baselines.

### Ablation (strict nested CV, prototypes re-selected in each inner fold)

| Variant | UCS R² | Permeability R² |
|---------|--------|-----------------|
| A1: no physical features, no RBF | 0.534 | 0.786 |
| A2: physical features, no RBF | 0.553 | 0.771 |
| A3: physical + unsupervised RBF (fixed) | 0.449 | 0.728 |
| A4: physical + supervised RBF (fixed) | 0.291 | 0.761 |
| A5: physical + unsupervised RBF + CV | 0.508 | 0.747 |
| A6: physical + supervised RBF + CV | 0.533 | 0.767 |

The main value of RBF prototype features comes from their combination with GBDT residual correction, not from standalone BLS prediction. Supervised prototype benefit is task-dependent.

### Sparse feature ablation (threshold 80% pre-defined, not tuned on results)

Removing 16 features with NaN ratio > 80% causes large performance drops:
- GBDT UCS: 0.822 → 0.553 (−0.269)
- GBDT permeability: 0.819 → 0.541 (−0.278)

Conclusion: "sparse ≠ useless" — the minority of samples with non-zero values carry key predictive information. This is a data-audit experiment; missing-rate statistics are computed on the full unified dataset (threshold does not depend on labels, so no target leakage).

### Prototype retention and sequential updating

This experiment evaluates two different behaviors:
- **Prototype records**: whether previously stored records remain retrievable → 100% (12/12 sources)
- **Predictive model**: how prediction error changes after sequential buffered re-solve updates → MAE drift +4% (7.0 → 7.3 kPa)

These should **not** be interpreted as measuring the same form of catastrophic forgetting.

### Leave-one-source-out (LOSO)

All models show negative R² with large source-level variance (UCS: −3.377 ± 4.306; permeability: −1.625 ± 1.968). This indicates that random-CV performance may overestimate cross-source deployability on this dataset (124 samples, 12 literature sources).

## Dataset

27 input features:
- 18 unified `f_` features: 15 raw composition/process variables + 3 aggregate/derived variables (`f_binder_cem_ggbs`, `f_slurry`, `f_binder_total`)
- 4 derived features (total binder, total bentonite, slag ratio, confining pressure)
- 5 physical interaction features (dimensionless ratios)

Targets: UCS compressive strength (kPa), permeability coefficient k (cm/s), −log₁₀(k).

Missing values: NaN in original `f_` columns means "this literature did not use that component" (dosage = 0), filled to 0 before computing derived features. Target columns have no missing values.

## Directory structure

```
纯净版/
├── bls.py                          # BLS core (closed-form ridge + sliding-window buffered re-solve)
├── ccs/
│   ├── __init__.py
│   ├── concept_space.py            # Concept Space (record-level merge/delete, Welford reconstruction)
│   └── memory_layer.py             # MaterialMemory wrapper
├── experiments/
│   ├── utils.py                    # data loading, physical features, RBF prototypes, model factory
│   ├── run_benchmark.py            # 9-model benchmark, 5-fold×3seed, LOSO, paired tests
│   ├── run_ablation.py             # staged ablation A1–A6, strict nested CV
│   ├── run_forgetting.py           # prototype retention + sequential update experiment
│   ├── run_sparse_ablation.py      # sparse feature removal ablation (27 vs 11 dims)
│   ├── make_figures.py             # paper figures fig1–fig6
│   └── make_sparse_figures.py      # sparse comparison figures fig7–fig8
├── optimization/
│   ├── nsga2.py                    # NSGA-II (pure numpy, feasible-region projection)
│   └── run_optimize.py             # Pareto search (nonnegative + sum constraint, derived vars recomputed)
├── data/
│   └── build_dataset.py            # dataset construction from raw xlsx
├── examples/
│   ├── bls_demo.py                 # BLS regression + incremental update demo
│   └── memory_demo.py              # CCS add/retrieve/record-delete/persistence demo
├── run_all.py                      # one-click: benchmark → ablation → retention → figures → optimization
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt

# Run all experiments (~15 min)
python run_all.py

# Or individually
python experiments/run_benchmark.py      # benchmark (~8 min)
python experiments/run_ablation.py       # ablation (~6 min, strict nested CV)
python experiments/run_forgetting.py     # prototype retention + sequential update
python experiments/make_figures.py       # generate fig1–fig6
python optimization/run_optimize.py      # NSGA-II Pareto search
```

## Reproducibility

All random seeds fixed (42). All preprocessing fitted within training folds only. A5/A6 ablation uses strict nested CV with prototype re-selection in each inner fold.

Results are reproducible under fixed dependency versions (numpy/scipy/sklearn/pandas/matplotlib) and random seeds. Cross-OS, cross-Python-version, or cross-dependency-version floating point results may differ slightly; image files are not guaranteed byte-identical.

## NSGA-II optimization note

The optimization searches over 5 controllable variables (`f_slag`, `f_water`, `f_bentonite`, `f_cement`, `f_steel_slag`) with the remaining 13 dimensions fixed at training-set means. Nonnegativity and a sum constraint are enforced via projection; derived aggregate variables are recomputed.

Results should be interpreted as **surrogate-space Pareto candidates in a restricted assumption space**, not globally optimal material formulations across the full composition space.

## Limitations

- The dataset is relatively small (124 samples, 12 literature sources).
- Prototype construction is target-informed and must therefore remain strictly inside the training portion of each evaluation split.
- The residual-correction stage may be sensitive to the choice of baseline model and regularization.
- Cross-validation folds are not independent statistical replicates.
- The proposed combination has not been established as a generally superior architecture outside this dataset/task.
- The "Concept Space" terminology is a project-specific working name, not a claim that the underlying operations constitute a new memory paradigm.
- NSGA-II optimization is restricted to 5 variables with the rest fixed; it does not explore the full material formulation space.
- In-sample residual (rather than OOF residual) is used for residual-BLS training; this is an empirical design choice, not a theoretically proven optimal strategy.

## License

CC BY-NC-SA 4.0 — share and adapt, non-commercial, same license, attribution. See [LICENSE](LICENSE).
