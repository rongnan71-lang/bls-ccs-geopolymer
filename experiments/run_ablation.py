# -*- coding: utf-8 -*-
"""消融实验 v2.1 — staged ablation design (分阶段消融, 非完整析因设计).

| Variant | 物理特征 | 原型选择   | Kernel | 超参CV | 回答的问题           |
|---------|---------|-----------|--------|--------|---------------------|
| A1      | ❌      | 无        | —      | 固定   | 基线(无物理特征)     |
| A2      | ✅      | 无        | —      | 固定   | 物理特征贡献(A2-A1)  |
| A3      | ✅      | 无监督RBF | RBF    | 固定   | RBF特征贡献(A3-A2)   |
| A4      | ✅      | 监督RBF   | RBF    | 固定   | 监督原型贡献(A4-A3)  |
| A5      | ✅      | 无监督RBF | RBF    | CV     | 超参优化(无监督)     |
| A6      | ✅      | 监督RBF   | RBF    | CV     | 完整模型(监督+CV)    |

注意: A5/A6的内层CV严格nested — 每个inner fold内只用inner train的y重新选prototype,
inner validation的y不参与prototype选择, 避免target-informed representation leakage.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bls import BroadLearningSystem
from experiments.utils import (
    load_xy, load_xy_no_physical,
    make_rbf_graph_feature_fn, make_unsupervised_rbf_fn,
    r2_score,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "unified_mix.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "ablation_report.txt")

FIXED_REG = 1.0
FIXED_SCALE = 5.0
CV_REGS = [0.01, 0.1, 1.0, 10.0, 100.0]
CV_SCALES = [1.0, 5.0, 20.0]


def _select_reg_scale_nested(X_tr, y_tr, build_feature_fn, use_cv):
    if not use_cv:
        return FIXED_REG, FIXED_SCALE
    best_reg, best_scale, best_s = FIXED_REG, FIXED_SCALE, -np.inf
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for reg in CV_REGS:
        for scale in CV_SCALES:
            scores = []
            for tr, va in kf.split(X_tr):
                fn_inner = build_feature_fn(X_tr[tr], y_tr[tr])
                m = BroadLearningSystem(n_features=X_tr.shape[1], n_enhance=300,
                                         reg=reg, feature_fn=fn_inner,
                                         scale=scale, rng_seed=42)
                m.fit(X_tr[tr], y_tr[tr])
                scores.append(m.score(X_tr[va], y_tr[va]))
            if np.mean(scores) > best_s:
                best_s, best_reg, best_scale = np.mean(scores), reg, scale
    return best_reg, best_scale


def eval_variant(X, y, variant, n_splits=5, n_seeds=3):
    r2s = []
    use_physical = variant in ("A2", "A3", "A4", "A5", "A6")
    use_rbf = variant in ("A3", "A4", "A5", "A6")
    supervised = variant in ("A4", "A6")
    use_cv = variant in ("A5", "A6")
    for seed in range(n_seeds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in kf.split(X):
            X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
            if not use_rbf:
                build_feature_fn = lambda X, y: None
                feature_fn = None
            elif supervised:
                build_feature_fn = lambda X, y: make_rbf_graph_feature_fn(X, y, n_prototypes=25)[0]
                feature_fn, _, _ = make_rbf_graph_feature_fn(X_tr, y_tr, n_prototypes=25)
            else:
                build_feature_fn = lambda X, y: make_unsupervised_rbf_fn(X, n_prototypes=25)[0]
                feature_fn, _, _ = make_unsupervised_rbf_fn(X_tr, n_prototypes=25)
            reg, scale = _select_reg_scale_nested(X_tr, y_tr, build_feature_fn, use_cv)
            m = BroadLearningSystem(n_features=X_tr.shape[1], n_enhance=300,
                                     reg=reg, feature_fn=feature_fn,
                                     scale=scale, rng_seed=42)
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)
            r2s.append(r2_score(y_te, pred))
    return np.mean(r2s), np.std(r2s)


def main():
    if not os.path.exists(DATA_PATH):
        print(f"数据不存在: {DATA_PATH}")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    variants = [
        ("A1", "无物理特征, 无RBF, 固定超参"),
        ("A2", "有物理特征, 无RBF, 固定超参"),
        ("A3", "有物理特征, 无监督RBF, 固定超参"),
        ("A4", "有物理特征, 监督RBF, 固定超参"),
        ("A5", "有物理特征, 无监督RBF, CV超参"),
        ("A6", "有物理特征, 监督RBF, CV超参"),
    ]
    report = ["=== 消融实验 v2.1 (staged ablation设计, 严格nested CV, 5折×3种子) ===", ""]
    report.append("设计: A1(无物理) → A2(+物理) → A3(+无监督RBF) → A4(+监督RBF) → A5/CV → A6/CV")
    report.append("")
    for target, label in [("ucs_kpa", "UCS 抗压强度"), ("neg_log10_k", "渗透系数 -log10(k)")]:
        print(f"\n--- {label} ---")
        report.append(f"--- {label} ---")
        X_nophys, y_nophys, _ = load_xy_no_physical(df, target)
        X_phys, y_phys, _ = load_xy(df, target)
        results = {}
        for vid, vdesc in variants:
            X_use = X_nophys if vid == "A1" else X_phys
            y_use = y_nophys if vid == "A1" else y_phys
            mean, std = eval_variant(X_use, y_use, vid)
            results[vid] = (mean, std)
            print(f"  {vid} ({vdesc}): R²={mean:.3f}±{std:.3f}")
            report.append(f"  {vid}  {vdesc:<35} R²={mean:+.3f}±{std:.3f}")
        report.append("")
        report.append("  增量贡献:")
        report.append(f"    物理特征 (A2-A1):       Δ={results['A2'][0]-results['A1'][0]:+.3f}")
        report.append(f"    RBF特征 (A3-A2):        Δ={results['A3'][0]-results['A2'][0]:+.3f}")
        report.append(f"    监督原型 (A4-A3):       Δ={results['A4'][0]-results['A3'][0]:+.3f}")
        report.append(f"    超参CV(监督) (A6-A4):   Δ={results['A6'][0]-results['A4'][0]:+.3f}")
        report.append(f"    监督vs无监督(CV) (A6-A5): Δ={results['A6'][0]-results['A5'][0]:+.3f}")
        report.append("")
    report_text = "\n".join(report)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告: {OUTPUT_PATH}")
    print(report_text)


if __name__ == "__main__":
    main()
