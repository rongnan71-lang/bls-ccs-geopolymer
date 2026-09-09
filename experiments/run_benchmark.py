# -*- coding: utf-8 -*-
"""基准对照实验 v1.3 — 9模型 + paired显著性检验."""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.utils import load_xy, make_models, r2_score, mae_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "unified_mix.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "benchmark_report.txt")

MODEL_NAMES = ["Dummy", "岭回归", "GBDT", "GP", "纯BLS", "图特征BLS",
               "RBF图增强BLS", "GBDT+BLS残差校正", "Stacking(BLS+GBDT+GP)"]


def run_kfold(df, target, n_splits=5, n_seeds=3):
    X, y, _ = load_xy(df, target)
    r2_all = {m: [] for m in MODEL_NAMES}
    mae_all = {m: [] for m in MODEL_NAMES}
    fold_r2 = {m: [] for m in MODEL_NAMES}
    for seed in range(n_seeds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_idx, te_idx in kf.split(X):
            models, _ = make_models(X[tr_idx], y[tr_idx])
            for name in MODEL_NAMES:
                pred = models[name](X[te_idx])
                r2 = r2_score(y[te_idx], pred)
                mae = mae_score(y[te_idx], pred)
                r2_all[name].append(r2)
                mae_all[name].append(mae)
                fold_r2[name].append(r2)
    summary = {m: (np.mean(r2_all[m]), np.std(r2_all[m]),
                   np.mean(mae_all[m]), np.std(mae_all[m])) for m in MODEL_NAMES}
    return summary, fold_r2


def run_loso(df, target):
    X, y, _ = load_xy(df, target)
    groups = df["source"].values
    r2_all = {m: [] for m in MODEL_NAMES}
    per_source = {}
    unique_sources = np.unique(groups)
    gkf = GroupKFold(n_splits=len(unique_sources))
    for tr_idx, te_idx in gkf.split(X, y, groups):
        src = groups[te_idx[0]]
        if len(te_idx) < 2 or len(tr_idx) < 20:
            continue
        models, _ = make_models(X[tr_idx], y[tr_idx])
        src_r2 = {}
        for name in MODEL_NAMES:
            pred = models[name](X[te_idx])
            r2 = r2_score(y[te_idx], pred)
            r2_all[name].append(r2)
            src_r2[name] = r2
        per_source[src[:40]] = src_r2
    return {m: (np.mean(r2_all[m]), np.std(r2_all[m])) for m in MODEL_NAMES}, per_source


def paired_significance(fold_r2, model_a, model_b, label_a, label_b):
    a = np.array(fold_r2[model_a])
    b = np.array(fold_r2[model_b])
    diff = b - a
    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    ci95 = stats.t.interval(0.95, n-1, loc=mean_diff, scale=std_diff/np.sqrt(n))
    t_stat, t_p = stats.ttest_rel(b, a)
    try:
        w_stat, w_p = stats.wilcoxon(b, a)
    except ValueError:
        w_stat, w_p = float('nan'), float('nan')
    return {"n": n, "mean_diff": mean_diff, "std_diff": std_diff,
            "ci95_low": ci95[0], "ci95_high": ci95[1],
            "t_stat": t_stat, "t_p": t_p, "w_stat": w_stat, "w_p": w_p}


def main():
    if not os.path.exists(DATA_PATH):
        print(f"数据不存在: {DATA_PATH}")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    print(f"数据集: {len(df)}行, {df['source'].nunique()}源, 特征维度: {load_xy(df,'ucs_kpa')[0].shape[1]}")
    report = [f"=== 基准对照 v1.3 (n={len(df)}, 源={df['source'].nunique()}, 模型=9) ===",
              "协议: 5折×3种子 + LOSO; 折内拟合; in-sample residual; 原型基于残差; shrinkage内层CV; 种子固定", ""]
    for target, label in [("ucs_kpa", "UCS 抗压强度(kPa)"), ("neg_log10_k", "-log10(k) 渗透")]:
        print(f"\n--- {label} ---")
        report.append(f"--- 目标: {label} ---")
        kf_results, fold_r2 = run_kfold(df, target)
        report.append("[5折×3种子 R² / MAE]")
        for m in MODEL_NAMES:
            r2, r2s, mae, maes = kf_results[m]
            marker = " ★" if m in ["RBF图增强BLS", "GBDT+BLS残差校正", "Stacking(BLS+GBDT+GP)"] else ""
            report.append(f"  {m:<22} R²={r2:+.3f}±{r2s:.3f}  MAE={mae:.3f}±{maes:.3f}{marker}")
            print(f"  {m}: R²={r2:.3f}")
        sig = paired_significance(fold_r2, "GBDT", "GBDT+BLS残差校正", "GBDT", "GBDT+BLS")
        report.append("")
        report.append("[Paired显著性检验: GBDT+BLS残差校正 vs GBDT]")
        report.append(f"  ΔR²均值 = {sig['mean_diff']:+.4f} (95% CI: [{sig['ci95_low']:+.4f}, {sig['ci95_high']:+.4f}])")
        report.append(f"  Paired t-test: t={sig['t_stat']:.3f}, p={sig['t_p']:.4f}")
        report.append(f"  Wilcoxon signed-rank: p={sig['w_p']:.4f}")
        sig_label = "显著(p<0.05)" if sig['t_p'] < 0.05 else "不显著(p≥0.05)"
        report.append(f"  结论: {sig_label}")
        print(f"  ΔR²={sig['mean_diff']:+.4f}, t-test p={sig['t_p']:.4f}, Wilcoxon p={sig['w_p']:.4f}")
        loso_results, _ = run_loso(df, target)
        report.append("")
        report.append("[留一文献 LOSO R²]")
        for m in MODEL_NAMES:
            r2, r2s = loso_results[m]
            report.append(f"  {m:<22} R²={r2:+.3f}±{r2s:.3f}")
        report.append("")
    report_text = "\n".join(report)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告: {OUTPUT_PATH}")
    print(report_text)


if __name__ == "__main__":
    main()
