# -*- coding: utf-8 -*-
"""
阶段1: 稀疏特征剔除消融实验
对比 27维全特征 vs 精简维(剔除NaN>80%特征) 的模型性能

红线: 阈值80%预先定义, 不得根据结果调整
      不得选择性报告有利结果
      两种特征集使用完全相同的评估协议(5折x3种子)
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from experiments.utils import load_xy, load_xy_no_sparse, make_models

def run_benchmark(X, y, n_splits=5, n_seeds=3, verbose=True):
    """运行基准实验: 9模型 x 5折 x 3种子"""
    models_tmp, _ = make_models(X, y)
    model_names = list(models_tmp.keys())
    n_models = len(model_names)
    
    all_scores = {name: [] for name in model_names}
    all_mae = {name: [] for name in model_names}
    
    for seed in range(n_seeds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            
            models, _ = make_models(X_tr, y_tr)
            for name in model_names:
                model = models[name]
                try:
                    pred = model(X_te)
                    r2 = r2_score(y_te, pred)
                    mae = mean_absolute_error(y_te, pred)
                except Exception as e:
                    r2 = np.nan
                    mae = np.nan
                all_scores[name].append(r2)
                all_mae[name].append(mae)
    
    results = {}
    for name in model_names:
        scores = np.array(all_scores[name])
        maes = np.array(all_mae[name])
        valid = ~np.isnan(scores)
        results[name] = {
            'r2_mean': np.mean(scores[valid]),
            'r2_std': np.std(scores[valid]),
            'mae_mean': np.mean(maes[valid]),
            'mae_std': np.std(maes[valid]),
            'n_valid': valid.sum()
        }
    return results

def main():
    df = pd.read_csv('output/unified_mix.csv')
    print("=" * 70)
    print("阶段1: 稀疏特征剔除消融实验")
    print("=" * 70)
    
    print("\n[1/3] 加载特征集...")
    X_full, y_ucs, names_full = load_xy(df, 'ucs_kpa')
    X_sparse, y_ucs2, names_kept, names_dropped = load_xy_no_sparse(df, 'ucs_kpa', nan_threshold=80.0)
    
    print(f"  全特征集: {X_full.shape[1]}维")
    print(f"  精简特征集: {X_sparse.shape[1]}维 (剔除{len(names_dropped)}个)")
    print(f"  剔除的特征: {', '.join(names_dropped)}")
    print(f"  保留的特征: {', '.join(names_kept)}")
    
    assert np.allclose(y_ucs, y_ucs2), "目标值不一致!"
    
    print("\n[2/3] 运行UCS基准实验 (全特征 vs 精简特征)...")
    print("  --- 全特征集 ---")
    res_full_ucs = run_benchmark(X_full, y_ucs, verbose=True)
    print("  --- 精简特征集 ---")
    res_sparse_ucs = run_benchmark(X_sparse, y_ucs, verbose=True)
    
    print("\n  运行渗透系数基准实验...")
    X_full_k, y_k, _ = load_xy(df, 'neg_log10_k')
    X_sparse_k, y_k2, _, _ = load_xy_no_sparse(df, 'neg_log10_k', nan_threshold=80.0)
    assert np.allclose(y_k, y_k2), "目标值不一致!"
    
    print("  --- 全特征集 ---")
    res_full_k = run_benchmark(X_full_k, y_k, verbose=True)
    print("  --- 精简特征集 ---")
    res_sparse_k = run_benchmark(X_sparse_k, y_k, verbose=True)
    
    print("\n" + "=" * 70)
    print("对比结果")
    print("=" * 70)
    
    print("\n--- UCS 抗压强度 ---")
    print(f"{'模型':<25} {'全特征R²':>12} {'精简R²':>12} {'ΔR²':>10} {'全MAE':>10} {'精简MAE':>10}")
    print("-" * 85)
    for name in res_full_ucs:
        r2_full = res_full_ucs[name]['r2_mean']
        r2_sparse = res_sparse_ucs[name]['r2_mean']
        delta = r2_sparse - r2_full
        mae_full = res_full_ucs[name]['mae_mean']
        mae_sparse = res_sparse_ucs[name]['mae_mean']
        marker = " ★" if delta > 0 else ""
        print(f"{name:<25} {r2_full:>12.3f} {r2_sparse:>12.3f} {delta:>+10.3f}{marker} {mae_full:>10.2f} {mae_sparse:>10.2f}")
    
    print("\n--- 渗透系数 -log10(k) ---")
    print(f"{'模型':<25} {'全特征R²':>12} {'精简R²':>12} {'ΔR²':>10} {'全MAE':>10} {'精简MAE':>10}")
    print("-" * 85)
    for name in res_full_k:
        r2_full = res_full_k[name]['r2_mean']
        r2_sparse = res_sparse_k[name]['r2_mean']
        delta = r2_sparse - r2_full
        mae_full = res_full_k[name]['mae_mean']
        mae_sparse = res_sparse_k[name]['mae_mean']
        marker = " ★" if delta > 0 else ""
        print(f"{name:<25} {r2_full:>12.3f} {r2_sparse:>12.3f} {delta:>+10.3f}{marker} {mae_full:>10.2f} {mae_sparse:>10.2f}")
    
    report = []
    report.append("=== 阶段1: 稀疏特征剔除消融实验报告 ===\n")
    report.append(f"全特征集: {X_full.shape[1]}维")
    report.append(f"精简特征集: {X_sparse.shape[1]}维 (剔除NaN>80%的{len(names_dropped)}个特征)")
    report.append(f"剔除特征: {', '.join(names_dropped)}\n")
    report.append("--- UCS 抗压强度 ---\n")
    for name in res_full_ucs:
        r2_full = res_full_ucs[name]['r2_mean']
        r2_sparse = res_sparse_ucs[name]['r2_mean']
        report.append(f"  {name}: 全特征={r2_full:.3f}, 精简={r2_sparse:.3f}, Δ={r2_sparse-r2_full:+.3f}")
    report.append("\n--- 渗透系数 ---\n")
    for name in res_full_k:
        r2_full = res_full_k[name]['r2_mean']
        r2_sparse = res_sparse_k[name]['r2_mean']
        report.append(f"  {name}: 全特征={r2_full:.3f}, 精简={r2_sparse:.3f}, Δ={r2_sparse-r2_full:+.3f}")
    
    with open('output/sparse_ablation_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"\n报告已保存: output/sparse_ablation_report.txt")

if __name__ == '__main__':
    main()
