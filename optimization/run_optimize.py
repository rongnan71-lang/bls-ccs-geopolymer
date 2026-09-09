"""运行NSGA-II多目标优化, 搜索帕累托最优材料配方.

目标 (全部最小化):
  f1 = -UCS (最大化抗压强度)
  f2 = -neg_log10(k) (最大化防渗性能)
  f3 = cost (最小化成本, 简化为胶凝材料总用量)

用法:
    py -3.12 optimization/run_optimize.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.utils import load_xy, make_models
from optimization.nsga2 import NSGA2


def main():
    print("=== NSGA-II 多目标材料配方优化 ===\n")

    df = pd.read_csv("output/unified_mix.csv")
    X_ucs, y_ucs, feat_cols = load_xy(df, "ucs_kpa")
    X_k, y_k, _ = load_xy(df, "neg_log10_k")

    print(f"数据: {len(df)}行, {X_ucs.shape[1]}维特征")

    models, _ = make_models(X_ucs, y_ucs)
    ucs_predict = models["GBDT+BLS残差校正"]
    models_k, _ = make_models(X_k, y_k)
    k_predict = models_k["GBDT+BLS残差校正"]
    print("  训练完成\n")

    n_opt = 5
    opt_idx = list(range(n_opt))
    opt_names = [feat_cols[i] for i in opt_idx]
    print(f"优化变量: {opt_names}")

    lo = np.maximum(X_ucs[:, opt_idx].min(axis=0), 0.0)
    hi = X_ucs[:, opt_idx].max(axis=0)
    bounds = np.column_stack([lo, hi])
    target_sum = X_ucs[:, opt_idx].sum(axis=1).mean()
    print(f"物理约束: 前5组分目标和 = {target_sum:.3f}, 非负")

    def project_feasible(x):
        x = np.maximum(x, 0.0)
        s = x.sum()
        if s > 1e-8:
            x = x * (target_sum / s)
        return x

    cost_weights = np.array([0.05, 0.001, 0.45, 0.08, 0.03])

    x_mean = X_ucs.mean(axis=0)
    IDX_BINDER_CEM_GGBS = 15
    IDX_SLURRY = 16
    IDX_BINDER_TOTAL = 17
    IDX_BINDER_COMPONENTS = [5, 7, 8, 13, 14]

    def objective(x_raw):
        x = x_mean.copy()
        x[opt_idx] = x_raw
        x[IDX_BINDER_CEM_GGBS] = x[3] + x[0] + x[4]
        x[IDX_BINDER_TOTAL] = x[IDX_BINDER_CEM_GGBS] + sum(x[i] for i in IDX_BINDER_COMPONENTS)
        x[IDX_SLURRY] = x[IDX_BINDER_TOTAL] + x[1] + x[2]
        x = x.reshape(1, -1)
        ucs = ucs_predict(x)[0]
        k_log = k_predict(x)[0]
        cost = np.sum(x_raw * cost_weights)
        return np.array([-ucs, -k_log, cost])

    optimizer = NSGA2(
        n_var=n_opt,
        n_obj=3,
        bounds=bounds,
        pop_size=60,
        n_gen=30,
        seed=42,
        project_fn=project_feasible,
    )

    print("运行NSGA-II (30代)...")
    pop, F = optimizer.run(objective)

    fronts = optimizer._fast_non_dominated_sort(F)
    pareto_idx = fronts[0]
    pareto_pop = pop[pareto_idx]
    pareto_F = F[pareto_idx]

    print(f"\n帕累托前沿: {len(pareto_idx)} 个解")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    result_df = pd.DataFrame(pareto_pop, columns=opt_names)
    result_df["pred_ucs_kpa"] = -pareto_F[:, 0]
    result_df["pred_neg_log10_k"] = -pareto_F[:, 1]
    result_df["cost"] = pareto_F[:, 2]
    result_df.to_csv(out_dir / "pareto_solutions.csv", index=False, encoding="utf-8-sig")
    print(f"已保存: {out_dir / 'pareto_solutions.csv'}")

    print("\n--- 代表性帕累托解 ---")
    i_ucs = np.argmax(-pareto_F[:, 0])
    i_k = np.argmax(-pareto_F[:, 1])
    i_cost = np.argmin(pareto_F[:, 2])

    for label, i in [("最高UCS", i_ucs), ("最低渗透", i_k), ("最低成本", i_cost)]:
        print(f"  [{label}] UCS={-pareto_F[i,0]:.1f} kPa, "
              f"-log10(k)={-pareto_F[i,1]:.3f}, cost={pareto_F[i,2]:.3f}")
        for j, name in enumerate(opt_names):
            print(f"    {name}={pareto_pop[i,j]:.2f}")


if __name__ == "__main__":
    main()
