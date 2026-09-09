# -*- coding: utf-8 -*-
"""Prototype retention and sequential updating experiment.

A. CCS记忆层: 记录级身份保持 — 早期文献的记录在全部文献到达后是否仍可被检索。
B. BLS增量: 统计拟合漂移 — 早期数据的MAE在逐批加入新文献后如何变化。

注意: 本实验是retention/replay实验, 不是严格的online learning benchmark。
CCS的embedding normalization在全数据集上预先计算(用于统一检索空间),
BLS的normalization在每个fit/add_data时独立计算。两者度量不同维度的"遗忘"。
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bls import BroadLearningSystem
from ccs import MaterialMemory
from experiments.utils import load_xy

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "unified_mix.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "forgetting_report.txt")


def run_ccs_memory(df, noise=0.05):
    sources = df["source"].unique()
    first_source = sources[0]
    early_df = df[df["source"] == first_source]
    n_early = len(early_df)
    X_all, _, _ = load_xy(df, "ucs_kpa")
    mu = X_all.mean(axis=0)
    sd = X_all.std(axis=0) + 1e-8
    X_norm = (X_all - mu) / sd
    X_norm = X_norm / (np.linalg.norm(X_norm, axis=1, keepdims=True) + 1e-12)
    mem = MaterialMemory(dim=X_all.shape[1], similarity_threshold=0.995)
    early_ids = []
    for si, src in enumerate(sources):
        mask = df["source"] == src
        for i in np.where(mask)[0]:
            nid, merged, _ = mem.add(X_norm[i], source=src,
                                     group=str(df.iloc[i]["group"]),
                                     row_id=int(df.iloc[i].get("row_in_xlsx", i)))
            if src == first_source:
                early_ids.append(nid)
    rng = np.random.default_rng(42)
    hits = 0
    early_mask = df["source"] == first_source
    for i in np.where(early_mask)[0]:
        v = X_norm[i] + rng.normal(0, noise, X_norm.shape[1])
        v = v / (np.linalg.norm(v) + 1e-12)
        results = mem.query(v, top_k=1)
        if results and results[0]["id"] in early_ids:
            hits += 1
    return {"early_source": first_source[:50], "n_early": n_early,
            "total_nodes": mem.node_count, "total_inserted": mem.total_inserted,
            "hits": hits, "hit_rate": hits / n_early, "noise": noise}


def run_bls_drift(df, target="ucs_kpa"):
    sources = df["source"].unique()
    first_source = sources[0]
    early_mask = df["source"] == first_source
    X, y, _ = load_xy(df, target)
    y_early = y[early_mask]
    bls = BroadLearningSystem(n_features=X.shape[1], reg=1.0, rng_seed=42, buffer_size=500)
    bls.fit(X[early_mask], y[early_mask])
    mae_early = np.mean(np.abs(bls.predict(X[early_mask]) - y_early))
    for src in sources[1:]:
        mask = df["source"] == src
        bls.add_data(X[mask], y[mask])
    mae_late = np.mean(np.abs(bls.predict(X[early_mask]) - y_early))
    return {"mae_early": mae_early, "mae_late": mae_late,
            "drift": mae_late - mae_early,
            "drift_pct": (mae_late - mae_early) / (mae_early + 1e-12) * 100}


def main():
    if not os.path.exists(DATA_PATH):
        print(f"数据不存在: {DATA_PATH}")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    sources = df["source"].unique()
    print(f"数据集: {len(df)}行, {len(sources)}源")
    print(f"早期知识 = 首篇 [{sources[0][:50]}], {len(df[df['source']==sources[0]])}组")
    report = ["=== Prototype Retention and Sequential Updating Experiment ===",
              f"早期知识 = 首篇 [{sources[0][:50]}]",
              f"后续到达 {len(sources)-1} 篇, 共 {len(df)} 组",
              "注: 本实验为retention/replay实验, CCS embedding normalization在全数据集预先计算",
              ""]
    print("\nA. CCS记忆层...")
    res_a = run_ccs_memory(df)
    report += ["A. CCS记忆层 (身份保持 — 检索命中率):",
               f"   全部 {res_a['total_inserted']}组入图 ({res_a['total_nodes']}节点)后,",
               f"   早期 {res_a['n_early']}组带噪({int(res_a['noise']*100)}%)精确找回率 = "
               f"{res_a['hits']}/{res_a['n_early']} = {res_a['hit_rate']*100:.0f}%"]
    print(f"   找回率: {res_a['hits']}/{res_a['n_early']}={res_a['hit_rate']*100:.0f}%")
    print("\nB. BLS增量...")
    res_b = run_bls_drift(df, "ucs_kpa")
    report += ["", "B. 纯BLS增量 (统计拟合 — 早期数据MAE漂移):",
               f"   早期数据MAE: 学完时 {res_b['mae_early']:.1f}kPa → "
               f"全部到齐后 {res_b['mae_late']:.1f}kPa "
               f"(漂移 +{res_b['drift']:.1f}, +{res_b['drift_pct']:.0f}%)"]
    print(f"   MAE: {res_b['mae_early']:.1f}→{res_b['mae_late']:.1f}kPa")
    report += ["", "解读: CCS节点级记忆是'身份保持'(查得到原始记录);",
               "BLS读出层是'统计拟合'(新数据必然重分配权重)。",
               "两者度量不同维度的'遗忘', 不是替代关系 — 记忆层负责可追溯, 回归层负责插值。"]
    report_text = "\n".join(report)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告: {OUTPUT_PATH}")
    print(report_text)


if __name__ == "__main__":
    main()
