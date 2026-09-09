# -*- coding: utf-8 -*-
"""生成论文全部图表 v1.1."""
import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 11
rcParams["figure.dpi"] = 150

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "output")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

sys.path.insert(0, BASE)
from experiments.utils import load_xy, make_models, r2_score, make_rbf_graph_feature_fn
from bls import BroadLearningSystem
from ccs import MaterialMemory
from optimization.nsga2 import NSGA2

COLORS = {"blue": "#2c3e50", "red": "#e74c3c", "green": "#27ae60",
          "orange": "#f39c12", "purple": "#8e44ad", "cyan": "#16a085",
          "gray": "#7f8c8d", "darkblue": "#2980b9"}


def parse_benchmark():
    """从 benchmark_report.txt 解析数据."""
    path = os.path.join(OUT, "benchmark_report.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        text = f.read()
    data = {}
    for target in ["UCS 抗压强度", "-log10(k) 渗透"]:
        data[target] = {"kfold": {}, "loso": {}}
    current = None
    for line in text.split("\n"):
        if "UCS 抗压强度" in line:
            current = "UCS 抗压强度"
        elif "-log10(k) 渗透" in line:
            current = "-log10(k) 渗透"
        m = re.match(r"\s+(\S.+?)\s+R²=([+\-\d.]+)±([\d.]+)\s+MAE=([\d.]+)", line)
        if m and current:
            name = m.group(1).strip()
            data[current]["kfold"][name] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        m2 = re.match(r"\s+(\S.+?)\s+R²=([+\-\d.]+)±([\d.]+)", line)
        if m2 and "LOSO" in text.split(line)[0][-200:] and current:
            name = m2.group(1).strip()
            if name not in data[current]["kfold"]:
                data[current]["loso"][name] = (float(m2.group(2)), float(m2.group(3)))
    return data


def fig1_model_comparison():
    """图1: 模型性能对比 (5折 R² + LOSO R²)."""
    data = parse_benchmark()
    if data is None:
        print("跳过图1: benchmark_report.txt 不存在")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (target, title) in zip(axes, data.items()):
        kfold = title["kfold"]
        loso = title["loso"]
        models = list(kfold.keys())
        x = np.arange(len(models))
        w = 0.38
        r2_vals = [kfold[m][0] for m in models]
        r2_errs = [kfold[m][1] for m in models]
        loso_vals = [loso.get(m, (0, 0))[0] for m in models]
        ax.bar(x - w/2, r2_vals, w, yerr=r2_errs, label="5折×3种子",
                       color=COLORS["darkblue"], capsize=3, alpha=0.85)
        ax.bar(x + w/2, loso_vals, w, label="留一文献LOSO",
                       color=COLORS["red"], capsize=3, alpha=0.85)
        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("Stacking(BLS+GBDT+GP)", "Stacking").replace("RBF图增强BLS", "RBF-BLS").replace("GBDT+BLS残差校正", "GBDT+BLS")
                            for m in models], rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("R²")
        ax.set_title(title.split(" ")[0] if " " in title else title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_ylim(min(min(loso_vals)-2, -3), max(r2_vals)+0.15)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("图1  模型性能对比：分布内精度 vs 跨文献外推", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG, "fig1_model_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  图1: {path}")


def fig2_ablation():
    """图2: 消融实验对比."""
    path = os.path.join(OUT, "ablation_report.txt")
    if not os.path.exists(path):
        print("跳过图2: ablation_report.txt 不存在")
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    variants = []
    ucs_r2, k_r2 = [], []
    current = None
    for line in text.split("\n"):
        if "UCS" in line and "---" in line:
            current = "ucs"
        elif "渗透" in line and "---" in line:
            current = "k"
        m = re.match(r"\s*(A\d)\s+(.+?)\s+R²=([+\-\d.]+)\s*±\s*([\d.]+)", line)
        if m:
            vid, desc, r2, std = m.group(1), m.group(2).strip(), float(m.group(3)), float(m.group(4))
            if current == "ucs":
                variants.append((vid, desc))
                ucs_r2.append((r2, std))
            elif current == "k":
                k_r2.append((r2, std))
    if not variants:
        print("跳过图2: 无法解析消融数据")
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(variants))
    w = 0.38
    labels = [v[0] for v in variants]
    ax.bar(x - w/2, [r[0] for r in ucs_r2], w, yerr=[r[1] for r in ucs_r2],
           label="UCS 抗压强度", color=COLORS["darkblue"], capsize=3, alpha=0.85)
    ax.bar(x + w/2, [r[0] for r in k_r2], w, yerr=[r[1] for r in k_r2],
           label="-log10(k) 渗透", color=COLORS["green"], capsize=3, alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("R² (5折×3种子)")
    ax.set_title("图2  消融实验：图特征形态与物理交互特征的贡献", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for i, (vid, desc) in enumerate(variants):
        ax.annotate(desc, xy=(i, ax.get_ylim()[0]+0.02), fontsize=7,
                    ha="center", va="bottom", color=COLORS["gray"],
                    rotation=15)
    plt.tight_layout()
    path = os.path.join(FIG, "fig2_ablation.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  图2: {path}")


def fig3_forgetting():
    """图3: 抗遗忘实验 — CCS找回率 vs BLS MAE漂移."""
    df = pd.read_csv(os.path.join(OUT, "unified_mix.csv"))
    sources = df["source"].unique()
    first = sources[0]
    X, y, _ = load_xy(df, "ucs_kpa")
    mu, sd = X.mean(0), X.std(0)+1e-8
    Xn = (X - mu) / sd
    Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True)+1e-12)

    mem = MaterialMemory(dim=X.shape[1], similarity_threshold=0.995)
    early_mask = df["source"] == first
    early_ids = []
    hit_rates = []
    n_nodes = []
    rng = np.random.default_rng(42)
    for si, src in enumerate(sources):
        mask = df["source"] == src
        for i in np.where(mask)[0]:
            nid, _, _ = mem.add(Xn[i], source=src, group=str(df.iloc[i]["group"]),
                                row_id=int(df.iloc[i].get("row_in_xlsx", i)))
            if src == first:
                early_ids.append(nid)
        hits = 0
        for i in np.where(early_mask)[0]:
            v = Xn[i] + rng.normal(0, 0.05, X.shape[1])
            v = v / (np.linalg.norm(v)+1e-12)
            res = mem.query(v, top_k=1)
            if res and res[0]["id"] in early_ids:
                hits += 1
        hit_rates.append(hits / early_mask.sum())
        n_nodes.append(mem.node_count)

    bls = BroadLearningSystem(n_features=X.shape[1], reg=1.0, rng_seed=42)
    bls.fit(X[early_mask], y[early_mask])
    mae_history = [np.mean(np.abs(bls.predict(X[early_mask]) - y[early_mask]))]
    for src in sources[1:]:
        mask = df["source"] == src
        bls.add_data(X[mask], y[mask])
        mae_history.append(np.mean(np.abs(bls.predict(X[early_mask]) - y[early_mask])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x_axis = np.arange(1, len(sources)+1)
    ax1.plot(x_axis, [r*100 for r in hit_rates], "o-", color=COLORS["green"],
             linewidth=2, markersize=8, label="CCS早期知识找回率")
    ax1.set_xlabel("已到达文献数")
    ax1.set_ylabel("早期知识找回率 (%)")
    ax1.set_title("(a) CCS记忆层：身份保持", fontsize=12, fontweight="bold")
    ax1.set_ylim(70, 105)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=10)
    for i, (xi, r) in enumerate(zip(x_axis, hit_rates)):
        ax1.annotate(f"{r*100:.0f}%", xy=(xi, r*100), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)
    ax2.plot(x_axis, mae_history, "s-", color=COLORS["red"],
             linewidth=2, markersize=8, label="BLS早期数据MAE")
    ax2.set_xlabel("已到达文献数")
    ax2.set_ylabel("早期数据 MAE (kPa)")
    ax2.set_title("(b) BLS读出层：统计拟合漂移", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=10)
    for i, (xi, m) in enumerate(zip(x_axis, mae_history)):
        ax2.annotate(f"{m:.1f}", xy=(xi, m), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)
    fig.suptitle("图3  流式抗遗忘：CCS身份保持 vs BLS读出漂移（两者度量不同维度的遗忘）",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG, "fig3_forgetting.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  图3: {path}")


def fig4_pareto():
    """图4: NSGA-II 帕累托前沿."""
    df = pd.read_csv(os.path.join(OUT, "unified_mix.csv"))
    X, y_ucs, _ = load_xy(df, "ucs_kpa")
    _, y_k, _ = load_xy(df, "neg_log10_k")
    models, _ = make_models(X, y_ucs)
    predict_ucs = models["GBDT+BLS残差校正"]
    models_k, _ = make_models(X, y_k)
    predict_k = models_k["GBDT+BLS残差校正"]

    n_opt = 5
    opt_idx = list(range(n_opt))
    lo = np.maximum(X[:, opt_idx].min(axis=0), 0.0)
    hi = X[:, opt_idx].max(axis=0)
    bounds = np.column_stack([lo, hi])
    target_sum = X[:, opt_idx].sum(axis=1).mean()
    cost_weights = np.array([0.05, 0.001, 0.45, 0.08, 0.03])

    def project_feasible(x):
        x = np.maximum(x, 0.0)
        s = x.sum()
        if s > 1e-8:
            x = x * (target_sum / s)
        return x

    mu_x = X.mean(0)
    IDX_BCG = 15
    IDX_SLURRY = 16
    IDX_BT = 17
    IDX_BINDER_COMP = [5, 7, 8, 13, 14]

    def objective(x_raw):
        xq = mu_x.copy()
        xq[opt_idx] = x_raw
        xq[IDX_BCG] = xq[3] + xq[0] + xq[4]
        xq[IDX_BT] = xq[IDX_BCG] + sum(xq[i] for i in IDX_BINDER_COMP)
        xq[IDX_SLURRY] = xq[IDX_BT] + xq[1] + xq[2]
        xq = xq.reshape(1, -1)
        ucs = float(predict_ucs(xq)[0])
        logk = float(predict_k(xq)[0])
        cost = float(np.sum(x_raw * cost_weights))
        return np.array([-ucs, -logk, cost])

    optimizer = NSGA2(n_var=n_opt, n_obj=3, bounds=bounds,
                      pop_size=60, n_gen=30, seed=42,
                      project_fn=project_feasible)
    pop, F = optimizer.run(objective)
    fronts = optimizer._fast_non_dominated_sort(F)
    pareto_idx = fronts[0]
    pareto_F = F[pareto_idx]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    sc = ax.scatter(pareto_F[:, 2], -pareto_F[:, 0], c=-pareto_F[:, 1],
                    cmap="viridis", s=60, alpha=0.8, edgecolors="black", linewidth=0.5)
    ax.set_xlabel("成本 (带权相对值)", fontsize=12)
    ax.set_ylabel("预测 UCS (kPa)", fontsize=12)
    ax.set_title("图4  NSGA-II 帕累托前沿（物理可行约束，颜色=防渗性能 -log10(k)）", fontsize=12, fontweight="bold")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("-log10(k) 防渗性能", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG, "fig4_pareto.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  图4: {path}")


def fig5_concept_space():
    """图5: 概念空间 t-SNE 可视化."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("跳过图5: sklearn.manifold.TSNE 不可用")
        return
    df = pd.read_csv(os.path.join(OUT, "unified_mix.csv"))
    X, y, _ = load_xy(df, "ucs_kpa")
    mu, sd = X.mean(0), X.std(0)+1e-8
    Xn = (X - mu) / sd
    Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True)+1e-12)
    mem = MaterialMemory(dim=X.shape[1], similarity_threshold=0.995)
    for i in range(len(df)):
        mem.add(Xn[i], source=df.iloc[i]["source"], group=str(df.iloc[i]["group"]),
                row_id=int(df.iloc[i].get("row_in_xlsx", i)),
                properties={"ucs": float(y[i])})
    nodes = mem.cs.all_nodes()
    if len(nodes) < 5:
        print("跳过图5: 节点太少")
        return
    mus = np.array([n.mu for n in nodes])
    ucs_vals = []
    sources = []
    for n in nodes:
        rec_ucs = [r.get("ucs", 0) for r in n.records if "ucs" in r]
        ucs_vals.append(np.mean(rec_ucs) if rec_ucs else n.metadata.get("ucs", 0))
        rec_srcs = [r.get("source", "unknown") for r in n.records]
        sources.append(max(set(rec_srcs), key=rec_srcs.count)[:15] if rec_srcs else n.metadata.get("source", "unknown")[:15])
    ucs_vals = np.array(ucs_vals)

    perp = min(30, len(nodes)-1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca")
    coords = tsne.fit_transform(mus)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    unique_src = list(set(sources))
    cmap = plt.cm.tab20
    for si, src in enumerate(unique_src):
        mask = np.array([s == src for s in sources])
        ax1.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(si % 20)],
                    label=src, s=50, alpha=0.8, edgecolors="black", linewidth=0.3)
    ax1.set_title("(a) 按文献源着色", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=7, loc="best", ncol=2)
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    ax1.grid(alpha=0.3)
    sc = ax2.scatter(coords[:, 0], coords[:, 1], c=ucs_vals, cmap="RdYlGn_r",
                     s=60, alpha=0.8, edgecolors="black", linewidth=0.3)
    ax2.set_title("(b) 按 UCS 强度着色", fontsize=12, fontweight="bold")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax2, label="UCS (kPa)")
    fig.suptitle(f"图5  概念空间可视化（{len(nodes)}个节点，t-SNE 2D投影）",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG, "fig5_concept_space.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  图5: {path}")


def fig6_rbf_mechanism():
    """图6: RBF原型特征机制示意."""
    df = pd.read_csv(os.path.join(OUT, "unified_mix.csv"))
    X, y, feat_cols = load_xy(df, "ucs_kpa")
    full_cols = list(feat_cols) + ["binder", "bent", "r_ggbs", "conf",
                                     "water_binder", "bent_binder", "solid_liquid",
                                     "slag_ratio", "cement_ratio"]
    idx_wb = full_cols.index("water_binder")
    idx_bb = full_cols.index("bent_binder")
    f1, f2 = X[:, idx_wb], X[:, idx_bb]
    fn, n_proto, sigma = make_rbf_graph_feature_fn(X, y, n_prototypes=25)
    proto_idx = fn.proto_idx
    rbf_activations = fn(X)
    max_activation = rbf_activations.max(axis=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(f1, f2, c=max_activation, cmap="YlOrRd", s=40, alpha=0.7,
                     label="训练样本 (颜色=最大RBF activation)")
    ax.scatter(f1[proto_idx], f2[proto_idx], c="blue", s=150, marker="*",
               edgecolors="black", linewidth=1.0, label=f"监督式RBF原型 (n={n_proto}, 按UCS分位数分层)", zorder=5)
    ax.set_xlabel("水胶比 (物理交互特征)", fontsize=12)
    ax.set_ylabel("膨润土胶凝比 (物理交互特征)", fontsize=12)
    ax.set_title("图6  目标引导RBF原型机制：按UCS分位数分层选25个原型，RBF activation在27维标准化空间计算",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("最大RBF activation (27维标准化空间)", fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIG, "fig6_rbf_mechanism.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  图6: {path}")


def main():
    print("生成论文图表...")
    fig1_model_comparison()
    fig2_ablation()
    fig3_forgetting()
    fig4_pareto()
    fig5_concept_space()
    fig6_rbf_mechanism()
    print(f"\n全部图表已保存到: {FIG}")


if __name__ == "__main__":
    main()
