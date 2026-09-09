# -*- coding: utf-8 -*-
"""实验公共工具 v1.3 — 物理交互特征 + 目标引导RBF原型嵌入 + GBDT+BLS残差校正."""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bls import BroadLearningSystem
from ccs import ConceptSpace

BINDER_COLS = ["f_cement", "f_slag", "f_steel_slag", "f_fly_ash", "f_lime", "f_mgo"]


def _load_xy_raw(df, target="ucs_kpa"):
    feat_cols = [c for c in df.columns if c.startswith("f_")]
    X = df[feat_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    idx = {c: i for i, c in enumerate(feat_cols)}
    binder = np.zeros(len(df))
    for c in BINDER_COLS:
        if c in idx:
            binder += X[:, idx[c]]
    bent = X[:, idx["f_bentonite"]] if "f_bentonite" in idx else np.zeros(len(df))
    if "f_bentonite_pac" in idx:
        bent = bent + X[:, idx["f_bentonite_pac"]]
    water = X[:, idx["f_water"]] if "f_water" in idx else np.zeros(len(df))
    r_ggbs = df.get("r_ggbs_rate", pd.Series(np.zeros(len(df)))).values.astype(np.float64) / 100.0
    conf = df.get("conf_kpa", pd.Series(np.zeros(len(df)))).values.astype(np.float64) / 100.0
    has_binder = binder > 1e-6
    has_water = water > 1e-6
    water_binder = np.where(has_binder, water / (binder + 1e-8), 0.0)
    bent_binder = np.where(has_binder, bent / (binder + 1e-8), 0.0)
    solid_liquid = np.where(has_water, (binder + bent) / (water + 1e-8), 0.0)
    slag_ratio = np.where(has_binder, (X[:, idx["f_slag"]] if "f_slag" in idx else 0) / (binder + 1e-8), 0.0)
    cement_ratio = np.where(has_binder, (X[:, idx["f_cement"]] if "f_cement" in idx else 0) / (binder + 1e-8), 0.0)
    all_names = list(feat_cols) + ["binder", "bent", "r_ggbs", "conf",
                                    "water_binder", "bent_binder", "solid_liquid",
                                    "slag_ratio", "cement_ratio"]
    X = np.hstack([X, binder.reshape(-1, 1), bent.reshape(-1, 1),
                   r_ggbs.reshape(-1, 1), conf.reshape(-1, 1),
                   water_binder.reshape(-1, 1), bent_binder.reshape(-1, 1),
                   solid_liquid.reshape(-1, 1), slag_ratio.reshape(-1, 1),
                   cement_ratio.reshape(-1, 1)])
    y = np.nan_to_num(df[target].values.astype(np.float64), nan=0.0)
    return X, y, all_names


def load_xy(df, target="ucs_kpa"):
    X, y, feat_cols = _load_xy_raw(df, target)
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    return X, y, feat_cols


def load_xy_no_sparse(df, target="ucs_kpa", nan_threshold=80.0):
    feat_cols = [c for c in df.columns if c.startswith("f_")]
    X_raw = df[feat_cols].values.astype(np.float64)
    n = len(df)
    kept_feat_cols = []
    dropped_names = []
    for i, name in enumerate(feat_cols):
        nan_ratio = np.isnan(X_raw[:, i]).sum() / n * 100.0
        if nan_ratio > nan_threshold:
            dropped_names.append(f"{name}({nan_ratio:.1f}%)")
        else:
            kept_feat_cols.append(name)
    X = df[kept_feat_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    idx = {c: i for i, c in enumerate(kept_feat_cols)}
    binder = np.zeros(n)
    for c in BINDER_COLS:
        if c in idx:
            binder += X[:, idx[c]]
    bent = X[:, idx["f_bentonite"]] if "f_bentonite" in idx else np.zeros(n)
    if "f_bentonite_pac" in idx:
        bent = bent + X[:, idx["f_bentonite_pac"]]
    water = X[:, idx["f_water"]] if "f_water" in idx else np.zeros(n)
    r_ggbs = df.get("r_ggbs_rate", pd.Series(np.zeros(n))).values.astype(np.float64) / 100.0
    conf = df.get("conf_kpa", pd.Series(np.zeros(n))).values.astype(np.float64) / 100.0
    has_binder = binder > 1e-6
    has_water = water > 1e-6
    water_binder = np.where(has_binder, water / (binder + 1e-8), 0.0)
    bent_binder = np.where(has_binder, bent / (binder + 1e-8), 0.0)
    solid_liquid = np.where(has_water, (binder + bent) / (water + 1e-8), 0.0)
    slag_ratio = np.where(has_binder, (X[:, idx["f_slag"]] if "f_slag" in idx else 0) / (binder + 1e-8), 0.0)
    cement_ratio = np.where(has_binder, (X[:, idx["f_cement"]] if "f_cement" in idx else 0) / (binder + 1e-8), 0.0)
    extra_cols = []
    extra_data = []
    for name, data in [("r_ggbs", r_ggbs), ("conf", conf)]:
        nan_ratio = np.isnan(data).sum() / n * 100.0
        if nan_ratio > nan_threshold:
            dropped_names.append(f"{name}({nan_ratio:.1f}%)")
        else:
            extra_cols.append(name)
            extra_data.append(np.nan_to_num(data, nan=0.0))
    all_names = list(kept_feat_cols) + ["binder", "bent"] + extra_cols + \
                ["water_binder", "bent_binder", "solid_liquid", "slag_ratio", "cement_ratio"]
    X = np.hstack([X, binder.reshape(-1, 1), bent.reshape(-1, 1)] +
                  [d.reshape(-1, 1) for d in extra_data] +
                  [water_binder.reshape(-1, 1), bent_binder.reshape(-1, 1),
                   solid_liquid.reshape(-1, 1), slag_ratio.reshape(-1, 1),
                   cement_ratio.reshape(-1, 1)])
    y = np.nan_to_num(df[target].values.astype(np.float64), nan=0.0)
    return X, y, all_names, dropped_names


def make_rbf_graph_feature_fn(X_train, y_train, n_prototypes=25, sigma_scale=1.0):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-8
    Xn = (X_train - mu) / sd
    n = len(X_train)
    n_bins = min(n_prototypes, 10)
    base_per_bin = n_prototypes // n_bins
    remainder = n_prototypes % n_bins
    n_per_bin = [base_per_bin + (1 if i < remainder else 0) for i in range(n_bins)]
    quantiles = np.quantile(y_train, np.linspace(0, 1, n_bins + 1))
    proto_idx = []
    rng = np.random.default_rng(42)
    for b in range(n_bins):
        mask = (y_train >= quantiles[b]) & (y_train <= quantiles[b + 1])
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            candidates = np.where(np.abs(y_train - (quantiles[b] + quantiles[b+1])/2) < np.std(y_train))[0]
        if len(candidates) > 0:
            chosen = rng.choice(candidates, size=min(n_per_bin[b], len(candidates)), replace=False)
            proto_idx.extend(chosen.tolist())
    if len(proto_idx) < n_prototypes:
        remaining = [i for i in range(n) if i not in set(proto_idx)]
        need = n_prototypes - len(proto_idx)
        if remaining and need > 0:
            extra = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
            proto_idx.extend(extra.tolist())
    proto_idx = proto_idx[:n_prototypes]
    prototypes = Xn[proto_idx]
    if len(prototypes) > 1:
        dists = []
        for i in range(len(prototypes)):
            for j in range(i+1, len(prototypes)):
                dists.append(np.linalg.norm(prototypes[i] - prototypes[j]))
        sigma = np.median(dists) * sigma_scale if dists else 1.0
    else:
        sigma = 1.0
    sigma = max(sigma, 0.1)
    def feature_fn(X):
        Xs = (X - mu) / sd
        diff = Xs[:, None, :] - prototypes[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        return np.exp(-dist_sq / (2 * sigma ** 2))
    feature_fn.proto_idx = proto_idx
    return feature_fn, len(prototypes), sigma


def make_unsupervised_rbf_fn(X_train, n_prototypes=25, sigma_scale=1.0):
    from sklearn.cluster import KMeans
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-8
    Xn = (X_train - mu) / sd
    n_clusters = min(n_prototypes, len(X_train))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(Xn)
    prototypes = km.cluster_centers_
    if len(prototypes) > 1:
        dists = []
        for i in range(len(prototypes)):
            for j in range(i+1, len(prototypes)):
                dists.append(np.linalg.norm(prototypes[i] - prototypes[j]))
        sigma = np.median(dists) * sigma_scale if dists else 1.0
    else:
        sigma = 1.0
    def feature_fn(X):
        Xs = (X - mu) / sd
        diff = Xs[:, None, :] - prototypes[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        return np.exp(-dist_sq / (2 * sigma ** 2))
    return feature_fn, len(prototypes), sigma


def load_xy_no_physical(df, target="ucs_kpa"):
    feat_cols = [c for c in df.columns if c.startswith("f_")]
    X = df[feat_cols].values.astype(np.float64)
    idx = {c: i for i, c in enumerate(feat_cols)}
    binder = np.zeros(len(df))
    for c in BINDER_COLS:
        if c in idx:
            binder += X[:, idx[c]]
    bent = X[:, idx["f_bentonite"]] if "f_bentonite" in idx else np.zeros(len(df))
    if "f_bentonite_pac" in idx:
        bent = bent + X[:, idx["f_bentonite_pac"]]
    r_ggbs = df.get("r_ggbs_rate", pd.Series(np.zeros(len(df)))).values.astype(np.float64) / 100.0
    conf = df.get("conf_kpa", pd.Series(np.zeros(len(df)))).values.astype(np.float64) / 100.0
    X = np.hstack([X, binder.reshape(-1, 1), bent.reshape(-1, 1),
                   r_ggbs.reshape(-1, 1), conf.reshape(-1, 1)])
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    y = np.nan_to_num(df[target].values.astype(np.float64), nan=0.0)
    return X, y, feat_cols


def inner_cv_select(X_train, y_train, model_fn, param_grid, n_splits=3):
    best_p, best_s = None, -np.inf
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for p in param_grid:
        scores = []
        for tr, va in kf.split(X_train):
            m = model_fn(p)
            m.fit(X_train[tr], y_train[tr])
            scores.append(m.score(X_train[va], y_train[va]))
        if np.mean(scores) > best_s:
            best_s, best_p = np.mean(scores), p
    return best_p, best_s


def make_models(X_train, y_train):
    models = {}
    reg_candidates = [0.01, 0.1, 1.0, 10.0, 100.0]
    y_mean = y_train.mean()
    models["Dummy"] = lambda X: np.full(len(X), y_mean)
    best_reg, _ = inner_cv_select(X_train, y_train, lambda r: Ridge(alpha=r), reg_candidates)
    ridge = Ridge(alpha=best_reg).fit(X_train, y_train)
    models["岭回归"] = ridge.predict
    gbdt = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)
    gbdt.fit(X_train, y_train)
    models["GBDT"] = gbdt.predict
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=42, normalize_y=True)
    gp.fit(X_train, y_train)
    models["GP"] = gp.predict
    best_reg, _ = inner_cv_select(X_train, y_train, lambda r: BroadLearningSystem(n_features=X_train.shape[1], reg=r, rng_seed=42), reg_candidates)
    bls = BroadLearningSystem(n_features=X_train.shape[1], reg=best_reg, rng_seed=42)
    bls.fit(X_train, y_train)
    models["纯BLS"] = bls.predict
    cos_fn, n_cos = _make_cos_graph_fn(X_train, threshold=0.7)
    best_reg, _ = inner_cv_select(X_train, y_train, lambda r: BroadLearningSystem(n_features=X_train.shape[1], reg=r, feature_fn=cos_fn, scale=1.0, rng_seed=42), reg_candidates)
    bls_cos = BroadLearningSystem(n_features=X_train.shape[1], reg=best_reg, feature_fn=cos_fn, scale=1.0, rng_seed=42)
    bls_cos.fit(X_train, y_train)
    models["图特征BLS"] = bls_cos.predict
    rbf_fn, n_rbf, sigma = make_rbf_graph_feature_fn(X_train, y_train, n_prototypes=25)
    best_reg, best_scale = 1.0, 5.0
    best_s = -np.inf
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for reg in reg_candidates:
        for scale in [1.0, 5.0, 20.0]:
            scores = []
            for tr, va in kf.split(X_train):
                fn_tr, _, _ = make_rbf_graph_feature_fn(X_train[tr], y_train[tr], n_prototypes=25)
                m = BroadLearningSystem(n_features=X_train.shape[1], reg=reg, feature_fn=fn_tr, scale=scale, rng_seed=42)
                m.fit(X_train[tr], y_train[tr])
                scores.append(m.score(X_train[va], y_train[va]))
            if np.mean(scores) > best_s:
                best_s, best_reg, best_scale = np.mean(scores), reg, scale
    bls_rbf = BroadLearningSystem(n_features=X_train.shape[1], reg=best_reg, feature_fn=rbf_fn, scale=best_scale, rng_seed=42)
    bls_rbf.fit(X_train, y_train)
    models["RBF图增强BLS"] = bls_rbf.predict
    in_sample_residual = y_train - gbdt.predict(X_train)
    res_rbf_fn, n_res_proto, res_sigma = make_rbf_graph_feature_fn(X_train, in_sample_residual, n_prototypes=25)
    best_res_reg = 1.0
    best_shrink = 0.3
    best_s = -np.inf
    kf_sh = KFold(n_splits=3, shuffle=True, random_state=42)
    for shrink in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        scores = []
        for tr, va in kf_sh.split(X_train):
            g = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)
            g.fit(X_train[tr], y_train[tr])
            res_tr_inner = y_train[tr] - g.predict(X_train[tr])
            fn_tr, _, _ = make_rbf_graph_feature_fn(X_train[tr], res_tr_inner, n_prototypes=25)
            b = BroadLearningSystem(n_features=X_train.shape[1], n_enhance=300, reg=best_res_reg, feature_fn=fn_tr, scale=best_scale, rng_seed=42)
            b.fit(X_train[tr], res_tr_inner)
            pred = g.predict(X_train[va]) + shrink * b.predict(X_train[va])
            ss_res = np.sum((y_train[va] - pred)**2)
            ss_tot = np.sum((y_train[va] - y_train[va].mean())**2)
            scores.append(1 - ss_res / (ss_tot + 1e-12))
        if np.mean(scores) > best_s:
            best_s, best_shrink = np.mean(scores), shrink
    bls_res = BroadLearningSystem(n_features=X_train.shape[1], n_enhance=300, reg=best_res_reg, feature_fn=res_rbf_fn, scale=best_scale, rng_seed=42)
    bls_res.fit(X_train, in_sample_residual)
    models["GBDT+BLS残差校正"] = lambda X, s=best_shrink: gbdt.predict(X) + s * bls_res.predict(X)
    kf_st = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(X_train), 3))
    for tr, va in kf_st.split(X_train):
        fn_tr, _, _ = make_rbf_graph_feature_fn(X_train[tr], y_train[tr], n_prototypes=25)
        m1 = BroadLearningSystem(n_features=X_train.shape[1], n_enhance=300, reg=best_reg, feature_fn=fn_tr, scale=best_scale, rng_seed=42)
        m1.fit(X_train[tr], y_train[tr])
        m2 = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)
        m2.fit(X_train[tr], y_train[tr])
        m3 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, random_state=42, normalize_y=True)
        m3.fit(X_train[tr], y_train[tr])
        oof[va, 0] = m1.predict(X_train[va])
        oof[va, 1] = m2.predict(X_train[va])
        oof[va, 2] = m3.predict(X_train[va])
    best_meta_alpha = 1.0
    meta_learner = Ridge(alpha=best_meta_alpha).fit(oof, y_train)
    meta_coef = meta_learner.coef_
    meta_intercept = meta_learner.intercept_
    models["Stacking(BLS+GBDT+GP)"] = lambda X, mc=meta_coef, mi=meta_intercept: (mc[0] * bls_rbf.predict(X) + mc[1] * gbdt.predict(X) + mc[2] * gp.predict(X) + mi)
    return models, {"n_rbf_prototypes": n_rbf, "rbf_sigma": sigma, "rbf_reg": best_reg, "rbf_scale": best_scale, "residual_shrinkage": best_shrink, "residual_reg": best_res_reg, "stacking_coef": meta_coef.tolist(), "stacking_intercept": float(meta_intercept), "stacking_meta_alpha": best_meta_alpha}


def _make_cos_graph_fn(X_train, threshold=0.7):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-8
    Xn = (X_train - mu) / sd
    Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12)
    cs = ConceptSpace(dim=X_train.shape[1], similarity_threshold=threshold, max_nodes=1000)
    for v in Xn:
        cs.add_concept(v)
    protos = np.array([cs._mu[i] for i in range(cs._next_id) if cs._alive[i]])
    def fn(X):
        Xs = (X - mu) / sd
        Xs = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-12)
        return Xs @ protos.T
    return fn, cs.node_count


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def mae_score(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()))
