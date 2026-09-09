# -*- coding: utf-8 -*-
"""宽度学习系统 (BLS) — 封闭解岭回归 + 滑动窗口缓冲重解式增量更新.

增量更新机制: add_data() 将新数据加入滑动窗口缓冲区,
若总样本数超过 buffer_size 则丢弃最旧的数据, 然后用缓冲区中
所有数据重新求解岭回归闭式解。这不是递归矩阵逆更新, 而是
可控窗口大小的缓冲重解 (sliding-window buffered re-solve)。
"""
import numpy as np


class BroadLearningSystem:
    def __init__(self, n_features=10, n_enhance=200, reg=1e-3,
                 scale=1.0, rng_seed=42, feature_fn=None, buffer_size=50):
        self.n_features = n_features
        self.n_enhance = n_enhance
        self.reg = reg
        self.scale = scale
        self.rng = np.random.default_rng(rng_seed)
        self.feature_fn = feature_fn
        self.buffer_size = buffer_size
        self.We = None
        self.be = None
        self.W = None
        self._A_buf = None
        self._Y_buf = None
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None

    def _build_features(self, X, fit=False):
        if fit:
            self._x_mean = X.mean(axis=0)
            self._x_std = X.std(axis=0) + 1e-8
        Xs = (X - self._x_mean) / self._x_std
        feat = Xs
        if self.feature_fn is not None:
            extra = self.feature_fn(X)
            if extra.ndim == 1:
                extra = extra.reshape(-1, 1)
            feat = np.hstack([Xs, extra * self.scale])
        n = feat.shape[1]
        if fit:
            self.We = self.rng.normal(0, 1.0, size=(n, self.n_enhance)).astype(np.float32)
            self.be = self.rng.normal(0, 0.1, size=self.n_enhance).astype(np.float32)
        H = np.tanh(feat @ self.We + self.be)
        return np.hstack([np.ones((len(X), 1)), feat, H])

    def _solve(self, A, Y):
        n_col = A.shape[1]
        AtA = A.T @ A + self.reg * np.eye(n_col, dtype=A.dtype)
        self.W = np.linalg.solve(AtA, A.T @ Y)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        self._y_mean = y.mean()
        self._y_std = y.std() + 1e-8
        Y = (y - self._y_mean) / self._y_std
        A = self._build_features(X, fit=True)
        self._A_buf = A.copy()
        self._Y_buf = Y.copy()
        self._solve(A, Y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        A = self._build_features(X, fit=False)
        Y = A @ self.W
        return (Y * self._y_std + self._y_mean).ravel()

    def add_data(self, X_new, y_new):
        """滑动窗口缓冲重解式增量更新.

        将新数据加入缓冲区, 若总样本数超过 buffer_size 则丢弃最旧数据,
        然后用缓冲区中所有数据重新求解岭回归闭式解。
        """
        X_new = np.asarray(X_new, dtype=np.float64)
        y_new = np.asarray(y_new, dtype=np.float64).reshape(-1, 1)
        Y_new = (y_new - self._y_mean) / self._y_std
        A_new = self._build_features(X_new, fit=False)
        self._A_buf = np.vstack([self._A_buf, A_new])
        self._Y_buf = np.vstack([self._Y_buf, Y_new])
        # 滑动窗口截断: 只保留最近 buffer_size 条数据
        if len(self._A_buf) > self.buffer_size:
            self._A_buf = self._A_buf[-self.buffer_size:]
            self._Y_buf = self._Y_buf[-self.buffer_size:]
        self._solve(self._A_buf, self._Y_buf)
        return self

    def score(self, X, y):
        y = np.asarray(y, dtype=np.float64).ravel()
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / (ss_tot + 1e-12)
