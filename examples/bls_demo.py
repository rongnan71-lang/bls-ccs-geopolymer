"""BLS基础用法演示: 合成数据上的回归与增量学习。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bls import BroadLearningSystem


def main():
    rng = np.random.RandomState(42)
    n_train, n_test, n_features = 200, 50, 10

    X_train = rng.normal(0, 1, (n_train, n_features))
    w_true = rng.normal(0, 1, n_features)
    y_train = X_train @ w_true + 0.1 * rng.normal(0, 1, n_train)

    X_test = rng.normal(0, 1, (n_test, n_features))
    y_test = X_test @ w_true + 0.1 * rng.normal(0, 1, n_test)

    bls = BroadLearningSystem(
        n_features=n_features,
        n_enhance=200,
        reg=1e-3,
        rng_seed=42,
    )
    bls.fit(X_train, y_train)

    pred = bls.predict(X_test)
    mae = np.mean(np.abs(pred - y_test))
    r2 = 1 - np.sum((y_test - pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
    print(f"初始训练: MAE={mae:.4f}, R2={r2:.4f}")

    X_new = rng.normal(0, 1, (50, n_features))
    y_new = X_new @ w_true + 0.1 * rng.normal(0, 1, 50)
    bls.add_data(X_new, y_new)

    pred2 = bls.predict(X_test)
    mae2 = np.mean(np.abs(pred2 - y_test))
    r2_2 = 1 - np.sum((y_test - pred2) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
    print(f"增量学习后: MAE={mae2:.4f}, R2={r2_2:.4f}")


if __name__ == "__main__":
    main()
