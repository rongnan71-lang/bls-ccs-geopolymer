"""NSGA-II 多目标优化: 在材料配方空间中搜索 (UCS最大化, 渗透率最小化, 成本最小化)。

纯numpy实现, 不依赖pymoo。目标函数通过训练好的RBF-BLS+集成模型预测。
"""
import numpy as np
from typing import Callable, List, Tuple


class NSGA2:
    def __init__(
        self,
        n_var: int,
        n_obj: int,
        bounds: np.ndarray,
        pop_size: int = 100,
        n_gen: int = 50,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        seed: int = 42,
        project_fn: Callable = None,
    ):
        self.n_var = n_var
        self.n_obj = n_obj
        self.bounds = bounds
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.rng = np.random.RandomState(seed)
        self.project_fn = project_fn

    def _project(self, ind):
        if self.project_fn is not None:
            return self.project_fn(ind)
        return ind

    def _init_pop(self) -> np.ndarray:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        pop = lo + self.rng.random((self.pop_size, self.n_var)) * (hi - lo)
        if self.project_fn is not None:
            pop = np.array([self._project(ind) for ind in pop])
        return pop

    def _evaluate(self, pop: np.ndarray, obj_fn: Callable) -> np.ndarray:
        return np.array([obj_fn(ind) for ind in pop])

    def _fast_non_dominated_sort(self, F: np.ndarray) -> List[List[int]]:
        n = len(F)
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        rank = np.zeros(n, dtype=int)
        fronts = [[]]
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    n_dom[p] += 1
            if n_dom[p] == 0:
                rank[p] = 0
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        fronts.pop()
        return fronts

    def _crowding_distance(self, F: np.ndarray, front: List[int]) -> np.ndarray:
        dist = np.zeros(len(front))
        if len(front) <= 2:
            return np.full(len(front), np.inf)
        f_arr = F[front]
        for m in range(self.n_obj):
            order = np.argsort(f_arr[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            f_range = f_arr[order[-1], m] - f_arr[order[0], m]
            if f_range == 0:
                continue
            for k in range(1, len(front) - 1):
                dist[order[k]] += (
                    f_arr[order[k + 1], m] - f_arr[order[k - 1], m]
                ) / f_range
        return dist

    def _tournament_select(self, pop: np.ndarray, F: np.ndarray, fronts, k: int = 2):
        idx = self.rng.choice(len(pop), size=k, replace=False)
        best = idx[0]
        for i in idx[1:]:
            for fi, front in enumerate(fronts):
                if i in front:
                    ri = fi
                    break
            for fi, front in enumerate(fronts):
                if best in front:
                    rb = fi
                    break
            if ri < rb:
                best = i
        return pop[best].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        alpha = self.rng.random(self.n_var)
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = (1 - alpha) * p1 + alpha * p2
        return self._project(c1), self._project(c2)

    def _mutate(self, ind: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        mask = self.rng.random(self.n_var) < self.mutation_rate
        sigma = (hi - lo) * 0.1
        ind[mask] += self.rng.normal(0, 1, mask.sum()) * sigma[mask]
        ind = np.clip(ind, lo, hi)
        return self._project(ind)

    def run(self, obj_fn: Callable) -> Tuple[np.ndarray, np.ndarray]:
        pop = self._init_pop()
        F = self._evaluate(pop, obj_fn)
        for gen in range(self.n_gen):
            fronts = self._fast_non_dominated_sort(F)
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, F, fronts)
                p2 = self._tournament_select(pop, F, fronts)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_pop.extend([c1, c2])
            new_pop = np.array(new_pop[: self.pop_size])
            combined = np.vstack([pop, new_pop])
            combined_F = self._evaluate(combined, obj_fn)
            fronts = self._fast_non_dominated_sort(combined_F)
            next_pop = []
            next_F = []
            for front in fronts:
                if len(next_pop) + len(front) <= self.pop_size:
                    next_pop.extend(combined[front])
                    next_F.extend(combined_F[front])
                else:
                    dist = self._crowding_distance(combined_F, front)
                    order = np.argsort(-dist)
                    need = self.pop_size - len(next_pop)
                    for k in order[:need]:
                        next_pop.append(combined[front[k]])
                        next_F.append(combined_F[front[k]])
                    break
            pop = np.array(next_pop)
            F = np.array(next_F)
            if gen % 10 == 0:
                n_pareto = len(fronts[0]) if fronts else 0
                print(f"  gen {gen:3d}: pareto front size = {n_pareto}")
        return pop, F
