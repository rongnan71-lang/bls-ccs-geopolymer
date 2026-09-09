# -*- coding: utf-8 -*-
"""Compact Concept Space — 材料信息学专用可审计图记忆层 (纯净版 v1.3).

v1.3 修复:
- 维护未归一化的 _mean_raw / _M2 (标准Welford), 检索时再对均值归一化
- 删除重建与在线更新使用同一套Welford统计口径, sigma2严格一致
- _mu 始终存储归一化后的均值, 仅用于余弦相似度检索
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Concept:
    id: int
    name: str
    mu: np.ndarray
    sigma2: np.ndarray
    count: int
    metadata: dict = field(default_factory=dict)
    records: list = field(default_factory=list)
    alive: bool = True


class ConceptSpace:
    def __init__(self, dim, max_nodes=5000, similarity_threshold=0.995, rng_seed=42):
        self.dim = dim
        self.max_nodes = max_nodes
        self.similarity_threshold = similarity_threshold
        self.rng = np.random.default_rng(rng_seed)
        self._mu = np.zeros((max_nodes, dim), dtype=np.float32)
        self._mean_raw = np.zeros((max_nodes, dim), dtype=np.float64)
        self._M2 = np.zeros((max_nodes, dim), dtype=np.float64)
        self._sigma2 = np.ones((max_nodes, dim), dtype=np.float32) * 0.5
        self._count = np.zeros(max_nodes, dtype=np.int32)
        self._alive = np.zeros(max_nodes, dtype=bool)
        self._names = [""] * max_nodes
        self._metadata = [None] * max_nodes
        self._records = [None] * max_nodes
        self._activation_count = np.zeros(max_nodes, dtype=np.int32)
        self._node_count = 0
        self._next_id = 0
        self._total_inserted = 0

    def add_concept(self, embedding, name=None, metadata=None):
        v = np.asarray(embedding, dtype=np.float32).ravel()
        if v.shape[0] != self.dim:
            raise ValueError(f"dim {v.shape[0]} != {self.dim}")
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            raise ValueError("zero embedding")
        v = v / norm
        self._total_inserted += 1
        record = dict(metadata) if metadata else {}
        record["_embedding"] = v.tolist()
        best_id, best_sim = self._best_match(v)
        if best_id is not None and best_sim >= self.similarity_threshold:
            self._update_gaussian(best_id, v)
            if self._records[best_id] is None:
                first_rec = dict(self._metadata[best_id] or {})
                if "_embedding" not in first_rec:
                    first_rec["_embedding"] = self._mu[best_id].tolist()
                self._records[best_id] = [first_rec]
            self._records[best_id].append(record)
            self._activation_count[best_id] += 1
            return best_id, True
        if self._node_count >= self.max_nodes:
            self._prune_least_active()
        nid = self._alloc_slot()
        self._mean_raw[nid] = v.astype(np.float64)
        self._M2[nid] = np.zeros(self.dim, dtype=np.float64)
        self._mu[nid] = v
        self._sigma2[nid] = np.ones(self.dim, dtype=np.float32) * 0.5
        self._count[nid] = 1
        self._alive[nid] = True
        self._names[nid] = name or f"concept_{nid}"
        self._metadata[nid] = record
        self._records[nid] = [record]
        self._activation_count[nid] = 1
        self._node_count += 1
        return nid, False

    def find_similar(self, embedding, top_k=5, threshold=0.0):
        v = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return []
        v = v / norm
        alive_idx = np.where(self._alive[:self._next_id])[0]
        if len(alive_idx) == 0:
            return []
        sims = self._mu[alive_idx] @ v
        order = np.argsort(-sims)[:top_k]
        results = []
        for pos in order:
            idx = alive_idx[pos]
            sim = float(sims[pos])
            if sim < threshold:
                break
            self._activation_count[idx] += 1
            results.append((int(idx), sim, {
                "metadata": dict(self._metadata[idx] or {}),
                "records": list(self._records[idx] or []),
                "n_records": len(self._records[idx] or []),
            }))
        return results

    def remove_concept(self, node_id):
        if node_id < 0 or node_id >= self._next_id or not self._alive[node_id]:
            return False
        self._alive[node_id] = False
        self._node_count -= 1
        return True

    def remove_where(self, predicate):
        removed_records = 0
        removed_nodes = 0
        for i in range(self._next_id):
            if not self._alive[i]:
                continue
            records = self._records[i] or []
            matched = [r for r in records if predicate(r)]
            remaining = [r for r in records if not predicate(r)]
            if matched:
                removed_records += len(matched)
                self._records[i] = remaining
                self._count[i] = len(remaining)
                if remaining:
                    self._metadata[i] = remaining[0]
                    self._rebuild_node_from_records(i, remaining)
                else:
                    self._alive[i] = False
                    self._node_count -= 1
                    removed_nodes += 1
        return removed_records, removed_nodes

    def _rebuild_node_from_records(self, node_id, records):
        embs = []
        for r in records:
            emb = r.get("_embedding")
            if emb is not None:
                embs.append(np.asarray(emb, dtype=np.float64))
        if not embs:
            return
        n = len(embs)
        mean = np.zeros(self.dim, dtype=np.float64)
        M2 = np.zeros(self.dim, dtype=np.float64)
        for i, x in enumerate(embs):
            count = i + 1
            delta = x - mean
            mean += delta / count
            delta2 = x - mean
            M2 += delta * delta2
        self._mean_raw[node_id] = mean
        self._M2[node_id] = M2
        self._count[node_id] = n
        norm = np.linalg.norm(mean)
        if norm > 1e-12:
            self._mu[node_id] = (mean / norm).astype(np.float32)
        else:
            self._mu[node_id] = mean.astype(np.float32)
        if n > 1:
            self._sigma2[node_id] = (M2 / (n - 1) + 1e-6).astype(np.float32)
        else:
            self._sigma2[node_id] = np.ones(self.dim, dtype=np.float32) * 0.5

    def _best_match(self, v):
        if self._node_count == 0:
            return None, 0.0
        alive_idx = np.where(self._alive[:self._next_id])[0]
        sims = self._mu[alive_idx] @ v
        pos = int(np.argmax(sims))
        return int(alive_idx[pos]), float(sims[pos])

    def _update_gaussian(self, nid, sample):
        self._count[nid] += 1
        count = int(self._count[nid])
        x = sample.astype(np.float64)
        delta = x - self._mean_raw[nid]
        self._mean_raw[nid] += delta / count
        delta2 = x - self._mean_raw[nid]
        self._M2[nid] += delta * delta2
        norm = np.linalg.norm(self._mean_raw[nid])
        if norm > 1e-12:
            self._mu[nid] = (self._mean_raw[nid] / norm).astype(np.float32)
        if count > 1:
            self._sigma2[nid] = (self._M2[nid] / (count - 1) + 1e-6).astype(np.float32)
        else:
            self._sigma2[nid] = np.ones(self.dim, dtype=np.float32) * 0.5

    def _alloc_slot(self):
        for i in range(self._next_id):
            if not self._alive[i]:
                return i
        nid = self._next_id
        self._next_id += 1
        return nid

    def _prune_least_active(self, n_prune=1):
        alive_idx = np.where(self._alive[:self._next_id])[0]
        if len(alive_idx) == 0:
            return
        counts = self._activation_count[alive_idx]
        order = np.argsort(counts)[:n_prune]
        for pos in order:
            self._alive[int(alive_idx[pos])] = False
            self._node_count -= 1

    @property
    def node_count(self):
        return self._node_count

    @property
    def total_inserted(self):
        return self._total_inserted

    @property
    def total_records(self):
        return sum(len(self._records[i] or [])
                   for i in range(self._next_id) if self._alive[i])

    def get_node(self, node_id):
        if node_id < 0 or node_id >= self._next_id or not self._alive[node_id]:
            return None
        return Concept(node_id, self._names[node_id], self._mu[node_id].copy(),
                       self._sigma2[node_id].copy(), int(self._count[node_id]),
                       dict(self._metadata[node_id] or {}),
                       list(self._records[node_id] or []), True)

    def all_nodes(self):
        return [self.get_node(i) for i in range(self._next_id) if self._alive[i]]

    def save_checkpoint(self, path, extra=None):
        base, _ = os.path.splitext(path)
        np.savez_compressed(base + ".npz",
            mu=self._mu[:self._next_id],
            mean_raw=self._mean_raw[:self._next_id],
            M2=self._M2[:self._next_id],
            sigma2=self._sigma2[:self._next_id],
            count=self._count[:self._next_id], alive=self._alive[:self._next_id],
            activation_count=self._activation_count[:self._next_id])
        meta = {"dim": self.dim, "max_nodes": self.max_nodes,
                "similarity_threshold": self.similarity_threshold,
                "node_count": self._node_count, "next_id": self._next_id,
                "total_inserted": self._total_inserted,
                "names": self._names[:self._next_id],
                "metadata": self._metadata[:self._next_id],
                "records": self._records[:self._next_id],
                "extra": extra or {}}
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_checkpoint(cls, path):
        base, _ = os.path.splitext(path)
        with open(base + ".json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        cs = cls(meta["dim"], meta["max_nodes"], meta["similarity_threshold"])
        data = np.load(base + ".npz", allow_pickle=True)
        n = meta["next_id"]
        cs._mu[:n] = data["mu"]
        if "mean_raw" in data:
            cs._mean_raw[:n] = data["mean_raw"]
        if "M2" in data:
            cs._M2[:n] = data["M2"]
        cs._sigma2[:n] = data["sigma2"]
        cs._count[:n] = data["count"]
        cs._alive[:n] = data["alive"]
        cs._activation_count[:n] = data["activation_count"]
        cs._names[:n] = meta["names"]
        cs._metadata[:n] = meta["metadata"]
        cs._records[:n] = meta.get("records", [None] * n)
        cs._node_count = meta["node_count"]
        cs._next_id = n
        cs._total_inserted = meta["total_inserted"]
        return cs

    def validate(self):
        alive_count = int(np.sum(self._alive[:self._next_id]))
        total_rec = sum(len(self._records[i] or [])
                        for i in range(self._next_id) if self._alive[i])
        return {"node_count_match": alive_count == self._node_count,
                "alive_nodes": alive_count, "next_id": self._next_id,
                "total_records": total_rec,
                "all_mu_normalized": bool(np.allclose(
                    np.linalg.norm(self._mu[:self._next_id][self._alive[:self._next_id]], axis=1),
                    1.0, atol=1e-5))}
