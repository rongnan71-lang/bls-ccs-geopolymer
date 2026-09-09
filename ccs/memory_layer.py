# -*- coding: utf-8 -*-
"""材料记忆层封装 — MaterialMemory (记录级可审计)."""
import os
from typing import Optional
import numpy as np
from .concept_space import ConceptSpace


class MaterialMemory:
    def __init__(self, dim, similarity_threshold=0.995, max_nodes=5000, rng_seed=42):
        self.cs = ConceptSpace(dim=dim, max_nodes=max_nodes,
                               similarity_threshold=similarity_threshold, rng_seed=rng_seed)
        self.dim = dim

    def add(self, mix_vector, source="", group="", row_id=-1, properties=None):
        v = np.asarray(mix_vector, dtype=np.float32).ravel()
        meta = {"source": source, "group": group, "row_id": int(row_id)}
        if properties:
            meta.update(properties)
        name = f"{source}_{group}_row{row_id}" if source else f"mix_{self.cs.total_inserted}"
        nid, merged = self.cs.add_concept(v, name=name, metadata=meta)
        node = self.cs.get_node(nid)
        return nid, merged, {"metadata": node.metadata, "records": node.records, "n_records": len(node.records)}

    def query(self, mix_vector, top_k=5, threshold=0.0):
        results = self.cs.find_similar(mix_vector, top_k=top_k, threshold=threshold)
        return [{"id": nid, "similarity": sim, **meta} for nid, sim, meta in results]

    def remove_source(self, source):
        return self.cs.remove_where(lambda r: r.get("source") == source)

    def remove_group(self, source, group):
        return self.cs.remove_where(
            lambda r: r.get("source") == source and r.get("group") == group)

    @property
    def node_count(self):
        return self.cs.node_count

    @property
    def total_inserted(self):
        return self.cs.total_inserted

    @property
    def total_records(self):
        return self.cs.total_records

    def sources(self):
        seen = set()
        for i in range(self.cs._next_id):
            if self.cs._alive[i]:
                for rec in (self.cs._records[i] or []):
                    s = rec.get("source", "")
                    if s:
                        seen.add(s)
        return sorted(seen)

    def source_counts(self):
        counts = {}
        for i in range(self.cs._next_id):
            if self.cs._alive[i]:
                for rec in (self.cs._records[i] or []):
                    s = rec.get("source", "unknown")
                    counts[s] = counts.get(s, 0) + 1
        return counts

    def save(self, path):
        self.cs.save_checkpoint(path)

    @classmethod
    def load(cls, path):
        cs = ConceptSpace.load_checkpoint(path)
        mem = cls(dim=cs.dim)
        mem.cs = cs
        return mem

    def stats(self):
        return {"nodes": self.cs.node_count,
                "total_inserted": self.cs.total_inserted,
                "total_records": self.cs.total_records,
                "sources": len(self.sources()),
                "source_list": self.sources()}
