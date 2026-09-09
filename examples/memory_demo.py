"""CCS材料记忆层演示: 添加、检索、按来源删除、持久化。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ccs.memory_layer import MaterialMemory


def main():
    mem = MaterialMemory(dim=5, similarity_threshold=0.99)

    rng = np.random.RandomState(0)
    for source_id in ["paper_A", "paper_B", "paper_C"]:
        for i in range(20):
            vec = rng.normal(0, 1, 5)
            vec[0] = {"paper_A": 1, "paper_B": -1, "paper_C": 0}[source_id]
            mem.add(
                mix_vector=vec,
                source=source_id,
                group="exp_1",
                row_id=i,
                properties={"ucs_kpa": float(rng.uniform(100, 500))},
            )

    print(f"总节点数: {mem.node_count}")
    print(f"总记录数: {mem.total_records}")
    print(f"来源分布: {mem.source_counts()}")

    query = np.array([1.0, 0, 0, 0, 0])
    results = mem.query(query, top_k=3)
    print("\n最近邻检索 (query=[1,0,0,0,0]):")
    for r in results:
        meta = r.get("metadata", {})
        print(f"  id={r['id']}, sim={r['similarity']:.3f}, "
              f"source={meta.get('source','?')}, ucs={meta.get('ucs_kpa', 0):.1f}, "
              f"n_records={r.get('n_records', 0)}")

    n_rec_removed, n_node_removed = mem.remove_source("paper_B")
    print(f"\n删除 paper_B: 移除 {n_rec_removed} 条记录, {n_node_removed} 个节点")
    print(f"剩余节点: {mem.node_count}, 记录: {mem.total_records}, 来源: {mem.source_counts()}")

    path = Path("output/memory_demo")
    path.parent.mkdir(exist_ok=True)
    mem.save(path)
    print(f"\n已保存到 {path}.npz + {path}.json")

    mem2 = MaterialMemory.load(path)
    print(f"重新加载: {mem2.node_count} 节点, {mem2.total_records} 记录")


if __name__ == "__main__":
    main()
