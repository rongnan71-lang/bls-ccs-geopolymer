"""从原始Excel构建统一混合数据集.

用法:
    py -3.12 data/build_dataset.py --input ../data/raw --output output/unified_mix.csv

原始数据格式: 每个文献一个xlsx, 包含 source, group, 组分列(f_*), ucs_kpa, k_cm_s 等.
本脚本做最小清洗: 去重、缺失值标记、统一列名, 不做任何插补.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RAW_COMPONENT_COLS = [
    "f_slag", "f_water", "f_bentonite", "f_cement",
    "f_steel_slag", "f_lime", "f_soil", "f_fly_ash",
    "f_mgo", "f_bentonite_pac", "f_activator", "f_polymer_sebs",
    "f_apam", "f_silica_fume", "f_waste_binder",
]
DERIVED_F_COLS = [
    "f_binder_cem_ggbs",
    "f_slurry",
    "f_binder_total",
]
COMPONENT_COLS = RAW_COMPONENT_COLS + DERIVED_F_COLS

TARGET_COLS = ["ucs_kpa", "neg_log10_k", "k_cm_s", "strain_pct"]
META_COLS = ["source", "group", "row_in_xlsx"]
DERIVED_COLS = ["dosage_sum", "r_ggbs_rate", "conf_kpa"]


def build(input_dir: Path, output_path: Path) -> pd.DataFrame:
    frames = []
    for xlsx in sorted(input_dir.glob("*.xlsx")):
        try:
            df = pd.read_excel(xlsx)
        except Exception as e:
            print(f"[skip] {xlsx.name}: {e}", file=sys.stderr)
            continue
        if "source" not in df.columns:
            df["source"] = xlsx.stem
        if "row_in_xlsx" not in df.columns:
            df["row_in_xlsx"] = range(1, len(df) + 1)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No xlsx found in {input_dir}")

    combined = pd.concat(frames, ignore_index=True)

    rename_map = {}
    for c in combined.columns:
        cl = c.strip().lower()
        rename_map[c] = cl
    combined = combined.rename(columns=rename_map)

    known = META_COLS + COMPONENT_COLS + TARGET_COLS + DERIVED_COLS
    keep = [c for c in known if c in combined.columns]
    extra = [c for c in combined.columns
             if c not in keep and not c.startswith("unnamed")]
    combined = combined[keep + extra]

    before = len(combined)
    combined = combined.drop_duplicates(subset=["source", "row_in_xlsx"], keep="first")
    combined = combined.reset_index(drop=True)
    print(f"去重: {before} -> {len(combined)} 行")

    f_count = len([c for c in combined.columns if c.startswith("f_")])
    print(f"特征: {f_count}个f_列, {len(combined.columns)}总列, {combined['source'].nunique()}文献源")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已写入: {output_path} ({len(combined)}行, {len(combined.columns)}列)")
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("../data/raw"))
    parser.add_argument("--output", type=Path, default=Path("output/unified_mix.csv"))
    args = parser.parse_args()
    build(args.input, args.output)


if __name__ == "__main__":
    main()
