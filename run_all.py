"""一键运行全部实验: 基准 -> 消融 -> 抗遗忘 -> 图表 -> 优化。

用法:
    py -3.12 run_all.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

STEPS = [
    ("基准实验 (9模型 x 5折 x 3种子 + LOSO)", "experiments/run_benchmark.py"),
    ("消融实验 (6变体)", "experiments/run_ablation.py"),
    ("抗遗忘实验", "experiments/run_forgetting.py"),
    ("生成论文图表 (6张)", "experiments/make_figures.py"),
    ("NSGA-II多目标优化", "optimization/run_optimize.py"),
]


def main():
    print("=" * 60)
    print("BLS+CCS 纯净版 - 全流程实验")
    print("=" * 60 + "\n")

    for i, (name, script) in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] {name}")
        print("-" * 40)
        result = subprocess.run(
            [PY, str(ROOT / script)],
            cwd=ROOT,
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"[FAIL] {name} 返回码 {result.returncode}")
            sys.exit(1)
        print(f"[OK] {name}")

    print("\n" + "=" * 60)
    print("全部实验完成! 产物在 output/ 目录")
    print("=" * 60)


if __name__ == "__main__":
    main()
