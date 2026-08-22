import argparse
import os
import random
import sys
import time

import numpy as np

from data_loader import load_data
from ga_engine import BusNetworkGA
import config
from config import GA_PARAMS, RESULTS_DIR
from config_override import apply_overrides
from deap import tools


def parse_args():
    p = argparse.ArgumentParser(description="NSGA-II 公交网络优化主程序")
    p.add_argument("--run-id", type=int, default=None,
                   help="运行编号（Friedman 检验用）；不传则单次实验模式")
    p.add_argument("--seed", type=int, default=None,
                   help="随机种子；不传则不固定")
    p.add_argument("--out-dir", type=str, default=None,
                   help="输出目录；不传则用 config.RESULTS_DIR")
    p.add_argument("--algo", type=str, default="NSGA2",
                   choices=["NSGA2", "LEHHA"],
                   help="算法标签（决定 run_summary 中的 algo 字段）")
    p.add_argument("--ops-dir", type=str, default=None,
                   help="LEHHA 精英算子目录；不传则用 src/temp/ 默认")
    # 消融实验模式：overall_only/visual_only/demand_only/default_only
    # 不传则使用 config.OPERATOR_SELECTION['ablation_mode'] 的默认值（None）
    p.add_argument("--ablation-mode", type=str, default=None,
                   choices=["overall_only", "visual_only",
                            "demand_only", "default_only"],
                   help="消融实验模式：只用某一类 LLM 算子（仅 LEHHA 有效）")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="覆盖 config 参数（可多次使用），"
                        "如 --set MODEL_CONSTRAINTS.DELTA_MAX=3.0")
    return p.parse_args()


def main():
    args = parse_args()

    # 种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # 消融模式：覆盖 config 全局配置，让 ga_engine 读到
    if args.ablation_mode is not None:
        config.OPERATOR_SELECTION['ablation_mode'] = args.ablation_mode
        config.OPERATOR_SELECTION['enabled'] = False  # 消融模式下禁用 LLM 动态权重
        print(f"\n⚠️  消融实验模式已启用：only {args.ablation_mode}")

    if args.set:                          # 超参数覆盖
        apply_overrides(args.set)

    # 输出目录覆盖
    out_dir = args.out_dir if args.out_dir else RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    # 加载数据
    edges, od, fixed_tasks, G, node_pos = load_data()
    ga = BusNetworkGA(G, od, fixed_tasks, node_pos)

    # 如果指定了 ops-dir，注入到 ga 实例（用于 LEHHA）
    if args.ops_dir:
        ga._ops_dir = args.ops_dir

    t0 = time.perf_counter()
    final_pop, log = ga.run(
        pop_size=GA_PARAMS['POP_SIZE'],
        n_gen=GA_PARAMS['NGEN'],
        out_dir=out_dir,
        algo_label=args.algo,
        run_id=args.run_id,
        seed=args.seed,
    )
    duration = time.perf_counter() - t0
    print(f"运行时间: {duration:.2f} 秒")

    # 4. 结果分析 (Pareto 前沿)
    front = tools.sortNondominated(final_pop, len(final_pop), first_front_only=True)[0]
    print(f"\n找到 {len(front)} 个 Pareto 最优解")
    if front:
        best_ind = front[0]
        z1, z2 = best_ind.fitness.values
        print(f"推荐方案: Z1(Avg Visual)={z1:.4f}, Z2(Demand)={z2:.0f}")


if __name__ == "__main__":
    main()
