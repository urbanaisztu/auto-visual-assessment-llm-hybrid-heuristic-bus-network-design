"""
实验结果统一落地器。
所有算法（NSGA-II/WSGATS/MOEAD/MOPSO/LEHHA）共用，保证 Friedman 检验输入格式一致。
"""
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd


def save_run_summary(
    algo: str,
    run_id: int,
    seed: int,
    summary: Dict[str, Any],
    out_dir: str,
    status: str = "success",
) -> str:
    """
    保存单次运行的汇总信息到 run_summary.json。

    :param algo: 算法名（LEHHA/NSGA2/WSGATS/MOEAD/MOPSO）
    :param run_id: 运行编号（1-10）
    :param seed: 随机种子
    :param summary: 包含 final_hv、duration_sec、evaluations 等字段的字典
    :param out_dir: 输出目录
    :param status: success / failed
    :return: 写入的文件路径
    """
    payload = {
        "algo": algo,
        "run_id": run_id,
        "seed": seed,
        "status": status,
        **summary,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "run_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def save_best_solutions(front, out_dir: str) -> str:
    """
    从帕累托前沿中提取 Z1 最优解和 Z2 最优解，保存为 best_solutions.csv。
    若 front 为空则写空文件并返回路径。

    :param front: DEAP 帕累托前沿（个体列表），每个个体有 .fitness.values=(z1, z2)
    :param out_dir: 输出目录
    :return: 写入的文件路径
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "best_solutions.csv")

    if not front:
        pd.DataFrame(columns=[
            "solution_type", "z1_visual", "z2_demand",
            "route_index_in_front"
        ]).to_csv(path, index=False)
        return path

    # 找到 Z1 最大和 Z2 最大的个体索引
    z1_values = [ind.fitness.values[0] for ind in front]
    z2_values = [ind.fitness.values[1] for ind in front]
    best_z1_idx = max(range(len(front)), key=lambda i: z1_values[i])
    best_z2_idx = max(range(len(front)), key=lambda i: z2_values[i])

    rows = []
    for sol_type, idx in [("best_z1", best_z1_idx), ("best_z2", best_z2_idx)]:
        rows.append({
            "solution_type": sol_type,
            "z1_visual": z1_values[idx],
            "z2_demand": z2_values[idx],
            "route_index_in_front": idx,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_final_pareto(front, out_dir: str) -> str:
    """
    保存最终帕累托前沿（含完整线路节点）为 final_pareto.csv。
    每行一个解，包含 index、z1、z2、routes（JSON 字符串）。
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "final_pareto.csv")
    rows = []
    for idx, ind in enumerate(front):
        routes = [[int(n) for n in route] for route in ind]
        rows.append({
            "index": idx,
            "z1_visual": ind.fitness.values[0],
            "z2_demand": ind.fitness.values[1],
            "routes_json": json.dumps(routes, ensure_ascii=False),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_best_solutions_from_records(records, out_dir: str) -> str:
    """
    从 records 列表提取 Z1 最优解和 Z2 最优解，保存为 best_solutions.csv。
    用于不依赖 DEAP 个体结构的基线算法。

    :param records: [(z1, z2, routes), ...] 列表
    :param out_dir: 输出目录
    :return: 写入的文件路径
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "best_solutions.csv")
    if not records:
        pd.DataFrame(columns=["solution_type", "z1_visual", "z2_demand",
                              "route_index_in_front"]).to_csv(path, index=False)
        return path
    z1s = [r[0] for r in records]
    z2s = [r[1] for r in records]
    best_z1_idx = max(range(len(records)), key=lambda i: z1s[i])
    best_z2_idx = max(range(len(records)), key=lambda i: z2s[i])
    rows = [
        {"solution_type": "best_z1", "z1_visual": z1s[best_z1_idx],
         "z2_demand": z2s[best_z1_idx], "route_index_in_front": best_z1_idx},
        {"solution_type": "best_z2", "z1_visual": z1s[best_z2_idx],
         "z2_demand": z2s[best_z2_idx], "route_index_in_front": best_z2_idx},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_final_pareto_from_records(records, out_dir: str) -> str:
    """
    从 records 列表保存最终帕累托前沿为 final_pareto.csv。
    用于不依赖 DEAP 个体结构的基线算法。

    :param records: [(z1, z2, routes), ...] 列表，routes 是 [[int, ...], ...]
    :param out_dir: 输出目录
    :return: 写入的文件路径
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "final_pareto.csv")
    rows = []
    for idx, (z1, z2, routes) in enumerate(records):
        rows.append({
            "index": idx, "z1_visual": z1, "z2_demand": z2,
            "routes_json": json.dumps([[int(n) for n in r] for r in routes],
                                       ensure_ascii=False),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_route_quality_from_records(records, out_dir: str, algo: str, run_id, G, node_positions) -> str:
    """
    从 records 列表计算路径质量（总长度、绕路系数）并保存为 route_quality.json。
    用于不依赖 DEAP 个体结构的基线算法。

    :param records: [(z1, z2, routes), ...] 列表，routes 是 [[int, ...], ...]
    :param out_dir: 输出目录
    :param algo: 算法名
    :param run_id: 运行编号
    :param G: NetworkX 图
    :param node_positions: {node_id: (lng, lat)}
    :return: 写入的文件路径；若无有效记录返回 None
    """
    from route_quality import evaluate_individual_quality

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "route_quality.json")

    quality_records = []
    for r in records:
        # records 元素支持 (z1, z2, routes) 或直接 routes 两种格式
        if isinstance(r, tuple) and len(r) >= 3:
            routes = r[2]
        elif isinstance(r, list):
            routes = r
        else:
            continue
        # 用每条线路的起终点构造临时 fixed_tasks
        fixed_tasks = [(-1, route[0], route[-1]) for route in routes if len(route) >= 2]
        try:
            q = evaluate_individual_quality(routes, G, node_positions, fixed_tasks)
            quality_records.append(q)
        except Exception:
            continue

    if not quality_records:
        return None

    n = len(quality_records)
    payload = {
        "algo": algo,
        "run_id": run_id,
        "optimized": {
            "pareto_front_size": n,
            "total_length_m_mean": sum(q["total_length_m"] for q in quality_records) / n,
            "avg_detour_ratio_mean": sum(q["avg_detour_ratio"] for q in quality_records) / n,
            "num_routes_mean": sum(q["num_routes"] for q in quality_records) / n,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
