"""
MOEA/D: Multi-Objective Evolutionary Algorithm based on Decomposition
基于分解的多目标进化算法

参考论文：
Zhang, Q., & Li, H. (2007). "MOEA/D: A multiobjective evolutionary algorithm
based on decomposition" IEEE Transactions on Evolutionary Computation

作者：Baseline实现
日期：2026-01-07
"""

import sys
import os
import random
import json
import numpy as np
import networkx as nx
from deap import tools
from typing import List, Tuple, Dict
from scipy.spatial.distance import cdist

# 添加父目录到路径以导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import evaluate_individual
from config import GA_PARAMS, STRATEGIES, RESULTS_DIR
from utils import calculate_haversine_distance


def calculate_hypervolume(fitnesses, reference_point):
    """
    计算Hypervolume指标

    Args:
        fitnesses: 适应度列表 [(Z1, Z2), ...]
        reference_point: 参考点 (ref_Z1, ref_Z2)

    Returns:
        float: HV值
    """
    if not fitnesses:
        return 0.0

    ref_Z1, ref_Z2 = reference_point
    hv = 0.0

    # 对所有非支配解按Z1排序
    sorted_fit = sorted(fitnesses, key=lambda x: x[0])

    # 计算HV
    prev_Z2 = ref_Z2
    for Z1, Z2 in sorted_fit:
        if Z1 > ref_Z1 and Z2 > ref_Z2:
            width = Z1 - ref_Z1
            height = Z2 - prev_Z2
            hv += width * height
            prev_Z2 = Z2

    return hv


class MOEAD:
    """MOEA/D算法主类"""

    def __init__(self, G, od_df, fixed_tasks, node_positions):
        """
        初始化MOEA/D

        Args:
            G: NetworkX图
            od_df: OD矩阵
            fixed_tasks: 固定任务列表 [(line_id, start, end), ...]
            node_positions: 节点位置字典
        """
        self.G = G
        self.od_df = od_df
        self.fixed_tasks = fixed_tasks
        self.node_positions = node_positions

        # 算法参数
        self.pop_size = GA_PARAMS['POP_SIZE']  # 100
        self.ngen = GA_PARAMS['NGEN']  # 200
        self.neighborhood_size = 20  # T
        self.cxpb = GA_PARAMS['CXPB']
        self.mutpb = GA_PARAMS['MUTPB']

        # 权重向量和邻域
        self.weight_vectors = None
        self.neighborhoods = None


        # 理想点（z*，最大值）和 最差点（z_min，最小值）
        self.ideal_point = None  # Max values
        self.nadir_point = None  # Min values (用于归一化)

    def _generate_weight_vectors(self, n_weights, H=99):
        """
        生成均匀分布的权重向量（简单单纯形网格方法）

        Args:
            n_weights: 需要的权重向量数量
            H: 分割粒度

        Returns:
            list: 权重向量列表 [[λ1, λ2], ...]
        """
        weights = []
        for h1 in range(H + 1):
            for h2 in range(H + 1 - h1):
                λ1 = h1 / H
                λ2 = h2 / H
                if λ1 + λ2 > 0:  # 避免全零
                    λ = [λ1, λ2]
                    λ = np.array(λ) / np.sum(λ)  # 归一化
                    weights.append(λ.tolist())

        # 如果不够，随机补充；如果多了，截断
        if len(weights) < n_weights:
            while len(weights) < n_weights:
                w = np.random.rand(2)
                w = w / np.sum(w)
                weights.append(w.tolist())
        else:
            weights = weights[:n_weights]

        return weights

    def _compute_neighborhoods(self):
        """
        计算每个权重向量的邻域（基于欧氏距离）

        Returns:
            list: 邻域索引列表，每个元素是长度为T的索引列表
        """
        weights_array = np.array(self.weight_vectors)

        # 计算权重向量间的欧氏距离
        distances = cdist(weights_array, weights_array, metric='euclidean')

        neighborhoods = []
        for i in range(self.pop_size):
            # 获取距离第i个权重向量最近的T个
            dist_i = distances[i]
            # 排除自己
            dist_i[i] = np.inf
            # 找到最小的T个索引
            neighbors = np.argsort(dist_i)[:self.neighborhood_size].tolist()
            neighborhoods.append(neighbors)

        return neighborhoods

    def _init_individual(self):
        """
        初始化个体：使用mixed策略（复用ga_engine逻辑）
        """
        routes = []
        strategy_mode = STRATEGIES.get('INITIALIZATION', 'random')
        ratios = STRATEGIES.get('INIT_RATIOS', {'physical': 1.0})

        for lid, s, e in self.fixed_tasks:
            path = []

            if strategy_mode == 'mixed':
                rand_val = random.random()
                p_phys = ratios['physical']
                p_vis = ratios['physical'] + ratios['visual']

                if rand_val < p_phys:
                    path = self._get_path_by_strategy(s, e, 'physical')
                elif rand_val < p_vis:
                    path = self._get_path_by_strategy(s, e, 'visual')
                else:
                    path = self._get_path_by_strategy(s, e, 'demand')
            else:
                path = self._get_path_by_strategy(s, e, 'physical')

            if not path:
                path = self._get_path_by_strategy(s, e, 'physical')
            routes.append(path)

        return routes

    def _get_path_by_strategy(self, start, end, strategy='physical'):
        """根据策略生成路径"""
        try:
            if strategy == 'visual':
                return nx.shortest_path(self.G, start, end, weight='visual_cost')
            elif strategy == 'demand':
                return nx.shortest_path(self.G, start, end, weight='demand_cost')
            else:
                return nx.shortest_path(self.G, start, end, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _tchebycheff(self, fitness, weight, ideal_point, nadir_point):
        """
        带归一化的 Tchebycheff 分解
        Args:
            fitness: 当前解 (f1, f2)
            weight: 权重 (w1, w2)
            ideal_point: 参考最大值 (z1_max, z2_max)
            nadir_point: 参考最小值 (z1_min, z2_min)
        """
        f = np.array(fitness)
        w = np.array(weight)
        z_max = np.array(ideal_point)
        z_min = np.array(nadir_point)

        # 1. 防止分母为0 (刚开始可能 max == min)
        # 如果 max 和 min 极其接近，给一个默认范围 1.0，防止报错
        denominator = z_max - z_min
        denominator = np.where(denominator < 1e-6, 1.0, denominator)

        # 2. 归一化计算距离
        # 公式: w_i * |z_max - f_i| / (z_max - z_min)
        # 因为是最大化问题，理想点是 z_max，差距是 z_max - f
        normalized_diff = np.abs(z_max - f) / denominator

        # 3. 加权并取最大值
        # 避免权重为0
        w = np.where(w < 1e-6, 1e-6, w)
        weighted_diff = w * normalized_diff

        return np.max(weighted_diff)

    def _smart_mutate(self, individual):
        """智能变异：复用ga_engine的逻辑"""
        new_individual = [list(route) for route in individual]

        probs = STRATEGIES.get('MUTATION_PROBS',
                               {'visual': 0.33, 'demand': 0.33, 'smooth': 0.34})

        idx = random.randint(0, len(new_individual) - 1)
        route = new_individual[idx]

        if len(route) < 4:
            return new_individual

        try:
            cut_u = random.randint(0, len(route) - 3)
            cut_v = random.randint(cut_u + 2, len(route) - 1)
            u, v = route[cut_u], route[cut_v]

            dice = random.random()
            w = 'weight'
            if dice < probs['visual']:
                w = 'visual_cost'
            elif dice < probs['visual'] + probs['demand']:
                w = 'demand_cost'

            subpath = nx.shortest_path(self.G, u, v, weight=w)
            new_individual[idx] = self._repair_route(
                route[:cut_u] + subpath + route[cut_v + 1:]
            )
        except:
            pass

        return new_individual

    def _repair_route(self, route):
        """修复路径中的重复节点"""
        if not route:
            return route

        seen, new_route = {}, []
        for node in route:
            if node in seen:
                new_route = new_route[:seen[node] + 1]
            else:
                seen[node] = len(new_route)
                new_route.append(node)
        return new_route

    def _dominates(self, fitness1, fitness2):
        """
        判断fitness1是否支配fitness2

        Args:
            fitness1, fitness2: (Z1, Z2) 元组

        Returns:
            bool: True表示fitness1支配fitness2
        """
        return (fitness1[0] >= fitness2[0] and fitness1[1] >= fitness2[1] and
                (fitness1[0] > fitness2[0] or fitness1[1] > fitness2[1]))

    def _update_ep(self, ep, individual, fitness):
        """
        更新外部存档EP (修复版：增加去重和容量限制)

        Args:
            ep: 当前外部存档 [(individual, fitness), ...]
            individual: 新解
            fitness: 新解的适应度

        Returns:
            更新后的EP
        """
        # 1. 【关键修复】去重检查
        # 如果新解的适应度与EP中已有的解极其相似，则认为是重复解，直接跳过
        for _, fit_ep in ep:
            # 判断浮点数相等，使用微小误差 epsilon
            if abs(fit_ep[0] - fitness[0]) < 1e-6 and abs(fit_ep[1] - fitness[1]) < 1e-6:
                return ep

        # 2. 支配检查
        dominated = False
        for _, fit_ep in ep:
            if self._dominates(fit_ep, fitness):
                dominated = True
                break

        if not dominated:
            # 3. 移除被新解支配的旧解
            new_ep = []
            for ind_ep, fit_ep in ep:
                if not self._dominates(fitness, fit_ep):
                    new_ep.append((ind_ep, fit_ep))

            # 添加新解
            new_ep.append((individual, fitness))

            # 4. 【额外保险】容量限制 (Pruning)
            # 如果存档太大（例如超过2倍种群规模），为了保持文件可读性和计算速度，需要修剪
            # 这里使用简单的随机修剪，标准做法是计算拥挤距离删除最拥挤的
            max_ep_size = self.pop_size * 2  # 设置上限，例如 200
            if len(new_ep) > max_ep_size:
                # 随机移除一个（保留最新的这个，移除旧的）
                # 更好的做法是移除最拥挤的，这里为了代码简洁使用随机移除
                remove_idx = random.randint(0, len(new_ep) - 2)  # 不移除刚加进去的最后一位
                new_ep.pop(remove_idx)

            return new_ep

        return ep

    def run(self):
        """
        运行MOEA/D算法

        Returns:
            tuple: (EP存档, convergence_data)
        """
        print("=" * 60)
        print("MOEA/D: Multi-Objective Evolutionary Algorithm based on Decomposition")
        print("=" * 60)
        print(f"种群规模: {self.pop_size}")
        print(f"迭代代数: {self.ngen}")
        print(f"邻域大小: {self.neighborhood_size}")
        print(f"总评估次数: {self.pop_size * self.ngen}")
        print("=" * 60)

        # 1. 生成权重向量
        print("\n正在生成权重向量...")
        self.weight_vectors = self._generate_weight_vectors(self.pop_size)
        print(f"生成了 {len(self.weight_vectors)} 个权重向量")

        # 2. 计算邻域
        print("正在计算邻域关系...")
        self.neighborhoods = self._compute_neighborhoods()
        print(f"每个子问题的邻域大小: {self.neighborhood_size}")

        # 3. 初始化种群
        print("正在初始化种群...")
        population = [self._init_individual() for _ in range(self.pop_size)]

        # 4. 评估种群并初始化理想点
        print("正在评估初始种群...")
        fitnesses = []
        for ind in population:
            Z1, Z2 = evaluate_individual(
                ind, self.G, self.od_df,
                self.node_positions, self.fixed_tasks
            )
            fitnesses.append((Z1, Z2))

        # 初始化理想点（最大化问题，取最大值）
        fit_array = np.array(fitnesses)
        self.ideal_point = np.max(fit_array, axis=0)  # [max_Z1, max_Z2]
        self.nadir_point = np.min(fit_array, axis=0)  # [min_Z1, min_Z2]

        # 5. 初始化外部存档EP
        ep = []
        for ind, fit in zip(population, fitnesses):
            ep = self._update_ep(ep, ind, fit)

        print(f"初始理想点: Z1={self.ideal_point[0]:.2f}, Z2={self.ideal_point[1]:.2f}")
        print(f"初始EP大小: {len(ep)}")

        # 设置HV参考点（比初始最差点更差，对于最大化问题）
        min_Z1 = min(f[0] for f in fitnesses)
        min_Z2 = min(f[1] for f in fitnesses)
        self.reference_point = (0, 0)  # 使用原点作为参考点

        # 收敛历史
        convergence_history = []

        # 6. 进化循环
        for gen in range(1, self.ngen + 1):
            for i in range(self.pop_size):
                # 从邻域B(i)中随机选择两个父代
                neighbors_idx = self.neighborhoods[i]
                parent_indices = random.sample(neighbors_idx, 2)
                parent1 = population[parent_indices[0]]
                parent2 = population[parent_indices[1]]

                # 交叉
                if random.random() < self.cxpb:
                    child1, child2 = tools.cxTwoPoint(
                        [list(route) for route in parent1],
                        [list(route) for route in parent2]
                    )
                else:
                    child1 = [list(route) for route in parent1]
                    child2 = [list(route) for route in parent2]

                # 变异
                if random.random() < self.mutpb:
                    child1 = self._smart_mutate(child1)
                if random.random() < self.mutpb:
                    child2 = self._smart_mutate(child2)

                # 只处理第一个子代（也可以两个都处理）
                child = child1

                # 评估新解
                Z1_child, Z2_child = evaluate_individual(
                    child, self.G, self.od_df,
                    self.node_positions, self.fixed_tasks
                )
                child_fitness = (Z1_child, Z2_child)

                # 更新理想点
                # 检查新解是否突破了历史最大值或最小值
                self.ideal_point[0] = max(self.ideal_point[0], Z1_child)
                self.ideal_point[1] = max(self.ideal_point[1], Z2_child)

                self.nadir_point[0] = min(self.nadir_point[0], Z1_child)
                self.nadir_point[1] = min(self.nadir_point[1], Z2_child)

                # 更新邻域解
                for j in neighbors_idx:
                    # 计算Tchebycheff值
                    weight_j = self.weight_vectors[j]
                    gte_old = self._tchebycheff(
                        fitnesses[j], weight_j, self.ideal_point,self.nadir_point
                    )
                    gte_new = self._tchebycheff(
                        child_fitness, weight_j, self.ideal_point,self.nadir_point
                    )

                    # 如果新解更好，替换
                    if gte_new < gte_old:
                        population[j] = [list(route) for route in child]
                        fitnesses[j] = child_fitness

                # 更新外部存档EP
                ep = self._update_ep(ep, child, child_fitness)

            # 记录指标
            if gen % 1 == 0 or gen == 1:
                Z1_list = [f[0] for f in fitnesses]
                Z2_list = [f[1] for f in fitnesses]

                # 计算HV
                ep_fitnesses = [fit for _, fit in ep]
                hv = calculate_hypervolume(ep_fitnesses, self.reference_point)

                record = {
                    "gen": gen,
                    "pareto_size": len(ep),
                    "hypervolume": float(hv),
                    "obj1": {
                        "max": float(np.max(Z1_list)),
                        "min": float(np.min(Z1_list)),
                        "mean": float(np.mean(Z1_list)),
                        "std": float(np.std(Z1_list))
                    },
                    "obj2": {
                        "max": float(np.max(Z2_list)),
                        "min": float(np.min(Z2_list)),
                        "mean": float(np.mean(Z2_list)),
                        "std": float(np.std(Z2_list))
                    },
                    "ideal_point": {
                        "Z1": float(self.ideal_point[0]),
                        "Z2": float(self.ideal_point[1])
                    }
                }
                convergence_history.append(record)

                print(f"Gen {gen}/{self.ngen} | "
                      f"EP Size: {len(ep)} | "
                      f"HV: {hv:.2f} | "
                      f"Z1_max: {record['obj1']['max']:.2f} | "
                      f"Z2_max: {record['obj2']['max']:.2f}")

        print("\n=== MOEA/D优化完成 ===")
        print(f"最终EP大小: {len(ep)}")
        print(f"最终理想点: Z1={self.ideal_point[0]:.2f}, Z2={self.ideal_point[1]:.2f}")

        return ep, convergence_history

    def save_results(self, ep, convergence_data,
                     run_id=None, seed=None, duration_sec=0.0):
        """
        保存结果到JSON文件

        Args:
            ep: 外部存档
            convergence_data: 收敛数据
            run_id: 运行编号（Friedman 检验用）
            seed: 随机种子
            duration_sec: 总耗时（秒）
        """
        # 保存收敛数据
        convergence_path = os.path.join(RESULTS_DIR, "MOEAD_convergence.json")
        with open(convergence_path, 'w', encoding='utf-8') as f:
            json.dump(convergence_data, f, indent=4)
        print(f"\n收敛数据已保存至: {convergence_path}")

        # 保存Pareto前沿（EP）
        pareto_data = []
        for idx, (individual, fitness) in enumerate(ep):
            pareto_data.append({
                "index": idx,
                "path": [[int(node) for node in route] for route in individual],
                "fitness": {
                    "visual": float(fitness[0]),
                    "satisfy": float(fitness[1])
                }
            })

        pareto_path = os.path.join(RESULTS_DIR, "MOEAD_pareto.json")
        with open(pareto_path, 'w', encoding='utf-8') as f:
            json.dump(pareto_data, f, indent=4, ensure_ascii=False)
        print(f"Pareto前沿已保存至: {pareto_path}")

        # =========================================================
        # 统一结果保存钩子（T10）
        # =========================================================
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from experiment_io import (save_run_summary, save_best_solutions_from_records,
                                       save_final_pareto_from_records, save_route_quality_from_records)
            from metrics_recorder import compute_hv
            import numpy as np

            records = [(d["fitness"]["visual"], d["fitness"]["satisfy"], d["path"])
                       for d in pareto_data]
            front_objs = np.array([[r[0], r[1]] for r in records]) if records else np.zeros((0, 2))
            hv_value = compute_hv(front_objs, np.array([0.0, 0.0]))

            save_best_solutions_from_records(records, RESULTS_DIR)
            save_final_pareto_from_records(records, RESULTS_DIR)
            save_run_summary(
                algo="MOEAD",
                run_id=run_id if run_id is not None else 0,
                seed=seed if seed is not None else 0,
                summary={
                    "duration_sec": round(duration_sec, 2),
                    "final_hv": hv_value,
                    "z1_max": float(front_objs[:, 0].max()) if len(front_objs) else 0.0,
                    "z2_max": float(front_objs[:, 1].max()) if len(front_objs) else 0.0,
                    "pareto_front_size": len(records),
                },
                out_dir=RESULTS_DIR,
            )
            save_route_quality_from_records(
                records, RESULTS_DIR, "MOEAD",
                run_id if run_id is not None else 0,
                self.G, self.node_positions
            )
            print(f"[钩子] MOEAD 统一结果已保存到 {RESULTS_DIR}")
        except Exception as e:
            import traceback
            print(f"[警告] MOEAD 统一结果保存失败: {e}")
            traceback.print_exc()


def main():
    """主函数：测试MOEA/D算法"""
    import argparse
    import config

    parser = argparse.ArgumentParser(description="MOEA/D 基线算法")
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--set", action="append", default=[],
                        metavar="SECTION.KEY=VALUE",
                        help="覆盖 config 参数（可多次使用），"
                             "如 --set MODEL_CONSTRAINTS.DELTA_MAX=3.0")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # 应用 --set 超参数覆盖（必须在 load_data() 之前）
    if args.set:
        from config_override import apply_overrides
        apply_overrides(args.set)

    global RESULTS_DIR
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        RESULTS_DIR = args.out_dir
        config.RESULTS_DIR = args.out_dir

    from data_loader import load_data
    import time

    print("正在加载数据...")
    edges_gdf, od_df, fixed_tasks, G, node_positions = load_data()

    print(f"图节点数: {G.number_of_nodes()}")
    print(f"图边数: {G.number_of_edges()}")
    print(f"固定任务数: {len(fixed_tasks)}")

    # 初始化MOEA/D
    moead = MOEAD(G, od_df, fixed_tasks, node_positions)

    t0 = time.perf_counter()
    # 运行优化
    ep, convergence_data = moead.run()
    duration_sec = time.perf_counter() - t0

    # 保存结果
    moead.save_results(ep, convergence_data,
                       run_id=args.run_id, seed=args.seed,
                       duration_sec=duration_sec)

    print("\n=== MOEA/D算法运行完成 ===")


if __name__ == "__main__":
    main()
