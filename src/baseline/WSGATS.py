"""
WSGATS: Weighted Sum GA with Taboo Search
带禁忌搜索的加权和遗传算法

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
from collections import deque
from typing import List, Tuple, Dict, Any

# 添加父目录到路径以导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import evaluate_individual
from config import GA_PARAMS, STRATEGIES, RESULTS_DIR
from utils import calculate_haversine_distance


class TabuList:
    """禁忌表类：使用FIFO策略管理禁忌解"""

    def __init__(self, max_size=50, tenure=10):
        """
        初始化禁忌表

        Args:
            max_size: 禁忌表最大容量
            tenure: 禁忌持续代数
        """
        self.max_size = max_size
        self.tenure = tenure
        # 存储格式: {solution_tuple: gen_added}
        self.solutions = {}

    def add(self, solution, gen):
        """
        添加解到禁忌表

        Args:
            solution: 个体（路径列表）
            gen: 当前代数
        """
        # 将解转换为tuple以便作为字典键
        sol_tuple = self._serialize(solution)
        self.solutions[sol_tuple] = gen

        # 如果超过容量，删除最老的
        if len(self.solutions) > self.max_size:
            oldest = min(self.solutions.items(), key=lambda x: x[1])
            del self.solutions[oldest[0]]

    def is_tabu(self, solution, gen):
        """
        检查解是否被禁忌

        Args:
            solution: 个体
            gen: 当前代数

        Returns:
            bool: True表示被禁忌
        """
        sol_tuple = self._serialize(solution)
        if sol_tuple not in self.solutions:
            return False

        # 检查是否超过tenure
        gen_added = self.solutions[sol_tuple]
        if gen - gen_added >= self.tenure:
            # 解除禁忌
            del self.solutions[sol_tuple]
            return False

        return True

    def _serialize(self, solution):
        """将解序列化为tuple"""
        return tuple(tuple(route) for route in solution)


class WSGATS:
    """WSGATS算法主类"""

    def __init__(self, G, od_df, fixed_tasks, node_positions):
        """
        初始化WSGATS

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
        self.weights_list = [
            (0.5, 0.5)
        ]
        self.pop_size = GA_PARAMS['POP_SIZE']  # 100
        self.gens_per_group = 200  # 每组权重40代
        self.cxpb = GA_PARAMS['CXPB']
        self.mutpb = GA_PARAMS['MUTPB']

        # 禁忌搜索参数
        self.tabu_size = 50
        self.tabu_tenure = 10
        self.neighborhood_size = 5
        self.use_aspiration = True

        # 归一化参数（用于消除量纲差异）
        self.Z1_max = 40.0    # obj1（视觉体验）的最大值估计
        self.Z2_max = 13700.0  # obj2（客流需求）的最大值估计

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
                # 混合策略初始化
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

            # 容错
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

    def _evaluate_weighted(self, individual, w1, w2):
        """
        计算加权适应度（带归一化处理）

        Args:
            individual: 个体
            w1, w2: 权重系数（w1 + w2 = 1）

        Returns:
            float: 加权适应度 F = w1*Z1_norm + w2*Z2_norm
            tuple: (Z1, Z2) 原始目标值
        """
        Z1, Z2 = evaluate_individual(
            individual, self.G, self.od_df,
            self.node_positions, self.fixed_tasks
        )

        # 归一化处理（消除量纲差异）
        # obj1最大值约40，obj2最大值约13700
        # 归一化到[0, 1]范围后再加权求和
        Z1_normalized = Z1 / self.Z1_max
        Z2_normalized = Z2 / self.Z2_max

        # 加权求和（归一化后）
        fitness = w1 * Z1_normalized + w2 * Z2_normalized

        return fitness, (Z1, Z2)

    def _smart_mutate(self, individual):
        """
        智能变异：复用ga_engine的逻辑

        Returns:
            新的个体（变异后的副本）
        """
        # 深拷贝个体
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

    def _generate_neighborhood(self, individual):
        """
        生成邻域解：调用变异算子

        Args:
            individual: 当前个体

        Returns:
            list: 邻域解列表
        """
        neighbors = []
        for _ in range(self.neighborhood_size):
            neighbor = self._smart_mutate(individual)
            neighbors.append(neighbor)
        return neighbors

    def _tabu_search(self, individual, w1, w2, tabu_list,
                     current_gen, global_best_fit, global_best_ind):
        """
        对个体执行禁忌搜索

        Args:
            individual: 待优化的个体
            w1, w2: 权重
            tabu_list: 禁忌表对象
            current_gen: 当前代数
            global_best_fit, global_best_ind: 全局最优

        Returns:
            改进后的个体
        """
        # 生成邻域
        neighborhood = self._generate_neighborhood(individual)

        # 评估邻域
        neighbor_evals = []
        for neighbor in neighborhood:
            fit, (Z1, Z2) = self._evaluate_weighted(neighbor, w1, w2)
            neighbor_evals.append((neighbor, fit, Z1, Z2))

        # 评估当前个体
        current_fit, (current_Z1, current_Z2) = self._evaluate_weighted(
            individual, w1, w2
        )

        # 找到最佳邻域解
        best_neighbor = None
        best_neighbor_fit = float('-inf')
        best_neighbor_Z1 = best_neighbor_Z2 = None

        for neighbor, fit, Z1, Z2 in neighbor_evals:
            # 检查特赦条件：如果优于全局最优，接受
            if self.use_aspiration and fit > global_best_fit:
                return neighbor, fit, Z1, Z2

            # 如果不被禁忌，记录最佳候选
            if not tabu_list.is_tabu(neighbor, current_gen):
                if fit > best_neighbor_fit:
                    best_neighbor = neighbor
                    best_neighbor_fit = fit
                    best_neighbor_Z1 = Z1
                    best_neighbor_Z2 = Z2

        # 如果找到非禁忌解且优于当前，接受
        if best_neighbor is not None and best_neighbor_fit > current_fit:
            # 将新解加入禁忌表
            tabu_list.add(best_neighbor, current_gen)
            return best_neighbor, best_neighbor_fit, best_neighbor_Z1, best_neighbor_Z2

        # 否则保持原解
        return individual, current_fit, current_Z1, current_Z2

    def _tournament_selection(self, population, fitnesses, k=3):
        """
        锦标赛选择

        Args:
            population: 种群列表
            fitnesses: 适应度列表
            k: 锦标赛大小

        Returns:
            选出的个体
        """
        participants = random.sample(list(zip(population, fitnesses)), k)
        winner = max(participants, key=lambda x: x[1])
        return winner[0]

    def _is_dominated(self, sol1, sol2):
        """
        检查sol1是否被sol2支配

        Args:
            sol1, sol2: (Z1, Z2) 元组

        Returns:
            bool: True表示sol1被sol2支配
        """
        return (sol2[0] >= sol1[0] and sol2[1] >= sol1[1] and
                (sol2[0] > sol1[0] or sol2[1] > sol1[1]))

    def _extract_pareto_front(self, solutions):
        """
        从解集中提取Pareto前沿

        Args:
            solutions: [(individual, (Z1, Z2)), ...] 列表

        Returns:
            Pareto前沿解列表
        """
        pareto_front = []
        for i, sol1 in enumerate(solutions):
            dominated = False
            for j, sol2 in enumerate(solutions):
                if i != j and self._is_dominated(sol1[1], sol2[1]):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(sol1)
        return pareto_front

    def run_single_weight_group(self, w1, w2, weight_label):
        """
        运行单个权重组的优化

        Args:
            w1, w2: 权重系数
            weight_label: 权重标签（如"1.0_0.0"）

        Returns:
            dict: 该组的结果
        """
        print(f"\n=== 开始优化权重组 {weight_label} (w1={w1}, w2={w2}) ===")

        # 初始化种群
        population = [self._init_individual() for _ in range(self.pop_size)]

        # 评估种群
        fitnesses = []
        Z1_list = []
        Z2_list = []
        for ind in population:
            fit, (Z1, Z2) = self._evaluate_weighted(ind, w1, w2)
            fitnesses.append(fit)
            Z1_list.append(Z1)
            Z2_list.append(Z2)

        # 初始化禁忌表
        tabu_list = TabuList(
            max_size=self.tabu_size,
            tenure=self.tabu_tenure
        )

        # 记录全局最优
        global_best_idx = np.argmax(fitnesses)
        global_best_ind = population[global_best_idx]
        global_best_fit = fitnesses[global_best_idx]
        global_best_Z1 = Z1_list[global_best_idx]
        global_best_Z2 = Z2_list[global_best_idx]

        # 收敛历史
        convergence_history = []

        # 进化循环
        for gen in range(1, self.gens_per_group + 1):
            offspring = []

            # 选择与繁殖
            while len(offspring) < self.pop_size:
                # 锦标赛选择
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)

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

                offspring.append(child1)
                if len(offspring) < self.pop_size:
                    offspring.append(child2)

            # 评估后代
            offspring_fitnesses = []
            offspring_Z1 = []
            offspring_Z2 = []

            for idx, ind in enumerate(offspring):
                # 禁忌搜索（对新个体的邻域探索）
                ind, fit, Z1, Z2 = self._tabu_search(
                    ind, w1, w2, tabu_list, gen,
                    global_best_fit, global_best_ind
                )
                offspring[idx] = ind
                offspring_fitnesses.append(fit)
                offspring_Z1.append(Z1)
                offspring_Z2.append(Z2)

            # 更新全局最优
            for idx, fit in enumerate(offspring_fitnesses):
                if fit > global_best_fit:
                    global_best_fit = fit
                    global_best_ind = offspring[idx]
                    global_best_Z1 = offspring_Z1[idx]
                    global_best_Z2 = offspring_Z2[idx]

            # 环境选择：合并父代子代，选最好的pop_size个
            combined = list(zip(
                population + offspring,
                fitnesses + offspring_fitnesses,
                Z1_list + offspring_Z1,
                Z2_list + offspring_Z2
            ))

            # 按适应度排序，选前pop_size个
            combined.sort(key=lambda x: x[1], reverse=True)
            selected = combined[:self.pop_size]

            population = [item[0] for item in selected]
            fitnesses = [item[1] for item in selected]
            Z1_list = [item[2] for item in selected]
            Z2_list = [item[3] for item in selected]

            # 记录指标
            record = {
                "gen": gen,
                "obj1_max": float(np.max(Z1_list)),
                "obj2_max": float(np.max(Z2_list)),
                "obj1_mean": float(np.mean(Z1_list)),
                "obj2_mean": float(np.mean(Z2_list)),
                "best_fitness": float(global_best_fit),
                "best_Z1": float(global_best_Z1),
                "best_Z2": float(global_best_Z2)
            }
            convergence_history.append(record)

            if gen % 10 == 0:
                print(f"  Gen {gen}/{self.gens_per_group} | "
                      f"Best Fitness: {global_best_fit:.2f} | "
                      f"Z1: {global_best_Z1:.2f} | Z2: {global_best_Z2:.2f}")

        print(f"=== 权重组 {weight_label} 优化完成 ===")
        print(f"  最终最优适应度: {global_best_fit:.2f}")
        print(f"  Z1: {global_best_Z1:.2f}, Z2: {global_best_Z2:.2f}")

        return {
            "weight_label": weight_label,
            "w1": w1,
            "w2": w2,
            "best_individual": global_best_ind,
            "best_Z1": global_best_Z1,
            "best_Z2": global_best_Z2,
            "best_fitness": global_best_fit,
            "convergence": convergence_history
        }

    def run(self):
        """
        运行完整的WSGATS算法

        Returns:
            tuple: (pareto_front, convergence_data)
        """
        print("=" * 60)
        print("WSGATS: Weighted Sum GA with Taboo Search")
        print("=" * 60)
        print(f"权重组数: {len(self.weights_list)}")
        print(f"每组种群规模: {self.pop_size}")
        print(f"每组迭代代数: {self.gens_per_group}")
        print(f"总评估次数: {len(self.weights_list) * self.pop_size * self.gens_per_group}")
        print("=" * 60)

        all_results = []
        convergence_data = {}

        # 对每个权重组运行优化
        for w1, w2 in self.weights_list:
            weight_label = f"{w1:.2f}_{w2:.2f}"

            result = self.run_single_weight_group(w1, w2, weight_label)
            all_results.append(result)
            convergence_data[weight_label] = result['convergence']

        # 构建近似Pareto前沿
        print("\n=== 构建近似Pareto前沿 ===")

        # 收集所有最终最优解
        final_solutions = []
        for result in all_results:
            final_solutions.append((
                result['best_individual'],
                (result['best_Z1'], result['best_Z2']),
                result['weight_label']
            ))

        # 提取非支配解
        pareto_front = self._extract_pareto_front(final_solutions)

        print(f"从 {len(final_solutions)} 个解中提取了 {len(pareto_front)} 个非支配解")

        return pareto_front, convergence_data

    def save_results(self, pareto_front, convergence_data,
                     run_id=None, seed=None, duration_sec=0.0):
        """
        保存结果到JSON文件

        Args:
            pareto_front: Pareto前沿解列表
            convergence_data: 收敛数据
            run_id: 运行编号（Friedman 检验用）
            seed: 随机种子
            duration_sec: 总耗时（秒）
        """
        # 保存收敛数据
        convergence_path = os.path.join(RESULTS_DIR, "WSGATS_convergence.json")
        with open(convergence_path, 'w', encoding='utf-8') as f:
            json.dump(convergence_data, f, indent=4)
        print(f"\n收敛数据已保存至: {convergence_path}")

        # 保存Pareto前沿
        pareto_data = []
        for idx, (individual, fitness, weight_label) in enumerate(pareto_front):
            pareto_data.append({
                "index": idx,
                "path": [[int(node) for node in route] for route in individual],
                "fitness": {
                    "visual": float(fitness[0]),
                    "satisfy": float(fitness[1])
                },
                "weight_group": weight_label
            })

        pareto_path = os.path.join(RESULTS_DIR, "WSGATS_pareto.json")
        with open(pareto_path, 'w', encoding='utf-8') as f:
            json.dump(pareto_data, f, indent=4, ensure_ascii=False)
        print(f"Pareto前沿已保存至: {pareto_path}")

        # =========================================================
        # 统一结果保存钩子（T9）
        # =========================================================
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from experiment_io import (save_run_summary, save_best_solutions_from_records,
                                       save_final_pareto_from_records, save_route_quality_from_records)
            from metrics_recorder import compute_hv

            # 构造 records 列表 [(z1, z2, routes), ...]
            records = [(d["fitness"]["visual"], d["fitness"]["satisfy"], d["path"])
                       for d in pareto_data]
            front_objs = np.array([[r[0], r[1]] for r in records]) if records else np.zeros((0, 2))
            hv_value = compute_hv(front_objs, np.array([0.0, 0.0]))

            save_best_solutions_from_records(records, RESULTS_DIR)
            save_final_pareto_from_records(records, RESULTS_DIR)
            save_run_summary(
                algo="WSGATS",
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
                records, RESULTS_DIR, "WSGATS",
                run_id if run_id is not None else 0,
                self.G, self.node_positions
            )
            print(f"[钩子] WSGATS 统一结果已保存到 {RESULTS_DIR}")
        except Exception as e:
            import traceback
            print(f"[警告] WSGATS 统一结果保存失败: {e}")
            traceback.print_exc()


def main():
    """主函数：测试WSGATS算法"""
    import argparse
    import config

    parser = argparse.ArgumentParser(description="WSGATS 基线算法")
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

    # 注入 out_dir 到 config.RESULTS_DIR 以便 save_results 使用
    global RESULTS_DIR
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        RESULTS_DIR = args.out_dir
        config.RESULTS_DIR = args.out_dir
    _args_ns = args  # 给 save_results 引用

    from data_loader import load_data
    import time
    total_start = time.perf_counter()
    print("正在加载数据...")
    edges_gdf, od_df, fixed_tasks, G, node_positions = load_data()

    print(f"图节点数: {G.number_of_nodes()}")
    print(f"图边数: {G.number_of_edges()}")
    print(f"固定任务数: {len(fixed_tasks)}")

    # 初始化WSGATS
    wsgats = WSGATS(G, od_df, fixed_tasks, node_positions)

    # 运行优化
    pareto_front, convergence_data = wsgats.run()
    total_end = time.perf_counter()
    duration_sec = total_end - total_start
    print(f"整体总耗时: {duration_sec:.4f} 秒")
    # 保存结果
    wsgats.save_results(pareto_front, convergence_data,
                        run_id=args.run_id, seed=args.seed,
                        duration_sec=duration_sec)

    print("\n=== WSGATS算法运行完成 ===")


if __name__ == "__main__":
    main()
