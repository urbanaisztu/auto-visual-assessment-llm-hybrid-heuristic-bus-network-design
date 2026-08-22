# encoding: utf-8
# author: zhaotianhong
# contact: zhaotianhong2016@email.szu.edu.cn
import json
import os
import pickle

import math
import random

import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from typing import List, Any, Tuple, Dict
# import environment
# import visualization
import copy
import itertools
import statistics




# from route_planning.NSGA_test.main import env


# from route_planning.NSGA_test.main import info_each_iteration, maxGen

def get_environment():
    path_edge = '../../data/reindex/edge.csv'
    path_node = '../../data/reindex/node.csv'
    path_cost_all = '../../data/reindex/matrix_all.csv'
    path_cost = '../../data/reindex/matrix.json'
    path_route = '../../data/reindex/route_OD.csv'
    path_init_route = '../../data/reindex/route_init.csv'
    path_demand = '../../data/reindex/demand_morning.csv'

    G = construct_graph(path_edge, path_node)

    # cost_matrix = load_cost_matrix_json(path_cost)
    cost_matrix = load_cost_matrix(path_cost_all)
    flow_network = assign_traffic(path_demand, path_edge, G)
    init_route = load_init_route(path_init_route)

    env = environment.Environment(G, cost_matrix, flow_network, init_route)

    return env


def readText(file_path):
    try:
        # 以只读模式打开txt文件，指定编码为utf-8（避免中文乱码）
        with open(file_path, "r", encoding="utf-8") as file:
            # 读取文件全部内容并赋值给变量
            txt = file.read().strip()  # strip() 去除首尾空白字符（换行、空格）

        # 验证读取结果（可选）
        if not txt:
            print("警告：txt文件内容为空！")
        else:
            return txt
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}，请检查文件路径是否正确")
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")

def logInfo(record, maxGen, pop_size, cxProb, mutateProb, gen,
            cx_weights=None, mt_weights=None,
            cx_usage_count=None, mt_usage_count=None):
    """
    记录每代信息

    参数:
        record: [record_pop, record_fronts]
        maxGen: 总代数
        pop_size: 种群大小
        cxProb: 交叉概率
        mutateProb: 变异概率
        gen: 当前代数
        cx_weights: 当前交叉算子权重 (可选)
        mt_weights: 当前变异算子权重 (可选)
        cx_usage_count: 交叉算子累计使用次数 (可选)
        mt_usage_count: 变异算子累计使用次数 (可选)
    """
    info_each_iteration = {}

    # 基本信息
    info_each_iteration["iteration"] = gen

    # 方法参数
    method_info = {
        "maxGen": maxGen,
        "pop_size": pop_size,
        "cxProb": cxProb,
        "mutateProb": mutateProb
    }

    # 新增：算子权重和使用率
    if cx_weights is not None:
        method_info["cx_weights"] = cx_weights.copy()
    if mt_weights is not None:
        method_info["mt_weights"] = mt_weights.copy()

    # 计算使用率
    if cx_usage_count is not None:
        total_cx = sum(cx_usage_count.values())
        if total_cx > 0:
            method_info["cx_usage_rate"] = {k: v/total_cx for k, v in cx_usage_count.items()}
        else:
            method_info["cx_usage_rate"] = {"overall": 0, "visual": 0, "demand": 0, "default": 0}

    if mt_usage_count is not None:
        total_mt = sum(mt_usage_count.values())
        if total_mt > 0:
            method_info["mt_usage_rate"] = {k: v/total_mt for k, v in mt_usage_count.items()}
        else:
            method_info["mt_usage_rate"] = {"overall": 0, "visual": 0, "demand": 0, "default": 0}

    info_each_iteration["method"] = method_info

    # 效果记录
    info_each_iteration["effect"] = record[0]
    info_each_iteration["pareto_fronts"] = record[1]
    info_each_iteration['num_of_unique_front'] = record[1]['front_count']
    info_each_iteration['hypervolume'] = record[1]['hypervolume']

    return info_each_iteration



def getBalanceOptimal(pareto_front_individuals,env):
    if not pareto_front_individuals:
        print("帕累托前沿为空，无法找到最优解。")
    else:
        # 2. 收集所有适应度值，用于后续归一化
        all_visual_values = [ind.fitness.values[0] for ind in pareto_front_individuals]
        all_satisfy_values = [ind.fitness.values[1] for ind in pareto_front_individuals]

        # 3. 计算每个目标的最大值和最小值
        min_visual, max_visual = min(all_visual_values), max(all_visual_values)
        min_satisfy, max_satisfy = min(all_satisfy_values), max(all_satisfy_values)

        # 防止除以零的错误（如果所有值都相同）
        range_visual = max_visual - min_visual if max_visual > min_visual else 1.0
        range_satisfy = max_satisfy - min_satisfy if max_satisfy > min_satisfy else 1.0

        best_individual = None
        best_score = -float('inf')

        # 4. 遍历前沿上的每个个体，计算其归一化加权分数
        for ind in pareto_front_individuals:
            visual, satisfy = ind.fitness.values

            # 进行Min-Max归一化
            normalized_visual = (visual - min_visual) / range_visual
            normalized_satisfy = (satisfy - min_satisfy) / range_satisfy

            # 计算加权分数 (权重相同)
            # 这里我们最大化 visual 和 satisfy，所以权重为正
            weighted_score = 0.5 * normalized_visual + 0.5 * normalized_satisfy

            # 5. 找到分数最高的个体
            if weighted_score > best_score:
                best_score = weighted_score
                best_individual = ind

        # 6. 输出并可视化找到的最优平衡解
        if best_individual:
            print("\n在 'visual' 和 'satisfy' 之间取得最佳平衡的个体 (权重各为0.5):")
            print(f"个体路径: {best_individual}")
            print(
                f"原始适应度: Visual={best_individual.fitness.values[0]:.4f}, Satisfy={best_individual.fitness.values[1]:.4f}")

            # 计算并打印归一化后的分数，方便验证
            norm_vis = (best_individual.fitness.values[0] - min_visual) / range_visual
            norm_sat = (best_individual.fitness.values[1] - min_satisfy) / range_satisfy
            print(f"归一化后分数: Visual={norm_vis:.4f}, Satisfy={norm_sat:.4f}")
            print(f"加权总分: {0.5 * norm_vis + 0.5 * norm_sat:.4f}")

            # 注意：这里的plot_network函数，我假设它能正确处理best_individual
            # 如果它需要特定格式的routes，请根据函数要求进行调整
            visualization.plot_network(env, best_individual, 10, 1, 'BEST Balanced Solution (visual & satisfy)')
        else:
            print("未能找到最优平衡解。")

def extract_avg_fitness(evolution_history):
    """从evolution_history中提取每代的平均视觉适应度和平均满足需求适应度"""
    generations = []  # 迭代次数（gen）
    avg_visual = []   # 平均视觉适应度
    avg_satisfy = []  # 平均满足需求适应度

    for item in evolution_history:
        # 解析logInfo记录的内容（根据你的logInfo函数输出结构调整，核心是拿到record_pop）
        # 假设logInfo返回的字典中，"record"字段对应你存储的[record_pop, record_fronts]
        record_pop = item["effect"]  # 对应代码中的 record = [record_pop, record_fronts]
        # record_pop = record[0]   # 种群整体统计数据（包含min/max/q1/median/q3）

        simple_avg_visual = record_pop["mean_visual"]
        simple_avg_satisfy = record_pop["mean_satisfy"]

        # 收集数据
        generations.append(item["iteration"])
        avg_visual.append(simple_avg_visual)
        avg_satisfy.append(simple_avg_satisfy)

    return np.array(generations), np.array(avg_visual), np.array(avg_satisfy)

def check_node_repeat(node_o, node_new):
    '''
    检查是否点重复
    :param node_o: 路径中原油节点
    :param node_new: 新增点节点
    :return:
    '''
    for n in node_new:
        if n in node_o:
            return False
    return True

def showConverge(generations, avg_visual, avg_satisfy):
    # 转为numpy数组
    generations = np.array(generations)
    avg_visual = np.array(avg_visual)
    avg_satisfy = np.array(avg_satisfy)

    # ---------------------- 计算斜率 ----------------------
    slope_visual = np.diff(avg_visual)
    slope_satisfy = np.diff(avg_satisfy)
    generations_slope = generations[:-1]

    # ================================
    #   图 1：Visual（双纵轴）
    # ================================
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    fig1.suptitle("Visual Fitness & Convergence Speed", fontsize=15, fontweight="bold")

    # 左轴：适应度
    line1 = ax1.plot(generations, avg_visual,
                     color="#1f77b4", linewidth=2.5, marker="o", markersize=3,
                     label="Average Visual Fitness")
    ax1.set_ylabel("Visual Fitness", fontsize=12)
    ax1.grid(alpha=0.3, linestyle="--")

    # 右轴：斜率
    ax1_r = ax1.twinx()
    line2 = ax1_r.plot(generations_slope, slope_visual,
                       color="#d62728", linewidth=2.0, linestyle="--", marker="^", markersize=3,
                       label="Visual Fitness Slope")
    ax1_r.set_ylabel("Visual Slope (Δ fitness / gen)", fontsize=12)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=10, loc="upper left")

    plt.tight_layout()
    plt.savefig("./results/visual_fitness_slope.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ================================
    #   图 2：Satisfaction（双纵轴）
    # ================================
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.suptitle("Satisfaction Fitness & Convergence Speed", fontsize=15, fontweight="bold")

    # 左轴：适应度
    line1 = ax2.plot(generations, avg_satisfy,
                     color="#ff7f0e", linewidth=2.5, marker="s", markersize=3,
                     label="Average Satisfaction Fitness")
    ax2.set_ylabel("Satisfaction Fitness", fontsize=12)
    ax2.grid(alpha=0.3, linestyle="--")

    # 右轴：斜率
    ax2_r = ax2.twinx()
    line2 = ax2_r.plot(generations_slope, slope_satisfy,
                       color="#2ca02c", linewidth=2.0, linestyle="--", marker="d", markersize=3,
                       label="Satisfaction Fitness Slope")
    ax2_r.set_ylabel("Satisfaction Slope (Δ fitness / gen)", fontsize=12)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=10, loc="upper left")

    plt.tight_layout()
    plt.savefig("./results/satisfy_fitness_slope.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---------------------- 统计输出 ----------------------
    print("=== 简单平均适应度趋势统计 ===")
    print(f"迭代次数范围：{generations.min()} ~ {generations.max()}")
    print(f"平均视觉适应度变化：{avg_visual.min():.2f} → {avg_visual.max():.2f}（提升{avg_visual.max() - avg_visual.min():.2f}）")
    print(f"平均满足需求适应度变化：{avg_satisfy.min():.2f} → {avg_satisfy.max():.2f}（提升{avg_satisfy.max() - avg_satisfy.min():.2f}）")

    print("\n=== 收敛速度（斜率）统计 ===")
    print(f"视觉适应度最大收敛速度：{slope_visual.max():.2f}（第{generations_slope[np.argmax(slope_visual)]}代）")
    print(f"视觉适应度最小收敛速度：{slope_visual.min():.2f}（第{generations_slope[np.argmin(slope_visual)]}代）")
    print(f"满足需求适应度最大收敛速度：{slope_satisfy.max():.2f}（第{generations_slope[np.argmax(slope_satisfy)]}代）")
    print(f"满足需求适应度最小收敛速度：{slope_satisfy.min():.2f}（第{generations_slope[np.argmin(slope_satisfy)]}代）")
    print(f"视觉适应度平均收敛速度：{slope_visual.mean():.2f}")
    print(f"满足需求适应度平均收敛速度：{slope_satisfy.mean():.2f}")

def get_evaluate_pop(pop):
    sorted_pop = sorted(
        pop,
        key=lambda ind: ind.fitness.values[0],
        reverse=True
    )

    # 选择前 3 个个体
    k = 100
    top_individuals: List[Any] = sorted_pop[:k]

    save_path = "./evaluate_pop.pkl"
    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存evaluate_pop到文件
    with open(save_path, 'wb') as f:
        pickle.dump(top_individuals, f)

    return top_individuals


def compile_and_load_code_string(code_string: str):
    global_dependencies = {
        # 外部函数/模块
        # 'mut_prob_index': mut_prob_index,
        'check_node_repeat': check_node_repeat,

        # 常用模块 (供 LLM 代码直接使用，无需 import)
        'random': random,  # 注入已导入的 random 模块
        'nx': nx,  # 注入已导入的 networkx 模块
        'math': math,
        'np': np,
        'pd': pd,
        'copy': copy
    }

    # 创建执行环境，并将依赖注入作为初始值
    # local_namespace 将作为 exec 的全局和局部命名空间
    local_namespace = global_dependencies.copy()

    # 执行代码字符串
    # 这将在 local_namespace 中创建函数定义 (例如 'crossover_operator')
    exec(code_string, local_namespace)

    # 从命名空间中取出我们期望的函数

    if 'cx_operator' in local_namespace:
        return local_namespace['cx_operator']
    elif 'mt_operator' in local_namespace:
        return local_namespace['mt_operator']
    else:
        # 如果代码执行了但没有定义期望的函数，则抛出错误
        raise NameError("error in function name")


import itertools
import statistics
import math


def calculate_jaccard_distance(seq1, seq2):
    """
    计算两个个体的 Jaccard 距离 (1 - Jaccard Index)。
    支持 List[int] 和 List[List[int]] (多线路) 结构。
    """

    # --- 1. 预处理助手函数：将 list 转为 tuple ---
    def make_hashable(seq):
        if not seq:
            return []
        # 判断是否为嵌套列表 (List of Lists)
        if isinstance(seq, list) and len(seq) > 0 and isinstance(seq[0], list):
            # 将每一条线路 (list) 转为 tuple
            # 例如: [[1,2], [3,4]] -> [(1,2), (3,4)]
            return [tuple(sub_list) for sub_list in seq]
        return seq

    # --- 2. 转换为集合 ---
    # 转换后，元素变成了 tuple，就可以放入 set 了
    s1 = set(make_hashable(seq1))
    s2 = set(make_hashable(seq2))

    # --- 3. 计算 Jaccard 距离 ---
    union_len = len(s1.union(s2))

    if union_len == 0:
        return 0.0

    intersection_len = len(s1.intersection(s2))
    jaccard_index = intersection_len / union_len

    return 1.0 - jaccard_index


import statistics
import random
import itertools
from func_timeout import func_timeout, FunctionTimedOut

import random
import itertools
import statistics
from func_timeout import func_timeout, FunctionTimedOut
from typing import Any, List


def evaluate_operator(population: list, evaluate_pop: list, toolbox: Any, env: Any, op_type: str,
                      seed: int = 42) -> list:
    """
    评估算子种群中每个算子的优越性。
    【修改版 - 适配多路径结构 [[r1], [r2]...]】：
    """

    # --- 基础配置 ---
    TIMEOUT_LIMIT = 20.0
    HARD_PENALTY_SCORE = -100.0
    SOFT_PENALTY_FACTOR = 0.01
    SCORE_SCALING = 100.0

    # --- 【关键修改】内部辅助：将多路径结构转换为平铺的边列表 ---
    def path_to_edges(individual):
        """
        将多路径结构 [[1,2,3], [4,5,6]] 转换为边列表 [(1,2), (2,3), (4,5), (5,6)]
        避免将路径之间的断点连接起来。
        """
        edges = []

        # 安全检查
        if not individual:
            return []

        # 检查是否为嵌套列表 (多路径结构)
        # 判断标准：列表非空，且第一个元素也是列表
        if isinstance(individual, list) and len(individual) > 0 and isinstance(individual[0], list):
            # 遍历每一条子路径 (Sub-route)
            for sub_route in individual:
                if len(sub_route) >= 2:
                    # 提取当前子路径的边
                    # 假设是无向边，排序以保证 (1,2) 和 (2,1) 被视为相同
                    # 如果是有向图 (VRP通常是有向的)，去掉 sorted
                    route_edges = [tuple(sorted((sub_route[i], sub_route[i + 1]))) for i in range(len(sub_route) - 1)]
                    edges.extend(route_edges)
        else:
            # 兼容旧逻辑：如果是单路径结构 [1, 2, 3, 4]
            if len(individual) >= 2:
                edges = [tuple(sorted((individual[i], individual[i + 1]))) for i in range(len(individual) - 1)]

        return edges

    # 1. 参数校验
    valid_ops = ['cx', 'mt']
    if op_type not in valid_ops:
        raise ValueError(f"op_type error: {op_type}")

    # 2. 准备测试用例
    if op_type == 'cx':
        all_combinations = list(itertools.combinations(evaluate_pop, 2))
        if len(all_combinations) > 50:
            rng = random.Random(seed)
            test_cases = rng.sample(all_combinations, 50)
        else:
            test_cases = all_combinations
    else:
        test_cases = [(ind,) for ind in evaluate_pop]

    # 3. 遍历评估每个算子
    for op_ind in population:
        gains_overall = []
        gains_obj1 = []
        gains_obj2 = []
        valid_runs = 0
        code_string = op_ind['code']

        try:
            # A. 编译
            op_func = compile_and_load_code_string(code_string)
            if not op_func:
                raise ValueError("编译失败")

            # B. 运行测试用例
            for test_case in test_cases:
                # 准备基准数据
                if op_type == 'cx':
                    parent1_orig, parent2_orig = test_case
                    base_f1 = max(parent1_orig.fitness.values[0], parent2_orig.fitness.values[0])
                    base_f2 = max(parent1_orig.fitness.values[1], parent2_orig.fitness.values[1])

                    p1_copy = convert_individual_nodes_to_int(toolbox.clone(parent1_orig))
                    p2_copy = convert_individual_nodes_to_int(toolbox.clone(parent2_orig))

                    # 选取较优父代作为基准路径
                    if parent1_orig.fitness.values[0] > parent2_orig.fitness.values[0]:
                        base_ind_raw = parent1_orig
                    else:
                        base_ind_raw = parent2_orig
                else:
                    parent1_orig = test_case[0]
                    base_f1 = parent1_orig.fitness.values[0]
                    base_f2 = parent1_orig.fitness.values[1]
                    p1_copy = convert_individual_nodes_to_int(toolbox.clone(parent1_orig))
                    base_ind_raw = parent1_orig

                # 【调用修改后的转换函数】获取基准解的所有边
                # 注意：这里假设 base_ind_raw 本身就是 [[...], [...]] 结构
                # 如果它是对象，请确保在这里提取出 list 结构，例如 base_ind_raw.genes 或 list(base_ind_raw)
                # DEAP 个体通常直接继承自 list，所以 list(base_ind_raw) 对于嵌套列表可能只会浅拷贝外层
                # 对于 List[List]，直接使用 base_ind_raw 即可，或者用 [list(r) for r in base_ind_raw] 确保纯净
                base_edges = path_to_edges(base_ind_raw)

                try:
                    # 封装执行逻辑
                    def _run_logic():
                        if op_type == 'cx':
                            return op_func(p1_copy, p2_copy, env)
                        elif op_type == 'mt':
                            return op_func(p1_copy, env)
                        return None

                    children = func_timeout(TIMEOUT_LIMIT, _run_logic)

                    if not isinstance(children, (list, tuple)):
                        children = [children]
                    valid_children = [c for c in children if c and len(c) > 0]
                    if not valid_children:
                        continue

                    # --- 评估子代 ---
                    best_metrics = None
                    best_overall_score = -float('inf')

                    for child in valid_children:
                        fit_values = toolbox.evaluate(child)
                        c_f1, c_f2 = fit_values[0], fit_values[1]

                        # 计算相对提升 (diff > 0 代表提升)
                        diff1 = (c_f1 - base_f1) / (abs(base_f1) + 1e-9)
                        diff2 = (c_f2 - base_f2) / (abs(base_f2) + 1e-9)

                        # --- 【修复点】加回软惩罚机制 ---
                        # 目的：防止灾难性个例的数值过大，污染历史平均分
                        # 如果提升，保持原样；如果下降，乘以 0.01
                        real_gain1 = diff1 if diff1 >= 0 else diff1 * SOFT_PENALTY_FACTOR
                        real_gain2 = diff2 if diff2 >= 0 else diff2 * SOFT_PENALTY_FACTOR

                        # 记录显示的 obj 分数 (转换为百分比)
                        s_obj1 = real_gain1 * 100
                        s_obj2 = real_gain2 * 100

                        # --- 综合分计算 (依然使用原始 diff 判断层级，保持严谨) ---
                        EPS = 1e-4
                        better_1 = diff1 > EPS
                        better_2 = diff2 > EPS
                        worse_1 = diff1 < -EPS
                        worse_2 = diff2 < -EPS

                        # A. 确定基础层级分 (Tier Score)
                        # 注意：这里判定层级建议还是用原始 diff，或者你也可以用 soft 后的，影响不大
                        # 但为了熔断机制有效，建议保留对 catastrophic 的判断

                        if better_1 and better_2:
                            tier_score = 100.0
                        elif (better_1 and not worse_2) or (better_2 and not worse_1):
                            tier_score = 80.0
                        elif (better_1 and worse_2) or (better_2 and worse_1):
                            tier_score = 50.0
                        elif not better_1 and not better_2 and not worse_1 and not worse_2:
                            tier_score = 0.0
                        else:
                            tier_score = -50.0

                        # 多样性加分
                        div_bonus = 0.0
                        if tier_score >= 0:
                            child_edges = path_to_edges(child)
                            jaccard_dist = calculate_jaccard_distance(base_edges, child_edges)

                            # 距离计算也建议 clamp 一下，防止溢出
                            obj_dist = math.sqrt(real_gain1 ** 2 + real_gain2 ** 2)
                            div_bonus = (jaccard_dist * 30.0) + (min(obj_dist, 1.0) * 20.0)

                        final_overall = tier_score + div_bonus

                        if final_overall > best_overall_score:
                            best_overall_score = final_overall
                            # 这里的 s_obj1 和 s_obj2 已经是经过软惩罚的值了
                            best_metrics = (final_overall, s_obj1, s_obj2)

                    if best_metrics is not None:
                        gains_overall.append(best_metrics[0])
                        gains_obj1.append(best_metrics[1])
                        gains_obj2.append(best_metrics[2])
                        valid_runs += 1

                except Exception:
                    continue

            # --- 汇总与平滑 ---
            if valid_runs > 0 and gains_overall:
                curr_total = statistics.mean(gains_overall)
                curr_obj1 = statistics.mean(gains_obj1)
                curr_obj2 = statistics.mean(gains_obj2)

                eval_count = op_ind.get('eval_count', 0)
                old_score = op_ind.get('score', None)

                if eval_count > 0 and isinstance(old_score, (tuple, list)) and len(old_score) == 3:
                    old_total, old_o1, old_o2 = old_score
                    new_total = (old_total * eval_count + curr_total) / (eval_count + 1)
                    new_obj1 = (old_o1 * eval_count + curr_obj1) / (eval_count + 1)
                    new_obj2 = (old_o2 * eval_count + curr_obj2) / (eval_count + 1)
                else:
                    new_total, new_obj1, new_obj2 = curr_total, curr_obj1, curr_obj2

                op_ind['score'] = (new_total, new_obj1, new_obj2)
                op_ind['eval_count'] = eval_count + 1

                print(
                    f"ID:{op_ind.get('idx', '?')} | Overall: {new_total:.2f} (Vis:{new_obj1:.2f}%, Dem:{new_obj2:.2f}%) | Count: {op_ind['eval_count']}")
            else:
                op_ind['score'] = HARD_PENALTY_SCORE

        except Exception as e:
            op_ind['score'] = HARD_PENALTY_SCORE
            print(f"Error ({op_ind.get('idx')}): {e}")

    return population



#单一评判标准
# def evaluate_operator(population: list, evaluate_pop: list, toolbox: Any, env: Any, op_type: str,
#                       seed: int = 42) -> list:
#     """
#     评估算子种群中每个算子的优越性。
#     【消融实验版 - Baseline】：
#     1. 移除 Monte Carlo 重复运行（每对父代只跑一次）。
#     2. 移除 多元评分机制（只计算平均提升率）。
#     3. 移除 历史分数平滑/精英保留（每次评估都是独立的，覆盖旧分）。
#     """
#
#     # --- 基础配置 ---
#     TIMEOUT_LIMIT = 20.0  # 超时限制
#     HARD_PENALTY_SCORE = -100.0  # 失败硬惩罚
#     SOFT_PENALTY_FACTOR = 0.01  # 负提升时的软惩罚保留，保证计算逻辑一致
#     SCORE_SCALING = 100.0  # 缩放系数，方便观察数据
#
#     # 1. 参数校验
#     valid_ops = ['cx', 'mt']
#     if op_type not in valid_ops:
#         raise ValueError(f"op_type 必须是 {valid_ops} 之一，当前为: {op_type}")
#
#     # 2. 准备测试用例
#     # 注意：为了控制变量，测试用例的生成方式应保持不变
#     if op_type == 'cx':
#         all_combinations = list(itertools.combinations(evaluate_pop, 2))
#         if len(all_combinations) > 50:
#             rng = random.Random(seed)
#             test_cases = rng.sample(all_combinations, 50)
#         else:
#             test_cases = all_combinations
#     else:
#         test_cases = [(ind,) for ind in evaluate_pop]
#
#     # 3. 遍历评估每个算子
#     for op_ind in population:
#         # 【移除】不再获取 old_score，本轮评估互不干扰，无状态
#
#         fitness_gains_pct = []  # 仅记录相对提升率
#         valid_runs = 0
#         code_string = op_ind['code']
#
#         try:
#             # A. 编译
#             op_func = compile_and_load_code_string(code_string)
#             if not op_func:
#                 raise ValueError("编译失败")
#
#             # B. 运行测试用例
#             for test_case in test_cases:
#                 # 准备基准数据
#                 if op_type == 'cx':
#                     parent1_orig, parent2_orig = test_case
#                     base_fitness1 = max(parent1_orig.fitness.values[0], parent2_orig.fitness.values[0])
#                     base_fitness2 = max(parent1_orig.fitness.values[1], parent2_orig.fitness.values[1])
#
#                     # 准备环境副本 (只做一次)
#                     p1_copy = convert_individual_nodes_to_int(toolbox.clone(parent1_orig))
#                     p2_copy = convert_individual_nodes_to_int(toolbox.clone(parent2_orig))
#                 else:
#                     parent1_orig = test_case[0]
#                     base_fitness1 = parent1_orig.fitness.values[0]
#                     base_fitness2 = parent1_orig.fitness.values[1]
#
#                     p1_copy = convert_individual_nodes_to_int(toolbox.clone(parent1_orig))
#
#                 # --- 【移除】Monte Carlo 循环，直接执行一次 ---
#                 try:
#                     # 封装逻辑
#                     def _run_logic():
#                         if op_type == 'cx':
#                             return op_func(p1_copy, p2_copy, env)
#                         elif op_type == 'mt':
#                             return op_func(p1_copy, env)
#                         return None
#
#                     # 执行 (带超时)
#                     children = func_timeout(TIMEOUT_LIMIT, _run_logic)
#
#                     # 格式校验
#                     if not isinstance(children, (list, tuple)):
#                         children = [children]
#
#                     valid_children = [c for c in children if c and len(c) > 0]
#
#                     # 如果没有有效子代，直接跳过当前测试用例
#                     if not valid_children:
#                         continue
#
#                     # 评估当前测试用例的最佳子代
#                     best_case_gain = -float('inf')
#
#                     for child in valid_children:
#                         fit_values = toolbox.evaluate(child)
#                         child_fit1 = fit_values[0]
#                         child_fit2 = fit_values[1]
#
#                         # 计算相对提升率 (保持计算公式一致，以便公平对比)
#                         raw_diff1 = (child_fit1 - base_fitness1) / (abs(base_fitness1) + 1e-9)
#                         raw_diff2 = (child_fit2 - base_fitness2) / (abs(base_fitness2) + 1e-9)
#
#                         # 综合两个目标
#                         raw_diff = 0.7 * raw_diff1 + 0.3 * raw_diff2
#
#                         # 软惩罚
#                         gain = raw_diff if raw_diff >= 0 else raw_diff * SOFT_PENALTY_FACTOR
#
#                         if gain > best_case_gain:
#                             best_case_gain = gain
#
#                     # 记录结果
#                     if best_case_gain != -float('inf'):
#                         fitness_gains_pct.append(best_case_gain)
#                         valid_runs += 1
#
#                 except Exception:
#                     continue  # 单次运行失败直接忽略
#
#             # --- 4. 算子最终打分 (无状态版) ---
#             if valid_runs > 0 and fitness_gains_pct:
#                 # 直接计算平均值
#                 avg_imp = statistics.mean(fitness_gains_pct)
#
#                 # 直接赋值，不与 old_score 进行加权平均
#                 current_score = avg_imp * SCORE_SCALING
#
#                 op_ind['score'] = current_score
#                 # 记录 eval_count 仅用于统计，不参与计算
#                 op_ind['eval_count'] = op_ind.get('eval_count', 0) + 1
#
#                 print(f"ID:{op_ind.get('idx', '?')} | Baseline Score: {current_score:.4f}")
#             else:
#                 op_ind['score'] = HARD_PENALTY_SCORE
#
#         except Exception as e:
#             op_ind['score'] = HARD_PENALTY_SCORE
#             print(f"评估严重错误 ({op_ind.get('idx', 'unknown')}): {e}")
#
#     return population




# True Function
# def evaluate_operator(population: list, evaluate_pop: list, toolbox: Any, env: Any, op_type: str, seed: int = 42) -> list:
#     """
#     评估算子种群中每个算子的优越性。
#     【改良版】：引入 Monte Carlo 重复采样、相对提升率、以及 Hard Elitism (精英锁定)。
#     """
#
#     # --- 配置权重 (侧重稳定性) ---
#     W_MEAN_IMP = 0.4  # 提升
#     W_WIN_RATE = 0.3  # 胜率
#     W_MAX_IMP = 0.1  # 最大潜力
#     W_DIV = 0.2  # 多样性
#
#     # --- 运行配置 ---
#     REPEAT_RUNS = 3  # 【新增】每对父代跑 3 次取平均，消除算子内部随机性
#     SOFT_PENALTY_FACTOR = 0.01  # 失败惩罚系数
#     HARD_PENALTY_SCORE = -100.0
#     TIMEOUT_LIMIT = 20.0
#     # 1. 参数校验
#     valid_ops = ['cx', 'mt']
#     if op_type not in valid_ops:
#         raise ValueError(f"op_type 必须是 {valid_ops} 之一，当前为: {op_type}")
#
#     # 2. 准备测试用例 (固定种子采样，保持不变)
#     if op_type == 'cx':
#         all_combinations = list(itertools.combinations(evaluate_pop, 2))
#         if len(all_combinations) > 50:  # 稍微减少测试用例数量，因为我们增加了内部循环 REPEAT_RUNS
#             rng = random.Random(seed)  # 这里的 42 必须锁死
#             test_cases = rng.sample(all_combinations, 50)
#         else:
#             test_cases = all_combinations
#     else:
#         test_cases = [(ind,) for ind in evaluate_pop]
#
#     # 3. 遍历评估每个算子
#     for op_ind in population:
#
#         # 获取旧分 (用于 Elitism)
#         old_score = op_ind.get('score', HARD_PENALTY_SCORE)
#         if old_score <= HARD_PENALTY_SCORE:
#             old_score = None
#
#         fitness_gains_pct = []  # 记录百分比提升
#         diversity_scores = []
#         win_counts = []  # 记录胜场
#         valid_runs = 0
#
#         code_string = op_ind['code']
#
#         try:
#             # A. 编译
#             op_func = compile_and_load_code_string(code_string)
#             if not op_func:
#                 raise ValueError("编译失败")
#
#             # B. 运行测试用例
#             for test_case in test_cases:
#                 # 准备数据副本
#                 if op_type == 'cx':
#                     parent1_orig, parent2_orig = test_case
#                     base_fitness1 = max(parent1_orig.fitness.values[0], parent2_orig.fitness.values[0])
#                     base_fitness2 = max(parent1_orig.fitness.values[1], parent2_orig.fitness.values[1])
#                 else:
#                     parent1_orig = test_case[0]
#                     base_fitness1 = parent1_orig.fitness.values[0]
#                     base_fitness2 = parent1_orig.fitness.values[1]
#
#                 # --- 【核心修改】Monte Carlo 内部循环 ---
#                 # 针对同一对父代，重复运行 REPEAT_RUNS 次，取平均表现
#                 case_gains_buffer = []
#                 case_div_buffer = []
#                 case_wins = 0
#
#                 for _ in range(REPEAT_RUNS):
#                     try:
#                         # A. 准备数据：每次 clone 保证环境独立
#                         if op_type == 'cx':
#                             # 交叉需要两个父代
#                             p1_copy = convert_individual_nodes_to_int(toolbox.clone(parent1_orig))
#                             p2_copy = convert_individual_nodes_to_int(toolbox.clone(parent2_orig))
#                         elif op_type == 'mt':
#                             # 变异只需要一个父代
#                             p1_copy = convert_individual_nodes_to_int(toolbox.clone(parent1_orig))
#                         else:
#                             # 未知类型直接跳过
#                             continue
#
#                         # B. 执行逻辑封装 (用于 func_timeout)
#                         def _run_logic():
#                             if op_type == 'cx':
#                                 return op_func(p1_copy, p2_copy, env)
#                             elif op_type == 'mt':
#                                 # 直接调用 op_func(individual, env)
#                                 return op_func(p1_copy, env)
#                             return None
#
#                         # C. 执行算子 (带超时保护)
#                         children = func_timeout(TIMEOUT_LIMIT, _run_logic)
#
#                         # D. 结果格式校验
#                         if not isinstance(children, (list, tuple)):
#                             children = [children]
#
#                         # 过滤无效子代
#                         valid_children = [c for c in children if c and len(c) > 0]
#                         if not valid_children:
#                             continue
#
#                         # E. 评估当前 Run 的最佳子代
#                         best_run_gain = -float('inf')
#                         best_run_div = 0.0
#
#                         for child in valid_children:
#                             # 计算适应度
#                             fit_values = toolbox.evaluate(child)
#                             child_fit1 = fit_values[0]
#                             child_fit2 = fit_values[1]
#
#                             # 计算相对提升率 (Relative Improvement)
#                             # base_fitness1/2 在外部定义 (通常取自父代或父代最大值)
#                             raw_diff1 = (child_fit1 - base_fitness1) / (abs(base_fitness1) + 1e-9)
#                             raw_diff2 = (child_fit2 - base_fitness2) / (abs(base_fitness2) + 1e-9)
#
#                             # 综合得分 (0.7 视觉 + 0.3 需求)
#                             raw_diff = 0.7 * raw_diff1 + 0.3 * raw_diff2
#
#                             # 软惩罚 (Soft Penalty): 如果是负提升，放大惩罚
#                             gain = raw_diff if raw_diff >= 0 else raw_diff * SOFT_PENALTY_FACTOR
#
#                             if gain > best_run_gain:
#                                 best_run_gain = gain
#
#                             # 计算多样性 (Jaccard Distance)
#                             if op_type == 'cx':
#                                 d1 = calculate_jaccard_distance(child, parent1_orig)
#                                 d2 = calculate_jaccard_distance(child, parent2_orig)
#                                 div = min(d1, d2)  # 取与两个父代差异的最小值
#                             else:  # op_type == 'mt'
#                                 div = calculate_jaccard_distance(child, parent1_orig)
#
#                             if div > best_run_div:
#                                 best_run_div = div
#
#                         # F. 记录本轮 Run 的最佳结果
#                         if best_run_gain != -float('inf'):
#                             case_gains_buffer.append(best_run_gain)
#                             case_div_buffer.append(best_run_div)
#                             if best_run_gain > 0:  # 如果有正向提升，计一胜
#                                 case_wins += 1
#
#                     except Exception:
#                         continue  # 单次运行失败忽略，不中断评估
#
#                 # --- 结束 Monte Carlo 循环 ---
#
#                 # 如果该测试用例至少成功了一次
#                 if case_gains_buffer:
#                     valid_runs += 1
#                     # 取平均值作为该测试用例的最终成绩
#                     fitness_gains_pct.append(statistics.mean(case_gains_buffer))
#                     diversity_scores.append(statistics.mean(case_div_buffer))
#                     # 计算该测试用例的胜率 (例如跑3次赢2次，就是 0.66)
#                     win_counts.append(case_wins / REPEAT_RUNS)
#
#             # --- 4. 算子最终打分 ---
#             if valid_runs > 0 and fitness_gains_pct:
#                 avg_imp = statistics.mean(fitness_gains_pct)
#                 max_imp = max(fitness_gains_pct)
#                 avg_win_rate = statistics.mean(win_counts)
#                 avg_div = statistics.mean(diversity_scores)
#
#                 # 归一化/缩放
#                 # 假设相对提升率平均在 0.01~0.1 (1%~10%)，放大 1000 倍方便观察
#                 score_basis = avg_imp * 1000
#                 max_basis = max_imp * 200
#                 win_basis = avg_win_rate * 100  # 胜率 0~1 -> 0~10
#                 div_bonus = avg_div * 100
#
#                 # 计算本轮新得分
#                 current_round_score = (W_MEAN_IMP * score_basis) + \
#                                       (W_WIN_RATE * win_basis) + \
#                                       (W_MAX_IMP * max_basis) + \
#                                       (W_DIV * div_bonus)
#                 print(f"加权前：提升得分：{score_basis}，最大潜力得分：{max_basis}，胜率得分：{win_basis}，多样性得分：{div_bonus}，加权后总分：{current_round_score}")
#                 # 1. 获取该算子历史评估次数 (如果没有则为 0)
#                 eval_count = op_ind.get('eval_count', 0)
#
#                 if old_score is not None and eval_count > 0:
#                     # 核心公式：加权平均
#                     # 随着评估次数增加，单次运气的权重会被稀释，分数越来越接近真实值
#                     new_total_score = (old_score * eval_count + current_round_score)
#                     new_count = eval_count + 1
#                     final_score = new_total_score / new_count
#
#                     op_ind['score'] = final_score
#                     op_ind['eval_count'] = new_count
#
#                     # Debug 输出 (可选)
#                     # print(f"算子修正: {old_score:.2f} (n={eval_count}) + {current_round_score:.2f} -> {final_score:.2f}")
#                 else:
#                     # 第一次出现的新算子
#                     op_ind['score'] = current_round_score
#                     op_ind['eval_count'] = 1
#             else:
#                 op_ind['score'] = HARD_PENALTY_SCORE
#
#         except Exception as e:
#             op_ind['score'] = HARD_PENALTY_SCORE
#             print(f"评估严重错误 ({op_ind['idx']}): {e}")
#
#     return population

def load_operators_checkpoint(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]] or None:
    """尝试从文件加载交叉和变异精英算子。"""
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                # 文件应该包含 (cx_elitist, mt_elitist) 两个元素
                checkpoint = pickle.load(f)
                print(f"成功从 {path} 加载 LLM 算子检查点。")
                return checkpoint
        except Exception as e:
            print(f"加载算子检查点失败 ({e})，将重新生成。")
            return None
    return None

def save_operators_checkpoint(path: str, cx_elitist: Dict[str, Any], mt_elitist: List[Dict[str, Any]]):
    """保存交叉和变异精英算子到文件。"""
    try:
        with open(path, 'wb') as f:
            pickle.dump((cx_elitist, mt_elitist), f)
        print(f"LLM 算子已成功保存到检查点: {path}")
    except Exception as e:
        print(f"保存算子检查点失败: {e}")


def convert_individual_nodes_to_int(individual: Any) -> Any:
    if not individual:
        return individual

    # 1. 递归转换逻辑：处理嵌套的 List of Lists 结构
    def recursive_convert(data):
        if isinstance(data, list):
            # 如果列表里的第一个元素还是列表，继续往下走（处理个体层）
            # 如果列表里的第一个元素不是列表，说明到了线路层，开始转 int
            if len(data) > 0 and isinstance(data[0], list):
                return [recursive_convert(sub) for sub in data]
            else:
                return [int(node) for node in data]
        return data

    # 2. 执行转换
    converted_data = recursive_convert(individual)

    # 3. 尝试保留原始类（DEAP Individual 等）
    try:
        original_class = individual.__class__
        # 传入转换后的嵌套数据
        new_individual = original_class(converted_data)

        # 复制适应度 (fitness)
        if hasattr(individual, 'fitness'):
            new_individual.fitness = copy.copy(individual.fitness)

        return new_individual
    except (TypeError, AttributeError):
        # 如果不是 DEAP 类型或实例化失败，返回转换后的原生 list/nested list
        return converted_data





