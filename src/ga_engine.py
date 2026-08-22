import random
import networkx as nx
import numpy as np
import pandas as pd
import os
import json
import shutil
import time
from deap import base, creator, tools
from matplotlib import pyplot as plt
from pymoo.indicators.hv import HV

from evaluation import evaluate_individual
from config import GA_PARAMS, STRATEGIES, RESULTS_DIR, EXPERIMENT_ID, OPERATOR_SELECTION
from function import readText, logInfo, compile_and_load_code_string
from agent import (llm_tuning_agent, get_summary, llm_decide_agent,
                  extract_true_response, extract_code_block,
                  llm_select_operators, smooth_operator_weights)
from utils import create_compatible_env
import functools

# --- DEAP Setup ---
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))

if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)


class BusNetworkGA:
    def __init__(self, G, od_df, fixed_tasks, node_positions):
        self.G = G
        self.od_df = od_df
        self.fixed_tasks = fixed_tasks
        self.node_positions = node_positions
        self.toolbox = base.Toolbox()

        # 建立输出目录
        self._setup_workspace()

        # 注册算子
        self._setup_toolbox()

    def _setup_workspace(self):
        """初始化实验文件夹结构，并备份配置"""
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        # 创建用于存放每一代 Pareto 前沿的文件夹
        self.history_dir = os.path.join(RESULTS_DIR, "pareto_history")
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)

        print(f"--- 实验初始化: {EXPERIMENT_ID} ---")
        print(f"--- 结果存储于: {RESULTS_DIR} ---")

        # 保存本次实验的配置参数 (Snapshot)，方便复盘
        config_backup = {
            "STRATEGIES": STRATEGIES,
            "GA_PARAMS": GA_PARAMS
        }
        with open(os.path.join(RESULTS_DIR, "experiment_config.json"), 'w') as f:
            json.dump(config_backup, f, indent=4)

    def _get_path_by_strategy(self, start, end, strategy='physical'):
        """根据策略生成路径 (Utility)"""
        try:
            if strategy == 'visual':
                return nx.shortest_path(self.G, start, end, weight='visual_cost')
            elif strategy == 'demand':
                return nx.shortest_path(self.G, start, end, weight='demand_cost')
            else:
                return nx.shortest_path(self.G, start, end, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _init_individual(self):
        """
        初始化个体：根据 config 选择 'random' 或 'mixed'
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
                # 默认/Baseline: 全部物理最短路 (或 K-shortest，这里简化为 shortest)
                path = self._get_path_by_strategy(s, e, 'physical')

            # 容错
            if not path:
                path = self._get_path_by_strategy(s, e, 'physical')
            routes.append(path)

        return creator.Individual(routes)

    def _mutate_router(self, individual):
        """
        变异路由：根据 config 选择 'random' 或 'smart'
        """
        mode = STRATEGIES.get('MUTATION', 'random')

        if mode == 'smart':
            return self._smart_mutate(individual)
        else:
            return self._random_mutate(individual)

    def _random_mutate(self, individual):
        """Baseline: 随机截断重连 (仅考虑物理距离)"""
        idx = random.randint(0, len(individual) - 1)
        route = individual[idx]
        if len(route) < 4: return individual,
        try:
            cut_u = random.randint(0, len(route) - 3)
            cut_v = random.randint(cut_u + 2, len(route) - 1)
            u, v = route[cut_u], route[cut_v]
            # 盲目搜索：只找物理最短
            subpath = nx.shortest_path(self.G, u, v, weight='weight')
            individual[idx] = self._repair_route(route[:cut_u] + subpath + route[cut_v + 1:])
        except:
            pass
        return individual,

    def _smart_mutate(self, individual):
        """Ours: 启发式变异"""
        probs = STRATEGIES.get('MUTATION_PROBS', {'visual': 0.33, 'demand': 0.33, 'smooth': 0.34})

        idx = random.randint(0, len(individual) - 1)
        route = individual[idx]
        if len(route) < 4: return individual,

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
            individual[idx] = self._repair_route(route[:cut_u] + subpath + route[cut_v + 1:])
        except:
            pass
        return individual,

    def _repair_route(self, route):
        if not route: return route
        seen, new_route = {}, []
        for node in route:
            if node in seen:
                new_route = new_route[:seen[node] + 1]
            else:
                seen[node] = len(new_route)
                new_route.append(node)
        return new_route

    def _setup_toolbox(self):
        self.toolbox.register("individual", self._init_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", evaluate_individual, G=self.G, od_df=self.od_df,
                              node_positions=self.node_positions, fixed_tasks=self.fixed_tasks)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_router)  # 路由到具体变异逻辑
        self.toolbox.register("select", tools.selNSGA2)

    def run(self, pop_size=GA_PARAMS['POP_SIZE'], n_gen=GA_PARAMS['NGEN'],
            out_dir=None, algo_label="NSGA2",
            run_id=None, seed=None):
        """
        手动展开的遗传算法主循环
        提供极高自由度的日志记录

        :param out_dir: 实验输出目录；None 时回退到 config.RESULTS_DIR
        :param algo_label: 算法标签（NSGA2 / LEHHA），写入 run_summary
        :param run_id: 运行编号
        :param seed: 随机种子
        """
        import config
        out_dir = out_dir if out_dir else config.RESULTS_DIR
        os.makedirs(out_dir, exist_ok=True)
        self._out_dir = out_dir
        # 每代数据写到 run 目录而非默认 RESULTS_DIR（便于按 run 隔离 + R 脚本读取）
        self.history_dir = os.path.join(out_dir, "pareto_history")
        os.makedirs(self.history_dir, exist_ok=True)
        # 暂存每代 nested 格式记录，run() 末尾统一写出 {algo}_convergence.json
        self._convergence_records = []
        self._algo_label = algo_label
        self._run_id = run_id
        self._seed = seed
        self._started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._run_start_ts = time.perf_counter()
        stats_visual = tools.Statistics(key=lambda ind: ind.fitness.values[0])

        stats_visual.register('min_visual', np.min)
        stats_visual.register('q1_visual', lambda x: np.percentile(x, 25))
        stats_visual.register('median_visual', np.median)
        stats_visual.register('q3_visual', lambda x: np.percentile(x, 75))
        stats_visual.register('max_visual', np.max)
        stats_visual.register('mean_visual', np.mean)

        stats_satisfaction = tools.Statistics(key=lambda ind: ind.fitness.values[1])

        stats_satisfaction.register('min_satisfy', np.min)
        stats_satisfaction.register('q1_satisfy', lambda x: np.percentile(x, 25))
        stats_satisfaction.register('median_satisfy', np.median)
        stats_satisfaction.register('q3_satisfy', lambda x: np.percentile(x, 75))
        stats_satisfaction.register('max_satisfy', np.max)
        stats_satisfaction.register('mean_satisfy', np.mean)
        Problem_Description = readText("./description/description.txt")

        # ==========================================
        # 0. 辅助函数：批量加载并编译算子
        # ==========================================
        def load_operator_functions(file_paths, op_name_prefix):
            """
            加载指定路径的代码文件，编译成函数列表。
            返回: 成功编译的函数列表
            """
            compiled_funcs = []
            for tag, path in file_paths.items():
                if os.path.exists(path):
                    try:
                        code = readText(path)  # 假设 readText 已定义
                        func = compile_and_load_code_string(code)
                        if func:
                            compiled_funcs.append(func)
                            print(f"✅ [{op_name_prefix}] 成功加载 {tag} 算子: {path}")
                        else:
                            print(f"⚠️ [{op_name_prefix}] 编译返回空: {path}")
                    except Exception as e:
                        print(f"❌ [{op_name_prefix}] 加载失败 {path}: {e}")
                else:
                    print(f"⚠️ [{op_name_prefix}] 文件不存在: {path}")
            return compiled_funcs

        # ==========================================
        # 1. 定义多策略动态交叉 Wrapper（支持权重）
        # ==========================================
        def dynamic_hybrid_mate_wrapper(ind1, ind2, llm_funcs, default_func, env, cx_weights, cx_usage_count):
            """
            动态调度：根据权重选择交叉算子

            参数:
                cx_weights: {"overall": 0.25, "visual": 0.25, "demand": 0.25, "default": 0.25}
                cx_usage_count: 算子使用计数字典（引用传递）
            """
            # 算子名称列表
            operator_names = ["overall", "visual", "demand", "default"]

            # 根据权重采样
            selected_name = random.choices(
                operator_names,
                weights=[cx_weights[name] for name in operator_names],
                k=1
            )[0]

            # 记录使用次数
            cx_usage_count[selected_name] += 1

            # 选择对应函数
            if selected_name == "default":
                selected_func = default_func
            else:
                # llm_funcs 顺序对应 ["overall", "visual", "demand"]
                idx = operator_names.index(selected_name)
                selected_func = llm_funcs[idx]

            # 执行
            try:
                children = selected_func(ind1, ind2, env)
            except Exception as e:
                # 失败时回退到默认算子
                selected_func = default_func
                children = selected_func(ind1, ind2)

            # 格式修正 (DEAP 要求返回 tuple)
            if isinstance(children, list): return tuple(children)
            if not isinstance(children, tuple): return (children,)
            return children

        # ==========================================
        # 2. 定义多策略动态变异 Wrapper（支持权重）
        # ==========================================
        def dynamic_hybrid_mutate_wrapper(ind, llm_funcs, default_func, env, mt_weights, mt_usage_count):
            """
            动态调度：根据权重选择变异算子

            参数:
                mt_weights: {"overall": 0.25, "visual": 0.25, "demand": 0.25, "default": 0.25}
                mt_usage_count: 算子使用计数字典（引用传递）
            """
            operator_names = ["overall", "visual", "demand", "default"]

            selected_name = random.choices(
                operator_names,
                weights=[mt_weights[name] for name in operator_names],
                k=1
            )[0]

            mt_usage_count[selected_name] += 1

            if selected_name == "default":
                selected_func = default_func
            else:
                idx = operator_names.index(selected_name)
                selected_func = llm_funcs[idx]

            try:
                result = selected_func(ind, env)
            except Exception as e:
                selected_func = default_func
                result = selected_func(ind)

            if isinstance(result, tuple): return result
            if isinstance(result, list): return (result,)
            return (ind,)

        # ==========================================
        # 3. 加载代码并注册到 Toolbox
        # ==========================================

        # --- A. 准备文件路径 ---
        # 算子来源控制：
        #   LEHHA → 加载 temp/ 下 6 个 LLM 进化算子
        #   其他（NSGA2 等基线）→ 空字典触发 fallback，使用 DEAP 默认（cxTwoPoint + _mutate_router）
        if self._algo_label == "LEHHA":
            cx_files = {
                "Overall": "./temp/cx_elitist_overall.py",
                "Visual": "./temp/cx_elitist_visual.py",
                "Demand": "./temp/cx_elitist_demand.py"
            }
            mt_files = {
                "Overall": "./temp/mt_elitist_overall.py",
                "Visual": "./temp/mt_elitist_visual.py",
                "Demand": "./temp/mt_elitist_demand.py"
            }
        else:
            print(f"\n[{self._algo_label} 基线模式] 跳过加载 LLM 进化算子，使用 DEAP 默认算子"
                  f"（mate=cxTwoPoint, mutate=_mutate_router）")
            cx_files = {}
            mt_files = {}

        # --- B. 加载并编译算子 ---
        print("-" * 30)
        print("正在初始化多策略算子池...")
        valid_cx_funcs = load_operator_functions(cx_files, "CX")
        valid_mt_funcs = load_operator_functions(mt_files, "MT")

        # ==========================================
        # 算子动态选择：初始化权重和使用计数
        # ==========================================
        operator_weights_history = []

        # 处理消融模式
        ablation_mode = OPERATOR_SELECTION.get('ablation_mode', None)
        if ablation_mode:
            # 消融模式：只使用某一类算子
            print(f"\n⚠️  消融实验模式：只使用 {ablation_mode} 算子")
            single_weight = 1.0
            zero_weight = 0.0

            initial_cx_weights = {
                "overall": single_weight if ablation_mode == "overall_only" else zero_weight,
                "visual": single_weight if ablation_mode == "visual_only" else zero_weight,
                "demand": single_weight if ablation_mode == "demand_only" else zero_weight,
                "default": single_weight if ablation_mode == "default_only" else zero_weight
            }
            initial_mt_weights = initial_cx_weights.copy()

        elif not OPERATOR_SELECTION.get('enabled', True):
            # LLM选择功能关闭，使用固定权重
            print("\n⚠️  LLM算子选择已禁用，使用固定权重")
            initial_cx_weights = OPERATOR_SELECTION['fixed_weights']['cx'].copy()
            initial_mt_weights = OPERATOR_SELECTION['fixed_weights']['mt'].copy()

        else:
            # 默认：等权重初始化
            initial_cx_weights = {"overall": 0.25, "visual": 0.25, "demand": 0.25, "default": 0.25}
            initial_mt_weights = {"overall": 0.25, "visual": 0.25, "demand": 0.25, "default": 0.25}

        # 当前使用的权重（可变，通过引用传递给wrapper）
        cx_weights = initial_cx_weights.copy()
        mt_weights = initial_mt_weights.copy()

        # 使用计数（用于记录实际使用率）
        cx_usage_count = {"overall": 0, "visual": 0, "demand": 0, "default": 0}
        mt_usage_count = {"overall": 0, "visual": 0, "demand": 0, "default": 0}

        print(f"初始交叉权重: {cx_weights}")
        print(f"初始变异权重: {mt_weights}")

        # --- C. 注册交叉算子 (CX) ---
        # 确保环境对象 env 存在
        if not hasattr(self, 'env_cache'):
            self.env_cache = create_compatible_env(self)

        # 注册 hybrid mate
        if valid_cx_funcs:
            print(f"交叉算子池已构建: {len(valid_cx_funcs)} LLM + 1 Default")
            # 使用容器来传递可变权重（因为functools.partial不会更新参数）
            class WeightContainer:
                def __init__(self, weights, usage_count):
                    self.weights = weights
                    self.usage_count = usage_count

            cx_container = WeightContainer(cx_weights, cx_usage_count)
            mt_container = WeightContainer(mt_weights, mt_usage_count)

            dynamic_cx = functools.partial(
                dynamic_hybrid_mate_wrapper,
                llm_funcs=valid_cx_funcs,
                default_func=tools.cxTwoPoint,
                env=self.env_cache,
                cx_weights=cx_container.weights,
                cx_usage_count=cx_container.usage_count
            )
            self.toolbox.register("mate", dynamic_cx)
        else:
            print("没有可用的 LLM 交叉算子，使用默认算子。")
            self.toolbox.register("mate", tools.cxTwoPoint)
            # 创建容器避免后续错误
            class WeightContainer:
                def __init__(self):
                    self.weights = {"default": 1.0}
                    self.usage_count = {"default": 0}
            cx_container = WeightContainer()
            mt_container = WeightContainer()

        # --- D. 注册变异算子 (MT) ---
        default_mutation = self._mutate_router if hasattr(self, '_mutate_router') else tools.mutShuffleIndexes

        if valid_mt_funcs:
            print(f"变异算子池已构建: {len(valid_mt_funcs)} LLM + 1 Default")
            dynamic_mt = functools.partial(
                dynamic_hybrid_mutate_wrapper,
                llm_funcs=valid_mt_funcs,
                default_func=default_mutation,
                env=self.env_cache,
                mt_weights=mt_container.weights,
                mt_usage_count=mt_container.usage_count
            )
            self.toolbox.register("mutate", dynamic_mt)
        else:
            print("没有可用的 LLM 变异算子，使用默认算子。")
            self.toolbox.register("mutate", default_mutation)

        print("算子动态加载完成。")
        print("-" * 30)

        cxProb = GA_PARAMS['CXPB']
        mutateProb = GA_PARAMS['MUTPB']
        # 1. 初始化种群
        pop = self.toolbox.population(n=pop_size)

        # 2. 初始评估
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 记录初始种群状态
        pop = self.toolbox.select(pop, len(pop))  # NSGA2 排序
        self._log_generation(0, pop)

        print(f"开始进化: {n_gen} 代...")
        evolution_history = []
        ref_point = np.array([0.0, 0.0])

        # 3. 进化循环
        gen = 1
        while gen <= n_gen:
            # A. 选择与克隆
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # B. 交叉与变异
            for i in range(0, len(offspring), 2):
                if i + 1 >= len(offspring): break
                if random.random() <= cxProb:
                    child1, child2 = self.toolbox.mate(offspring[i], offspring[i + 1])
                    offspring[i][:] = offspring[i].__class__(child1)
                    offspring[i + 1][:] = offspring[i + 1].__class__(child2)
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values

            for i in range(len(offspring)):
                if random.random() <= mutateProb:
                    child, = self.toolbox.mutate(offspring[i])
                    offspring[i][:] = offspring[i].__class__(child)
                    del offspring[i].fitness.values

            # C. 评估新后代
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # D. 环境选择
            pop = self.toolbox.select(pop + offspring, pop_size)

            # =========================================================
            # 计算并记录每一代的收敛指标
            # =========================================================

            # 1. 获取目标函数值
            fits_obj1 = [ind.fitness.values[0] for ind in pop]
            fits_obj2 = [ind.fitness.values[1] for ind in pop]

            # 2. 计算 Pareto 前沿 & Hypervolume
            non_dominated_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

            try:
                front_objs = np.array([ind.fitness.values for ind in non_dominated_front])
                ind_hv = HV(ref_point=ref_point)
                current_hv = ind_hv.do(-front_objs)
                print(f"HV: {current_hv}")  # 保留每轮打印HV
            except Exception as e:
                print(f"HV 计算出错: {e}")
                current_hv = 0.0

            # 3. 组装统计数据（用于logInfo）
            record_pop = {
                "min_visual": float(np.min(fits_obj1)),
                "q1_visual": float(np.percentile(fits_obj1, 25)),
                "median_visual": float(np.median(fits_obj1)),
                "q3_visual": float(np.percentile(fits_obj1, 75)),
                "max_visual": float(np.max(fits_obj1)),
                "mean_visual": float(np.mean(fits_obj1)),
                "min_satisfy": float(np.min(fits_obj2)),
                "q1_satisfy": float(np.percentile(fits_obj2, 25)),
                "median_satisfy": float(np.median(fits_obj2)),
                "q3_satisfy": float(np.percentile(fits_obj2, 75)),
                "max_satisfy": float(np.max(fits_obj2)),
                "mean_satisfy": float(np.mean(fits_obj2))
            }

            record_fronts = {
                "hypervolume": current_hv,
                "front_count": len(non_dominated_front)
            }

            # 4. 记录到 evolution_history
            evolution_history.append(
                logInfo(
                    [record_pop, record_fronts],
                    n_gen,
                    pop_size,
                    cxProb,
                    mutateProb,
                    gen,
                    cx_weights=cx_container.weights if 'cx_container' in locals() else None,
                    mt_weights=mt_container.weights if 'mt_container' in locals() else None,
                    cx_usage_count=cx_container.usage_count if 'cx_container' in locals() else None,
                    mt_usage_count=mt_container.usage_count if 'mt_container' in locals() else None
                )
            )

            # =========================================================

            # E. 详细日志记录
            self._log_generation(gen, pop, hv=current_hv)

            # =========================================================
            # 算子动态选择：每N轮调用LLM
            # =========================================================
            call_interval = OPERATOR_SELECTION.get('call_interval', 10)
            start_gen = OPERATOR_SELECTION.get('start_gen', 20)
            enabled = OPERATOR_SELECTION.get('enabled', True)
            no_ablation = OPERATOR_SELECTION.get('ablation_mode', None) is None

            if enabled and no_ablation and gen >= start_gen and gen % call_interval == 0 and gen != n_gen:
                print(f"\n{'='*70}")
                print(f"[Gen {gen}] 触发 LLM 算子选择...")
                print(f"{'='*70}\n")

                # --- 1. 计算双窗口对比数据 ---
                def extract_window_stats(window_data):
                    """提取窗口统计信息"""
                    if not window_data or len(window_data) == 0:
                        return None

                    first = window_data[0]
                    last = window_data[-1]

                    # HV增长
                    hv_first = first['pareto_fronts']['hypervolume']
                    hv_last = last['pareto_fronts']['hypervolume']
                    hv_growth = hv_last - hv_first
                    hv_growth_pct = (hv_growth / (hv_first + 1e-9)) * 100

                    # 停滞步数
                    stagnation = 0
                    for i in range(1, len(window_data)):
                        curr_hv = window_data[i]['pareto_fronts']['hypervolume']
                        prev_hv = window_data[i-1]['pareto_fronts']['hypervolume']
                        step_growth = (curr_hv - prev_hv) / (prev_hv + 1e-9)
                        if step_growth < 0.0001:
                            stagnation += 1

                    # 前沿数量变化
                    front_first = first['pareto_fronts']['front_count']
                    front_last = last['pareto_fronts']['front_count']
                    front_change = front_last - front_first

                    # 适应度中位数变化
                    visual_median_change = (
                        last['effect']['median_visual'] - first['effect']['median_visual']
                    )
                    satisfy_median_change = (
                        last['effect']['median_satisfy'] - first['effect']['median_satisfy']
                    )

                    # IQR变化
                    iqr_first = first['effect']['q3_visual'] - first['effect']['q1_visual']
                    iqr_last = last['effect']['q3_visual'] - last['effect']['q1_visual']
                    iqr_change = iqr_last - iqr_first

                    return {
                        "hv_growth": hv_growth,
                        "hv_growth_pct": hv_growth_pct,
                        "stagnation": stagnation,
                        "front_change": front_change,
                        "visual_median_change": visual_median_change,
                        "satisfy_median_change": satisfy_median_change,
                        "visual_iqr_change": iqr_change
                    }

                # 窗口1：刚结束的10轮
                window1_start = gen - call_interval + 1
                window1_end = gen
                window1_data = evolution_history[window1_start - 1:window1_end]
                window1_stats = extract_window_stats(window1_data)

                # 窗口2：之前的10轮
                window2_start = window1_start - call_interval
                window2_end = window1_start - 1
                window2_data = evolution_history[window2_start - 1:window2_end] if window2_start >= 1 else []
                window2_stats = extract_window_stats(window2_data)

                # ==========================================
                # 2. 获取权重字符串
                # ==========================================
                if gen == start_gen:
                    # 首次调用，窗口2使用默认等权重
                    window2_cx_weights_str = "overall:0.25, visual:0.25, demand:0.25, default:0.25"
                    window2_mt_weights_str = "overall:0.25, visual:0.25, demand:0.25, default:0.25"
                else:
                    # 从 operator_weights_history 获取历史权重
                    # 取倒数第2条记录（倒数第1条是当前窗口1刚用过的）
                    hist_idx = -2 if len(operator_weights_history) >= 2 else -1
                    if hist_idx == -2:
                        prev_entry = operator_weights_history[-2]
                    else:
                        # 如果历史记录不足，回退到初始配置
                        prev_entry = {"cx_weights": initial_cx_weights, "mt_weights": initial_mt_weights}

                    window2_cx = prev_entry["cx_weights"]
                    window2_mt = prev_entry["mt_weights"]
                    window2_cx_weights_str = ", ".join([f"{k}:{v:.2f}" for k, v in window2_cx.items()])
                    window2_mt_weights_str = ", ".join([f"{k}:{v:.2f}" for k, v in window2_mt.items()])

                # 当前窗口1的权重 (直接从容器获取当前状态)
                window1_cx_weights_str = ", ".join([f"{k}:{v:.2f}" for k, v in cx_container.weights.items()])
                window1_mt_weights_str = ", ".join([f"{k}:{v:.2f}" for k, v in mt_container.weights.items()])

                # ==========================================
                # 3. 预计算所有描述性文本 (关键修改：避免在模板中写逻辑)
                # ==========================================

                # --- 基础数值 ---
                phase_status = "EXPLORATION" if gen < n_gen * 0.75 else "CONVERGENCE"
                phase_desc_cn = "探索期" if phase_status == "EXPLORATION" else "收敛期"

                # 安全获取窗口2的数据（防止第一代为空）
                w2_hv_pct = window2_stats["hv_growth_pct"] if window2_stats else 0.0
                w2_stag = window2_stats["stagnation"] if window2_stats else 0
                w2_vis_med = window2_stats["visual_median_change"] if window2_stats else 0.0
                w2_sat_med = window2_stats["satisfy_median_change"] if window2_stats else 0.0
                w2_iqr = window2_stats["visual_iqr_change"] if window2_stats else 0.0
                w2_front = window2_stats["front_change"] if window2_stats else 0

                # --- 差异计算 ---
                hv_growth_diff = window1_stats["hv_growth_pct"] - w2_hv_pct
                stagnation_diff = window1_stats["stagnation"] - w2_stag

                # --- 1. HV增长差异描述 ---
                hv_diff_desc = "优于" if hv_growth_diff > 0 else "差于"

                # --- 2. 停滞改善描述 ---
                if stagnation_diff < 0:
                    stagnation_desc = "窗口1更少停滞"
                elif stagnation_diff > 0:
                    stagnation_desc = "窗口2更少停滞"
                else:
                    stagnation_desc = "停滞情况持平"

                # --- 3. 视觉/客流进展对比描述 ---
                visual_progress_desc = "窗口1更优" if window1_stats["visual_median_change"] > w2_vis_med else "窗口2更优"
                demand_progress_desc = "窗口1更优" if window1_stats["satisfy_median_change"] > w2_sat_med else "窗口2更优"

                # --- 4. HV增长率等级评价 ---
                w1_hv_pct = window1_stats["hv_growth_pct"]
                if w1_hv_pct > 5:
                    hv_rank_desc = "优秀(>5%)"
                elif w1_hv_pct > 2:
                    hv_rank_desc = "良好(2-5%)"
                elif w1_hv_pct > 1:
                    hv_rank_desc = "一般(1-2%)"
                else:
                    hv_rank_desc = "停滞(<1%)"

                # --- 5. 停滞步数健康度评价 ---
                w1_stag = window1_stats["stagnation"]
                if w1_stag <= 3:
                    stagnation_health_desc = "健康(≤3)"
                elif w1_stag <= 6:
                    stagnation_health_desc = "边缘(4-6)"
                else:
                    stagnation_health_desc = "异常(≥7)"

                # --- 6. 相对表现评价 ---
                if hv_growth_diff > 0:
                    relative_perf_desc = "表现更好"
                elif abs(hv_growth_diff) < 2:
                    relative_perf_desc = "表现相近"
                else:
                    relative_perf_desc = "表现更差"

                # --- 7. 增长预期符合度评价 ---
                is_converging = (phase_status == 'CONVERGENCE')
                is_exploring = (phase_status == 'EXPLORATION')

                if (is_converging and w1_hv_pct > 1) or (is_exploring and w1_hv_pct > 3):
                    expectation_desc = "符合预期"
                else:
                    expectation_desc = "低于预期"

                # ==========================================
                # 4. 构建 LLM 输入字典
                # ==========================================
                llm_input_info = {
                    "problem_desc": readText("./description/description.txt"),
                    "current_gen": gen,
                    "total_gen": n_gen,
                    "phase_status": phase_status,

                    # 窗口1数据
                    "window1_start": window1_start,
                    "window1_end": window1_end,
                    "window1_cx_weights": window1_cx_weights_str,
                    "window1_mt_weights": window1_mt_weights_str,
                    "window1_hv_growth": window1_stats["hv_growth"],
                    "window1_hv_growth_pct": w1_hv_pct,
                    "window1_stagnation": w1_stag,
                    "window1_front_change": window1_stats["front_change"],
                    "window1_visual_median_change": window1_stats["visual_median_change"],
                    "window1_satisfy_median_change": window1_stats["satisfy_median_change"],
                    "window1_visual_iqr_change": window1_stats["visual_iqr_change"],

                    # 窗口2数据
                    "window2_start": window2_start,
                    "window2_end": window2_end,
                    "window2_cx_weights": window2_cx_weights_str,
                    "window2_mt_weights": window2_mt_weights_str,
                    "window2_hv_growth": window2_stats["hv_growth"] if window2_stats else 0.0,
                    "window2_hv_growth_pct": w2_hv_pct,
                    "window2_stagnation": w2_stag,
                    "window2_front_change": w2_front,
                    "window2_visual_median_change": w2_vis_med,
                    "window2_satisfy_median_change": w2_sat_med,
                    "window2_visual_iqr_change": w2_iqr,

                    # 预先计算好的描述性文本 (直接用于模板替换)
                    "hv_growth_diff": hv_growth_diff,
                    "stagnation_diff": stagnation_diff,

                    "hv_diff_desc": hv_diff_desc,  # 对应 {hv_diff_desc}
                    "stagnation_desc": stagnation_desc,  # 对应 {stagnation_desc}
                    "visual_progress_desc": visual_progress_desc,  # 对应 {visual_progress_desc}
                    "demand_progress_desc": demand_progress_desc,  # 对应 {demand_progress_desc}

                    "hv_rank_desc": hv_rank_desc,  # 对应 {hv_rank_desc}
                    "stagnation_health_desc": stagnation_health_desc,  # 对应 {stagnation_health_desc}
                    "relative_perf_desc": relative_perf_desc,  # 对应 {relative_perf_desc}
                    "phase_desc_cn": phase_desc_cn,  # 对应 {phase_desc_cn}
                    "expectation_desc": expectation_desc,  # 对应 {expectation_desc}

                    # 未来窗口
                    "next_gen_start": gen + 1,
                    "next_gen_end": min(gen + call_interval, n_gen)
                }

                # --- 2. 调用 LLM ---
                try:
                    smoothing_params = OPERATOR_SELECTION.get('smoothing', {})
                    llm_result = llm_select_operators(llm_input_info)

                    if llm_result and "cx_weights" in llm_result:
                        # --- 3. 权重平滑 ---
                        new_cx_weights = smooth_operator_weights(
                            cx_container.weights,
                            llm_result.get("cx_weights", cx_container.weights),
                            min_val=smoothing_params.get('min_weight', 0.05),
                            max_val=smoothing_params.get('max_weight', 0.7),
                            max_change=smoothing_params.get('max_change', 0.2)
                        )
                        new_mt_weights = smooth_operator_weights(
                            mt_container.weights,
                            llm_result.get("mt_weights", mt_container.weights),
                            min_val=smoothing_params.get('min_weight', 0.05),
                            max_val=smoothing_params.get('max_weight', 0.7),
                            max_change=smoothing_params.get('max_change', 0.2)
                        )

                        phase = llm_result.get("phase", "unknown")
                        rationale = llm_result.get("rationale", "")

                        print(f"  [LLM分析] 阶段: {phase}")
                        print(f"  [LLM分析] 理由: {rationale[:200]}...")
                        print(f"  [权重更新] 交叉: {cx_container.weights}")
                        print(f"          --> {new_cx_weights}")
                        print(f"  [权重更新] 变异: {mt_container.weights}")
                        print(f"          --> {new_mt_weights}")

                        # --- 4. 更新当前权重 ---
                        cx_container.weights.clear()
                        cx_container.weights.update(new_cx_weights)
                        mt_container.weights.clear()
                        mt_container.weights.update(new_mt_weights)

                    else:
                        print(f"  [警告] LLM返回结果无效，保持当前权重不变")
                        phase = "error"
                        rationale = "LLM返回结果无效"

                except Exception as e:
                    print(f"  [错误] LLM调用失败: {e}")
                    print(f"  [回退] 保持当前权重不变")
                    phase = "error"
                    rationale = f"LLM调用失败: {str(e)}"

                # --- 5. 记录到历史 ---
                operator_weights_history.append({
                    "gen": gen,
                    "cx_weights": cx_container.weights.copy(),
                    "mt_weights": mt_container.weights.copy(),
                    "phase": phase,
                    "rationale": rationale
                })

                print(f"{'='*70}\n")

            # 其他代数只打印简要信息
            if gen % 10 == 0 and not (enabled and no_ablation and gen >= start_gen and gen % call_interval == 0 and gen != n_gen):
                print(f"Gen {gen}/{n_gen} 完成. 当前 Pareto 前沿数量: {len(self._get_front(pop))}")

            gen = gen + 1

        print("优化结束。")

        # =========================================================
        # 保存收敛数据到 JSON
        # =========================================================
        output_path = os.path.join(RESULTS_DIR, "convergence_metrics_test.json")

        # 确保 directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(evolution_history, f, indent=4, ensure_ascii=False)
            print(f"\n收敛性分析数据已保存至: {output_path}")
        except Exception as e:
            print(f"\n保存 JSON 失败: {e}")


        # 将最终的帕累托第0层前沿记录到文件

        print("\n正在将帕累托第0层前沿保存到文件...")
        front_sort = tools.emo.sortNondominated(pop, len(pop))
        # 1. 定义要保存的数据
        pareto_front_data = []
        # front_sort[0] 就是帕累托第0层前沿
        for idx, individual in enumerate(front_sort[0]):
            # 为每个个体创建一个字典，包含其索引、路径和适应度
            individual_info = {
                "index": idx,
                "path": [[int(node) for node in route] for route in individual],  # 个体的基因序列，即路径节点列表
                "fitness": {
                    "visual": individual.fitness.values[0],
                    "satisfy": individual.fitness.values[1]
                }
            }
            pareto_front_data.append(individual_info)

        # 2. 定义文件名和路径
        # 使用注入的实验目录（T8 钩子）
        output_dir = getattr(self, "_out_dir", None) or 'results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 文件名包含一些关键参数，方便识别
        file_path = os.path.join(output_dir, "final_pareto_legacy.json")

        # 3. 将数据写入JSON文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(pareto_front_data, f, ensure_ascii=False, indent=4)

        print(f"帕累托前沿已成功保存到: {file_path}")
        print(f"共保存了 {len(pareto_front_data)} 个非支配解。")

        # =========================================================
        # 保存算子权重历史
        # =========================================================
        if operator_weights_history:
            operator_weights_path = os.path.join(RESULTS_DIR, "operator_weights_history_test.json")
            try:
                with open(operator_weights_path, "w", encoding="utf-8") as f:
                    json.dump(operator_weights_history, f, indent=4, ensure_ascii=False)
                print(f"\n算子权重历史已保存至: {operator_weights_path}")
            except Exception as e:
                print(f"\n保存权重历史失败: {e}")

            # =========================================================
            # 绘制算子权重变化趋势
            # =========================================================
            try:
                print("\n正在绘制算子权重变化趋势...")

                # 提取数据
                gens = [entry["gen"] for entry in operator_weights_history]

                # 交叉算子权重
                cx_overall = [entry["cx_weights"]["overall"] for entry in operator_weights_history]
                cx_visual = [entry["cx_weights"]["visual"] for entry in operator_weights_history]
                cx_demand = [entry["cx_weights"]["demand"] for entry in operator_weights_history]
                cx_default = [entry["cx_weights"]["default"] for entry in operator_weights_history]

                # 变异算子权重
                mt_overall = [entry["mt_weights"]["overall"] for entry in operator_weights_history]
                mt_visual = [entry["mt_weights"]["visual"] for entry in operator_weights_history]
                mt_demand = [entry["mt_weights"]["demand"] for entry in operator_weights_history]
                mt_default = [entry["mt_weights"]["default"] for entry in operator_weights_history]

                # 创建双子图
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

                # 子图1：交叉算子权重
                ax1.stackplot(gens, cx_overall, cx_visual, cx_demand, cx_default,
                              labels=['Overall', 'Visual', 'Demand', 'Default'],
                              alpha=0.8, colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                ax1.set_title('Crossover Operator Weights Over Generations', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Generation', fontsize=12)
                ax1.set_ylabel('Weight', fontsize=12)
                ax1.legend(loc='upper left', fontsize=10)
                ax1.set_ylim(0, 1)
                ax1.grid(True, linestyle=':', alpha=0.3, axis='y')

                # 添加权重数值标签（每隔一个点）
                for i, gen in enumerate(gens):
                    if i % 2 == 0:  # 每隔一个点显示
                        y_offset = 0
                        for name, vals in [('Overall', cx_overall), ('Visual', cx_visual),
                                          ('Demand', cx_demand), ('Default', cx_default)]:
                            if vals[i] > 0.05:  # 只显示较大的权重
                                ax1.text(gen, y_offset + vals[i]/2, f'{vals[i]:.2f}',
                                        ha='center', va='center', fontsize=8, color='white')
                            y_offset += vals[i]

                # 子图2：变异算子权重
                ax2.stackplot(gens, mt_overall, mt_visual, mt_demand, mt_default,
                              labels=['Overall', 'Visual', 'Demand', 'Default'],
                              alpha=0.8, colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                ax2.set_title('Mutation Operator Weights Over Generations', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Generation', fontsize=12)
                ax2.set_ylabel('Weight', fontsize=12)
                ax2.legend(loc='upper left', fontsize=10)
                ax2.set_ylim(0, 1)
                ax2.grid(True, linestyle=':', alpha=0.3, axis='y')

                # 添加权重数值标签
                for i, gen in enumerate(gens):
                    if i % 2 == 0:
                        y_offset = 0
                        for name, vals in [('Overall', mt_overall), ('Visual', mt_visual),
                                          ('Demand', mt_demand), ('Default', mt_default)]:
                            if vals[i] > 0.05:
                                ax2.text(gen, y_offset + vals[i]/2, f'{vals[i]:.2f}',
                                        ha='center', va='center', fontsize=8, color='white')
                            y_offset += vals[i]

                plt.tight_layout()

                # 保存
                weights_plot_path = os.path.join(RESULTS_DIR, "operator_weights_trend.png")
                plt.savefig(weights_plot_path, dpi=150, bbox_inches='tight')
                print(f"算子权重趋势图已保存至: {weights_plot_path}")

                # plt.show()  # 可选：显示图像

            except Exception as e:
                print(f"\n绘制权重趋势图失败: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("\n没有权重历史数据（LLM算子选择未启用或未触发），跳过保存和绘图。")

        # # =============参数变化曲线=====================
        # try:
        #     # import matplotlib.pyplot as plt
        #
        #     print("\nStarting to plot parameter evolution graph...")
        #
        #     # 1. 提取数据
        #     # 确保 evolution_history 不为空
        #     if evolution_history:
        #         iterations = [entry['iteration'] for entry in evolution_history]
        #         pop_sizes = [entry['method']['pop_size'] for entry in evolution_history]
        #         cx_probs = [entry['method']['cxProb'] for entry in evolution_history]
        #         mut_probs = [entry['method']['mutateProb'] for entry in evolution_history]
        #
        #         # 2. 设置画布和左侧坐标轴 (Population Size)
        #         fig, ax1 = plt.subplots(figsize=(10, 6))
        #
        #         # 设置标题和网格
        #         plt.title('Parameter Evolution over Iterations', fontsize=14)
        #         ax1.grid(True, linestyle=':', alpha=0.6)
        #
        #         # 绘制种群规模曲线 (蓝色实线)
        #         color_pop = 'tab:blue'
        #         ax1.set_xlabel('Iteration (Gen)')
        #         ax1.set_ylabel('Population Size', color=color_pop, fontsize=12)
        #         line1, = ax1.plot(iterations, pop_sizes, color=color_pop, label='Population Size', linewidth=2)
        #         ax1.tick_params(axis='y', labelcolor=color_pop)
        #
        #         # 3. 设置右侧坐标轴 (Probability 0-1)
        #         ax2 = ax1.twinx()  # 共享x轴
        #         ax2.set_ylabel('Probability', color='black', fontsize=12)
        #         ax2.set_ylim(0, 1.05)  # 严格限制在 0-1 范围，稍微多一点空间给图例
        #
        #         # 绘制交叉率曲线 (橙色虚线)
        #         color_cx = 'tab:orange'
        #         line2, = ax2.plot(iterations, cx_probs, color=color_cx, label='Crossover Rate (cxProb)', linestyle='--',
        #                           linewidth=2)
        #
        #         # 绘制变异率曲线 (绿色点划线)
        #         color_mut = 'tab:green'
        #         line3, = ax2.plot(iterations, mut_probs, color=color_mut, label='Mutation Rate (mutateProb)',
        #                           linestyle='-.',
        #                           linewidth=2)
        #
        #         # 4. 计算并标注平均种群规模
        #         avg_pop_size = sum(pop_sizes) / len(pop_sizes)
        #         text_str = f"Avg Pop Size: {avg_pop_size:.2f}"
        #
        #         # 在图表左上角添加文本框
        #         props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        #         ax1.text(0.05, 0.95, text_str, transform=ax1.transAxes, fontsize=10,
        #                  verticalalignment='top', bbox=props, color=color_pop, fontweight='bold')
        #
        #         # 5. 合并图例
        #         lines = [line1, line2, line3]
        #         labels = [l.get_label() for l in lines]
        #         ax1.legend(lines, labels, loc='center right', frameon=True)
        #
        #         # 6. 保存及显示
        #         plt.tight_layout()
        #         save_plot_path = './results/parameter_trend.png'
        #         plt.savefig(save_plot_path)
        #         print(f"参数演化趋势图已保存至: {save_plot_path}")
        #         plt.show()
        #     else:
        #         print("Evolution history is empty, skipping plot.")
        #
        # except Exception as e:
        #     print(f"Error occurred while plotting parameter evolution: {str(e)}")

        # =========================================================
        # 统一结果保存钩子（供 Friedman 检验使用，T8）
        # =========================================================
        try:
            from experiment_io import save_run_summary, save_best_solutions, save_final_pareto
            from metrics_recorder import extract_pareto_front, compute_hv
            from evaluation import evaluate_with_details

            front = extract_pareto_front(pop)

            # 最终帕累托 CSV（含完整线路）
            save_final_pareto(front, out_dir)

            # best_solutions
            save_best_solutions(front, out_dir)

            # 路径质量（按帕累托均值）
            quality_records = []
            for ind in front:
                q = evaluate_with_details(ind, self.G, self.od_df,
                                          self.node_positions, self.fixed_tasks)
                quality_records.append(q)
            n_q = max(len(quality_records), 1)
            route_quality_payload = {
                "algo": algo_label,
                "run_id": run_id,
                "optimized": {
                    "pareto_front_size": len(quality_records),
                    "total_length_m_mean": sum(q["total_length_m"] for q in quality_records) / n_q,
                    "avg_detour_ratio_mean": sum(q["avg_detour_ratio"] for q in quality_records) / n_q,
                    "num_routes_mean": sum(q["num_routes"] for q in quality_records) / n_q,
                },
            }
            with open(os.path.join(out_dir, "route_quality.json"), "w", encoding="utf-8") as f:
                json.dump(route_quality_payload, f, ensure_ascii=False, indent=2)

            # run_summary
            front_objs = np.array([[ind.fitness.values[0], ind.fitness.values[1]]
                                   for ind in front]) if front else np.zeros((0, 2))
            ref_point = np.array([0.0, 0.0])
            hv_value = compute_hv(front_objs, ref_point)
            duration_val = time.perf_counter() - getattr(self, "_run_start_ts", time.perf_counter())

            save_run_summary(
                algo=algo_label, run_id=run_id if run_id is not None else 0,
                seed=seed if seed is not None else 0,
                summary={
                    "started_at": getattr(self, "_started_at", None),
                    "duration_sec": round(duration_val, 2),
                    "evaluations": pop_size * n_gen,
                    "pop_size": pop_size,
                    "n_gen": n_gen,
                    "final_hv": hv_value,
                    "ref_point": [0.0, 0.0],
                    "z1_max": float(front_objs[:, 0].max()) if len(front_objs) else 0.0,
                    "z2_max": float(front_objs[:, 1].max()) if len(front_objs) else 0.0,
                    "pareto_front_size": len(front),
                },
                out_dir=out_dir,
            )
            print(f"[钩子] 统一结果已保存到 {out_dir}")
        except Exception as e:
            import traceback
            print(f"[警告] 统一结果保存失败: {e}")
            traceback.print_exc()

        # 写出每代 convergence JSON（与 MOEAD/MOPSO/WSGATS 的 {algo}_convergence.json 对齐）
        # 供 plot/curve.R 的 nested type 直接读取
        try:
            conv_path = os.path.join(out_dir, f"{algo_label}_convergence.json")
            with open(conv_path, "w", encoding="utf-8") as f:
                json.dump(self._convergence_records, f, ensure_ascii=False, indent=2)
            print(f"[convergence] 每代记录已保存到 {conv_path} "
                  f"(共 {len(self._convergence_records)} 代)")
        except Exception as e:
            print(f"[警告] convergence JSON 写出失败: {e}")

        return pop, None

    def _get_front(self, pop):
        """获取当前种群的非支配解集 (Pareto Front)"""
        return tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

    def _log_generation(self, gen, pop, hv=None):
        """
        核心日志函数：
        1. 记录当代的统计指标 (Metrics)
        2. 记录当代的 Pareto 前沿详细数据 (用于动态绘图)
        3. 暂存 nested 格式记录到 self._convergence_records（run 末尾统一写 JSON）

        :param hv: 当前代的 hypervolume（None 表示未计算，如 gen=0 初始化）
        """
        # 1. 计算基础统计
        fits = [ind.fitness.values for ind in pop]
        z1_list = [f[0] for f in fits]
        z2_list = [f[1] for f in fits]
        front = self._get_front(pop)

        stats = {
            'gen': gen,
            'hypervolume': hv if hv is not None else None,
            'avg_z1': np.mean(z1_list),
            'max_z1': np.max(z1_list),
            'avg_z2': np.mean(z2_list),
            'max_z2': np.max(z2_list),
            'std_z1': np.std(z1_list),
            'std_z2': np.std(z2_list)
        }

        # 追加写入 metrics.csv（写到 run 目录，便于按 run 隔离 + R 脚本读取）
        metrics_file = os.path.join(self._out_dir, "metrics.csv")
        df_stats = pd.DataFrame([stats])
        if gen == 0:
            df_stats.to_csv(metrics_file, index=False)
        else:
            df_stats.to_csv(metrics_file, mode='a', header=False, index=False)

        # 暂存 nested 格式记录（与 MOEAD/MOPSO/WSGATS 的 {algo}_convergence.json 对齐）
        # 字段：gen, pareto_size, hypervolume, obj1:{max,min,mean,std}, obj2:{...}, ideal_point
        self._convergence_records.append({
            'gen': gen,
            'pareto_size': len(front),
            'hypervolume': hv if hv is not None else None,
            'obj1': {
                'max': float(np.max(z1_list)),
                'min': float(np.min(z1_list)),
                'mean': float(np.mean(z1_list)),
                'std': float(np.std(z1_list)),
            },
            'obj2': {
                'max': float(np.max(z2_list)),
                'min': float(np.min(z2_list)),
                'mean': float(np.mean(z2_list)),
                'std': float(np.std(z2_list)),
            },
            'ideal_point': {
                'Z1': float(np.max(z1_list)),
                'Z2': float(np.max(z2_list)),
            },
        })

        # 2. 保存 Pareto 前沿 (详细解)
        # 我们只保存 Pareto Front 的解，节省空间，但足够画图分析
        front_data = []
        for ind in front:
            front_data.append({
                'gen': gen,
                'z1_visual': ind.fitness.values[0],
                'z2_demand': ind.fitness.values[1],
                # 如果需要分析具体路线，可以将 ind (路线列表) 转 string 存入
                # 'routes_str': str(ind)
            })

        df_front = pd.DataFrame(front_data)
        # 每个世代存一个文件，方便后续制作 GIF 或 Timeline 分析
        df_front.to_csv(os.path.join(self.history_dir, f"front_gen_{gen:03d}.csv"), index=False)