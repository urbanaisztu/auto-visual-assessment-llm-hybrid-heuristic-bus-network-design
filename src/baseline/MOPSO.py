"""
MOPSO: Multi-Objective Particle Swarm Optimization
多目标粒子群优化算法

参考论文：
Coello Coello, C. A., Pulido, G. T., & Lechuga, M. S. (2004).
"MOPSO: A proposal for multiple objective particle swarm optimization"
IEEE Transactions on Evolutionary Computation, 8(3), 256-279.

作者：Baseline实现
日期：2026-01-07
"""

import sys
import os
import random
import json
import copy
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
from collections import defaultdict

# 添加父目录到路径以导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import evaluate_individual
from config import GA_PARAMS, STRATEGIES, RESULTS_DIR
from utils import calculate_haversine_distance


def calculate_hypervolume(fitnesses, reference_point=(0, 0)):
    """
    【修正】适用于最大化问题的HV计算（面积法）
    """
    if not fitnesses:
        return 0.0

    # 1. 按第一个目标排序
    # 假设输入已经是帕累托前沿（互不支配）
    # 为了保险，先按Z1排序
    sorted_fit = sorted(fitnesses, key=lambda x: x[0])

    # 简单过滤：确保 Z1 递增时 Z2 递减（剔除被支配解）
    clean_fits = []
    if sorted_fit:
        clean_fits.append(sorted_fit[0])
        for i in range(1, len(sorted_fit)):
            # 如果当前解的 Z2 比上一个解的 Z2 大（或相等），说明上一个解被支配了（或重复），
            # 但因为我们排了序，正常帕累托前沿 Z2 应该是下降的。
            # 这里简化处理，直接计算所有阶梯面积
            pass

    hv = 0.0
    prev_Z1 = reference_point[0]

    # 2. 累加阶梯面积
    for Z1, Z2 in sorted_fit:
        width = Z1 - prev_Z1
        height = Z2

        # 只计算参考点右上方的面积
        if width > 0 and height > reference_point[1]:
            hv += width * (height - reference_point[1])

        prev_Z1 = Z1

    return hv


class Velocity:
    """
    速度类：针对路径规划问题的离散编码

    速度 = 子路径替换操作序列
    每个操作：(route_idx, cut_start, cut_end, new_subpath)
    表示：将路径route_idx的[cut_start:cut_end]段替换为new_subpath

    这种方式保证：
    1. 起点和终点不变
    2. 替换的子路径是连通的（通过shortest_path生成）
    3. 适用于路径规划问题而非TSP
    """

    def __init__(self, operations=None):
        """
        初始化速度

        Args:
            operations: 操作列表，每个元素是(route_idx, cut_start, cut_end, new_subpath)
        """
        self.operations = operations if operations is not None else []

    def __len__(self):
        return len(self.operations)

    def __add__(self, other):
        """速度相加：合并操作序列"""
        return Velocity(self.operations + other.operations)

    def __mul__(self, scalar):
        """速度标量乘：随机保留部分操作"""
        probability = min(1.0, scalar)
        new_ops = [op for op in self.operations if random.random() < probability]
        return Velocity(new_ops)

    def limit(self, max_ops):
        """限制速度大小（最大操作次数）"""
        if len(self.operations) > max_ops:
            return Velocity(random.sample(self.operations, max_ops))
        return Velocity(self.operations)


class Grid:
    """
    网格类：用于Archive的密度估计
    """

    def __init__(self, n_divisions=10):
        """
        初始化网格

        Args:
            n_divisions: 每个维度的分割数（默认10×10）
        """
        self.n_divisions = n_divisions
        self.bounds = {'obj1': (0, 1), 'obj2': (0, 1)}  # 动态更新
        self.grid = defaultdict(int)  # (cell_x, cell_y) -> count

    def update_bounds(self, archive):
        """根据Archive更新边界"""
        if not archive:
            return

        obj1_values = [fit[0] for _, fit in archive]
        obj2_values = [fit[1] for _, fit in archive]

        self.bounds['obj1'] = (min(obj1_values), max(obj1_values))
        self.bounds['obj2'] = (min(obj2_values), max(obj2_values))

    def locate(self, fitness):
        """
        将适应度定位到网格单元

        Args:
            fitness: (Z1, Z2)

        Returns:
            tuple: (cell_x, cell_y)
        """
        obj1_min, obj1_max = self.bounds['obj1']
        obj2_min, obj2_max = self.bounds['obj2']

        # 避免除零
        if obj1_max == obj1_min:
            cell_x = 0
        else:
            ratio_x = (fitness[0] - obj1_min) / (obj1_max - obj1_min)
            cell_x = int(ratio_x * (self.n_divisions - 1))

        if obj2_max == obj2_min:
            cell_y = 0
        else:
            ratio_y = (fitness[1] - obj2_min) / (obj2_max - obj2_min)
            cell_y = int(ratio_y * (self.n_divisions - 1))

        return (cell_x, cell_y)

    def update_grid(self, archive):
        """更新网格计数"""
        self.grid.clear()
        for _, fitness in archive:
            cell = self.locate(fitness)
            self.grid[cell] += 1

    def get_density(self, fitness, archive):
        """
        获取适应度所在位置的密度（用于leader选择）

        Args:
            fitness: (Z1, Z2)
            archive: 当前Archive

        Returns:
            float: 密度值（越大越拥挤）
        """
        if not archive:
            return 0

        cell = self.locate(fitness)
        return self.grid.get(cell, 0)

    def select_leader(self, archive):
        """
        基于网格密度选择leader（低密度区域概率更高）

        Args:
            archive: [(individual, fitness), ...]

        Returns:
            individual: 选择的leader个体
        """
        if not archive:
            return None

        # 计算每个解的密度倒数
        probs = []
        for _, fitness in archive:
            density = self.get_density(fitness, archive)
            # 密度越小，概率越大
            prob = 1.0 / (density + 1)
            probs.append(prob)

        # 归一化
        total = sum(probs)
        probs = [p / total for p in probs]

        # 轮盘赌选择
        idx = np.random.choice(len(archive), p=probs)
        return archive[idx][0]


class Particle:
    """粒子类"""

    def __init__(self, position, fitness, velocity=None):
        """
        初始化粒子

        Args:
            position: 位置（路径列表）
            fitness: 适应度 (Z1, Z2)
            velocity: 速度对象
        """
        self.position = position
        self.fitness = fitness
        self.velocity = velocity if velocity is not None else Velocity()
        self.pbest_position = position  # 个人历史最优位置
        self.pbest_fitness = fitness    # 个人历史最优适应度


class MOPSO:
    """MOPSO算法主类"""

    def __init__(self, G, od_df, fixed_tasks, node_positions, debug=False):
        """
        初始化MOPSO

        Args:
            G: NetworkX图
            od_df: OD矩阵
            fixed_tasks: 固定任务列表 [(line_id, start, end), ...]
            node_positions: 节点位置字典
            debug: 是否输出调试信息
        """
        self.G = G
        self.od_df = od_df
        self.fixed_tasks = fixed_tasks
        self.node_positions = node_positions
        self.debug = debug

        # 算法参数
        self.pop_size = GA_PARAMS['POP_SIZE']  # 100
        self.ngen = GA_PARAMS['NGEN']  # 200
        self.archive_size = 100
        self.grid_division = 10

        # PSO参数
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.max_velocity = 5  # 最大操作次数
        self.mutation_rate = 0.3  # 提高变异率以增强探索能力
        self.mutation_force_rate = 0.05  # 强制变异率（即使速度为空也变异）

        # 网格和存档
        self.grid = Grid(n_divisions=self.grid_division)
        self.archive = []  # [(individual, fitness), ...]

    def _init_individual(self):
        """初始化个体：使用mixed策略"""
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

    def _evaluate(self, individual):
        """评估个体"""
        Z1, Z2 = evaluate_individual(
            individual, self.G, self.od_df,
            self.node_positions, self.fixed_tasks
        )
        return (Z1, Z2)

    def _dominates(self, fitness1, fitness2):
        """判断fitness1是否支配fitness2"""
        return (fitness1[0] >= fitness2[0] and fitness1[1] >= fitness2[1] and
                (fitness1[0] > fitness2[0] or fitness1[1] > fitness2[1]))

    def _generate_path_replacement_operations(self, current_pos, target_pos, max_ops=3):
        """
        【修正】基于公共节点的路径片段复制（模拟交叉操作）
        不再重新生成路径，而是直接复制 target_pos 的优秀片段
        """
        operations = []
        if not current_pos or not target_pos:
            return operations

        num_routes = min(len(current_pos), len(target_pos))

        # 尝试多次寻找可交换片段
        attempts = 0
        while len(operations) < max_ops and attempts < 10:
            attempts += 1
            route_idx = random.randint(0, num_routes - 1)

            route_curr = current_pos[route_idx]
            route_targ = target_pos[route_idx]

            if len(route_curr) < 2 or len(route_targ) < 2:
                continue

            # 1. 寻找公共节点（交点）
            common_nodes = list(set(route_curr) & set(route_targ))

            # 如果公共节点太少，无法形成有效的段替换，退化为随机重连（旧策略）
            if len(common_nodes) < 2:
                # [保留旧逻辑作为备选] 随机选择子段重生成
                if len(route_curr) > 5:
                    cut_u = random.randint(1, len(route_curr) - 4)
                    cut_v = random.randint(cut_u + 2, len(route_curr) - 2)
                    u, v = route_curr[cut_u], route_curr[cut_v]
                    try:
                        # 随机选一种权重，增加多样性
                        w_attr = random.choice(['weight', 'visual_cost', 'demand_cost'])
                        new_sub = nx.shortest_path(self.G, u, v, weight=w_attr)
                        if len(new_sub) > 1:
                            operations.append((route_idx, cut_u, cut_v + 1, new_sub))
                    except:
                        pass
                continue

            # 2. 基于公共节点进行片段复制
            # 随机选两个公共节点 u, v
            u = random.choice(common_nodes)
            v = random.choice(common_nodes)

            if u == v:
                continue

            # 获取 u, v 在 current 和 target 中的索引
            try:
                curr_u_idx = route_curr.index(u)
                curr_v_idx = route_curr.index(v)
                targ_u_idx = route_targ.index(u)
                targ_v_idx = route_targ.index(v)

                # 确保顺序一致 (u 在 v 前面)
                if curr_u_idx > curr_v_idx:
                    curr_u_idx, curr_v_idx = curr_v_idx, curr_u_idx

                # 检查 target 中 u 是否也在 v 前面（如果不一致则很难复制）
                if targ_u_idx > targ_v_idx:
                    # 尝试反转 target 的片段？或者直接跳过
                    continue

                # 3. 提取 Target 的基因片段
                target_subpath = route_targ[targ_u_idx: targ_v_idx + 1]

                # 4. 生成操作：用 Target 的片段替换 Current 的片段
                # 注意：curr_v_idx + 1 是为了包含 v
                operations.append((route_idx, curr_u_idx, curr_v_idx + 1, target_subpath))

            except ValueError:
                continue

        return operations

    def _is_subpath_connected(self, subpath):
        """
        检查子路径是否连通

        Args:
            subpath: 节点序列

        Returns:
            bool: 是否连通
        """
        if len(subpath) < 2:
            return True

        for i in range(len(subpath) - 1):
            if not self.G.has_edge(subpath[i], subpath[i + 1]):
                return False
        return True

    def _apply_velocity(self, position, velocity):
        """
        将速度应用到位置（子路径替换）

        Args:
            position: 当前位置
            velocity: 速度对象（包含替换操作）

        Returns:
            新位置
        """
        new_position = copy.deepcopy(position)

        for (route_idx, cut_start, cut_end, new_subpath) in velocity.operations:
            if route_idx < len(new_position):
                route = new_position[route_idx]
                if 0 <= cut_start < cut_end <= len(route):
                    # 替换子路径
                    new_route = route[:cut_start] + new_subpath + route[cut_end:]
                    new_position[route_idx] = new_route

        return new_position

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

    def _smart_mutate(self, individual):
        """
        智能变异：随机选择一条路径的子段，用启发式策略重新生成
        (已修复 randint 范围报错问题)
        """
        # 深拷贝，避免修改原对象
        new_individual = [list(route) for route in individual]

        if len(new_individual) == 0:
            return new_individual

        # 1. 随机选择一条路径
        route_idx = random.randint(0, len(new_individual) - 1)
        route = new_individual[route_idx]

        # 2. 长度检查 [关键修改]
        # 我们需要：起点(1) + u(1) + 间隔(1) + v(1) + 终点(1) = 5个节点
        if len(route) < 5:
            return new_individual

        # 3. 严格计算索引范围 [关键修改]
        # 约束条件：
        # a. 1 <= cut_u
        # b. cut_v >= cut_u + 2
        # c. cut_v <= len(route) - 2 (保留终点)
        # 推导 a & b & c -> cut_u + 2 <= len(route) - 2  =>  cut_u <= len(route) - 4

        max_u = len(route) - 4
        cut_u = random.randint(1, max_u)

        # 此时 cut_u + 2 一定小于等于 len(route) - 2，范围有效
        cut_v = random.randint(cut_u + 2, len(route) - 2)

        u, v = route[cut_u], route[cut_v]

        # 4. 随机选择策略
        dice = random.random()
        weight_attr = 'weight'  # 默认物理距离
        if dice < 0.33:
            weight_attr = 'visual_cost'
        elif dice < 0.66:
            weight_attr = 'demand_cost'

        # 5. 生成新子路径并拼接
        try:
            # 使用 NetworkX 寻找两点间的新路径
            subpath = nx.shortest_path(self.G, u, v, weight=weight_attr)

            # 拼接：头部(不含u) + 新子路径(含u和v) + 尾部(不含v)
            # route[:cut_u] 拿到 0 到 u-1
            # subpath 拿到 u 到 v
            # route[cut_v + 1:] 拿到 v+1 到 end
            new_route = route[:cut_u] + subpath + route[cut_v + 1:]

            # 修复可能产生的环并更新
            new_individual[route_idx] = self._repair_route(new_route)

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # 如果找不到路，静默失败，保持原样
            pass

        return new_individual

    def _update_archive(self, new_individual, new_fitness):
        """
        更新Archive存档

        Args:
            new_individual: 新个体
            new_fitness: 新适应度
        """
        # 检查是否被Archive中的解支配
        dominated = False
        for _, fit_archive in self.archive:
            if self._dominates(fit_archive, new_fitness):
                dominated = True
                break

        if not dominated:
            # 移除被新解支配的解
            new_archive = []
            for ind_archive, fit_archive in self.archive:
                if not self._dominates(new_fitness, fit_archive):
                    new_archive.append((ind_archive, fit_archive))

            # 添加新解
            new_archive.append((new_individual, new_fitness))

            # 如果超过容量，删除最拥挤的
            if len(new_archive) > self.archive_size:
                self.grid.update_bounds(new_archive)
                self.grid.update_grid(new_archive)

                # 计算每个解的密度
                densities = []
                for _, fitness in new_archive:
                    density = self.grid.get_density(fitness, new_archive)
                    densities.append(density)

                # 删除密度最大的
                max_density_idx = np.argmax(densities)
                del new_archive[max_density_idx]

            self.archive = new_archive

            # 更新网格
            self.grid.update_bounds(self.archive)
            self.grid.update_grid(self.archive)

    def run(self):
        """
        运行MOPSO算法

        Returns:
            tuple: (archive存档, convergence_data)
        """
        print("=" * 60)
        print("MOPSO: Multi-Objective Particle Swarm Optimization")
        print("=" * 60)
        print(f"种群规模: {self.pop_size}")
        print(f"迭代代数: {self.ngen}")
        print(f"Archive容量: {self.archive_size}")
        print(f"网格划分: {self.grid_division}×{self.grid_division}")
        print(f"惯性权重: {self.w_max} → {self.w_min}")
        print(f"总评估次数: {self.pop_size * self.ngen}")
        print("=" * 60)

        # 1. 初始化粒子群
        print("\n正在初始化粒子群...")
        particles = []
        for _ in range(self.pop_size):
            position = self._init_individual()
            fitness = self._evaluate(position)
            velocity = Velocity()
            particle = Particle(position, fitness, velocity)
            particles.append(particle)

        # 2. 初始化Archive
        print("正在初始化Archive...")
        for p in particles:
            self._update_archive(p.position, p.fitness)

        print(f"初始Archive大小: {len(self.archive)}")

        # 设置HV参考点（对于最大化问题，使用原点）
        self.reference_point = (0, 0)

        # 收敛历史
        convergence_history = []

        # 3. 进化循环
        for gen in range(1, self.ngen + 1):
            # 计算当前惯性权重（线性递减）
            w = self.w_max - (self.w_max - self.w_min) * gen / self.ngen

            for particle in particles:
                # --- 选择 Leader ---
                leader = self.grid.select_leader(self.archive)
                if leader is None:
                    leader = particle.pbest_position

                # --- 速度更新 (修正版) ---
                # 1. 惯性: 保留旧操作的概率降低，防止无效操作堆积
                v_inertia = particle.velocity * (w * 0.5)

                # 2. 认知 (向Pbest学习)
                r1 = random.random()
                if random.random() < self.c1 * r1:  # 简化概率判断
                    # 只允许生成 1-2 个操作，不要太多
                    cognitive_ops = self._generate_path_replacement_operations(
                        particle.position, particle.pbest_position, max_ops=1
                    )
                    v_cognitive = Velocity(cognitive_ops)
                else:
                    v_cognitive = Velocity()

                # 3. 社会 (向Leader学习)
                r2 = random.random()
                if random.random() < self.c2 * r2:
                    # 向 Leader 学习至多 2 个操作
                    social_ops = self._generate_path_replacement_operations(
                        particle.position, leader, max_ops=2
                    )
                    v_social = Velocity(social_ops)
                else:
                    v_social = Velocity()

                # 合并速度：降低最大操作数，防止破坏性更新
                # 建议 max_velocity 设为 2 或 3 (原代码为5，太大了)
                new_velocity = v_inertia + v_cognitive + v_social
                new_velocity = new_velocity.limit(max_ops=3)

                particle.velocity = new_velocity

                # --- 位置更新 ---
                new_position = self._apply_velocity(particle.position, particle.velocity)
                new_position = [self._repair_route(route) for route in new_position]

                # --- 变异 ---
                # 降低变异率，因为 MOPSO 主要是靠跟随 Leader 收敛
                if random.random() < 0.1:  # 原为 mutation_rate
                    new_position = self._smart_mutate(new_position)

                # 强制变异保留 (防止死锁)
                if len(particle.velocity.operations) == 0 and random.random() < 0.05:
                    new_position = self._smart_mutate(new_position)

                # --- 评估 ---
                new_fitness = self._evaluate(new_position)

                # --- 更新 PBest (修正版：严格保护非支配解) ---
                if self._dominates(new_fitness, particle.pbest_fitness):
                    # 新解支配旧解 -> 更新
                    particle.pbest_position = new_position
                    particle.pbest_fitness = new_fitness
                elif self._dominates(particle.pbest_fitness, new_fitness):
                    # 旧解支配新解 -> 保持不动
                    pass
                else:
                    # 【关键修改】互不支配时
                    # NSGA-II 可能会保留两者，但在 PSO 必须二选一。
                    # 策略：以极大概率保留旧 PBest，防止在 Pareto 前沿上无意义震荡
                    # 只有极小概率 (如 10%) 切换到新解，或者当新解非常“独特”时切换(这里简化处理)
                    if random.random() < 0.1:
                        particle.pbest_position = new_position
                        particle.pbest_fitness = new_fitness

                # 更新当前状态
                particle.position = new_position
                particle.fitness = new_fitness

                # 更新 Archive
                self._update_archive(particle.position, particle.fitness)

            # 记录指标
            if gen % 1 == 0 or gen == 1:
                archive_fitnesses = [fit for _, fit in self.archive]

                Z1_list = [f[0] for f in archive_fitnesses]
                Z2_list = [f[1] for f in archive_fitnesses]

                # 计算HV
                hv = calculate_hypervolume(archive_fitnesses, self.reference_point)

                record = {
                    "gen": gen,
                    "archive_size": len(self.archive),
                    "hypervolume": float(hv),
                    "obj1": {
                        "max": float(np.max(Z1_list)) if Z1_list else 0,
                        "min": float(np.min(Z1_list)) if Z1_list else 0,
                        "mean": float(np.mean(Z1_list)) if Z1_list else 0,
                        "std": float(np.std(Z1_list)) if Z1_list else 0
                    },
                    "obj2": {
                        "max": float(np.max(Z2_list)) if Z2_list else 0,
                        "min": float(np.min(Z2_list)) if Z2_list else 0,
                        "mean": float(np.mean(Z2_list)) if Z2_list else 0,
                        "std": float(np.std(Z2_list)) if Z2_list else 0
                    }
                }
                convergence_history.append(record)

                print(f"Gen {gen}/{self.ngen} | "
                      f"Archive: {len(self.archive)} | "
                      f"HV: {hv:.2f} | "
                      f"Z1_max: {record['obj1']['max']:.2f} | "
                      f"Z2_max: {record['obj2']['max']:.2f}")

        print("\n=== MOPSO优化完成 ===")
        print(f"最终Archive大小: {len(self.archive)}")

        return self.archive, convergence_history

    def save_results(self, archive, convergence_data,
                     run_id=None, seed=None, duration_sec=0.0):
        """
        保存结果到JSON文件

        Args:
            archive: Archive存档
            convergence_data: 收敛数据
            run_id: 运行编号（Friedman 检验用）
            seed: 随机种子
            duration_sec: 总耗时（秒）
        """
        # 保存收敛数据
        convergence_path = os.path.join(RESULTS_DIR, "MOPSO_convergence.json")
        with open(convergence_path, 'w', encoding='utf-8') as f:
            json.dump(convergence_data, f, indent=4)
        print(f"\n收敛数据已保存至: {convergence_path}")

        # 保存Pareto前沿（Archive）
        pareto_data = []
        for idx, (individual, fitness) in enumerate(archive):
            pareto_data.append({
                "index": idx,
                "path": [[int(node) for node in route] for route in individual],
                "fitness": {
                    "visual": float(fitness[0]),
                    "satisfy": float(fitness[1])
                }
            })

        pareto_path = os.path.join(RESULTS_DIR, "MOPSO_pareto.json")
        with open(pareto_path, 'w', encoding='utf-8') as f:
            json.dump(pareto_data, f, indent=4, ensure_ascii=False)
        print(f"Pareto前沿已保存至: {pareto_path}")

        # =========================================================
        # 统一结果保存钩子（T11）
        # =========================================================
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from experiment_io import (save_run_summary, save_best_solutions_from_records,
                                       save_final_pareto_from_records, save_route_quality_from_records)
            from metrics_recorder import compute_hv

            records = [(d["fitness"]["visual"], d["fitness"]["satisfy"], d["path"])
                       for d in pareto_data]
            front_objs = np.array([[r[0], r[1]] for r in records]) if records else np.zeros((0, 2))
            hv_value = compute_hv(front_objs, np.array([0.0, 0.0]))

            save_best_solutions_from_records(records, RESULTS_DIR)
            save_final_pareto_from_records(records, RESULTS_DIR)
            save_run_summary(
                algo="MOPSO",
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
                records, RESULTS_DIR, "MOPSO",
                run_id if run_id is not None else 0,
                self.G, self.node_positions
            )
            print(f"[钩子] MOPSO 统一结果已保存到 {RESULTS_DIR}")
        except Exception as e:
            import traceback
            print(f"[警告] MOPSO 统一结果保存失败: {e}")
            traceback.print_exc()


def main():
    """主函数：测试MOPSO算法"""
    import argparse
    import config

    parser = argparse.ArgumentParser(description="MOPSO 基线算法")
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

    # 初始化MOPSO
    mopso = MOPSO(G, od_df, fixed_tasks, node_positions)

    t0 = time.perf_counter()
    # 运行优化
    archive, convergence_data = mopso.run()
    duration_sec = time.perf_counter() - t0

    # 保存结果
    mopso.save_results(archive, convergence_data,
                       run_id=args.run_id, seed=args.seed,
                       duration_sec=duration_sec)

    print("\n=== MOPSO算法运行完成 ===")


if __name__ == "__main__":
    main()
