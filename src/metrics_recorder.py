"""
统一指标计算：HV、帕累托前沿提取、参考点生成。
所有算法共用，保证 Friedman 检验口径一致。
"""
import numpy as np
from deap import tools
from pymoo.indicators.hv import HV


def compute_hv(front_objs: np.ndarray, ref_point: np.ndarray) -> float:
    """
    计算超体积（Hypervolume）。最大化问题自动取负号（pymoo 约定最小化）。

    :param front_objs: (n, 2) 数组，每行是 (z1, z2)
    :param ref_point: (2,) 参考点
    :return: HV 标量
    """
    front_objs = np.asarray(front_objs, dtype=float)
    if front_objs.size == 0:
        return 0.0
    ref_point = np.asarray(ref_point, dtype=float)
    # 取负值转换为最小化问题
    ind = HV(ref_point=ref_point)
    try:
        return float(ind.do(-front_objs))
    except Exception:
        return 0.0


def compute_dynamic_ref_point(all_objs: np.ndarray, margin_ratio: float = 1.1) -> np.ndarray:
    """
    动态参考点：每个维度取所有 run 最小值 × margin_ratio。
    适用于最大化问题。

    :param all_objs: (n, 2) 所有运行的 (z1, z2)
    :param margin_ratio: 余量系数，默认 1.1
    :return: (2,) 参考点
    """
    all_objs = np.asarray(all_objs, dtype=float)
    if all_objs.size == 0:
        return np.array([0.0, 0.0])
    mins = all_objs.min(axis=0)
    return mins * margin_ratio


def extract_pareto_front(population):
    """
    从 DEAP 种群中提取第一层非支配前沿。

    :param population: DEAP 种群
    :return: 前沿个体列表
    """
    return tools.sortNondominated(population, len(population), first_front_only=True)[0]
