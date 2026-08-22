"""
单次实验运行模块

支持5种实验模式：
1. equal_weights - 算子平分概率
2. llm_control - LLM把控概率
3. ablation_overall - 只用overall算子
4. ablation_visual - 只用visual算子
5. ablation_demand - 只用demand算子
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_single_experiment(exp_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行单次实验

    Args:
        exp_config: 实验配置字典，包含：
            - exp_id: 实验编号
            - exp_type: 实验类型
            - run_id: 运行编号 (1-5)
            - seed: 随机种子
            - operator_config: 算子配置

    Returns:
        dict: 实验结果摘要
    """
    # 导入模块（在函数内部导入，避免多进程问题）
    from ga_engine import BusNetworkGA
    from data_loader import load_data
    from config import GA_PARAMS, OPERATOR_SELECTION, EXPERIMENT_ID, RESULTS_DIR, STRATEGIES

    exp_id = exp_config['exp_id']
    exp_type = exp_config['exp_type']
    run_id = exp_config['run_id']
    seed = exp_config['seed']
    operator_config = exp_config['operator_config']

    print(f"\n{'='*80}")
    print(f"【实验 {exp_id}】运行 {run_id}/5 - {exp_type}")
    print(f"{'='*80}")
    print(f"随机种子: {seed}")
    print(f"配置: {operator_config}")

    # 设置随机种子
    import random
    random.seed(seed)
    np.random.seed(seed)

    # 修改算子配置
    original_enabled = OPERATOR_SELECTION.get('enabled', False)
    original_ablation = OPERATOR_SELECTION.get('ablation_mode', None)
    original_fixed_weights = OPERATOR_SELECTION.get('fixed_weights', {
        'cx': {'overall': 0.25, 'visual': 0.25, 'demand': 0.25, 'default': 0.25},
        'mt': {'overall': 0.25, 'visual': 0.25, 'demand': 0.25, 'default': 0.25}
    }).copy()

    # 应用新配置
    OPERATOR_SELECTION['enabled'] = operator_config.get('enabled', False)
    OPERATOR_SELECTION['ablation_mode'] = operator_config.get('ablation_mode', None)
    if 'fixed_weights' in operator_config:
        OPERATOR_SELECTION['fixed_weights'] = operator_config['fixed_weights']

    # 加载数据
    print("正在加载数据...")
    edges, od, fixed_tasks, G, node_pos = load_data()

    # 初始化GA
    print("初始化GA引擎...")
    ga = BusNetworkGA(G, od, fixed_tasks, node_pos)

    # 记录时间
    start_time = time.time()
    start_perf = time.perf_counter()

    # 运行优化
    print(f"开始优化 (种群={GA_PARAMS['POP_SIZE']}, 迭代={GA_PARAMS['NGEN']}代)...")
    try:
        final_pop, _ = ga.run(
            pop_size=GA_PARAMS['POP_SIZE'],
            n_gen=GA_PARAMS['NGEN']
        )
    except Exception as e:
        print(f"实验运行出错: {e}")
        import traceback
        traceback.print_exc()

        # 恢复原始配置
        OPERATOR_SELECTION['enabled'] = original_enabled
        OPERATOR_SELECTION['ablation_mode'] = original_ablation
        OPERATOR_SELECTION['fixed_weights'] = original_fixed_weights

        return {
            'exp_id': exp_id,
            'run_id': run_id,
            'success': False,
            'error': str(e)
        }

    # 记录结束时间
    end_time = time.time()
    end_perf = time.perf_counter()

    wall_time = end_time - start_time
    perf_time = end_perf - start_perf

    print(f"\n实验完成！")
    print(f"Wall Clock耗时: {wall_time:.2f}秒 ({wall_time/60:.2f}分钟)")
    print(f"Perf Counter耗时: {perf_time:.2f}秒 ({perf_time/60:.2f}分钟)")

    # 提取Pareto前沿
    from deap import tools
    front = tools.sortNondominated(final_pop, len(final_pop), first_front_only=True)[0]

    # 计算统计信息
    Z1_list = [ind.fitness.values[0] for ind in front]
    Z2_list = [ind.fitness.values[1] for ind in final_pop]

    # 计算HV
    try:
        from pymoo.indicators.hv import HV
        ref_point = np.array([0.0, 0.0])
        front_objs = np.array([ind.fitness.values for ind in front])
        ind_hv = HV(ref_point=ref_point)
        final_hv = ind_hv.do(-front_objs)
    except:
        final_hv = 0.0

    # 保存最终Pareto前沿
    pareto_data = []
    for idx, individual in enumerate(front):
        pareto_data.append({
            "index": idx,
            "path": [[int(node) for node in route] for route in individual],
            "fitness": {
                "visual": float(individual.fitness.values[0]),
                "demand": float(individual.fitness.values[1])
            }
        })

    # 结果保存路径
    # 创建 {实验类型}/run_{序号} 的独立目录结构
    type_dir = os.path.join(RESULTS_DIR, exp_type)
    os.makedirs(type_dir, exist_ok=True)
    run_dir = os.path.join(type_dir, f"run_{run_id:03d}")
    os.makedirs(run_dir, exist_ok=True)

    # 保存Pareto前沿
    pareto_path = os.path.join(run_dir, "pareto_front.json")
    with open(pareto_path, 'w', encoding='utf-8') as f:
        json.dump(pareto_data, f, indent=4, ensure_ascii=False)

    # 保存摘要
    result_summary = {
        "exp_id": exp_id,
        "exp_type": exp_type,
        "run_id": run_id,
        "seed": seed,
        "success": True,
        "wall_time_seconds": float(wall_time),
        "perf_time_seconds": float(perf_time),
        "pareto_size": len(front),
        "final_hypervolume": float(final_hv),
        "Z1_visual": {
            "max": float(np.max(Z1_list)) if Z1_list else 0,
            "min": float(np.min(Z1_list)) if Z1_list else 0,
            "mean": float(np.mean(Z1_list)) if Z1_list else 0,
            "std": float(np.std(Z1_list)) if Z1_list else 0,
            "median": float(np.median(Z1_list)) if Z1_list else 0
        },
        "Z2_demand": {
            "max": float(np.max(Z2_list)) if Z2_list else 0,
            "min": float(np.min(Z2_list)) if Z2_list else 0,
            "mean": float(np.mean(Z2_list)) if Z2_list else 0,
            "std": float(np.std(Z2_list)) if Z2_list else 0,
            "median": float(np.median(Z2_list)) if Z2_list else 0
        },
        "operator_config": operator_config,
        "pareto_front_path": pareto_path
    }

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(result_summary, f, indent=4, ensure_ascii=False)

    print(f"结果已保存至: {run_dir}")

    # 恢复原始配置
    OPERATOR_SELECTION['enabled'] = original_enabled
    OPERATOR_SELECTION['ablation_mode'] = original_ablation
    OPERATOR_SELECTION['fixed_weights'] = original_fixed_weights

    return result_summary


if __name__ == "__main__":
    # 测试运行
    test_config = {
        'exp_id': 0,
        'exp_type': 'equal_weights',
        'run_id': 1,
        'seed': 1001,
        'operator_config': {
            'enabled': False,
            'ablation_mode': None,
            'fixed_weights': {
                'cx': {'overall': 0.25, 'visual': 0.25, 'demand': 0.25, 'default': 0.25},
                'mt': {'overall': 0.25, 'visual': 0.25, 'demand': 0.25, 'default': 0.25}
            }
        }
    }

    result = run_single_experiment(test_config)
    print("\n测试结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
