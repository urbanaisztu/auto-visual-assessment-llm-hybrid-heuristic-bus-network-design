import os
import time

# --- 路径配置 (核心修复) ---
# 无论从哪个子目录运行，都通过此文件定位项目根目录
# os.path.abspath(__file__) 获取当前 config.py 的绝对路径
# .dirname 向上退一级得到项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 确保 DATA_INPUT_DIR 指向正确的绝对路径
DATA_INPUT_DIR = os.path.join(BASE_DIR, '../','data', 'input')

FILE_PATHS = {
    'edges': os.path.join(DATA_INPUT_DIR, 'edges.geojson'),
    'nodes': os.path.join(DATA_INPUT_DIR, 'nodes.geojson'),
    'od': os.path.join(DATA_INPUT_DIR, 'OD.csv'),
    'route_nodes': os.path.join(DATA_INPUT_DIR, 'route_nodes_updated.geojson')
}

# --- 实验管理与版本控制 ---
CUSTOM_TAG = "equal"
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
EXPERIMENT_ID = f"{CUSTOM_TAG}_{TIMESTAMP}" if CUSTOM_TAG else f"Exp_{TIMESTAMP}"

# 结果输出根目录（所有算法的父目录，用于 Friedman 检验扫描）
RESULTS_BASE_DIR = os.path.join(BASE_DIR, '..', 'results')   # /workspace/2025/results/

# 单次实验目录（保持向后兼容；旧脚本仍可用）
RESULTS_BASE = os.path.join(BASE_DIR, 'results')             # /workspace/2025/src/results/（旧）
RESULTS_DIR = os.path.join(RESULTS_BASE, EXPERIMENT_ID)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR, exist_ok=True)

# 算法清单（Friedman 检验扫描时用）
ALGORITHMS = ["LEHHA", "NSGA2", "WSGATS", "MOEAD", "MOPSO"]

# --- 算法策略开关 (保持原样) ---
STRATEGIES = {
    'INITIALIZATION': 'mixed',
    'MUTATION': 'smart',
    'INIT_RATIOS': {
        'physical': 0.4,
        'visual': 0.3,
        'demand': 0.3
    },
    'MUTATION_PROBS': {
        'visual': 0.35,
        'demand': 0.35,
        'smooth': 0.30
    }
}

# --- 遗传算法参数 ---
GA_PARAMS = {
    'POP_SIZE': 100,
    'NGEN': 200,
    'CXPB': 0.6,
    'MUTPB': 0.4
}

# --- 模型约束参数 ---
MODEL_CONSTRAINTS = {
    'N_MIN': 5,
    'N_MAX': 50,
    'DELTA_MAX': 2.5,
    'K_OVERLAP': 3,
    'PENALTY_FACTOR': 1e5
}

# --- 视觉评价权重 ---
VISUAL_WEIGHTS = {
    'W1': 1.0,
    'W2': 0.2
}

# --- 目标函数归一化参数（详见 docs/plans/2026-07-16-objective-normalization-design.md） ---
NORMALIZATION = {
    # Z1 理论边界（net_visual = W1*PI - W2*NI, PI/NI ∈ [0,1]）
    'Z1_RAW_MIN': -0.2,          # 对应 PI=0, NI=1
    'Z1_RAW_MAX': 1.0,           # 对应 PI=1, NI=0
    'Z1_EMPTY_PENALTY': -0.5,    # 空路径硬编码值（必被淘汰）

    # Z2 理论上界 = OD 表总需求（由 data_loader.py 加载后动态填充）
    'Z2_TOTAL_DEMAND': None,

    # 约束违反度归一化
    'V_SCALE': 5.0,              # 经验上限（3 类约束各贡献约 1.5）
    'PENALTY_ALPHA': 1.0,        # 软惩罚强度（α=1 保证可行解严格优于违反解）
}

# --- 算子动态选择控制（LLM驱动） ---
OPERATOR_SELECTION = {
    # 是否启用LLM动态选择算子权重
    'enabled': False,

    # 消融实验模式：None/'overall_only'/'visual_only'/'demand_only'/'default_only'
    'ablation_mode': None,

    # 调用频率：每N代调用一次LLM
    'call_interval': 10,

    # 首次调用时机：从第N代开始
    'start_gen': 20,

    # 当enabled=False或ablation_mode!=None时使用固定权重
    'fixed_weights': {
        'cx': {'overall': 0.25, 'visual': 0.25, 'demand': 0.25, 'default': 0.25},
        'mt': {'overall': 0.25, 'visual': 0.25, 'demand': 0.25, 'default': 0.25}
    },

    # 权重平滑参数
    'smoothing': {
        'min_weight': 0.05,      # 单个权重最小值
        'max_weight': 0.7,       # 单个权重最大值
        'max_change': 0.2        # 单次最大变化幅度
    }
}