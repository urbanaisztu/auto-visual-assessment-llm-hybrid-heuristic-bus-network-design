import networkx as nx
from config import MODEL_CONSTRAINTS
# --- 修改点 1: 引入 utils 中的距离计算函数 ---
from utils import calculate_haversine_distance
from config import NORMALIZATION


def normalize_z1_raw(z1_raw):
    """
    将原始 Z1（段平均 net_visual）归一化到 [0, 1]
    理论边界: net_visual ∈ [-0.2, 1.0]
    公式: (z1_raw - Z1_RAW_MIN) / (Z1_RAW_MAX - Z1_RAW_MIN)
    """
    lo = NORMALIZATION['Z1_RAW_MIN']
    hi = NORMALIZATION['Z1_RAW_MAX']
    return (z1_raw - lo) / (hi - lo)


def normalize_z2_raw(z2_raw):
    """
    将原始 Z2（被服务的总需求）归一化到 [0, 1]
    上界: D_total = Σ OD.count
    """
    d_total = NORMALIZATION['Z2_TOTAL_DEMAND']
    if d_total is None or d_total <= 0:
        # 容错: data_loader 未填充时退化为不归一化
        return z2_raw
    return z2_raw / d_total


def normalize_violation(v_raw):
    """
    将无量纲违反度 V 归一化到 [0, 1]，超出 V_SCALE 时饱和
    """
    v_scale = NORMALIZATION['V_SCALE']
    if v_scale <= 0:
        return 0.0
    return min(v_raw / v_scale, 1.0)


def compute_violation_degree(route_stats, edge_counts):
    """
    计算无量纲约束违反度 V = v_n + v_δ + v_k

    :param route_stats: [{num_stops, length, straight_dist, ...}, ...]
    :param edge_counts: {(u,v): overlap_count, ...}
    :return: float, 无量纲违反度
    """
    n_min = MODEL_CONSTRAINTS['N_MIN']
    n_max = MODEL_CONSTRAINTS['N_MAX']
    delta_max = MODEL_CONSTRAINTS['DELTA_MAX']
    k_overlap = MODEL_CONSTRAINTS['K_OVERLAP']

    # v_n: 站点数违反率
    if route_stats:
        v_n = sum(1 for s in route_stats
                  if s['num_stops'] < n_min or s['num_stops'] > n_max) / len(route_stats)
    else:
        v_n = 0.0

    # v_δ: 平均非直线系数超标量
    if route_stats:
        v_delta = sum(max(0.0, s['length'] / s['straight_dist'] - delta_max)
                      for s in route_stats) / len(route_stats)
    else:
        v_delta = 0.0

    # v_k: 平均路线重叠超标量
    if edge_counts:
        v_k = sum(max(0.0, c - k_overlap) for c in edge_counts.values()) / len(edge_counts)
    else:
        v_k = 0.0

    return v_n + v_delta + v_k


def evaluate_individual(individual, G, od_df, node_positions, fixed_tasks):
    """
    计算个体的双目标适应度 + 约束惩罚（归一化版本）

    Z1, Z2 ∈ [0, 1]（违反约束时可能 < 0）
    - Z1_norm = (Z1_raw + 0.2) / 1.2，Z1_raw = mean(net_visual)
    - Z2_norm = Z2_raw / D_total，Z2_raw = sum(served OD count)
    - 惩罚: Z_final = Z_norm - α * V_norm，α=1.0

    :return: (Z1_final, Z2_final)
    """

    # --- 1. 解码与基础统计 ---
    all_edges = []
    edge_counts = {}
    route_stats = []

    transit_subgraph = nx.Graph()
    total_segments = 0

    for idx, route in enumerate(individual):
        if not route or len(route) < 2:
            continue

        line_id, s_r, t_r = fixed_tasks[idx]

        route_len = 0
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]

            if G.has_edge(u, v):
                all_edges.append((u, v))
                total_segments += 1

                edge_sig = tuple(sorted((u, v)))  # 无向图重叠统计 key 需排序
                edge_counts[edge_sig] = edge_counts.get(edge_sig, 0) + 1

                w = G[u][v].get('weight', 0)
                route_len += w

                transit_subgraph.add_edge(u, v)

        # Haversine 直线距离（米）
        if s_r in node_positions and t_r in node_positions:
            coord_start = node_positions[s_r]
            coord_end = node_positions[t_r]
            straight_dist = calculate_haversine_distance(coord_start, coord_end)
        else:
            straight_dist = 0.1
        if straight_dist == 0:
            straight_dist = 0.1

        route_stats.append({
            'num_stops': len(route),
            'length': route_len,
            'straight_dist': straight_dist,
        })

    # --- 2. 计算 Z1（段平均净视觉值 → 归一化） ---
    if total_segments > 0:
        z1_raw = sum(G[u][v].get('net_visual', 0) for u, v in all_edges) / total_segments
        z1_norm = normalize_z1_raw(z1_raw)
    else:
        z1_norm = NORMALIZATION['Z1_EMPTY_PENALTY']  # 空路径硬编码值

    # --- 3. 计算 Z2（被服务需求 → 归一化） ---
    z2_raw = 0
    sub_nodes = set(transit_subgraph.nodes())
    valid_od = od_df[od_df['from_node'].isin(sub_nodes) & od_df['to_node'].isin(sub_nodes)]

    for _, row in valid_od.iterrows():
        o, d = int(row['from_node']), int(row['to_node'])
        if nx.has_path(transit_subgraph, o, d):
            z2_raw += row['count']

    z2_norm = normalize_z2_raw(z2_raw)

    # --- 4. 计算无量纲违反度 → 归一化 ---
    v_raw = compute_violation_degree(route_stats, edge_counts)
    v_norm = normalize_violation(v_raw)

    # --- 5. 应用对称软惩罚 ---
    alpha = NORMALIZATION['PENALTY_ALPHA']
    final_z1 = z1_norm - alpha * v_norm
    final_z2 = z2_norm - alpha * v_norm

    return final_z1, final_z2


def evaluate_with_details(individual, G, od_df, node_positions, fixed_tasks):
    """
    增强版评估：除 (Z1, Z2) 外，返回路径质量细节用于持久化记录。
    保证与 evaluate_individual 返回值一致。

    :return: {
        "z1": float, "z2": float,
        "total_length_m": float, "avg_detour_ratio": float,
        "num_routes": int, "total_segments": int,
        "route_stats": [{num_stops, length, straight_dist, detour_ratio}, ...]
    }
    """
    all_edges = []
    edge_counts = {}
    route_stats = []
    transit_subgraph = nx.Graph()
    total_segments = 0

    for idx, route in enumerate(individual):
        if not route or len(route) < 2:
            continue
        line_id, s_r, t_r = fixed_tasks[idx]
        route_len = 0
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if G.has_edge(u, v):
                all_edges.append((u, v))
                total_segments += 1
                edge_sig = tuple(sorted((u, v)))
                edge_counts[edge_sig] = edge_counts.get(edge_sig, 0) + 1
                route_len += G[u][v].get('weight', 0)
                transit_subgraph.add_edge(u, v)

        if s_r in node_positions and t_r in node_positions:
            straight_dist = calculate_haversine_distance(
                node_positions[s_r], node_positions[t_r]
            )
        else:
            straight_dist = 0.1
        if straight_dist == 0:
            straight_dist = 0.1

        detour_ratio = route_len / straight_dist
        route_stats.append({
            'num_stops': len(route),
            'length': route_len,
            'straight_dist': straight_dist,
            'detour_ratio': detour_ratio,
        })

    # Z1 归一化
    if total_segments > 0:
        z1_raw = sum(G[u][v].get('net_visual', 0) for u, v in all_edges) / total_segments
        z1_norm = normalize_z1_raw(z1_raw)
    else:
        z1_norm = NORMALIZATION['Z1_EMPTY_PENALTY']

    # Z2 归一化
    z2_raw = 0
    sub_nodes = set(transit_subgraph.nodes())
    valid_od = od_df[od_df['from_node'].isin(sub_nodes) & od_df['to_node'].isin(sub_nodes)]
    for _, row in valid_od.iterrows():
        o, d = int(row['from_node']), int(row['to_node'])
        if nx.has_path(transit_subgraph, o, d):
            z2_raw += row['count']
    z2_norm = normalize_z2_raw(z2_raw)

    # 对称软惩罚（与 evaluate_individual 一致）
    v_raw = compute_violation_degree(route_stats, edge_counts)
    v_norm = normalize_violation(v_raw)
    alpha = NORMALIZATION['PENALTY_ALPHA']

    total_length = sum(s['length'] for s in route_stats)
    avg_detour = (sum(s['detour_ratio'] for s in route_stats) / len(route_stats)
                  if route_stats else 0.0)

    return {
        "z1": z1_norm - alpha * v_norm,
        "z2": z2_norm - alpha * v_norm,
        "total_length_m": total_length,
        "avg_detour_ratio": avg_detour,
        "num_routes": len(route_stats),
        "total_segments": total_segments,
        "route_stats": route_stats,
    }