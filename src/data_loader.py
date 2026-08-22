import geopandas as gpd
import pandas as pd
import networkx as nx
from config import FILE_PATHS, VISUAL_WEIGHTS


def load_data():
    """
    读取数据并构建网络
    更新：增加节点需求热度统计，用于启发式搜索
    """
    print("--- 正在加载数据 ---")

    # 1. 读取基础数据
    edges_gdf = gpd.read_file(FILE_PATHS['edges'])
    nodes_gdf = gpd.read_file(FILE_PATHS['nodes'])
    route_nodes_gdf = gpd.read_file(FILE_PATHS['route_nodes'])
    od_df = pd.read_csv(FILE_PATHS['od'])

    # 2. 强制类型转换 (确保 ID 一致性)
    try:
        edges_gdf['source'] = edges_gdf['source'].astype(int)
        edges_gdf['target'] = edges_gdf['target'].astype(int)
        nodes_gdf['node_id'] = nodes_gdf['node_id'].astype(int)
        route_nodes_gdf['node_id'] = route_nodes_gdf['node_id'].astype(int)
        od_df['from_node'] = od_df['from_node'].astype(int)
        od_df['to_node'] = od_df['to_node'].astype(int)
    except Exception as e:
        print(f"数据类型转换错误: {e}")
        raise

    # 3. 预计算节点需求热度 (Node Demand Heat)
    # 目的：为图中的节点赋予"热度"属性，指导算法优先搜索高需求区域
    print("--- 正在计算节点需求热度 ---")
    origin_counts = od_df.groupby('from_node')['count'].sum()
    dest_counts = od_df.groupby('to_node')['count'].sum()

    # 将 Series 转为字典，并合并 (O + D)
    # 使用 fill_value=0 保证某个点只做起点或只做终点时不出错
    total_demand = origin_counts.add(dest_counts, fill_value=0)
    node_demand_map = total_demand.to_dict()

    # 4. 构建图 (传递 demand_map 以计算启发式权重)
    G, node_positions = build_graph(edges_gdf, nodes_gdf, node_demand_map)

    # 5. 解析固定任务
    fixed_tasks = parse_fixed_route_tasks(route_nodes_gdf)

    # 6. 填充归一化常量 Z2_TOTAL_DEMAND（详见 docs/plans/2026-07-16-objective-normalization-design.md）
    import config as _config
    _config.NORMALIZATION['Z2_TOTAL_DEMAND'] = float(od_df['count'].sum())
    print(f"--- 已填充 Z2_TOTAL_DEMAND = {_config.NORMALIZATION['Z2_TOTAL_DEMAND']:.0f} ---")

    return edges_gdf, od_df, fixed_tasks, G, node_positions


def build_graph(edges_gdf, nodes_gdf, node_demand_map):
    """
    构建无向图，并计算多种权重的阻抗
    :param node_demand_map: {node_id: total_passenger_count}
    """
    G = nx.Graph()

    # 1. 建立节点坐标索引 & 写入需求属性
    node_positions = {}
    for _, row in nodes_gdf.iterrows():
        nid = int(row['node_id'])
        node_positions[nid] = (row.geometry.x, row.geometry.y)

        # 将需求热度写入节点属性
        heat = node_demand_map.get(nid, 0)
        G.add_node(nid, demand_heat=heat)

    # 2. 添加边属性 (计算多种 Cost)
    print("--- 正在构建图并计算启发式权重 ---")
    for idx, row in edges_gdf.iterrows():
        u = int(row['source'])
        v = int(row['target'])
        length = float(row['length'])

        # --- A. 视觉评分计算 ---
        pi = row.get('PI', 0)
        ni = row.get('NI', 0)
        # 净视觉值 (用于目标函数 Z1 的计算)
        net_visual = (VISUAL_WEIGHTS['W1'] * pi) - (VISUAL_WEIGHTS['W2'] * ni)

        # --- B. 启发式权重设计 (用于寻路变异) ---

        # 1. Visual Cost (视觉阻抗)
        # 逻辑：视觉越好，阻抗越小 (看似距离更短)。
        # 假设 net_visual 可能为负 (e.g. -20 到 100)，先平移保证为正
        visual_score = max(0, net_visual + 20)
        # 公式：实际长度 / (1 + 视觉加成)
        # 视觉分很高时，cost 会显著低于物理长度
        visual_cost = length / (1.0 + 0.5 * visual_score)

        # 2. Demand Cost (需求阻抗)
        # 逻辑：连接高需求点的边，阻抗越小。
        u_heat = node_demand_map.get(u, 0)
        v_heat = node_demand_map.get(v, 0)
        avg_heat = (u_heat + v_heat) / 2.0
        # 公式：实际长度 / (1 + 需求加成)
        # 这是一个简单的缩放，可根据实际客流数值范围(几十还是几万)调整系数 0.05
        demand_cost = length / (1.0 + 0.05 * avg_heat)

        # 3. 添加边
        G.add_edge(u, v,
                   weight=length,  # 物理长度 (用于计算约束和统计)
                   net_visual=net_visual,  # 真实视觉值 (用于评估 Z1)
                   visual_cost=visual_cost,  # 启发式权重 (用于生成高视觉初值)
                   demand_cost=demand_cost,  # 启发式权重 (用于生成高客流初值)
                   edge_id=row.get('edge_id'))

    print(f"图构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    return G, node_positions


def parse_fixed_route_tasks(route_nodes_gdf):
    """解析固定起终点任务 (保持不变)"""
    tasks = {}
    for _, row in route_nodes_gdf.iterrows():
        lid = row['line_id']
        ntype = row['type']
        nid = int(row['node_id'])

        if lid not in tasks: tasks[lid] = {}
        tasks[lid][ntype] = nid

    task_list = []
    for lid, nodes in tasks.items():
        if 'origin' in nodes and 'destination' in nodes:
            task_list.append((lid, nodes['origin'], nodes['destination']))

    return task_list

