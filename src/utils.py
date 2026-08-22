import math
import networkx as nx
import copy

def calculate_haversine_distance(coord1, coord2):
    """
    计算两点经纬度之间的球面距离 (Haversine 公式)
    :param coord1: (lon, lat) tuple, e.g., (113.90, 22.50)
    :param coord2: (lon, lat) tuple
    :return: 距离 (米)
    """
    lon1, lat1 = coord1
    lon2, lat2 = coord2

    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine 公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    r = 6371000  # 地球平均半径，单位为米
    return c * r


import networkx as nx
import copy
import pandas as pd


class CompatibilityEnv:
    """
    适配器类：将 BusNetworkGA 实例转换为 LLM 期望的 env 结构。
    严格基于 load_data 和 build_graph 的数据结构构建，无假设。
    """

    def __init__(self, ga_instance):
        # 1. 复制并适配图结构 (Core Graph Structure)
        # 这里的 G 来源于 ga_instance.G，由 build_graph 构建
        self.G = copy.deepcopy(ga_instance.G)
        self._adapt_graph_attributes(ga_instance.node_positions)

        # 2. 转换流数据 (Traffic Flow Data)
        # 数据来源：ga_instance.od_df，由 load_data 读取
        self.flow = self._build_flow_dict(ga_instance.od_df)

        # 3. 构建辅助映射 (Helper Attributes)
        self.tourist_map = {}
        self.distance_map = {}
        self.PI_map = {}
        self.NI_map = {}
        self._build_helper_maps()

        # 4. 构建代价矩阵 (Cost Matrix)
        # 初始化为空字典结构，防止算子直接调用报错
        # 实际逻辑应引导 LLM 使用 nx.shortest_path
        self.cost_matrix = {n: {} for n in self.G.nodes}

        # 5. 传递固定任务
        # 数据来源：ga_instance.fixed_tasks
        self.fixed_tasks = ga_instance.fixed_tasks

        # 5.5. 传递 OD 数据（新增）
        # 数据来源：ga_instance.od_df
        self.od = ga_instance.od_df

        # 6. Route Initialization (结构占位，用于兼容性)
        self.init_route = {'arc': [], 'stop_num': 0, 'start_end': []}

    def _adapt_graph_attributes(self, node_positions):
        """
        将 ga.G 的属性映射为 env.G 期望的格式。
        基于 build_graph 函数逻辑：
        - 节点有 'demand_heat'
        - 边有 'weight', 'net_visual', 'visual_cost', 'demand_cost'
        """

        # --- 节点属性适配 ---
        for node_id in self.G.nodes:
            # node_positions 来源于 load_data -> build_graph 返回
            if node_id in node_positions:
                x, y = node_positions[node_id]
                self.G.nodes[node_id]['lng'] = x
                self.G.nodes[node_id]['lat'] = y

            # 你的 build_graph 中没有设置 'tourist'，这里显式初始化为 0 以防报错
            if 'tourist' not in self.G.nodes[node_id]:
                self.G.nodes[node_id]['tourist'] = 0

        # --- 边属性适配 ---
        for u, v, data in self.G.edges(data=True):
            # 你的 build_graph 中使用的是 'weight'，LLM 文档里是 'length'
            # 必须做映射，否则 LLM 生成的代码用 data['length'] 会 KeyError
            if 'weight' in data:
                data['length'] = data['weight']

            # 你的 build_graph 计算了 'net_visual'
            # LLM 文档需要 PI 和 NI，这里根据净值反推（确保字段存在）
            net_v = data.get('net_visual', 0)
            if 'PI' not in data:
                data['PI'] = max(0, net_v)
            if 'NI' not in data:
                data['NI'] = max(0, -net_v)

    def _build_flow_dict(self, od_df):
        """
        将 DataFrame 转换为嵌套字典: flow[u][v] = count
        基于 load_data 中的 groupby 逻辑，列名必然是 from_node, to_node, count
        """
        flow_dict = {}

        # 检查必要列是否存在，虽然 load_data 已经隐式检查过了，但这能防止意外传入错误数据
        required_cols = {'from_node', 'to_node', 'count'}
        if not required_cols.issubset(od_df.columns):
            raise ValueError(f"od_df 缺少必要列: {required_cols - set(od_df.columns)}")

        # 使用 itertuples 遍历，效率极高且不依赖列索引位置
        for row in od_df.itertuples(index=False):
            # load_data 中已强制转换为 int，这里直接使用
            u = getattr(row, 'from_node')
            v = getattr(row, 'to_node')
            c = float(getattr(row, 'count'))

            if u not in flow_dict:
                flow_dict[u] = {}
            # 累加流量 (防止同一OD对有多行记录)
            flow_dict[u][v] = flow_dict[u].get(v, 0.0) + c

        return flow_dict

    def _build_helper_maps(self):
        """构建快速查找表，完全基于 adapt 后的图属性"""
        for u, v, data in self.G.edges(data=True):
            # 此时 data['length'] 肯定存在（由 _adapt_graph_attributes 保证）
            dist = data.get('length', 0.0)
            self.distance_map[(u, v)] = dist
            self.distance_map[(v, u)] = dist

            self.PI_map[(u, v)] = data.get('PI', 0.0)
            self.NI_map[(u, v)] = data.get('NI', 0.0)

        for n, data in self.G.nodes(data=True):
            self.tourist_map[n] = data.get('tourist', 0)


def create_compatible_env(ga_instance):
    """工厂函数"""
    return CompatibilityEnv(ga_instance)