"""
路径质量评估：总长度、平均绕路系数。
基准（现有公交路线）与优化（帕累托前沿均值）共用此模块。
"""
import networkx as nx
from typing import Any, Dict, List, Tuple

from utils import calculate_haversine_distance


def evaluate_individual_quality(
    individual: List[List[int]],
    G: nx.Graph,
    node_positions: Dict[int, Tuple[float, float]],
    fixed_tasks: List[Tuple],
) -> Dict[str, Any]:
    """
    计算单个个体（多线路方案）的总长度与平均绕路系数。

    :param individual: [[node1, node2, ...], ...] 线路列表
    :param G: NetworkX 图
    :param node_positions: {node_id: (lng, lat)}
    :param fixed_tasks: [(line_id, start, end), ...]
    :return: {
        "total_length_m": float,
        "avg_detour_ratio": float,  # 所有线路算术平均
        "num_routes": int,
        "route_details": [{"length_m": ..., "detour_ratio": ..., "num_stops": ...}, ...]
    }
    """
    route_details = []
    total_length = 0.0

    for idx, route in enumerate(individual):
        if not route or len(route) < 2:
            continue

        # 累加路径实际长度
        route_len = 0.0
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if G.has_edge(u, v):
                route_len += G[u][v].get("weight", 0.0)

        # 起终点直线距离
        line_id, s_r, t_r = fixed_tasks[idx] if idx < len(fixed_tasks) else (-1, route[0], route[-1])
        straight_dist = 0.1
        if s_r in node_positions and t_r in node_positions:
            straight_dist = calculate_haversine_distance(
                node_positions[s_r], node_positions[t_r]
            )
        if straight_dist == 0:
            straight_dist = 0.1

        detour_ratio = route_len / straight_dist
        route_details.append({
            "length_m": route_len,
            "detour_ratio": detour_ratio,
            "num_stops": len(route),
        })
        total_length += route_len

    num_routes = len(route_details)
    avg_detour = sum(r["detour_ratio"] for r in route_details) / num_routes if num_routes else 0.0

    return {
        "total_length_m": total_length,
        "avg_detour_ratio": avg_detour,
        "num_routes": num_routes,
        "route_details": route_details,
    }


def evaluate_existing_routes_from_geodataframe(
    lines_gdf,
    nodes_gdf=None,
    line_id_field: str = "layer",
) -> Dict[str, Any]:
    """
    从 route_lines.geojson 计算现有公交路线的总长度与平均绕路系数。

    该实现完全基于 GeoDataFrame 自带的 LINESTRING 几何信息：
    - 路径长度：使用 shapely geometry.length（注意：geographic CRS 下单位为度，
      乘以 111320 近似转为米）。如果 GeoDataFrame 已是 projected CRS，
      则直接为米。这里使用 Haversine 公式按坐标点逐段累加得到真实米数。
    - 起终点直线距离：取 LINESTRING 起点和终点坐标，用 Haversine 计算。
    - 绕路系数 = 路径长度 / 起终点直线距离。

    :param lines_gdf: GeoDataFrame，至少包含 line_id_field 字段和 LINESTRING geometry。
                     通常来源于 data/input/route_lines.geojson。
    :param nodes_gdf: 兼容性参数，未使用。route_nodes.geojson 只含 origin/destination，
                     不能用于路径长度计算；保留参数以便未来扩展。
    :param line_id_field: 用于分组的线路 ID 字段名（route_lines.geojson 中默认是 'layer'）。
    :return: 同 evaluate_individual_quality 的返回格式（不含 fixed_tasks 信息）
    """
    import math
    from shapely.geometry import LineString, MultiLineString

    if lines_gdf is None or len(lines_gdf) == 0:
        return {
            "total_length_m": 0.0,
            "avg_detour_ratio": 0.0,
            "num_routes": 0,
            "route_details": [],
        }

    def _haversine_m(p1, p2):
        """(lon, lat) -> meters"""
        lon1, lat1 = p1
        lon2, lat2 = p2
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return c * 6371000.0

    def _linestring_coords(geom):
        """统一从 LineString / MultiLineString 提取坐标点序列"""
        if isinstance(geom, LineString):
            return list(geom.coords)
        if isinstance(geom, MultiLineString):
            coords = []
            for part in geom.geoms:
                coords.extend(list(part.coords))
            return coords
        return []

    def _segment_length_m(coords):
        """按相邻坐标逐段 Haversine 累加"""
        total = 0.0
        for i in range(len(coords) - 1):
            total += _haversine_m(coords[i], coords[i + 1])
        return total

    # 按 line_id_field 分组聚合每条线路的所有线段
    if line_id_field not in lines_gdf.columns:
        # 兜底：尝试常见字段名
        for fallback in ("line_id", "line_name", "ArcID"):
            if fallback in lines_gdf.columns:
                line_id_field = fallback
                break
        else:
            # 实在找不到分组字段，把所有 geometry 合并成单条线路处理
            line_id_field = None

    route_details = []
    if line_id_field is not None:
        grouped = lines_gdf.groupby(line_id_field)
    else:
        # 用一个虚拟分组
        grouped = [("ALL_LINE", lines_gdf)]

    for line_key, group in grouped:
        # 合并该线路下所有 LINESTRING 段的坐标点
        all_coords = []
        for geom in group.geometry:
            all_coords.extend(_linestring_coords(geom))

        if len(all_coords) < 2:
            continue

        # 路径长度（米）
        length_m = _segment_length_m(all_coords)

        # 起终点直线距离（米）
        start = all_coords[0]
        end = all_coords[-1]
        straight_m = _haversine_m(start, end)
        if straight_m < 0.1:
            straight_m = 0.1

        detour = length_m / straight_m
        route_details.append({
            "length_m": length_m,
            "detour_ratio": detour,
            "num_stops": len(all_coords),
            "line_id": str(line_key),
        })

    total_length = sum(r["length_m"] for r in route_details)
    num_routes = len(route_details)
    avg_detour = (sum(r["detour_ratio"] for r in route_details) / num_routes
                  if num_routes else 0.0)

    return {
        "total_length_m": total_length,
        "avg_detour_ratio": avg_detour,
        "num_routes": num_routes,
        "route_details": route_details,
    }
