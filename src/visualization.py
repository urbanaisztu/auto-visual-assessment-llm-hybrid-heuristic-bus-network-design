"""
可视化结果路径图（适配新版数据结构）。

旧版来自 old_version_temp/visualization.py，主要修改：
  1. plot_network 签名：旧 (env, routes, show_num, save_name, title, show) 顺序混乱
     → 新 (G, node_positions, routes, title, save_path, show, show_num)
  2. 节点坐标：旧 env.G._node[id]['lng'/'lat'] → 新 node_positions[id] = (lng, lat)
  3. 输入结构：旧版 routes 是单条线路（一维列表，需 get_sub_route 拆分）
     → 新版 routes 是 10 条线路的嵌套列表（直接对应 final_pareto_legacy.json 的 path）
  4. 删除对 function.get_sub_route 的依赖（新版未实现）
  5. 增加 CLI：直接读取 final_pareto_legacy.json / <algo>_pareto.json 中的某个解并绘图
  6. 完善图例、起终点标记、保存高质量 PNG/PDF
"""
import os
import sys
import json
import argparse

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# 项目模块（保证从 src/ 运行）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data

# 配色（保留旧版色板）+ 10 条线路足够使用
COLORS = [
    '#C6B8FF', '#F86287', '#56498A', '#791288', '#05D74D',
    '#1381AD', '#E3DD93', '#F50DCC', '#5D278E', '#B981BC',
    '#D481D6', '#FB45F1', '#50C3EB', '#0DF80A', '#5C2EF2',
]
MARKERS = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']


def plot_network(G, node_positions, routes, title='Bus Routes',
                 save_path=None, show=False, show_num=None,
                 show_background=True, bg_color='#DCDCDC',
                 figsize=(12, 8), dpi=300):
    """
    在地理路网底图上绘制多条公交路径。

    :param G: networkx.Graph，仅用于绘制底图边
    :param node_positions: {node_id: (lng, lat)}
    :param routes: [[node_id, ...], ...]  嵌套列表，每条子列表为一条线路
    :param title: 图标题
    :param save_path: 保存路径（.png/.pdf/.jpg）；None 则不保存
    :param show: 是否 plt.show()（脚本运行建议 False）
    :param show_num: 最多绘制前几条线路；None 表示全部
    :param show_background: 是否绘制路网底图
    :param bg_color: 底图颜色
    :param figsize: 画布大小
    :param dpi: 保存分辨率
    :return: matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # === 1. 路网底图 ===
    if show_background:
        xs_a, xs_b = [], []
        ys_a, ys_b = [], []
        for u, v in G.edges():
            if u in node_positions and v in node_positions:
                xs_a.append(node_positions[u][0]); xs_b.append(node_positions[v][0])
                ys_a.append(node_positions[u][1]); ys_b.append(node_positions[v][1])
        # 用循环逐边画太慢；批量构造线段集合后统一画
        for i in range(len(xs_a)):
            ax.plot([xs_a[i], xs_b[i]], [ys_a[i], ys_b[i]],
                    '-', color=bg_color, linewidth=0.4, zorder=1)

    # === 2. 公交线路 ===
    n = len(routes) if show_num is None else min(show_num, len(routes))
    legend_handles = []
    for c in range(n):
        route = routes[c]
        if not route or len(route) < 2:
            continue
        # 仅保留图中存在的节点（防止 JSON 中混入未知 ID）
        pts = [nid for nid in route if nid in node_positions]
        if len(pts) < 2:
            continue
        xs = [node_positions[nid][0] for nid in pts]
        ys = [node_positions[nid][1] for nid in pts]

        color = COLORS[c % len(COLORS)]
        marker = MARKERS[c % len(MARKERS)]

        # 路径线
        ax.plot(xs, ys, '-', color=color, linewidth=2.0, alpha=0.85, zorder=3)
        # 中间站点点
        ax.scatter(xs[1:-1], ys[1:-1], s=18, color=color,
                   marker='o', edgecolors='white', linewidths=0.4, zorder=4)
        # 起点（实心大点）
        ax.scatter([xs[0]], [ys[0]], s=110, color=color,
                   marker=marker, edgecolors='black', linewidths=0.8, zorder=5)
        # 终点（空心大点）
        ax.scatter([xs[-1]], [ys[-1]], s=110, facecolors='white', edgecolors=color,
                   marker=marker, linewidths=1.6, zorder=5)

        legend_handles.append(
            mlines.Line2D([], [], color=color, marker=marker, linestyle='-', linewidth=2,
                          markersize=8, markerfacecolor=color,
                          markeredgecolor='black', label=f'Route {c+1}: {pts[0]} → {pts[-1]}')
        )

    # === 3. 装饰 ===
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle=':', alpha=0.3)
    if legend_handles:
        ax.legend(handles=legend_handles, loc='best', fontsize=8,
                  framealpha=0.9, frameon=True)

    plt.tight_layout()

    # === 4. 保存/显示 ===
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f">>> 图像已保存：{save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def load_solution(json_path, sol_index=0):
    """从 final_pareto_legacy.json / <algo>_pareto.json 读取指定解。
    返回 (routes, fitness)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list) or sol_index >= len(data):
        raise IndexError(f"解索引 {sol_index} 超出范围（共 {len(data) if isinstance(data, list) else 0} 个解）")
    sol = data[sol_index]
    routes = sol.get('path', sol.get('routes', []))
    fitness = sol.get('fitness', {})
    return routes, fitness


def main():
    parser = argparse.ArgumentParser(description="可视化某个帕累托解的 10 条公交线路")
    parser.add_argument('--algo', default='LEHHA', help='算法名（如 LEHHA / NSGA2 / MOEAD / MOPSO / WSGATS）')
    parser.add_argument('--run', type=int, default=1, help='run 编号 1-10')
    parser.add_argument('--sol', type=int, default=0, help='解索引（帕累托前沿中的第几个）')
    parser.add_argument('--out', default=None, help='输出文件路径（默认 plot/routes_<algo>_runXX_solYY.png）')
    parser.add_argument('--show', action='store_true', help='交互式显示（默认关闭，仅保存）')
    parser.add_argument('--no-bg', action='store_true', help='不绘制路网底图')
    args = parser.parse_args()

    # === 1. 加载图与节点坐标 ===
    _, _, _, G, node_positions = load_data()

    # === 2. 定位解文件 ===
    run_dir = f"/workspace/2025/results/{args.algo}/run_{args.run:02d}"
    legacy = os.path.join(run_dir, 'final_pareto_legacy.json')
    alt = os.path.join(run_dir, f"{args.algo}_pareto.json")
    json_path = legacy if os.path.exists(legacy) else alt
    if not os.path.exists(json_path):
        print(f"错误：找不到解文件，尝试过：\n  {legacy}\n  {alt}")
        sys.exit(1)

    # === 3. 读取解 ===
    routes, fitness = load_solution(json_path, args.sol)
    fv = fitness.get('visual', fitness.get('z1_visual', float('nan')))
    fs = fitness.get('satisfy', fitness.get('z2_demand', float('nan')))
    title = (f"{args.algo} run_{args.run:02d}  Solution #{args.sol}  "
             f"(Z1={fv:.4f}, Z2={fs:.4f}, {len(routes)} routes)")

    # === 4. 输出路径 ===
    out_path = args.out or f"/workspace/2025/plot/routes_{args.algo}_run{args.run:02d}_sol{args.sol:02d}.png"

    # === 5. 绘制 ===
    plot_network(G, node_positions, routes,
                 title=title, save_path=out_path,
                 show=args.show, show_background=not args.no_bg)


if __name__ == '__main__':
    main()
