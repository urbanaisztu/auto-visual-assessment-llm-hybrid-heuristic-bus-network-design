"""
Friedman 检验 + Nemenyi 事后检验。
读取 results/{ALGO}/run_NN/run_summary.json，构建 HV 矩阵，输出统计结果。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 保证能 import 上级目录模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics_recorder import compute_dynamic_ref_point


def collect_run_summaries(results_root: Path):
    """
    扫描 results/{ALGO}/run_NN/run_summary.json
    :return: DataFrame，index=[(algo, run_id)], cols=[z1_max, z2_max, final_hv, ...]
    """
    records = []
    for algo_dir in sorted(results_root.iterdir()):
        if not algo_dir.is_dir() or algo_dir.name.startswith("_"):
            continue
        algo = algo_dir.name
        for run_dir in sorted(algo_dir.glob("run_*")):
            summary_file = run_dir / "run_summary.json"
            if not summary_file.exists():
                continue
            try:
                with open(summary_file, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("status") != "success":
                    continue
                records.append({
                    "algo": algo,
                    "run_id": data.get("run_id", int(run_dir.name.split("_")[1])),
                    "z1_max": data.get("z1_max", 0.0),
                    "z2_max": data.get("z2_max", 0.0),
                    "final_hv": data.get("final_hv", 0.0),
                })
            except Exception as e:
                print(f"[WARN] 读取 {summary_file} 失败: {e}")
    return pd.DataFrame(records)


def build_hv_matrix(df: pd.DataFrame, runs: int):
    """构建 algo × run_id 的 HV 矩阵"""
    pivot = df.pivot(index="algo", columns="run_id", values="final_hv")
    # 只保留前 N 次 run
    pivot = pivot.iloc[:, :runs]
    return pivot


def rebuild_hv_with_dynamic_ref(df: pd.DataFrame, results_root: Path, runs: int):
    """
    用动态参考点重新计算 HV。
    需要 final_pareto.csv 中有 z1/z2 数据。
    """
    all_objs = []
    algo_run_objs = {}  # {(algo, run_id): [(z1, z2), ...]}

    for _, row in df.iterrows():
        algo = row["algo"]
        run_id = row["run_id"]
        fp_path = results_root / algo / f"run_{run_id:02d}" / "final_pareto.csv"
        if not fp_path.exists():
            continue
        try:
            pareto_df = pd.read_csv(fp_path)
            objs = pareto_df[["z1_visual", "z2_demand"]].values
            algo_run_objs[(algo, run_id)] = objs
            all_objs.append(objs)
        except Exception:
            continue

    if not all_objs:
        return None

    all_objs_concat = np.vstack(all_objs)
    ref_dynamic = compute_dynamic_ref_point(all_objs_concat, margin_ratio=1.1)

    from metrics_recorder import compute_hv
    hv_dict = {}
    for (algo, run_id), objs in algo_run_objs.items():
        hv_dict[(algo, run_id)] = compute_hv(objs, ref_dynamic)

    ref_zero = np.array([0.0, 0.0])
    hv_dict_zero = {}
    for (algo, run_id), objs in algo_run_objs.items():
        hv_dict_zero[(algo, run_id)] = compute_hv(objs, ref_zero)

    # 转 DataFrame
    matrix_dyn = pd.Series(hv_dict).unstack(level=0).T
    matrix_zero = pd.Series(hv_dict_zero).unstack(level=0).T
    return matrix_dyn, matrix_zero, ref_dynamic, ref_zero


def run_friedman(matrix: pd.DataFrame, out_dir: Path, label: str):
    """运行 Friedman + Nemenyi 检验"""
    from scipy.stats import friedmanchisquare
    import scikit_posthocs as sp

    out_dir.mkdir(parents=True, exist_ok=True)

    # Friedman
    cols_data = [matrix.iloc[i, :].dropna().values for i in range(matrix.shape[0])]
    if any(len(d) < 3 for d in cols_data):
        print(f"[WARN] {label}：样本量不足（<3），跳过")
        return
    stat, p = friedmanchisquare(*cols_data)

    friedman_result = {
        "label": label,
        "chi_square": float(stat),
        "df": int(matrix.shape[0] - 1),
        "p_value": float(p),
        "alpha": 0.05,
        "significant": bool(p < 0.05),
        "n_runs": int(matrix.shape[1]),
        "n_algorithms": int(matrix.shape[0]),
    }
    with open(out_dir / f"friedman_result_{label}.json", "w", encoding="utf-8") as f:
        json.dump(friedman_result, f, ensure_ascii=False, indent=2)
    print(f"[Friedman/{label}] χ²={stat:.4f}, p={p:.4e}, "
          f"显著={friedman_result['significant']}")

    # 平均排名（越大越好）
    ranks = matrix.rank(axis=0, ascending=False).mean(axis=1)
    ranks_df = pd.DataFrame({"avg_rank": ranks}).sort_values("avg_rank")
    ranks_df.to_csv(out_dir / f"hv_ranks_{label}.csv")

    # Nemenyi 事后检验（仅当 Friedman 显著）
    if friedman_result["significant"]:
        try:
            # scikit_posthocs 要求 (n_samples, n_groups)
            posthoc = sp.posthoc_nemenyi_friedman(matrix.T.values)
            posthoc.index = matrix.index
            posthoc.columns = matrix.index
            posthoc.to_csv(out_dir / f"posthoc_nemenyi_{label}.csv")
            print(f"[Nemenyi/{label}] 已保存事后检验结果")
        except Exception as e:
            print(f"[WARN] Nemenyi 失败: {e}")

    # 保存矩阵
    matrix.to_csv(out_dir / f"hv_matrix_{label}.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="../results",
                        help="结果根目录（默认相对 src/）")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    results_root = Path(args.results).resolve()
    if not results_root.is_absolute():
        results_root = (Path(__file__).resolve().parent.parent.parent / args.results).resolve()

    print(f"扫描结果目录: {results_root}")
    df = collect_run_summaries(results_root)
    if df.empty:
        print("[FAIL] 未找到任何 run_summary.json")
        return

    out_dir = results_root / "_friedman"
    out_dir.mkdir(exist_ok=True)

    # 1. 用各 run_summary 中已有的 final_hv 构建（zero ref）
    matrix_zero = build_hv_matrix(df, args.runs)
    run_friedman(matrix_zero, out_dir, "zero_ref_from_summary")

    # 2. 重新计算动态参考点 HV
    result = rebuild_hv_with_dynamic_ref(df, results_root, args.runs)
    if result is not None:
        matrix_dyn, matrix_zero2, ref_dyn, ref_zero = result
        # 保存参考点
        with open(out_dir / "ref_point.json", "w", encoding="utf-8") as f:
            json.dump({
                "dynamic_ref": ref_dyn.tolist(),
                "zero_ref": ref_zero.tolist(),
            }, f, indent=2)
        run_friedman(matrix_dyn, out_dir, "dynamic_ref")
        run_friedman(matrix_zero2, out_dir, "zero_ref_rebuilt")

    print(f"\n所有 Friedman 结果已保存到 {out_dir}")


if __name__ == "__main__":
    main()
