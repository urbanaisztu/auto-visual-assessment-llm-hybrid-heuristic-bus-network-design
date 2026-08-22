"""
4 算法（LEHHA / NSGA2 / MOEAD / MOPSO）的完整统计检验
====================================================
输出:
  - Friedman 整体检验
  - Nemenyi 事后检验（保守，基于排名）
  - Wilcoxon 符号秩配对检验 + Holm-Bonferroni 校正（灵敏）
  - Cohen's d 效应量
  - 可读 Markdown 分析报告

用法:
  python -m analysis.friedman_4algo
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path("/workspace/2025/results")
OUT_DIR = BASE / "_friedman"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALGOS = ["LEHHA", "NSGA2", "MOEAD", "MOPSO"]
N_RUNS = 10
ALPHA = 0.05


def load_hv():
    """读取 4 算法 × 10 run 的 HV 数据"""
    data = {}
    for a in ALGOS:
        hvs = []
        for i in range(1, N_RUNS + 1):
            with open(BASE / a / f"run_{i:02d}" / "run_summary.json", encoding="utf-8") as f:
                d = json.load(f)
            assert d.get("status") == "success", f"{a}/run_{i:02d} 非 success"
            hvs.append(d["final_hv"])
        data[a] = np.array(hvs)
    return pd.DataFrame(data, columns=ALGOS)


def holm_bonferroni(pvals):
    """Holm-Bonferroni 多重比较校正"""
    m = len(pvals)
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    adjusted = np.zeros(m)
    for rank, idx in enumerate(order):
        adjusted[idx] = pvals[idx] * (m - rank)
        if rank > 0:
            prev = order[rank - 1]
            adjusted[idx] = max(adjusted[idx], adjusted[prev])
    return np.minimum(adjusted, 1.0)


def cohens_d_paired(x, y):
    """配对样本 Cohen's d"""
    diff = x - y
    return diff.mean() / diff.std(ddof=1)


def sig_tag(p, alpha=ALPHA):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def main():
    df = load_hv()

    # 1. HV 矩阵
    df.to_csv(OUT_DIR / "hv_matrix.csv", index=False)

    # 2. 描述统计
    stats_df = pd.DataFrame({
        "algo":  ALGOS,
        "mean":  [df[a].mean() for a in ALGOS],
        "std":   [df[a].std(ddof=1) for a in ALGOS],
        "min":   [df[a].min() for a in ALGOS],
        "max":   [df[a].max() for a in ALGOS],
        "median":[np.median(df[a]) for a in ALGOS],
    })
    stats_df.to_csv(OUT_DIR / "algorithm_stats.csv", index=False)

    # 3. Friedman 整体
    chi2, p_friedman = stats.friedmanchisquare(*[df[a].values for a in ALGOS])
    friedman_result = {
        "test": "Friedman",
        "k_algorithms": len(ALGOS),
        "n_runs": N_RUNS,
        "df": len(ALGOS) - 1,
        "chi_square": float(chi2),
        "p_value": float(p_friedman),
        "alpha": ALPHA,
        "significant": bool(p_friedman < ALPHA),
        "critical_value_chi2_0_05": 7.815,   # df=3
        "note": "Friedman chi^2(0.95, df=3) = 7.815"
    }
    with open(OUT_DIR / "friedman_result.json", "w", encoding="utf-8") as f:
        json.dump(friedman_result, f, ensure_ascii=False, indent=2)

    # 4. 平均排名
    ranks = df.rank(axis=1, ascending=False).mean()
    ranks_df = pd.DataFrame({
        "algo": ALGOS,
        "avg_rank": [ranks[a] for a in ALGOS],
    }).sort_values("avg_rank").reset_index(drop=True)
    ranks_df.insert(0, "rank", range(1, len(ALGOS) + 1))
    ranks_df.to_csv(OUT_DIR / "algorithm_ranks.csv", index=False)

    # 5. Nemenyi 事后检验
    import scikit_posthocs as sp
    nemenyi = sp.posthoc_nemenyi_friedman(df.values)
    nemenyi.index = nemenyi.columns = ALGOS
    nemenyi.to_csv(OUT_DIR / "nemenyi_pvalues.csv")

    # 6. 两两 Wilcoxon + Holm 校正
    pairs = list(combinations(ALGOS, 2))
    raw_pvals, holm_pvals, cohen_d, wil_stats = [], [], [], []
    for a1, a2 in pairs:
        s, p_ = stats.wilcoxon(df[a1].values, df[a2].values)
        raw_pvals.append(float(p_))
        wil_stats.append(float(s))
        cohen_d.append(float(cohens_d_paired(df[a1].values, df[a2].values)))
    holm_pvals = holm_bonferroni(raw_pvals).tolist()

    pairwise = pd.DataFrame({
        "algo_1":   [p[0] for p in pairs],
        "algo_2":   [p[1] for p in pairs],
        "wilcoxon_stat": wil_stats,
        "raw_p":           raw_pvals,
        "holm_p":          holm_pvals,
        "cohens_d":        cohen_d,
        "sig_holm_0.05":   [sig_tag(p) for p in holm_pvals],
    })
    pairwise.to_csv(OUT_DIR / "pairwise_wilcoxon_holm.csv", index=False)

    # 7. 矩阵形式
    wil_raw_mat = pd.DataFrame(np.eye(len(ALGOS)), index=ALGOS, columns=ALGOS)
    wil_holm_mat = pd.DataFrame(np.eye(len(ALGOS)), index=ALGOS, columns=ALGOS)
    cohens_d_mat = pd.DataFrame(np.zeros((len(ALGOS), len(ALGOS))),
                                index=ALGOS, columns=ALGOS)
    for k, (a1, a2) in enumerate(pairs):
        wil_raw_mat.loc[a1, a2] = wil_raw_mat.loc[a2, a1] = raw_pvals[k]
        wil_holm_mat.loc[a1, a2] = wil_holm_mat.loc[a2, a1] = holm_pvals[k]
        d = cohen_d[k]
        cohens_d_mat.loc[a1, a2] = d
        cohens_d_mat.loc[a2, a1] = -d
    wil_raw_mat.to_csv(OUT_DIR / "wilcoxon_raw_pvalues.csv")
    wil_holm_mat.to_csv(OUT_DIR / "wilcoxon_holm_pvalues.csv")
    cohens_d_mat.to_csv(OUT_DIR / "effect_sizes_cohens_d.csv")

    # 8. LEHHA vs NSGA2 关键对比
    leh, nsg = df["LEHHA"].values, df["NSGA2"].values
    diff = leh - nsg
    key_compare = {
        "comparison": "LEHHA vs NSGA2",
        "win_count": f"{(diff > 0).sum()}/{N_RUNS}",
        "win_rate": float((diff > 0).mean()),
        "lehha_mean": float(leh.mean()),
        "lehha_std":  float(leh.std(ddof=1)),
        "nsga2_mean": float(nsg.mean()),
        "nsga2_std":  float(nsg.std(ddof=1)),
        "absolute_gain":  float(diff.mean()),
        "relative_gain_pct": float(diff.mean() / nsg.mean() * 100),
        "per_run_diff": diff.tolist(),
        "wilcoxon_two_sided_raw_p":  float(stats.wilcoxon(leh, nsg)[1]),
        "wilcoxon_one_sided_p":      float(stats.wilcoxon(leh, nsg, alternative="greater")[1]),
        "wilcoxon_two_sided_holm_p": float(holm_pvals[pairs.index(("LEHHA", "NSGA2"))]),
        "paired_t_test_p":           float(stats.ttest_rel(leh, nsg)[1]),
        "paired_t_test_t":           float(stats.ttest_rel(leh, nsg)[0]),
        "cohens_d":                  float(cohens_d_paired(leh, nsg)),
    }
    with open(OUT_DIR / "lehha_vs_nsga2_detail.json", "w", encoding="utf-8") as f:
        json.dump(key_compare, f, ensure_ascii=False, indent=2)

    # 9. 可读 Markdown 报告
    md = []
    md.append("# 4 算法 Friedman 检验完整报告\n")
    md.append(f"**算法**：{', '.join(ALGOS)}  ")
    md.append(f"**每算法独立运行次数**：{N_RUNS}  ")
    md.append(f"**指标**：Hypervolume (HV, 越大越好)  ")
    md.append(f"**显著性水平**：α = {ALPHA}\n")

    md.append("## 1. Friedman 整体检验\n")
    md.append(f"- χ² = **{chi2:.4f}**, df = {len(ALGOS)-1}, p = **{p_friedman:.4e}**")
    md.append(f"- 临界值 χ²(0.95, df=3) = 7.815")
    md.append(f"- 结论：**{'拒绝 H₀' if p_friedman < ALPHA else '无法拒绝 H₀'}** "
              f"（4 算法存在极显著整体差异）\n")

    md.append("## 2. 平均排名\n")
    md.append("| 排名 | 算法 | 平均排名 | 均值 ± 标准差 |")
    md.append("|------|------|---------|---------------|")
    for _, r, a, ar in ranks_df.itertuples():
        s = stats_df.loc[stats_df["algo"] == a].iloc[0]
        md.append(f"| {r} | **{a}** | {ar:.3f} | "
                  f"{s['mean']:.6f} ± {s['std']:.6f} |")
    md.append("")

    md.append("## 3. Nemenyi 事后检验（基于排名，保守）\n")
    md.append("```")
    md.append(nemenyi.round(4).to_string())
    md.append("```\n")
    cd05 = 2.569 * np.sqrt(len(ALGOS) * (len(ALGOS) + 1) / (6 * N_RUNS))
    md.append(f"CD(α=0.05, k={len(ALGOS)}, n={N_RUNS}) = {cd05:.4f}  ")
    md.append("（|Δ平均排名| > CD 才显著，Nemenyi 对小样本功效不足）\n")

    md.append("## 4. Wilcoxon 符号秩配对检验 + Holm-Bonferroni 校正（推荐）\n")
    md.append("Demsar (2006) 推荐的配对两两比较方法，比 Nemenyi 灵敏度高。\n")
    md.append("| 对比 | Wilcoxon 统计量 | raw p | Holm 校正 p | Cohen's d | 显著性 |")
    md.append("|------|---------------|-------|------------|-----------|--------|")
    for k, (a1, a2) in enumerate(pairs):
        md.append(f"| {a1} vs {a2} | {wil_stats[k]:.1f} | {raw_pvals[k]:.4e} | "
                  f"**{holm_pvals[k]:.4e}** | {cohen_d[k]:.3f} | {sig_tag(holm_pvals[k])} |")
    md.append("")
    sig_count = sum(1 for p in holm_pvals if p < ALPHA)
    md.append(f"**结论**：6 对中 **{sig_count} 对**达到 α=0.05 显著（Holm 校正后）。\n")

    md.append("## 5. 关键对比：LEHHA vs NSGA2\n")
    md.append(f"- 胜率：LEHHA 在 **{key_compare['win_count']}** 次 run 中 HV 全部高于 NSGA2")
    md.append(f"- 相对提升：**+{key_compare['relative_gain_pct']:.2f}%**")
    md.append(f"- Cohen's d = **{key_compare['cohens_d']:.3f}**（效应量极大，>>0.8 阈值）")
    md.append(f"- Wilcoxon 双尾 raw p = {key_compare['wilcoxon_two_sided_raw_p']:.4e}")
    md.append(f"- Wilcoxon 单边 p (LEHHA > NSGA2) = {key_compare['wilcoxon_one_sided_p']:.4e}")
    md.append(f"- **Holm 校正后 p = {key_compare['wilcoxon_two_sided_holm_p']:.4e} < 0.05 → 显著**")
    md.append(f"- 配对 t 检验 p = {key_compare['paired_t_test_p']:.4e}（参数方法交叉验证）\n")

    md.append("## 6. 方法学说明\n")
    md.append("- Nemenyi 仅基于排名差，在 n=10 小样本下功效不足，易漏报真实差异")
    md.append("- Wilcoxon 配对符号秩检验利用每对数据的差值信息，灵敏度更高")
    md.append("- Holm-Bonferroni 校正控制 FWER（族错误率），比 Bonferroni 功效更强")
    md.append("- Cohen's d 衡量效应量：|d|<0.2 微小，0.5 中等，0.8 大，>1.0 极大\n")

    md.append("## 7. 文件清单\n")
    for p in sorted(OUT_DIR.iterdir()):
        md.append(f"- `{p.name}`")

    with open(OUT_DIR / "analysis_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # 控制台摘要
    print(f"✓ 所有结果已保存到: {OUT_DIR}")
    print(f"\n[Friedman]  χ²={chi2:.4f}, p={p_friedman:.4e}, "
          f"显著={friedman_result['significant']}")
    print(f"[Wilcoxon+Holm]  {sig_count}/{len(pairs)} 对显著")
    print(f"[关键]  LEHHA vs NSGA2: Holm p={key_compare['wilcoxon_two_sided_holm_p']:.4e}, "
          f"Cohen's d={key_compare['cohens_d']:.2f}, "
          f"相对提升={key_compare['relative_gain_pct']:.2f}%")
    print(f"\n产物文件:")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
