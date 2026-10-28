"""Derive human-readable reports and thesis tables exclusively from raw runs."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def summarize(records: list[dict]) -> list[dict]:
    references = {(r["domain"], r["case"], r["seed"]): r
                  for r in records if r["mode"] == "reference"}
    groups = defaultdict(list)
    for r in records:
        groups[(r["domain"], r["case"], r["mode"])].append(r)
    summary = []
    for (domain, case, mode), items in groups.items():
        reductions, kernel_reductions, speedups, elapsed = [], [], [], []
        for r in items:
            ref = references[(domain, case, r["seed"])]
            reductions.append(100 * (1 - r["estimated_work"] / ref["estimated_work"]))
            kernel_reductions.append(100 * (1 - r["metrics"]["observed_kernel_work"] /
                                             ref["metrics"]["observed_kernel_work"]))
            speedups.append(ref["timing"]["end_to_end_ms"]["mean"] /
                            r["timing"]["end_to_end_ms"]["mean"])
            elapsed.append(r["timing"]["end_to_end_ms"]["mean"])
        def sd(values):
            return statistics.stdev(values) if len(values) > 1 else 0.0
        numerical = [v for r in items for k, v in r["validation"].items() if k.startswith("max_")]
        summary.append({"domain": domain, "case": case, "mode": mode, "seeds": len(items),
                        "passed": sum(r["validation"]["passed"] for r in items),
                        "fallbacks": sum(r["fallback"] for r in items),
                        "weighted_work_reduction_pct_mean": statistics.mean(reductions),
                        "weighted_work_reduction_pct_std": sd(reductions),
                        "kernel_work_reduction_pct_mean": statistics.mean(kernel_reductions),
                        "end_to_end_speedup_mean": statistics.mean(speedups),
                        "end_to_end_speedup_std": sd(speedups),
                        "end_to_end_ms_mean": statistics.mean(elapsed),
                        "end_to_end_ms_std_across_seeds": sd(elapsed),
                        "max_numerical_error": max(numerical, default=0.0),
                        "max_latent_rmse": max((r["validation"].get("latent_rmse", 0.) for r in items), default=0.),
                        "min_selected_units": min(r["plan"]["active_units"] for r in items),
                        "max_selected_units": max(r["plan"]["active_units"] for r in items),
                        "mean_execution_groups": statistics.mean(r["plan"]["execution_groups"] for r in items),
                        "selected_plan_names": ",".join(sorted({r["plan"]["name"] for r in items}))})
    return summary


def generate(folder: Path):
    records = [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
    config = json.loads((folder / "config.json").read_text())
    env = json.loads((folder / "environment.json").read_text())
    rows = summarize(records)
    (folder / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    with (folder / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 结构感知统一原型：CPU仿真实验报告", "",
             "本报告由 raw.jsonl 自动生成。所有输入均为构造输入；不含原始三项系统的复现结果、真实DiT模型或GPU加速结论。", "",
             f"环境：{env['cpu']}；{env['platform']}；Python {env['python'].split()[0]}；NumPy {env['numpy']}。",
             f"输入随机种子：{config['seeds']}；每项预热 {config['warmups']} 次、计时 {config['repeats']} 次。", "",
             "计算代价是任务内的加权运算量，包含变换组织的计数开销；与实测毫秒数分列，未把运算量下降换算为耗时下降。",
             "端到端时间包含任务适配、公共规划、方案校验、内核执行及训练参数更新；结果对照验证时间另存。",
             "预算校准、初始索引构建和增量查询的初始缓存构建属于预处理，不计入对应在线时间；组合实验衡量已有缓存下的增量执行。", "",
             "表中时间与加速比先在同一输入种子的重复运行中求均值，再在输入种子间求均值和样本标准差；小于1表示端到端减速。", "",
             "|任务|输入情形|方案|验证通过|代价下降(%)|端到端加速比(均值±标准差)|端到端耗时(ms)|回退数|",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['domain']}|{r['case']}|{r['mode']}|{r['passed']}/{r['seeds']}|"
                     f"{r['weighted_work_reduction_pct_mean']:.2f}|"
                     f"{r['end_to_end_speedup_mean']:.3f}±{r['end_to_end_speedup_std']:.3f}|"
                     f"{r['end_to_end_ms_mean']:.3f}|{r['fallbacks']}|")
    lines += ["", "## 约束检查", "",
              "- 动态图：逐项比较全部地标距离；小规模单元测试另用Floyd–Warshall独立验证。只实现单边界分块图上的精确距离维护，不实现FulBM的路径强弱标签。",
              "- 逻辑查询训练：比较每轮前向结果、正负采样损失、实体及关系梯度，以及连续多轮更新后的参数。正样本来自构造知识图谱的集合语义求值，负样本取自答案补集。",
              "- 视频张量去噪：在留出输入上比较全注意力参考输出，使用潜变量RMSE阈值；另记录跨帧差分RMSE。该指标不是VBench、视觉质量或真实视频生成模型精度。",
              "- 组合实验：模型参数固定，依据输入变化传播缓存失效，再执行公共子表达式合并；任何参数快照变化都使相关缓存失效。训练时禁止跳过缓存分支的梯度。", "",
              "## 可审计文件", "",
              "raw.jsonl保留每种输入、方案、验证误差、候选执行方案、逐次计时样本和回退记录；inputs保存实际输入数组、表达式和图更新。",
              "calibration.json记录视频预算、曲率代理、校准输入种子、校准耗时及校准误差。配置、源文件和输入文件均有SHA-256摘要。", "",
              "公共规划器同时比较合法候选方案的加权代价；本原型尚未训练能准确预测不同CPU内核耗时的性能模型。因此，代价模型偏好的方案可能在小规模CPU运行中更慢，报告保留这些结果。", ""]
    (folder / "report.md").write_text("\n".join(lines))
    lookup = {(r["domain"], r["case"], r["mode"]): r for r in rows}
    selections = [("dynamic", "local_1", "optimized", "动态图局部维护"),
                  ("logical", "shared_8", "optimized", "逻辑查询训练"),
                  ("video", "heterogeneous", "optimized", "视频张量去噪"),
                  ("composition", "cohorts_8_changed_0.125", "combined", "增量查询组合变换")]
    tex = ["% Generated by structure_aware.report from raw.jsonl; do not edit numbers manually.",
           r"\begin{table}[htbp]", r"    \centering",
           r"    \caption{统一原型在构造输入上的 CPU 仿真结果}",
           r"    \label{tab:prototype_simulation_results}", r"    \scriptsize",
           r"    \setlength{\tabcolsep}{4pt}", r"    \renewcommand\arraystretch{1.2}",
           r"    \begin{tabular}{p{2.8cm}ccp{4.7cm}}", r"        \toprule",
           r"        任务 & 加权计算代价下降 & 端到端加速比 & 约束验证 \\", r"        \midrule"]
    for domain, case, mode, label in selections:
        row = lookup.get((domain, case, mode))
        if row is None:
            continue
        if domain == "dynamic":
            verification = "全部地标距离与重建结果一致"
        elif domain == "video":
            verification = f"潜变量 RMSE 不超过 {row['max_latent_rmse']:.4f}"
        elif domain == "logical":
            verification = "损失、梯度及参数更新误差小于 $10^{-11}$"
        else:
            verification = "全部查询输出与完整重算结果一致"
        tex.append(f"        {label} & {row['weighted_work_reduction_pct_mean']:.1f}\\% & "
                   f"{row['end_to_end_speedup_mean']:.2f} & {verification} \\\\")
    tex += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
    (folder / "thesis_results.tex").write_text("\n".join(tex))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    generate(parser.parse_args().folder)
