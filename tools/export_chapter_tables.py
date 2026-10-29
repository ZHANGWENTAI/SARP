"""Export Chapter 6 tables from the existing audited simulation records."""
from pathlib import Path
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "full"


def write_table(filename, caption, label, header, rows, columns):
    lines = ["% Generated from audited results/full records; no experiments are rerun.",
             r"\begin{table}[htbp]", r"    \centering", r"    \scriptsize",
             r"    \setlength{\tabcolsep}{4pt}", r"    \renewcommand\arraystretch{1.2}",
             f"    \\caption{{{caption}}}", f"    \\label{{{label}}}",
             f"    \\begin{{tabular}}{{{columns}}}", r"        \toprule",
             "        " + " & ".join(header) + r" \\", r"        \midrule"]
    lines += ["        " + " & ".join(row) + r" \\" for row in rows]
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
    (RESULTS / filename).write_text("\n".join(lines))


def main():
    summary = json.loads((RESULTS / "summary.json").read_text())
    lookup = {(r["domain"], r["case"], r["mode"]): r for r in summary}
    def reduction(row):
        return f"{row['weighted_work_reduction_pct_mean']:.1f}\\%"
    def speed(row):
        return f"${row['end_to_end_speedup_mean']:.2f}\\pm{row['end_to_end_speedup_std']:.2f}$"
    rows = []
    for domain, case, label in [
        ("dynamic", "local_1", "1个局部块更新"),
        ("dynamic", "local_8", "8个局部块更新"),
        ("dynamic", "global_32", "全部32个局部块更新"),
        ("dynamic", "cross_block", "跨块更新"),
        ("logical", "shared_8", "8组公共前缀"),
        ("logical", "shared_64", "64组公共前缀"),
        ("logical", "shared_384", "384组公共前缀")]:
        row = lookup[(domain, case, "optimized")]
        rows.append(["动态图维护" if domain == "dynamic" else "逻辑查询训练", label,
                     reduction(row), speed(row),
                     {"scope": "范围收缩", "merge": "节点合并", "reference": "完整执行"}[row["selected_plan_names"]]])
    write_table("chapter_structure_effects.tex", "拓扑局部性与逻辑复用程度对原型收益的影响",
                "tab:prototype_structure_effects", ["任务", "结构条件", "计算代价下降", "端到端加速比", "选中方案"], rows,
                "p{2.0cm}p{3.4cm}ccc")
    calibration = json.loads((RESULTS / "calibration.json").read_text())
    rows = []
    for case, label in [("heterogeneous", "异质阶段"), ("longer", "较长序列"),
                        ("homogeneous", "同质阶段"), ("strict", "严格误差要求")]:
        row = lookup[("video", case, "optimized")]
        cal = calibration[case]
        cfg = cal["config"]
        full = cfg["steps"] * (cfg["frames"] * cfg["height"] * cfg["width"] // cfg["block_size"])
        rows.append([label, str(full), str(sum(cal["uniform"])), str(sum(cal["adaptive"])),
                     reduction(row), speed(row), f"\\num{{{row['max_latent_rmse']:.4g}}}"])
    write_table("chapter_video_budget.tex", "不同去噪阶段条件下的预算与留出输入验证结果",
                "tab:prototype_video_budget", ["情形", "完整预算", "固定预算", "阶段预算", "代价下降", "加速比", "最大 RMSE"],
                rows, "p{2.1cm}cccccc")
    raw = [json.loads(line) for line in (RESULTS / "raw.jsonl").read_text().splitlines()]
    rows = []
    case = "cohorts_8_changed_0.125"
    for mode, label in [("reference", "完整重算"), ("scope", "仅范围收缩"), ("merge", "仅节点合并"),
                        ("combined", "组合执行"), ("optimized", "自动选案")]:
        row = lookup[("composition", case, mode)]
        selected = [r for r in raw if (r["domain"], r["case"], r["mode"]) == ("composition", case, mode)]
        elapsed = [f"{statistics.mean(r['timing'][key]['mean'] for r in selected):.3f}"
                   for key in ["prepare_ms", "planning_ms", "execution_ms", "end_to_end_ms"]]
        rows.append([label, reduction(row), *elapsed, speed(row)])
    write_table("chapter_composition.tex", "范围收缩与节点合并的组合验证及耗时分解",
                "tab:prototype_composition", ["执行方案", "代价下降", "准备/ms", "规划/ms", "执行/ms", "总耗时/ms", "加速比"],
                rows, "p{2.0cm}cccccc")
    print("Exported three Chapter 6 tables from existing raw results.")


if __name__ == "__main__":
    main()
