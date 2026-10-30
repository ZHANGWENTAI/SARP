"""Reproducible extension demonstrations: spreadsheet and image-processing DAGs."""
from __future__ import annotations

# Set numerical-library thread limits before importing NumPy.
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import statistics
import sys

import numpy as np

from .adapters import image_pipeline, spreadsheet
from .sarr import SARR, Session, output_digest


MODES = ("reference", "scope", "merge", "combined", "auto")
CASES = (
    {"name": "cold_repeated", "duplicate": True, "changed": 0., "warm_cache": False},
    {"name": "partial_update", "duplicate": True, "changed": .1, "warm_cache": True},
    {"name": "unchanged", "duplicate": True, "changed": 0., "warm_cache": True},
    {"name": "all_changed_unique", "duplicate": False, "changed": 1., "warm_cache": True},
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected):
    if set(actual) != set(expected):
        return {"passed": False, "max_absolute_error": None}
    error = 0.
    for name, value in expected.items():
        a, b = np.asarray(actual[name]), np.asarray(value)
        if a.shape != b.shape or a.dtype != b.dtype:
            return {"passed": False, "max_absolute_error": None}
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return {"passed": False, "max_absolute_error": None}
        error = max(error, float(np.max(np.abs(a.astype(float) - b.astype(float)))))
    return {"passed": error == 0., "max_absolute_error": error}


def summarize(records):
    baseline = {(r["domain"], r["case"], r["seed"]): r for r in records if r["mode"] == "reference"}
    grouped = {}
    for r in records:
        grouped.setdefault((r["domain"], r["case"], r["mode"]), []).append(r)
    result = []
    for (domain, case, mode), rows in sorted(grouped.items()):
        speeds, work_savings, call_savings = [], [], []
        for row in rows:
            ref = baseline[(domain, case, row["seed"])]
            speeds.append(statistics.mean(ref["timing"]["online"]) / statistics.mean(row["timing"]["online"]))
            work_savings.append(1 - row["plan"]["estimated_work"] / ref["plan"]["estimated_work"])
            call_savings.append(1 - row["metrics"]["kernel_calls"] / ref["metrics"]["kernel_calls"])
        result.append({"domain": domain, "case": case, "mode": mode,
                       "speedup_mean": statistics.mean(speeds),
                       "speedup_std": statistics.stdev(speeds) if len(speeds) > 1 else 0.,
                       "work_reduction": statistics.mean(work_savings),
                       "kernel_call_reduction": statistics.mean(call_savings),
                       "kernel_calls": statistics.mean(r["metrics"]["kernel_calls"] for r in rows),
                       "total_units": rows[0]["metrics"]["total_units"],
                       "active_units": statistics.mean(r["metrics"]["active_units"] for r in rows),
                       "execution_groups": statistics.mean(r["metrics"]["executed_groups"] for r in rows),
                       "selected_plans": sorted({r["plan"]["name"] for r in rows}),
                       "online_ms": statistics.mean(statistics.mean(r["timing"]["online"]) for r in rows) * 1000,
                       "max_absolute_error": max(r["validation"]["max_absolute_error"] for r in rows)})
    return result


def write_report(folder, records):
    summary = summarize(records)
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = ["# SARR 任务扩展验证", "",
             "两个适配器均只声明计算图与算子，未修改规划器，也未实现任务专用执行器。",
             "输入复制、内容指纹、计算图构建、方案规划和执行均计入在线耗时；初始缓存构建与参考核验分别记录。",
             "", "|任务|条件|模式|算子调用|调用减少|加权代价减少|加速比（均值±标准差）|最大误差|",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"|{r['domain']}|{r['case']}|{r['mode']}|{r['kernel_calls']:.0f}|"
                     f"{100*r['kernel_call_reduction']:.1f}%|{100*r['work_reduction']:.1f}%|"
                     f"{r['speedup_mean']:.3f}±{r['speedup_std']:.3f}|{r['max_absolute_error']:g}|")
    (folder / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def run(folder: Path, *, seeds=(7, 19, 31, 43, 59), repeats=7, warmups=2,
        regions=12, rows=65536, views=5, images=6, size=192):
    if folder.exists():
        raise ValueError("use a new output directory to preserve existing results")
    folder.mkdir(parents=True)
    (folder / "inputs").mkdir()
    config = {"seeds": list(seeds), "repeats": repeats, "warmups": warmups,
              "regions": regions, "rows": rows, "views": views, "images": images,
              "image_size": size, "cases": CASES, "modes": MODES}
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    root = Path(__file__).resolve().parents[1]
    sources = {}
    for source in sorted((root / "structure_aware").rglob("*.py")):
        rel = source.relative_to(root)
        dest = folder / "source_snapshot" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        sources[str(rel)] = sha(source)
    (folder / "source_manifest.json").write_text(json.dumps(sources, indent=2) + "\n")
    (folder / "environment.json").write_text(json.dumps({
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "processor": platform.processor(), "logical_processors": os.cpu_count(),
        "thread_limits": {k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                                     "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}, indent=2) + "\n")
    records = []
    runtime = SARR()
    for domain, module, adapter in (
            ("spreadsheet", spreadsheet, spreadsheet.SpreadsheetAdapter()),
            ("image_pipeline", image_pipeline, image_pipeline.ImagePipelineAdapter())):
        for case in CASES:
            for seed in seeds:
                kwargs = ({"regions": regions, "rows": rows, "views": views} if domain == "spreadsheet"
                          else {"images": images, "size": size})
                initial = module.construct(seed, duplicate=case["duplicate"], **kwargs)
                request = module.changed(initial, case["changed"])
                inputs = request.ranges if domain == "spreadsheet" else request.images
                base_inputs = initial.ranges if domain == "spreadsheet" else initial.images
                filename = f"{domain}_{seed}_{int(case['duplicate'])}.npz"
                input_path = folder / "inputs" / filename
                if not input_path.exists():
                    np.savez_compressed(input_path, **base_inputs)
                expected = module.expected(request)
                expected_digest = output_digest(expected)
                cache = Session()
                initial_cache_timing = None
                if case["warm_cache"]:
                    initial_run = runtime.run(adapter, initial, session=cache, mode="reference")
                    initial_cache_timing = initial_run.timing
                metadata = {"domain": domain, "case": case, "seed": seed,
                            "base_input_file": f"inputs/{filename}", "base_input_sha256": sha(input_path),
                            "current_input_digest": output_digest(inputs),
                            "reference_output_digest": expected_digest,
                            "initial_cache_timing": initial_cache_timing,
                            "formulas": dict(request.formulas) if domain == "spreadsheet" else None,
                            "outputs": list(expected)}
                metadata_path = folder / "inputs" / f"{domain}_{case['name']}_{seed}.json"
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
                timing = {mode: {k: [] for k in ("prepare", "planning", "execution", "online", "validation")}
                          for mode in MODES}
                last = {}
                order = random.Random(seed)
                for iteration in range(warmups + repeats):
                    modes = list(MODES)
                    order.shuffle(modes)
                    for mode in modes:
                        result = runtime.run(adapter, request, session=cache.clone(), mode=mode,
                                             validator=lambda out: compare(out, expected))
                        if result.validation["fallback"]:
                            raise AssertionError("an exact DAG transformation changed an output")
                        assert output_digest(result.outputs) == expected_digest
                        if iteration >= warmups:
                            for key in timing[mode]:
                                timing[mode][key].append(result.timing[key])
                            last[mode] = result
                for mode in MODES:
                    result = last[mode]
                    record = {"domain": domain, "case": case["name"], "seed": seed, "mode": mode,
                              "plan": result.plan.record(), "candidates": result.candidates,
                              "metrics": result.metrics, "timing": timing[mode],
                              "validation": result.validation["attempts"][-1]["validation"],
                              "fallback": result.validation["fallback"],
                              "output_digest": expected_digest,
                              "input_metadata": str(metadata_path.relative_to(folder)),
                              "input_metadata_sha256": sha(metadata_path)}
                    records.append(record)
                print(f"{domain}: {case['name']} seed={seed} completed", flush=True)
    raw = folder / "raw.jsonl"
    raw.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    summary = write_report(folder, records)
    (folder / "status.json").write_text(json.dumps({
        "status": "complete", "records": len(records), "timed_executions": len(records) * repeats,
        "raw_sha256": sha(raw), "config_sha256": sha(folder / "config.json")}, indent=2) + "\n")
    return summary


def audit(folder: Path):
    config = json.loads((folder / "config.json").read_text())
    records = [json.loads(x) for x in (folder / "raw.jsonl").read_text().splitlines()]
    status = json.loads((folder / "status.json").read_text())
    expected = {(domain, case["name"], seed, mode)
                for domain in ("spreadsheet", "image_pipeline") for case in config["cases"]
                for seed in config["seeds"] for mode in config["modes"]}
    assert {(r["domain"], r["case"], r["seed"], r["mode"]) for r in records} == expected
    assert len(records) == len(expected) == status["records"]
    assert sha(folder / "raw.jsonl") == status["raw_sha256"]
    assert sha(folder / "config.json") == status["config_sha256"]
    for path, signature in json.loads((folder / "source_manifest.json").read_text()).items():
        assert sha(folder / "source_snapshot" / path) == signature
    checked_metadata, checked_inputs = set(), set()
    for record in records:
        assert record["validation"]["passed"] and not record["fallback"]
        assert record["validation"]["max_absolute_error"] == 0
        assert sha(folder / record["input_metadata"]) == record["input_metadata_sha256"]
        metadata = json.loads((folder / record["input_metadata"]).read_text())
        if metadata["base_input_file"] not in checked_inputs:
            assert sha(folder / metadata["base_input_file"]) == metadata["base_input_sha256"]
            checked_inputs.add(metadata["base_input_file"])
        assert record["output_digest"] == metadata["reference_output_digest"]
        if record["input_metadata"] not in checked_metadata:
            with np.load(folder / metadata["base_input_file"], allow_pickle=False) as archive:
                inputs = {name: archive[name] for name in archive.files}
            if record["domain"] == "spreadsheet":
                initial = spreadsheet.Workbook(inputs, metadata["formulas"], tuple(metadata["outputs"]))
                request = spreadsheet.changed(initial, metadata["case"]["changed"])
                expected_outputs = spreadsheet.expected(request)
                request_inputs = request.ranges
            else:
                branches = ("edges", "mask", "preview") if metadata["case"]["duplicate"] else ("edges",)
                request = image_pipeline.changed(image_pipeline.ImageRequest(inputs, branches),
                                                 metadata["case"]["changed"])
                expected_outputs = image_pipeline.expected(request)
                request_inputs = request.images
            assert output_digest(request_inputs) == metadata["current_input_digest"]
            assert output_digest(expected_outputs) == metadata["reference_output_digest"]
            checked_metadata.add(record["input_metadata"])
        for samples in record["timing"].values():
            assert len(samples) == config["repeats"] and all(x >= 0 for x in samples)
        if record["mode"] == "auto":
            assert record["plan"]["estimated_work"] == min(x["estimated_work"] for x in record["candidates"])
    assert json.loads((folder / "summary.json").read_text()) == summarize(records)
    result = {"passed": True, "records": len(records),
              "timed_executions": len(records) * config["repeats"],
              "max_absolute_error": 0., "fallbacks": 0}
    (folder / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit(args.audit)))
        return
    if not args.output:
        parser.error("--output or --audit is required")
    kwargs = {"seeds": (7,), "repeats": 2, "warmups": 1, "regions": 6,
              "rows": 8192, "views": 3, "images": 3, "size": 64} if args.quick else {}
    run(args.output, **kwargs)
    print(json.dumps(audit(args.output)))


if __name__ == "__main__":
    main()
