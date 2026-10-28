"""Reproducible data construction, shared execution harness, raw performance logs."""
from __future__ import annotations

import os
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Callable
import numpy as np

from .core import Runtime
from .adapters import dynamic, logical, video


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_input(folder: Path, stem: str, arrays: dict, metadata: dict) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    npz = folder / f"{stem}.npz"
    meta = folder / f"{stem}.json"
    np.savez_compressed(npz, **arrays)
    dump(meta, metadata)
    return {"arrays": f"inputs/{npz.name}", "metadata": f"inputs/{meta.name}",
            "arrays_sha256": file_hash(npz), "metadata_sha256": file_hash(meta)}


def environment() -> dict:
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        cpu = platform.processor()
    return {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
            "cpu": cpu, "logical_cpu_count": os.cpu_count(),
            "timer": "time.perf_counter_ns", "execution_device": "CPU",
            "thread_limits": {k: os.environ.get(k) for k in
                              ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
            "simulation": True, "original_FulBM_FlashEKGR_PASA_reproduction": False}


def choose(runtime, task, mode, levels=None):
    planner = runtime.planner
    if mode == "reference":
        return planner.make(task)
    if mode == "optimized":
        return planner.select(task)[0]
    if mode == "scope":
        return planner.make(task, scope=True)
    if mode == "merge":
        return planner.make(task, merge=True)
    if mode == "combined":
        return planner.make(task, scope=True, merge=True)
    if mode == "uniform":
        return planner.make(task, budgets=levels, name="uniform")
    raise ValueError(mode)


def one_trial(runtime: Runtime, backend, prepare: Callable, validator: Callable,
              mode: str, levels=None) -> dict:
    start = time.perf_counter_ns()
    task = prepare()
    prepared = time.perf_counter_ns()
    if mode == "optimized":
        plan, candidate_plans = runtime.planner.select(task)
    else:
        plan = choose(runtime, task, mode, levels)
        candidate_plans = [plan.record()]
    planned = time.perf_counter_ns()
    outcome, plan = runtime.execute(task, plan)
    executed = time.perf_counter_ns()
    check = validator(outcome)
    checked = time.perf_counter_ns()
    initial_check = check
    attempts = [plan.record()]
    attempted_kernel_work = outcome.metrics["observed_kernel_work"]
    preparation_ms = (prepared - start) / 1e6
    planning_ms = (planned - prepared) / 1e6
    execution_ms = (executed - planned) / 1e6
    validation_ms = (checked - executed) / 1e6
    if not check["passed"]:
        # The simulation has an oracle. Charge actual fallback execution, but
        # separately record oracle-based validation time.
        before = time.perf_counter_ns()
        plan = runtime.planner.make(task)
        after_plan = time.perf_counter_ns()
        outcome, plan = runtime.execute(task, plan)
        attempted_kernel_work += outcome.metrics["observed_kernel_work"]
        after_run = time.perf_counter_ns()
        check = validator(outcome)
        after_check = time.perf_counter_ns()
        planning_ms += (after_plan - before) / 1e6
        execution_ms += (after_run - after_plan) / 1e6
        validation_ms += (after_check - after_run) / 1e6
        attempts.append(plan.record())
    if not check["passed"]:
        raise AssertionError(f"{task.name}: reference validation failed: {check}")
    estimated_work = sum(p["estimated_work"] for p in attempts)
    return {"task": task, "outcome": outcome, "plan": plan.record(),
            "candidate_plans": candidate_plans,
            "attempted_plans": attempts, "estimated_work": estimated_work,
            "validation": check, "initial_validation": initial_check,
            "fallback": len(attempts) > 1,
            "timing": {"prepare_ms": preparation_ms, "planning_ms": planning_ms,
                       "execution_ms": execution_ms,
                       "end_to_end_ms": preparation_ms + planning_ms + execution_ms,
                       "validation_ms": validation_ms},
            "metrics": {**outcome.metrics, "observed_kernel_work": attempted_kernel_work}}


def measure(trial: Callable, warmups: int, repeats: int) -> tuple[dict, dict]:
    for _ in range(warmups):
        trial()
    trials = [trial() for _ in range(repeats)]
    sample = trials[0]
    timing = {}
    for name in sample["timing"]:
        values = [t["timing"][name] for t in trials]
        timing[name] = {"mean": statistics.mean(values), "std": statistics.stdev(values) if repeats > 1 else 0,
                        "median": statistics.median(values), "samples": values}
    if any(t["plan"] != sample["plan"] or t["validation"] != sample["validation"] for t in trials):
        raise AssertionError("deterministic input produced inconsistent plans or results")
    record = {k: sample[k] for k in ("plan", "candidate_plans", "attempted_plans", "estimated_work", "validation",
                                      "initial_validation", "fallback", "metrics")}
    record.update(timing=timing, repeats=repeats, warmups=warmups,
                  structural_units=len(sample["task"].structure.units))
    return record, sample


def training_trial(runtime, backend, data, mode, reference_steps, reference_final):
    accumulated = {k: 0.0 for k in ("prepare_ms", "planning_ms", "execution_ms", "end_to_end_ms", "validation_ms")}
    losses, errors = [], {}
    total_work = total_ops = total_projections = 0
    current = data
    last = None
    for reference in reference_steps:
        trial = one_trial(runtime, backend, lambda: backend.prepare(current),
                          lambda out: backend.validate(out, reference), mode)
        start = time.perf_counter_ns()
        current = logical.step(current, trial["outcome"])
        update_ms = (time.perf_counter_ns() - start) / 1e6
        trial["timing"]["execution_ms"] += update_ms
        trial["timing"]["end_to_end_ms"] += update_ms
        for key, value in trial["timing"].items():
            accumulated[key] += value
        for key, value in trial["validation"].items():
            if key.startswith("max_"):
                errors[key] = max(errors.get(key, 0), value)
        total_work += trial["estimated_work"]
        total_ops += trial["metrics"]["operator_executions"]
        total_projections += trial["metrics"]["project_executions"]
        losses.append(trial["outcome"].values["loss"])
        last = trial
    errors["max_entity_update_error"] = float(np.max(np.abs(current.entities - reference_final.entities)))
    errors["max_relation_update_error"] = float(np.max(np.abs(current.relations - reference_final.relations)))
    last.update(timing=accumulated, estimated_work=total_work,
                validation={"passed": all(e <= 1e-11 for e in errors.values()), **errors},
                metrics={**last["metrics"], "losses": losses, "training_steps": len(losses),
                         "operator_executions": total_ops, "project_executions": total_projections,
                         "observed_kernel_work": last["plan"]["kernel_work"] * len(losses)})
    if not last["validation"]["passed"]:
        raise AssertionError("multi-step training diverged from reference")
    return last


def run(config: dict, output: Path):
    if config["repeats"] < 2 or config["warmups"] < 0:
        raise ValueError("use at least two timing repeats")
    if set(config["seeds"]) & set(config["calibration_seeds"]):
        raise ValueError("calibration and held-out seeds overlap")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "raw.jsonl").exists():
        raise FileExistsError("output already contains results; choose a new directory")
    dump(output / "config.json", config)
    dump(output / "environment.json", environment())
    source_root = Path(__file__).resolve().parent.parent
    source_files = {str(path.relative_to(source_root)): file_hash(path)
                    for path in sorted((source_root / "structure_aware").rglob("*.py"))}
    dump(output / "source_manifest.json", source_files)
    status = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "records": 0, "config_sha256": file_hash(output / "config.json")}
    dump(output / "status.json", status)
    runtime = Runtime()
    backends = [dynamic.DynamicAdapter(), logical.LogicalAdapter(), video.VideoAdapter()]
    for backend in backends:
        runtime.register(backend)
    records = []
    stream = (output / "raw.jsonl").open("w")
    def emit(domain, case, seed, mode, trial, input_record):
        record, sample = measure(trial, config["warmups"], config["repeats"])
        record.update(domain=domain, case=case, seed=seed, mode=mode,
                      input=input_record, simulation=True)
        records.append(record)
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        status["records"] = len(records)
        dump(output / "status.json", status)
        print(f"[{len(records):03d}] {domain}/{case}/seed={seed}/{mode}: "
              f"{record['timing']['end_to_end_ms']['mean']:.3f} ms; "
              f"validation={record['validation']['passed']}; fallback={record['fallback']}", flush=True)
        return sample
    for item in config["dynamic"]:
        backend = backends[0]
        case = item["name"]
        for seed in config["seeds"]:
            data = dynamic.construct(seed, **{k: v for k, v in item.items() if k != "name"})
            edges = np.array([(u, v) for u, ns in enumerate(data.adjacency) for v in sorted(ns) if u < v])
            inp = save_input(output / "inputs", f"dynamic_{case}_{seed}",
                             {"edges": edges, "old_distances": data.old_distances,
                              "landmarks": data.landmarks, "block_of": data.block_of},
                             {"seed": seed, "case": item, "updates": data.updates,
                              "roots": data.roots})
            task = backend.prepare(data)
            reference = backend.execute(task, runtime.planner.make(task))
            np.savez_compressed(output / "inputs" / f"dynamic_{case}_{seed}_reference.npz",
                                distances=reference.values)
            for mode in ordered_modes(("reference", "optimized"), seed):
                emit("dynamic", case, seed, mode,
                     lambda mode=mode: one_trial(runtime, backend, lambda: backend.prepare(data),
                                                  lambda out: backend.validate(out, reference), mode), inp)
    for item in config["logical"]:
        backend = backends[1]
        case = item["name"]
        for seed in config["seeds"]:
            data = logical.construct(seed, **{k: v for k, v in item.items() if k != "name"})
            inp = save_input(output / "inputs", f"logical_{case}_{seed}",
                             {"entities": data.entities, "relations": data.relations,
                              "triples": np.array(data.triples), "positives": data.positives,
                              "negatives": data.negatives},
                             {"seed": seed, "case": item, "expressions": data.expressions})
            reference_steps = []
            current = data
            for _ in range(config["training_steps"]):
                task = backend.prepare(current)
                reference = backend.execute(task, runtime.planner.make(task))
                reference_steps.append(reference)
                current = logical.step(current, reference)
            np.savez_compressed(output / "inputs" / f"logical_{case}_{seed}_reference.npz",
                                entities=current.entities, relations=current.relations,
                                losses=[r.values["loss"] for r in reference_steps])
            for mode in ordered_modes(("reference", "optimized"), seed):
                emit("logical", case, seed, mode,
                     lambda mode=mode: training_trial(runtime, backend, data, mode, reference_steps, current), inp)
    comp = config["composition"]
    for cohorts in comp["cohorts"]:
        for fraction in comp["changed_fractions"]:
            case = f"cohorts_{cohorts}_changed_{fraction:g}"
            backend = backends[1]
            for seed in config["seeds"]:
                data = logical.construct(seed, queries=comp["queries"], dimension=comp["dimension"], cohorts=cohorts)
                initial_task = backend.prepare(data, training=False)
                initial = backend.execute(initial_task, runtime.planner.make(initial_task))
                changed = logical.change_inputs(data, fraction)
                cache = initial.values["cache"]
                inp = save_input(output / "inputs", f"composition_{case}_{seed}",
                                 {"entities": data.entities, "relations": data.relations,
                                  "positives": data.positives, "negatives": data.negatives},
                                 {"seed": seed, "cohorts": cohorts, "changed_fraction": fraction,
                                  "old_expressions": data.expressions, "new_expressions": changed.expressions,
                                  "cache_setup_excluded_from_warm_incremental_timing": True,
                                  "initial_cache_work": runtime.planner.make(initial_task).estimated_work})
                task = backend.prepare(changed, training=False, cache=cache)
                reference = backend.execute(task, runtime.planner.make(task))
                for mode in ordered_modes(("reference", "scope", "merge", "combined", "optimized"), seed):
                    emit("composition", case, seed, mode,
                         lambda mode=mode: one_trial(runtime, backend,
                            lambda: backend.prepare(changed, training=False, cache=cache),
                            lambda out: backend.validate(out, reference), mode), inp)
    calibrations = {}
    for item in config["video"]:
        backend = backends[2]
        case = item["name"]
        cfg = video.VideoConfig(**{k: v for k, v in item.items() if k != "name"})
        start = time.perf_counter_ns()
        calibration = video.calibrate(cfg, tuple(config["calibration_seeds"]))
        calibration["offline_wall_ms"] = (time.perf_counter_ns() - start) / 1e6
        calibration["config"] = asdict(cfg)
        calibrations[case] = calibration
        dump(output / "calibration.json", calibrations)
        for seed in config["seeds"]:
            data = video.construct(seed, cfg)
            inp = save_input(output / "inputs", f"video_{case}_{seed}",
                             {"clean": data.clean, "initial": data.initial,
                              "weights": np.stack(data.weights), "gains": data.gains},
                             {"seed": seed, "config": asdict(cfg), "calibration_case": case})
            prepare = lambda: backend.prepare(data, tuple(calibration["adaptive"]), tuple(calibration["signal"]))
            task = prepare()
            reference = backend.execute(task, runtime.planner.make(task))
            for mode in ordered_modes(("reference", "uniform", "optimized"), seed):
                sample = emit("video", case, seed, mode,
                     lambda mode=mode: one_trial(runtime, backend, prepare,
                         lambda out: backend.validate(out, reference, data), mode,
                         tuple(calibration["uniform"])), inp)
                if seed == config["seeds"][0]:
                    np.savez_compressed(output / "inputs" / f"video_{case}_{seed}_{mode}_output.npz",
                                        latent=sample["outcome"].values["latent"],
                                        trajectory=sample["outcome"].values["trajectory"])
    stream.close()
    status.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                  all_validated=all(r["validation"]["passed"] for r in records),
                  raw_sha256=file_hash(output / "raw.jsonl"))
    dump(output / "status.json", status)
    from .report import generate
    generate(output)
    print(f"Complete: {len(records)} records, report={output / 'report.md'}", flush=True)


def ordered_modes(modes, seed):
    """Deterministic shuffled order reduces systematic reference-first bias."""
    return list(np.random.default_rng(seed + 83017).permutation(modes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/full.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
