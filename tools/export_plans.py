"""Export small inspectable graphs and actual candidate plans for all adapters."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from structure_aware.core import Planner
from structure_aware.adapters import dynamic, logical, video


def export(task, filename):
    planner = Planner()
    plans = planner.candidates(task)
    selected, _ = planner.select(task)
    record = {"task": task.name, "domain": task.domain, "cyclic": task.structure.cyclic,
              "units": [asdict(u) for u in task.structure.units],
              "range_certificate": task.range_certificate,
              "required": sorted(task.required) if task.required is not None else None,
              "constraint": task.constraint_name, "selected": selected.name,
              "candidate_plans": [{**p.record(), "active": list(p.active),
                                   "groups": list(p.groups)} for p in plans]}
    path = ROOT / "examples" / filename
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


def main():
    da = dynamic.DynamicAdapter()
    export(da.prepare(dynamic.construct(7, blocks=8, block_size=8, touched=1)), "dynamic_plan.json")
    la = logical.LogicalAdapter()
    data = logical.construct(7, queries=12, dimension=8, cohorts=2)
    export(la.prepare(data), "logical_training_plan.json")
    task = la.prepare(data, training=False)
    cache = la.execute(task, Planner().make(task)).values["cache"]
    export(la.prepare(logical.change_inputs(data, .5), training=False, cache=cache),
           "composition_plan.json")
    va = video.VideoAdapter()
    cfg = video.VideoConfig()
    calibration = video.calibrate(cfg)
    export(va.prepare(video.construct(7, cfg), tuple(calibration["adaptive"]),
                      tuple(calibration["signal"])), "video_plan.json")
    print("Exported four graphs and their candidate execution plans to examples/.")


if __name__ == "__main__":
    main()
