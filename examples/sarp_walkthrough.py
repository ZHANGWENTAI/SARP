"""Run both new task adapters and print inspectable execution arrangements."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structure_aware.sarp import SARP, Session, output_digest
from structure_aware.adapters import spreadsheet, image_pipeline
from structure_aware.generalization import compare


def demonstrate(runtime, name, adapter, initial, updated, expected):
    cache = Session()
    runtime.run(adapter, initial, session=cache, mode="reference")
    prepared = runtime.prepare(adapter, updated, cache)
    payload = prepared.task.payload
    records = []
    for mode in ("reference", "scope", "merge", "combined", "auto"):
        result = runtime.run(adapter, updated, session=cache.clone(), mode=mode,
                             validator=lambda out: compare(out, expected))
        records.append({"mode": mode, "chosen_plan": result.plan.name,
                        "kernel_calls": result.metrics["kernel_calls"],
                        "active_units": len(result.plan.active),
                        "execution_groups": len(result.plan.groups),
                        "estimated_work": result.plan.estimated_work,
                        "passed": result.validation["passed"],
                        "output_digest": output_digest(result.outputs),
                        "groups": [[payload.names[i] for i in group] for group in result.plan.groups]})
    graph = {"task": name, "planner": type(runtime.runtime.planner).__name__,
             "executor": type(runtime.runtime.backends[prepared.task.domain]).__name__,
             "adapter_implements_execution": hasattr(adapter, "execute"),
             "inputs": {k: {"shape": list(getattr(v, "shape", ())),
                             "dtype": str(getattr(v, "dtype", type(v).__name__))}
                        for k, v in payload.inputs.items()},
             "outputs": list(payload.outputs),
             "nodes": [{"name": node.name, "operation": node.operation,
                        "dependencies": list(node.dependencies),
                        "parameters": dict(node.parameters), "work": node.work}
                       for node in payload.nodes.values()],
             "required": [payload.names[i] for i in sorted(prepared.task.required)],
             "plans": records}
    filename = ROOT / "examples" / f"generalization_{name}.json"
    filename.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n")
    print(f"\n{name}: mode | actual kernel calls | selected plan | correct")
    for row in records:
        print(f"{row['mode']:10} | {row['kernel_calls']:3} | {row['chosen_plan']:12} | {row['passed']}")
    print("Graph and plans:", filename)


def main():
    runtime = SARP()
    book = spreadsheet.construct(regions=3, rows=1024, views=3)
    updated = spreadsheet.changed(book, 1 / 3)
    demonstrate(runtime, "spreadsheet", spreadsheet.SpreadsheetAdapter(), book,
                updated, spreadsheet.expected(updated))
    images = image_pipeline.construct(images=3, size=32)
    updated = image_pipeline.changed(images, 1 / 3)
    demonstrate(runtime, "image_pipeline", image_pipeline.ImagePipelineAdapter(), images,
                updated, image_pipeline.expected(updated))


if __name__ == "__main__":
    main()
