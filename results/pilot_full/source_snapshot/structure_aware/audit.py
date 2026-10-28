"""Check coverage, raw samples, source/input provenance, and derived summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from .report import summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder: Path, check_source=True) -> dict:
    config = json.loads((folder / "config.json").read_text())
    status = json.loads((folder / "status.json").read_text())
    records = [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
    assert status["status"] == "complete"
    assert status["raw_sha256"] == digest(folder / "raw.jsonl")
    assert status["config_sha256"] == digest(folder / "config.json")
    comp = config["composition"]
    expected = len(config["seeds"]) * (2 * len(config["dynamic"]) + 2 * len(config["logical"])
               + 5 * len(comp["cohorts"]) * len(comp["changed_fractions"]) + 3 * len(config["video"]))
    assert len(records) == expected == status["records"], (len(records), expected)
    identities = {(r["domain"], r["case"], r["seed"], r["mode"]) for r in records}
    assert len(identities) == expected
    expected_identities = set()
    for domain in ("dynamic", "logical", "video"):
        modes = ("reference", "uniform", "optimized") if domain == "video" else ("reference", "optimized")
        for case in config[domain]:
            for seed in config["seeds"]:
                expected_identities.update((domain, case["name"], seed, mode) for mode in modes)
    for cohorts in comp["cohorts"]:
        for fraction in comp["changed_fractions"]:
            for seed in config["seeds"]:
                expected_identities.update(("composition", f"cohorts_{cohorts}_changed_{fraction:g}", seed, mode)
                                          for mode in ("reference", "scope", "merge", "combined", "optimized"))
    assert identities == expected_identities
    for r in records:
        assert r["simulation"] and r["validation"]["passed"]
        assert r["repeats"] == config["repeats"] and r["warmups"] == config["warmups"]
        for key, value in r["validation"].items():
            if key.startswith("max_"):
                assert value <= 1e-11
        if r["domain"] == "dynamic":
            assert r["validation"]["distance_mismatches"] == 0
        if r["domain"] == "logical":
            assert len(r["metrics"]["losses"]) == config["training_steps"]
        if r["domain"] == "video":
            assert r["validation"]["latent_rmse"] <= r["validation"]["quality_limit"]
        if r["mode"] == "optimized" and not r["fallback"]:
            assert r["plan"]["estimated_work"] == min(p["estimated_work"] for p in r["candidate_plans"])
        for timing in r["timing"].values():
            assert len(timing["samples"]) == config["repeats"]
            assert all(t >= 0 for t in timing["samples"])
            assert abs(statistics.mean(timing["samples"]) - timing["mean"]) < 1e-9
        for kind in ("arrays", "metadata"):
            assert digest(folder / r["input"][kind]) == r["input"][f"{kind}_sha256"]
    calibrations = json.loads((folder / "calibration.json").read_text())
    for case in config["video"]:
        calibration = calibrations[case["name"]]
        assert set(calibration["calibration_seeds"]).isdisjoint(config["seeds"])
        assert calibration["offline_calibration_excluded_from_online_timing"]
        steps = calibration["config"]["steps"]
        assert len(calibration["adaptive"]) == len(calibration["uniform"]) == steps
        assert calibration["adaptive_calibration_rmse"] <= calibration["margin_limit"]
        assert calibration["uniform_calibration_rmse"] <= calibration["margin_limit"]
    if check_source:
        root = Path(__file__).resolve().parent.parent
        sources = json.loads((folder / "source_manifest.json").read_text())
        for path, signature in sources.items():
            assert digest(root / path) == signature, f"source changed after run: {path}"
    assert json.loads((folder / "summary.json").read_text()) == summarize(records)
    return {"passed": True, "records": expected, "input_seeds": config["seeds"],
            "timed_executions": expected * config["repeats"],
            "source_checked": check_source, "fallback_records": sum(r["fallback"] for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    result = audit(args.folder)
    (args.folder / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
