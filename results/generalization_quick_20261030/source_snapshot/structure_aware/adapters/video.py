"""Low-dimensional conditional video-tensor denoising with block attention.

No DiT checkpoint, learned video generator, CUDA kernel, or VBench metric is used.
Full attention is the numerical reference; quality means latent RMSE against that
reference. Offline calibration and held-out generation inputs are disjoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..core import (BudgetDomain, Outcome, Plan, Planner, Structure, Task, Unit,
                    allocate_budget)


@dataclass(frozen=True)
class VideoConfig:
    frames: int = 4
    height: int = 4
    width: int = 4
    channels: int = 12
    steps: int = 12
    block_size: int = 4
    quality_limit: float = 0.004
    heterogeneous: bool = True


@dataclass
class VideoInput:
    clean: np.ndarray
    initial: np.ndarray
    weights: tuple[np.ndarray, ...]
    gains: np.ndarray
    config: VideoConfig
    seed: int


def construct(seed=0, config=VideoConfig()) -> VideoInput:
    n = config.frames * config.height * config.width
    if n % config.block_size or config.steps < 4:
        raise ValueError("invalid attention geometry")
    rng = np.random.default_rng(seed)
    f, y, x, c = np.indices((config.frames, config.height, config.width, config.channels))
    clean = (0.5 * np.sin(0.7 * x + 0.5 * y + 0.4 * f + c * 0.23)
             + 0.25 * np.cos(0.8 * x - 0.2 * y - 0.5 * f + c * 0.11))
    clean = clean.reshape(n, config.channels)
    initial = clean + rng.normal(0, 0.55, clean.shape)
    model_rng = np.random.default_rng(20261028)
    weights = tuple(model_rng.normal(0, 1 / np.sqrt(config.channels),
                                     (config.channels, config.channels)) for _ in range(4))
    time = np.linspace(0, 1, config.steps)
    gains = (0.06 + 2.2 * np.exp(-((time - 0.32) / 0.105) ** 2)
             if config.heterogeneous else np.full(config.steps, 0.5))
    return VideoInput(clean, initial, weights, gains, config, seed)


def attention(x: np.ndarray, weights: tuple[np.ndarray, ...], exact_blocks: int,
              block_size: int) -> tuple[np.ndarray, int]:
    wq, wk, wv, wo = weights
    q, k, v = x @ wq, x @ wk, x @ wv
    n, d = x.shape
    blocks = n // block_size
    kb, vb = k.reshape(blocks, block_size, d), v.reshape(blocks, block_size, d)
    km, vm = kb.mean(axis=1), vb.mean(axis=1)
    # Simple deterministic block router; the prototype studies per-step budget.
    variance = np.sum((kb - km[:, None, :]) ** 2, axis=(1, 2))
    order = np.argsort(-variance, kind="stable")
    selected, approximate = order[:exact_blocks], order[exact_blocks:]
    keys = np.concatenate((kb[selected].reshape(-1, d), km[approximate]), axis=0)
    vals = np.concatenate((vb[selected].reshape(-1, d), vm[approximate]), axis=0)
    logits = q @ keys.T / np.sqrt(d)
    # Each centroid represents block_size original keys.
    if len(approximate):
        logits[:, exact_blocks * block_size:] += np.log(block_size)
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    return (probability @ vals) @ wo, n * len(keys)


class VideoAdapter:
    domain = "video"

    def prepare(self, data: VideoInput, budget: tuple[int, ...] | None = None,
                weights: tuple[float, ...] | None = None) -> Task:
        cfg = data.config
        n, d = data.initial.shape
        blocks = n // cfg.block_size
        base = float(8 * n * d * d + 4 * n * blocks * d + 6 * n * d)
        increment = float(4 * n * (cfg.block_size - 1) * d)
        domain = BudgetDomain(0, blocks, weights or (1.0,) * cfg.steps,
                              (base,) * cfg.steps, (increment,) * cfg.steps)
        units = tuple(Unit(t, "denoising_step", () if t == 0 else (t - 1,),
                           work=domain.work(t, blocks)) for t in range(cfg.steps))
        return Task("conditional_video_tensor_denoising", self.domain, Structure(units),
                    budget=domain, permitted_budget=budget,
                    constraint_name="held_out_latent_rmse",
                    payload=data)

    def execute(self, task: Task, plan: Plan) -> Outcome:
        data = task.payload
        x = data.initial.copy()
        velocities, states = [], []
        interactions = 0
        for t, b in enumerate(plan.budgets):
            attended, pairs = attention(x, data.weights, b, data.config.block_size)
            velocity = data.gains[t] * (x - data.clean + 0.65 * np.tanh(attended))
            x = x - 0.12 * velocity
            velocities.append(velocity)
            states.append(x.copy())
            interactions += pairs
        return Outcome({"latent": x, "trajectory": np.stack(states),
                        "velocity": np.stack(velocities)},
                       {"exact_blocks": sum(plan.budgets),
                        "attention_interactions": interactions,
                        "observed_kernel_work": (data.config.steps *
                            (8 * x.shape[0] * x.shape[1] ** 2 + 6 * x.shape[0] * x.shape[1])
                            + 4 * x.shape[1] * interactions)})

    @staticmethod
    def validate(result: Outcome, reference: Outcome, data: VideoInput) -> dict:
        difference = result.values["latent"] - reference.values["latent"]
        rmse = float(np.sqrt(np.mean(difference ** 2)))
        cfg = data.config
        shape = (cfg.frames, cfg.height, cfg.width, cfg.channels)
        temporal = np.diff(result.values["latent"].reshape(shape), axis=0)
        reference_temporal = np.diff(reference.values["latent"].reshape(shape), axis=0)
        temporal_rmse = float(np.sqrt(np.mean((temporal - reference_temporal) ** 2)))
        return {"passed": bool(np.isfinite(rmse) and rmse <= cfg.quality_limit),
                "latent_rmse": rmse, "temporal_difference_rmse": temporal_rmse,
                "quality_limit": cfg.quality_limit,
                "clean_target_rmse": float(np.sqrt(np.mean(
                    (result.values["latent"] - data.clean) ** 2)))}


def calibrate(config=VideoConfig(), seeds=(101, 102, 103)) -> dict:
    """Offline policy choice, checked against full-attention calibration outputs.

    The trajectory proxy merely proposes allocations. Actual calibration RMSE
    decides acceptance; held-out inputs are never consulted during this function.
    """
    backend, planner = VideoAdapter(), Planner()
    datasets = [construct(seed, config) for seed in seeds]
    tasks = [backend.prepare(d) for d in datasets]
    references = [backend.execute(t, planner.make(t)) for t in tasks]
    signals = []
    for result in references:
        velocities = result.values["velocity"]
        curvature = np.linalg.norm(np.diff(velocities, axis=0).reshape(config.steps - 1, -1), axis=1)
        curvature = np.concatenate(([curvature[0]], curvature))
        speed = np.linalg.norm(velocities.reshape(config.steps, -1), axis=1)
        signals.append(curvature + 0.25 * speed + 1e-9)
    signal = np.mean(signals, axis=0)
    signal /= signal.sum()
    domain = backend.prepare(datasets[0], weights=tuple(map(float, signal))).budget
    margin_limit = config.quality_limit * 0.7
    evaluations = 0
    def check(levels):
        nonlocal evaluations
        errors = []
        for data, task, reference in zip(datasets, tasks, references):
            result = backend.execute(task, planner.make(task, budgets=levels))
            evaluations += 1
            errors.append(backend.validate(result, reference, data)["latent_rmse"])
            if errors[-1] > margin_limit:
                return False, max(errors)
        return True, max(errors)
    max_blocks = domain.maximum
    uniform, uniform_error = None, None
    for b in range(max_blocks + 1):
        candidate = (b,) * config.steps
        ok, error = check(candidate)
        if ok:
            uniform, uniform_error = candidate, error
            break
    adaptive, adaptive_error = None, None
    for total in range(config.steps * max_blocks + 1):
        candidate = allocate_budget(domain, total)
        ok, error = check(candidate)
        if ok:
            adaptive, adaptive_error = candidate, error
            break
    if uniform is None or adaptive is None:
        raise RuntimeError("full attention must pass calibration")
    return {"calibration_seeds": list(seeds), "margin_limit": margin_limit,
            "signal": list(map(float, signal)), "uniform": list(uniform),
            "adaptive": list(adaptive), "uniform_calibration_rmse": uniform_error,
            "adaptive_calibration_rmse": adaptive_error,
            "calibration_evaluations": evaluations,
            "offline_calibration_excluded_from_online_timing": True}
