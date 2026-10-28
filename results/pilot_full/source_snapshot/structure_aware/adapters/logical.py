"""Synthetic KG query training with exact common-subexpression autodiff.

The small differentiable model implements projection, intersection, union and
negation. It is an executable simulation of FlashEKGR's reuse mechanism, not a
reproduction of its embedding model or GPU pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any
import numpy as np

from ..core import Outcome, Plan, Structure, Task, Unit


@dataclass
class LogicalInput:
    entities: np.ndarray
    relations: np.ndarray
    triples: tuple[tuple[int, int, int], ...]
    expressions: tuple[tuple, ...]
    positives: np.ndarray
    negatives: np.ndarray
    seed: int


def answers(expression: tuple, adjacency: dict, universe: frozenset[int]) -> frozenset[int]:
    op, *args = expression
    if op == "entity":
        return frozenset([args[0]])
    if op == "project":
        src, relation = args
        return frozenset(v for u in answers(src, adjacency, universe)
                         for v in adjacency[(u, relation)])
    if op == "negate":
        return universe - answers(args[0], adjacency, universe)
    left, right = (answers(a, adjacency, universe) for a in args)
    return left & right if op == "intersect" else left | right


def construct(seed=0, queries=192, dimension=24, cohorts=24,
              entity_count=128, relation_count=6) -> LogicalInput:
    rng = np.random.default_rng(seed)
    adjacency = {}
    triples = []
    for e in range(entity_count):
        for r in range(relation_count):
            tails = tuple(sorted(map(int, rng.choice(entity_count, 3, replace=False))))
            adjacency[(e, r)] = tails
            triples.extend((e, r, t) for t in tails)
    universe = frozenset(range(entity_count))
    cohort_data = [tuple(map(int, (rng.integers(entity_count),
                                   rng.integers(relation_count),
                                   rng.integers(relation_count)))) for _ in range(cohorts)]
    expressions, positives, negatives = [], [], []
    for q in range(queries):
        anchor, r0, r1 = cohort_data[q % cohorts]
        prefix = ("project", ("project", ("entity", anchor), r0), r1)
        other = ("project", ("entity", int(rng.integers(entity_count))),
                 int(rng.integers(relation_count)))
        typ = q % 5
        expr = (prefix if typ == 0 else
                ("project", prefix, q % relation_count) if typ == 1 else
                ("intersect", prefix, other) if typ == 2 else
                ("union", prefix, other) if typ == 3 else
                ("intersect", prefix, ("negate", other)))
        truth = answers(expr, adjacency, universe)
        if not truth or len(truth) == entity_count:
            expr, truth = prefix, answers(prefix, adjacency, universe)
        expressions.append(expr)
        positives.append(int(rng.choice(sorted(truth))))
        negatives.append(rng.choice(sorted(universe - truth), 8, replace=False))
    return LogicalInput(rng.normal(0, 0.22, (entity_count, dimension)),
                        rng.normal(0, 0.16, (relation_count, dimension, dimension)),
                        tuple(triples), tuple(expressions), np.array(positives),
                        np.array(negatives), seed)


@dataclass
class Cache:
    model_digest: str
    unit_descriptors: tuple[tuple, ...]
    values: dict[int, np.ndarray]


def model_digest(data: LogicalInput) -> str:
    return hashlib.sha256(data.entities.tobytes() + data.relations.tobytes()).hexdigest()


def graph_for(data: LogicalInput, training: bool) -> tuple[Structure, tuple[int, ...]]:
    nodes: list[Unit] = []
    dim = data.entities.shape[1]
    def emit(expr):
        op, *args = expr
        if op == "entity":
            deps, params, work = (), (int(args[0]),), float(dim)
        elif op == "project":
            deps, params = (emit(args[0]),), (int(args[1]),)
            work = float((6 if training else 2) * dim * dim + 5 * dim)
        else:
            deps, params = tuple(emit(a) for a in args), ()
            work = float((4 if training else 2) * dim)
        uid = len(nodes)
        nodes.append(Unit(uid, op, deps, params, work, shareable=True))
        return uid
    roots = tuple(emit(e) for e in data.expressions)
    return Structure(tuple(nodes)), roots


class LogicalAdapter:
    domain = "logical"

    def prepare(self, data: LogicalInput, *, training=True, cache: Cache | None = None) -> Task:
        structure, roots = graph_for(data, training)
        digest = model_digest(data)
        descriptors = tuple((u.op, u.deps, u.parameters) for u in structure.units)
        required, certificate, visits = None, None, 0
        if cache is not None:
            if training:
                raise ValueError("training caches cannot skip gradient contributions")
            if digest != cache.model_digest or len(descriptors) != len(cache.unit_descriptors):
                changed = set(structure.by_id)
            else:
                changed = {u for u, (a, b) in enumerate(zip(descriptors, cache.unit_descriptors))
                           if a != b or u not in cache.values}
            required, visits = structure.affected(changed)
            certificate = "input_and_parameter_versions"
        count, dim = len(roots), data.entities.shape[1]
        # Positive/negative scoring is common to both graph execution plans.
        fixed_work = count * (1 + data.negatives.shape[1]) * dim * (6 if training else 2)
        return Task("kg_query_training" if training else "incremental_query_evaluation",
                    self.domain, structure, required=required,
                    range_certificate=certificate, analysis_work=float(visits),
                    fixed_work=float(fixed_work), allow_merge=True,
                    constraint_name="loss_gradient_parameter_equivalence" if training else "exact_outputs",
                    payload={"data": data, "training": training, "roots": roots,
                             "cache": cache, "digest": digest, "descriptors": descriptors})

    def execute(self, task: Task, plan: Plan) -> Outcome:
        p = task.payload
        data, training, roots = p["data"], p["training"], p["roots"]
        cached = p["cache"] if "scope" in plan.transforms else None
        values = dict(cached.values) if cached is not None else {}
        representatives = {u: g[0] for g in plan.groups for u in g}
        executed = {"entity": 0, "project": 0, "intersect": 0, "union": 0, "negate": 0}
        for group in plan.groups:
            unit = task.structure.by_id[group[0]]
            deps = [values[d] for d in unit.deps]
            if unit.op == "entity":
                value = data.entities[unit.parameters[0]].copy()
            elif unit.op == "project":
                value = np.tanh(deps[0] @ data.relations[unit.parameters[0]])
            elif unit.op == "intersect":
                value = (deps[0] + deps[1]) * 0.5
            elif unit.op == "union":
                value = np.maximum(deps[0], deps[1])
            elif unit.op == "negate":
                value = -deps[0]
            else:
                raise ValueError(unit.op)
            for u in group:
                values[u] = value
            executed[unit.op] += 1
        query = np.stack([values[r] for r in roots])
        ids = np.column_stack((data.positives, data.negatives))
        targets = data.entities[ids]
        scores = np.einsum("qd,qkd->qk", query, targets)
        labels = np.zeros_like(scores)
        labels[:, 0] = 1.0
        loss = float(np.mean(np.logaddexp(0, scores) - labels * scores))
        metrics = {"loss": loss, "operator_executions": sum(executed.values()),
                   "project_executions": executed["project"],
                   "operators": executed, "observed_kernel_work": plan.kernel_work}
        output = {"query": query, "scores": scores, "loss": loss,
                  "cache": Cache(p["digest"], p["descriptors"], values)}
        if training:
            grad_entities = np.zeros_like(data.entities)
            grad_relations = np.zeros_like(data.relations)
            ds = (1 / (1 + np.exp(-scores)) - labels) / scores.size
            dq = np.einsum("qk,qkd->qd", ds, targets)
            np.add.at(grad_entities, ids.ravel(),
                      (ds[:, :, None] * query[:, None, :]).reshape(-1, query.shape[1]))
            grads: dict[int, np.ndarray] = {}
            def add(uid, grad):
                rep = representatives[uid]
                if rep in grads:
                    grads[rep] += grad
                else:
                    grads[rep] = grad.copy()
            for root, g in zip(roots, dq):
                add(root, g)
            for group in reversed(plan.groups):
                u = task.structure.by_id[group[0]]
                gradient = grads.get(u.uid)
                if gradient is None:
                    continue
                if u.op == "entity":
                    grad_entities[u.parameters[0]] += gradient
                elif u.op == "project":
                    dz = gradient * (1 - values[u.uid] ** 2)
                    rid = u.parameters[0]
                    grad_relations[rid] += np.outer(values[u.deps[0]], dz)
                    add(u.deps[0], dz @ data.relations[rid].T)
                elif u.op == "intersect":
                    add(u.deps[0], gradient * 0.5)
                    add(u.deps[1], gradient * 0.5)
                elif u.op == "union":
                    mask = values[u.deps[0]] >= values[u.deps[1]]
                    add(u.deps[0], gradient * mask)
                    add(u.deps[1], gradient * ~mask)
                else:
                    add(u.deps[0], -gradient)
            output.update(grad_entities=grad_entities, grad_relations=grad_relations)
        return Outcome(output, metrics)

    @staticmethod
    def validate(result: Outcome, reference: Outcome, tolerance=1e-11) -> dict:
        errors = {}
        for key in ("query", "scores", "loss", "grad_entities", "grad_relations"):
            if key in reference.values:
                errors[f"max_{key}_error"] = float(np.max(np.abs(
                    np.asarray(result.values[key]) - np.asarray(reference.values[key]))))
        return {"passed": all(e <= tolerance for e in errors.values()), **errors}


def step(data: LogicalInput, result: Outcome, learning_rate=0.1) -> LogicalInput:
    return replace(data, entities=data.entities - learning_rate * result.values["grad_entities"],
                   relations=data.relations - learning_rate * result.values["grad_relations"])


def change_inputs(data: LogicalInput, fraction: float) -> LogicalInput:
    """Keep AST layout and change a reproducible fraction of anchor leaves."""
    if not 0 <= fraction <= 1:
        raise ValueError("invalid fraction")
    def rewrite(expr):
        op, *args = expr
        if op == "entity":
            return (op, (args[0] + 17) % len(data.entities))
        if op == "project":
            return (op, rewrite(args[0]), args[1])
        return (op, *(rewrite(a) for a in args))
    count = int(round(len(data.expressions) * fraction))
    expressions = tuple(rewrite(e) if i < count else e for i, e in enumerate(data.expressions))
    # Evaluation output equivalence uses unchanged scoring targets; this is not a
    # training batch with labels for the modified expressions.
    return replace(data, expressions=expressions)
