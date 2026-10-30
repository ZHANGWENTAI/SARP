"""SARR: adapter-described DAGs executed by the existing structural planner.

Adapters declare nodes, dependencies, kernels and costs. They do not implement
invalidation, common-expression elimination, scheduling or cache management.
Purity and operation versions are explicit contracts; effects are conservative.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import hashlib
import json
import math
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .core import Outcome, Plan, Runtime, Structure, Task, Unit


def _key(value: Any) -> Any:
    """Content keys for supported immutable values, including array dtype/shape."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays require a domain-specific backend")
        a = np.ascontiguousarray(value)
        return ("array", a.dtype.str, a.shape,
                hashlib.sha256(a.tobytes()).hexdigest())
    if isinstance(value, np.generic):
        return _key(np.asarray(value))
    if value is None or isinstance(value, (str, bool, int)):
        return (type(value).__name__, value)
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, (tuple, list)):
        return ("sequence", tuple(_key(x) for x in value))
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("parameter and input mappings need string keys")
        return ("mapping", tuple((k, _key(v)) for k, v in sorted(value.items())))
    raise TypeError(f"unsupported graph value: {type(value).__name__}")


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.flags.writeable = False
        return result
    if isinstance(value, np.generic):
        return value.copy()
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(x) for x in value)
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("kernels must return arrays, scalars or tuples of these")


@dataclass(frozen=True)
class Operation:
    kernel: Callable[..., Any]
    # Only explicitly pure kernels may be merged or cached across executions.
    pure: bool = False
    version: str = "1"


@dataclass(frozen=True)
class Node:
    name: str
    operation: str
    dependencies: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    work: float = 1.0


@dataclass(frozen=True)
class Graph:
    namespace: str
    inputs: Mapping[str, Any]
    nodes: tuple[Node, ...]
    outputs: tuple[str, ...]


class Adapter(Protocol):
    """Kernels must declare hidden model state in parameters or their version.

    Side effects use pure=False; order-dependent effects need dependency edges.
    Loops/cycles must be lowered to a DAG or supplied through a native backend.
    """
    operations: Mapping[str, Operation]

    def build(self, request: Any) -> Graph: ...


@dataclass
class Session:
    """Per-client cache; no result is shared implicitly across clients."""
    namespace: str | None = None
    signatures: dict[str, Any] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> Session:
        # Cached arrays are read-only, so independent dictionaries suffice.
        return Session(self.namespace, dict(self.signatures), dict(self.values))


@dataclass
class _Payload:
    names: tuple[str, ...]
    nodes: Mapping[str, Node]
    operations: Mapping[str, Operation]
    inputs: Mapping[str, Any]
    cached: Mapping[str, Any]
    outputs: tuple[str, ...]
    pure: Mapping[str, bool]


class _GraphBackend:
    domain = "sarr.dag"

    def execute(self, task: Task, plan: Plan) -> Outcome:
        data: _Payload = task.payload
        values = dict(data.cached)
        calls: Counter[str] = Counter()
        computed = 0
        for group in plan.groups:
            uid = group[0]
            name = data.names[uid]
            if name in data.inputs:
                result = data.inputs[name]
            else:
                node = data.nodes[name]
                arguments = [values[d] for d in node.dependencies]
                result = _freeze(data.operations[node.operation].kernel(
                    *arguments, **dict(node.parameters)))
                calls[node.operation] += 1
            computed += 1
            for member in group:
                values[data.names[member]] = result
        outputs = {name: values[name] for name in data.outputs}
        cache = {name: value for name, value in values.items() if data.pure.get(name)}
        return Outcome({"outputs": outputs, "cache": cache},
                       {"kernel_calls": sum(calls.values()),
                        "calls_by_operation": dict(calls),
                        "executed_groups": computed,
                        "active_units": len(plan.active),
                        "total_units": len(data.names)})


@dataclass
class Prepared:
    task: Task
    namespace: str
    signatures: dict[str, Any]


@dataclass
class Result:
    outputs: dict[str, Any]
    plan: Plan
    candidates: list[dict[str, Any]]
    metrics: dict[str, Any]
    timing: dict[str, float]
    validation: dict[str, Any] | None


class SARR:
    """Structure-Aware Reasoning Runtime; one executor, any declared DAG adapter."""
    def __init__(self):
        self.runtime = Runtime()
        self.runtime.register(_GraphBackend())

    def prepare(self, adapter: Adapter, request: Any,
                session: Session | None = None) -> Prepared:
        graph = adapter.build(request)
        if not graph.namespace or not graph.outputs:
            raise ValueError("a graph needs a namespace and at least one output")
        inputs = {name: _freeze(value) for name, value in graph.inputs.items()}
        nodes = {node.name: Node(node.name, node.operation, node.dependencies,
                                _freeze(node.parameters), node.work)
                 for node in graph.nodes}
        if len(nodes) != len(graph.nodes) or set(nodes) & set(inputs):
            raise ValueError("duplicate node/input name")
        if any(not isinstance(name, str) for name in (*inputs, *nodes)):
            raise TypeError("node/input names must be strings")
        known = set(inputs) | set(nodes)
        if not set(graph.outputs) <= known:
            raise ValueError("unknown requested output")
        users: dict[str, list[str]] = {name: [] for name in known}
        indegree = {name: 0 for name in inputs}
        for name, node in nodes.items():
            if node.operation not in adapter.operations:
                raise ValueError(f"unregistered operation: {node.operation}")
            if not math.isfinite(node.work) or node.work < 0:
                raise ValueError("node work must be finite and nonnegative")
            if not set(node.dependencies) <= known:
                raise ValueError(f"unknown dependency of {name}")
            indegree[name] = len(node.dependencies)
            for dependency in node.dependencies:
                users[dependency].append(name)
        queue = deque(name for name, degree in indegree.items() if degree == 0)
        names = []
        while queue:
            name = queue.popleft()
            names.append(name)
            for user in users[name]:
                indegree[user] -= 1
                if indegree[user] == 0:
                    queue.append(user)
        if len(names) != len(known):
            raise ValueError("cyclic task: lower loops to a DAG or use a native backend")
        ids = {name: i for i, name in enumerate(names)}
        signatures, pure, units = {}, {}, []
        for name in names:
            if name in inputs:
                signature = ("input", name, _key(inputs[name]))
                pure[name] = True
                unit = Unit(ids[name], "__input__", parameters=signature,
                            work=1, shareable=True)
            else:
                node = nodes[name]
                operation = adapter.operations[node.operation]
                signature = (node.operation, operation.version, node.dependencies,
                             _key(node.parameters))
                pure[name] = operation.pure
                unit = Unit(ids[name], node.operation,
                            tuple(ids[d] for d in node.dependencies),
                            (operation.version, _key(node.parameters)),
                            node.work, operation.pure)
            signatures[name] = signature
            units.append(unit)
        structure = Structure(tuple(units))
        previous = session if session and session.namespace == graph.namespace else Session()
        changed = [ids[name] for name in names
                   if not pure[name] or previous.signatures.get(name) != signatures[name]
                   or name not in previous.values]
        affected, visits = structure.affected(changed)
        # Keep output ancestors and all effectful nodes (and their ancestors).
        live: set[str] = set()
        stack = list(graph.outputs) + [name for name in names if not pure[name]]
        while stack:
            name = stack.pop()
            if name in live:
                continue
            live.add(name)
            if name in nodes:
                stack.extend(nodes[name].dependencies)
        required = frozenset(ids[name] for name in live if ids[name] in affected)
        cached = {name: value for name, value in previous.values.items()
                  if name in ids and ids[name] not in affected and pure[name]}
        payload = _Payload(tuple(names), nodes, dict(adapter.operations), inputs,
                           cached, graph.outputs, pure)
        task = Task(graph.namespace, _GraphBackend.domain, structure,
                    required=required,
                    range_certificate="dependency-closure-and-versioned-cache",
                    analysis_work=float(visits + len(names)),
                    allow_merge=any(pure[name] for name in nodes), payload=payload)
        return Prepared(task, graph.namespace, signatures)

    def run(self, adapter: Adapter, request: Any, *, session: Session | None = None,
            mode: str = "auto",
            validator: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None) -> Result:
        start = perf_counter()
        prepared = self.prepare(adapter, request, session)
        prepared_at = perf_counter()
        planner = self.runtime.planner
        if mode == "auto":
            plan, candidates = planner.select(prepared.task)
        else:
            if mode not in {"reference", "scope", "merge", "combined"}:
                raise ValueError("unknown execution mode")
            plan = planner.make(prepared.task, scope=mode in {"scope", "combined"},
                                merge=mode in {"merge", "combined"})
            candidates = [plan.record()]
        planned_at = perf_counter()
        outcome, plan = self.runtime.execute(prepared.task, plan)
        executed_at = perf_counter()
        validation = None
        if validator is not None:
            check = validator(outcome.values["outputs"])
            validation = {"passed": bool(check["passed"]), "fallback": False,
                          "attempts": [{"plan": plan.record(), "validation": check}]}
            if not check["passed"]:
                if not all(prepared.task.payload.pure.values()):
                    raise RuntimeError("validation failed; effectful tasks cannot be replayed automatically")
                plan = planner.make(prepared.task)
                fallback_start = perf_counter()
                outcome, plan = self.runtime.execute(prepared.task, plan)
                # Charge fallback computation to online execution, not validation.
                fallback_duration = perf_counter() - fallback_start
                executed_at += fallback_duration
                check = validator(outcome.values["outputs"])
                validation.update(passed=bool(check["passed"]), fallback=True)
                validation["attempts"].append({"plan": plan.record(), "validation": check})
                if not check["passed"]:
                    raise RuntimeError("reference execution failed validation")
        finish = perf_counter()
        if session is not None:
            session.namespace = prepared.namespace
            session.signatures = prepared.signatures
            session.values = outcome.values["cache"]
        online = executed_at - start
        return Result(outcome.values["outputs"], plan, candidates, outcome.metrics,
                      {"prepare": prepared_at - start,
                       "planning": planned_at - prepared_at,
                       "execution": executed_at - planned_at,
                       "online": online, "validation": finish - start - online},
                      validation)


def output_digest(outputs: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(_key(outputs), separators=(",", ":")).encode()).hexdigest()
