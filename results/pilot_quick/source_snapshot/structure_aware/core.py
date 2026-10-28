"""Executable counterpart of G_s, A_s, P_s, beta and W in the thesis.

Costs are within-task weighted operation counts, NOT milliseconds. The planner
shares transforms across adapters; backends supply semantics and execute kernels.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
import heapq
import math
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class Unit:
    uid: int
    op: str
    deps: tuple[int, ...] = ()
    parameters: tuple[Any, ...] = ()
    work: float = 1.0
    shareable: bool = False


@dataclass
class Structure:
    units: tuple[Unit, ...]
    cyclic: bool = False

    def __post_init__(self):
        self.by_id = {u.uid: u for u in self.units}
        if len(self.by_id) != len(self.units):
            raise ValueError("duplicate unit ids")
        self.users: dict[int, list[int]] = defaultdict(list)
        seen: set[int] = set()
        for u in self.units:
            for d in u.deps:
                if d not in self.by_id:
                    raise ValueError(f"missing dependency {d}")
                if not self.cyclic and d not in seen:
                    raise ValueError("DAG units must be topologically ordered")
                self.users[d].append(u.uid)
            seen.add(u.uid)

    def affected(self, changed: Iterable[int]) -> tuple[frozenset[int], int]:
        """Shared version-invalidation propagation, also valid on cyclic graphs."""
        reached = set(changed)
        if not reached.issubset(self.by_id):
            raise ValueError("unknown changed unit")
        todo = deque(sorted(reached))
        visits = len(reached)
        while todo:
            for v in self.users[todo.popleft()]:
                visits += 1
                if v not in reached:
                    reached.add(v)
                    todo.append(v)
        return frozenset(reached), visits


@dataclass(frozen=True)
class BudgetDomain:
    """Discrete budgets and an offline-calibrated sensitivity surrogate."""
    minimum: int
    maximum: int
    weights: tuple[float, ...]
    base_work: tuple[float, ...]
    incremental_work: tuple[float, ...]
    # Positive surrogate slope, calibrated against reference trajectories.
    power: float = 1.0

    def residual(self, budget: tuple[int, ...]) -> float:
        return sum(w / (b + 1) ** self.power
                   for w, b in zip(self.weights, budget))

    def work(self, position: int, level: int) -> float:
        return self.base_work[position] + self.incremental_work[position] * level


@dataclass
class Task:
    name: str
    domain: str
    structure: Structure
    required: frozenset[int] | None = None
    range_certificate: str | None = None
    analysis_work: float = 0.0
    fixed_work: float = 0.0
    allow_merge: bool = False
    budget: BudgetDomain | None = None
    permitted_budget: tuple[int, ...] | None = None
    constraint_name: str = "exact"
    payload: Any = None
    legality: Callable[["Plan"], bool] | None = None


@dataclass(frozen=True)
class Plan:
    name: str
    active: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]
    budgets: tuple[int, ...] = ()
    transforms: tuple[str, ...] = ()
    kernel_work: float = 0.0
    organization_work: float = 0.0

    @property
    def estimated_work(self) -> float:
        return self.kernel_work + self.organization_work

    def record(self) -> dict[str, Any]:
        return {"name": self.name, "active_units": len(self.active),
                "execution_groups": len(self.groups),
                "budgets": list(self.budgets), "transforms": list(self.transforms),
                "kernel_work": self.kernel_work,
                "organization_work": self.organization_work,
                "estimated_work": self.estimated_work}


def allocate_budget(domain: BudgetDomain, total: int) -> tuple[int, ...]:
    """One generic marginal-benefit allocator, shared by budget-enabled tasks.

    For the separable decreasing surrogate and equal unit increments this greedily
    chooses the largest next error reduction. No claim of actual-quality optimality.
    """
    n = len(domain.weights)
    if not n or not n * domain.minimum <= total <= n * domain.maximum:
        raise ValueError("budget outside feasible interval")
    if len(domain.base_work) != n or len(domain.incremental_work) != n:
        raise ValueError("inconsistent budget domain")
    if any(not math.isfinite(w) or w < 0 for w in domain.weights):
        raise ValueError("invalid sensitivity")
    levels = [domain.minimum] * n
    def gain(i: int) -> float:
        b = levels[i]
        return domain.weights[i] * ((b + 1) ** -domain.power
                                    - (b + 2) ** -domain.power)
    heap = [(-gain(i), i) for i in range(n) if levels[i] < domain.maximum]
    heapq.heapify(heap)
    for _ in range(total - sum(levels)):
        _, i = heapq.heappop(heap)
        levels[i] += 1
        if levels[i] < domain.maximum:
            heapq.heappush(heap, (-gain(i), i))
    return tuple(levels)


class Planner:
    """Common scope/merge/budget passes and measured-work-based plan selection."""

    def _groups(self, task: Task, active: tuple[int, ...], merge: bool):
        if not merge:
            return tuple((i,) for i in active), 0
        if task.structure.cyclic:
            raise ValueError("common-subexpression merge requires a DAG")
        classes: dict[tuple, int] = {}
        identity: dict[int, int] = {}
        buckets: dict[int, list[int]] = {}
        chosen = set(active)
        visits = 0
        for u in task.structure.units:
            visits += 1 + len(u.deps)
            key = (u.op, u.parameters, tuple(identity[d] for d in u.deps))
            if u.shareable:
                cid = classes.setdefault(key, len(classes))
                identity[u.uid] = cid
            else:
                # Namespace prevents collisions with shareable expression classes.
                identity[u.uid] = -(u.uid + 1)
            if u.uid in chosen:
                buckets.setdefault(identity[u.uid], []).append(u.uid)
        return tuple(tuple(v) for v in buckets.values()), visits

    def make(self, task: Task, *, scope=False, merge=False,
             budgets: tuple[int, ...] | None = None, name: str | None = None) -> Plan:
        if scope and (task.required is None or not task.range_certificate):
            raise ValueError("scope needs a domain correctness certificate")
        if merge and not task.allow_merge:
            raise ValueError("merge forbidden by domain semantics")
        all_ids = tuple(u.uid for u in task.structure.units)
        wanted = task.required if scope else frozenset(all_ids)
        if not wanted.issubset(task.structure.by_id):
            raise ValueError("unknown active unit")
        active = tuple(i for i in all_ids if i in wanted)
        groups, visits = self._groups(task, active, merge)
        if task.budget is not None:
            if scope or merge:
                raise ValueError("this backend has no legality rule for that combination")
            levels = budgets or (task.budget.maximum,) * len(all_ids)
            if len(levels) != len(all_ids) or any(
                    not isinstance(b, int) or not task.budget.minimum <= b <= task.budget.maximum
                    for b in levels):
                raise ValueError("invalid per-unit budget")
            kernel = task.fixed_work + sum(task.budget.work(i, b)
                                            for i, b in enumerate(levels))
        else:
            if budgets:
                raise ValueError("exact task does not permit approximate budgets")
            levels = ()
            kernel = task.fixed_work + sum(task.structure.by_id[g[0]].work for g in groups)
        transforms = tuple(k for k, enabled in (("scope", scope), ("merge", merge),
                           ("budget", bool(task.budget and levels !=
                                          (task.budget.maximum,) * len(all_ids)))) if enabled)
        organization = (task.analysis_work if scope else 0.0) + visits
        plan = Plan(name or ("+".join(transforms) or "reference"), active, groups,
                    levels, transforms, kernel, organization)
        if task.legality is not None and not task.legality(plan):
            raise ValueError("domain legality check rejected plan")
        self.validate(task, plan)
        return plan

    def validate(self, task: Task, plan: Plan) -> None:
        active_set = set(plan.active)
        ordered = tuple(u.uid for u in task.structure.units if u.uid in active_set)
        if ordered != plan.active:
            raise ValueError("active units must be known and in structural order")
        if any(not g for g in plan.groups):
            raise ValueError("empty execution group")
        flat = [u for g in plan.groups for u in g]
        if len(flat) != len(set(flat)) or set(flat) != set(plan.active):
            raise ValueError("groups must partition the actual processing range")
        if task.required is not None and not task.required.issubset(plan.active):
            raise ValueError("plan drops required computation")
        if "merge" in plan.transforms and not task.allow_merge:
            raise ValueError("illegal merge")
        if "merge" in plan.transforms:
            expected, _ = self._groups(task, plan.active, True)
            if expected != plan.groups:
                raise ValueError("grouping violates common-subexpression semantics")
        elif any(len(g) != 1 for g in plan.groups):
            raise ValueError("unregistered merge")
        if task.budget:
            if len(plan.budgets) != len(task.structure.units) or any(
                not isinstance(b, int) or not task.budget.minimum <= b <= task.budget.maximum
                for b in plan.budgets):
                raise ValueError("invalid budget")
        elif plan.budgets:
            raise ValueError("approximation is forbidden for this task")
        if task.legality and not task.legality(plan):
            raise ValueError("domain legality check rejected plan")

    def candidates(self, task: Task) -> list[Plan]:
        result = [self.make(task)]
        scopes = [False, True] if task.required is not None else [False]
        merges = [False, True] if task.allow_merge else [False]
        for scope in scopes:
            for merge in merges:
                if scope or merge:
                    result.append(self.make(task, scope=scope, merge=merge))
        if task.budget and task.permitted_budget:
            result.append(self.make(task, budgets=task.permitted_budget))
        return result

    def select(self, task: Task) -> tuple[Plan, list[dict[str, Any]]]:
        choices = self.candidates(task)
        # Stable reference preference when all gains vanish.
        chosen = min(choices, key=lambda p: p.estimated_work)
        return chosen, [p.record() for p in choices]


@dataclass
class Outcome:
    values: Any
    metrics: dict[str, Any] = field(default_factory=dict)


class Runtime:
    """One planning/validation/dispatch entry point for all three backends."""
    def __init__(self):
        self.planner = Planner()
        self.backends: dict[str, Any] = {}

    def register(self, backend):
        if backend.domain in self.backends:
            raise ValueError("duplicate backend")
        self.backends[backend.domain] = backend

    def execute(self, task: Task, plan: Plan | None = None) -> tuple[Outcome, Plan]:
        plan = plan or self.planner.select(task)[0]
        self.planner.validate(task, plan)
        return self.backends[task.domain].execute(task, plan), plan

    def checked(self, task: Task, validator: Callable[[Outcome], dict],
                plan: Plan | None = None) -> tuple[Outcome, Plan, dict]:
        """Validation feedback and safe fallback; all attempts remain observable.

        An oracle-based validator is appropriate for this simulation harness.
        Production quality monitoring would need its own deployable validator.
        """
        result, selected = self.execute(task, plan)
        check = validator(result)
        attempts = [{"plan": selected.record(), "validation": check}]
        if not check["passed"]:
            selected = self.planner.make(task)
            result, selected = self.execute(task, selected)
            check = validator(result)
            attempts.append({"plan": selected.record(), "validation": check})
            if not check["passed"]:
                raise RuntimeError("reference execution failed its validation")
        return result, selected, {"passed": check["passed"], "fallback": len(attempts) > 1,
                                  "validation": check, "attempts": attempts}
