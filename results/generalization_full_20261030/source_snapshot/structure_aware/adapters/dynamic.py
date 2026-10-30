"""Exact landmark-distance maintenance on a certified block graph.

This simulates FulBM's *range restriction mechanism*. It does not reproduce its
highway labels, strong/weak paths, or the original C++ implementation. Pendant
blocks have a single boundary vertex; edits within such a block cannot alter
distances outside it when landmarks are on the backbone. Cross-block edits
invalidate the certificate and trigger full recomputation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np

from ..core import Outcome, Plan, Structure, Task, Unit

INF = 1_000_000_000


def bfs(adjacency: list[set[int]], source: int) -> tuple[np.ndarray, int, int]:
    distance = np.full(len(adjacency), INF, dtype=np.int64)
    distance[source] = 0
    queue = deque([source])
    visited = edges = 0
    while queue:
        u = queue.popleft()
        visited += 1
        for v in adjacency[u]:
            edges += 1
            if distance[v] == INF:
                distance[v] = distance[u] + 1
                queue.append(v)
    return distance, visited, edges


@dataclass
class DynamicInput:
    adjacency: list[set[int]]
    landmarks: tuple[int, ...]
    block_of: tuple[int, ...]
    roots: tuple[int, ...]
    updates: tuple[tuple[str, int, int], ...]
    old_distances: np.ndarray
    seed: int


def construct(seed=0, blocks=32, block_size=64, touched=2,
              edits_per_block=4, cross_block=False) -> DynamicInput:
    if not 1 <= touched <= blocks or block_size < 4 or blocks < 4:
        raise ValueError("invalid block graph size")
    rng = np.random.default_rng(seed)
    n = blocks * (block_size + 1)
    adjacency = [set() for _ in range(n)]
    def edge(u, v):
        if u != v:
            adjacency[u].add(v)
            adjacency[v].add(u)
    for root in range(1, blocks):
        edge(root - 1, root)
    membership = [-1] * blocks
    for block in range(blocks):
        start = blocks + block * block_size
        edge(block, start)
        for j in range(block_size):
            membership.append(block)
            if j:
                edge(start + j, start + int(rng.integers(j)))
        for _ in range(block_size):
            u, v = rng.choice(block_size, 2, replace=False) + start
            edge(int(u), int(v))
    landmarks = tuple(sorted({0, blocks // 3, 2 * blocks // 3, blocks - 1}))
    old = np.stack([bfs(adjacency, r)[0] for r in landmarks])
    updates: list[tuple[str, int, int]] = []
    for block in rng.choice(blocks, touched, replace=False):
        block = int(block)
        start = blocks + block * block_size
        existing = [(u, v) for u in range(start, start + block_size)
                    for v in sorted(adjacency[u]) if u < v]
        indices = rng.choice(len(existing), min(edits_per_block, len(existing)), replace=False)
        for idx in indices:
            u, v = existing[int(idx)]
            updates.append(("delete", u, v))
        missing = [(u, v) for u in range(start, start + block_size)
                   for v in range(u + 1, start + block_size) if v not in adjacency[u]]
        indices = rng.choice(len(missing), min(edits_per_block, len(missing)), replace=False)
        updates.extend(("insert", *missing[int(i)]) for i in sorted(indices))
    if cross_block:
        updates.append(("insert", blocks, blocks + block_size))
    return DynamicInput(adjacency, landmarks, tuple(membership), tuple(range(blocks)),
                        tuple(updates), old, seed)


class DynamicAdapter:
    domain = "dynamic"

    def prepare(self, data: DynamicInput) -> Task:
        # Copying the graph and validating updates is included in online prepare time.
        adjacency = [set(neighbors) for neighbors in data.adjacency]
        touched = set()
        certified = True
        for kind, u, v in data.updates:
            if kind not in ("insert", "delete") or u == v:
                raise ValueError("invalid update")
            if kind == "insert":
                if v in adjacency[u]:
                    raise ValueError("duplicate insertion")
                adjacency[u].add(v)
                adjacency[v].add(u)
            else:
                if v not in adjacency[u]:
                    raise ValueError("deleting absent edge")
                adjacency[u].remove(v)
                adjacency[v].remove(u)
            bu, bv = data.block_of[u], data.block_of[v]
            if bu >= 0 and bu == bv:
                touched.add(bu)
            elif bu >= 0 and v == data.roots[bu]:
                touched.add(bu)
            elif bv >= 0 and u == data.roots[bv]:
                touched.add(bv)
            else:
                certified = False
        # Generated initial structure is certified once; arbitrary external graphs
        # must be checked by validate_partition before producing the certificate.
        if not self.validate_partition(data):
            certified = False
        if any(data.block_of[r] != -1 for r in data.landmarks):
            certified = False
        active = (frozenset(i for i, b in enumerate(data.block_of) if b in touched)
                  if certified else frozenset(range(len(adjacency))))
        units = tuple(Unit(i, "distance_label", tuple(sorted(neighbors)),
                           work=len(data.landmarks) * (len(neighbors) + 1))
                      for i, neighbors in enumerate(adjacency))
        # Full partition validation scans initial adjacency; count it explicitly.
        validation_work = sum(map(len, data.adjacency)) + len(data.adjacency)
        return Task("landmark_distance_update", self.domain, Structure(units, cyclic=True),
                    required=active,
                    range_certificate=("single_boundary_block" if certified else "full_fallback"),
                    analysis_work=float(validation_work + len(data.updates)),
                    payload=(data, adjacency, certified, tuple(sorted(touched))))

    @staticmethod
    def validate_partition(data: DynamicInput) -> bool:
        for u, neighbors in enumerate(data.adjacency):
            for v in neighbors:
                bu, bv = data.block_of[u], data.block_of[v]
                if bu == -1 and bv == -1:
                    continue
                if bu >= 0 and (bv == bu or v == data.roots[bu]):
                    continue
                if bv >= 0 and u == data.roots[bv]:
                    continue
                return False
        return True

    def execute(self, task: Task, plan: Plan) -> Outcome:
        data, adjacency, certified, touched = task.payload
        visited = scans = 0
        if "scope" not in plan.transforms or not certified:
            rows = []
            for r in data.landmarks:
                distances, vv, ee = bfs(adjacency, r)
                rows.append(distances)
                visited += vv
                scans += ee
            distances = np.stack(rows)
        else:
            distances = data.old_distances.copy()
            active = set(plan.active)
            for row in range(len(data.landmarks)):
                distances[row, list(active)] = INF
                queue = deque()
                for block in touched:
                    root = data.roots[block]
                    for v in adjacency[root]:
                        scans += 1
                        if v in active and data.block_of[v] == block:
                            candidate = distances[row, root] + 1
                            if candidate < distances[row, v]:
                                distances[row, v] = candidate
                                queue.append(v)
                while queue:
                    u = queue.popleft()
                    visited += 1
                    for v in adjacency[u]:
                        scans += 1
                        if v in active and distances[row, v] > distances[row, u] + 1:
                            distances[row, v] = distances[row, u] + 1
                            queue.append(v)
        return Outcome(distances, {"vertices_processed": visited,
                                   "edge_examinations": scans,
                                   "observed_kernel_work": visited + scans,
                                   "certificate_valid": certified,
                                   "changed_distance_entries": int(np.count_nonzero(
                                       distances != data.old_distances))})

    @staticmethod
    def validate(result: Outcome, reference: Outcome) -> dict:
        mismatch = int(np.count_nonzero(result.values != reference.values))
        return {"passed": mismatch == 0, "distance_mismatches": mismatch}
