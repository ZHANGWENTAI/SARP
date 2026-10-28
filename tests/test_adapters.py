from dataclasses import replace
import unittest
import numpy as np

from structure_aware.core import Planner, Runtime
from structure_aware.adapters import dynamic, logical, video


class DynamicTests(unittest.TestCase):
    def test_mixed_edits_against_independent_floyd_warshall(self):
        backend, planner = dynamic.DynamicAdapter(), Planner()
        for seed in range(8):
            data = dynamic.construct(seed, blocks=4, block_size=5, touched=2, edits_per_block=1)
            task = backend.prepare(data)
            out = backend.execute(task, planner.make(task, scope=True))
            adjacency = task.payload[1]
            n = len(adjacency)
            distances = np.full((n, n), dynamic.INF, dtype=np.int64)
            np.fill_diagonal(distances, 0)
            for u, neighbors in enumerate(adjacency):
                distances[u, list(neighbors)] = 1
            for k in range(n):
                distances = np.minimum(distances, distances[:, k, None] + distances[None, k, :])
            np.testing.assert_array_equal(out.values, distances[list(data.landmarks)])

    def test_cross_block_edit_falls_back(self):
        backend = dynamic.DynamicAdapter()
        task = backend.prepare(dynamic.construct(blocks=4, block_size=8, cross_block=True))
        self.assertFalse(task.payload[2])
        self.assertEqual(Planner().select(task)[0].name, "reference")

    def test_disconnection_and_reconnection(self):
        backend, planner = dynamic.DynamicAdapter(), Planner()
        data = dynamic.construct(blocks=4, block_size=8, touched=1)
        data = replace(data, updates=(("delete", 0, 4),))
        task = backend.prepare(data)
        out = backend.execute(task, planner.make(task, scope=True))
        ref = backend.execute(task, planner.make(task))
        np.testing.assert_array_equal(out.values, ref.values)
        self.assertTrue(np.all(out.values[:, 4:12] == dynamic.INF))
        disconnected = [set(a) for a in data.adjacency]
        disconnected[0].remove(4)
        disconnected[4].remove(0)
        data = replace(data, adjacency=disconnected, old_distances=out.values,
                       updates=(("insert", 0, 4),))
        task = backend.prepare(data)
        np.testing.assert_array_equal(backend.execute(task, planner.make(task, scope=True)).values,
                                      backend.execute(task, planner.make(task)).values)


class LogicalTests(unittest.TestCase):
    def setUp(self):
        self.backend, self.planner = logical.LogicalAdapter(), Planner()
        self.data = logical.construct(seed=1, queries=20, dimension=5, cohorts=2)

    def test_merged_forward_backward_and_multistep_update(self):
        left, right = self.data, self.data
        for _ in range(5):
            tl, tr = self.backend.prepare(left), self.backend.prepare(right)
            ref = self.backend.execute(tl, self.planner.make(tl))
            out = self.backend.execute(tr, self.planner.make(tr, merge=True))
            self.assertTrue(self.backend.validate(out, ref)["passed"])
            left, right = logical.step(left, ref), logical.step(right, out)
            np.testing.assert_allclose(left.entities, right.entities, atol=1e-14)
            np.testing.assert_allclose(left.relations, right.relations, atol=1e-14)

    def test_gradient_against_finite_differences(self):
        task = self.backend.prepare(self.data)
        out = self.backend.execute(task, self.planner.make(task, merge=True))
        epsilon = 1e-5
        rng = np.random.default_rng(5)
        for key, gradient in (("entities", "grad_entities"), ("relations", "grad_relations")):
            direction = rng.normal(size=getattr(self.data, key).shape)
            direction /= np.linalg.norm(direction)
            loss = []
            for sign in (-1, 1):
                data = replace(self.data, **{key: getattr(self.data, key) + sign * epsilon * direction})
                task = self.backend.prepare(data)
                loss.append(self.backend.execute(task, self.planner.make(task)).values["loss"])
            numeric = (loss[1] - loss[0]) / (2 * epsilon)
            analytic = np.sum(out.values[gradient] * direction)
            self.assertAlmostEqual(numeric, analytic, delta=2e-8)

    def test_training_targets_follow_kg_set_semantics(self):
        adjacency = {}
        for h, r, t in self.data.triples:
            adjacency.setdefault((h, r), set()).add(t)
        for expr, positive, negatives in zip(self.data.expressions, self.data.positives, self.data.negatives):
            truth = logical.answers(expr, adjacency, frozenset(range(len(self.data.entities))))
            self.assertIn(int(positive), truth)
            self.assertFalse(set(map(int, negatives)) & truth)

    def test_incremental_combinations_and_parameter_invalidation(self):
        initial = self.backend.prepare(self.data, training=False)
        cache = self.backend.execute(initial, self.planner.make(initial)).values["cache"]
        changed = logical.change_inputs(self.data, 0.2)
        task = self.backend.prepare(changed, training=False, cache=cache)
        ref = self.backend.execute(task, self.planner.make(task))
        for scope, merge in ((True, False), (False, True), (True, True)):
            out = self.backend.execute(task, self.planner.make(task, scope=scope, merge=merge))
            self.assertTrue(self.backend.validate(out, ref)["passed"])
        changed = replace(changed, relations=changed.relations + 0.01)
        task = self.backend.prepare(changed, training=False, cache=cache)
        self.assertEqual(len(task.required), len(task.structure.units))
        with self.assertRaises(ValueError):
            self.backend.prepare(changed, training=True, cache=cache)


class VideoTests(unittest.TestCase):
    def test_full_block_attention_matches_direct_attention(self):
        data = video.construct()
        x = data.initial
        wq, wk, wv, wo = data.weights
        logits = (x @ wq) @ (x @ wk).T / np.sqrt(x.shape[1])
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        direct = (p @ (x @ wv)) @ wo
        grouped, _ = video.attention(x, data.weights, len(x) // data.config.block_size,
                                      data.config.block_size)
        np.testing.assert_allclose(grouped, direct, atol=1e-14)

    def test_quality_check_and_runtime_fallback(self):
        backend = video.VideoAdapter()
        runtime = Runtime()
        runtime.register(backend)
        data = video.construct(config=video.VideoConfig(quality_limit=1e-12))
        task = backend.prepare(data)
        reference = backend.execute(task, runtime.planner.make(task))
        approximate = runtime.planner.make(task, budgets=(0,) * data.config.steps)
        self.assertFalse(backend.validate(backend.execute(task, approximate), reference, data)["passed"])
        result, selected, record = runtime.checked(
            task, lambda out: backend.validate(out, reference, data), approximate)
        self.assertTrue(record["fallback"])
        self.assertEqual(selected.name, "reference")
        self.assertEqual(len(record["attempts"]), 2)


if __name__ == "__main__":
    unittest.main()
