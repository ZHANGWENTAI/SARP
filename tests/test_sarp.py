from dataclasses import replace
import unittest

import numpy as np

from structure_aware.sarp import Graph, Node, Operation, SARP, Session
from structure_aware.adapters import spreadsheet, image_pipeline, dynamic, video


class ToyAdapter:
    operations = {
        "square": Operation(lambda x: x * x, pure=True),
        "add": Operation(lambda x, amount: x + amount, pure=True),
    }

    def build(self, request):
        return Graph("unseen-domain", {"x": request}, (
            Node("out", "add", ("left",), {"amount": 3}, 100),
            Node("left", "square", ("x",), work=100),
            Node("right", "square", ("x",), work=100),
        ), ("out", "right"))


class SARPTests(unittest.TestCase):
    def test_unseen_adapter_without_executor_and_unordered_nodes(self):
        result = SARP().run(ToyAdapter(), 4)
        self.assertEqual(result.outputs, {"out": 19, "right": 16})
        self.assertEqual(result.metrics["kernel_calls"], 2)

    def test_no_change_reuses_cache_and_inplace_input_change_is_detected(self):
        runtime, cache, adapter = SARP(), Session(), ToyAdapter()
        value = np.array([2., 3.])
        runtime.run(adapter, value, session=cache)
        unchanged = runtime.run(adapter, value, session=cache)
        self.assertEqual(unchanged.metrics["kernel_calls"], 0)
        value += 1
        updated = runtime.run(adapter, value, session=cache)
        np.testing.assert_array_equal(updated.outputs["right"], value * value)
        with self.assertRaises(ValueError):
            updated.outputs["right"][0] = 999

    def test_parameters_operation_version_and_namespace_invalidate_cache(self):
        adapter, runtime, cache = ToyAdapter(), SARP(), Session()
        runtime.run(adapter, 4, session=cache)
        base = adapter.build(4)
        class Other:
            operations = dict(adapter.operations)
            def build(self, request):
                return replace(base, nodes=(replace(base.nodes[0], parameters={"amount": 10}),
                                             *base.nodes[1:]))
        other = Other()
        self.assertEqual(runtime.run(other, None, session=cache).outputs["out"], 26)
        other.operations["square"] = Operation(lambda x: x ** 3, pure=True, version="2")
        self.assertEqual(runtime.run(other, None, session=cache).outputs["out"], 74)
        other.build = lambda _: replace(base, namespace="another-task")
        self.assertGreater(runtime.run(other, None, session=cache).metrics["kernel_calls"], 0)

    def test_effects_are_not_pruned_merged_or_cached(self):
        calls = []
        class Adapter:
            operations = {"effect": Operation(lambda: calls.append(1) or len(calls))}
            def build(self, request):
                return Graph("effects", {"x": 1}, (Node("event", "effect", work=100),), ("x",))
        runtime, cache = SARP(), Session()
        for _ in range(2):
            runtime.run(Adapter(), None, session=cache)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("event", cache.values)

    def test_dead_pure_nodes_pruned_but_newly_requested_output_is_calculated(self):
        class Adapter(ToyAdapter):
            def build(self, request):
                graph = super().build(request)
                return replace(graph, nodes=graph.nodes + (
                    Node("dead", "add", ("x",), {"amount": 99}, 10000),),
                    outputs=("out",) if request == 4 else ("out", "dead"))
        runtime, cache, adapter = SARP(), Session(), Adapter()
        first = runtime.run(adapter, 4, session=cache)
        self.assertNotIn("dead", cache.values)
        self.assertLess(first.metrics["active_units"], first.metrics["total_units"])
        self.assertEqual(runtime.run(adapter, 5, session=cache).outputs["dead"], 104)

    def test_unknown_dependencies_and_cycles_are_rejected(self):
        adapter = ToyAdapter()
        for nodes in ((Node("a", "square", ("missing",)),),
                      (Node("a", "square", ("b",)), Node("b", "square", ("a",)))):
            adapter.build = lambda _, nodes=nodes: Graph("invalid", {}, nodes, ("a",))
            with self.assertRaises(ValueError):
                SARP().run(adapter, None)

    def test_failed_validation_falls_back_and_failed_reference_does_not_commit(self):
        runtime, cache, calls = SARP(), Session(), []
        def validator(out):
            calls.append(out)
            return {"passed": len(calls) > 1}
        result = runtime.run(ToyAdapter(), 4, session=cache, validator=validator)
        self.assertTrue(result.validation["fallback"])
        self.assertEqual(result.plan.name, "reference")
        before = dict(cache.signatures)
        with self.assertRaises(RuntimeError):
            runtime.run(ToyAdapter(), 5, session=cache, validator=lambda _: {"passed": False})
        self.assertEqual(cache.signatures, before)

    def test_shared_ancestor_does_not_merge_different_input_sources(self):
        class Adapter(ToyAdapter):
            def build(self, request):
                return Graph("different-inputs", {"x": 2, "y": 3}, (
                    Node("a", "square", ("x",), work=100),
                    Node("b", "square", ("y",), work=100)), ("a", "b"))
        result = SARP().run(Adapter(), None)
        self.assertEqual(result.outputs, {"a": 4, "b": 9})

    def test_effects_are_not_replayed_on_validation_failure(self):
        calls = []
        class Adapter:
            operations = {"effect": Operation(lambda: calls.append(1) or 1)}
            def build(self, request):
                return Graph("effect-check", {}, (Node("out", "effect"),), ("out",))
        with self.assertRaises(RuntimeError):
            SARP().run(Adapter(), None, validator=lambda _: {"passed": False})
        self.assertEqual(calls, [1])

    def test_original_cyclic_and_budget_backends_use_the_same_runtime(self):
        runtime = SARP()
        backend = dynamic.DynamicAdapter()
        runtime.register_backend(backend)
        task = backend.prepare(dynamic.construct(7, blocks=4, block_size=8, touched=1))
        out, _ = runtime.execute_task(task)
        reference = backend.execute(task, runtime.runtime.planner.make(task))
        np.testing.assert_array_equal(out.values, reference.values)
        backend = video.VideoAdapter()
        runtime.register_backend(backend)
        task = backend.prepare(video.construct())
        out, plan = runtime.execute_task(task)
        self.assertEqual(len(plan.budgets), 12)
        self.assertEqual(out.values["latent"].shape, (64, 12))


class GenericAdapterTests(unittest.TestCase):
    def test_workbook_modes_and_incremental_update_against_direct_formula(self):
        adapter, runtime = spreadsheet.SpreadsheetAdapter(), SARP()
        book = spreadsheet.construct(9, regions=3, rows=128, views=3)
        cache = Session()
        runtime.run(adapter, book, session=cache)
        updated = spreadsheet.changed(book, 1 / 3)
        expected = spreadsheet.expected(updated)
        for mode in ("reference", "scope", "merge", "combined", "auto"):
            out = runtime.run(adapter, updated, mode=mode, session=cache.clone())
            self.assertEqual(out.outputs, expected)
        changed = runtime.run(adapter, updated, mode="combined", session=cache.clone())
        reference = runtime.run(adapter, updated, mode="reference")
        self.assertLess(changed.metrics["kernel_calls"], reference.metrics["kernel_calls"])

    def test_forward_formula_references_and_syntax_rejection(self):
        adapter = spreadsheet.SpreadsheetAdapter()
        book = spreadsheet.Workbook({"x": 3}, {"a": "b + 1", "b": "x * 2"}, ("a",))
        self.assertEqual(SARP().run(adapter, book).outputs, {"a": 7})
        for formula in ("__import__('os')", "x.__class__", "a + 1"):
            with self.assertRaises(ValueError):
                SARP().run(adapter, spreadsheet.Workbook({"x": 3}, {"a": formula}, ("a",)))

    def test_image_modes_and_change_against_direct_pipeline(self):
        runtime, adapter = SARP(), image_pipeline.ImagePipelineAdapter()
        data = image_pipeline.construct(3, images=3, size=16)
        cache = Session()
        runtime.run(adapter, data, session=cache)
        updated = image_pipeline.changed(data, 1 / 3)
        expected = image_pipeline.expected(updated)
        for mode in ("reference", "scope", "merge", "combined", "auto"):
            result = runtime.run(adapter, updated, session=cache.clone(), mode=mode)
            for name in expected:
                np.testing.assert_array_equal(result.outputs[name], expected[name])

    def test_gaussian_kernel_against_independent_nested_convolution(self):
        image = np.arange(20, dtype=float).reshape(4, 5)
        weights = np.outer([1, 4, 6, 4, 1], [1, 4, 6, 4, 1]) / 256
        padded = np.pad(image, 2, mode="edge")
        expected = np.empty_like(image)
        for i in range(4):
            for j in range(5):
                expected[i, j] = np.sum(padded[i:i + 5, j:j + 5] * weights)
        np.testing.assert_array_equal(image_pipeline.gaussian(image, 1), expected)

    def test_no_redundancy_full_change_chooses_reference(self):
        runtime, adapter = SARP(), image_pipeline.ImagePipelineAdapter()
        data = image_pipeline.construct(images=1, size=16, duplicate=False)
        cache = Session()
        runtime.run(adapter, data, session=cache)
        result = runtime.run(adapter, image_pipeline.changed(data, 1), session=cache)
        self.assertEqual(result.plan.name, "reference")


if __name__ == "__main__":
    unittest.main()
