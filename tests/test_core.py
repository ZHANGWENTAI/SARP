from dataclasses import replace
import unittest
from structure_aware.core import (BudgetDomain, Planner, Runtime, Structure, Task,
                                  Unit, allocate_budget)


class CoreTests(unittest.TestCase):
    def test_dependency_invalidation_handles_cycles(self):
        graph = Structure((Unit(0, "x", (1,)), Unit(1, "x", (0,)), Unit(2, "x")), cyclic=True)
        reached, _ = graph.affected([0])
        self.assertEqual(reached, frozenset([0, 1]))

    def test_scope_and_merge_are_composable(self):
        graph = Structure((Unit(0, "leaf", parameters=(1,), shareable=True),
                           Unit(1, "leaf", parameters=(1,), shareable=True),
                           Unit(2, "project", (0,), work=100, shareable=True),
                           Unit(3, "project", (1,), work=100, shareable=True),
                           Unit(4, "leaf", parameters=(2,), shareable=True)))
        task = Task("test", "test", graph, required=frozenset([0, 1, 2, 3]),
                    range_certificate="test", allow_merge=True)
        plan = Planner().make(task, scope=True, merge=True)
        self.assertEqual(plan.groups, ((0, 1), (2, 3)))
        self.assertEqual(set(plan.active), {0, 1, 2, 3})
        with self.assertRaises(ValueError):
            Planner().validate(task, replace(plan, groups=((0, 2), (1, 3))))

    def test_fallback_when_transform_overhead_exceeds_gain(self):
        graph = Structure((Unit(0, "x"), Unit(1, "x")))
        task = Task("small", "test", graph, required=frozenset([0]),
                    range_certificate="test", analysis_work=100)
        self.assertEqual(Planner().select(task)[0].name, "reference")

    def test_budget_conservation_bounds_and_determinism(self):
        domain = BudgetDomain(1, 4, (1.0, 10.0, 2.0), (1.,) * 3, (2.,) * 3)
        for total in range(3, 13):
            levels = allocate_budget(domain, total)
            self.assertEqual(sum(levels), total)
            self.assertTrue(all(1 <= b <= 4 for b in levels))
            self.assertEqual(levels, allocate_budget(domain, total))
        self.assertGreaterEqual(allocate_budget(domain, 7)[1], allocate_budget(domain, 7)[0])
        with self.assertRaises(ValueError):
            allocate_budget(domain, 13)

    def test_exact_tasks_reject_approximate_budgets(self):
        task = Task("exact", "test", Structure((Unit(0, "x"),)))
        with self.assertRaises(ValueError):
            Planner().make(task, budgets=(1,))

    def test_candidate_plans_share_structural_analysis(self):
        graph = Structure((Unit(0, "leaf", parameters=(1,), shareable=True),
                           Unit(1, "leaf", parameters=(1,), shareable=True)))
        task = Task("shared", "test", graph, required=frozenset([0]),
                    range_certificate="test", allow_merge=True)
        planner = Planner()
        choices = planner.candidates(task)
        identity = graph._identity
        planner.validate(task, choices[-1])
        self.assertIs(identity, graph._identity)
        self.assertEqual(len(graph._group_cache), 4)

    def test_tampered_budget_rejected_by_runtime_validation(self):
        domain = BudgetDomain(0, 2, (1.,), (1.,), (1.,))
        task = Task("budget", "test", Structure((Unit(0, "x"),)), budget=domain)
        plan = Planner().make(task)
        with self.assertRaises(ValueError):
            Planner().validate(task, replace(plan, budgets=(3,)))

    def test_unknown_or_missing_dependency_rejected(self):
        with self.assertRaises(ValueError):
            Structure((Unit(0, "x", (100,)),))
        with self.assertRaises(ValueError):
            Structure((Unit(0, "x", (1,)), Unit(1, "x")))


if __name__ == "__main__":
    unittest.main()
