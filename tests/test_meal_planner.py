"""Checks the hand-written linear algebra against NumPy and against each other.

Run with:  python -m unittest discover tests
"""
import unittest

import numpy as np

from meal_planner import linalg
from meal_planner.data import DEFAULT_FOODS, DEPENDENT_FOODS, STUDENTS
from meal_planner.planner import build_system, plan_for_students, plan_meal, target_warning


class TestLinearAlgebra(unittest.TestCase):
    def setUp(self):
        self.A, self.c = build_system(DEFAULT_FOODS)
        self.b = np.array(STUDENTS["A - regular student"][:3], dtype=float)

    def test_determinant_matches_numpy(self):
        rng = np.random.default_rng(0)
        for n in (1, 2, 3, 4):
            M = rng.normal(size=(n, n))
            self.assertAlmostEqual(linalg.det_cofactor(M), np.linalg.det(M))

    def test_gauss_jordan_solution_satisfies_system(self):
        result = linalg.solve(self.A, self.b)
        self.assertEqual(result["kind"], linalg.UNIQUE)
        np.testing.assert_allclose(self.A @ result["x"], self.b)
        np.testing.assert_allclose(result["x"], np.linalg.solve(self.A, self.b))

    def test_cramer_matches_gauss_jordan(self):
        np.testing.assert_allclose(linalg.cramer(self.A, self.b), linalg.solve(self.A, self.b)["x"])

    def test_inverse(self):
        np.testing.assert_allclose(self.A @ linalg.inverse(self.A), np.eye(3), atol=1e-12)

    def test_singular_has_no_inverse(self):
        A_dep, _ = build_system(DEPENDENT_FOODS)
        with self.assertRaises(ValueError):
            linalg.inverse(A_dep)

    def test_rank(self):
        self.assertEqual(linalg.rank(self.A), 3)
        self.assertEqual(linalg.rank(build_system(DEPENDENT_FOODS)[0]), 2)


class TestDependentFoods(unittest.TestCase):
    def setUp(self):
        self.A, _ = build_system(DEPENDENT_FOODS)

    def test_determinant_is_zero(self):
        self.assertAlmostEqual(linalg.det_cofactor(self.A), 0.0, places=6)

    def test_target_off_the_plane_has_no_solution(self):
        self.assertEqual(linalg.solve(self.A, [600, 25, 80])["kind"], linalg.NONE)

    def test_target_on_the_plane_has_infinitely_many(self):
        b = 2 * self.A[:, 0] + self.A[:, 1]
        self.assertEqual(linalg.solve(self.A, b)["kind"], linalg.INFINITE)

    def test_null_space_vector(self):
        np.testing.assert_allclose(self.A @ np.array([-1, -1, 1]), 0, atol=1e-12)


class TestPlanner(unittest.TestCase):
    def test_all_students(self):
        A, _, B, X, rows = plan_for_students(DEFAULT_FOODS, STUDENTS)
        np.testing.assert_allclose(A @ X, B)
        verdicts = {r["student"][0]: r["verdict"] for r in rows}
        self.assertEqual(verdicts, {"A": "OK", "B": "OK", "C": "OK",
                                    "D": "Not feasible (negative servings)"})

    def test_cost_is_dot_product(self):
        result = plan_meal(DEFAULT_FOODS, [600, 25, 80], budget=80)
        self.assertAlmostEqual(result["cost"], float(result["c"] @ result["x"]))
        self.assertEqual(result["verdict"], "OK")

    def test_impossible_target_warns(self):
        # 4 x (100 + 100) = 800 kcal > 700 kcal
        self.assertIn("Impossible target", target_warning((700, 100, 100)))
        self.assertIsNotNone(plan_meal(DEFAULT_FOODS, (700, 100, 100))["warning"])

    def test_negative_target_warns(self):
        self.assertIsNotNone(target_warning((600, -5, 80)))

    def test_realistic_targets_do_not_warn(self):
        for s, values in STUDENTS.items():
            self.assertIsNone(target_warning(values[:3]), s)
        self.assertIsNone(target_warning((700, 40, 80)))

    def test_every_food_passes_its_own_check(self):
        # A real food must never be flagged as impossible.
        from meal_planner.data import FOODS
        for name, values in FOODS.items():
            self.assertIsNone(target_warning(values[:3]), name)

    def test_unknown_food(self):
        with self.assertRaises(KeyError):
            build_system(["Pizza", "Egg (1 large)", "Milk (1 cup)"])


if __name__ == "__main__":
    unittest.main()
