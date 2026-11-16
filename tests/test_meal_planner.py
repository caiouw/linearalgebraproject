"""Checks the hand-written linear algebra against NumPy and against each other.

Run with:  py -m unittest discover tests
"""
import unittest

import numpy as np

from meal_planner import explain, linalg
from meal_planner.data import DEFAULT_FOODS, DEPENDENT_FOODS, FOODS, STUDENTS
from meal_planner.planner import build_system, plan_for_students, plan_meal, target_warning

A, c = build_system(DEFAULT_FOODS)
A_DEP, _ = build_system(DEPENDENT_FOODS)
B_STUDENT_A = np.array([600, 25, 80], dtype=float)


class TestLinearAlgebra(unittest.TestCase):
    def test_det3_matches_numpy(self):
        self.assertAlmostEqual(linalg.det3(A), np.linalg.det(A))
        M = np.array([[1, 2, 3], [2, 1, 3], [3, 1, 2]])  # the solved example in Lesson 3
        self.assertAlmostEqual(linalg.det3(M), np.linalg.det(M))

    def test_solution_satisfies_system(self):
        result = linalg.solve(A, B_STUDENT_A)
        self.assertEqual(result["kind"], "unique")
        np.testing.assert_allclose(A @ result["x"], B_STUDENT_A)
        np.testing.assert_allclose(result["x"], np.linalg.solve(A, B_STUDENT_A))

    def test_cramer_matches_gauss_jordan(self):
        np.testing.assert_allclose(linalg.cramer(A, B_STUDENT_A),
                                   linalg.solve(A, B_STUDENT_A)["x"])

    def test_inverse(self):
        np.testing.assert_allclose(A @ linalg.inverse(A), np.eye(3), atol=1e-12)

    def test_rank(self):
        self.assertEqual(linalg.rank(A), 3)
        self.assertEqual(linalg.rank(A_DEP), 2)


class TestDependentFoods(unittest.TestCase):
    def test_determinant_is_zero(self):
        self.assertAlmostEqual(linalg.det3(A_DEP), 0.0, places=6)

    def test_no_inverse(self):
        with self.assertRaises(ValueError):
            linalg.inverse(A_DEP)

    def test_target_off_the_plane_has_no_solution(self):
        self.assertEqual(linalg.solve(A_DEP, B_STUDENT_A)["kind"], "none")

    def test_target_on_the_plane_has_infinitely_many(self):
        b = 2 * A_DEP[:, 0] + A_DEP[:, 1]
        self.assertEqual(linalg.solve(A_DEP, b)["kind"], "infinitely many")

    def test_null_space_vector(self):
        np.testing.assert_allclose(A_DEP @ np.array([-1, -1, 1]), 0, atol=1e-12)


class TestPlanner(unittest.TestCase):
    def test_all_students(self):
        A_, _, B, X, rows = plan_for_students(DEFAULT_FOODS, STUDENTS)
        np.testing.assert_allclose(A_ @ X, B)
        verdicts = [row["verdict"] for row in rows]
        self.assertEqual(verdicts, ["OK", "OK", "OK", "Not feasible (negative servings)"])

    def test_cost_is_dot_product(self):
        result = plan_meal(DEFAULT_FOODS, (600, 25, 80), budget=80)
        self.assertAlmostEqual(result["cost"], c @ result["x"])
        self.assertEqual(result["verdict"], "OK")

    def test_unknown_food(self):
        with self.assertRaises(KeyError):
            build_system(["Pizza", "Egg (1 large)", "Milk (1 cup)"])


class TestTargetWarning(unittest.TestCase):
    def test_impossible_target(self):
        self.assertIn("Impossible target", target_warning((700, 100, 100)))

    def test_negative_target(self):
        self.assertIsNotNone(target_warning((600, -5, 80)))

    def test_realistic_targets(self):
        for values in STUDENTS.values():
            self.assertIsNone(target_warning(values[:3]))

    def test_no_real_food_is_flagged(self):
        for values in FOODS.values():
            self.assertIsNone(target_warning(values[:3]))


class TestPlainLanguage(unittest.TestCase):
    def story(self, foods, target, budget):
        return explain.story(foods, plan_meal(foods, target, budget))

    def test_good_plan(self):
        self.assertIn("fits your ₱80 budget", self.story(DEFAULT_FOODS, (600, 25, 80), 80))

    def test_negative_servings(self):
        self.assertIn("NOT POSSIBLE", self.story(DEFAULT_FOODS, (700, 10, 140), 60))

    def test_over_budget(self):
        self.assertIn("more than your ₱30 budget", self.story(DEFAULT_FOODS, (900, 45, 110), 30))

    def test_dependent_no_plan(self):
        self.assertIn("There is NO plan", self.story(DEPENDENT_FOODS, (600, 25, 80), 80))

    def test_dependent_many_plans(self):
        self.assertIn("MANY different ways", self.story(DEPENDENT_FOODS, (575, 39.6, 89), 80))


if __name__ == "__main__":
    unittest.main()
