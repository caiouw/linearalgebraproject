"""Turns the food data into Ax = b and judges the meal plan."""
import numpy as np

from . import linalg
from .data import FOODS


def build_system(food_names):
    """A has one column per food (its nutrient vector); c holds the prices."""
    for name in food_names:
        if name not in FOODS:
            raise KeyError(f"Unknown food: {name}")
    A = np.array([FOODS[name][:3] for name in food_names], dtype=float).T
    c = np.array([FOODS[name][3] for name in food_names], dtype=float)
    return A, c


def target_warning(target):
    """Warn if no real foods could ever reach this target.

    Protein and carbs each give about 4 kcal per gram, so a meal has at least
    4 x (protein + carbs) kcal. We allow 10% because labels round and fibre
    counts as carbs but gives fewer kcal.
    """
    kcal, protein, carbs = target
    if kcal < 0 or protein < 0 or carbs < 0:
        return "Targets cannot be negative."
    minimum = 4 * (protein + carbs)
    if minimum > 1.10 * kcal:
        return (f"Impossible target: {protein:g} g protein + {carbs:g} g carbs already give "
                f"4 x ({protein:g} + {carbs:g}) = {minimum:.0f} kcal, more than your "
                f"{kcal:g} kcal target. No combination of real foods can do this. "
                f"Raise calories to at least {minimum:.0f} kcal or lower protein/carbs.")
    return None


def verdict(x, cost, budget):
    """Can the plan actually be eaten, and is it affordable?"""
    if np.any(x < -linalg.TOL):
        return "Not feasible (negative servings)"
    if budget is not None and cost > budget:
        return "Over budget"
    return "OK"


def plan_meal(food_names, target, budget=None):
    """Plan one meal: solve Ax = b, then work out the cost c . x."""
    A, c = build_system(food_names)
    b = np.array(target, dtype=float)
    result = linalg.solve(A, b)
    result.update(A=A, b=b, c=c, budget=budget,
                  det=linalg.det3(A), warning=target_warning(target))
    if result["kind"] == "unique":
        result["cost"] = c @ result["x"]
        result["verdict"] = verdict(result["x"], result["cost"], budget)
    else:
        result["cost"] = None
        result["verdict"] = "No unique plan"
    return result


def plan_for_students(food_names, students):
    """Plan for every student at once: X = A^-1 B, one column of B per student."""
    A, c = build_system(food_names)
    names = list(students)
    B = np.array([students[s][:3] for s in names], dtype=float).T
    A_inv = linalg.inverse(A)
    X = A_inv @ B

    rows = []
    for k, name in enumerate(names):
        x = X[:, k]
        cost = c @ x
        budget = students[name][3]
        rows.append({"student": name, "x": x, "cost": cost, "budget": budget,
                     "verdict": verdict(x, cost, budget)})
    return A, A_inv, B, X, rows
