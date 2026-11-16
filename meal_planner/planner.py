"""Turns food data into the system Ax = b and judges the resulting meal plan."""
import numpy as np

from . import linalg
from .data import FOODS


def build_system(food_names, foods=FOODS):
    """Columns of A are the food nutrient vectors a_j; c is the price vector."""
    missing = [f for f in food_names if f not in foods]
    if missing:
        raise KeyError(f"Unknown food(s): {', '.join(missing)}")
    A = np.array([foods[f][:3] for f in food_names], dtype=float).T  # foods -> columns
    c = np.array([foods[f][3] for f in food_names], dtype=float)
    return A, c


KCAL_PER_GRAM = 4      # protein and carbohydrate each give about 4 kcal per gram
LABEL_ALLOWANCE = 1.10 # food labels round, and fibre counts as carbs but gives fewer kcal


def target_warning(target):
    """Return a warning if no real food could ever meet this target, else None.

    Protein and carbs alone supply 4(P + C) kcal, and fat can only add more, so a
    target whose calories are below 4(P + C) is impossible whatever foods are chosen.
    """
    kcal, protein, carbs = (float(v) for v in target)
    if min(kcal, protein, carbs) < 0:
        return "Targets cannot be negative."
    minimum = KCAL_PER_GRAM * (protein + carbs)
    if minimum > LABEL_ALLOWANCE * kcal:
        return (f"Impossible target: {protein:g} g protein + {carbs:g} g carbs already give "
                f"4 x ({protein:g} + {carbs:g}) = {minimum:.0f} kcal, more than your "
                f"{kcal:g} kcal target. No combination of real foods can do this. "
                f"Raise calories to at least {minimum:.0f} kcal or lower protein/carbs.")
    return None


def judge(x, c, budget=None):
    """Cost is the dot product c . x; a plan with a negative serving cannot be eaten."""
    cost = float(c @ x)
    if np.any(x < -linalg.TOL):
        verdict = "Not feasible (negative servings)"
    elif budget is not None and cost > budget + linalg.TOL:
        verdict = "Over budget"
    else:
        verdict = "OK"
    return cost, verdict


def plan_meal(food_names, target, budget=None, foods=FOODS):
    """Plan one meal. Returns a dict with det, the solve() result, cost and verdict."""
    A, c = build_system(food_names, foods)
    b = np.asarray(target, dtype=float)
    result = linalg.solve(A, b)
    result["det"] = linalg.det_cofactor(A)
    result["A"], result["b"], result["c"] = A, b, c
    result["warning"] = target_warning(b)
    result["budget"] = budget
    if result["kind"] == linalg.UNIQUE:
        result["cost"], result["verdict"] = judge(result["x"], c, budget)
    else:
        result["cost"], result["verdict"] = None, "No unique plan"
    return result


def plan_for_students(food_names, students, foods=FOODS):
    """Plan for every student at once: X = A^-1 B, one column of B per student."""
    A, c = build_system(food_names, foods)
    names = list(students)
    B = np.array([students[s][:3] for s in names], dtype=float).T
    A_inv = linalg.inverse(A)
    X = A_inv @ B
    rows = []
    for k, s in enumerate(names):
        cost, verdict = judge(X[:, k], c, students[s][3])
        rows.append({"student": s, "x": X[:, k], "cost": cost,
                     "budget": students[s][3], "verdict": verdict})
    return A, A_inv, B, X, rows
