"""Plain-language explanations of each result, for people who don't know linear algebra."""
import textwrap

import numpy as np

from . import linalg
from .data import NUTRIENTS

UNITS = ["kcal", "g protein", "g carbs"]


def wrap(text, indent="  "):
    """Wrap a paragraph to the terminal width."""
    return textwrap.fill(" ".join(text.split()), width=76,
                         initial_indent=indent, subsequent_indent=indent)


def plain_words(text):
    """An 'In plain words' note printed under a maths step."""
    print(wrap("In plain words: " + text, indent="  | "))


def equations(A, b, food_names):
    """The system Ax = b written out as three everyday equations."""
    lines = ["Unknowns (servings to find):"]
    lines += [f"  x{j+1} = servings of {name}" for j, name in enumerate(food_names)]
    lines.append("")
    lines.append("One equation per nutrient (row of A times x = target):")
    for i, nutrient in enumerate(NUTRIENTS):
        terms = " + ".join(f"{A[i, j]:g}*x{j+1}" for j in range(A.shape[1]))
        lines.append(f"  {terms} = {b[i]:g}   <- {nutrient}")
    return "\n".join(lines)


def story(food_names, result):
    """Explain what the plan_meal() result means, in plain language."""
    out = []
    kind, rank, det = result["kind"], result["rank"], result["det"]

    if kind != linalg.UNIQUE:
        out.append(wrap(
            f"Your foods overlap: one of them is just a mix of the others, so it adds nothing "
            f"new. Maths check: det(A) = 0 and rank(A) = {rank}, meaning only {rank} of your "
            f"3 foods are truly different."))
        if kind == linalg.NONE:
            out.append(wrap(
                "Because of that, these foods can only reach some nutrient totals (in maths: "
                "their span is a plane, not all of 3D space), and your target is not one of "
                "them. There is NO plan. Swap one food for a different one."))
        else:
            out.append(wrap(
                "Your target happens to be reachable, but in MANY different ways, because you "
                "can trade the 'mixed' food for its ingredients without changing the nutrients "
                "(in maths: the null space is not just zero). Pick the cheapest one, or swap a "
                "food to get a single answer."))
        return "\n\n".join(out)

    out.append(wrap(
        f"Your 3 foods are different enough from each other (det(A) = {det:,.0f}, which is "
        f"not 0), so there is exactly ONE combination of servings that hits your target."))

    x, c = result["x"], result["c"]
    negative = [(name, xi) for name, xi in zip(food_names, x) if xi < -linalg.TOL]
    if negative:
        for name, xi in negative:
            out.append(wrap(
                f"That combination needs {xi:.2f} servings of {name}. A negative amount means "
                f"you would have to take that food AWAY from the meal, which is impossible."))
        out.append(wrap(
            "Result: NOT POSSIBLE with these foods. Try swapping "
            + " / ".join(n for n, _ in negative)
            + " for a different food, or change your target."))
        return "\n\n".join(out)

    meal = ["  Your meal:"]
    for name, xi, ci in zip(food_names, x, c):
        meal.append(f"    {xi:5.2f} x {name:<26} ₱{xi * ci:6.2f}")
    meal.append(f"    {'':5}   {'Total cost':<26} ₱{result['cost']:6.2f}")
    out.append("\n".join(meal))

    budget = result.get("budget")
    if result["verdict"] == "Over budget":
        out.append(wrap(
            f"Result: the meal works, but it costs ₱{result['cost'] - budget:.2f} more than "
            f"your ₱{budget:g} budget. Try cheaper foods."))
    else:
        tail = f" and fits your ₱{budget:g} budget" if budget is not None else ""
        out.append(wrap(f"Result: this meal hits your target exactly{tail}."))
    check = result["A"] @ x
    out.append(wrap("Check (A times x gives back the target): "
                    + ", ".join(f"{v:.0f} {u}" for v, u in zip(check, UNITS))))
    return "\n\n".join(out)
