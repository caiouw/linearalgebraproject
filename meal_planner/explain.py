"""Plain-language explanations of each result, for people who don't know linear algebra."""
import textwrap

from .data import NUTRIENTS


def wrap(text, indent="  "):
    """Wrap a paragraph so it fits the terminal."""
    return textwrap.fill(" ".join(text.split()), width=76,
                         initial_indent=indent, subsequent_indent=indent)


def plain_words(text):
    """Print an 'In plain words' note under a maths step."""
    print(wrap("In plain words: " + text, indent="  | "))


def equations(A, b, food_names):
    """Write Ax = b out as three everyday equations, one per nutrient."""
    lines = ["Unknowns (servings to find):"]
    for j in range(3):
        lines.append(f"  x{j+1} = servings of {food_names[j]}")
    lines.append("")
    lines.append("One equation per nutrient:")
    for i in range(3):
        left = f"{A[i, 0]:g}*x1 + {A[i, 1]:g}*x2 + {A[i, 2]:g}*x3"
        lines.append(f"  {left} = {b[i]:g}   <- {NUTRIENTS[i]}")
    return "\n".join(lines)


def story(food_names, result):
    """Explain a plan_meal() result in plain language."""
    if result["kind"] == "none":
        return "\n\n".join([
            wrap(f"Your foods overlap: one of them is just a mix of the others "
                 f"(det(A) = 0, rank(A) = {result['rank']}), so only {result['rank']} "
                 f"of your 3 foods are truly different."),
            wrap("These foods can only reach some nutrient totals (their span is a plane), "
                 "and your target is not one of them. There is NO plan. Swap one food."),
        ])

    if result["kind"] == "infinitely many":
        return "\n\n".join([
            wrap(f"Your foods overlap: one of them is just a mix of the others "
                 f"(det(A) = 0, rank(A) = {result['rank']})."),
            wrap("Your target can still be reached, but in MANY different ways, because "
                 "the mixed food can be traded for its ingredients. Pick the cheapest one, "
                 "or swap a food to get a single answer."),
        ])

    parts = [wrap(f"Your 3 foods are different enough from each other (det(A) = "
                  f"{result['det']:,.0f}, which is not 0), so there is exactly ONE "
                  f"combination of servings that hits your target.")]

    x, c = result["x"], result["c"]
    if result["verdict"] == "Not feasible (negative servings)":
        for name, servings in zip(food_names, x):
            if servings < 0:
                parts.append(wrap(
                    f"That combination needs {servings:.2f} servings of {name}. A negative "
                    f"amount means taking that food AWAY from the meal, which is impossible."))
        parts.append(wrap("Result: NOT POSSIBLE with these foods. Swap a food or change "
                          "your target."))
        return "\n\n".join(parts)

    meal = "  Your meal:\n"
    for name, servings, price in zip(food_names, x, c):
        meal += f"    {servings:5.2f} x {name:<26} ₱{servings * price:6.2f}\n"
    meal += f"            {'Total cost':<26} ₱{result['cost']:6.2f}"
    parts.append(meal)

    budget = result["budget"]
    if result["verdict"] == "Over budget":
        parts.append(wrap(f"Result: the meal works, but it costs ₱{result['cost'] - budget:.2f} "
                          f"more than your ₱{budget:g} budget. Try cheaper foods."))
    elif budget is not None:
        parts.append(wrap(f"Result: this meal hits your target exactly and fits your "
                          f"₱{budget:g} budget."))
    else:
        parts.append(wrap("Result: this meal hits your target exactly."))

    check = result["A"] @ x
    parts.append(wrap(f"Check (A times x gives back the target): {check[0]:.0f} kcal, "
                      f"{check[1]:.0f} g protein, {check[2]:.0f} g carbs"))
    return "\n\n".join(parts)
