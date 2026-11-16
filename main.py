"""Budget Meal Planner - command-line program.

    python main.py                 full demo report, figures saved to figures/
    python main.py --show          same, and also open the figure windows
    python main.py --interactive   choose your own foods, target and budget
"""
import argparse
import os
import sys

import matplotlib
import numpy as np

from meal_planner import linalg
from meal_planner.data import DEFAULT_FOODS, DEPENDENT_FOODS, FOODS, NUTRIENTS, STUDENTS
from meal_planner.planner import build_system, plan_for_students, plan_meal, target_warning

np.set_printoptions(precision=3, suppress=True)


def heading(text):
    print("\n" + "=" * 72 + f"\n{text}\n" + "=" * 72)


def print_plan(food_names, result):
    if result.get("warning"):
        print(f"WARNING: {result['warning']}\n")
    print(f"det(A) = {result['det']:,.2f}   rank(A) = {result['rank']}   "
          f"solutions: {result['kind']}")
    if result["kind"] != linalg.UNIQUE:
        return
    x, c = result["x"], result["c"]
    print(f"\n{'Food':<26}{'Servings':>10}{'Cost (₱)':>12}")
    for name, xi, ci in zip(food_names, x, c):
        print(f"{name:<26}{xi:>10.2f}{xi * ci:>12.2f}")
    print(f"\nTotal cost c·x = ₱{result['cost']:.2f}  ->  {result['verdict']}")


def demo(figure_dir, show):
    # 1. Formulation
    heading("1. Formulation: Ax = b  (columns of A are the food vectors)")
    A, c = build_system(DEFAULT_FOODS)
    name_a = "A - regular student"
    b = np.array(STUDENTS[name_a][:3], dtype=float)
    print("Foods:", ", ".join(DEFAULT_FOODS))
    print("A (rows = kcal, protein, carbs):\n", A)
    print("b (Student A's target):", b)
    print("c (price per serving): ", c)

    # 2. Determinant
    heading("2. Unique plan?  det(A) by cofactor expansion")
    d = linalg.det_cofactor(A)
    print(f"det(A) = {d:,.2f}  (numpy check: {np.linalg.det(A):,.2f})")
    print("det(A) != 0, so the foods are linearly independent and the plan is unique.")

    # 3. Gauss-Jordan
    heading("3. Gauss-Jordan elimination on [A | b]")
    result = plan_meal(DEFAULT_FOODS, b, budget=STUDENTS[name_a][3])
    print("\n".join(result["steps"]))
    print("\nRREF =\n", result["rref"])
    print()
    print_plan(DEFAULT_FOODS, result)

    # 4. Cramer's rule
    heading("4. Check with Cramer's rule")
    x_cramer = linalg.cramer(A, b)
    print("Gauss-Jordan: ", result["x"])
    print("Cramer's rule:", x_cramer)
    print("Same answer:", np.allclose(result["x"], x_cramer))

    # 5. All students
    heading("5. Every student at once: X = A^-1 B")
    A, A_inv, B, X, rows = plan_for_students(DEFAULT_FOODS, STUDENTS)
    print("A^-1 =\n", A_inv)
    print("A A^-1 = I:", np.allclose(A @ A_inv, np.eye(3)))
    short = ["Rice", "PB", "Chicken"]
    print(f"\n{'Student':<24}" + "".join(f"{s:>9}" for s in short)
          + f"{'Cost':>9}{'Budget':>8}  Verdict")
    for row in rows:
        print(f"{row['student']:<24}" + "".join(f"{v:>9.2f}" for v in row["x"])
              + f"{row['cost']:>9.2f}{row['budget']:>8}  {row['verdict']}")

    # 6. Dependent foods
    heading("6. Linearly dependent foods: combo = rice + chicken")
    A_dep, c_dep = build_system(DEPENDENT_FOODS)
    print("A_dep =\n", A_dep)
    print(f"det(A_dep) = {linalg.det_cofactor(A_dep):.6f}   rank = {linalg.rank(A_dep)}")
    on_plane = 2 * A_dep[:, 0] + A_dep[:, 1]
    for label, target in [("Student A's target", b), ("2 rice + 1 chicken", on_plane)]:
        r = linalg.solve(A_dep, target)
        print(f"\n{label}: b = {target}")
        print("RREF of [A | b] =\n", r["rref"])
        print("Solutions:", r["kind"])
    print("\nOn-plane target, x = (2, 1, 0) + t(-1, -1, 1)  [null space direction]:")
    for t in (0, 0.5, 1):
        xt = np.array([2, 1, 0]) + t * np.array([-1, -1, 1])
        print(f"  t = {t:<3}  servings = {xt}  Ax = {A_dep @ xt}  cost = ₱{c_dep @ xt:.2f}")

    # Figures
    from meal_planner import plots
    import matplotlib.pyplot as plt
    os.makedirs(figure_dir, exist_ok=True)
    f1 = plots.linear_combination_chart(A, X, B, DEFAULT_FOODS, list(STUDENTS))
    f2 = plots.span_plane_chart(A_dep, DEPENDENT_FOODS, b)
    p1 = os.path.join(figure_dir, "figure1_linear_combination.png")
    p2 = os.path.join(figure_dir, "figure2_span_plane.png")
    f1.savefig(p1, dpi=150)
    f2.savefig(p2, dpi=150)
    heading("Figures")
    print(f"Saved {p1}\nSaved {p2}")
    if show:
        plt.show()


def ask_number(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  Please enter a number.")


def interactive():
    names = list(FOODS)
    heading("Budget Meal Planner - choose 3 foods")
    for i, n in enumerate(names, 1):
        kcal, p, carb, cost = FOODS[n]
        print(f"{i:>2}. {n:<26} {kcal:>5} kcal {p:>5} g protein {carb:>5} g carbs  ₱{cost}")
    while True:
        picks = input("\nEnter 3 numbers separated by spaces (e.g. 1 4 5): ").split()
        try:
            idx = [int(p) - 1 for p in picks]
            if len(idx) == 3 and len(set(idx)) == 3 and all(0 <= i < len(names) for i in idx):
                break
        except ValueError:
            pass
        print("  Please enter 3 different numbers from the list.")
    chosen = [names[i] for i in idx]
    while True:
        target = [ask_number(f"Target {n}: ") for n in NUTRIENTS]
        warning = target_warning(target)
        if warning is None:
            break
        print(f"\n  WARNING: {warning}")
        if input("  Enter a different target? (y/n): ").strip().lower() not in ("y", "yes"):
            break
        print()
    budget = ask_number("Budget (₱): ")
    heading("Your meal plan")
    print_plan(chosen, plan_meal(chosen, target, budget))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # for the ₱ sign on Windows
    parser = argparse.ArgumentParser(description="Plan a meal by solving Ax = b.")
    parser.add_argument("--interactive", action="store_true", help="pick your own foods and target")
    parser.add_argument("--show", action="store_true", help="open the figure windows")
    parser.add_argument("--figures", default="figures", help="folder for saved figures")
    args = parser.parse_args()
    if args.interactive:
        interactive()
        return
    if not args.show:
        matplotlib.use("Agg")  # save figures without opening windows
    demo(args.figures, args.show)


if __name__ == "__main__":
    main()
