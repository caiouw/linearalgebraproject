"""Budget Meal Planner - finds how many servings of 3 foods hit a nutrition target.

    py main.py                  menu: plan your own meal, or see the full walkthrough
    py main.py --plan           go straight to planning your own meal
    py main.py --report         full linear algebra walkthrough; figures saved to figures/
    py main.py --report --show  same, and open the figure windows
"""
import argparse
import os
import sys

import matplotlib
import numpy as np

from meal_planner import explain, linalg
from meal_planner.data import DEFAULT_FOODS, DEPENDENT_FOODS, FOODS, NUTRIENTS, STUDENTS
from meal_planner.explain import plain_words
from meal_planner.planner import build_system, plan_for_students, plan_meal, target_warning

np.set_printoptions(precision=3, suppress=True)


def heading(text):
    print("\n" + "=" * 76 + f"\n{text}\n" + "=" * 76)


# --------------------------------------------------------------------------- report

def report(figure_dir, show):
    name_a = "A - regular student"
    b = np.array(STUDENTS[name_a][:3], dtype=float)
    A, c = build_system(DEFAULT_FOODS)

    heading("STEP 1. Turn the problem into Ax = b")
    print("Foods:", ", ".join(DEFAULT_FOODS))
    print("\nA (rows = kcal, protein, carbs; one column per food):\n", A)
    print("b (Student A's target):", b)
    print("c (price per serving):  ", c)
    print("\n" + explain.equations(A, b, DEFAULT_FOODS))
    plain_words(
        "each food's nutrition label is a list of 3 numbers (a vector). Put the three "
        "labels side by side and you get the table A. Eating x1 servings of food 1, x2 of "
        "food 2 and x3 of food 3 adds the labels up (a linear combination). We want that "
        "total to equal the target b, so we solve Ax = b for the servings x.")

    heading("STEP 2. Is there exactly one answer? The determinant")
    d = linalg.det_cofactor(A)
    print(f"det(A) by cofactor expansion = {d:,.2f}   (numpy check: {np.linalg.det(A):,.2f})")
    plain_words(
        "the determinant is a single number that tells us whether the 3 foods are really "
        "different from each other. If it is 0, one food is just a mix of the other two and "
        "there is no single answer. Here it is not 0, so exactly one meal plan exists.")

    heading("STEP 3. Solve it with Gauss-Jordan elimination on [A | b]")
    result = plan_meal(DEFAULT_FOODS, b, budget=STUDENTS[name_a][3])
    print("\n".join(result["steps"]))
    print("\nRREF =\n", result["rref"])
    plain_words(
        "we simplify the three equations step by step using only three allowed moves: swap "
        "two equations (Type I), multiply one by a number (Type II), and subtract a multiple "
        "of one from another (Type III). At the end each row has a single 1, so each row "
        "reads 'x_i = number' and the last column is the answer.")
    print()
    print(explain.story(DEFAULT_FOODS, result))

    heading("STEP 4. Double-check with Cramer's rule")
    x_cramer = linalg.cramer(A, b)
    print("Gauss-Jordan: ", result["x"])
    print("Cramer's rule:", x_cramer)
    print("Same answer:  ", np.allclose(result["x"], x_cramer))
    plain_words(
        "Cramer's rule is a completely different method: each serving is one determinant "
        "divided by another, x_i = det(A_i) / det(A), where A_i is A with column i replaced "
        "by the target. Two different methods agreeing is strong evidence the answer is right.")

    heading("STEP 5. Plan for every student at once with the inverse: X = A^-1 B")
    A, A_inv, B, X, rows = plan_for_students(DEFAULT_FOODS, STUDENTS)
    print("A^-1 =\n", A_inv)
    print("A A^-1 = I:", np.allclose(A @ A_inv, np.eye(3)))
    short = ["Rice", "PB", "Chicken"]
    print(f"\n{'Student':<24}" + "".join(f"{s:>9}" for s in short)
          + f"{'Cost':>9}{'Budget':>8}  Verdict")
    for row in rows:
        print(f"{row['student']:<24}" + "".join(f"{v:>9.2f}" for v in row["x"])
              + f"{row['cost']:>9.2f}{row['budget']:>8}  {row['verdict']}")
    plain_words(
        "the inverse A^-1 is a reusable 'undo' for A. We compute it once, then every "
        "student's servings come from a single multiplication instead of re-solving. Student "
        "D wants lots of carbs but little protein; rice brings too much protein along, so the "
        "only exact answer needs negative chicken, which is impossible.")

    heading("STEP 6. What if one food is a mix of the others? (combo = rice + chicken)")
    A_dep, c_dep = build_system(DEPENDENT_FOODS)
    print("A_dep =\n", A_dep)
    print(f"det(A_dep) = {linalg.det_cofactor(A_dep):.6f}   rank = {linalg.rank(A_dep)}")
    on_plane = 2 * A_dep[:, 0] + A_dep[:, 1]
    for label, target in [("Student A's target", b), ("2 rice + 1 chicken", on_plane)]:
        r = linalg.solve(A_dep, target)
        print(f"\n{label}: b = {target}")
        print("RREF of [A | b] =\n", r["rref"])
        print("Solutions:", r["kind"])
    print("\nOn-plane target, x = (2, 1, 0) + t(-1, -1, 1):")
    for t in (0, 0.5, 1):
        xt = np.array([2, 1, 0]) + t * np.array([-1, -1, 1])
        print(f"  t = {t:<3}  servings = {xt}  Ax = {A_dep @ xt}  cost = ₱{c_dep @ xt:.2f}")
    plain_words(
        "the combo meal is exactly 1 rice + 1 chicken, so the 3 foods are linearly dependent: "
        "det = 0 and the rank is 2 (only 2 truly different foods). Everything they can make "
        "lies on one flat sheet (a plane, their span). Student A's target is off that sheet, "
        "so the last RREF row reads 0 = -26.2, a contradiction: no plan. A target on the "
        "sheet has endless plans, because swapping 1 rice + 1 chicken for 1 combo changes "
        "nothing (that swap, (-1, -1, 1), is the null space). Only the price changes, so we "
        "pick the cheapest: t = 1.")

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
    print(f"Saved {p1}\n  Each bar is a student's target, built from the food pieces x_j * a_j.")
    print(f"Saved {p2}\n  The flat sheet is everything the dependent foods can make.")
    if show:
        plt.show()


# --------------------------------------------------------------------------- your own meal

def ask_number(prompt, allow_negative=False):
    while True:
        text = input(prompt).strip()
        try:
            value = float(text)
        except ValueError:
            print("    Please type a number, e.g. 600")
            continue
        if value < 0 and not allow_negative:
            print("    Please type a number that is 0 or more.")
            continue
        return value


def ask_yes(prompt):
    return input(prompt).strip().lower() in ("y", "yes")


def choose_foods():
    names = list(FOODS)
    print("\nSTEP 1 of 3 - Choose 3 foods\n")
    print(f"    {'Food (1 serving)':<26}{'kcal':>6}{'protein':>9}{'carbs':>8}{'price':>7}")
    for i, n in enumerate(names, 1):
        kcal, p, carb, cost = FOODS[n]
        print(f"{i:>2}. {n:<26}{kcal:>6g}{p:>8g}g{carb:>7g}g{'₱' + format(cost, 'g'):>7}")
    print("\n  Tip: mix a carb food (rice, banana, bread), a protein food (egg, chicken)")
    print("  and something in between (peanut butter, milk).")
    while True:
        picks = input("\n  Type 3 numbers with spaces between them (e.g. 1 4 5): ").split()
        try:
            idx = [int(p) - 1 for p in picks]
        except ValueError:
            idx = []
        if len(idx) == 3 and len(set(idx)) == 3 and all(0 <= i < len(names) for i in idx):
            return [names[i] for i in idx]
        print("    Please type 3 DIFFERENT numbers from the list, e.g. 1 4 5")


def choose_target():
    print("\nSTEP 2 of 3 - Your target for one meal")
    print("  A typical student lunch is about 500-800 kcal, 20-40 g protein, 60-100 g carbs.\n")
    while True:
        target = [ask_number(f"  {n}: ") for n in NUTRIENTS]
        warning = target_warning(target)
        if warning is None:
            return target
        print("\n" + explain.wrap("WARNING: " + warning, indent="  "))
        if not ask_yes("\n  Type a different target? (y/n): "):
            return target
        print()


def plan_your_meal():
    heading("Budget Meal Planner")
    print(explain.wrap(
        "Pick 3 foods and tell me how many calories, grams of protein and grams of carbs you "
        "want in one meal. I will work out exactly how many servings of each food to eat, "
        "and whether it fits your budget."))
    while True:
        foods = choose_foods()
        target = choose_target()
        print("\nSTEP 3 of 3 - Your budget")
        budget = ask_number("  Budget for this meal (₱): ")
        result = plan_meal(foods, target, budget)

        heading("The maths behind your meal")
        print(explain.equations(result["A"], result["b"], foods))
        heading("Your result")
        if result["warning"]:
            print(explain.wrap("WARNING: " + result["warning"]) + "\n")
        print(explain.story(foods, result))
        if not ask_yes("\nPlan another meal? (y/n): "):
            print("Goodbye!")
            return


# --------------------------------------------------------------------------- start

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # for the ₱ sign on Windows
    parser = argparse.ArgumentParser(description="Plan a meal by solving Ax = b.")
    parser.add_argument("--plan", "--interactive", dest="plan", action="store_true",
                        help="plan your own meal")
    parser.add_argument("--report", action="store_true", help="full linear algebra walkthrough")
    parser.add_argument("--show", action="store_true", help="open the figure windows (with --report)")
    parser.add_argument("--figures", default="figures", help="folder for saved figures")
    args = parser.parse_args()

    if not args.plan and not args.report:
        heading("Budget Meal Planner")
        print("  1. Plan my own meal")
        print("  2. Show the full linear algebra walkthrough (and save the figures)")
        print("  3. Quit")
        choice = input("\nType 1, 2 or 3: ").strip()
        if choice == "1":
            args.plan = True
        elif choice == "2":
            args.report = True
        else:
            return
    if args.plan:
        plan_your_meal()
    else:
        if not args.show:
            matplotlib.use("Agg")  # save figures without opening windows
        report(args.figures, args.show)


if __name__ == "__main__":
    main()
