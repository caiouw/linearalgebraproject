# Budget Meal Planner (MATH 2138 – Linear Algebra)

Finds how many servings of three foods meet a student's calorie, protein and carb targets
by solving the linear system **Ax = b**, then checks the cost against the student's budget.

The project comes in two forms with the same maths:

* `Budget_Meal_Planner.ipynb` – a self-contained notebook for Google Colab (the submission).
* `main.py` + `meal_planner/` – a regular Python project you can run on your own computer.

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com) → *File → Upload notebook* → `Budget_Meal_Planner.ipynb`.
2. *Runtime → Run all*.
3. To try your own foods or targets, edit the `plan_meal(...)` cell in Section 10.

## Run as a Python project

```bash
pip install -r requirements.txt
python main.py                  # full report; figures saved to figures/
python main.py --show           # same, and open the figure windows
python main.py --interactive    # pick your own 3 foods, target and budget
python -m unittest discover tests
```

| File | What it holds |
|---|---|
| `meal_planner/data.py` | Food nutrients and prices, student targets and budgets |
| `meal_planner/linalg.py` | Hand-written cofactor determinant, Gauss-Jordan, rank, Cramer's rule, inverse |
| `meal_planner/planner.py` | Builds A, b, c from the data; judges feasibility and budget |
| `meal_planner/plots.py` | Figure 1 (linear combination bars) and Figure 2 (span plane) |
| `main.py` | Command-line report and interactive mode |
| `tests/test_meal_planner.py` | 13 checks against NumPy and between methods |

## What each part shows

| Part | Linear algebra | Lesson |
|---|---|---|
| Formulation | Matrices, vectors, linear combinations, dot product (cost) | 1 |
| Unique plan? | Determinant by cofactor expansion | 3 |
| Solve | Gauss-Jordan elimination, RREF, elementary row operations | 2 |
| Check | Cramer's rule | 3 |
| All students | Inverse via [A \| I], X = A⁻¹B | 1, 2 |
| Combo meal | Linear dependence, span, rank, null space | 4 |
| Testing | Hand-written results compared with NumPy and with each other | – |

Nutrient values are approximate (USDA FoodData Central); prices are rough estimates in pesos.
