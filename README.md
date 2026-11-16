# Budget Meal Planner (MATH 2138 – Linear Algebra)

**What it does:** you pick 3 foods and say how many calories, grams of protein and grams of carbs
you want in one meal. The program works out exactly how many servings of each food to eat, and
whether the meal fits your budget. Behind the scenes this is one linear system, **Ax = b**.

There are two versions with the same maths:

* `Budget_Meal_Planner.ipynb` – a notebook for Google Colab (the class submission).
* `main.py` + `meal_planner/` – a Python program you run on your own computer.

## Run in Google Colab (nothing to install)

1. Go to [Google Colab](https://colab.research.google.com).
2. *File → Open notebook → GitHub*, paste this repository's URL, and open `Budget_Meal_Planner.ipynb`.
3. *Runtime → Run all*.
4. To try your own meal, scroll to **Section 10**, change the foods, target and budget, and run that cell.

## Run the Python program in Google Colab

Open a new notebook in Colab and run these in a cell, one line at a time or all together:

```
!git clone https://github.com/caiouw/linearalgebraproject.git
%cd linearalgebraproject
%run main.py --report
```

* `%run main.py --plan` plans your own meal (Colab shows a box to type your answers in).
* `!python -m unittest discover tests` runs the 22 checks.
* The figures are saved in `linearalgebraproject/figures` (open the folder icon on the left).

## Run on your computer (Windows)

1. Open the project folder in File Explorer, right-click an empty space, choose **Open in Terminal**.
2. The first time only, install the two libraries it needs:

   ```bash
   py -m pip install -r requirements.txt
   ```

3. Start the program:

   ```bash
   py main.py
   ```

   A menu appears: type **1** to plan your own meal, or **2** for the full step-by-step
   linear algebra walkthrough (which also saves two figures into a `figures` folder).

Type these commands at the terminal prompt (`PS C:\...>`), not inside Python. If you see `>>>`,
type `exit()` first. On Mac or Linux, use `python3` instead of `py`.

Shortcuts: `py main.py --plan` goes straight to planning, `py main.py --report --show` runs the
walkthrough and opens the figures, and `py -m unittest discover tests` runs the 22 checks.

## Reading your result

| You see | It means |
|---|---|
| A list of servings and **"hits your target exactly"** | Eat those amounts; the total cost is shown |
| **"NOT POSSIBLE"** with a negative number | The only exact answer would need you to remove a food. Swap that food or change the target |
| **"over your budget"** | The meal works but costs too much. Try cheaper foods |
| **"Your foods overlap"** | One food is a mix of the others (e.g. the combo meal with rice and chicken). Swap one |
| **"WARNING: Impossible target"** | Protein and carbs alone already exceed your calories, so no foods can work |

## Linear algebra in plain words

| Term | In this project it means… |
|---|---|
| **Vector** | A food's nutrition label as a list of numbers, e.g. rice = (205 kcal, 4.3 g protein, 44.5 g carbs) |
| **Matrix A** | The three labels placed side by side, one column per food |
| **Linear combination** | Eating some servings of each food and adding up the nutrients |
| **Ax = b** | "Which servings x add up to my target b?" |
| **Determinant** | One number saying whether the foods are truly different; 0 means one food is just a mix of the others |
| **Gauss-Jordan / RREF** | Simplifying the equations step by step until each one reads "x = number" |
| **Cramer's rule** | A second, formula-based way to get the same answer, used as a check |
| **Inverse A⁻¹** | A reusable "undo" for A: any target times A⁻¹ gives the servings |
| **Rank** | How many of the foods are truly different |
| **Span** | Every nutrient total the foods can make |
| **Null space** | Swaps between foods that change nothing (1 combo meal ↔ 1 rice + 1 chicken) |

## Project files

| File | What it holds |
|---|---|
| `meal_planner/data.py` | Food nutrients and prices, student targets and budgets (edit prices here) |
| `meal_planner/linalg.py` | Hand-written cofactor determinant, Gauss-Jordan, rank, Cramer's rule, inverse |
| `meal_planner/planner.py` | Builds A, b, c from the data; warns about impossible targets; checks feasibility and budget |
| `meal_planner/explain.py` | Turns each result into plain-language sentences |
| `meal_planner/plots.py` | Figure 1 (linear combination bars) and Figure 2 (span plane) |
| `main.py` | The menu, the guided meal planner and the walkthrough |
| `tests/test_meal_planner.py` | 22 checks against NumPy, between methods, and of the explanations |

## Which lesson each part uses

| Part | Linear algebra | Lesson |
|---|---|---|
| Formulation | Matrices, vectors, linear combinations, dot product (cost) | 1 |
| Unique plan? | Determinant by cofactor expansion | 3 |
| Solve | Gauss-Jordan elimination, RREF, elementary row operations | 2 |
| Check | Cramer's rule | 3 |
| All students | Inverse via [A \| I], X = A⁻¹B | 1, 2 |
| Combo meal | Linear dependence, span, rank, null space | 4 |

Nutrient values are approximate (USDA FoodData Central); prices are rough estimates in pesos.
