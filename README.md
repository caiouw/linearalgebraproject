# Budget Meal Planner (MATH 2138 – Linear Algebra)

Finds how many servings of three foods meet a student's calorie, protein and carb targets
by solving the linear system **Ax = b**, then checks the cost against the student's budget.

## Run it

1. Open [Google Colab](https://colab.research.google.com) → *File → Upload notebook* → `Budget_Meal_Planner.ipynb`.
2. *Runtime → Run all*.
3. To try your own foods or targets, edit the `plan_meal(...)` cell in Section 10.

## What each section shows

| Section | Linear algebra | Lesson |
|---|---|---|
| 3. Formulation | Matrices, vectors, linear combinations, dot product (cost) | 1 |
| 4. Unique plan? | Determinant by cofactor expansion | 3 |
| 5. Solve | Gauss-Jordan elimination, RREF, elementary row operations | 2 |
| 6. Check | Cramer's rule | 3 |
| 7. All students | Inverse via [A \| I], X = A⁻¹B | 1, 2 |
| 9. Combo meal | Linear dependence, span, rank, null space | 4 |
| 11. Testing | Hand-written results compared with NumPy and with each other | – |

Nutrient values are approximate (USDA FoodData Central); prices are rough estimates in pesos.
