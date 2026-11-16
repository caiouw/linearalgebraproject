"""Food and student data.

Nutrient values are approximate, per serving, based on USDA FoodData Central.
Costs are rough Philippine peso estimates for food prepared at home; edit them freely.
"""

NUTRIENTS = ["Calories (kcal)", "Protein (g)", "Carbs (g)"]

# name: (kcal, protein g, carbs g, cost ₱)
FOODS = {
    "Rice (1 cup cooked)":    (205, 4.3, 44.5, 5),
    "Egg (1 large)":          (78, 6.3, 0.6, 9),
    "Banana (1 medium)":      (105, 1.3, 27.0, 8),
    "Peanut butter (2 tbsp)": (191, 7.1, 7.7, 14),
    "Chicken breast (100 g)": (165, 31.0, 0.0, 35),
    "White bread (1 slice)":  (67, 1.9, 12.7, 4),
    "Milk (1 cup)":           (149, 7.7, 11.7, 20),
    # Exactly 1 rice + 1 chicken, sold cheaper as a combo: used to show linear dependence.
    "Chicken rice meal":      (370, 35.3, 44.5, 35),
}

# name: (kcal, protein g, carbs g, budget ₱) for one meal
STUDENTS = {
    "A - regular student":   (600, 25, 80, 80),
    "B - varsity athlete":   (900, 45, 110, 100),
    "C - high-protein diet": (500, 40, 30, 70),
    "D - carb-loading":      (700, 10, 140, 60),
}

DEFAULT_FOODS = ["Rice (1 cup cooked)", "Peanut butter (2 tbsp)", "Chicken breast (100 g)"]
DEPENDENT_FOODS = ["Rice (1 cup cooked)", "Chicken breast (100 g)", "Chicken rice meal"]
