"""The two figures used in the report."""
import numpy as np
import matplotlib.pyplot as plt

from .data import NUTRIENTS

COLORS = ["#e0b84a", "#a0522d", "#d9826b"]


def linear_combination_chart(A, X, B, food_names, student_names):
    """Figure 1: each bar is x1*a1 + x2*a2 + x3*a3, stacked up to the student's target."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = [name.split(" - ")[0] for name in student_names]
    positions = np.arange(len(student_names))

    for i in range(3):  # one chart per nutrient
        ax = axes[i]
        top = np.zeros(len(student_names))     # positive pieces stack upward from 0
        under = np.zeros(len(student_names))   # negative pieces stack downward from 0
        for j in range(3):  # stack the three foods
            piece = X[j] * A[i, j]  # x_j times (nutrient i in food j)
            start = np.where(piece >= 0, top, under)
            ax.bar(positions, piece, bottom=start, color=COLORS[j], label=food_names[j])
            top = top + np.maximum(piece, 0)
            under = under + np.minimum(piece, 0)
        ax.scatter(positions, B[i], color="black", marker="_", s=600, zorder=3, label="Target")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_ylim(top=1.1 * max(top.max(), B[i].max()))
        ax.set_xticks(positions, labels)
        ax.set_title(NUTRIENTS[i])
        ax.set_xlabel("Student")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Figure 1. Each target as a linear combination of the food vectors")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def span_plane_chart(A_dep, food_names, target):
    """Figure 2: dependent foods only reach a flat plane; the target is off it."""
    a1, a2, a3 = A_dep[:, 0], A_dep[:, 1], A_dep[:, 2]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")

    # the plane spanned by a1 and a2: all points s*a1 + t*a2
    s, t = np.meshgrid(np.linspace(0, 3, 10), np.linspace(0, 2, 10))
    X = s * a1[0] + t * a2[0]
    Y = s * a1[1] + t * a2[1]
    Z = s * a1[2] + t * a2[2]
    ax.plot_surface(X, Y, Z, alpha=0.25, color="steelblue")

    for vector, name, color in [(a1, food_names[0], COLORS[0]),
                                (a2, food_names[1], COLORS[2]),
                                (a3, food_names[2], "green")]:
        ax.quiver(0, 0, 0, *vector, color=color, lw=2, arrow_length_ratio=0.08)
        ax.text(*vector, " " + name, fontsize=8)

    ax.scatter(*target, color="red", s=60)
    ax.text(*target, "  target (off the plane)", color="red", fontsize=9)
    ax.set_xlabel("kcal")
    ax.set_ylabel("protein (g)")
    ax.set_zlabel("carbs (g)")
    ax.set_title(r"Figure 2. Dependent foods span only a plane in $\mathbb{R}^3$")
    fig.tight_layout()
    return fig
