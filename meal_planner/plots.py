"""The two figures used in the report."""
import numpy as np
import matplotlib.pyplot as plt

from .data import NUTRIENTS

COLORS = ["#e0b84a", "#a0522d", "#d9826b"]


def linear_combination_chart(A, X, B, food_names, student_names):
    """Figure 1: each bar is x1*a1 + x2*a2 + x3*a3 stacked up to the student's target."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    idx = np.arange(len(student_names))
    for i, (ax, nutrient) in enumerate(zip(axes, NUTRIENTS)):
        pos = np.zeros(len(student_names))
        neg = np.zeros(len(student_names))
        for j, food in enumerate(food_names):
            part = X[j] * A[i, j]  # x_j * (nutrient i in food j)
            ax.bar(idx, part, bottom=np.where(part >= 0, pos, neg), color=COLORS[j], label=food)
            pos += np.clip(part, 0, None)
            neg += np.clip(part, None, 0)
        ax.scatter(idx, B[i], color="black", marker="_", s=600, zorder=3, label="Target")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(idx, [n.split(" - ")[0] for n in student_names])
        ax.set_title(nutrient)
        ax.set_xlabel("Student")
        ax.set_ylim(top=1.1 * max(pos.max(), B[i].max()))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Figure 1. Each target as a linear combination of the food vectors")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def span_plane_chart(A_dep, food_names, target):
    """Figure 2: dependent foods span only a plane; a target off the plane is unreachable."""
    a1, a2, a3 = A_dep[:, 0], A_dep[:, 1], A_dep[:, 2]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    s, t = np.meshgrid(np.linspace(0, 3, 10), np.linspace(0, 2, 10))
    plane = s[..., None] * a1 + t[..., None] * a2  # span{a1, a2}
    ax.plot_surface(plane[..., 0], plane[..., 1], plane[..., 2], alpha=0.25, color="steelblue")
    for vec, name, col in [(a1, food_names[0], COLORS[0]), (a2, food_names[1], COLORS[2]),
                           (a3, food_names[2], "green")]:
        ax.quiver(0, 0, 0, *vec, color=col, lw=2, arrow_length_ratio=0.08)
        ax.text(*vec, " " + name, fontsize=8)
    ax.scatter(*target, color="red", s=60)
    ax.text(*target, "  target (off the plane)", color="red", fontsize=9)
    ax.set_xlabel("kcal")
    ax.set_ylabel("protein (g)")
    ax.set_zlabel("carbs (g)")
    ax.set_title(r"Figure 2. Dependent foods span only a plane in $\mathbb{R}^3$")
    fig.tight_layout()
    return fig
