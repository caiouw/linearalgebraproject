"""Hand-written linear algebra used by the planner (Lessons 1-4).

NumPy is used only to store arrays; every algorithm here is written out by hand.
"""
import numpy as np

TOL = 1e-9  # numbers smaller than this are treated as zero


def det_cofactor(M):
    """Determinant by cofactor expansion along the first row (Lesson 3).

    det(A) = sum_j (-1)^(1+j) a_1j det(M_1j), where M_1j is A without row 1 and column j.
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    if n == 1:
        return M[0, 0]
    if n == 2:
        return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]  # ad - bc
    total = 0.0
    for j in range(n):
        minor = np.delete(np.delete(M, 0, axis=0), j, axis=1)
        total += (-1) ** j * M[0, j] * det_cofactor(minor)  # (-1)^(1+j) with 0-based j
    return total


def gauss_jordan(M, n_coef_cols=None):
    """Reduce M to reduced row echelon form with elementary row operations (Lesson 2).

    Only the first n_coef_cols columns may hold pivots, so the right-hand side of
    [A | b] or [A | I] is never used as a pivot column.

    Returns (rref, pivot_columns, steps) where steps describes each row operation.
    """
    M = np.array(M, dtype=float)
    rows, cols = M.shape
    n_coef_cols = cols if n_coef_cols is None else n_coef_cols
    pivots, steps, r = [], [], 0
    for col in range(n_coef_cols):
        if r == rows:
            break
        p = r + int(np.argmax(np.abs(M[r:, col])))  # largest pivot limits rounding error
        if abs(M[p, col]) < TOL:
            continue  # no pivot in this column: free variable
        if p != r:
            M[[r, p]] = M[[p, r]]
            steps.append(f"Type I:   R{r+1} <-> R{p+1}")
        piv = M[r, col]
        M[r] = M[r] / piv
        steps.append(f"Type II:  R{r+1} <- (1/{piv:.4g}) R{r+1}")
        for i in range(rows):
            if i != r and abs(M[i, col]) > TOL:
                f = M[i, col]
                M[i] = M[i] - f * M[r]
                steps.append(f"Type III: R{i+1} <- R{i+1} - ({f:.4g}) R{r+1}")
        pivots.append(col)
        r += 1
    M[np.abs(M) < TOL] = 0.0
    return M, pivots, steps


def rank(A):
    """Rank = number of pivots in the RREF (Lesson 4)."""
    return len(gauss_jordan(A)[1])


def cramer(A, b):
    """Cramer's rule: x_i = det(A_i) / det(A), A_i = A with column i replaced by b (Lesson 3)."""
    A = np.asarray(A, dtype=float)
    dA = det_cofactor(A)
    if abs(dA) < TOL:
        raise ValueError("det(A) = 0: Cramer's rule does not apply.")
    x = np.zeros(len(b))
    for i in range(len(b)):
        Ai = A.copy()
        Ai[:, i] = b
        x[i] = det_cofactor(Ai) / dA
    return x


def inverse(A):
    """Inverse by row-reducing [A | I] to [I | A^-1] (Lesson 2)."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    R, pivots, _ = gauss_jordan(np.hstack([A, np.eye(n)]), n_coef_cols=n)
    if len(pivots) < n:
        raise ValueError("A is singular: it has no inverse.")
    return R[:, n:]


UNIQUE, INFINITE, NONE = "unique", "infinitely many", "none"


def solve(A, b):
    """Solve Ax = b by Gauss-Jordan and classify the solution set.

    Returns a dict with keys: kind (UNIQUE / INFINITE / NONE), x (the solution when
    unique, else None), rank, rref, steps.
    """
    A = np.asarray(A, dtype=float)
    R, pivots, steps = gauss_jordan(np.column_stack([A, b]), n_coef_cols=A.shape[1])
    r = len(pivots)
    inconsistent = any(np.all(np.abs(row[:-1]) < TOL) and abs(row[-1]) > TOL for row in R)
    if inconsistent:
        kind = NONE
    elif r == A.shape[1]:
        kind = UNIQUE
    else:
        kind = INFINITE
    x = R[: A.shape[1], -1].copy() if kind == UNIQUE else None
    return {"kind": kind, "x": x, "rank": r, "rref": R, "steps": steps}
