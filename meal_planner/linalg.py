"""The linear algebra behind the planner, written out by hand (Lessons 1-4)."""
import numpy as np

TOL = 1e-9  # numbers closer to 0 than this count as 0


def det2(M):
    """Determinant of a 2x2 matrix: ad - bc."""
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]


def det3(A):
    """Determinant of a 3x3 matrix by cofactor expansion along row 1 (Lesson 3).

    det(A) = a11*det(M11) - a12*det(M12) + a13*det(M13)
    """
    A = np.asarray(A, dtype=float)
    M11 = A[1:, [1, 2]]  # A without row 1 and column 1
    M12 = A[1:, [0, 2]]  # A without row 1 and column 2
    M13 = A[1:, [0, 1]]  # A without row 1 and column 3
    return A[0, 0] * det2(M11) - A[0, 1] * det2(M12) + A[0, 2] * det2(M13)


def gauss_jordan(M, n):
    """Row-reduce M to reduced row echelon form (Lesson 2).

    n is the number of unknowns: only the first n columns get a leading 1.
    Returns the RREF, the pivot columns, and a list of the row operations used.
    """
    M = np.array(M, dtype=float)
    pivots, steps = [], []
    row = 0
    for col in range(n):
        # find a row (from `row` down) with a nonzero entry in this column
        candidates = [r for r in range(row, len(M)) if abs(M[r, col]) > TOL]
        if not candidates:
            continue  # no leading 1 in this column: a free variable

        r = candidates[0]
        if r != row:  # Type I: swap two rows
            M[[row, r]] = M[[r, row]]
            steps.append(f"Type I:   swap R{row+1} and R{r+1}")

        value = M[row, col]  # Type II: divide the row to get a leading 1
        M[row] = M[row] / value
        steps.append(f"Type II:  R{row+1} = R{row+1} / {value:.4g}")

        for i in range(len(M)):  # Type III: clear the rest of the column
            if i != row and abs(M[i, col]) > TOL:
                factor = M[i, col]
                M[i] = M[i] - factor * M[row]
                steps.append(f"Type III: R{i+1} = R{i+1} - ({factor:.4g}) R{row+1}")

        pivots.append(col)
        row += 1
        if row == len(M):
            break

    M[np.abs(M) < TOL] = 0.0
    return M, pivots, steps


def rank(A):
    """Rank = number of leading 1s in the RREF (Lesson 4)."""
    A = np.asarray(A, dtype=float)
    return len(gauss_jordan(A, A.shape[1])[1])


def solve(A, b):
    """Solve Ax = b with Gauss-Jordan on [A | b] and say what kind of answer it has."""
    A = np.asarray(A, dtype=float)
    n = A.shape[1]
    R, pivots, steps = gauss_jordan(np.column_stack([A, b]), n)

    # a row [0 0 0 | nonzero] means 0 = nonzero: impossible
    impossible = any(np.all(row[:n] == 0) and row[n] != 0 for row in R)
    if impossible:
        kind = "none"
    elif len(pivots) == n:
        kind = "unique"
    else:
        kind = "infinitely many"

    x = R[:n, n] if kind == "unique" else None
    return {"kind": kind, "x": x, "rank": len(pivots), "rref": R, "steps": steps}


def cramer(A, b):
    """Cramer's rule: x_i = det(A_i) / det(A), A_i = A with column i replaced by b (Lesson 3)."""
    A = np.asarray(A, dtype=float)
    d = det3(A)
    if abs(d) < TOL:
        raise ValueError("det(A) = 0, so Cramer's rule cannot be used.")
    x = np.zeros(3)
    for i in range(3):
        Ai = A.copy()
        Ai[:, i] = b
        x[i] = det3(Ai) / d
    return x


def inverse(A):
    """Inverse by row-reducing [A | I] to [I | A^-1] (Lesson 2)."""
    A = np.asarray(A, dtype=float)
    R, pivots, _ = gauss_jordan(np.hstack([A, np.eye(3)]), 3)
    if len(pivots) < 3:
        raise ValueError("A has no inverse because det(A) = 0.")
    return R[:, 3:]
