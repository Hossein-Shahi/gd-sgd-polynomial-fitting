"""Verify the closed-form least-squares minima quoted in the paper (Sections 4.1 and 5.4).

Solves the least-squares problem exactly for polynomial degrees 2, 3, and 4 on the
standardized data and confirms the values reported in the paper:
    degree 2 -> 0.398247   (reached by GD at learning rates 0.01 and 0.1)
    degree 3 -> 0.015294   (reached by GD at learning rate 0.1)
    degree 4 -> 0.015237   (< 0.4% below the cubic optimum)
This is the only script in the repository that requires NumPy.
Run from the repository root:  python verify_closed_form.py
"""
import numpy as np

X, Y = [], []
with open("Part1_x_y_Values.txt") as f:
    next(f)  # header
    for line in f:
        if line.strip():
            a, b = line.split()
            X.append(float(a)); Y.append(float(b))
X, Y = np.array(X), np.array(Y)
Xn = (X - X.mean()) / X.std()   # population std, matching main.py
Yn = (Y - Y.mean()) / Y.std()

expected = {2: 0.398247, 3: 0.015294, 4: 0.015237}
for d in (2, 3, 4):
    A = np.vander(Xn, d + 1)                     # columns x^d ... x^0, as in the paper
    w, *_ = np.linalg.lstsq(A, Yn, rcond=None)
    mse = float(np.mean((A @ w - Yn) ** 2))
    print(f"degree {d}: exact minimum MSE = {mse:.6f}   (R^2 = {1 - mse:.4f})   "
          f"weights (hi->lo) = {np.round(w, 4)}")
    assert abs(mse - expected[d]) < 5e-7, f"degree {d}: got {mse:.6f}, expected {expected[d]}"
print("All closed-form minima match the values reported in the paper.")
