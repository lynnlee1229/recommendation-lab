"""CPU-only constructed mechanisms; no domain model or experiment registry writes."""
import numpy as np


def quadratic_descent(learning_rate, steps=20, initial=1.0):
    """Return w including initial point for L(w)=w², gradient 2w."""
    values = [float(initial)]
    for _ in range(steps):
        values.append(values[-1] - learning_rate * 2 * values[-1])
    return np.array(values)


def polynomial_fit(train_x, train_y, evaluation_x, degree):
    """Fit only on train; return aligned train/evaluation predictions, no labels needed."""
    design = np.polynomial.polynomial.polyvander(train_x, degree)
    coefficients = np.linalg.lstsq(design, train_y, rcond=None)[0]
    return design @ coefficients, np.polynomial.polynomial.polyval(evaluation_x, coefficients)
