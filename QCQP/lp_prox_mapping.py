import numpy as np


def lp_prox_mapping(y: np.ndarray, lam: float, p: float, r: float) -> np.ndarray:
    """
    Compute Lp proximal mapping with infinity norm constraint
    """
    # Handle L1 case separately
    if abs(p - 1.0) < 1e-8:
        return l1_prox(y, lam, r)

    # General Lp case (0 < p < 1)
    b = np.abs(y)
    n_b = len(b)

    # Active set calculation
    base_term = 2 * lam * (1 - p)
    exponent1 = 1 / (2 - p)
    exponent2 = (p - 1) / (2 - p)
    tau = (base_term ** exponent1) + lam * p * (base_term ** exponent2)
    active_mask = b > tau

    if not np.any(active_mask):
        return np.zeros(n_b)

    # Solve for active variables
    active_b = b[active_mask]
    active_y = y[active_mask]
    tmpx = active_b.copy()

    # Newton iterations
    max_iter = 1000
    tol_newton = 1e-12
    lam_p = lam * p
    lam_p_p1 = lam * p * (p - 1)

    for i in range(max_iter):
        gx = tmpx - active_b + lam_p * (tmpx ** (p - 1))
        if np.max(np.abs(gx)) < tol_newton:
            break

        hx = 1 + lam_p_p1 * (tmpx ** (p - 2))
        hx[hx == 0] = 1  # Avoid division by zero.
        tmpx = tmpx - gx / hx
        tmpx[tmpx < 1e-12] = 1e-12  # Ensure non-negativity

    # Apply infinity norm constraint
    violate_mask = tmpx > r
    if np.any(violate_mask):
        violate_active_b = active_b[violate_mask]
        f_r = 0.5 * (r - violate_active_b) ** 2 + lam * (r ** p)
        f_0 = 0.5 * (violate_active_b ** 2)
        better_r = f_r < f_0
        tmpx[violate_mask] = better_r * r + (~better_r) * 0

    # Reconstruct solution with sign
    x = np.zeros(n_b)
    x[active_mask] = tmpx * np.sign(active_y)

    return x


def l1_prox(y: np.ndarray, lam: float, r: float) -> np.ndarray:
    """
    L1 proximal mapping with infinity norm constraint
    """
    abs_y = np.abs(y)
    soft_threshold = np.maximum(abs_y - lam, 0)
    soft_threshold_inf_norm = np.minimum(soft_threshold, r)
    x = np.sign(y) * soft_threshold_inf_norm
    return x