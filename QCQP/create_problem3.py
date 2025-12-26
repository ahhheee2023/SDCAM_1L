import numpy as np
from scipy.linalg import qr
from lp_prox_mapping import lp_prox_mapping


def create_problem3(n: int, m: int, alpha: float, p: float):
    """
    Create optimization problem similar to MATLAB version
    """
    problem = {}
    problem['n'] = n
    problem['m'] = m
    problem['alpha'] = alpha
    problem['p'] = p

    # Generate objective function matrices
    scale0 = 5.0
    scale = 5.0

    # Q0 is identity matrix
    problem['Q0'] = np.eye(n)

    # Generate b0
    np.random.seed(42)  # For reproducibility
    b0 = np.random.randn(n) * scale0

    # Compute x0 using lp_prox_mapping
    x0 = lp_prox_mapping(-b0, alpha, p, 1e5)
    if np.linalg.norm(x0) == 0:
        raise ValueError('norm(x0) = 0')

    # Generate constraint matrices
    Qi = np.zeros((m, n, n))
    bi = np.zeros((m, n))
    ci = np.zeros(m)

    for i in range(m):
        # Generate random orthogonal matrix using QR decomposition
        A = np.random.randn(n, n)
        Q, R = qr(A)

        # Generate diagonal matrix with positive entries
        d = np.random.rand(n) * scale
        D = np.diag(d)

        Qi_i = Q @ D @ Q.T
        bi_i = np.zeros(n)

        # Compute quadratic term
        Qval = 0.5 * x0.T @ (Qi_i @ x0)

        ci_i = -Qval * 0.5  # A smaller ratio will make the problem harder

        Qi[i, :, :] = Qi_i
        bi[i, :] = bi_i
        ci[i] = ci_i

    problem['Qi'] = Qi
    problem['bi'] = bi
    problem['ci'] = ci
    problem['b0'] = b0
    problem['r'] = np.max(np.abs(x0))  # infinity norm
    print(f'The inf norm is {problem['r']:2.1e}')

    return problem
