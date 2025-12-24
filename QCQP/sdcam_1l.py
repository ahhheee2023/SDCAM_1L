import numpy as np
from typing import Dict, Tuple
from lp_prox_mapping import lp_prox_mapping


def evaluate_objective(x: np.ndarray, problem: Dict) -> float:
    """
    Evaluate objective function
    """
    quadratic = 0.5 * x.T @ problem['Q0'] @ x
    linear = problem['b0'].T @ x
    lp_norm = problem['alpha'] * np.sum(np.abs(x) ** problem['p'])
    return quadratic + linear + lp_norm


def SDCAM_1L(x_t: np.ndarray, y_t: np.ndarray, beta_0: float, last_mu: float,
             problem: Dict, solver_params: Dict) -> Tuple[np.ndarray, np.ndarray, float, float, int, Dict]:
    """
    SDCAM_1L algorithm implementation
    """
    n = problem['n']
    m = problem['m']

    mu_max = solver_params.get('mu_max', 1e7)
    rho = solver_params.get('rho', 0.8)
    eta = solver_params.get('eta', 1.2)
    delta = solver_params.get('delta', 0.3)
    max_inner_iter = solver_params.get('max_inner_iter', 100)
    tol = solver_params.get('tol', 1e-10)
    max_iter = solver_params.get('max_iter', 1000)
    print_freq = solver_params.get('print_freq', 100)

    Q0_sym = 0.5 * (problem['Q0'] + problem['Q0'].T)
    Qi_reshaped = problem['Qi'].reshape(m * n, n)

    trial1 = 0
    trial2 = 0

    # History tracking
    history = {
        'iterations': [],
        'objective': [],
        'rel_feas': [],
        'x_diff_over_mu_square': []
    }

    for t in range(max_iter):
        # Initialize mu
        if last_mu is None:
            mu = solver_params.get('mu', 1.0)
        else:
            mu = last_mu

        # Update beta_t
        beta_t = beta_0 * ((t + 1) ** delta)

        # Compute c(x_t)
        Qx = Qi_reshaped @ x_t
        Qx_reshaped = Qx.reshape(m, n)
        quadratic_terms = 0.5 * np.sum(x_t * Qx_reshaped, axis=1)
        linear_terms = problem['bi'] @ x_t
        c_xt = quadratic_terms + linear_terms + problem['ci']
        print(f"x_t shape: {x_t.shape}")
        print(f"Qx_reshaped shape: {Qx_reshaped.shape}")

        # Compute gradient of f
        grad_f = Q0_sym @ x_t + problem['b0']

        # Compute Jacobian term Jc
        Jc = Qx_reshaped + problem['bi']
        Jc_term = Jc.T @ (c_xt - y_t)

        obj_x_t = evaluate_objective(x_t, problem)

        inner_iter = 0
        while inner_iter < max_inner_iter:
            #Lp proximal mapping
            target_point = x_t - (mu / 2) * (grad_f + beta_t * Jc_term)
            tilde_x = lp_prox_mapping(target_point, (mu / 2) * problem['alpha'], problem['p'], problem['r'])

            # Compute c(tilde_x)
            Q_tilde_x = Qi_reshaped @ tilde_x
            Q_tilde_x_reshaped = Q_tilde_x.reshape(m, n)
            quadratic_terms_tilde = 0.5 * np.sum(tilde_x * Q_tilde_x_reshaped, axis=1)
            linear_terms_tilde = problem['bi'] @ tilde_x
            c_tilde_x = quadratic_terms_tilde + linear_terms_tilde + problem['ci']

            # Check conditions
            constraint_diff_norm = np.linalg.norm(c_xt - c_tilde_x)
            x_diff_norm = np.linalg.norm(tilde_x - x_t)

            condition1 = constraint_diff_norm <= np.sqrt(1 / (mu * beta_t)) * x_diff_norm

            obj_tilde_x = evaluate_objective(tilde_x, problem)
            term1 = (obj_tilde_x - obj_x_t) / beta_t
            term2 = 0.5 * (np.linalg.norm(c_tilde_x - y_t) ** 2 - np.linalg.norm(c_xt - y_t) ** 2)
            term3 = - (1 / (2 * mu * beta_t)) * (x_diff_norm ** 2)
            condition2 = (term1 + term2) <= term3

            inner_iter += 1

            if condition1 and condition2:
                x_next = tilde_x
                y_next = np.minimum(c_tilde_x, 0)  # Project onto non-positive orthant
                mu = min(mu_max, eta * mu)
                break
            else:
                mu = rho * mu

        if inner_iter == max_inner_iter:
            x_next = x_t
            y_next = y_t

        last_mu = mu

        objective_val = evaluate_objective(x_next, problem)
        x_norm = np.linalg.norm(x_t)

        c_x_next = c_tilde_x
        ci_abs = np.abs(problem['ci'])
        ci_abs[ci_abs == 0] = 1
        rel_feas_current = np.linalg.norm(np.maximum(c_x_next, 0) / ci_abs)

        # Record history
        history['iterations'].append(t + 1)
        history['objective'].append(objective_val)
        history['rel_feas'].append(rel_feas_current)
        history['x_diff_over_mu_square'].append(x_diff_norm / mu)

        # Print progress for first iteration
        if t == 0:
            print(f'when t=1, mu is {mu:.1e}, norm of x_diff is {x_diff_norm:.1e}')

        # Check termination conditions
        if x_diff_norm <= tol * x_norm / 10:
            trial1 += 1
            if np.max(np.maximum(c_x_next, 0)) <= tol * np.max(np.abs(problem['ci'])) and trial1 >= 5:
                print(f'Terminate: feas {np.max(c_x_next):.4e}')
                print(
                    f' iter {t + 1:8d}, fval {objective_val:6.5e}, norm_cval {np.linalg.norm(np.maximum(c_x_next, 0)):2.1e}, '
                    f'rel.feas = {rel_feas_current:2.1e}, beta {beta_t:2.1e}, mu {mu:2.1e}')
                break
        else:
            trial1 = 0

        if mu < 1e-10:
            trial2 += 1
            if trial2 >= 5:
                print(' Termination due to small stepsize')
                print(
                    f' iter {t + 1:8d}, fval {objective_val:6.5e}, norm_cval {np.linalg.norm(np.maximum(c_x_next, 0)):2.1e}, '
                    f'rel.feas = {rel_feas_current:2.1e}, beta {beta_t:2.1e}, mu {mu:2.1e}')
                break
            mu = 1.0  # restart mu
        else:
            trial2 = 0

        # Update variables for next iteration
        x_t = x_next.copy()
        y_t = y_next.copy()

        # Print progress
        if (t + 1) % print_freq == 0:
            print(
                f' iter {t + 1:8d}, fval {objective_val:6.5e}, norm_cval {np.linalg.norm(np.maximum(c_x_next, 0)):2.1e}, '
                f'rel.feas = {rel_feas_current:2.1e}, beta {beta_t:2.1e}, mu {mu:2.1e}')

    return x_next, y_next, x_diff_norm, mu, inner_iter, history