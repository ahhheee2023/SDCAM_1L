import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from mlp import MLP


class SDCAM_1L:
    """
    Implementation of SDCAM_1L algorithm

    Features:
    1. g = delta_C + lambda * ||.||_1 where C is l_inf ball
    2. Objective: (1/m) * sum {rho(MLP(a_i; v) - y_i)} + lambda * ||v||_1
    3. Constraint set: l_inf ball
    """

    def __init__(self, problem: 'OptimizationProblem', mu_max: float = 1.0,
                 mu_init: float = 0.1, rho: float = 0.15, eta: float = 12.0,
                 beta_0: float = 1.0, beta_delta: float = 0.1,
                 max_inner_iter: int = 100, tol: float = 1e-6,
                 newton_max_iter: int = 100, newton_tol: float = 1e-8):
        """
        Initialize SDCAM_1L algorithm
        """
        self.problem = problem
        self.mu_max = mu_max
        self.mu = mu_init
        self.rho = rho
        self.eta = eta
        self.beta_0 = beta_0
        self.beta_delta = beta_delta
        self.max_inner_iter = max_inner_iter
        self.tol = tol
        self.newton_max_iter = newton_max_iter
        self.newton_tol = newton_tol

        # Algorithm state
        self.current_iteration = 0
        self.x_t = None
        self.y_t = None

        # History tracking
        self.history = {
            'iterations': [],
            'objective_values': [],
            'constraint_norms': [],
            'c_minus_y_norms': [],
            'x_diff_norms': [],
            'mu_values': [],
            'beta_values': [],
            'inner_iter_counts': [],
            'is_successful': [],
            'l1_norms': [],
            'linf_norms': [],
            'x_values': []
        }

    def compute_proximal_y(self, c_value: jnp.ndarray, beta_t: float) -> jnp.ndarray:
        """
        Compute proximal mapping for y: y^{t+1} in prox_{(1/beta_t) h}(c(x^{t+1}))

        Implements Lp proximal mapping for 0 < p <= 1

        Solves: min_y (1/(beta_t * p)) * ||y||_p^p + 1/2 * ||y - c||^2
        """
        # Convert to numpy for Lp proximal mapping computation
        y = np.array(c_value)
        p = float(self.problem.p)
        lambda_val = 1.0 / (beta_t * p)  # Regularization parameter

        # Call the Lp proximal mapping function
        x = self._lp_prox_mapping(y, lambda_val, p)

        return jnp.array(x)

    def _lp_prox_mapping(self, y: np.ndarray, lambda_val: float, p: float) -> np.ndarray:
        """
        Compute the proximal mapping of Lp regularization
        Solves: min f(x) := lambda * ||x||_p^p + 1/2 * ||x - y||^2 (0 < p <= 1)
        """
        iter_count = 0

        if abs(p - 1.0) < 1e-8:
            # For p = 1: soft thresholding
            x = np.sign(y) * np.maximum(np.abs(y) - lambda_val, 0.0)
        else:
            # For 0 < p < 1: use Newton's method
            b = np.abs(y)
            n = len(b)
            x = np.zeros_like(b)

            # Compute tau threshold for active set
            base_term = 2 * lambda_val * (1 - p)
            if base_term < 1e-15:
                tau = 0.0
            else:
                tau = (base_term ** (1 / (2 - p))) + lambda_val * p * (base_term ** ((p - 1) / (2 - p)))

            # Find active indices where b > tau
            index = np.where(b > tau)[0]

            if len(index) > 0:
                tmpx = b[index].copy()  # Initial guess
                tmpY = b[index]  # Target values

                # Compute initial gradient
                gx = tmpx - tmpY + lambda_val * p * (tmpx ** (p - 1))

                # Newton's method iterations
                while np.max(np.abs(gx)) > self.newton_tol and iter_count < self.newton_max_iter:
                    # Compute Hessian
                    hx = 1 + lambda_val * p * (p - 1) * (tmpx ** (p - 2))

                    # Avoid division by zero
                    hx_safe = np.where(np.abs(hx) < 1e-12, 1.0, hx)

                    # Newton update
                    tmpx = tmpx - gx / hx_safe

                    # Ensure non-negativity
                    tmpx = np.maximum(tmpx, 1e-12)

                    # Recompute gradient
                    gx = tmpx - tmpY + lambda_val * p * (tmpx ** (p - 1))

                    iter_count += 1

                # Store results with original signs
                x[index] = tmpx * np.sign(y[index])

        return x

    def initialize(self, x_init: Optional[Dict] = None, y_init: Optional[jnp.ndarray] = None):
        """
        Initialize algorithm with starting points
        """
        if x_init is None:
            self.x_t = self.problem.mlp.get_parameters()
        else:
            self.x_t = x_init.copy()

        if y_init is None:
            self.y_t = jnp.zeros(self.problem.m)
        else:
            self.y_t = y_init.copy()

        print("SDCAM_1L Algorithm Initialized")
        print(f"  Samples (m): {self.problem.m}")
        print(f"  Constraint C: {self.problem.C:.4f}")
        print(f"  p value: {self.problem.p}")
        print(f"  Initial mu: {self.mu:.4f}")
        print(f"  beta_0: {self.beta_0:.4f}, delta: {self.beta_delta}")
        print(f"  Regularization lambda: {self.problem.lambda_reg:.4f}")
        print(f"  Newton max iterations: {self.newton_max_iter}")
        print(f"  Newton tolerance: {self.newton_tol}")

    def compute_beta_t(self, t: int) -> float:
        """Compute beta_t = beta_0 * (t+1)^delta"""
        return self.beta_0 * ((t + 1) ** self.beta_delta)

    def compute_tilde_x(self, x_t: Dict, y_t: jnp.ndarray, beta_t: float, mu: float) -> Dict:
        """
        Compute tilde_x using proximal mapping
        tilde_x in argmin_x {<grad_f + beta_t * J_c^T * (c - y), x> + (1/mu) * ||x - x_t||^2 + g(x)}
        """
        gradient_term = self.problem.compute_gradient_term_vjp(x_t, y_t, beta_t)

        # Compute target point: x_t - (mu/2) * gradient_term
        target_point = {}
        for key in x_t.keys():
            target_point[key] = x_t[key] - (mu / 2.0) * gradient_term[key]

        # Soft thresholding for L1 regularization
        threshold = mu * self.problem.lambda_reg / 2.0
        soft_thresholded = self.problem.soft_thresholding(target_point, threshold)

        # Project onto l_inf ball constraint
        tilde_x = self.problem.project_to_constraint_set(soft_thresholded)

        return tilde_x

    def check_condition_1(self, c_tilde: jnp.ndarray, c_t: jnp.ndarray,
                          tilde_x: Dict, x_t: Dict, beta_t: float, mu: float) -> bool:
        """
        Check Condition (i): ||c(tilde_x) - c(x_t)|| <= sqrt(1/(mu * beta_t)) * ||tilde_x - x_t||
        """
        c_diff_norm = jnp.linalg.norm(c_tilde - c_t)
        x_diff_norm = self._compute_param_norm_diff(tilde_x, x_t)
        rhs = jnp.sqrt(1.0 / (mu * beta_t)) * x_diff_norm
        return float(c_diff_norm) <= float(rhs)

    def check_condition_2(self, tilde_x: Dict, x_t: Dict, c_tilde: jnp.ndarray,
                          c_t: jnp.ndarray, y_t: jnp.ndarray, beta_t: float, mu: float) -> bool:

        # Compute g values
        g_tilde = self.problem.compute_g_value(tilde_x)
        g_t = self.problem.compute_g_value(x_t)

        # If either point is not feasible, condition fails
        if g_tilde == float('inf') or g_t == float('inf'):
            return False

        # Left side
        left_term = g_tilde + (beta_t / 2.0) * jnp.sum(jnp.square(c_tilde - y_t))

        # Right side
        right_term = g_t + (beta_t / 2.0) * jnp.sum(jnp.square(c_t - y_t))
        x_diff_norm_sq = self._compute_param_norm_diff_squared(tilde_x, x_t)
        right_term -= (1.0 / (2.0 * mu)) * x_diff_norm_sq

        tolerance =0
        return float(left_term) <= float(right_term) + tolerance

    def run_iteration(self) -> bool:
        """
        Run one iteration of SDCAM_1L algorithm
        """
        t = self.current_iteration
        beta_t = self.compute_beta_t(t)

        inner_iter = 0
        mu_current = self.mu

        while inner_iter < self.max_inner_iter:
            # Compute tilde_x
            tilde_x = self.compute_tilde_x(self.x_t, self.y_t, beta_t, mu_current)

            # Compute constraint values
            c_tilde = self.problem.compute_constraints(tilde_x)
            c_t = self.problem.compute_constraints(self.x_t)

            # Check conditions
            cond1 = self.check_condition_1(c_tilde, c_t, tilde_x, self.x_t, beta_t, mu_current)
            cond2 = self.check_condition_2(tilde_x, self.x_t, c_tilde, c_t, self.y_t, beta_t, mu_current)

            if cond1 and cond2:
                # Successful iteration
                self.x_t = tilde_x
                self.y_t = self.compute_proximal_y(c_tilde, beta_t)
                self.mu = min(self.mu_max, self.eta * mu_current)

                # Record history
                self._record_history(t, beta_t, mu_current, inner_iter + 1, True)
                self.current_iteration += 1
                return True
            else:
                # Unsuccessful iteration, decrease mu
                mu_current = self.rho * mu_current
                inner_iter += 1


        self._record_history(t, beta_t, mu_current, inner_iter, False)
        self.current_iteration += 1
        return False

    def _compute_param_norm_diff(self, params1: Dict, params2: Dict) -> float:
        """Compute Euclidean norm of difference between parameter sets"""
        total_norm_sq = 0.0
        for key in params1.keys():
            diff = params1[key] - params2[key]
            total_norm_sq += jnp.sum(diff ** 2)
        return float(jnp.sqrt(total_norm_sq))

    def _compute_param_norm_diff_squared(self, params1: Dict, params2: Dict) -> float:
        """Compute squared Euclidean norm of difference between parameter sets"""
        total_norm_sq = 0.0
        for key in params1.keys():
            diff = params1[key] - params2[key]
            total_norm_sq += jnp.sum(diff ** 2)
        return float(total_norm_sq)

    def _record_history(self, iteration: int, beta_t: float, mu: float,
                        inner_iter: int, is_successful: bool):
        """Record iteration history"""
        # Current values
        obj_val = self.problem.objective_function_scalar(self.x_t)
        # Compute c(x)
        c_values = self.problem.compute_constraints(self.x_t)
        c_norm = jnp.linalg.norm(c_values)

        # Compute ||c(x) - y||
        c_minus_y_norm = jnp.linalg.norm(c_values - self.y_t)
        l1_norm = self.problem.l1_norm_scalar(self.x_t)
        linf_norm = self.problem.linf_norm(self.x_t)


        # x difference from previous
        if len(self.history['x_diff_norms']) > 0 and self.history['x_values']:
            prev_x = self.history['x_values'][-1]
            x_diff_norm = self._compute_param_norm_diff(self.x_t, prev_x)
        else:
            x_diff_norm = 0.0

        # Store history
        self.history['iterations'].append(iteration + 1)
        self.history['objective_values'].append(obj_val)
        self.history['constraint_norms'].append(float(c_norm))
        self.history['c_minus_y_norms'].append(float(c_minus_y_norm))
        self.history['x_diff_norms'].append(x_diff_norm)
        self.history['mu_values'].append(mu)
        self.history['beta_values'].append(beta_t)
        self.history['inner_iter_counts'].append(inner_iter)
        self.history['is_successful'].append(is_successful)
        self.history['l1_norms'].append(l1_norm)
        self.history['linf_norms'].append(linf_norm)
        self.history['x_values'].append({k: v.copy() for k, v in self.x_t.items()})

    def _print_progress(self, iteration: int, success: bool):
        """Print iteration progress"""
        if iteration == 1:
            print(f"\n{'Iter':>6s} {'Objective':>12s} {'||c(x)-y||':>10s}  "
                  f"{'||x_diff||':>10s} {'beta_t':>10s} {'mu':>10s} {'Inner':>8s} ")
            print("-" * 90)

        if len(self.history['objective_values']) == 0:
            return

        obj_val = self.history['objective_values'][-1]
        c_norm = self.history['constraint_norms'][-1]
        l1_norm = self.history['l1_norms'][-1]
        x_diff = self.history['x_diff_norms'][-1]
        beta_t = self.history['beta_values'][-1]
        mu_val = self.history['mu_values'][-1]
        inner_iter = self.history['inner_iter_counts'][-1]
        c_minus_y_norm = self.history['c_minus_y_norms'][-1]
        # status = "Success" if success else "Fail"

        print(f"{iteration:6d} {obj_val:12.6e} {c_minus_y_norm:10.3e}  "
              f"{x_diff:10.3e} {beta_t:10.3e} {mu_val:10.3e} {inner_iter:8d} ")


    def optimize(self, max_iterations: int = 1000, print_freq: int = 100) -> Dict:
        """
        Run SDCAM_1L optimization
        """
        print("\n" + "=" * 60)
        print("Starting SDCAM_1L Optimization")
        print(f"Constraint: l_inf ball with radius C = {self.problem.C:.4f}")
        print("=" * 60)

        # Initialize if not already initialized
        if self.x_t is None:
            self.initialize()

        for iteration in range(max_iterations):
            success = self.run_iteration()

            if (iteration + 1) % print_freq == 0 or iteration == 0:
                self._print_progress(iteration + 1, success)


        print("\n" + "=" * 60)
        print("SDCAM_1L Optimization Complete")
        print("=" * 60)

        return self.x_t

    def plot_convergence(self, save_path: Optional[str] = None):
        """
    Plot convergence curves - only objective function, ||c(x) - y|| and (1/μ) * x_diff
    """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].semilogy(self.history['iterations'], self.history['objective_values'])
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Objective value')
        axes[0].set_title('Objective function ')
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(self.history['iterations'], self.history['c_minus_y_norms'])
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel(r'$\|c(x^t) - y^t\|$')
        axes[1].set_title(r'$\|c(x) - y\|$')
        axes[1].grid(True, alpha=0.3)


        if len(self.history['x_diff_norms']) > 1 and len(self.history['mu_values']) > 1:
            start_idx = 1
            iterations = self.history['iterations'][start_idx:]
            x_diffs = self.history['x_diff_norms'][start_idx:]

            mu_values = self.history['mu_values'][-len(x_diffs):]

            one_over_mu_xdiff = []
            for i in range(min(len(x_diffs), len(mu_values))):
                one_over_mu_xdiff.append(x_diffs[i] / (mu_values[i] + 1e-12))

            axes[2].semilogy(iterations[:len(one_over_mu_xdiff)], one_over_mu_xdiff)
        else:
            axes[2].plot([], [])

        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel(r'$(1/\mu_t) \cdot \|x^{t+1} -x^t\|$')
        axes[2].set_title(r'$(1/\mu_t) \cdot \|x^{t+1} -x^t\|$')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            # 分别保存三个图表
            base_path = save_path.rstrip('.png')

            # 保存目标函数图表
            fig_obj, ax_obj = plt.subplots(figsize=(8, 6))
            ax_obj.semilogy(self.history['iterations'], self.history['objective_values'])
            ax_obj.set_xlabel('Iteration')
            ax_obj.set_ylabel('Objective Value')
            ax_obj.set_title('Objective Function Convergence')
            ax_obj.grid(True, alpha=0.3)
            plt.tight_layout()
            obj_path = f"{base_path}_objective.png"
            plt.savefig(obj_path, dpi=300, bbox_inches='tight')
            plt.close(fig_obj)
            print(f"Objective function plot saved to: {obj_path}")

            # 保存 ||c(x) - y|| 图表
            fig_cnorm, ax_cnorm = plt.subplots(figsize=(8, 6))
            ax_cnorm.semilogy(self.history['iterations'], self.history['c_minus_y_norms'])
            ax_cnorm.set_xlabel('Iteration')
            ax_cnorm.set_ylabel('||c(x) - y||')
            ax_cnorm.set_title('||c(x) - y|| Convergence')
            ax_cnorm.grid(True, alpha=0.3)
            plt.tight_layout()
            cnorm_path = f"{base_path}_c_minus_y_norm.png"
            plt.savefig(cnorm_path, dpi=300, bbox_inches='tight')
            plt.close(fig_cnorm)
            print(f"||c(x) - y|| plot saved to: {cnorm_path}")

            # 保存 (1/μ) * x_diff 图表
            fig_xdiff, ax_xdiff = plt.subplots(figsize=(8, 6))
            if len(self.history['x_diff_norms']) > 1 and len(self.history['mu_values']) > 1:
                ax_xdiff.semilogy(iterations[:len(one_over_mu_xdiff)], one_over_mu_xdiff)
            ax_xdiff.set_xlabel('Iteration')
            ax_xdiff.set_ylabel(r'$(1/\mu_t) \cdot \|x^{t+1} -x^t\|$')
            ax_xdiff.set_title(r'$(1/\mu_t) \cdot \|x^{t+1} -x^t\|$')
            ax_xdiff.grid(True, alpha=0.3)
            plt.tight_layout()
            xdiff_path = f"{base_path}_x_diff.png"
            plt.savefig(xdiff_path, dpi=300, bbox_inches='tight')
            plt.close(fig_xdiff)
            print(f"(1/μ) * ||x_diff|| plot saved to: {xdiff_path}")


            combo_path = f"{base_path}_combined.png"
            plt.savefig(combo_path, dpi=300, bbox_inches='tight')
            print(f"Combined plot saved to: {combo_path}")
            plt.close(fig)
        else:
            plt.show()

    def get_convergence_summary(self) -> Dict:
        """Get convergence summary"""
        if not self.history['iterations']:
            return {}

        linf_norm = self.problem.linf_norm(self.x_t)
        l1_norm = self.problem.l1_norm_scalar(self.x_t)

        return {
            'total_iterations': len(self.history['iterations']),
            'final_objective': self.history['objective_values'][-1],
            'final_constraint_norm': self.history['constraint_norms'][-1],
            'final_x_diff_norm': self.history['x_diff_norms'][-1],
            'final_l1_norm': l1_norm,
            'final_linf_norm': linf_norm,
            'final_beta': self.history['beta_values'][-1],
            'final_mu': self.history['mu_values'][-1],
            'successful_iterations': sum(self.history['is_successful']),
            'failed_iterations': len(self.history['is_successful']) - sum(self.history['is_successful']),
            'in_constraint': linf_norm <= self.problem.C + 1e-8
        }