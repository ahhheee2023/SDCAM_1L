import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, List
import numpy as np


class OptimizationProblem:
    """
    Optimization problem with autodiff support
    Uses l_inf ball constraint and L1 regularization
    """

    def __init__(self, mlp: 'MLP', dataset: Tuple,
                 p: float = 0.5, lambda_reg: float = 0.01):
        """
        Args:
            mlp: MLP model
            dataset: (A, Y) where A in R^{m x n0}, Y in R^m
            p: parameter for rho function, p in (0,1)
            lambda_reg: regularization coefficient for L1 penalty
        """
        self.mlp = mlp
        self.A, self.Y = dataset
        self.m = self.A.shape[0]  # Number of samples
        self.p = p
        self.lambda_reg = lambda_reg

        # Compute constraint set C (l_inf ball radius)
        self.C = self._compute_constraint_set()

        # JIT compile key functions
        self._compiled_objective = jax.jit(self._objective_function_internal)
        self._compiled_gradient = jax.jit(jax.grad(self._objective_function_internal))

    def _compute_constraint_set(self) -> float:
        """Compute constraint set C (l_inf ball radius)"""
        zero_params = {}
        for key in self.mlp.v.keys():
            zero_params[key] = jnp.zeros_like(self.mlp.v[key])

        predictions = jax.vmap(lambda a: self.mlp.forward(zero_params, a))(self.A)
        residuals = predictions - self.Y
        total_rho = jnp.sum(self.rho_function(residuals))

        # Radius for l_inf ball constraint
        radius = float(total_rho) / self.lambda_reg/self.m
        return radius

    def rho_function(self, u) -> jnp.ndarray:
        """rho(u) = |u|^p / p"""
        return jnp.abs(u) ** self.p / self.p

    def _objective_function_internal(self, params: Dict) -> jnp.ndarray:
        """
        Vectorized objective function with average loss per sample

        Returns: (1/m) * sum {rho(MLP(a_i; v) - y_i)} + lambda * ||v||_1
        """
        predictions = jax.vmap(lambda a: self.mlp.forward(params, a))(self.A)
        residuals = predictions - self.Y

        # Average rho-loss over all samples
        data_fit = jnp.mean(self.rho_function(residuals))

        # Regularization term: L1 penalty
        reg_term = self.lambda_reg * self.l1_norm(params)

        return data_fit + reg_term

    def objective_function(self, params: Dict) -> jnp.ndarray:
        """Compute objective function value"""
        return self._compiled_objective(params)

    def objective_function_scalar(self, params: Dict) -> float:
        """Compute objective function and return as Python float"""
        return float(self.objective_function(params))

    def compute_gradient(self, params: Dict) -> Dict:
        """
        Compute gradient using automatic differentiation
        """
        return self._compiled_gradient(params)

    def l1_norm(self, params: Dict) -> jnp.ndarray:
        """Compute L1 norm of parameters ||v||_1 = sum|v_i|"""
        total_norm = 0.0
        for key in params.keys():
            total_norm += jnp.sum(jnp.abs(params[key]))
        return total_norm

    def l1_norm_scalar(self, params: Dict) -> float:
        """Compute L1 norm and return as Python float"""
        return float(self.l1_norm(params))

    def linf_norm(self, params: Dict) -> float:
        """Compute l_inf norm of parameters ||v||_inf = max|v_i|"""
        max_abs = 0.0
        for key in params.keys():
            max_abs = max(max_abs, float(jnp.max(jnp.abs(params[key]))))
        return max_abs

    def project_to_constraint_set(self, params: Dict) -> Dict:
        """
        Project onto l_inf ball constraint set C: {v: ||v||_inf <= C}

        Projection: v_proj = sign(v) * min(|v|, C)
        """
        projected_params = {}
        for key in params.keys():
            projected_params[key] = jnp.sign(params[key]) * jnp.minimum(
                jnp.abs(params[key]), self.C
            )
        return projected_params

    def compute_constraints(self, params: Dict) -> jnp.ndarray:
        """
        Compute constraint values c(v) = MLP(a_i; v) - y_i for all samples
        """
        predictions = jax.vmap(lambda a: self.mlp.forward(params, a))(self.A)
        return predictions - self.Y

    def constraint_violation_scalar(self, params: Dict) -> float:
        """
        Compute constraint violation norm
        """
        constraints = self.compute_constraints(params)
        return float(jnp.linalg.norm(constraints))

    def compute_constraint_jacobian(self, params: Dict) -> Dict:
        """
        Compute Jacobian matrix of constraints J_c(v)
        """

        def constraint_vector(v):
            predictions = jax.vmap(lambda a: self.mlp.forward(v, a))(self.A)
            return predictions - self.Y

        return jax.jacfwd(constraint_vector)(params)



    def compute_gradient_term_vjp(self, params: Dict, y: jnp.ndarray, beta: float) -> Dict:

        # Define function for vjp
        def constraints_func(p):
            return self.compute_constraints(p)

        # Use vjp to compute J_c^T * (c - y) efficiently
        c_values, vjp_func = jax.vjp(constraints_func, params)
        c_minus_y = c_values - y
        gradient_term = vjp_func(c_minus_y)[0]

        # Multiply by beta
        for key in gradient_term.keys():
            gradient_term[key] = beta * gradient_term[key]

        #Chenck the gradient_term
        check_count = 0
        test_d = {}
        eps = 1e-6
        if check_count:
            for key in params.keys():
                shape = params[key].shape

                rand_dir = np.random.randn(*shape)
                norm = np.sqrt(np.sum(rand_dir ** 2))
                test_d[key] = jnp.array(rand_dir / (norm + 1e-12))


            #left hand side
            left_sum = 0.0
            for key in gradient_term.keys():
                left_sum += jnp.sum((gradient_term[key] / beta) * test_d[key])

            #right hand side
            params_eps = {}
            for key in params.keys():
                params_eps[key] = params[key] + eps * test_d[key]

            c_eps = self.compute_constraints(params_eps)
            delta_c_square = 0.5*(np.linalg.norm(c_eps) ** 2 - np.linalg.norm(c_minus_y) ** 2) / eps

            print(f'delta_c_square: {delta_c_square}------J_c_term:{left_sum}' )

        return gradient_term

    def soft_thresholding(self, params: Dict, threshold: float) -> Dict:
        """
        Apply soft thresholding operation for L1 regularization
        Returns: sign(v) * max(|v| - threshold, 0)
        """
        thresholded_params = {}
        for key in params.keys():
            thresholded_params[key] = jnp.sign(params[key]) * jnp.maximum(
                jnp.abs(params[key]) - threshold, 0.0
            )
        return thresholded_params

    def check_feasibility(self, params: Dict) -> bool:
        """Check if parameters are within l_inf ball constraint"""
        return self.linf_norm(params) <= self.C + 1e-8

    def compute_g_value(self, params: Dict) -> float:
        """
        Compute g(v) = delta_C(v) + lambda * ||v||_1
        Returns lambda * ||v||_1 if v in C, otherwise infinity
        """
        if not self.check_feasibility(params):
            return float('inf')
        return float(self.lambda_reg * self.l1_norm(params))