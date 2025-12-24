import numpy as np
import matplotlib.pyplot as plt
from create_problem3 import create_problem3
from sdcam_1l import SDCAM_1L
import time




def run_experiment():
    seed = 2046
    np.random.seed(seed)

    n = 1000
    m = int(np.ceil(n / 10))
    alpha = 0.05

    # Choose p value
    p = 0.8  # Change this to test p<1 or p=1

    print(f'=== SDCAM Algorithm - Testing p={p:.1f} ===\n')

    if p < 1:
        # ========== p < 1: Test different beta_0 values ==========
        beta_0_values = [1e-4, 1e-2, 1]
        colors = ['b', 'r', 'g']
        delta = 0.3

        # Create problem
        problem = create_problem3(n, m, alpha, p)
        histories = []

        for beta_idx, beta_0 in enumerate(beta_0_values):
            print(f'\n--- Testing beta_0 = {beta_0:.1e} ---\n')

            # Solver parameters
            solver_params = {
                'mu': 1.0,
                'mu_max': 1e7,
                'rho': 0.8,
                'eta': 1.2,
                'delta': delta,
                'max_inner_iter': 100,
                'tol': 1e-10,
                'max_iter': 3000,
                'print_freq': 100
            }

            # Initialize variables
            x_t = -problem['b0'].copy()
            y_t = np.zeros(m)

            print(f'Starting iterations (max: {solver_params["max_iter"]})')

            start_time = time.time()

            # Run SDCAM algorithm
            x_next, y_next, x_diff_norm, mu, inner_iter, history = SDCAM_1L(
                x_t, y_t, beta_0, None, problem, solver_params
            )

            end_time = time.time()

            print(f'beta_0 = {beta_0:.1e}: completed {len(history["iterations"])} '
                  f'iterations in {end_time - start_time:.2f} seconds')

            histories.append(history)

        # Plot
        plt.figure(figsize=(10, 6))
        for beta_idx, (beta_0, color) in enumerate(zip(beta_0_values, colors)):
            history = histories[beta_idx]
            plt.loglog(history['iterations'], history['rel_feas'],
                       color=color, linewidth=1,
                       label=f'beta_0={beta_0:.1e}')
        plt.xlabel('Iterations')
        plt.ylabel('Relative Feasibility',fontsize= 15)
        plt.title(f'Relative Feasibility (p={p:.1f})', fontsize= 15)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('plot_feasibility_all_beta.png', dpi=300, bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(10, 6))
        for beta_idx, (beta_0, color) in enumerate(zip(beta_0_values, colors)):
            history = histories[beta_idx]
            plt.loglog(history['iterations'], history['x_diff_over_mu_square'],
                       color=color, linewidth=1,
                       label=f'beta_0={beta_0:.1e}')
        plt.xlabel('Iterations')
        plt.ylabel(r'$\|x^{t+1}-x^t\|/\mu_t$', fontsize= 15)
        plt.title(r'$\|x^{t+1}-x^t\|/\mu_t$ (p=0.8)', fontsize= 15)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('plot_diff_over_mu_square_all_beta.png', dpi=300, bbox_inches='tight')
        plt.show()

    else:
        # ========== p = 1: Test different delta values ==========
        beta_0 = 0.01
        delta_values = [0.15, 0.3, 0.45]
        max_iterations = 1000
        histories = []

        print(f'Starting iterations (max: {max_iterations})')

        for delta in delta_values:
            # Create problem
            problem = create_problem3(n, m, alpha, p)

            solver_params = {
                'mu': 1.0,
                'mu_max': 1e7,
                'rho': 0.8,
                'eta': 1.2,
                'delta': delta,
                'max_inner_iter': 100,
                'tol': 1e-10,
                'max_iter': max_iterations,
                'print_freq': 1000
            }

            x_t = -problem['b0']
            y_t = np.zeros(m)

            start_time = time.time()

            # Run SDCAM algorithm
            x_next, y_next, x_diff_norm, mu, inner_iter, history = SDCAM_1L(
                x_t, y_t, beta_0, None, problem, solver_params
            )

            end_time = time.time()

            print(f'delta = {delta:.2f}: completed {len(history["iterations"])} '
                  f'iterations in {end_time - start_time:.2f} seconds')

            histories.append(history)



if __name__ == '__main__':
    run_experiment()