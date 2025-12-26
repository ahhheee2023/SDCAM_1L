import numpy as np
import matplotlib.pyplot as plt
from create_problem3 import create_problem3
from sdcam_1l import SDCAM_1L
import time
import os

def format_beta_label(beta_value):
    exponent = int(np.floor(np.log10(beta_value)))
    coeff = beta_value / (10 ** exponent)

    if abs(coeff - 1.0) < 1e-12:
        return r'$\beta_0 = 10^{' + f'{exponent}' + '}$'
    else:
        if abs(coeff - int(coeff)) < 1e-12:
            coeff_str = str(int(coeff))
        else:
            coeff_str = f'{coeff:.1f}'.rstrip('0').rstrip('.')
        return r'$\beta_0 = ' + coeff_str + r'\times10^{' + f'{exponent}' + '}$'



def run_experiment():
    seed = 2046
    np.random.seed(seed)
    save_dir = "sdcam_results"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

        plt.rcParams['text.usetex'] = True
        # Plot 1: Relative Feasibility
        fig, ax = plt.subplots(figsize=(10, 7))
        for history, color, beta_0 in zip(histories, colors, beta_0_values):
            iterations = history['iterations']
            rel_feas = history['rel_feas']
            label = format_beta_label(beta_0)
            ax.loglog(iterations, rel_feas, color=color,
                      label=label, linewidth=2.5)

        ax.set_xlabel('Iteration', fontsize=30)
        ax.set_ylabel('Relative Feasibility', fontsize=30)
        ax.legend(fontsize=20, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=20)

        plt.tight_layout()
        feas_path = os.path.join(save_dir, f"relative_feasibility_all_beta.png")
        plt.savefig(feas_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Relative feasibility plot saved to: {feas_path}")

        # Plot 2: ||x^{t+1} - x^t|| / μ_t
        fig, ax = plt.subplots(figsize=(10, 7))
        for history, color, beta_0 in zip(histories, colors, beta_0_values):
            if len(history['iterations']) > 1:
                iterations = history['iterations']
                x_diff_over_mu = history['x_diff_over_mu_square']
                label = format_beta_label(beta_0)
                ax.loglog(iterations, x_diff_over_mu, color=color,
                          label=label, linewidth=2.5)

        ax.set_xlabel('Iteration', fontsize=30)
        ax.set_ylabel(r'$\|x^{t+1}-x^t\|/\mu_t$', fontsize=30)
        ax.legend(fontsize=20, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=20)

        plt.tight_layout()
        diff_path = os.path.join(save_dir, f"x_diff_over_mu_all_beta.png")
        plt.savefig(diff_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"x_diff_over_mu plot saved to: {diff_path}")
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
