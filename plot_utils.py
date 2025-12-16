import matplotlib.pyplot as plt
import os
from typing import List, Dict


def plot_beta_comparison(histories: List[Dict],
                         colors: List[str],
                         beta_0_values: List[float],
                         save_dir: str = "beta_comparison_results"):


    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plot 1: Objective function comparison
    fig, ax = plt.subplots(figsize=(10, 7))
    for history, color, beta_0 in zip(histories, colors, beta_0_values):
        iterations = history['iterations']
        obj_values = history['objective_values']
        ax.semilogy(iterations, obj_values, color=color,
                    label=f'$\\beta_0 = {beta_0:.1e}$', linewidth=2.5)

    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel('Objective Value', fontsize=13)
    ax.set_title('Objective Function Convergence', fontsize=15, pad=15)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    obj_path = os.path.join(save_dir, "objective_comparison.png")
    plt.savefig(obj_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Objective function plot saved to: {obj_path}")

    # Plot 2: ||c(x) - y|| comparison
    fig, ax = plt.subplots(figsize=(10, 7))
    for history, color, beta_0 in zip(histories, colors, beta_0_values):
        iterations = history['iterations']
        c_minus_y_norms = history['c_minus_y_norms']
        ax.semilogy(iterations, c_minus_y_norms, color=color,
                    label=f'$\\beta_0 = {beta_0:.1e}$', linewidth=2.5)

    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel(r'$\|c(x^t) - y^t\|$', fontsize=13)
    ax.set_title(r'$\|c(x^t) - y^t\|$ Convergence', fontsize=15, pad=15)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    cnorm_path = os.path.join(save_dir, "c_minus_y_norm_comparison.png")
    plt.savefig(cnorm_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"||c(x) - y|| plot saved to: {cnorm_path}")

    # Plot 3: (1/mu) * x_diff comparison
    fig, ax = plt.subplots(figsize=(10, 7))
    for history, color, beta_0 in zip(histories, colors, beta_0_values):
        if len(history['iterations']) > 1:
            # Get data starting from iteration 1
            start_idx = 1
            iterations = history['iterations'][start_idx:]
            x_diffs = history['x_diff_norms'][start_idx:]
            mu_values = history['mu_values']

            # Ensure we have enough mu_values
            if len(mu_values) >= len(x_diffs):
                available_mu = mu_values[-len(x_diffs):]
            else:
                available_mu = mu_values

            # Calculate (1/mu) * x_diff
            one_over_mu_xdiff = []
            for j in range(min(len(x_diffs), len(available_mu))):
                one_over_mu_xdiff.append(x_diffs[j] / (available_mu[j] + 1e-12))

            if one_over_mu_xdiff:
                ax.semilogy(iterations[:len(one_over_mu_xdiff)],
                            one_over_mu_xdiff, color=color,
                            label=f'$\\beta_0 = {beta_0:.1e}$', linewidth=2.5)

    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel(r'$(1/\mu_t) \cdot \|x^{t+1} - x^t\|$', fontsize=13)
    ax.set_title(r'$(1/\mu_t) \cdot \|x^{t+1} - x^t\|$', fontsize=15, pad=15)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    xdiff_path = os.path.join(save_dir, "x_diff_comparison.png")
    plt.savefig(xdiff_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"(1/μ) * x_diff plot saved to: {xdiff_path}")


def print_summary(optimizers: List, beta_0_values: List[float]):
    """Print summary table of all experiments"""
    print("\n" + "=" * 70)
    print(f"{'Beta_0 Comparison Summary':^70}")
    print("=" * 70)
    print(f"{'beta_0':<12} {'Final Obj':<15} {'Final ||c-y||':<15} {'Iterations':<12}")
    print("-" * 70)

    for optimizer, beta_0 in zip(optimizers, beta_0_values):
        history = optimizer.history
        if history['objective_values'] and history['c_minus_y_norms']:
            final_obj = history['objective_values'][-1]
            final_c_minus_y = history['c_minus_y_norms'][-1]
            iterations = len(history['iterations'])
            print(f"{beta_0:<12.2e} {final_obj:<15.6e} {final_c_minus_y:<15.6e} {iterations:<12}")

    print("=" * 70)