import jax
import jax.numpy as jnp
from mlp import MLP
from problem import OptimizationProblem
from sdcam import SDCAM_1L
from data_loader import load_mnist_subset
from plot_utils import plot_beta_comparison, print_summary
import os

# Enable float64 precision in JAX
jax.config.update("jax_enable_x64", True)


def run_single_experiment(beta_0, data_directory):
    """Run optimization with specific beta_0 value"""
    print(f"\nRunning experiment with beta_0 = {beta_0:.2e}")
    print("-" * 50)

    # Load MNIST dataset
    A, Y = load_mnist_subset(
        num_samples=1000,
        input_dim=784,
        data_dir=data_directory
    )

    # Create MLP
    mlp = MLP([784, 128, 64, 1], activation='tanh')

    # Create optimization problem
    problem = OptimizationProblem(mlp, (A, Y), p=0.5, lambda_reg=0.05)
    initial_params = mlp.get_parameters()
    projected_params = problem.project_to_constraint_set(initial_params)
    mlp.set_parameters(projected_params)

    # Create and initialize SDCAM optimizer
    optimizer = SDCAM_1L(
        problem=problem,
        mu_max=1e7,
        mu_init=0.01,
        rho=0.5,
        eta=2,
        beta_0=beta_0,
        beta_delta=0.5,
        max_inner_iter=1e7,
        tol=1e-6,
        newton_max_iter=100,
        newton_tol=1e-8
    )

    # Initialize and run optimization
    optimizer.initialize()
    optimized_params = optimizer.optimize(max_iterations=3000, print_freq=100)

    return optimizer


def main():
    # Data directory
    data_directory = os.path.join(os.getcwd(), 'data', 'MNIST', 'raw')

    # Define beta_0 values and corresponding colors
    beta_0_values = [1e-5, 1e-3, 1e-2]
    colors = ['r', 'g', 'b']  # Red, Green, Blue

    # Run all experiments
    optimizers = []

    for beta_0 in beta_0_values:
        optimizer = run_single_experiment(beta_0, data_directory)
        optimizers.append(optimizer)

    # Get histories for plotting
    histories = [optimizer.history for optimizer in optimizers]

    # Plot comparison
    plot_beta_comparison(histories, colors, beta_0_values)

    # Print summary
    print_summary(optimizers, beta_0_values)


if __name__ == "__main__":
    main()