# <h1 align="center">SDCAM Numerical Experiments</h1>

<p align="center">
  <b>Implementation of the Algorithm:</b>
</p>

$$\large \text{SDCAM}_{\mathbb{1}\ell}$$

<p align="center">
  <i>Verified on Python 3.8+ | Powered by JAX & NumPy</i>
</p>

---

## 1. Reference Paper
The implementations in this repository are based on the following research paper:
* **Paper Title**: Complexity and convergence analysis of a single-loop SDCAM for Lipschitz composite optimization and beyond
* **Link**: <a href="">View Paper on </a>

---

## 2. Environment & Requirements

### **Python Version**
Required: `Python 3.8` or higher.

### **Dependencies**
| Library | Purpose |
| :--- | :--- |
| **JAX** | High-performance autodiff & Vector-Jacobian Product |
| **NumPy** | Newton-based $L_p$ proximal mapping logic |
| **Matplotlib** | Convergence and performance visualization |

---

## 3. Projects Organization

### **Project 1: MLP**
<font size="4">This project applies the algorithm to a Multi-Layer Perceptron (MLP) using the MNIST dataset.</font>

#### **Mathematical Formulation**
The objective is to solve the following optimization problem:

$$\min_{v} \frac1m\sum_{i =1}^m \rho\left({\rm MLP}(a_i;v) - y_i\right) + \lambda \|v\|_1$$

**Where** $v$ represents the MLP parameters, $m$ represents the number of samples in the dataset, $\rho(u) = \frac{|u|^p}{p}$ is the loss function where $0 < p \le 1$, $C$ is the $\ell_\infty$-ball constraint, and $\lambda$ is the regularization coefficient for $L_1$ penalty.

#### **Code Structure**
* **`main.py`**: The entry point of the project. It loads the dataset, initializes the MLP, and runs the SDCAM optimization with different $\beta_0$ values.
* **`sdcam.py`**: Contains the `SDCAM_1L` class, implementing the core idea of $\text{SDCAM}_{\mathbb{1}\ell}$.
* **`problem.py`**: Defines the `OptimizationProblem` class. It contains objective function calculations, constraint projections, and leverages JAX for automatic differentiation and Vector-Jacobian Products.
* **`mlp.py`**: Implements the Multi-Layer Perceptron with 3 layers and activation functions using JAX.
* **`data_loader.py`**: Handles loading and preprocessing the MNIST dataset, including normalization and subset selection.
* **`plot_utils.py`**: Provides utilities for plotting convergence curves (Objective value, $\|c(x)-y\|$, etc.) and printing result summaries.

#### **Usage**
```bash
python main.py
