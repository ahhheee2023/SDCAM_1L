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
```


### **Project 2: QCQP**
<font size="4">Implementation of the algorithm for solving a penalized Quadratically Constrained Quadratic Program (QCQP).</font>

#### **Mathematical Formulation**
The objective is to solve the following optimization problem:

$$\min_{x} \frac{1}{2} x^T Q_0 x + b_0^T x + \alpha \|x\|_p^p \quad \text{s.t. } \frac{1}{2} x^T Q_i x + b_i^T x + c_i \le 0, \quad \|x\|_\infty \le r$$

**Where** $Q_0$ is the identity matrix, $b_0$ is a scaled vector with standard Gaussian entries, $Q_i$ are positive semi-definite matrices constructed via random orthogonal matrices and diagonal matrices, $c_i$ are constraints constants derived from a reference minimizer $\bar{x}$, $\alpha$ is the regularization parameter, and $r$ is the radius of the $\ell_\infty$-ball constraint defined by $\|\bar{x}\|_\infty$.

#### **Code Structure**
* ** `run_code.py`**: The main script to execute QCQP experiments, supporting tests for different $p$ values and $\beta_0$ settings.
* **`sdcam_1l.py`**:  Core solver implementing the $\text{SDCAM}_{\mathbb{1}\ell}$ algorithm for the QCQP model, including feasibility checks and dual variable updates.
* **`create_problem3.py`**: Generates the synthetic QCQP problem instance, including matrices $Q_i$, vectors $b_i$, and the reference point $\bar{x}$.
* **`lp_prox_mapping.py`**: Implements the Newton-based root-finding scheme for the $L_p$ proximal mapping with $\ell_\infty$ norm constraints.

#### **Usage**
```bash
python run_code.py
