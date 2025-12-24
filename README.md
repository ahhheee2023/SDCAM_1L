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

$$\begin{aligned}
\min_{x\in \mathbb{R}^n} \quad & \frac{1}{2} x^TQ_0x+ b_0^Tx+ \alpha \|x\|^p_p, \\
\text{s.t.} \quad & \frac{1}{2} x^TQ_ix+ b_i^Tx+ r_i \le 0, \quad i=1,2,\cdots, m, \\
& \|x\|_\infty \le r,
\end{aligned}$$

**Where** $Q_0$ is the identity matrix, $b_0$ is a scaled vector with standard Gaussian entries, $Q_i$ are positive semi-definite matrices, $c_i$ are constraints constants , $\alpha$ is the regularization parameter, and $r$ is the radius of the $\ell_\infty$-ball constraint.

#### **Code Structure**
* **`run_code.py`**: running QCQP experiments, supporting tests for different $p$ values and $\beta_0$ settings.
* **`sdcam_1l.py`**:  Core solver implementing the $\text{SDCAM}_{\mathbb{1}\ell}$ algorithm for the QCQP model.
* **`create_problem3.py`**: Generating the  QCQP problem instance, including matrices $Q_i$, vectors $b_i$, for $i=0,cdots,m$, and the $r$.
* **`lp_prox_mapping.py`**: Implementing the Newton-based root-finding scheme for the $\ell_p$ proximal mapping with $\ell_\infty$ norm constraints.

#### **Usage**
```bash
python run_code.py
