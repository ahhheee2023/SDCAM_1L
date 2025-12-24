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

**Where:**
* **$v$**: the MLP parameters.
* **$\rho(u) = \frac{|u|^p}{p}$**: the loss function where $0 < p \le 1$.
* **$C = \{x : \|x\|_\infty \le \mathcal{C}\}$**: the $\ell_\infty$-ball constraint.
* **$\lambda$**: the regularization coefficient for $L_1$ penalty.

#### **Algorithm Settings**
* **Penalty Parameter**: $\beta_t = \beta_0(t+1)^\delta$ with $\delta = 0.5$.
* **Initialization**: Xavier initialization projected onto $C$, with $y^0 = 0$.

#### **Usage**
```bash
python main.py
