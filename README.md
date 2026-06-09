<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=32&pause=1000&color=6C63FF&center=true&vCenter=true&width=900&lines=Machine+Learning+%26+Deep+Learning;Transformers+%7C+LangChain+%7C+LLMs;The+Ultimate+Reference+Repository" alt="Typing SVG" />

<br/>

# 🤖 Machine Learning, Deep Learning & LLM Engineering
### The Most Complete Theory-First Reference — Beginner to Production

<br/>

[![Stars](https://img.shields.io/github/stars/ravidaliparthy/Machine-Learning-Deep-Learning-Modules?style=for-the-badge&color=6C63FF)](https://github.com/ravidaliparthy)
[![Forks](https://img.shields.io/github/forks/ravidaliparthy/Machine-Learning-Deep-Learning-Modules?style=for-the-badge&color=FF6B6B)](https://github.com/ravidaliparthy)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **Every algorithm. Every derivation. Every formula. Every intuition.**
> Theory → Math → Code → Production.

<br/>

</div>

---

## 📑 Table of Contents

<details>
<summary><b>Click to expand full contents</b></summary>

1. [Mathematical Foundations](#1-mathematical-foundations)
2. [Core Machine Learning Theory](#2-core-machine-learning-theory)
3. [Data Preprocessing & Feature Engineering](#3-data-preprocessing--feature-engineering)
4. [Regression Algorithms](#4-regression-algorithms)
5. [Classification Algorithms](#5-classification-algorithms)
6. [Clustering & Unsupervised Learning](#6-clustering--unsupervised-learning)
7. [Dimensionality Reduction](#7-dimensionality-reduction)
8. [Ensemble Learning](#8-ensemble-learning)
9. [Time Series Analysis](#9-time-series-analysis)
10. [Deep Learning Foundations](#10-deep-learning-foundations)
11. [Convolutional Neural Networks (CNN)](#11-convolutional-neural-networks-cnn)
12. [Recurrent Neural Networks (RNN)](#12-recurrent-neural-networks-rnn)
13. [LSTM & GRU](#13-long-short-term-memory-lstm--gru)
14. [Advanced Deep Learning](#14-advanced-deep-learning)
15. [🔥 Transformers — Complete Deep Dive](#15-transformers--complete-deep-dive)
16. [🦜 LangChain & LLM Application Engineering](#16-langchain--llm-application-engineering)
17. [Handling Imbalanced Data](#17-handling-imbalanced-data)
18. [Model Evaluation & Selection](#18-model-evaluation--selection)
19. [Regularisation & Optimisation](#19-regularisation--optimisation)
20. [Probabilistic Machine Learning](#20-probabilistic-machine-learning)
21. [Repository Structure](#21-repository-structure)

</details>

---

## 1. Mathematical Foundations

### 1.1 Linear Algebra

#### Vectors and Norms

```
L0 norm:  ||x||₀ = count(xᵢ ≠ 0)             — number of non-zero elements
L1 norm:  ||x||₁ = Σᵢ |xᵢ|                   — Manhattan distance
L2 norm:  ||x||₂ = √(Σᵢ xᵢ²)                 — Euclidean distance
L∞ norm:  ||x||∞ = maxᵢ |xᵢ|                  — Chebyshev distance
Lp norm:  ||x||ₚ = (Σᵢ |xᵢ|ᵖ)^(1/p)
```

**Geometric Intuition:** The unit ball `{x : ||x||ₚ ≤ 1}` changes shape with p:
- p=1 → diamond (corners at axes → sparsity in optimisation)
- p=2 → sphere (smooth → shrinkage without sparsity)
- p=∞ → hypercube

**Why Norms Matter in ML:**
Norms are not merely measurement tools — they encode geometric assumptions about the space in which learning takes place. When you choose a loss function, a regulariser, or a distance metric, you are implicitly choosing a norm, and that choice shapes what the algorithm considers "close", "large", or "important". The L1 norm, for instance, treats all dimensions symmetrarily in terms of absolute deviation but geometrically has sharp corners. These corners are exactly what cause solutions to land on sparse coordinate axes during optimisation — a property exploited by Lasso. The L2 norm, being a smooth ellipsoid, does not produce this corner-touching behaviour, so it shrinks all weights but never zeros any of them exactly.

**Triangle Inequality and its Consequences:**
```
||x + y||ₚ ≤ ||x||ₚ + ||y||ₚ   (triangle inequality, holds for all p ≥ 1)
||x - z||ₚ ≤ ||x - y||ₚ + ||y - z||ₚ  (metric property)
```
This is the foundational property that makes norms valid distance metrics. It ensures that a "detour" through an intermediate point is never shorter than the direct path — a fact relied upon in convergence proofs for gradient-based optimisation and in the proof that clustering objectives are well-defined.

**Dual Norms:**
```
The dual norm of ||·||ₚ is ||·||_q where 1/p + 1/q = 1
Dual of L1 = L∞
Dual of L2 = L2 (self-dual)
Dual of Lp = Lq

Hölder's inequality: |xᵀy| ≤ ||x||ₚ · ||y||_q
Cauchy-Schwarz (p=q=2): |xᵀy| ≤ ||x||₂ · ||y||₂
```
The dual norm appears naturally in subgradient analysis. The subdifferential of `||x||₁` at a non-zero point is `sign(x)`, and at zero it is the entire dual ball `{v : ||v||∞ ≤ 1}`. This is why coordinate descent on L1-penalised objectives leads to soft-thresholding operators.

#### Matrix Operations

```
Transpose:          (AB)ᵀ = BᵀAᵀ
Inverse:            AA⁻¹ = I          (square, non-singular only)
Pseudo-inverse:     A⁺ = VΣ⁺Uᵀ       (Moore-Penrose; works for any shape)
Trace:              tr(A) = Σᵢ Aᵢᵢ = Σᵢ λᵢ
Determinant:        det(A) = Πᵢ λᵢ    (product of all eigenvalues)
Rank:               rank(A) = number of non-zero singular values
Frobenius norm:     ||A||_F = √(Σᵢ Σⱼ Aᵢⱼ²) = √tr(AᵀA) = √(Σᵢ σᵢ²)
Nuclear norm:       ||A||_* = Σᵢ σᵢ   (sum of singular values; convex relaxation of rank)
```

**Useful Identities:**
```
tr(AB) = tr(BA)                        (cyclic property)
∂/∂A tr(AB) = Bᵀ
∂/∂A tr(AᵀA) = 2A
∂/∂x (xᵀAx) = (A + Aᵀ)x = 2Ax   (if A symmetric)
∂/∂x (aᵀx) = a
∂/∂A det(A) = det(A) A⁻ᵀ
```

**Woodbury Matrix Identity (crucial for efficient computation):**
```
(A + UCV)⁻¹ = A⁻¹ - A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹VA⁻¹
```
This identity is pivotal whenever a large matrix `A` is easy to invert but `A + UCV` is not. In ML it appears in: kernel ridge regression (avoiding n×n matrix inversion when n > d), Gaussian Process posterior updates, online learning updates, and computing the inverse of covariance matrices after adding a low-rank perturbation. The key insight is that if `U` and `V` have few columns (low rank), the right-hand side requires inverting only a small matrix.

**Schur Complement:**
```
For block matrix M = [[A, B],[C, D]]:
  det(M) = det(A) · det(D - CA⁻¹B)
  M/A = D - CA⁻¹B   (Schur complement of A in M)
```
The Schur complement arises in conditioning of multivariate Gaussians. When marginalising a joint Gaussian `P(x,y)`, the conditional `P(y|x)` has covariance equal to the Schur complement `Σ_yy - Σ_yx Σ_xx⁻¹ Σ_xy`. This is the theoretical basis for Gaussian Process posterior updates.

#### Eigendecomposition

```
Av = λv            (v = eigenvector, λ = eigenvalue)
A = Q Λ Qᵀ         (symmetric A: Q = orthogonal eigenvectors, Λ = diagonal eigenvalues)
```

**Properties:**
- Symmetric matrices always have real eigenvalues and orthogonal eigenvectors
- Positive Definite (PD): all λᵢ > 0 — guarantees unique minimum in quadratic optimisation
- Positive Semi-Definite (PSD): all λᵢ ≥ 0 — required for valid covariance matrices and kernels
- Condition number κ = λ_max / λ_min — large κ → ill-conditioned → optimisation is slow/unstable

**Why Ill-Conditioning Hurts Gradient Descent:**
Consider minimising the quadratic `f(x) = (1/2)xᵀAx` where `A` is PD with condition number `κ`. Gradient descent with step size `α = 2/(λ_min + λ_max)` converges at rate:
```
||xₜ - x*||² ≤ ((κ-1)/(κ+1))^{2t} ||x₀ - x*||²
```
When `κ` is large (say 10,000), the contraction factor `(κ-1)/(κ+1) ≈ 0.9998` means thousands of iterations to converge. The loss landscape is a highly elongated ellipsoid — gradient steps bounce between the steep walls rather than moving directly toward the minimum. Preconditioning (e.g., dividing each coordinate by the corresponding eigenvalue) transforms the landscape to a sphere and restores fast convergence. This is exactly what Adam does adaptively: it normalises gradients by their approximate second moment, achieving an implicit diagonal preconditioning.

**Spectral Theorem:**
Every real symmetric matrix `A ∈ ℝⁿˣⁿ` has a complete orthonormal eigenbasis. The spectral decomposition `A = QΛQᵀ = Σᵢ λᵢ qᵢqᵢᵀ` reveals that `A` is a weighted sum of rank-1 projection matrices `qᵢqᵢᵀ`. Each projection extracts the component of a vector along `qᵢ` and scales it by `λᵢ`. PCA is the application of this theorem to the data covariance matrix: the eigenvectors define directions of maximum variance, and the eigenvalues tell us exactly how much variance lies along each direction.

**Matrix Functions via Eigendecomposition:**
```
For symmetric A = QΛQᵀ:
  Aᵏ = QΛᵏQᵀ
  exp(A) = Q diag(exp(λᵢ)) Qᵀ
  A^{1/2} = Q diag(√λᵢ) Qᵀ   (only if A ≻ 0)
  log(A)  = Q diag(log λᵢ) Qᵀ
```
Matrix exponentials appear in continuous-time dynamical systems (e.g., neural ODEs). Matrix square roots appear in whitening transforms and in the computation of the Mahalanobis distance.

#### Singular Value Decomposition (SVD)

```
A = U Σ Vᵀ
```

Where:
- `U` (m×m): left singular vectors — eigenvectors of `AAᵀ`
- `V` (n×n): right singular vectors — eigenvectors of `AᵀA`
- `Σ` (m×n): diagonal with singular values σᵢ = √λᵢ(AᵀA) ≥ 0

**Truncated (rank-k) SVD:**
```
Aₖ = Uₖ Σₖ Vₖᵀ   (best rank-k approximation by Eckart-Young theorem)
||A - Aₖ||_F² = Σᵢ>k σᵢ²
```

**Eckart-Young Theorem (proof sketch):**
The theorem states that among all rank-k matrices `B`, the truncated SVD minimises `||A - B||_F`. The proof relies on the fact that any rank-k matrix can be written as `B = Σᵢ≤k uᵢσᵢvᵢᵀ` (a sum of k rank-1 terms), and the Frobenius norm decomposes as `||A - B||_F² = Σᵢ (σᵢ - σ̃ᵢ)²` when `B` shares the same singular vectors as `A`. Since the σᵢ are ordered, setting `σ̃ᵢ = σᵢ` for i ≤ k and 0 otherwise is optimal. This underpins: PCA (best linear dimensionality reduction), collaborative filtering (low-rank matrix completion), and data compression.

**SVD and the Four Fundamental Subspaces:**
```
Column space of A   = span(U₁,...,Uᵣ)       (r = rank(A))
Left null space     = span(Uᵣ₊₁,...,Uₘ)
Row space of A      = span(V₁,...,Vᵣ)
Right null space    = span(Vᵣ₊₁,...,Vₙ)
```
Understanding these subspaces clarifies why certain linear systems have solutions, why the normal equations `XᵀXβ = Xᵀy` may have infinitely many solutions (when X is rank-deficient), and why the pseudo-inverse gives the minimum-norm solution.

**Applications:** PCA, LSA (NLP), collaborative filtering, pseudo-inverse, matrix completion.

#### Covariance and Correlation

```
Covariance matrix:   Σ = (1/(n-1)) Xᵀ X     (X is mean-centred, n×d)
Σᵢⱼ = Cov(Xᵢ, Xⱼ) = E[(Xᵢ - μᵢ)(Xⱼ - μⱼ)]

Correlation matrix:  R = D⁻¹/² Σ D⁻¹/²     (D = diag(Σ))
Rᵢⱼ = Σᵢⱼ / (σᵢ σⱼ)   ∈ [-1, 1]
```

**Geometry of Covariance:**
The covariance matrix encodes the shape of the data cloud. Geometrically, `Σ` defines an ellipsoid `{x : xᵀΣ⁻¹x ≤ 1}` — the level set of the Mahalanobis distance. The axes of this ellipsoid are the eigenvectors of `Σ`, and the lengths of the axes are `√λᵢ`. When `Σ = σ²I`, the ellipsoid is a sphere (all directions equally variable). When `Σ` has one dominant eigenvalue, the ellipsoid is elongated in that direction — the data is most spread along the first principal component.

**Sample vs Population Covariance:**
```
Population: Σ = (1/n) Σᵢ (xᵢ-μ)(xᵢ-μ)ᵀ   (biased)
Sample:     S = (1/(n-1)) Σᵢ (xᵢ-x̄)(xᵢ-x̄)ᵀ  (unbiased, Bessel's correction)
```
The `n-1` denominator corrects for the fact that we estimated the mean `μ` from the same data. Replacing the true mean with the sample mean introduces a downward bias in the variance estimate, which the `n-1` correction exactly cancels. For large n the difference is negligible, but for small samples it matters.

---

### 1.2 Calculus & Optimisation

#### Gradient, Jacobian, Hessian

```
Gradient (scalar → vector):
  ∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ   ∈ ℝⁿ

Jacobian (vector → vector):
  J(f)ᵢⱼ = ∂fᵢ/∂xⱼ                            ∈ ℝᵐˣⁿ

Hessian (scalar → matrix of 2nd derivatives):
  H(f)ᵢⱼ = ∂²f / (∂xᵢ ∂xⱼ)                   ∈ ℝⁿˣⁿ
```

**Hessian Definiteness → Identifies Critical Points:**
```
H ≻ 0  (positive definite)  → local minimum
H ≺ 0  (negative definite)  → local maximum
H indefinite                 → saddle point
H ⪰ 0  (PSD)               → local minimum or saddle (need higher-order test)
```

**Convexity and Its Importance:**
A function `f` is convex if for all `x, y` and `λ ∈ [0,1]`:
```
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
```
Geometrically: the chord between any two points on the graph lies above the graph. Convexity is the central structural property that makes optimisation tractable:
- Every local minimum is a global minimum
- The sublevel sets `{x : f(x) ≤ c}` are convex sets
- Gradient descent converges to the global optimum
- Duality theory gives tight bounds via the conjugate function

**First-Order Convexity Condition:**
```
f is convex ⟺ f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)  ∀x,y
```
The tangent hyperplane at any point is a global underestimate. This gives the gradient descent update a clean interpretation: we are moving in the direction of steepest descent of the tangent approximation, and convexity guarantees this does not lead us away from the global minimum.

**Second-Order Convexity Condition:**
```
f is convex ⟺ H(x) ⪰ 0  ∀x   (Hessian is PSD everywhere)
```

#### Chain Rule

```
Scalar:   dz/dx = (dz/dy)(dy/dx)

Vector:   ∂z/∂x = Jᵀ_y→x · (∂z/∂y)     (Jacobian-vector product)

General (backprop):
  If z = f(g(x)):  ∂z/∂xᵢ = Σⱼ (∂z/∂yⱼ)(∂yⱼ/∂xᵢ)
```

**Computational Graph Perspective:**
The chain rule applied to a computational graph is precisely what backpropagation computes. Every operation in a neural network is a node; edges carry values (forward pass) and gradients (backward pass). The key insight of reverse-mode automatic differentiation is that computing `∂L/∂θ` for ALL parameters simultaneously costs only O(1) times the cost of a forward pass, regardless of the number of parameters. This is why training deep networks with millions of parameters is computationally feasible.

**Forward-Mode vs Reverse-Mode AD:**
```
Forward-mode: propagate Jacobian-vector products (JVPs) from inputs → output
  Cost: O(n_inputs) passes to compute full Jacobian
  Efficient when: n_outputs >> n_inputs (e.g., computing sensitivities)

Reverse-mode: propagate vector-Jacobian products (VJPs) from output → inputs
  Cost: O(n_outputs) passes to compute full Jacobian
  Efficient when: n_inputs >> n_outputs (e.g., gradient of scalar loss wrt all weights)
  This is what backpropagation does: 1 pass to get ∂L/∂θ for ALL θ
```

#### Taylor Series & Optimisation Proofs

```
f(x + δ) ≈ f(x) + ∇f(x)ᵀδ + (1/2)δᵀH(x)δ + O(||δ||³)
```

**Newton's Method** (uses 2nd-order approximation):
```
δ* = -H⁻¹ ∇f(x)
x_{t+1} = xₜ - H⁻¹ ∇f(xₜ)
```
Quadratic convergence but O(n³) per step. In ML: used in Fisher scoring, IRLS for logistic regression.

**Gradient Descent Convergence (L-smooth f):**
```
If ||∇²f|| ≤ L (Lipschitz gradient), then GD with α = 1/L converges:
  f(xₜ) - f* ≤ (L||x₀ - x*||²) / (2t)
```

**Derivation of GD Convergence Bound:**
From L-smoothness: `f(y) ≤ f(x) + ∇f(x)ᵀ(y-x) + (L/2)||y-x||²`. Setting `y = x - (1/L)∇f(x)`:
```
f(y) ≤ f(x) - (1/L)||∇f(x)||² + (1/(2L))||∇f(x)||²
      = f(x) - (1/(2L))||∇f(x)||²
```
This gives a decrease of at least `(1/2L)||∇f(x)||²` per step. Summing over t steps and using the fact that the minimum of `||∇f(xₜ)||²` over all steps bounds the average, the O(1/t) convergence rate follows.

**Strong Convexity (μ-strongly convex):**
```
f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (μ/2)||y-x||²
Linear convergence: ||xₜ - x*||² ≤ (1 - μ/L)ᵗ ||x₀ - x*||²
Condition number κ = L/μ — smaller → faster convergence
```

**Conjugate Gradient Method:**
For `Ax = b` (or equivalently, minimising `(1/2)xᵀAx - bᵀx`):
```
Generates conjugate directions p₀, p₁, ..., pₙ₋₁ such that pᵢᵀApⱼ = 0 for i≠j
Exact convergence in n steps (theoretically); in practice, converges in far fewer
Step sizes chosen exactly: αₜ = rₜᵀrₜ / (pₜᵀApₜ)   (rₜ = residual)
Convergence rate: ||xₜ - x*||_A ≤ 2((√κ-1)/(√κ+1))^t ||x₀ - x*||_A
Faster than GD: √κ instead of κ governs the rate
```

---

### 1.3 Probability & Statistics

#### Bayes' Theorem & Its Geometry

```
P(A|B) = P(B|A) P(A) / P(B)

Prior:      P(θ)           — belief before observing data
Likelihood: P(data|θ)      — probability of data given parameter
Posterior:  P(θ|data)      — updated belief after observing data
Evidence:   P(data) = ∫ P(data|θ) P(θ) dθ  — normalisation constant
```

**Bayes as Information Update:**
The posterior is the prior "sharpened" by the data. In the Gaussian case with prior `θ ~ N(μ₀, τ²)` and likelihood `x|θ ~ N(θ, σ²)` for n observations:
```
Posterior: θ|x ~ N(μₙ, τₙ²)
  μₙ = (τ⁻² μ₀ + nσ⁻² x̄) / (τ⁻² + nσ⁻²)   (precision-weighted average)
  τₙ⁻² = τ⁻² + nσ⁻²                           (precisions add)
```
The posterior mean is a convex combination of the prior mean and the data mean, weighted by their precisions (inverse variances). As n → ∞, the posterior collapses onto x̄ (the data overwhelm the prior). As n → 0, it collapses onto μ₀ (we know nothing, fall back to prior). This is the Bayesian analogue of regularisation: the prior pulls the estimate toward μ₀, exactly like L2 regularisation.

**Sequential Bayesian Updating:**
Bayes' theorem is inherently sequential. Given data `x₁, x₂, ..., xₙ` arriving one at a time:
```
P(θ|x₁,...,xₙ) ∝ P(θ) × Πᵢ P(xᵢ|θ)
```
The posterior after n observations becomes the prior for the (n+1)-th observation. This Markov property in belief space means full Bayesian inference is incremental — a fundamental advantage in online learning settings.

#### Expectation, Variance, Covariance

```
E[X] = Σₓ x P(X=x)           (discrete)
E[X] = ∫ x f(x) dx            (continuous)
E[aX + bY] = aE[X] + bE[Y]   (linearity — always holds)

Var(X) = E[(X-μ)²] = E[X²] - (E[X])²
Var(aX + b) = a² Var(X)
Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)

Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] = E[XY] - E[X]E[Y]
Cov(X,Y) = 0 ⟹ uncorrelated  (independent ⟹ uncorrelated, NOT vice versa)
```

**Law of Total Expectation and Total Variance:**
```
Law of Total Expectation: E[X] = E[E[X|Y]]
Law of Total Variance:    Var(X) = E[Var(X|Y)] + Var(E[X|Y])
                                    ← within-group variance + between-group variance
```
The law of total variance is the theoretical basis for the ANOVA decomposition and for understanding variance reduction in ensemble methods. The "within-group" term is the irreducible noise conditioned on Y; the "between-group" term is the variance in the conditional mean — what a model of Y can explain about X.

**Moment Generating Functions (MGF) and Characteristic Functions:**
```
MGF:  M_X(t) = E[e^{tX}]           (exists in neighbourhood of t=0 iff all moments finite)
      E[Xⁿ] = M_X^{(n)}(0)         (nth moment = nth derivative at 0)
      
Characteristic function: φ_X(t) = E[e^{itX}]  (always exists)
      Uniquely determines distribution
      For sums of independent RVs: φ_{X+Y}(t) = φ_X(t) × φ_Y(t)
```
MGFs transform the problem of computing moments into differentiation, and the product rule for independent sums into multiplication — both major simplifications. The Central Limit Theorem is most cleanly proved via characteristic functions.

#### Key Distributions

| Distribution | PMF/PDF | Mean | Variance | Use in ML |
|---|---|---|---|---|
| Bernoulli(p) | pˣ(1-p)^(1-x) | p | p(1-p) | Binary classification output |
| Binomial(n,p) | C(n,x)pˣ(1-p)^(n-x) | np | np(1-p) | Count data |
| Gaussian(μ,σ²) | (1/√(2πσ²))exp(-(x-μ)²/2σ²) | μ | σ² | Continuous output, weight prior |
| Categorical(π) | Πₖ πₖ^xₖ | πₖ | πₖ(1-πₖ) | Multi-class output |
| Dirichlet(α) | ∝ Πₖ xₖ^(αₖ-1) | αₖ/α₀ | — | Prior over Categorical |
| Laplace(μ,b) | (1/2b)exp(-|x-μ|/b) | μ | 2b² | L1 prior, robust regression |
| Beta(α,β) | ∝ xα⁻¹(1-x)^(β-1) | α/(α+β) | — | Prior over Bernoulli/Binomial |
| Exponential(λ) | λe^(-λx) | 1/λ | 1/λ² | Time between events |
| Poisson(λ) | λˣe^(-λ)/x! | λ | λ | Count of rare events |

**Why the Gaussian is Central:**
The Gaussian distribution arises so frequently because of the Central Limit Theorem: the sum of n i.i.d. random variables with finite mean μ and variance σ² converges in distribution to N(nμ, nσ²) as n → ∞, regardless of the original distribution. In ML, this justifies modelling residuals as Gaussian (they are often sums of many small errors), and it underlies the use of squared loss as the canonical regression objective (MLE under Gaussian noise).

**The Gaussian is also Maximum Entropy:** Among all distributions with fixed mean μ and variance σ², the Gaussian maximises the differential entropy `H(X) = -∫ p(x) log p(x) dx`. This makes it the "least informative" distribution given only these two constraints — a useful default when we have no additional structural knowledge.

#### Exponential Family (Unified Framework)

Most common distributions belong to the exponential family:
```
p(x|η) = h(x) exp(ηᵀT(x) - A(η))
```
Where:
- `η`: natural (canonical) parameters
- `T(x)`: sufficient statistics
- `A(η)`: log-partition function (normaliser)
- `h(x)`: base measure

**Key Property:** `∂A/∂η = E[T(x)]`, `∂²A/∂η² = Var[T(x)]` — cumulants from derivatives.

**Why the Exponential Family Matters:**
The exponential family is not just a mathematical convenience — it has deep structural properties that make it the right family for statistical modelling:

1. **Sufficient Statistics:** By the Fisher-Neyman factorisation theorem, T(x) is a sufficient statistic — it captures all information about η that the data contains. Knowing T(x₁,...,xₙ) = Σᵢ T(xᵢ) is equivalent to knowing all the raw data for the purpose of estimating η. This is why, for a Gaussian, only the sample mean and variance matter; for a Bernoulli, only the count of successes matters.

2. **MLE has Closed Form:** The MLE satisfies `∂A/∂η = (1/n)Σᵢ T(xᵢ)` — mean sufficient statistics equal the empirical mean. For the Gaussian, this gives `μ̂ = x̄` directly.

3. **Conjugate Priors exist:** For every exponential family likelihood, there is a conjugate prior in the same family, leading to tractable posterior updates.

4. **GLMs are built on this:** Generalised Linear Models assume the response belongs to an exponential family with a link function connecting the linear predictor to the natural parameter.

#### Information Theory

```
Entropy (uncertainty in X):
  H(X) = -Σₓ P(x) log₂ P(x)    ≥ 0, = 0 if deterministic

Joint Entropy:
  H(X,Y) = -Σₓ Σᵧ P(x,y) log P(x,y)

Conditional Entropy:
  H(Y|X) = H(X,Y) - H(X) = -Σₓ P(x) Σᵧ P(y|x) log P(y|x)

Mutual Information (shared info between X and Y):
  I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)  ≥ 0

KL Divergence (how different Q is from P):
  D_KL(P||Q) = Σₓ P(x) log(P(x)/Q(x))  ≥ 0  (= 0 iff P = Q)
  Non-symmetric: D_KL(P||Q) ≠ D_KL(Q||P)
  Forward KL (P||Q): "mean-seeking" — Q covers all of P's mass
  Reverse KL (Q||P): "mode-seeking" — Q concentrates on P's mode

Cross-Entropy:
  H(P,Q) = H(P) + D_KL(P||Q) = -Σₓ P(x) log Q(x)
  = expected log-loss when using Q to encode P

Jensen-Shannon Divergence (symmetric, bounded):
  JSD(P||Q) = (1/2)D_KL(P||M) + (1/2)D_KL(Q||M),  M = (P+Q)/2
  JSD ∈ [0, 1],  √JSD is a proper metric
```

**Why Cross-Entropy is the Natural Classification Loss:**
Suppose the true distribution of labels given input x is P(y|x), and our model produces Q(y|x). The expected description length of labels under our model (but true distribution P) is H(P,Q) = H(P) + D_KL(P||Q). Since H(P) is fixed (irreducible entropy of the true distribution), minimising the cross-entropy is exactly equivalent to minimising the KL divergence from our model to the true distribution. This is also MLE: maximising log-likelihood Σᵢ log Q(yᵢ|xᵢ) is the same as minimising the cross-entropy between the empirical distribution and the model.

**Forward vs Reverse KL — Profound Implications for Variational Inference:**
```
Forward KL (P||Q): min_Q E_P[log P/Q]
  → Q must be non-zero wherever P is non-zero
  → Q tends to "spread out" to cover all modes of P (mean-seeking)
  → Used in MLE / cross-entropy training

Reverse KL (Q||P): min_Q E_Q[log Q/P]
  → Q avoids placing mass where P is zero (or near-zero)
  → Q concentrates on a single mode of P (mode-seeking)
  → Used in variational inference (ELBO maximisation)
```
This asymmetry has profound consequences. A model trained with forward KL (standard neural network training) will produce "blurry" outputs when asked to model a multimodal distribution — it spreads probability across all modes. A model trained with reverse KL (variational autoencoder decoder) will collapse to one mode. Neither is universally better — the choice depends on whether coverage or sharpness is more important for the application.

**Data Processing Inequality:**
```
If X → Y → Z is a Markov chain:
  I(X;Z) ≤ I(X;Y)   and   I(X;Z) ≤ I(Y;Z)
```
Processing data can only reduce (or maintain) mutual information — you cannot create information by transforming data. This is the theoretical basis for the Information Bottleneck principle: a good representation Z of input X for predicting output Y should maximise I(Z;Y) while minimising I(X;Z) — compressing X as much as possible while preserving relevant information for Y.

---

## 2. Core Machine Learning Theory

### 2.1 The Learning Problem

**Formal Setup:**
- Input space `X`, output space `Y`, unknown target `f: X → Y`
- Training data: `D = {(x₁,y₁), ..., (xₙ,yₙ)}` i.i.d. from `P(X,Y)`
- Hypothesis class: `H` — the set of functions the algorithm considers
- Loss function: `L: Y × Y → ℝ` — measures prediction error

**Empirical Risk Minimisation (ERM):**
```
ĥ = argmin_{h ∈ H}  R̂(h) = (1/n) Σᵢ L(h(xᵢ), yᵢ)
```

**Expected (True) Risk:**
```
R(h) = E_{(x,y)~P} [L(h(x), y)]
```

**Generalisation Gap:**
```
R(h) - R̂(h) ≤ ε   with probability ≥ 1 - δ
```

The gap depends on: sample size n, hypothesis class complexity, confidence δ.

**The Fundamental Tension in Learning:**
There are two ways a learning algorithm can fail, and they pull in opposite directions:
- **Approximation error (bias):** The hypothesis class H may not contain a function close to f. This is a property of H alone, not the data. Making H larger reduces approximation error.
- **Estimation error (variance):** Even if f ∈ H, finite data means we cannot perfectly identify f within H. A larger H gives more "room" for ERM to overfit noise. Making H smaller reduces estimation error.

The optimal H balances these two terms. Formal bounds (VC theory, Rademacher complexity) precisely quantify how the estimation error grows with the complexity of H.

**Rademacher Complexity:**
```
Empirical Rademacher complexity:
  R̂ₙ(H) = E_σ[ sup_{h∈H} (1/n) Σᵢ σᵢ h(xᵢ) ]
  σᵢ ∈ {-1,+1} i.i.d. uniform (Rademacher random variables)

Rademacher complexity:
  Rₙ(H) = E_{D,σ} [ sup_{h∈H} (1/n) Σᵢ σᵢ h(xᵢ) ]
```
Intuitively, R̂ₙ(H) measures how well the hypothesis class can fit random ±1 labels on the training data. If H can perfectly fit any labelling (high complexity), R̂ₙ → 1. If H is very rigid, R̂ₙ → 0. The key generalisation bound is:
```
With probability ≥ 1-δ:
  R(ĥ) ≤ R̂(ĥ) + 2Rₙ(H) + √(log(1/δ)/(2n))
```
This is tighter and more widely applicable than VC theory for modern neural networks.

**Fundamental ML Assumption:** Training distribution ≈ test distribution. When violated → distribution shift (covariate shift, label shift, concept drift).

---

### 2.2 Bias-Variance Trade-off

For squared loss:
```
E[(y - ĥ(x))²] = Bias²(ĥ(x)) + Var(ĥ(x)) + σ²_noise
```

Where:
```
Bias(ĥ(x))  = E[ĥ(x)] - f(x)               (systematic error; model too simple)
Var(ĥ(x))   = E[(ĥ(x) - E[ĥ(x)])²]         (sensitivity to training data)
σ²_noise     = irreducible noise = Var(y|x)
```

**The Bias-Variance Decomposition Derivation:**
```
MSE = E[(y - ĥ)²]
    = E[((y-f) + (f-Eĥ) + (Eĥ-ĥ))²]
    = E[(y-f)²] + (f-Eĥ)² + E[(Eĥ-ĥ)²]   (cross-terms = 0)
    = σ²_noise + Bias² + Variance
```

**Why Cross-Terms Vanish:**
The cross-term `E[(y-f)(f-Eĥ)]` is zero because `(f-Eĥ)` is a constant (averaged over datasets) and `E[(y-f)] = 0` by definition (f is the true conditional mean). The cross-term `E[(y-f)(Eĥ-ĥ)]` is zero because `(y-f)` is independent noise while `(Eĥ-ĥ)` depends only on the training set D.

**Implications:**
- **High Bias:** Model too simple (e.g., linear model for non-linear data). Both train and test error are high.
- **High Variance:** Model too complex (e.g., deep tree). Low train error, high test error.
- **Optimal complexity:** Minimises total expected error. Found via cross-validation.

**Bias-Variance for k-Nearest Neighbours:**
For KNN regression:
```
Bias²(x) ≈ [f(x_NN) - f(x)]² ≈ (C_f/k)² × k² × h²   (depends on smoothness of f)
Variance(x) ≈ σ²/k
```
As k → 1: zero bias (uses the single exact nearest neighbour), infinite variance.
As k → n: smoothest possible estimate, maximum bias, minimum variance.
The optimal k trades these off, typically scaling as k ~ n^(4/(d+4)) for d-dimensional smooth functions.

**Double Descent Phenomenon (modern DL):**
Beyond the interpolation threshold (zero training error), error can *decrease again* as model size grows — classical bias-variance doesn't fully explain deep learning behaviour. This occurs because overparameterised models that can perfectly fit training data can still generalise well if they are implicitly regularised (by SGD's preference for flat minima, early stopping, or the geometry of the loss landscape). The double descent curve has three regimes: classical underfitting → overfitting peak at the interpolation threshold → second descent into the modern "benign overfit" regime.

---

### 2.3 The No Free Lunch Theorem

**Statement:** For any two algorithms A₁ and A₂, averaged over all possible data-generating distributions:
```
Σ_f E[L(f, A₁, n)] = Σ_f E[L(f, A₂, n)]
```

**Implication:** No learning algorithm is universally superior. Every algorithm has inductive biases — assumptions baked in about the hypothesis class. Good algorithm selection requires domain knowledge.

**What This Means in Practice:**
The NFL theorem is often misread as saying "all algorithms are equal." It says no algorithm is universally equal on all possible tasks. In practice, tasks we care about are not random — they have structure (images have spatial correlations; language has syntactic structure; tabular data has feature interactions). Algorithms that exploit the right structure win on those tasks. CNNs win on images because they exploit translation invariance. Transformers win on sequences because they exploit pairwise token relationships. The "free lunch" we pay for is the inductive bias we build in.

---

### 2.4 VC Dimension & PAC Learning

**VC Dimension:** The largest set of points that H can *shatter* (correctly classify for ALL 2ⁿ possible labellings).

```
VCdim(H) = d  ⟺  ∃ set of d points that H shatters AND no set of d+1 points is shattered
```

**Examples:**
```
Linear classifiers in ℝᵈ:    VCdim = d + 1
Circles in ℝ²:                VCdim = 3
Convex polygons in ℝ²:        VCdim = ∞
Neural networks (W weights):  VCdim ≈ O(W log W)
```

**Why VC Dimension Cannot be d (only d+1) for Linear Classifiers:**
In ℝᵈ, any d+1 points in "general position" (no d of them collinear, etc.) can be shattered by a hyperplane. The argument uses the fact that any binary labelling of d+1 points in general position is linearly separable. However, for d+2 points, by Radon's theorem, any set of d+2 points can be partitioned into two subsets whose convex hulls intersect — and this particular partition cannot be separated by a hyperplane.

**Sauer-Shelah Lemma (growth function):**
```
mH(n) ≤ Σᵢ₌₀^d C(n,i)   where d = VCdim(H)
```
For n > d: `mH(n) ≤ (en/d)^d` — polynomial, not exponential.

**Significance of Polynomial Growth:**
The growth function `mH(n)` counts the number of distinct labellings of n points that H can produce. If H can shatter any n points, `mH(n) = 2ⁿ` — exponential. The Sauer-Shelah lemma shows that for finite VC dimension, the growth is only polynomial. This polynomial growth is exactly what permits generalisation: if H can only produce `(en/d)^d` distinct labellings (far fewer than 2ⁿ), then with enough samples each distinct labelling is well-estimated.

**PAC Learning Bound:**
```
With probability ≥ 1-δ, for sample size n:

  R(ĥ) - R̂(ĥ) ≤ √(2 VCdim(H) log(en/VCdim(H)) / n) + √(log(1/δ)/(2n))

Simplified: n ≥ (1/ε²)[VCdim(H) log(1/ε) + log(1/δ)]
```

---

### 2.5 Maximum Likelihood Estimation (MLE)

Given observations `{x₁,...,xₙ}` i.i.d. from `p(x|θ)`:

```
θ_MLE = argmax_θ  L(θ) = Π_i p(xᵢ|θ)
       = argmax_θ  ℓ(θ) = Σᵢ log p(xᵢ|θ)
```

**Log-likelihood is preferred:** Converts products to sums (numerically stable), same argmax.

**MLE as KL Minimisation:**
```
θ_MLE = argmin_θ D_KL(p_data || p_θ)
       = argmin_θ E_{x~p_data}[-log p_θ(x)]
       = argmin_θ H(p_data, p_θ)
```
MLE is equivalent to finding the model distribution `p_θ` closest (in KL divergence) to the true data distribution `p_data`. Since `D_KL(p_data||p_θ) = H(p_data) + D_KL(p_data||p_θ)` and `H(p_data)` is fixed, minimising KL is the same as minimising cross-entropy, which is the same as maximising log-likelihood. This unification connects information theory, statistics, and machine learning through a single framework.

**Score Function:**
```
s(θ) = ∇_θ ℓ(θ) = Σᵢ ∇_θ log p(xᵢ|θ) = 0   (at MLE)
```

**Fisher Information Matrix:**
```
I(θ) = E[-∇²_θ log p(x|θ)] = E[s(θ)s(θ)ᵀ]
```
- Measures curvature of the log-likelihood — how much data tells us about θ
- Cramér-Rao Bound: `Var(θ̂) ≥ I(θ)⁻¹` — MLE achieves this asymptotically

**Natural Gradient:**
Standard gradient descent uses the Euclidean metric in parameter space. But the Fisher information matrix defines a Riemannian metric on the space of probability distributions — the natural (intrinsic) geometry. The natural gradient is:
```
∇̃_θ ℓ = I(θ)⁻¹ ∇_θ ℓ
```
This preconditions the gradient by the inverse Fisher information, making updates invariant to reparameterisation. Natural gradient descent converges in fewer steps than standard GD for probabilistic models. K-FAC and related methods approximate the Fisher inverse efficiently for neural networks.

**Asymptotic Properties of MLE:**
```
√n (θ_MLE - θ*) →_d N(0, I(θ*)⁻¹)   (consistency + asymptotic normality)
```

---

### 2.6 Maximum A Posteriori (MAP) Estimation

Incorporates a prior `P(θ)`:
```
θ_MAP = argmax_θ  log P(θ|X)
       = argmax_θ  [Σᵢ log P(xᵢ|θ) + log P(θ)]
       = MLE objective + regularisation
```

**Prior → Regularisation Correspondence:**
```
Gaussian prior  β ~ N(0, σ²/λ I)  ↔  L2 regularisation (Ridge)
Laplace prior   β ~ Laplace(0,b)  ↔  L1 regularisation (Lasso)
Spike-and-slab  ↔  L0 sparsity
Beta prior      ↔  Dirichlet regularisation (for Categorical)
```

**Why L2 Regularisation Corresponds to a Gaussian Prior:**
Under a Gaussian prior `β ~ N(0, τ²I)` and Gaussian likelihood:
```
log P(β|X,y) ∝ log P(y|X,β) + log P(β)
             = -1/(2σ²)||y - Xβ||² - 1/(2τ²)||β||²  + const
```
Setting `λ = σ²/τ²`, this is exactly `||y-Xβ||² + λ||β||²` — the Ridge objective. Larger `τ²` (weaker prior, more permissive) corresponds to smaller `λ` (less regularisation). This connection gives a principled way to choose `λ`: it encodes our prior belief about the typical scale of the weights.

**Full Bayesian Inference** (goes beyond MAP):
```
P(θ|X) = P(X|θ)P(θ) / P(X)          (exact posterior)
P(x_new|X) = ∫ P(x_new|θ) P(θ|X) dθ  (posterior predictive)
```
MAP = mode of posterior. Full Bayes = full posterior distribution. The difference matters when the posterior is multi-modal or asymmetric — the mode can be very unrepresentative. Full Bayesian inference marginalises over parameter uncertainty, giving predictions that are automatically calibrated.

---

## 3. Data Preprocessing & Feature Engineering

### 3.1 Handling Missing Values

**Types of Missingness (Rubin, 1976):**
```
MCAR: P(missing | data_observed, data_missing) = P(missing)   — no relationship
MAR:  P(missing | data_observed, data_missing) = P(missing | data_observed)  — depends on observed
MNAR: P(missing | data_observed, data_missing) = P(missing | data_missing)  — depends on missing value itself
```

- MCAR → safe to drop rows (complete case analysis)
- MAR → imputation works; listwise deletion biases estimates
- MNAR → most dangerous; need model for missingness mechanism

**Theoretical Underpinning of Imputation:**
Complete case analysis (dropping missing rows) is valid only under MCAR. Under MAR, it produces biased estimates because the missingness depends on observed covariates — the complete cases are a non-random subsample. Imputation under MAR works because we can model the conditional distribution of the missing values given the observed ones. Under MNAR, even sophisticated imputation is biased without a model for the missingness mechanism. In practice, one should always test the MCAR hypothesis (Little's test) and justify the MAR assumption by domain knowledge.

**Imputation Methods:**

```
Mean/Median/Mode:
  x_imputed = mean(x_observed)     — use median for skewed distributions
  x_imputed = mode(x_observed)     — use for categorical

Regression Imputation:
  x_missing = β₀ + β₁x₁ + β₂x₂ + ε   — predict from other features

KNN Imputation:
  x_imputed = Σₖ wₖ xₖ / Σₖ wₖ,   wₖ = 1/d(x, xₖ)^p   (inverse distance weighting)

Multiple Imputation (MICE — Multivariate Imputation by Chained Equations):
  For each variable j with missing values:
    Regress xⱼ on all other variables using observed cases
    Sample from the predictive distribution
  Repeat for all variables; iterate until convergence
  Run m complete datasets; pool results via Rubin's rules:
    Q̄ = (1/m)Σ Qₘ,   T = W̄ + (1 + 1/m)B
    W̄ = within-imputation variance, B = between-imputation variance
```

**Why Multiple Imputation Outperforms Single Imputation:**
Single imputation (e.g., mean imputation) treats imputed values as if they were observed, artificially deflating standard errors and distorting the covariance structure. Multiple imputation correctly propagates uncertainty about the missing values by sampling from the imputation model's predictive distribution m times. Rubin's combining rules aggregate the m analyses in a way that correctly accounts for both within-imputation variability (estimation error given the imputed values) and between-imputation variability (uncertainty due to not knowing the missing values). The result is valid confidence intervals and hypothesis tests.

**Missing Indicator Variables:**
```
Create binary column:  missing_j = 1 if xⱼ is missing, 0 otherwise
Then impute xⱼ with any method
— allows model to learn if missingness itself is informative
```

---

### 3.2 Feature Scaling

```
Min-Max Normalisation (scales to [0,1]):
  x' = (x - x_min) / (x_max - x_min)
  Sensitive to outliers. Use when distribution is bounded and known.

Z-Score Standardisation (mean=0, std=1):
  x' = (x - μ) / σ
  Required for: PCA, SVMs, KNN, regularised regression, neural networks.

Robust Scaling (median/IQR — robust to outliers):
  x' = (x - median) / IQR,    IQR = Q3 - Q1

Max-Abs Scaling (preserves sign, scales to [-1,1]):
  x' = x / max(|x|)   — useful for sparse data

Log Transform (right-skewed data):
  x' = log(x + 1)      — +1 handles x=0 (log1p)

Power Transforms (approximate Gaussian):
  Box-Cox:   x'(λ) = (xλ-1)/λ  if λ≠0;  log(x) if λ=0   (requires x > 0)
  Yeo-Johnson: extends Box-Cox to handle x ≤ 0
  λ estimated by maximising log-likelihood of normality

Unit Vector Normalisation (per-sample, used in NLP/cosine similarity):
  x' = x / ||x||₂   — each sample becomes a unit vector
```

**Why Scaling is Mandatory for Some Algorithms:**
The need for scaling is not arbitrary — it is dictated by the geometry of the algorithm's objective function. SVMs maximise the margin `2/||w||`, which is a Euclidean norm — if one feature has values in [0, 1000] and another in [0, 1], the first feature will dominate the norm, and the SVM will effectively ignore the second. PCA finds directions of maximum variance — without standardisation, features with larger numerical ranges contribute disproportionately variance. KNN uses distance directly; unscaled features make distance meaningless. Gradient-based methods converge faster when the loss landscape is isotropic, which requires scaled features.

**Tree-Based Methods Don't Need Scaling:**
Decision trees, random forests, and gradient boosting split features based on threshold comparisons: `xⱼ ≤ t`. This operation is scale-invariant — multiplying xⱼ by any positive constant just rescales t proportionally. Therefore, scaling has no effect on the model's split decisions or predictions.

---

### 3.3 Encoding Categorical Variables

```
Label Encoding: [Low, Med, High] → [0, 1, 2]
  ONLY for ordinal variables. Implies magnitude relationship.

One-Hot Encoding: [Red, Green, Blue] → 3 binary columns
  Use k-1 columns (drop first) to avoid multicollinearity in linear models.
  Creates high-dimensional sparse feature space for high-cardinality categories.

Ordinal Encoding: maps to meaningful integers preserving order

Target Encoding (Mean Encoding):
  x_encoded = E[y | x = category]  (with regularisation/smoothing)
  Smoothed: x_enc = (n_c × ȳ_c + α × ȳ) / (n_c + α)

Frequency/Count Encoding:
  x_encoded = count(category) / total_count

Binary Encoding:
  Label encode → convert to binary → split bits into columns

Hashing (Feature Hashing / Hashing Trick):
  x_encoded[h(category) mod m] += 1

Embedding Encoding (learned):
  Map category c to dense vector e_c ∈ ℝᵈ via lookup table
  Dimensionality rule of thumb: d = min(50, ceil(cardinality/2))

Leave-One-Out Encoding:
  x_enc_i = (Σⱼ≠ᵢ [xⱼ=xᵢ] yⱼ) / (count(xᵢ) - 1)
```

**The Target Leakage Problem in Target Encoding:**
Target encoding (using the mean target value per category) is powerful but dangerous. If you compute `ȳ_c` using the same rows that will be used for training, you are directly leaking information about y into the feature — the model sees a feature that is perfectly correlated with the target for training data, but this correlation partially disappears at test time. The solution is to use out-of-fold statistics: compute `ȳ_c` only on the training folds, never including the current validation fold. Smoothing `(n_c × ȳ_c + α × ȳ)/(n_c + α)` also helps by pulling rare categories toward the global mean, reducing the variance of the encoded values.

**Why Embeddings Outperform One-Hot for High Cardinality:**
For a categorical variable with 50,000 unique values, one-hot encoding produces a 50,000-dimensional sparse vector — mostly zeros. Two distinct cities in a one-hot encoding are orthogonal (distance √2 apart), with no notion of similarity. Learned embeddings solve both problems: they are dense (say, 50-dimensional), so the model needs to learn far fewer parameters, and they can encode semantic similarity — cities in the same region end up with similar embedding vectors because the model learns to exploit geographic patterns in the target variable.

---

### 3.4 Feature Selection

**Filter Methods (model-independent, fast):**
```
Pearson Correlation:   r = Cov(x,y) / (σₓ σᵧ)    — linear relationships only
Spearman Correlation:  ρ = 1 - 6Σdᵢ² / (n(n²-1)) — monotonic, non-parametric
Mutual Information:    I(X;Y) = H(X) - H(X|Y)     — any relationship (non-parametric)
Chi-Squared test:      χ² = Σ (O-E)²/E            — categorical features vs target
ANOVA F-statistic:     F = MS_between / MS_within  — continuous features vs categorical target
Variance threshold:    remove features with Var < threshold (near-constant → uninformative)
```

**Wrapper Methods (use model performance):**
```
Forward Selection, Backward Elimination, Recursive Feature Elimination (RFE)
Sequential Floating (SFFS/SBFS): adds backward step in forward — better but slower.
```

**Embedded Methods (regularisation-based):**
```
L1 (Lasso), L2 (Ridge), Elastic Net, Tree-based importances, SHAP
```

**SHAP Values — Theoretical Foundation:**
SHAP (SHapley Additive exPlanations) is grounded in cooperative game theory. Given a model `f` and an instance `x`, the Shapley value `φⱼ` of feature j is:
```
φⱼ = Σ_{S⊆F\{j}} [|S|!(|F|-|S|-1)!/|F|!] × [v(S∪{j}) - v(S)]
```
where `v(S)` is the model's expected prediction when only the features in S are known (others marginalised out), and the sum is over all feature subsets not containing j. The weights give each subset the probability that it would arise if features joined in a random order — the Shapley value is the expected marginal contribution of feature j across all possible orderings.

SHAP satisfies three axioms that make it the unique fair attribution:
1. **Efficiency:** `Σⱼ φⱼ = f(x) - E[f(X)]` (attributions sum to model output minus baseline)
2. **Symmetry:** If j and k are interchangeable, `φⱼ = φₖ`
3. **Dummy:** If feature j never changes predictions, `φⱼ = 0`

---

### 3.5 Outlier Detection

```
Z-Score, Modified Z-Score, IQR (Tukey)
Isolation Forest, Local Outlier Factor (LOF), DBSCAN-based
One-Class SVM, Autoencoder-based
```

**Why Isolation Forest Works:**
The key insight of Isolation Forest is that anomalies are "few and different" — they are sparse in the feature space and far from other points. A random tree recursively partitions the space with random splits. For normal points (dense regions), many splits are needed to isolate any single point. For anomalies (sparse regions), they are isolated in very few splits. Therefore, the average path length to isolation across many trees is a direct measure of anomaly score — short path = anomalous. The method requires no distance computation, scales linearly with n, and handles high dimensions well because isolation doesn't require dense coverage of the space.

**LOF Theory:**
LOF compares the local density of a point to the local density of its neighbours. If a point is in a much sparser neighbourhood than its neighbours, its LOF score is much greater than 1, indicating an outlier. The "reachability distance" ensures that the distance to very nearby points is not underestimated (replacing small distances with the k-th nearest neighbour distance), making LOF robust to clusters of varying densities. LOF can detect outliers that are not globally extreme but are locally anomalous — a capability flat methods like Z-score lack.

---

### 3.6 Correlation Analysis

```
Pearson, Spearman, Kendall, Point-biserial, Cramér's V

VIF (detects multicollinearity):
  VIF_j = 1 / (1 - R²_j)
  VIF > 5: moderate; VIF > 10: severe multicollinearity
```

**Why Multicollinearity Inflates Variance:**
When features are highly correlated, the matrix `XᵀX` becomes nearly singular (nearly rank-deficient). The OLS estimator is `β* = (XᵀX)⁻¹Xᵀy`, and its variance is `σ²(XᵀX)⁻¹`. Near-singularity makes `(XᵀX)⁻¹` have very large eigenvalues — the variance of the estimates explodes. Geometrically: with two perfectly correlated features, there are infinitely many hyperplanes that fit the data equally well (the loss is flat in the direction of perfect collinearity), so the estimated coefficients are essentially random.

Ridge regression directly solves this by adding `λI` to `XᵀX`, guaranteeing a well-conditioned matrix regardless of collinearity. This is precisely why Ridge is preferred over OLS in the presence of multicollinearity.

---

## 4. Regression Algorithms

### 4.1 Linear Regression

**Model:**
```
ŷ = β₀ + β₁x₁ + ... + βₙxₙ = Xβ    (matrix form, X has intercept column)
```

**OLS — Minimise MSE:**
```
L(β) = ||y - Xβ||² = (y-Xβ)ᵀ(y-Xβ) = yᵀy - 2βᵀXᵀy + βᵀXᵀXβ
```

**Closed-Form (Normal Equation):**
```
∂L/∂β = 2Xᵀ(Xβ - y) = 0
β* = (XᵀX)⁻¹ Xᵀy
```

**Geometric Interpretation of OLS:**
OLS projects y onto the column space of X. The fitted values `ŷ = Xβ* = X(XᵀX)⁻¹Xᵀy = Hᵧ` where `H = X(XᵀX)⁻¹Xᵀ` is the hat (projection) matrix. The residuals `e = y - ŷ = (I - H)y` are orthogonal to the column space of X: `Xᵀe = 0`. This is why OLS residuals are uncorrelated with any linear combination of the predictors. The hat matrix H is idempotent (`H² = H`) and symmetric, reflecting its nature as an orthogonal projection.

**Hat Matrix Properties:**
```
H = X(XᵀX)⁻¹Xᵀ    (n×n projection matrix)
H² = H               (idempotent)
Hᵀ = H               (symmetric)
trace(H) = p+1        (number of parameters; = effective degrees of freedom)
0 ≤ Hᵢᵢ ≤ 1          (leverage of observation i)
```
The diagonal entries `Hᵢᵢ` are called leverages — they measure how much influence observation i has on its own fitted value. High leverage points (Hᵢᵢ close to 1) can distort the regression: if their y-values are also unusual (high residual), they become influential outliers. Cook's distance combines leverage and residual to quantify influence:
```
Dᵢ = (ŷ - ŷ₍ᵢ₎)ᵀ(ŷ - ŷ₍ᵢ₎) / (p × MSE) = eᵢ² Hᵢᵢ / (p × MSE × (1-Hᵢᵢ)²)
```

**Numerical Stability:**
```
Via QR decomposition: X = QR → β* = R⁻¹ Qᵀy     (O(nd²), more stable)
Via SVD: X = UΣVᵀ → β* = V Σ⁺ Uᵀy               (handles rank deficiency)
```

**Gauss-Markov Theorem (OLS is BLUE):**
Under assumptions:
1. `E[ε|X] = 0` (zero conditional mean)
2. `Var(ε|X) = σ²I` (homoscedasticity + no autocorrelation)
3. No perfect multicollinearity

OLS is the Best Linear Unbiased Estimator (BLUE).

**Understanding Gauss-Markov:**
The theorem compares OLS to any other linear unbiased estimator `β̃ = Cy` for some matrix C with `CX = I`. The OLS estimator achieves the minimum variance in the sense that `Var(aᵀβ̃) ≥ Var(aᵀβ*)` for any vector a. The proof uses the fact that any alternative `C = (XᵀX)⁻¹Xᵀ + D` for some D with `DX = 0`, and the extra term `DD'σ²` in the variance is positive semi-definite. Gauss-Markov does NOT say OLS is the best of ALL estimators — biased estimators (like Ridge) can have lower MSE if they trade a small bias for a large variance reduction.

**Inference:**
```
Var(β*) = σ² (XᵀX)⁻¹
SE(βⱼ) = √[σ²(XᵀX)⁻¹ⱼⱼ]
t-statistic: tⱼ = βⱼ / SE(βⱼ) ~ t(n-p-1)

R² = 1 - SS_res/SS_tot = 1 - Σ(yᵢ-ŷᵢ)² / Σ(yᵢ-ȳ)²
Adjusted R² = 1 - (1-R²)(n-1)/(n-p-1)
F-statistic = (R²/p) / ((1-R²)/(n-p-1))  — tests joint significance
```

---

### 4.2 Ridge Regression (L2)

```
L(β) = ||y - Xβ||² + λ||β||²₂
β_ridge = (XᵀX + λI)⁻¹ Xᵀy
```

**Bias-Variance of Ridge:**
```
Bias(β_ridge) = -λ(XᵀX + λI)⁻¹ β*         (non-zero — biased)
Var(β_ridge)  = σ² (XᵀX+λI)⁻¹ XᵀX (XᵀX+λI)⁻¹  (< OLS variance)
```

**Effect via SVD:** If `X = UΣVᵀ`:
```
β_ridge = V diag(σᵢ²/(σᵢ²+λ)) Uᵀy
```
Each component is shrunk by `σᵢ²/(σᵢ²+λ)` — small singular values (noisy directions) shrunk most.

**Why Ridge Shrinks Small Singular Values Most:**
The shrinkage factor `σᵢ²/(σᵢ²+λ)` approaches 1 when `σᵢ >> λ` (large singular value, well-identified direction — kept intact) and approaches 0 when `σᵢ << λ` (small singular value, poorly identified direction — heavily regularised). Directions with small singular values correspond to near-dependencies in the data (near-multicollinearity). Ridge correctly identifies these as unreliable and shrinks them toward zero.

**Effective Degrees of Freedom:**
```
df(λ) = tr(X(XᵀX+λI)⁻¹Xᵀ) = Σᵢ σᵢ²/(σᵢ²+λ)
```
λ=0 → df=p (full OLS); λ→∞ → df→0.

**Ridge and the Bias-Variance Trade-off (Formally):**
Define the total MSE as `MSE(β_ridge) = Bias² + Variance`. There always exists some `λ > 0` for which `MSE(β_ridge) < MSE(β_OLS)`. The reason is that OLS, while unbiased, may have very high variance (especially with collinearity). A small amount of bias introduced by Ridge can reduce variance by much more, giving net MSE improvement. The optimal λ minimises this MSE and depends on the unknown true β — in practice, we estimate it via cross-validation.

---

### 4.3 Lasso Regression (L1)

```
L(β) = ||y - Xβ||² + λΣⱼ|βⱼ|
```

No closed form — use subgradient or coordinate descent.

**Coordinate Descent:**
```
ρⱼ = Σᵢ xᵢⱼ (yᵢ - Σₖ≠ⱼ xᵢₖ βₖ)   (partial residual)
βⱼ = S(ρⱼ, λ/2) / Σᵢ xᵢⱼ²
Soft-threshold: S(ρ, λ) = sign(ρ) max(|ρ| - λ, 0)
```

**Why the Soft-Threshold Operator:**
The subdifferential of `L` with respect to `βⱼ` is:
```
∂L/∂βⱼ = -2ρⱼ + 2Σᵢxᵢⱼ² βⱼ + λ∂|βⱼ|
```
Setting this to zero and solving:
- If `βⱼ > 0`: `2Σxᵢⱼ² βⱼ = 2ρⱼ - λ` → `βⱼ = (ρⱼ - λ/2)/Σxᵢⱼ²` (valid only if > 0)
- If `βⱼ < 0`: `βⱼ = (ρⱼ + λ/2)/Σxᵢⱼ²` (valid only if < 0)
- If `0 ∈ ∂|βⱼ||_{β=0}`: `βⱼ = 0` (when `|ρⱼ| ≤ λ/2`)

The soft-threshold combines these cases. Coefficients with small partial correlation `|ρⱼ| ≤ λ/2` are set exactly to zero (true sparsity, not just near-zero).

**Why L1 Gives Sparsity (geometric argument):**
The constraint region `{β: ||β||₁ ≤ t}` is a polytope (diamond in 2D) with corners on coordinate axes. Loss contours (ellipses) typically first touch this region at a corner → one coordinate = 0.

**Lasso Solution Path and LARS:**
```
Lasso Solution Path: As λ decreases from λ_max to 0:
  λ_max = max_j |Xⱼᵀy|/n — all coefficients zero
  Features enter the model one by one
  LARS (Least Angle Regression) efficiently computes the full path in O(np²) time
```

**LARS Algorithm (sketch):**
At each step, find the feature most correlated with the current residual. Move in a direction equiangular between this feature and all currently active features — equiangular means maintaining equal correlation with all active features. This produces the exact Lasso solution path as a piecewise linear function of λ. The "least angle" refers to the update direction making the smallest angle (closest to equiangular) among all possible directions respecting the active constraint set.

---

### 4.4 Elastic Net

```
L(β) = ||y-Xβ||² + λ[α||β||₁ + (1-α)||β||²]
```

**Why Lasso Fails with Correlated Groups:**
When two features `x₁` and `x₂` are highly correlated, Lasso arbitrarily selects one and drops the other. The selection is unstable — a small perturbation in the data can switch which one is chosen. This is problematic in settings like genomics (correlated genes in a pathway) or NLP (synonyms). Elastic Net solves this by adding an L2 term that encourages correlated features to be selected together and to have similar coefficient magnitudes.

**Grouping Effect:**
For two perfectly correlated features `x₁ = x₂`:
- Lasso: picks one, sets the other to zero
- Elastic Net: sets both to the same value `β₁ = β₂ = (some shared value)`

The L2 term's gradient `2λ(1-α)βⱼ` penalises any imbalance between `β₁` and `β₂`, driving them toward equality.

**Advantages over Lasso:**
- Selects groups of correlated features (Lasso picks one arbitrarily)
- Always selects ≤ n features (Lasso limited to n features when p > n)
- More numerically stable when p >> n

---

### 4.5 Polynomial & Basis Function Regression

```
Feature expansion: φ(x) = [1, x, x², ..., xᵖ]
ŷ = βᵀφ(x)   — still linear in parameters, apply OLS
```

**The Universality of Basis Expansions:**
Any sufficiently smooth function f on a compact interval can be approximated arbitrarily closely by a linear combination of basis functions — this is guaranteed by the Weierstrass approximation theorem (polynomial basis) and related results for other basis families. The choice of basis encodes prior knowledge about the function's structure. Polynomial bases are natural for smooth global behaviour but can oscillate wildly (Runge's phenomenon). Spline bases are locally supported and avoid oscillation. Fourier bases are natural for periodic functions. Wavelet bases adapt to both local and global structure.

**Splines in Detail:**
A natural cubic spline with K knots `τ₁ < ... < τₖ` is a function that:
1. Is a cubic polynomial on each of the K+1 intervals
2. Has continuous first and second derivatives at each knot
3. Is linear beyond the boundary knots (natural boundary condition)

The smoothing spline minimises:
```
Σᵢ (yᵢ - f(xᵢ))² + λ ∫ [f''(x)]² dx
```
The roughness penalty `∫[f'']²dx` penalises curvature. The solution is a natural cubic spline — exactly a basis expansion with K basis functions, and λ controls the smoothness-fit tradeoff. This connects basis function regression to non-parametric smoothing.

---

### 4.6 Bayesian Linear Regression

```
Prior:     β ~ N(μ₀, Σ₀)
Likelihood: y|X,β ~ N(Xβ, σ²I)

Posterior:
  Σₙ = (Σ₀⁻¹ + (1/σ²)XᵀX)⁻¹
  μₙ = Σₙ(Σ₀⁻¹μ₀ + (1/σ²)Xᵀy)
  β|X,y ~ N(μₙ, Σₙ)

Predictive Distribution (for new x*):
  y*|x*,X,y ~ N(μₙᵀx*, σ² + x*ᵀΣₙx*)
  — gives BOTH prediction AND uncertainty estimate
```

**Conjugate prior:** Gaussian prior + Gaussian likelihood = Gaussian posterior (closed form).

**The Predictive Distribution Captures Two Sources of Uncertainty:**
- `σ²`: aleatoric uncertainty — irreducible noise in the data-generating process
- `x*ᵀΣₙx*`: epistemic uncertainty — our uncertainty about the true parameter β, which decreases as we collect more data

This distinction is critical for safety-critical applications. A model confident about predictions should have small `x*ᵀΣₙx*` (the new input is well-covered by training data) even if σ² is large (the problem is inherently noisy). Near extrapolation regions, `x*ᵀΣₙx*` grows, correctly flagging that predictions are uncertain.

**Online Bayesian Updating:**
As new data `(xₙ₊₁, yₙ₊₁)` arrives:
```
Σₙ₊₁⁻¹ = Σₙ⁻¹ + (1/σ²) xₙ₊₁ xₙ₊₁ᵀ
μₙ₊₁   = Σₙ₊₁(Σₙ⁻¹μₙ + (1/σ²) xₙ₊₁ yₙ₊₁)
```
Using the Woodbury identity:
```
Σₙ₊₁ = Σₙ - Σₙxₙ₊₁(σ² + xₙ₊₁ᵀΣₙxₙ₊₁)⁻¹xₙ₊₁ᵀΣₙ
```
This is a rank-1 update — each new data point updates the covariance matrix with O(d²) work rather than O(d³), enabling efficient sequential learning.

---

## 5. Classification Algorithms

### 5.1 Logistic Regression

**Sigmoid & Log-Odds:**
```
σ(z) = 1/(1+e⁻ᶻ),   σ'(z) = σ(z)(1-σ(z))

Log-odds = logit(P) = log[P/(1-P)] = β₀ + β₁x₁ + ... = βᵀx
P(y=1|x) = σ(βᵀx) = 1/(1+exp(-βᵀx))
```

**Why the Logit Link?**
The logit function is the natural link function for the Bernoulli distribution in the GLM (Generalised Linear Model) framework. The Bernoulli distribution belongs to the exponential family with natural parameter `η = log(p/(1-p))` — the log-odds. The canonical link sets the linear predictor equal to the natural parameter: `βᵀx = η = logit(p)`. This gives logistic regression its statistical justification as an exponential family GLM. The logit also satisfies desirable properties: it maps probabilities in (0,1) to the full real line, is symmetric around p=0.5, and its inverse (sigmoid) is bounded, smooth, and monotone.

**Binary Cross-Entropy Loss (Negative Log-Likelihood):**
```
L = -(1/n) Σᵢ [yᵢ log σ(βᵀxᵢ) + (1-yᵢ) log(1-σ(βᵀxᵢ))]
```

**Why Cross-Entropy, Not MSE?**
- MSE with sigmoid creates non-convex loss → many local minima
- Cross-entropy is convex in β → guaranteed global minimum
- MLE justification: minimising cross-entropy = maximising likelihood
- MSE treats probability errors as if they were in a Euclidean space; cross-entropy respects the geometry of the simplex

**Gradient:**
```
∂L/∂β = (1/n) Xᵀ(σ(Xβ) - y)   — elegant: same form as linear regression
```

**Why the Gradient Has This Clean Form:**
The gradient simplicity arises from the relationship between the sigmoid and the exponential family. For any GLM, the gradient of the negative log-likelihood is `Xᵀ(μ - y)` where `μ = E[y|x]`. For logistic regression, `μ = σ(βᵀx)`. This is a general property: the gradient of the log-likelihood for an exponential family GLM always equals `Xᵀ(observation - expected value)`. This makes all GLM gradients structurally identical, unifying Poisson regression (log link), linear regression (identity link), and logistic regression (logit link) in a single framework.

**Hessian (for Newton's method):**
```
H = (1/n) XᵀWX,   W = diag[σ(xᵢ)(1-σ(xᵢ))]
β := β - H⁻¹∇L   (IRLS — Iteratively Reweighted Least Squares)
```

**IRLS Connection to Weighted Least Squares:**
Each Newton step of logistic regression is equivalent to solving a weighted least squares problem with working response `z = Xβ + W⁻¹(y - μ)` and weights W. The weights `wᵢ = σᵢ(1-σᵢ)` are larger near the decision boundary (p ≈ 0.5) where the sigmoid is steepest and information is richest, and smaller near the extremes (p ≈ 0 or 1) where the sigmoid is flat. IRLS re-weights observations at each step, focusing computational effort where it is most informative.

**Multinomial Logistic (Softmax):**
```
P(y=k|x) = exp(βₖᵀx) / Σⱼ exp(βⱼᵀx)

Softmax temperature scaling:
  P(y=k|x,T) = exp(βₖᵀx/T) / Σⱼ exp(βⱼᵀx/T)
  T→0: argmax (deterministic); T→∞: uniform
```

**Softmax Non-Identifiability:**
For K classes, logistic regression has K weight vectors β₁,...,βₖ. Adding any constant vector c to all βₖ leaves the softmax probabilities unchanged (the constant cancels in numerator and denominator). This means the parameterisation is not identified — there are infinitely many (β₁,...,βₖ) giving the same predictions. To resolve this, one typically fixes β₁ = 0 (reference class). L2 regularisation also breaks the tie by penalising large weights, picking the minimum-norm solution.

---

### 5.2 K-Nearest Neighbours (KNN)

**Distance Metrics:**
```
Euclidean, Manhattan, Minkowski, Cosine, Mahalanobis, Hamming

Classification: ŷ = argmax_c Σ_{xⱼ∈Nₖ(x)} 𝟙[yⱼ=c]
Regression:     ŷ = (1/k) Σ_{xⱼ∈Nₖ(x)} yⱼ
Weighted KNN:   ŷ = Σⱼ (1/dⱼ) yⱼ / Σⱼ (1/dⱼ)
```

**Mahalanobis Distance — Why It Matters:**
The Mahalanobis distance `d(x,y) = √((x-y)ᵀΣ⁻¹(x-y))` accounts for the correlation structure of the data. It is equivalent to Euclidean distance after whitening (transforming data so that the covariance is the identity). Without this, KNN treats all features as equally important and uncorrelated — if two features are highly correlated, they effectively contribute twice to the distance. The Mahalanobis distance is invariant to linear transformations of the feature space, making it a more principled measure of "similarity."

**KNN as a Non-Parametric Density Estimator:**
The KNN classification rule implicitly estimates class conditional densities via:
```
P̂(x|C=c) ∝ (nₖc / nₖ) / Vol(R_k(x))
```
where `nₖc` is the number of class-c neighbours among the k nearest, and `Vol(R_k(x))` is the volume of the k-nearest-neighbour ball. This connects KNN to kernel density estimation: KNN uses a variable bandwidth (the k-th nearest neighbour distance), while kernel density estimation uses a fixed bandwidth. Variable bandwidth KNN adapts to local density — smaller bandwidth in dense regions, larger in sparse regions.

**Curse of Dimensionality:**
```
Volume of d-hypersphere / volume of d-hypercube → 0 as d → ∞
Consequence: in high-d, all points are approximately equidistant.
For KNN to work: n must grow exponentially in d to maintain neighbourhood density.
Practical remedy: dimensionality reduction (PCA, UMAP) before KNN.
```

---

### 5.3 Naive Bayes

```
P(C|x) ∝ P(x|C) P(C) = P(C) Πⱼ P(xⱼ|C)   (naive independence)
ĉ = argmax_c [log P(C=c) + Σⱼ log P(xⱼ|C=c)]
```

**When Does Naive Bayes Work Despite Wrong Assumptions?**
The independence assumption is almost always violated in practice — yet Naive Bayes often performs competitively. The reason is that we need the decision boundary to be in the right place, not the probability values to be correct. Even if `P(x|C)` is badly approximated, if the ordering of class posteriors is preserved, classification is correct. Analysis by Domingos and Pazzani (1997) shows that the Naive Bayes decision boundary is correct whenever the dependency structure among features is the same across all classes (even if dependencies exist). Furthermore, Naive Bayes has low variance — only `O(d × K)` parameters — making it excellent for small datasets and high-dimensional inputs (like text).

**Gaussian NB, Multinomial NB, Bernoulli NB, Complement NB:**
(See prior section for formulas.)

**Calibration:** Despite often good accuracy, Naive Bayes probabilities are poorly calibrated — apply Platt scaling or isotonic regression if probabilities matter.

**Zero-Frequency Problem and Laplace Smoothing:**
In text classification, the word "quantum" might never appear in training spam emails. Without smoothing, `P("quantum"|spam) = 0`, which makes the entire product `Πⱼ P(xⱼ|spam) = 0` regardless of other evidence — one unseen word completely overrides all other signals. Laplace (add-α) smoothing adds a pseudo-count of α to each word count:
```
P(wⱼ|C=c) = (count(wⱼ,c) + α) / (count(all words in c) + α×|V|)
```
This ensures no probability is exactly zero. α=1 is "add-one smoothing"; smaller α for large vocabularies.

---

### 5.4 Decision Trees

**Splitting Criteria:**
```
Entropy:       H(S) = -Σₖ pₖ log₂ pₖ
Gini:          G(S) = 1 - Σₖ pₖ²
Misclass rate: E(S) = 1 - max_k pₖ
```

**Why Gini ≈ Entropy:**
A Taylor expansion of `-p log p` around p=0.5 gives `-p log p ≈ p(1-p) + O((p-0.5)²)`. Therefore `H ≈ 2Σpₖ(1-pₖ) = 2G`, showing the two criteria are proportional for small impurity. The difference emerges near the extremes: entropy penalises near-pure splits slightly less than Gini does near probability 0 or 1.

**Information Gain:**
```
IG(S, A) = H(S) - Σᵥ (|Sᵥ|/|S|) H(Sᵥ)
```

**Gain Ratio (C4.5 — prevents bias toward many-valued attributes):**
```
GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)
SplitInfo(S, A) = -Σᵥ (|Sᵥ|/|S|) log₂(|Sᵥ|/|S|)
```

**Why Information Gain Biases Toward High-Cardinality Features:**
A feature like "unique customer ID" with one distinct value per sample can achieve perfect information gain (every leaf is pure with one sample). But this generalises terribly. Gain ratio normalises by the entropy of the split itself (SplitInfo). A split with many equal-sized branches has high SplitInfo, penalising the feature. A binary split with unequal sizes has lower SplitInfo, rewarding it less. This normalisation makes the criterion fairer across features with different numbers of values.

**Cost-Complexity Pruning (α-pruning / CART):**
```
R_α(T) = R(T) + α|T|
```

**The Bias-Variance Trade-off in Decision Trees:**
A fully grown tree memorises the training set — each leaf contains a single training point, giving zero training error but high variance (overfitting). Pruning increases bias (coarser decision regions) while reducing variance. Cost-complexity pruning generates a nested sequence of trees `T₀ ⊃ T₁ ⊃ ... ⊃ T₁` by successively removing the subtree with the smallest increase in misclassification rate per leaf removed. Cross-validation then selects the tree in this sequence with lowest validation error. This is a principled way to sweep the bias-variance curve.

---

### 5.5 Support Vector Machines (SVM)

**Hard-Margin:**
```
Minimise:   (1/2)||w||²
Subject to: yᵢ(wᵀxᵢ+b) ≥ 1   ∀i
```

**Margin:** γ = 2/||w||. Maximum margin ↔ minimum ||w||².

**Why Maximum Margin?**
The margin is a measure of "confidence" in the decision boundary. A larger margin means the classifier is further from the training points, making it more robust to small perturbations. The VC dimension of maximum-margin classifiers is bounded by `O(R²/γ²)` where R is the radius of the data and γ is the margin. Maximising the margin therefore minimises the VC bound on generalisation error — the maximum-margin principle has a direct theoretical justification in terms of structural risk minimisation.

**Lagrangian & Dual:**
```
Primal: L = (1/2)||w||² - Σᵢ αᵢ[yᵢ(wᵀxᵢ+b)-1]
Dual:   max_α Σᵢαᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼxᵢᵀxⱼ
        s.t. αᵢ ≥ 0, Σᵢαᵢyᵢ = 0
Primal-dual: w* = Σᵢ αᵢyᵢxᵢ   (only support vectors with αᵢ > 0 contribute)
```

**Why the Dual is Preferred:**
1. The dual depends on data only through inner products `xᵢᵀxⱼ` — enabling the kernel trick without explicitly computing feature maps.
2. For n << d (many features, few samples), the dual is cheaper: O(n²) vs O(d²) variables.
3. The dual is always a convex QP regardless of the feature space dimension, so standard solvers guarantee global optimality.

**KKT Conditions and Support Vectors:**
```
αᵢ[yᵢ(wᵀxᵢ+b) - 1] = 0  (complementary slackness)
```
This means αᵢ > 0 only for points exactly on the margin `yᵢ(wᵀxᵢ+b) = 1` — the support vectors. All other points have αᵢ = 0 and do not contribute to w*. This sparsity makes SVM predictions efficient at test time and robust to the choice of non-support-vector training points.

**Soft-Margin, Hinge Loss, Kernel Trick:** (See prior section for formulas.)

**Reproducing Kernel Hilbert Space (RKHS):**
A kernel `K(x,z)` defines an RKHS `H_K` — a Hilbert space of functions with the "reproducing property" `f(x) = ⟨f, K(x,·)⟩_{H_K}`. The feature map `φ: X → H_K` is defined by `φ(x) = K(x,·)`, and the inner product in `H_K` satisfies `⟨φ(x),φ(z)⟩_{H_K} = K(x,z)`. The kernel trick replaces the (possibly infinite-dimensional) explicit feature map computation with the kernel evaluation — O(d) work instead of O(∞). This is possible exactly because the dual formulation depends only on inner products.

---

### 5.6 Discriminant Analysis

**LDA (Shared Covariance):**
(See prior section for formulas.)

**LDA as Dimensionality Reduction:** (See prior section.)

**Theoretical Connection Between LDA and Logistic Regression:**
Both LDA and logistic regression produce linear decision boundaries in the original feature space. The difference is in how the parameters are estimated:
- LDA assumes Gaussian class-conditional distributions and estimates parameters generatively (jointly model P(x,y)). It is optimal when this assumption holds.
- Logistic regression estimates parameters discriminatively (model P(y|x) directly). It is more robust to non-Gaussian distributions.

When the Gaussian assumption holds, LDA is more efficient (lower asymptotic variance) because it uses the structural assumption to "borrow strength" from the marginal distribution of x. When it does not hold, logistic regression is more robust. In practice, logistic regression is preferred for classification unless the dataset is small and the Gaussian assumption is credible.

---

## 6. Clustering & Unsupervised Learning

### 6.1 K-Means Clustering

**Objective:**
```
J = Σₖ Σᵢ∈Cₖ ||xᵢ-μₖ||²   (WCSS — Within-Cluster Sum of Squares)
```

**K-means as EM on a Mixture of Gaussians:**
K-means is the hard-assignment limit of the EM algorithm for a mixture of spherical Gaussians with equal, vanishing variances. In the GMM framework:
- E-step: compute soft assignments `γᵢₖ = P(zᵢ=k|xᵢ)` (probabilities)
- M-step: update means using weighted averages

As the variance σ² → 0, the soft assignments become hard (binary), and the EM updates reduce exactly to the K-means assignment and centroid update steps. This connection explains why K-means is sensitive to initialisation (EM has local optima), why it assumes spherical equal-variance clusters (the Gaussian assumption), and why K-means++ works (it mimics a good initialisation for EM).

**Lloyd's Algorithm and K-means++ Initialisation:** (See prior section for formulas.)

**Convergence Proof:** Each step (assignment + update) monotonically decreases J, bounded below by 0 → convergence. However, local optima are common.

**Why K-means Can Fail:**
K-means implicitly assumes clusters are convex, roughly equal in size, and have similar within-cluster variance. It will fail on:
- Elongated clusters (use GMM with full covariance)
- Clusters of vastly different sizes (centroids are pulled toward large clusters)
- Non-convex clusters like rings or crescents (use DBSCAN or spectral clustering)
- High-dimensional data where Euclidean distance is uninformative (use cosine distance or dimensionality reduction first)

**Spectral Clustering:**
```
1. Compute similarity matrix W: Wᵢⱼ = exp(-||xᵢ-xⱼ||²/(2σ²))
2. Compute unnormalised graph Laplacian: L = D - W  (D = degree matrix)
   Normalised Laplacian: L_sym = D^{-1/2}LD^{-1/2} = I - D^{-1/2}WD^{-1/2}
3. Compute first k eigenvectors of L_sym
4. Embed each point as its row in the eigenvector matrix
5. Run K-means on the embedded points
```
Spectral clustering can find non-convex clusters because the Laplacian eigenvectors encode graph connectivity, not Euclidean distance. Two points in the same cluster are well-connected in the graph even if they are far apart in Euclidean space. The eigenvalues of L count connected components: the multiplicity of eigenvalue 0 equals the number of connected components.

**Choosing K:** Elbow method, Silhouette, Gap Statistic, Davies-Bouldin Index, Calinski-Harabasz Index. (See prior section for formulas.)

---

### 6.2 DBSCAN

(See prior section for core definitions and algorithm.)

**Why DBSCAN is Robust to Noise:**
Unlike K-means, DBSCAN makes no assumptions about cluster shapes or number. It defines clusters as dense regions separated by sparse regions. Noise points (in sparse regions) are not assigned to any cluster. This makes DBSCAN naturally robust to outliers — they simply become noise. The key parameters ε and minPts define what "dense" means: a region is core if it contains at least minPts points within radius ε. Setting minPts = 2×d (twice the dimensionality) is a common heuristic justified by the fact that the intrinsic dimensionality of most datasets is lower than the ambient dimensionality.

**DBSCAN Complexity and HDBSCAN:** (See prior section.)

---

### 6.3 Hierarchical Clustering

**Agglomerative (bottom-up)** and **Divisive (top-down):** (See prior section.)

**Linkage Criteria:** (See prior section.)

**Ward's Method — Theoretical Justification:**
Ward's linkage merges the pair of clusters that minimises the increase in total within-cluster variance:
```
Δ(A,B) = (|A||B|/(|A|+|B|)) × ||μ_A - μ_B||²
```
This is equivalent to K-means with K decremented by 1 at each step. Ward's method tends to produce compact, spherical clusters of similar size — the same assumption as K-means — but unlike K-means, it is deterministic and hierarchical, giving a complete nested structure. However, it is sensitive to outliers (single outlier can drastically change the merge order) and assumes Euclidean distance.

**Dendrogram Interpretation:**
The height at which two clusters merge is the dissimilarity at the time of merging (depends on linkage criterion). Cutting the dendrogram at height h gives a flat clustering. The "ultrametric" property: the dissimilarity between clusters in a hierarchy satisfies `d(A,C) ≤ max(d(A,B), d(B,C))` — the indirect path through B is never shorter than the direct path. This stronger-than-triangle-inequality property is what makes hierarchical clusterings representable as trees.

---

### 6.4 Gaussian Mixture Models (GMM)

```
P(x) = Σₖ πₖ N(x | μₖ, Σₖ),   Σₖ πₖ = 1
```

**EM Algorithm and Why It Works:**
The EM algorithm maximises the log-likelihood `ℓ(θ) = Σᵢ log Σₖ πₖN(xᵢ|μₖ,Σₖ)`. The sum inside the log makes direct maximisation hard. EM introduces a lower bound via Jensen's inequality:
```
ℓ(θ) = Σᵢ log Σₖ πₖN(xᵢ|μₖ,Σₖ) ≥ Σᵢ Σₖ γᵢₖ log[πₖN(xᵢ|μₖ,Σₖ)/γᵢₖ] = Q(θ,θ_old)
```
The E-step computes the posterior assignments `γᵢₖ` (soft cluster memberships) that make this bound tight at the current θ. The M-step maximises the lower bound Q over θ, producing a new θ that is guaranteed to increase ℓ (since Q was tight at the old θ). Repeating this guarantees convergence to a local maximum. The key insight: the lower bound is separable in k after the E-step, so the M-step decomposes into independent weighted MLE problems for each component.

**EM Convergence:** Each iteration increases log-likelihood. Guaranteed convergence to a local maximum.

**Model Selection:** BIC = -2ℓ + k log n. **Covariance Constraints:** (See prior section.)

---

## 7. Dimensionality Reduction

### 7.1 Principal Component Analysis (PCA)

(See prior section for algorithm and formulas.)

**PCA as Maximum Variance Projection:**
The first PC `v₁` solves: `max_{||v||=1} vᵀCv` where `C = (1/n)XᵀX` is the sample covariance. By the variational characterisation of eigenvalues (`max_{||v||=1} vᵀCv = λ₁` achieved at `v = q₁`), the first PC is the leading eigenvector. Subsequent PCs are the leading eigenvectors of the residual covariance after projecting out previous components. The constraint of orthogonality ensures each PC captures independent variation.

**PCA as Minimum Reconstruction Error:**
Equivalently, the k-dimensional PCA projection minimises the average squared reconstruction error:
```
min_{V: VᵀV=I} (1/n)||X - XV VᵀXᵀ||²_F = Σᵢ>k λᵢ
```
Both characterisations (maximum variance and minimum reconstruction error) give the same solution — this is the Eckart-Young theorem applied to the sample covariance matrix.

**PCA Assumptions and When They Fail:**
PCA assumes:
1. Linear relationships between variables — captured by covariance
2. Variance = importance — high-variance directions are the signal, not noise
3. Orthogonal components — components are independent in the linear sense

These fail when: data lies on a non-linear manifold (use UMAP, kernel PCA), low-variance directions contain important discriminative information (PCA may remove features needed for classification), or independent components are desired (use ICA, not PCA — ICA finds statistically independent, not uncorrelated, components).

**Independent Component Analysis (ICA):**
```
ICA model: x = As  where s = [s₁,...,sₙ] are statistically independent sources
Goal: find W = A⁻¹ such that Wx ≈ s

FastICA algorithm:
  1. Whiten: z = Cx^{-1/2} x
  2. For each component w:
     a. w ← E[z g(wᵀz)] - E[g'(wᵀz)] w    (fixed-point update)
     b. w ← w / ||w||
     c. Orthogonalise against previous components
  g(u) = u³ (kurtosis), tanh(u) (super-Gaussian), exp(-u²/2) (sub-Gaussian)
```
ICA maximises non-Gaussianity (measured by kurtosis or negentropy) rather than variance. By the Central Limit Theorem, the sum of independent components is more Gaussian than any individual component — so maximising non-Gaussianity separates the mixture into its original non-Gaussian sources.

**Variance Explained, Whitening, Kernel PCA, PCA Limitations:** (See prior section.)

---

### 7.2 t-SNE

(See prior section for full formulas and practical notes.)

**Why Student-t in Low Dimensions:**
The "crowding problem" is fundamental: in d dimensions, moderate distances can span a wide range; in 2D, there is insufficient room. If we model low-dimensional similarities with the same Gaussian as high-dimensional, the moderate distances get "crushed" together — the visualisation loses inter-cluster structure. The Student-t distribution with 1 degree of freedom has much heavier tails than a Gaussian: the probability of large low-d distances decreases as `(1+||yᵢ-yⱼ||²)⁻¹` rather than exponentially. This "stretches" moderate distances in 2D, resolving the crowding problem and preserving inter-cluster separations.

**Gradient Interpretation:**
```
∂C/∂yᵢ = 4Σⱼ(pᵢⱼ-qᵢⱼ)(yᵢ-yⱼ)(1+||yᵢ-yⱼ||²)⁻¹
```
- When `pᵢⱼ > qᵢⱼ`: nearby in high-d but not in low-d → attractive force pulling yᵢ toward yⱼ
- When `pᵢⱼ < qᵢⱼ`: nearby in low-d but not in high-d → repulsive force pushing yᵢ away from yⱼ

The balance between these two forces shapes the visualisation. Early exaggeration (artificially increasing pᵢⱼ by 4× for the first 250 iterations) creates strong attractive forces, helping clusters form before repulsion takes over and spreads them apart.

---

### 7.3 UMAP

(See prior section for formulas and comparison with t-SNE.)

**UMAP's Theoretical Foundation:**
UMAP is based on the theory of fuzzy simplicial sets and Riemannian geometry. The core idea: assume data lies on a Riemannian manifold with uniform distribution. The geodesic distance on this manifold can be approximated by constructing a fuzzy simplicial set (a probabilistic extension of a simplicial complex — a generalisation of a graph). The membership strength `wᵢⱼ` between points decays with distance, normalised by the local density at each point (via `ρᵢ`, the distance to the nearest neighbour). This local normalisation makes UMAP adaptive to varying density — unlike t-SNE which uses a global bandwidth for each point (determined by perplexity).

---

## 8. Ensemble Learning

### 8.1 Why Ensembles Work

(See prior section for variance reduction formulas and Condorcet's theorem.)

**Diversity and Error Decomposition:**
The ensemble error can be decomposed as:
```
E[L_ensemble] = Ē[L_individual] - Ā[Ambiguity]
```
where the ambiguity measures how much individual models disagree with the ensemble mean. This is the "ambiguity decomposition" (Krogh & Vedelsby, 1995) for squared loss:
```
(f̄(x) - y)² = (1/B)Σ(fᵦ(x)-y)² - (1/B)Σ(fᵦ(x)-f̄(x))²
error of ensemble = avg individual error - avg disagreement
```
To minimise ensemble error: (1) reduce individual errors (good models), (2) increase disagreement (diverse models). This formalises the intuition that ensembles benefit from diversity — even accurate models that agree too strongly don't benefit from being combined.

**Bagging, Random Forests, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost, Stacking:** (See prior sections for all formulas and detailed theory.)

**Gradient Boosting as Functional Gradient Descent:**
Standard gradient descent operates in parameter space: `θₜ₊₁ = θₜ - α ∂L/∂θ`. Gradient boosting operates in function space: `Fₜ₊₁ = Fₜ + α hₜ` where `hₜ` is a new model fit to the negative functional gradient `rᵢ = -∂L/∂F|_{F=Fₜ₋₁}`. The functional gradient is:
```
rᵢ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]_{F=Fₜ₋₁}
```
By fitting a regression tree to these residuals, we find the best weak learner (restricted to the tree function class) that points in the direction of steepest descent in function space. This perspective unifies boosting with gradient descent and immediately suggests generalisations to any differentiable loss function, not just squared error.

**Why XGBoost Outperforms Vanilla GBM:**
The second-order Taylor expansion in XGBoost is key. Standard GBM uses only the first-order (gradient) information — like gradient descent. XGBoost uses second-order (Hessian) information — like Newton's method. The Newton step has faster convergence because it accounts for the curvature of the loss. Additionally, the regularisation terms `γT + (λ/2)||w||²` directly penalise model complexity, making XGBoost better calibrated and less prone to overfitting. The optimal leaf score `wⱼ* = -Gⱼ/(Hⱼ+λ)` is the Newton step for each leaf, exactly optimal under the local quadratic approximation.

---

## 9. Time Series Analysis

### 9.1 Stationarity

(See prior section for definitions, tests, and differencing.)

**Why Stationarity is Required for ARIMA:**
The theoretical basis for ARIMA and related models rests on the Wold decomposition theorem: every covariance-stationary process `{Yₜ}` can be written as `Yₜ = Σⱼ ψⱼ εₜ₋ⱼ + vₜ` — a sum of a moving average of white noise (the non-deterministic part) and a linearly deterministic component. This decomposition underpins all stationary time series modelling. For non-stationary series (unit root), the Wold decomposition does not apply. Differencing transforms an I(d) series to an I(0) (stationary) series, restoring the applicability of the Wold decomposition and hence of ARIMA.

**Spurious Regression:**
If two non-stationary I(1) series `Yₜ` and `Xₜ` are independent random walks, regressing Y on X gives a high R² and a significant t-statistic with probability approaching 1 as n → ∞. This is spurious regression — statistical significance without any causal or structural relationship. The reason: both series drift, and any two drifting series will appear correlated over a finite sample. Cointegration testing determines whether non-stationary series have a genuine long-run relationship (a stationary linear combination exists) — if so, regression is valid despite non-stationarity.

---

### 9.2 ACF and PACF

(See prior section for definitions and model identification.)

**ACF of an AR(p) Process:**
For `Yₜ = φ₁Yₜ₋₁ + ... + φₚYₜ₋ₚ + εₜ`, the autocorrelations satisfy the Yule-Walker equations:
```
ρₖ = φ₁ρₖ₋₁ + φ₂ρₖ₋₂ + ... + φₚρₖ₋ₚ   for k ≥ 1
```
This is a linear recurrence relation, with solution `ρₖ = Σⱼ Aⱼ λⱼᵏ` where `λⱼ` are the roots of the characteristic polynomial `λᵖ - φ₁λᵖ⁻¹ - ... - φₚ = 0`. For a stationary AR(p), all roots are inside the unit circle, so `|λⱼ| < 1` and the ACF decays geometrically — matching the signature in the table above. For a unit root (`λⱼ = 1`), the ACF decays very slowly — the signature of non-stationarity.

---

### 9.3 ARIMA, SARIMAX, Exponential Smoothing

(See prior sections for all formulas.)

**ETS State-Space Representation:**
The exponential smoothing methods (SES, Holt, Holt-Winters) are special cases of a general state-space framework. For Simple Exponential Smoothing:
```
Observation:   yₜ = lₜ₋₁ + εₜ
State:         lₜ = lₜ₋₁ + α εₜ
```
The parameter α controls how quickly the level responds to new observations. `εₜ` is white noise. This state-space formulation enables exact likelihood computation (Kalman filter), principled estimation of initial conditions, and automatic model selection via AIC across all error/trend/seasonality combinations.

**Information Criteria for Model Selection:**
```
AIC  = -2ℓ + 2k
AICc = AIC + 2k(k+1)/(n-k-1)   (small-sample correction)
BIC  = -2ℓ + k log n
```
AIC is derived from minimising the expected KL divergence between the true model and the fitted model — it penalises complexity at 2 per parameter. BIC is derived from Bayesian model comparison and penalises more strongly at `log n` per parameter, favouring simpler models for large n. For large n, BIC tends to select the true model (consistency) while AIC does not. For small n or complex models, AICc should be preferred over AIC.

---

## 10. Deep Learning Foundations

### 10.1 Perceptron & MLP

**Forward Pass, Universal Approximation Theorem, Depth vs Width:** (See prior section.)

**Expressivity: What Functions Neural Networks Can Represent:**
The universal approximation theorem guarantees existence but not learnability. More specific results:
- ReLU networks with L layers and N total neurons compute piecewise linear functions with at most O(N^L / L!) linear regions
- Deep networks create exponentially more linear regions than shallow networks with the same number of neurons
- Boolean functions: a depth-2 network can represent any Boolean function on n bits with O(2ⁿ) neurons; a depth-O(log n) network can do the same with O(n) neurons — exponential depth vs width tradeoff

**Representational Capacity vs Sample Complexity:**
A network can represent an exponential number of functions, but learning the right one from data requires sufficient samples. The sample complexity grows with the VC dimension (O(W log W) for W-parameter networks), but the effective capacity relevant to generalisation is much lower for networks trained with SGD on real data — SGD has an implicit regularisation bias toward simple, smooth functions.

---

### 10.2 Activation Functions

(See prior section for all formulas.)

**Why Activation Functions Must Be Non-Linear:**
Without non-linearities, a deep network `f(x) = W_L ... W_2 W_1 x` is equivalent to a single linear transformation (the product of weight matrices). No matter how deep, a network of linear layers can only represent linear functions. Non-linearity is what allows deep networks to represent complex decision boundaries.

**Dying ReLU Problem:**
If a ReLU neuron's pre-activation `zᵢ` is always negative (e.g., due to a large negative bias), the gradient `g'(zᵢ) = 𝟙[zᵢ>0] = 0` for all inputs. The neuron permanently outputs 0 and receives no gradient signal — it is "dead." Dead neurons are caused by: large learning rates (which can drive weights/biases to large negative values), poor initialisation, or aggressive regularisation. Leaky ReLU (g'(z) = α < 1 for z ≤ 0) prevents this by allowing a small gradient even for negative activations.

**GELU's Stochastic Interpretation:**
GELU(x) = x × Φ(x) where Φ is the standard normal CDF. This can be interpreted as: "keep the input with probability Φ(x), otherwise drop it." For large positive x, Φ(x) ≈ 1 (always kept). For large negative x, Φ(x) ≈ 0 (always dropped). This is a soft, data-dependent version of Dropout — the gating depends on the value of the input itself, creating an input-dependent stochastic regularisation. This is why GELU outperforms ReLU for Transformers: the stochastic gating provides implicit regularisation in the attention mechanism.

---

### 10.3 Backpropagation — Full Derivation

(See prior section for error signals, parameter gradients, and vanishing gradient analysis.)

**Backpropagation as Reverse-Mode Automatic Differentiation:**
Backpropagation is a specific instance of reverse-mode automatic differentiation applied to computational graphs. Every operation in the forward pass is recorded in a computation graph. During the backward pass, the chain rule is applied to every edge in the graph in reverse topological order. The key efficiency insight: each intermediate gradient is computed exactly once (memoisation), giving total complexity O(forward pass cost) regardless of the number of parameters. For a network with P parameters, this is O(P) — far better than the O(P²) naive finite-difference approach.

**Exploding Gradients and Gradient Clipping:**
```
g := g × min(1, c/||g||)   (norm clipping, preserves direction)
gⱼ := sign(gⱼ) × min(|gⱼ|, c)  (coordinate clipping)
```
Norm clipping is preferred because it preserves the direction of the gradient update, only scaling its magnitude. Coordinate clipping distorts the direction. For RNNs, gradient explosion is catastrophic because one large update can destroy learned parameters. The threshold c is a hyperparameter, typically set between 0.5 and 5.

---

### 10.4 Optimisation Algorithms

(See prior section for SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW, Lion, and learning rate schedules.)

**Why Adam Works (and Sometimes Doesn't):**
Adam adapts the learning rate for each parameter individually: parameters with consistently large gradients get small effective learning rates; those with small or rare gradients get large ones. This is ideal for sparse gradients (many zero gradients, few large ones, as in NLP embeddings). The bias correction terms `(1-β₁ᵗ)` and `(1-β₂ᵗ)` are critical in the early steps: without them, `m` and `v` are heavily underestimated (they start at zero), causing too-large early updates.

Adam's known failure: in some settings (certain image models, convex problems), Adam converges to worse solutions than SGD+momentum. The reason: Adam's adaptive learning rates effectively change the geometry of the loss landscape. "Generalisable" directions (flat loss directions) may receive larger effective learning rates, while "generalisation-harming" directions (sharp) may also receive large rates. SGD with momentum traverses the loss landscape more faithfully. AdamW partially addresses this by correcting L2 regularisation, but the fundamental geometry issue remains.

**Learning Rate Warmup — Why It Is Needed:**
At the start of training, the estimates of `m` (first moment) and `v` (second moment) are close to zero. Even with bias correction, the noise in these estimates is high. A large initial learning rate combined with noisy gradient estimates leads to catastrophic updates that destroy the pre-trained initialisation (for fine-tuning) or put the parameters in a bad region of the loss landscape. Linear warmup gradually increases the learning rate, allowing `m` and `v` to stabilise before large updates are made.

---

### 10.5 Weight Initialisation

(See prior section for Xavier, He, LeCun, Orthogonal, and Sparse init.)

**The Exploding/Vanishing Variance Problem:**
Consider a deep network with L layers, each `zˡ = Wˡ aˡ⁻¹` with `aˡ = ReLU(zˡ)`. If each `Wˡ` has i.i.d. entries from `N(0, σ²)`:
```
Var(zˡⱼ) = n_{l-1} × σ² × Var(aˡ⁻¹ᵢ)
```
For ReLU: `Var(a) = Var(z)/2` (ReLU zeros half the activations, reducing variance by half). So:
```
Var(zˡ) = (n_{l-1} × σ² / 2) × Var(zˡ⁻¹)
```
For variance to be preserved: `n_{l-1} × σ² / 2 = 1` → `σ² = 2/n_{l-1}`. This is exactly He initialisation. The `2` compensates for ReLU's variance halving. Without this, variances grow or shrink exponentially with depth, causing activations to explode or vanish.

---

## 11. Convolutional Neural Networks (CNN)

### 11.1 The Convolution Operation

(See prior section for formulas and inductive biases.)

**Why Translation Equivariance is the Right Prior for Images:**
A cat is a cat regardless of where it appears in the image. Translation equivariance encodes this prior: if the input shifts by δ pixels, the feature map shifts by δ pixels (scaled by stride). No new learning is needed — the same filter detects the cat everywhere. This is enormously sample-efficient: instead of learning a separate "cat detector" for each possible position (which would require n_positions × n_filters parameters), we learn a single filter that works everywhere (n_filters parameters).

**The Trade-Off Between Receptive Field and Parameters:**
Two 3×3 conv layers have the same receptive field (5×5) as one 5×5 layer, but fewer parameters (`2 × 9C² < 25C²` for C channels) and two non-linearities (more expressive). This is why modern architectures stack small kernels. Depth compensates for kernel size.

**Convolution Types, Receptive Field, Output Shape:** (See prior section.)

---

### 11.2 Pooling & Normalisation

(See prior section for max/average pooling, Batch Norm, Layer Norm, Instance Norm, Group Norm, RMS Norm.)

**Why Batch Normalisation Accelerates Training:**
Batch normalisation reduces "internal covariate shift" — the change in the distribution of layer inputs as parameters of previous layers change during training. Without BN, each layer must constantly adapt to a shifting input distribution, slowing learning. With BN, each layer always receives approximately zero-mean, unit-variance inputs, making the optimisation landscape much smoother and allowing larger learning rates. BN also provides slight regularisation (each mini-batch is a noisy estimate of the normalisation statistics), reducing dependence on Dropout.

**The Mechanics of Batch Norm During Training vs Inference:**
During training, each mini-batch computes its own statistics `(μ_B, σ²_B)`. This introduces noise into the normalisation, acting as a regulariser. During inference, we use running statistics (exponential moving averages of training batch statistics) for deterministic, consistent predictions. The learnable parameters `(γ, β)` allow the network to "undo" the normalisation if needed — without them, the normalisation might destroy useful information (e.g., if the mean of a pre-activation is genuinely informative for the task).

---

### 11.3 CNN Architectures

(See prior section for AlexNet, VGG, GoogLeNet, ResNet, DenseNet, EfficientNet, ViT.)

**ResNet's Skip Connection — The Residual Principle:**
The hypothesis: it is easier for a layer to learn a small residual (deviation from identity) than to learn the full mapping from scratch. Formally, if the desired mapping is `H(x)`, a residual block learns `F(x) = H(x) - x`, and the output is `H(x) = F(x) + x`. If the optimal mapping is close to identity, `F(x) → 0` is easy to achieve with near-zero weights, whereas forcing a stack of layers to approximate the identity is harder.

**Skip Connections and Gradient Flow:**
The gradient through a residual block is:
```
∂L/∂x = ∂L/∂y × (∂F(x)/∂x + I)
```
The identity term `I` ensures the gradient is always at least as large as 1 (in the direction of the identity) — preventing vanishing. For very deep networks (1000+ layers), this enables training that would be impossible without residual connections. Huang et al. (2016) showed that ResNet can be interpreted as an ensemble of paths of varying length, where the shorter paths dominate the gradient flow.

---

## 12. Recurrent Neural Networks (RNN)

### 12.1 Vanilla RNN

(See prior section for equations, BPTT, and vanishing/exploding gradient analysis.)

**The Vanishing Gradient is a Memory Problem:**
The vanishing gradient in RNNs is not merely a numerical issue — it is a fundamental limitation on what RNNs can learn. If the gradient of the loss with respect to hidden state at time step k vanishes as we look further back (larger t-k), the network cannot learn dependencies involving events more than a few steps apart. In practice, vanilla RNNs struggle with dependencies beyond 10-20 steps. The fundamental cause: each step multiplies by `diag(g'(zᵢ))Wₕ`, and if this contraction factor is less than 1, the gradient decays exponentially. LSTM solves this with the cell state highway.

**Echo State Networks (ESN) — An Alternative:**
ESNs fix the recurrent weights `Wₕ` (initialised with `|λ_max(Wₕ)| < 1`, the "echo state property") and only train the output layer. The fixed recurrent weights create a "reservoir" of diverse nonlinear temporal features. Training is just a linear regression on top of these features — no backpropagation through time needed. ESNs are fast to train and can handle some long-term dependencies because the reservoir state encodes a rich history. However, they are less expressive than trained RNNs.

---

## 13. Long Short-Term Memory (LSTM) & GRU

### 13.1 LSTM

(See prior section for all gate equations and parameter count.)

**How LSTM Solves the Vanishing Gradient:**
The cell state `Cₜ` flows through time with gradient:
```
∂Cₜ/∂Cₜ₋₁ = fₜ
```
No matrix multiplication — just element-wise multiplication by the forget gate. If `fₜ ≈ 1`, the gradient flows unchanged backward through time, enabling learning of arbitrarily long dependencies. If `fₜ ≈ 0`, old information is selectively forgotten. The LSTM learns to set `fₜ ≈ 1` for time steps where long-range dependencies exist, and `fₜ ≈ 0` when a new pattern begins.

**Comparison with Gated Recurrent Unit (GRU):**

| Property | LSTM | GRU |
|---|---|---|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| State | Separate cell + hidden | Single hidden state |
| Parameters | 4× | 3× |
| Performance | Better on very long sequences | Competitive on shorter |
| Memory | Higher | Lower |

**When to Choose LSTM vs GRU:**
GRU is generally preferred for shorter sequences and when computational efficiency matters. LSTM is preferred for very long sequences where the separate cell state highway provides a cleaner gradient path. For sequences > 500 steps, LSTM typically outperforms GRU. For sequences of 50-200 steps, GRU is competitive with 25% fewer parameters.

---

### 13.2 Sequence-to-Sequence & Attention

(See prior section for encoder-decoder architecture and Bahdanau/Luong attention formulas.)

**Why Fixed-Size Context Bottleneck Fails:**
In vanilla seq2seq, the encoder compresses the entire input sequence into a single fixed-size vector c (typically the last hidden state). For long sequences, this bottleneck is catastrophically information-lossy — the encoder must compress an arbitrarily long sequence into a constant-dimension vector. Attention solves this: instead of a single context vector, the decoder at each step computes a dynamic weighted sum `cₜ = Σⱼ αₜⱼ hⱼ` over all encoder hidden states. The alignment weights `αₜⱼ` are learned — they determine which source positions are most relevant for generating each target token. This gives the decoder access to the entire source sequence, not a compressed summary.

**Attention as Soft Template Matching:**
The attention weight `αₜⱼ` can be interpreted as the probability that source position j is aligned with target position t. The alignment model `eₜⱼ = score(sₜ₋₁, hⱼ)` measures compatibility between the decoder state (what the decoder is currently trying to produce) and the encoder state (what the encoder has captured at position j). High compatibility → high attention weight → the context vector draws heavily from position j. This is soft template matching: the decoder finds the source positions that match its current "query."

---

## 14. Advanced Deep Learning

### 14.1 VAE, GAN, Diffusion Models

(See prior section for ELBO, reparameterisation trick, GAN objective, WGAN, and DDPM formulas.)

**VAE ELBO — Information-Theoretic Interpretation:**
The ELBO (Evidence Lower Bound) is:
```
L = E_q[log p(x|z)] - KL(q(z|x) || p(z))
```
The first term is the reconstruction quality (decoder quality). The second term is the KL divergence between the approximate posterior (encoder output) and the prior — it penalises the encoder for deviating from the prior, acting as a regulariser on the latent space. Maximising the ELBO is equivalent to minimising `KL(q(z|x,λ) || p(z|x)) - log p(x)` — the gap between the approximate and true posterior, plus a constant. The variational approximation `q(z|x)` is the best we can do within the Gaussian family.

**Posterior Collapse in VAEs:**
A known failure mode: the KL term becomes dominant, forcing q(z|x) → p(z) (the posterior collapses to the prior). The decoder then ignores z entirely and learns a non-conditional generative model. This happens when the decoder is powerful enough to model p(x) without z. Solutions: β-VAE with small β (weight KL less), annealing β from 0 to 1 during training, or using a discrete latent space (VQ-VAE).

**Score Matching and Diffusion Models:**
The score function of a distribution p is `s(x) = ∇_x log p(x)`. Score matching trains a neural network `s_θ(x)` to approximate this score without computing the intractable normalisation constant. Diffusion models connect to score matching via Tweedie's formula: the optimal denoiser for Gaussian noise is `E[x₀|xₜ] = xₜ + σₜ² s(xₜ,t)`. Therefore, training a denoising network is equivalent to training a score estimator, and the denoising objective `||ε - ε_θ(xₜ,t)||²` is a re-parameterised score matching loss.

---

### 14.2 Regularisation

(See prior section for Dropout, Early Stopping, Data Augmentation, Label Smoothing, Spectral Normalisation.)

**Label Smoothing — Information-Theoretic View:**
Label smoothing replaces one-hot targets with `y_smooth = (1-ε)y_hard + ε/K`. The optimal logits under cross-entropy with smooth labels are `fₖ*(x) = log p(y=k|x) + c` for some constant c — the model should output calibrated probabilities, not extreme logits. Without smoothing, cross-entropy drives the correct logit to +∞ (and others to -∞), making the model overconfident. Smoothing adds a KL divergence penalty that prevents logits from becoming too large:
```
H(y_smooth, softmax(f)) = (1-ε) H(y_hard, softmax(f)) + ε H(Uniform, softmax(f))
```
The entropy term `H(Uniform, softmax(f))` is maximised when all logits are equal — preventing collapse to a single class and encouraging calibrated uncertainty.

---

## 15. 🔥 Transformers — Complete Deep Dive

### 15.1 Motivation — Why Transformers?

(See prior section.)

**The Fundamental Advantage: Parallelism Over Sequential Structure:**
RNNs process sequences one token at a time. At training time, computing the hidden state at position t requires the hidden state at position t-1 — the computation is sequential and cannot be parallelised. For a sequence of length T and model dimension d, the serial depth is T operations. Transformers compute all positions simultaneously: the attention operation `softmax(QKᵀ/√dₖ)V` processes all T positions in parallel using matrix operations — the serial depth is just 1 (plus the depth of MLP layers). On modern GPUs with thousands of parallel cores, this difference in serial depth is the key to Transformer scalability.

---

### 15.2 Scaled Dot-Product Attention

(See prior section for full formulas.)

**Attention as Soft Dictionary Lookup:**
A dictionary stores (key, value) pairs and retrieves the value corresponding to an exact query match. Attention is a soft (differentiable) version: given a query q, it computes a weighted sum of all values, where the weight for value `vⱼ` is proportional to the similarity between query q and key `kⱼ`. The similarity is measured by the dot product `q·kⱼ/√dₖ`, softmaxed to obtain a probability distribution over values. This allows the model to retrieve a blend of information from multiple memory locations — unlike a hard dictionary which retrieves exactly one entry.

**Why Attention is O(n²):**
The score matrix `S = QKᵀ` has shape `(n × n)` — every query attends to every key. For n=1024 tokens, this is 1 million entries. For n=100,000 tokens, it's 10 billion entries, requiring ~40GB memory for float32. This quadratic scaling with sequence length is the core bottleneck of standard Transformers. Sparse and linear attention variants aim to reduce this, at the cost of either approximation error or reduced expressive power.

**Multi-Head, Transformer Block, Positional Encodings, Architecture Types, Pre-training, Scaling Laws, Efficient Attention, Post-Training, PEFT, Inference Optimisation, LLM Architectures, ViT:** (See prior sections for all formulas and detailed theory.)

**The Induction Head Mechanism:**
Mechanistic interpretability reveals that Transformers learn fundamental computational primitives. "Induction heads" are a two-layer attention circuit that implements in-context learning: if the sequence contains the pattern `[A][B]...[A]`, the induction head predicts `[B]` at the second `[A]`. This is implemented by:
1. Layer 1: a "previous token head" that rotates the residual stream of each position to contain information from the previous token
2. Layer 2: a "key-query" matching head that attends to positions whose previous token matches the current token

This circuit is thought to be responsible for the "few-shot learning" capability of language models — they learn to recognise and complete patterns from context.

**Emergent Abilities and Phase Transitions:**
As language models scale, new capabilities emerge sharply at certain scale thresholds — they are near-zero below the threshold and non-trivial above it. Wei et al. (2022) documented these "emergent abilities" for tasks like multi-step arithmetic, analogy, and chain-of-thought reasoning. The theoretical explanation is debated: one view holds that emergent abilities represent genuine phase transitions in learned representations; another argues they are artefacts of metrics that discretise continuous underlying capabilities.

---

## 16. 🦜 LangChain & LLM Application Engineering

### 16.1 LangChain Overview

(See prior section for core abstractions and installation.)

**Why RAG Instead of Just Fine-Tuning?**
Fine-tuning encodes knowledge in the model's weights — it is static, expensive, and requires re-training to incorporate new information. RAG externalises knowledge in a retrieval database — it is dynamic (update the database without retraining), cheaper (no GPU training), and interpretable (you can inspect which documents were retrieved). Fine-tuning is better for learning new skills (style, format, reasoning patterns) or aligning model behaviour. RAG is better for providing up-to-date factual knowledge. Modern systems often combine both.

**RAG Failure Modes and Mitigations:**

| Failure Mode | Description | Mitigation |
|---|---|---|
| Retrieval failure | Correct document not retrieved | Hybrid search (dense + BM25), query expansion |
| Context not used | LLM ignores retrieved context | System prompt engineering, context formatting |
| Hallucination despite context | LLM fabricates despite good context | Faithfulness evaluation (RAGAS), constrained decoding |
| Context window overflow | Too many chunks exceed limit | Contextual compression, chunk size tuning |
| Semantic search miss | Query and document use different vocabulary | Keyword/BM25 backup, re-ranking |

**LLM/Chat Models, Prompt Templates, LCEL, Output Parsers, Document Loaders, Text Splitters, Embeddings, Vector Stores, RAG, Memory, Agents, LangGraph, Production Pipeline, Evaluation:** (See prior sections for all code and theory.)

**Chunking Strategy Theory:**
The optimal chunk size trades retrieval precision against context completeness. Small chunks (200-400 tokens) retrieve precise, relevant passages but may miss context that spans multiple sentences. Large chunks (1500-2000 tokens) provide more context but retrieve more irrelevant text, diluting the signal. The chunk overlap `(100-200 tokens typically)` ensures that sentences near chunk boundaries are not split — a sentence straddling two chunks would be "lost" without overlap. Semantic chunking (splitting at content boundaries rather than fixed token counts) improves both precision and context completeness by keeping semantically coherent units together.

**Vector Search Theory — HNSW:**
Hierarchical Navigable Small World (HNSW) graphs underpin most production approximate nearest-neighbour search (FAISS, Weaviate, Pinecone). HNSW builds a multi-layer graph where:
- Layer 0: all nodes, connected to their K nearest neighbours
- Layer 1: a random subset of nodes (skip list style)
- ...
- Layer L: very few "hub" nodes with long-range connections

Search starts at the top layer (few nodes, fast traversal to approximate region) and greedily descends, refining at each layer. The multi-scale structure gives O(log n) search complexity. The theoretical guarantee: HNSW achieves near-exact recall at query-time with much less work than exact search, because the graph structure mirrors the intrinsic geometry of the embedding space.

---

## 17. Handling Imbalanced Data

### 17.1 Understanding Class Imbalance

(See prior section for Imbalance Ratio, metric choices.)

**The Theoretical Problem with Imbalanced Data:**
Standard ERM minimises the average loss `(1/n)Σᵢ L(yᵢ, ŷᵢ)`. With 99% negative examples, a model that always predicts negative achieves 99% accuracy — it minimises the ERM objective without learning anything about the positive class. The issue is that the loss function weights all samples equally, but the classes are not equally important. Solutions all modify this weighting:
- Oversampling: increases the effective count of minority examples
- Undersampling: decreases the effective count of majority examples
- Cost-sensitive learning: directly modifies the loss function weights
- Threshold moving: adjusts the decision boundary post-training

**SMOTE, ADASYN, Borderline-SMOTE, Focal Loss, Class Weights:** (See prior section for formulas.)

**Why Focal Loss is Theoretically Superior to Resampling:**
Resampling changes the training data distribution, which changes what the model is learning. A model trained on oversampled data learns `P(y=1|x) = (oversampled rate)` — not the true posterior probability. Focal loss keeps the data distribution unchanged but modifies the loss:
```
FL(p) = -α(1-p)^γ log(p)
```
The factor `(1-p)^γ` down-weights easy examples (high p for positives) and up-weights hard ones (low p for positives or high p for negatives). This dynamically focuses training on the informative, difficult examples without distorting the distribution. The model still learns the true posterior probability, just more efficiently.

---

## 18. Model Evaluation & Selection

### 18.1 Classification Metrics

(See prior section for all formulas.)

**The Precision-Recall Trade-off — Decision Threshold:**
A classifier outputs scores `s(x)` which are thresholded at τ to produce class predictions: `ŷ = 1 if s(x) ≥ τ else 0`. As τ increases:
- Precision increases (we only call positive when very confident)
- Recall decreases (we miss many positives)

The PR curve traces this trade-off as τ varies. The area under this curve (AUPRC) summarises classifier quality across all thresholds. For imbalanced datasets, AUPRC is more informative than AUC-ROC because the ROC curve's x-axis (FPR) is dominated by the large negative class — even a bad positive recall can give a high AUC-ROC when true negatives are plentiful.

**Matthews Correlation Coefficient (MCC) — Why It is the Most Informative Single Metric:**
MCC is the phi coefficient applied to the binary confusion matrix. It is the only metric that correctly handles all four quadrants of the confusion matrix simultaneously:
```
MCC = (TP·TN - FP·FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]
```
MCC = +1 only when all four rates are simultaneously good. Accuracy can be maximised by predicting only the majority class. F1 can be maximised without caring about TN. MCC = 0 means no better than chance. Chicco & Jurman (2020) demonstrated that MCC is more informative than F1 and accuracy for binary classification with imbalanced classes.

---

### 18.2 Regression Metrics

(See prior section for MAE, MSE, RMSE, MAPE, R², Huber loss, Quantile loss.)

**Why MSE Penalises Outliers Disproportionately:**
MSE = `(1/n)Σ(yᵢ-ŷᵢ)²`. An error of 10 contributes 100 to MSE; an error of 1 contributes 1 — a 10× larger error contributes 100× to the loss. This quadratic penalisation makes MSE very sensitive to outliers: a single large error can dominate the training signal, distorting the model toward fitting that one point. MAE penalises linearly (error 10 contributes 10, error 1 contributes 1) — much more robust. Huber loss interpolates: MAE for large errors (robust), MSE for small errors (smooth gradient at 0).

**Prediction Intervals via Quantile Regression:**
Quantile loss `L_q(y,ŷ) = q(y-ŷ)⁺ + (1-q)(ŷ-y)⁺` with q=0.05 and q=0.95 produces the 5th and 95th percentile predictions. The 90% prediction interval `[ŷ₀.₀₅, ŷ₀.₉₅]` contains the true y 90% of the time (if the model is well-calibrated). Unlike confidence intervals (which concern the mean), prediction intervals cover individual observations — they account for both parameter uncertainty and irreducible noise.

---

### 18.3 Cross-Validation Strategies

(See prior section for all CV methods.)

**Nested CV — Why It's Required for Unbiased Model Selection:**
Suppose we use the same data to (1) select hyperparameters and (2) estimate test performance. The selected hyperparameters are those that happened to work best on that particular validation fold — an optimistic estimate. Nested CV separates these: the inner loop selects hyperparameters (on the inner folds), and the outer loop estimates performance (on the outer test fold, which was never used for hyperparameter selection). The outer performance estimate is therefore unbiased. In non-nested CV, the commonly reported performance is optimistic — sometimes dramatically so for small datasets.

---

### 18.4 Hyperparameter Tuning

(See prior section for Grid Search, Random Search, Bayesian Optimisation, Hyperband, BOHB, PBT, NAS.)

**Why Random Search Beats Grid Search:**
Bergstra & Bengio (2012) showed that most hyperparameters are irrelevant — the objective function is nearly constant in some directions. In a grid, many evaluations are wasted exploring irrelevant dimensions. Random search samples the important dimensions more thoroughly: if only 2 of 10 dimensions matter, random search with 100 trials provides ~10 unique values for each important dimension, while a grid with the same budget provides only `100^{1/10} ≈ 1.6` unique values per dimension. The argument extends: the more irrelevant dimensions, the more random search outperforms grid search.

---

## 19. Regularisation & Optimisation

### 19.1 Norms & Regularisation Theory

(See prior section for all norm formulas and regularisation connections.)

**Implicit Regularisation by SGD:**
Beyond explicit regularisation (L1, L2, Dropout), SGD itself has an implicit regularisation effect. For overparameterised linear models, gradient descent from zero initialisation converges to the minimum-norm solution (equivalent to L2 regularisation with λ → 0⁺). For neural networks, the implicit bias of SGD depends on the learning rate, batch size, and architecture. Large learning rate → stronger implicit regularisation, preference for flat minima. Small batch size → noisier gradients → more exploration → better generalisation (Keskar et al., 2017).

**Flat vs Sharp Minima:**
Two solutions with the same training loss may generalise differently. A "flat" minimum has a wide basin — nearby points in parameter space also have low loss. A "sharp" minimum has a narrow basin — small perturbations increase loss greatly. Intuitively, flat minima generalise better: if the test distribution differs slightly from train (inevitable due to finite samples), a flat minimum stays in a low-loss region. SGD with large learning rates preferentially finds flat minima (the gradient noise effectively "bounces" out of sharp narrow basins but stays in wide flat ones).

---

### 19.2 Normalisation Layers

(See prior section for BN, LN, IN, GN, RMS Norm, Layer Scale.)

**Normalisation and the Optimisation Landscape:**
Batch normalisation dramatically improves the conditioning of the optimisation landscape, not just by reducing internal covariate shift. Santurkar et al. (2018) showed that BN makes the loss function smoother (smaller Lipschitz constant of gradients) — the landscape has fewer sharp peaks and valleys, and gradient descent steps are more reliably downhill. This allows larger learning rates, faster convergence, and reduced sensitivity to initialisation. The regularisation effect is secondary.

---

## 20. Probabilistic Machine Learning

### 20.1 Gaussian Processes

(See prior section for all formulas and kernel types.)

**GP as Distribution Over Functions:**
A Gaussian Process is a prior distribution over functions: `f ~ GP(m, k)` means that for any finite set of inputs `{x₁,...,xₙ}`, the function values `[f(x₁),...,f(xₙ)]` are jointly Gaussian with mean `[m(xᵢ)]` and covariance `[k(xᵢ,xⱼ)]`. The kernel encodes all structural assumptions about the function: smoothness (RBF kernel → infinitely differentiable functions), periodicity (periodic kernel), linear trends (linear kernel), or combinations thereof. GPs provide principled uncertainty estimates: predictions far from training data have high variance (wide posterior), while interpolations between data points have low variance (narrow posterior).

**Sparse GPs and Inducing Points:**
Exact GP inference requires `O(n³)` for the Cholesky decomposition of the `n×n` kernel matrix. For large n (>10,000), this is infeasible. Sparse GP methods introduce M << n "inducing points" `Z = {z₁,...,zₘ}` and approximate the posterior:
```
p(f*|X,y) ≈ q(f*) = ∫ p(f*|u) q(u) du
q(u) = N(μₘ, Aₘ)   (variational posterior over inducing outputs)
```
Training minimises `KL(q(f) || p(f|y))`, or equivalently maximises the ELBO. Complexity: `O(nm² + m³)` — linear in n if m is fixed. Good inducing points are spread throughout the input space to cover the training data; optimal inducing points can be learned by gradient descent on the ELBO.

---

### 20.2 Hidden Markov Models (HMM)

(See prior section for all three algorithms: Forward, Viterbi, Baum-Welch.)

**The Three HMM Problems and Why They Require Different Algorithms:**
1. **Evaluation** (Forward algorithm): Computing `P(x₁,...,xT|λ)` naively requires summing over all `N^T` possible state sequences — exponential. The forward algorithm uses dynamic programming to factorise this sum, reducing to `O(TN²)`.
2. **Decoding** (Viterbi): Finding the most likely state sequence requires maximising over `N^T` paths. Viterbi replaces the sum with a max in the forward recursion, giving the same `O(TN²)` complexity.
3. **Learning** (Baum-Welch): Standard MLE for HMMs is intractable because the state sequence is latent (unobserved). Baum-Welch is the EM algorithm applied to HMMs — the E-step computes posterior expectations over hidden states, and the M-step updates parameters using these expected counts.

---

### 20.3 Calibration

(See prior section for reliability diagrams, calibration metrics, and calibration methods.)

**Why Neural Networks are Overconfident:**
Guo et al. (2017) showed that modern neural networks are significantly overconfident — their predicted probabilities are much higher than empirical accuracies. A network predicting 95% confidence is correct only 80% of the time (for example). The cause is the combination of (1) large model capacity — the network memorises training data with near-perfect confidence, (2) cross-entropy loss — it rewards higher probabilities for the correct class without bound, driving logits to extremes, and (3) batch normalisation and weight decay — they have unintended effects on calibration.

Temperature scaling is the simplest fix: divide all logits by a single scalar T > 1 before softmax. This "softens" the distribution, reducing overconfidence. T is fit on a separate validation set by minimising NLL. The key observation: temperature scaling does not change the ranking of classes (same top-1 accuracy), only the confidence values.

---

## 21. Repository Structure

```
📦 Machine-Learning-Deep-Learning-Modules/
│
├── 📂 Mathematical-Foundations/
│   ├── 📓 linear_algebra_review.ipynb
│   ├── 📓 probability_statistics.ipynb
│   └── 📓 calculus_optimisation.ipynb
│
├── 📂 Regression/
│   ├── 📓 linear_ridge_lasso.ipynb
│   ├── 📓 ANN_for_regression.ipynb
│   └── 📓 house_pricing.ipynb
│
├── 📂 Classification/
│   ├── 📓 logistic_regression.ipynb
│   ├── 📓 KNN_practice.ipynb
│   ├── 📓 naive_bayes.ipynb
│   ├── 📓 decision_trees_svm.ipynb
│   ├── 📓 diabetes_prediction.ipynb
│   ├── 📓 credit_card_fraud_detection.ipynb
│   └── 📓 email_spam.ipynb
│
├── 📂 Ensemble-Learning/
│   ├── 📓 random_forest_boosting.ipynb
│   ├── 📓 xgboost_lightgbm.ipynb
│   └── 📓 majority_voting_stacking.ipynb
│
├── 📂 Deep-Learning-ANN/
│   ├── 📓 backpropagation_scratch.ipynb
│   ├── 📓 ANN_MNIST.ipynb
│   └── 📓 optimisers_comparison.ipynb
│
├── 📂 Deep-Learning-CNN/
│   ├── 📓 CNN_MNIST.ipynb
│   ├── 📓 CIFAR10_CNN.ipynb
│   ├── 📓 resnet_transfer_learning.ipynb
│   └── 📓 image_processing.ipynb
│
├── 📂 Deep-Learning-RNN-LSTM/
│   ├── 📓 vanilla_rnn.ipynb
│   ├── 📓 LSTM_sequence.ipynb
│   └── 📓 gru_text_classification.ipynb
│
├── 📂 Transformers/
│   ├── 📓 attention_from_scratch.ipynb
│   ├── 📓 transformer_encoder.ipynb
│   ├── 📓 bert_fine_tuning.ipynb
│   ├── 📓 gpt_text_generation.ipynb
│   └── 📓 vit_image_classification.ipynb
│
├── 📂 LangChain-LLM-Engineering/
│   ├── 📓 langchain_basics.ipynb
│   ├── 📓 rag_pipeline.ipynb
│   ├── 📓 agents_tools.ipynb
│   ├── 📓 langgraph_stateful_agent.ipynb
│   └── 📓 production_chatbot.ipynb
│
├── 📂 Time-Series/
│   ├── 📓 arima_sarimax.ipynb
│   └── 📓 temperature_forecasting.ipynb
│
├── 📂 Clustering-Unsupervised/
│   ├── 📓 kmeans_dbscan.ipynb
│   └── 📓 pca_tsne_umap.ipynb
│
├── 📂 Preprocessing-Balancing/
│   ├── 📓 oversampling_undersampling.ipynb
│   ├── 📓 feature_engineering.ipynb
│   └── 📓 correlation_analysis.ipynb
│
├── 📂 Computer-Vision/
│   └── 🐍 body_detection.py
│
├── 📂 NLP-Recommendation/
│   └── 📓 movie_recommendation.ipynb
│
├── 📂 Probabilistic-ML/
│   └── 📓 gaussian_processes_hmm.ipynb
│
└── 📄 README.md
```

---

## 📚 References

<details>
<summary><b>📖 Foundational Textbooks</b></summary>

- Bishop, C. M. — *Pattern Recognition and Machine Learning* (2006)
- Goodfellow, Bengio, Courville — *Deep Learning* ([deeplearningbook.org](https://deeplearningbook.org))
- Hastie, Tibshirani, Friedman — *Elements of Statistical Learning* ([free PDF](https://hastie.su.domains/ElemStatLearn/))
- Murphy, K. P. — *Probabilistic Machine Learning* (2022, [free PDF](https://probml.github.io/pml-book/))
- Géron, A. — *Hands-On ML with Scikit-Learn, Keras & TensorFlow* (3rd ed.)
- Nielsen, M. — *Neural Networks and Deep Learning* ([neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com))
- Szeliski, R. — *Computer Vision: Algorithms and Applications* (2nd ed.)

</details>

<details>
<summary><b>📜 Landmark Papers</b></summary>

- Vaswani et al. (2017) — *Attention Is All You Need* — Transformer architecture
- Devlin et al. (2018) — *BERT* — Bidirectional Encoder Representations
- Radford et al. (2018, 2019, 2020) — *GPT-1, GPT-2, GPT-3*
- He et al. (2015) — *Deep Residual Learning (ResNet)*
- Dosovitskiy et al. (2020) — *An Image is Worth 16×16 Words (ViT)*
- Brown et al. (2020) — *Language Models are Few-Shot Learners (GPT-3)*
- Wei et al. (2022) — *Emergent Abilities of Large Language Models*
- Hoffmann et al. (2022) — *Training Compute-Optimal LLMs (Chinchilla)*
- Hu et al. (2021) — *LoRA: Low-Rank Adaptation*
- Ouyang et al. (2022) — *InstructGPT / RLHF*
- Rafailov et al. (2023) — *Direct Preference Optimisation (DPO)*
- Dao et al. (2022) — *FlashAttention*
- Chen et al. (2016) — *XGBoost*
- Ke et al. (2017) — *LightGBM*
- Ho et al. (2020) — *Denoising Diffusion Probabilistic Models (DDPM)*
- Bergstra & Bengio (2012) — *Random Search for Hyper-Parameter Optimisation*
- Guo et al. (2017) — *On Calibration of Modern Neural Networks*
- Santurkar et al. (2018) — *How Does Batch Normalization Help Optimization?*
- Krogh & Vedelsby (1995) — *Neural Network Ensembles, Cross Validation, and Active Learning*
- Domingos & Pazzani (1997) — *On the Optimality of the Simple Bayesian Classifier*

</details>

<details>
<summary><b>🔗 Online Resources</b></summary>

- [Papers With Code](https://paperswithcode.com) — SOTA benchmarks with code
- [Arxiv ML](https://arxiv.org/list/cs.LG/recent) — Latest ML papers
- [Hugging Face](https://huggingface.co) — Models, datasets, demos
- [LangChain Docs](https://docs.langchain.com)
- [LangSmith](https://smith.langchain.com) — LLM observability
- [Scikit-learn](https://scikit-learn.org) | [PyTorch](https://pytorch.org) | [TensorFlow](https://tensorflow.org)
- [Distill.pub](https://distill.pub) — Visual ML explanations
- [3Blue1Brown Neural Networks](https://www.3blue1brown.com/topics/neural-networks)
- [Andrej Karpathy's Blog](https://karpathy.github.io)
- [Sebastian Ruder's NLP Progress](https://nlpprogress.com)
- [Jay Alammar's Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

</details>

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-FFD21E?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-E87722?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-0A9B5F?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-0064A5?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF69B4?style=for-the-badge)

</div>

---

<div align="center">

## ⭐ Star this repo if it helped you!

```
Made with ❤️, mathematics, and too much caffeine
by @ravidaliparthy
```

*Theory → Math → Code → Production*

[![GitHub followers](https://img.shields.io/github/followers/ravidaliparthy?style=social)](https://github.com/ravidaliparthy)
[![Twitter Follow](https://img.shields.io/twitter/follow/ravidaliparthy?style=social)](https://twitter.com/ravidaliparthy)

</div>
