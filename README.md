# 🤖 Machine Learning, Deep Learning & LLM Engineering
### The Most Complete Theory-First Reference — Beginner to Production

> **Every algorithm. Every derivation. Every formula. Every intuition.**
> Theory → Math → Code → Production.

---

## 📑 Table of Contents

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
15. [Transformers — Complete Deep Dive](#15-transformers--complete-deep-dive)
16. [LangChain & LLM Application Engineering](#16-langchain--llm-application-engineering)
17. [Handling Imbalanced Data](#17-handling-imbalanced-data)
18. [Model Evaluation & Selection](#18-model-evaluation--selection)
19. [Regularisation & Optimisation](#19-regularisation--optimisation)
20. [Probabilistic Machine Learning](#20-probabilistic-machine-learning)
21. [Repository Structure](#21-repository-structure)

---

## 1. Mathematical Foundations

### 1.1 Linear Algebra

#### Vectors and Norms

A **norm** is a function that assigns a non-negative length or size to a vector. Norms are not just measurement tools — they encode geometric assumptions about the space where learning happens. When you choose a loss function, a regulariser, or a distance metric, you are implicitly choosing a norm, and that choice shapes what the algorithm considers "close", "large", or "important".

```
L0 norm:  ||x||₀ = count(xᵢ ≠ 0)             — number of non-zero elements
L1 norm:  ||x||₁ = Σᵢ |xᵢ|                   — Manhattan distance
L2 norm:  ||x||₂ = √(Σᵢ xᵢ²)                 — Euclidean distance
L∞ norm:  ||x||∞ = maxᵢ |xᵢ|                  — Chebyshev distance
Lp norm:  ||x||ₚ = (Σᵢ |xᵢ|ᵖ)^(1/p)
```

**Geometric Intuition:** The unit ball `{x : ||x||ₚ ≤ 1}` changes shape with p:
- `p=1` → diamond shape (corners at coordinate axes → sparsity in optimisation)
- `p=2` → sphere (smooth → shrinkage without sparsity)
- `p=∞` → hypercube

**Why Norms Drive Sparsity or Shrinkage:**
The L1 norm has sharp corners exactly on the coordinate axes. When you minimise a loss subject to an L1 constraint, the optimal solution tends to land on one of these corners, making some coefficients exactly zero (sparsity). The L2 norm is a smooth ellipsoid with no corners — it shrinks all weights proportionally but never drives any exactly to zero. This geometric difference is the entire reason Lasso produces sparse models and Ridge does not.

**Triangle Inequality (foundation of all distance reasoning):**
```
||x + y||ₚ ≤ ||x||ₚ + ||y||ₚ   (holds for all p ≥ 1)
||x - z||ₚ ≤ ||x - y||ₚ + ||y - z||ₚ  (metric property)
```

**Dual Norms:**
```
The dual norm of ||·||ₚ is ||·||_q  where 1/p + 1/q = 1
Dual of L1 = L∞
Dual of L2 = L2  (self-dual)

Hölder's inequality:  |xᵀy| ≤ ||x||ₚ · ||y||_q
Cauchy-Schwarz (p=q=2):  |xᵀy| ≤ ||x||₂ · ||y||₂
```

The dual norm appears in subgradient analysis. The subdifferential of `||x||₁` at a non-zero point is `sign(x)`; at zero it is the entire dual ball `{v : ||v||∞ ≤ 1}`. This is why coordinate descent on L1-penalised objectives leads to soft-thresholding operators — a direct algebraic consequence of the geometry.

---

#### Matrix Operations

```
Transpose:          (AB)ᵀ = BᵀAᵀ
Inverse:            AA⁻¹ = I          (square, non-singular only)
Pseudo-inverse:     A⁺ = VΣ⁺Uᵀ       (Moore-Penrose; works for any shape)
Trace:              tr(A) = Σᵢ Aᵢᵢ = Σᵢ λᵢ
Determinant:        det(A) = Πᵢ λᵢ    (product of all eigenvalues)
Rank:               rank(A) = number of non-zero singular values
Frobenius norm:     ||A||_F = √(Σᵢ Σⱼ Aᵢⱼ²) = √tr(AᵀA) = √(Σᵢ σᵢ²)
Nuclear norm:       ||A||_* = Σᵢ σᵢ   (convex relaxation of rank)
```

**Matrix Calculus Identities (essential for deriving ML algorithms):**
```
tr(AB) = tr(BA)                        (cyclic property — very useful)
∂/∂A tr(AB)      = Bᵀ
∂/∂A tr(AᵀA)     = 2A
∂/∂x (xᵀAx)      = (A + Aᵀ)x = 2Ax   (when A symmetric)
∂/∂x (aᵀx)       = a
∂/∂A det(A)       = det(A) A⁻ᵀ
```

**Woodbury Matrix Identity (critical for efficient computation):**
```
(A + UCV)⁻¹ = A⁻¹ − A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹VA⁻¹
```

This identity is pivotal whenever a large matrix `A` is easy to invert but `A + UCV` is not. In ML it appears in kernel ridge regression (avoiding n×n inversion when n > d), Gaussian Process posterior updates, and online learning updates. The key insight: if `U` and `V` have few columns (low rank), the right-hand side requires inverting only a small matrix — massive computational savings.

**Schur Complement:**
```
For block matrix M = [[A, B],[C, D]]:
  M/A = D − CA⁻¹B   (Schur complement of A in M)
  det(M) = det(A) · det(M/A)
```

The Schur complement is the theoretical basis for Gaussian Process posterior updates. When conditioning a joint Gaussian `P(x,y)`, the conditional `P(y|x)` has covariance equal to `Σ_yy − Σ_yx Σ_xx⁻¹ Σ_xy` — the Schur complement.

---

#### Eigendecomposition

```
Av = λv            (v = eigenvector, λ = eigenvalue)
A = Q Λ Qᵀ         (symmetric A: Q = orthogonal eigenvectors, Λ = diagonal eigenvalues)
```

**Properties:**
- Symmetric matrices always have real eigenvalues and orthogonal eigenvectors (Spectral Theorem)
- Positive Definite (PD): all λᵢ > 0 — unique minimum in quadratic optimisation
- Positive Semi-Definite (PSD): all λᵢ ≥ 0 — required for valid covariance matrices and kernels
- Condition number κ = λ_max / λ_min — large κ → ill-conditioned system

**Why Ill-Conditioning Hurts Gradient Descent:**
When minimising `f(x) = (1/2)xᵀAx` with condition number κ, gradient descent with optimal step size converges at rate:
```
||xₜ − x*||² ≤ ((κ−1)/(κ+1))^{2t} ||x₀ − x*||²
```
When κ = 10,000, the contraction factor ≈ 0.9998 — thousands of iterations to converge. The loss landscape is a highly elongated ellipsoid — gradient steps bounce between steep walls. Preconditioning (dividing each coordinate by its eigenvalue) transforms the landscape to a sphere. This is exactly what Adam does adaptively: normalising gradients by their approximate second moment achieves implicit diagonal preconditioning.

**Spectral Theorem:**
Every real symmetric matrix `A ∈ ℝⁿˣⁿ` decomposes as `A = QΛQᵀ = Σᵢ λᵢ qᵢqᵢᵀ` — a weighted sum of rank-1 projection matrices. PCA is the direct application: eigenvectors are directions of maximum variance, eigenvalues quantify how much variance lies in each direction.

**Matrix Functions via Eigendecomposition:**
```
Aᵏ      = Q Λᵏ Qᵀ
exp(A)  = Q diag(exp(λᵢ)) Qᵀ
A^{1/2} = Q diag(√λᵢ) Qᵀ     (only if A ≻ 0)
log(A)  = Q diag(log λᵢ) Qᵀ
```

Matrix square roots appear in whitening transforms (PCA pre-processing); matrix exponentials appear in continuous-time dynamical systems (neural ODEs).

---

#### Singular Value Decomposition (SVD)

```
A = U Σ Vᵀ
```

- `U` (m×m): left singular vectors — eigenvectors of `AAᵀ`
- `V` (n×n): right singular vectors — eigenvectors of `AᵀA`
- `Σ` (m×n): diagonal with singular values σᵢ = √λᵢ(AᵀA) ≥ 0

**Truncated (rank-k) SVD:**
```
Aₖ = Uₖ Σₖ Vₖᵀ       (best rank-k approximation — Eckart-Young theorem)
||A − Aₖ||_F² = Σᵢ>k σᵢ²
```

**Eckart-Young Theorem:** Among all rank-k matrices B, the truncated SVD minimises `||A − B||_F`. The proof: any rank-k matrix B can be written as a sum of k rank-1 terms, and the Frobenius norm decomposes as `||A − B||_F² = Σᵢ(σᵢ − σ̃ᵢ)²` when B shares singular vectors with A. Setting `σ̃ᵢ = σᵢ` for i ≤ k and 0 otherwise is optimal. This underpins PCA (best linear dimensionality reduction), collaborative filtering, and data compression.

**SVD and the Four Fundamental Subspaces:**
```
Column space of A   = span(U₁,...,Uᵣ)       (r = rank(A))
Left null space     = span(Uᵣ₊₁,...,Uₘ)
Row space of A      = span(V₁,...,Vᵣ)
Right null space    = span(Vᵣ₊₁,...,Vₙ)
```

Understanding these subspaces explains why linear systems may have infinitely many solutions (rank-deficient X), and why the pseudo-inverse gives the minimum-norm solution.

**Applications of SVD in ML:** PCA, LSA (NLP), collaborative filtering, pseudo-inverse, matrix completion, latent semantic analysis.

---

#### Covariance and Correlation

```
Covariance matrix:   Σ = (1/(n−1)) Xᵀ X     (X is mean-centred, n×d)
Σᵢⱼ = Cov(Xᵢ, Xⱼ)  = E[(Xᵢ − μᵢ)(Xⱼ − μⱼ)]

Correlation matrix:  R = D^{−1/2} Σ D^{−1/2}   (D = diag(Σ))
Rᵢⱼ = Σᵢⱼ / (σᵢ σⱼ)   ∈ [−1, 1]
```

**Geometry of Covariance:** The covariance matrix defines an ellipsoid `{x : xᵀΣ⁻¹x ≤ 1}` — the Mahalanobis unit ball. Eigenvectors of Σ are the axes of this ellipsoid; axis lengths are `√λᵢ`. When Σ = σ²I (sphere), all directions equally variable. When Σ has one dominant eigenvalue, the data is mostly spread along the first principal component.

**Sample vs Population Covariance:**
```
Population: (1/n) Σᵢ (xᵢ−μ)(xᵢ−μ)ᵀ   (biased)
Sample:     (1/(n−1)) Σᵢ (xᵢ−x̄)(xᵢ−x̄)ᵀ  (Bessel's correction — unbiased)
```
The n−1 denominator corrects for the fact that we estimated the mean from the same data, introducing a downward bias in raw variance estimation.

---

### 1.2 Calculus & Optimisation

#### Gradient, Jacobian, Hessian

```
Gradient (scalar → vector):
  ∇f(x) = [∂f/∂x₁, ..., ∂f/∂xₙ]ᵀ   ∈ ℝⁿ

Jacobian (vector → vector):
  J(f)ᵢⱼ = ∂fᵢ/∂xⱼ                   ∈ ℝᵐˣⁿ

Hessian (scalar → matrix of 2nd derivatives):
  H(f)ᵢⱼ = ∂²f / (∂xᵢ ∂xⱼ)            ∈ ℝⁿˣⁿ
```

**Hessian Definiteness → Identifies Critical Points:**
```
H ≻ 0  (positive definite)  → local minimum
H ≺ 0  (negative definite)  → local maximum
H indefinite                 → saddle point
H ⪰ 0  (PSD)               → need higher-order test
```

**Convexity — Why It Matters Deeply:**
A function f is convex if for all x, y and λ ∈ [0,1]:
```
f(λx + (1−λ)y) ≤ λf(x) + (1−λ)f(y)
```

Convexity is the central structural property that makes optimisation tractable:
- Every local minimum is a global minimum
- Gradient descent converges to the global optimum
- Tight bounds via duality theory

**First-Order Convexity Condition:**
```
f is convex ⟺ f(y) ≥ f(x) + ∇f(x)ᵀ(y−x)  ∀x,y
```
The tangent hyperplane at any point is a global underestimate. This gives gradient descent a clean interpretation: we move in the direction of steepest descent of the tangent, and convexity guarantees this never leads away from the global minimum.

**Second-Order Convexity Condition:**
```
f is convex ⟺ H(x) ⪰ 0  ∀x
```

---

#### Chain Rule

```
Scalar:   dz/dx = (dz/dy)(dy/dx)

Vector:   ∂z/∂x = Jᵀ_{y→x} · (∂z/∂y)     (Jacobian-vector product)

General (backprop):
  If z = f(g(x)):  ∂z/∂xᵢ = Σⱼ (∂z/∂yⱼ)(∂yⱼ/∂xᵢ)
```

**Computational Graph Perspective:**
The chain rule applied to a computational graph is precisely what backpropagation computes. Every operation is a node; edges carry values (forward pass) and gradients (backward pass). The key insight of reverse-mode automatic differentiation: computing `∂L/∂θ` for ALL parameters simultaneously costs only O(1) times the cost of a forward pass, regardless of parameter count. This is why training networks with billions of parameters is computationally feasible.

**Forward-Mode vs Reverse-Mode AD:**
```
Forward-mode: propagate JVPs from inputs → output
  Efficient when: n_outputs >> n_inputs (e.g., sensitivity analysis)

Reverse-mode: propagate VJPs from output → inputs
  Efficient when: n_inputs >> n_outputs (e.g., ∂L/∂θ for all θ in one pass)
  This is what backpropagation does
```

---

#### Taylor Series & Optimisation Proofs

```
f(x + δ) ≈ f(x) + ∇f(x)ᵀδ + (1/2)δᵀH(x)δ + O(||δ||³)
```

**Newton's Method (2nd-order):**
```
δ* = −H⁻¹ ∇f(x)
x_{t+1} = xₜ − H⁻¹ ∇f(xₜ)
```
Quadratic convergence but O(n³) per step. Used in IRLS for logistic regression.

**Gradient Descent Convergence (L-smooth f):**
```
If ||∇²f|| ≤ L (Lipschitz gradient), then GD with α = 1/L:
  f(xₜ) − f* ≤ (L||x₀ − x*||²) / (2t)    O(1/t) sublinear
```

**Derivation Sketch:** From L-smoothness: `f(y) ≤ f(x) + ∇f(x)ᵀ(y−x) + (L/2)||y−x||²`. Setting `y = x − (1/L)∇f(x)` gives descent of at least `(1/2L)||∇f(x)||²` per step. Summing over t steps yields the O(1/t) rate.

**Strong Convexity (μ-strongly convex):**
```
f(y) ≥ f(x) + ∇f(x)ᵀ(y−x) + (μ/2)||y−x||²
Linear convergence: ||xₜ − x*||² ≤ (1 − μ/L)ᵗ ||x₀ − x*||²
Condition number κ = L/μ — smaller → faster convergence
```

**Conjugate Gradient Method:**
```
For Ax = b — minimises (1/2)xᵀAx − bᵀx:
  Generates A-conjugate directions: pᵢᵀApⱼ = 0 for i≠j
  Exact convergence in n steps (theoretically)
  Convergence rate: O((√κ−1)/(√κ+1))^t — faster than GD's O((κ−1)/(κ+1))^t
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
In the Gaussian case with prior `θ ~ N(μ₀, τ²)` and n i.i.d. observations `x|θ ~ N(θ, σ²)`:
```
Posterior: θ|x ~ N(μₙ, τₙ²)
  μₙ  = (τ⁻² μ₀ + nσ⁻² x̄) / (τ⁻² + nσ⁻²)   (precision-weighted average)
  τₙ⁻² = τ⁻² + nσ⁻²                           (precisions add)
```

The posterior mean is a convex combination of prior mean and data mean, weighted by their precisions. As n → ∞, the posterior collapses onto x̄ (data overwhelm the prior). As n → 0, it collapses onto μ₀. This is the Bayesian analogue of regularisation: the prior pulls the estimate toward μ₀, exactly like L2 regularisation.

**Sequential Bayesian Updating:**
Bayes' theorem is inherently sequential. The posterior after n observations becomes the prior for observation n+1. This Markov property means full Bayesian inference is incremental — a fundamental advantage in online learning.

---

#### Expectation, Variance, Covariance

```
E[X] = Σₓ x P(X=x)           (discrete)
E[X] = ∫ x f(x) dx            (continuous)
E[aX + bY] = aE[X] + bE[Y]   (linearity — always holds, no independence needed)

Var(X) = E[(X−μ)²] = E[X²] − (E[X])²
Var(aX + b) = a² Var(X)
Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)

Cov(X,Y) = E[(X−μₓ)(Y−μᵧ)] = E[XY] − E[X]E[Y]
Cov(X,Y) = 0 ⟹ uncorrelated  (independent ⟹ uncorrelated, NOT vice versa)
```

**Law of Total Expectation and Total Variance:**
```
Law of Total Expectation: E[X] = E[E[X|Y]]
Law of Total Variance:    Var(X) = E[Var(X|Y)] + Var(E[X|Y])
                                  ← within-group  +  between-group variance
```

The law of total variance underpins the ANOVA decomposition and explains why ensembles reduce variance: the "between-group" term is what a model of Y explains about X; the "within-group" term is irreducible noise.

---

#### Key Distributions

| Distribution | PMF/PDF | Mean | Variance | Use in ML |
|---|---|---|---|---|
| Bernoulli(p) | pˣ(1−p)^(1−x) | p | p(1−p) | Binary classification output |
| Binomial(n,p) | C(n,x)pˣ(1−p)^(n−x) | np | np(1−p) | Count data |
| Gaussian(μ,σ²) | (1/√(2πσ²))exp(−(x−μ)²/2σ²) | μ | σ² | Continuous output, weight prior |
| Categorical(π) | Πₖ πₖ^xₖ | πₖ | πₖ(1−πₖ) | Multi-class output |
| Dirichlet(α) | ∝ Πₖ xₖ^(αₖ−1) | αₖ/α₀ | — | Prior over Categorical |
| Laplace(μ,b) | (1/2b)exp(−|x−μ|/b) | μ | 2b² | L1 prior, robust regression |
| Beta(α,β) | ∝ x^{α−1}(1−x)^{β−1} | α/(α+β) | — | Prior over Bernoulli/Binomial |
| Exponential(λ) | λe^{−λx} | 1/λ | 1/λ² | Time between events |
| Poisson(λ) | λˣe^{−λ}/x! | λ | λ | Count of rare events |

**Why the Gaussian is Central:**
The Gaussian arises from the Central Limit Theorem: the sum of n i.i.d. random variables with finite mean μ and variance σ² converges to N(nμ, nσ²) regardless of original distribution. This justifies modelling residuals as Gaussian (they are sums of many small errors) and using squared loss as the canonical regression objective (MLE under Gaussian noise).

**Maximum Entropy Property:** Among all distributions with fixed mean μ and variance σ², the Gaussian maximises differential entropy `H(X) = −∫ p(x) log p(x) dx`. It is the "least informative" distribution given only first two moments — a principled default when no additional structural knowledge is available.

---

#### Exponential Family (Unified Framework)

```
p(x|η) = h(x) exp(ηᵀT(x) − A(η))
```

Where:
- `η`: natural (canonical) parameters
- `T(x)`: sufficient statistics
- `A(η)`: log-partition function (normaliser)
- `h(x)`: base measure

**Key Cumulant Property:**
```
∂A/∂η  = E[T(x)]        (first cumulant = mean)
∂²A/∂η² = Var[T(x)]     (second cumulant = variance)
```

**Why the Exponential Family Matters:**

1. **Sufficient Statistics:** By the Fisher-Neyman factorisation theorem, T(x) captures all information about η that the data contains. For a Gaussian, knowing sample mean and variance is equivalent to knowing all the raw data for the purpose of estimating parameters.

2. **MLE has Closed Form:** The MLE satisfies `∂A/∂η = (1/n)Σᵢ T(xᵢ)` — setting mean sufficient statistics equal to empirical mean. For the Gaussian, this gives `μ̂ = x̄` directly.

3. **Conjugate Priors Exist:** For every exponential family likelihood, there is a conjugate prior in the same family, leading to tractable posterior updates.

4. **GLMs are Built on This:** Generalised Linear Models assume the response belongs to an exponential family with a link function connecting the linear predictor to the natural parameter.

---

#### Information Theory

```
Entropy (uncertainty in X):
  H(X) = −Σₓ P(x) log₂ P(x)    ≥ 0

Conditional Entropy:
  H(Y|X) = H(X,Y) − H(X)

Mutual Information (shared information):
  I(X;Y) = H(X) − H(X|Y) = H(Y) − H(Y|X) = H(X) + H(Y) − H(X,Y)   ≥ 0

KL Divergence (how different Q is from P):
  D_KL(P||Q) = Σₓ P(x) log(P(x)/Q(x))  ≥ 0   (= 0 iff P = Q, non-symmetric)
  Forward KL  (P||Q): "mean-seeking" — Q covers all of P's mass
  Reverse KL  (Q||P): "mode-seeking" — Q concentrates on P's mode

Cross-Entropy:
  H(P,Q) = H(P) + D_KL(P||Q) = −Σₓ P(x) log Q(x)

Jensen-Shannon Divergence (symmetric, bounded [0,1]):
  JSD(P||Q) = (1/2)D_KL(P||M) + (1/2)D_KL(Q||M),   M = (P+Q)/2
  √JSD is a proper metric
```

**Why Cross-Entropy is the Natural Classification Loss:**
Minimising cross-entropy `H(P,Q)` is equivalent to minimising `D_KL(P||Q)` (since H(P) is fixed). This is also identical to maximising log-likelihood — MLE, KL minimisation, and cross-entropy minimisation are the same objective in three different framings. This unification connects information theory, statistics, and deep learning through a single principle.

**Forward vs Reverse KL — Profound Implications:**
```
Forward KL (P||Q): Q must cover all mass of P → mean-seeking → "blurry" outputs
Reverse KL (Q||P): Q avoids zero-mass regions of P → mode-seeking → sharp, collapsed outputs
```
A neural network classifier trained with cross-entropy (forward KL) will spread probability across all modes of a multimodal distribution. A variational autoencoder decoder trained with reverse KL will collapse to one mode. Neither is universally better — the choice depends on whether coverage or sharpness is more important.

**Data Processing Inequality:**
```
If X → Y → Z is a Markov chain:  I(X;Z) ≤ I(X;Y)
```
Processing data can only reduce mutual information — you cannot create information by transforming data. This is the theoretical basis for the Information Bottleneck principle: a good representation Z of input X for predicting Y should maximise I(Z;Y) while minimising I(X;Z).

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
R(h) − R̂(h) ≤ ε   with probability ≥ 1 − δ
```

The gap depends on: sample size n, hypothesis class complexity, confidence δ.

**The Fundamental Tension in Learning:**
There are two ways a learning algorithm can fail, pulling in opposite directions:
- **Approximation error (bias):** H may not contain a function close to f. Making H larger reduces approximation error.
- **Estimation error (variance):** Finite data means we cannot perfectly identify f within H. A larger H allows ERM to overfit noise. Making H smaller reduces estimation error.

**Rademacher Complexity (modern generalisation bound):**
```
R̂ₙ(H) = E_σ[ sup_{h∈H} (1/n) Σᵢ σᵢ h(xᵢ) ]
  σᵢ ∈ {−1,+1} i.i.d. uniform (Rademacher random variables)

Generalisation bound:  R(ĥ) ≤ R̂(ĥ) + 2Rₙ(H) + √(log(1/δ)/(2n))
```

Intuitively, R̂ₙ(H) measures how well H can fit random ±1 labels. If H can perfectly fit any labelling (high complexity), R̂ₙ → 1. This bound is tighter and more widely applicable than VC theory for modern neural networks.

**Fundamental ML Assumption:** Training distribution ≈ test distribution. When violated → distribution shift (covariate shift, label shift, concept drift).

---

### 2.2 Bias-Variance Trade-off

For squared loss, the expected error of any estimator ĥ decomposes as:
```
E[(y − ĥ(x))²] = Bias²(ĥ(x)) + Var(ĥ(x)) + σ²_noise
```

Where:
```
Bias(ĥ(x))  = E[ĥ(x)] − f(x)           (systematic error; model too simple)
Var(ĥ(x))   = E[(ĥ(x) − E[ĥ(x)])²]     (sensitivity to training data)
σ²_noise     = Var(y|x)                  (irreducible noise)
```

**Full Derivation:**
```
MSE = E[(y − ĥ)²]
    = E[((y−f) + (f−Eĥ) + (Eĥ−ĥ))²]
    = E[(y−f)²] + (f−Eĥ)² + E[(Eĥ−ĥ)²]   (cross-terms = 0)
    = σ²_noise   +   Bias²   +   Variance
```

**Why Cross-Terms Vanish:**
- `E[(y−f)(f−Eĥ)]` = 0 because `(f−Eĥ)` is a constant and `E[(y−f)] = 0` by definition
- `E[(y−f)(Eĥ−ĥ)]` = 0 because noise `(y−f)` is independent of training set D

**Implications for Model Selection:**
- **High Bias:** Model too simple (linear model for non-linear data). Both train and test error are high. → Increase model complexity, add features.
- **High Variance:** Model too complex (deep tree). Low train error, high test error. → Reduce complexity, add regularisation, get more data.
- **Optimal complexity:** Minimises total expected error — found via cross-validation.

**Bias-Variance for KNN:**
```
Bias²(x) ≈ [f(x_NN) − f(x)]² ∝ h²   (smoother f → lower bias for given k)
Variance(x) ≈ σ²/k
```
As k → 1: zero bias, maximum variance. As k → n: maximum bias, minimum variance. Optimal k balances these, typically scaling as k ~ n^{4/(d+4)} for d-dimensional smooth functions.

**Double Descent (modern deep learning):**
Beyond the interpolation threshold (zero training error), error can *decrease again* as model size grows — classical bias-variance does not fully explain deep learning. Overparameterised models that perfectly fit training data can still generalise well via implicit regularisation: SGD's preference for flat minima, the geometry of the loss landscape, and early stopping all contribute. The double descent curve has three regimes: underfitting → overfitting peak at interpolation → second descent in the "benign overfit" regime.

---

### 2.3 The No Free Lunch Theorem

**Statement:** For any two algorithms A₁ and A₂, averaged over all possible data-generating distributions:
```
Σ_f E[L(f, A₁, n)] = Σ_f E[L(f, A₂, n)]
```

**Implication:** No learning algorithm is universally superior. Every algorithm has inductive biases — assumptions baked in about the hypothesis class. Good algorithm selection requires domain knowledge.

**What This Means in Practice:**
The NFL theorem is often misread as "all algorithms are equal." It says: no algorithm is universally superior on ALL possible tasks. In practice, tasks we care about are not random — they have structure (images have spatial correlations; language has syntactic structure). Algorithms that exploit the right structure win. CNNs win on images because of translation invariance. Transformers win on sequences because of pairwise token relationships. The "free lunch" we pay for is the inductive bias we build in.

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

**Why Linear Classifiers in ℝᵈ Have VCdim = d+1:**
Any d+1 points in "general position" can be shattered by a hyperplane. For d+2 points, by Radon's theorem, any set of d+2 points can be partitioned into two subsets whose convex hulls intersect — this partition cannot be separated by a hyperplane.

**Sauer-Shelah Lemma (growth function):**
```
mH(n) ≤ (en/d)^d  for n > d    (polynomial, not exponential)
```

**Significance:** If H can only produce polynomially many distinct labellings, each is well-estimated with enough samples. This polynomial growth permits generalisation — the bridge from finite-sample ERM to true risk.

**PAC Learning Bound:**
```
With probability ≥ 1−δ:

  R(ĥ) − R̂(ĥ) ≤ √(2 VCdim(H) log(en/VCdim(H)) / n) + √(log(1/δ)/(2n))

Simplified: n ≥ (1/ε²)[VCdim(H) log(1/ε) + log(1/δ)]
```

---

### 2.5 Maximum Likelihood Estimation (MLE)

Given observations `{x₁,...,xₙ}` i.i.d. from `p(x|θ)`:

```
θ_MLE = argmax_θ  L(θ) = Πᵢ p(xᵢ|θ)
       = argmax_θ  ℓ(θ) = Σᵢ log p(xᵢ|θ)
```

Log-likelihood is preferred: converts products to sums (numerically stable), same argmax.

**MLE as KL Minimisation:**
```
θ_MLE = argmin_θ D_KL(p_data || p_θ)
       = argmin_θ H(p_data, p_θ)
```

MLE finds the model distribution `p_θ` closest in KL divergence to the true data distribution. This unification connects information theory, statistics, and machine learning through a single framework.

**Score Function and Fisher Information:**
```
Score:    s(θ) = ∇_θ ℓ(θ) = Σᵢ ∇_θ log p(xᵢ|θ) = 0   (at MLE)

Fisher Information: I(θ) = E[−∇²_θ log p(x|θ)] = E[s(θ)s(θ)ᵀ]
Cramér-Rao Bound: Var(θ̂) ≥ I(θ)⁻¹   (MLE achieves this asymptotically)
```

**Natural Gradient:**
Standard gradient descent uses the Euclidean metric in parameter space. The natural gradient uses the Fisher information matrix — the intrinsic Riemannian geometry of the space of probability distributions:
```
∇̃_θ ℓ = I(θ)⁻¹ ∇_θ ℓ
```
This preconditions the gradient by the inverse Fisher information, making updates invariant to reparameterisation. Natural gradient descent converges in fewer steps for probabilistic models.

**Asymptotic Properties of MLE:**
```
√n (θ_MLE − θ*) →_d N(0, I(θ*)⁻¹)   (consistency + asymptotic normality)
```

---

### 2.6 Maximum A Posteriori (MAP) Estimation

Incorporates a prior `P(θ)`:
```
θ_MAP = argmax_θ  [Σᵢ log P(xᵢ|θ) + log P(θ)]
       = MLE objective + regularisation term
```

**Prior → Regularisation Correspondence:**
```
Gaussian prior  β ~ N(0, σ²/λ I)    ↔   L2 regularisation (Ridge)
Laplace prior   β ~ Laplace(0, b)   ↔   L1 regularisation (Lasso)
Spike-and-slab                       ↔   L0 sparsity
```

**Why L2 Regularisation Corresponds to a Gaussian Prior:**
Under Gaussian prior `β ~ N(0, τ²I)`:
```
log P(β|X,y) ∝ −1/(2σ²)||y − Xβ||² − 1/(2τ²)||β||²  + const
```
Setting `λ = σ²/τ²` gives exactly the Ridge objective. Larger τ² (weaker prior) → smaller λ (less regularisation). This gives a principled way to choose λ: it encodes prior belief about the scale of weights.

**Full Bayesian Inference** (goes beyond MAP):
```
P(θ|X) = P(X|θ)P(θ) / P(X)           (exact posterior)
P(x_new|X) = ∫ P(x_new|θ) P(θ|X) dθ  (posterior predictive)
```

MAP = mode of posterior. Full Bayes = full posterior distribution. When the posterior is multimodal or asymmetric, the mode can be very unrepresentative. Full Bayesian inference marginalises over parameter uncertainty, giving automatically calibrated predictions.

---

## 3. Data Preprocessing & Feature Engineering

### 3.1 Handling Missing Values

**Types of Missingness (Rubin, 1976):**
```
MCAR: P(missing | data) = P(missing)               — no relationship to data
MAR:  P(missing | data) = P(missing | observed)    — depends on observed values only
MNAR: P(missing | data) = P(missing | missing)     — depends on the missing value itself
```

**Practical Guidance:**
- **MCAR** → safe to drop rows (complete case analysis)
- **MAR** → imputation works; listwise deletion biases estimates
- **MNAR** → most dangerous; need a model for the missingness mechanism itself

**Theoretical Underpinning:**
Complete case analysis under MAR is biased because the missingness depends on observed covariates — the complete cases are a non-random subsample. Imputation under MAR works because we can model the conditional distribution of missing values given observed ones. Under MNAR, even sophisticated imputation is biased without a model for the missingness mechanism.

**Imputation Methods:**
```
Mean/Median/Mode:
  x_imputed = mean(x_observed)    — use median for skewed, mode for categorical

Regression Imputation:
  x_missing = β₀ + β₁x₁ + β₂x₂ + ε   — predict from other features

KNN Imputation:
  x_imputed = Σₖ wₖ xₖ / Σₖ wₖ,   wₖ = 1/d(x, xₖ)ᵖ   (inverse distance)

Multiple Imputation (MICE):
  For each variable j with missing values:
    Regress xⱼ on all others using observed cases
    Sample from the predictive distribution
  Repeat for all variables; iterate until convergence
  Run m complete datasets; pool via Rubin's rules:
    Q̄ = (1/m)Σ Qₘ,   T = W̄ + (1 + 1/m)B
    W̄ = within-imputation variance, B = between-imputation variance
```

**Why Multiple Imputation Outperforms Single Imputation:**
Single imputation treats imputed values as if they were observed — artificially deflating standard errors and distorting covariance structure. Multiple imputation correctly propagates uncertainty by sampling from the imputation model m times. Rubin's combining rules correctly account for both within-imputation variability and between-imputation variability (uncertainty about missing values). Result: valid confidence intervals and hypothesis tests.

**Missing Indicator Variables:**
```
missing_j = 1 if xⱼ is missing, 0 otherwise
Then impute xⱼ with any method
→ Allows the model to learn if missingness itself is informative
```

---

### 3.2 Feature Scaling

```
Min-Max Normalisation — scales to [0,1]:
  x' = (x − x_min) / (x_max − x_min)
  Sensitive to outliers. Use when distribution is bounded and known.

Z-Score Standardisation — mean=0, std=1:
  x' = (x − μ) / σ
  Required for: PCA, SVMs, KNN, regularised regression, neural networks.

Robust Scaling — median/IQR (resistant to outliers):
  x' = (x − median) / IQR,    IQR = Q3 − Q1

Max-Abs Scaling — preserves sign, scales to [−1,1]:
  x' = x / max(|x|)    — useful for sparse data

Log Transform — right-skewed data:
  x' = log(x + 1)       — +1 handles x=0 (log1p)

Box-Cox Transform — approximate Gaussian:
  x'(λ) = (xλ−1)/λ  if λ≠0;  log(x) if λ=0   (requires x > 0)
  λ estimated by maximising log-likelihood of normality

Yeo-Johnson Transform — extends Box-Cox to x ≤ 0:
  x'(λ) = [(x+1)λ−1]/λ           if λ≠0, x≥0
           log(x+1)                if λ=0, x≥0
          −[(−x+1)^{2−λ}−1]/(2−λ) if λ≠2, x<0
          −log(−x+1)               if λ=2, x<0

Unit Vector Normalisation — per-sample (cosine similarity):
  x' = x / ||x||₂    — each sample becomes a unit vector
```

**Why Scaling is Mandatory for Some Algorithms:**
The need for scaling is dictated by the geometry of the algorithm's objective. SVMs maximise margin `2/||w||` (Euclidean norm) — unscaled features with values in [0,1000] vs [0,1] make the first feature dominate the norm, effectively ignoring the second. PCA finds directions of maximum variance — without standardisation, features with larger numerical ranges dominate. KNN uses distance directly; unscaled features make distances meaningless. Tree-based methods (Random Forest, XGBoost) do NOT need scaling — split decisions are scale-invariant threshold comparisons.

---

### 3.3 Encoding Categorical Variables

```
Label Encoding: [Low, Med, High] → [0, 1, 2]
  ONLY for ordinal variables. Implies magnitude relationship.

One-Hot Encoding: [Red, Green, Blue] → 3 binary columns
  Drop first column to avoid multicollinearity in linear models (k−1 columns).
  Creates high-dimensional sparse space for high-cardinality categories.

Target Encoding (Mean Encoding):
  x_enc = E[y | x = category]  with smoothing:
  x_enc = (n_c × ȳ_c + α × ȳ) / (n_c + α)
  n_c = count of category c, ȳ_c = mean target for c, α = smoothing factor

Frequency/Count Encoding:
  x_encoded = count(category) / total_count

Binary Encoding:
  Label encode → convert integer to binary → split bits into columns
  e.g. category 5 → 101 → [1, 0, 1]

Hashing (Feature Hashing / Hashing Trick):
  x_encoded[h(category) mod m] += 1
  Fixed-size output; handles unseen categories; risk of collisions

Embedding Encoding (learned in neural networks):
  Map category c to dense vector e_c ∈ ℝᵈ via lookup table
  Rule of thumb: d = min(50, ceil(cardinality/2))

Leave-One-Out Encoding:
  x_enc_i = (Σⱼ≠ᵢ [xⱼ=xᵢ] yⱼ) / (count(xᵢ) − 1)
  Avoids target leakage without separate folds
```

**The Target Leakage Problem:**
If `ȳ_c` is computed using the same rows used for training, information about y is directly leaked into the feature — the model sees a feature perfectly correlated with the target during training, but this correlation partially disappears at test time (data leakage). Solution: compute statistics only on out-of-fold training data, never including the current validation fold. Smoothing also helps by pulling rare categories toward the global mean.

**Why Embeddings Outperform One-Hot for High Cardinality:**
For 50,000 unique values, one-hot encoding produces a 50,000-dimensional sparse vector — mostly zeros. Two distinct cities are orthogonal (distance √2) with no notion of similarity. Learned embeddings (say, 50-dimensional) are dense, require far fewer parameters, and can encode semantic similarity — cities in the same region end up with similar vectors because the model learns to exploit geographic patterns.

---

### 3.4 Feature Selection

**Filter Methods (model-independent, fast):**
```
Pearson Correlation:   r = Cov(x,y) / (σₓ σᵧ)      — linear relationships only
Spearman Correlation:  ρ = 1 − 6Σdᵢ² / (n(n²−1))  — monotonic, non-parametric
Mutual Information:    I(X;Y) = H(X) − H(X|Y)       — any relationship
Chi-Squared test:      χ² = Σ (O−E)²/E              — categorical vs target
ANOVA F-statistic:     F = MS_between / MS_within    — continuous vs categorical
Variance threshold:    remove features with Var < threshold
```

**Wrapper Methods:**
```
Forward Selection:    start ∅, greedily add feature maximising CV score
Backward Elimination: start full, greedily remove worst feature
RFE:                  fit model → rank by importance → remove worst → repeat
Complexity: O(p²) model fits. Expensive but captures interactions.
```

**Embedded Methods:**
```
L1 (Lasso):   drives irrelevant coefficients to exactly 0
L2 (Ridge):   shrinks all — no elimination
Elastic Net:  sparse but handles correlated groups
Tree-based:   sum of impurity reduction per feature across all splits
SHAP:         game-theory-based — consistent, locally accurate
```

**SHAP Values — Theoretical Foundation:**
Based on cooperative game theory (Shapley values):
```
φⱼ = Σ_{S⊆F\{j}} [|S|!(|F|−|S|−1)!/|F|!] × [v(S∪{j}) − v(S)]
```
where `v(S)` is the model's expected prediction knowing only features in S. The weight gives each subset the probability it would arise in a random feature-entry ordering. SHAP is the unique attribution satisfying:
1. **Efficiency:** `Σⱼ φⱼ = f(x) − E[f(X)]`
2. **Symmetry:** Interchangeable features get equal attribution
3. **Dummy:** Non-contributing features get zero attribution

---

### 3.5 Outlier Detection

```
Z-Score:          z = (x − μ)/σ;  flag |z| > 3
Modified Z-Score: Mᵢ = 0.6745(xᵢ − x̃)/MAD;  flag |M| > 3.5   (robust)
IQR (Tukey):      Lower = Q1 − 1.5×IQR;  Upper = Q3 + 1.5×IQR

Isolation Forest:
  Anomaly Score = 2^{−E[h(x)] / c(n)}
  h(x) = path length;  c(n) = 2H(n−1) − 2(n−1)/n   (normalisation)
  Anomalies are isolated with fewer splits → shorter average path

Local Outlier Factor (LOF):
  lrd_k(x) = 1 / (mean of reach_dist_k(x, neighbours))
  LOF_k(x) = mean_{o ∈ Nₖ(x)} lrd_k(o) / lrd_k(x)
  LOF >> 1 → much sparser than neighbourhood → outlier

One-Class SVM:
  min_{R,c,ξ} R² + (1/νn) Σᵢ ξᵢ  s.t. ||xᵢ−c||² ≤ R² + ξᵢ
  ν ≈ upper bound on outlier fraction

Autoencoder-based:
  Train AE on normal data → high reconstruction error = outlier
  Threshold: ||x − AE(x)||² > τ
```

**Why Isolation Forest Works:**
The insight: anomalies are "few and different" — sparse in feature space, far from other points. A random tree recursively partitions space with random splits. For normal (dense) points, many splits are needed to isolate any single point. For anomalies (sparse regions), isolation requires very few splits. Average path length across many trees is a direct anomaly score — short path = anomalous. Linear complexity O(n), handles high dimensions well.

**LOF Theory:**
LOF compares local density of a point to the local density of its neighbours. A point in a much sparser neighbourhood than its neighbours has LOF >> 1. The "reachability distance" prevents underestimating distances to very nearby points. LOF can detect locally anomalous points that are not globally extreme — a capability flat methods like Z-score lack.

---

### 3.6 Correlation Analysis

```
Pearson:     r = Σ(xᵢ−x̄)(yᵢ−ȳ) / √[Σ(xᵢ−x̄)² Σ(yᵢ−ȳ)²]   (linear)
Spearman:    ρ = 1 − 6Σdᵢ² / (n(n²−1))                       (monotonic)
Kendall:     τ = (concordant − discordant) / C(n,2)            (ordinal)
Point-biserial: r_pb = (ȳ₁−ȳ₀)/sᵧ × √(n₁n₀/n²)             (continuous vs binary)
Cramér's V:  V = √(χ²/(n × min(r−1,c−1)))                     (categorical vs categorical)

VIF (multicollinearity):
  VIF_j = 1 / (1 − R²_j)    where R²_j = R² of regressing xⱼ on all others
  VIF > 5: moderate; VIF > 10: severe → remove feature or apply PCA
```

**Why Multicollinearity Inflates Variance:**
When features are highly correlated, `XᵀX` becomes nearly singular. The OLS variance `σ²(XᵀX)⁻¹` has very large eigenvalues — variance explodes. Geometrically: with perfectly correlated features, infinitely many hyperplanes fit equally well (flat loss landscape in that direction), so estimated coefficients are essentially random. Ridge regression adds `λI` to `XᵀX`, guaranteeing well-conditioning regardless of collinearity.

---

## 4. Regression Algorithms

### 4.1 Linear Regression

**Model:**
```
ŷ = β₀ + β₁x₁ + ... + βₙxₙ = Xβ    (matrix form; X includes intercept column)
```

**OLS — Minimise MSE:**
```
L(β) = ||y − Xβ||² = yᵀy − 2βᵀXᵀy + βᵀXᵀXβ

∂L/∂β = 2Xᵀ(Xβ − y) = 0
β* = (XᵀX)⁻¹ Xᵀy    (Normal Equation)
```

**Numerical Stability Alternatives:**
```
Via QR decomposition: X = QR → β* = R⁻¹ Qᵀy       (O(nd²), more stable)
Via SVD: X = UΣVᵀ    → β* = V Σ⁺ Uᵀy              (handles rank deficiency)
```

**Geometric Interpretation of OLS:**
OLS projects y onto the column space of X. The hat (projection) matrix `H = X(XᵀX)⁻¹Xᵀ` produces fitted values `ŷ = Hy`. Key properties:
```
H² = H   (idempotent — projecting twice is the same as projecting once)
Hᵀ = H   (symmetric)
tr(H) = p+1  (effective degrees of freedom)
Hᵢᵢ ∈ [0,1] (leverage of observation i)
```
Residuals `e = (I − H)y` are orthogonal to the column space: `Xᵀe = 0`.

**Leverage and Influence:**
```
Cook's distance: Dᵢ = eᵢ² Hᵢᵢ / (p × MSE × (1−Hᵢᵢ)²)
```
High-leverage points (Hᵢᵢ near 1) with large residuals are highly influential — they can dramatically distort the regression.

**Gauss-Markov Theorem (OLS is BLUE):**
Under assumptions:
1. `E[ε|X] = 0` (zero conditional mean)
2. `Var(ε|X) = σ²I` (homoscedasticity + no autocorrelation)
3. No perfect multicollinearity

OLS is the Best Linear Unbiased Estimator (BLUE). Gauss-Markov does NOT say OLS beats ALL estimators — biased estimators (like Ridge) can have lower MSE by trading a small bias for large variance reduction.

**Inference:**
```
Var(β*) = σ² (XᵀX)⁻¹
SE(βⱼ)  = √[σ²(XᵀX)⁻¹ⱼⱼ]
t-statistic: tⱼ = βⱼ / SE(βⱼ) ~ t(n−p−1)

R² = 1 − SS_res/SS_tot = 1 − Σ(yᵢ−ŷᵢ)² / Σ(yᵢ−ȳ)²
Adjusted R² = 1 − (1−R²)(n−1)/(n−p−1)
F-statistic = (R²/p) / ((1−R²)/(n−p−1))   — tests joint significance of all predictors
```

**Gradient Descent for Linear Regression:**
```
∇L = (2/n) Xᵀ(Xβ − y)
β := β − α∇L
```

---

### 4.2 Ridge Regression (L2)

```
L(β) = ||y − Xβ||² + λ||β||²₂
β_ridge = (XᵀX + λI)⁻¹ Xᵀy   (always invertible — key advantage)
```

**Bias-Variance of Ridge:**
```
Bias(β_ridge) = −λ(XᵀX + λI)⁻¹ β*    (non-zero — biased estimator)
Var(β_ridge)  = σ² (XᵀX+λI)⁻¹ XᵀX (XᵀX+λI)⁻¹   (< OLS variance)
```

**Effect via SVD (deep insight):** If `X = UΣVᵀ`:
```
β_ridge = V diag(σᵢ²/(σᵢ²+λ)) Uᵀy
```
Each component is shrunk by `σᵢ²/(σᵢ²+λ)`. When `σᵢ >> λ` (well-identified direction): shrinkage factor ≈ 1, kept intact. When `σᵢ << λ` (near-collinear direction): factor ≈ 0, heavily regularised. Ridge correctly identifies near-dependencies as unreliable and shrinks them.

**Effective Degrees of Freedom:**
```
df(λ) = Σᵢ σᵢ²/(σᵢ²+λ)
```
λ=0 → df=p (full OLS); λ→∞ → df→0.

**Why Ridge Improves MSE Over OLS:**
There always exists some `λ > 0` for which `MSE(β_ridge) < MSE(β_OLS)`. The reason: OLS, while unbiased, may have very high variance with collinearity. A small bias from Ridge can reduce variance far more, giving net MSE improvement. The optimal λ is found via cross-validation.

---

### 4.3 Lasso Regression (L1)

```
L(β) = ||y − Xβ||² + λΣⱼ|βⱼ|
```

No closed form — use coordinate descent.

**Coordinate Descent Update:**
```
ρⱼ = Σᵢ xᵢⱼ (yᵢ − Σₖ≠ⱼ xᵢₖ βₖ)   (partial residual)
βⱼ = S(ρⱼ, λ/2) / Σᵢ xᵢⱼ²

Soft-threshold: S(ρ, λ) = sign(ρ) max(|ρ| − λ, 0)
```

**Why the Soft-Threshold Operator Gives Exact Zeros:**
Setting the subdifferential of L to zero:
- If `|ρⱼ| ≤ λ/2`: the zero solution is stationary → `βⱼ = 0` (true sparsity)
- If `|ρⱼ| > λ/2`: `βⱼ = (ρⱼ − λ/2·sign(ρⱼ)) / Σxᵢⱼ²` (shrunk but non-zero)

Soft-thresholding combines these cases into a single operator. This is not just near-zero — it is exactly zero for coefficients where the partial correlation is below the threshold.

**Why L1 Gives Sparsity (geometric argument):**
The constraint `{β: ||β||₁ ≤ t}` is a polytope (diamond in 2D) with corners on coordinate axes. Loss contours (ellipses) typically first touch this region at a corner where one coordinate = 0 — producing sparse solutions.

**Lasso Solution Path:**
```
λ_max = max_j |Xⱼᵀy|/n  — all coefficients zero at this point
As λ decreases: features enter the model one by one
LARS algorithm computes the full path in O(np²) time
```

---

### 4.4 Elastic Net

```
L(β) = ||y−Xβ||² + λ[α||β||₁ + (1−α)||β||²]
```

**Why Lasso Fails with Correlated Groups:**
When features x₁ and x₂ are highly correlated, Lasso arbitrarily selects one and drops the other. The selection is unstable — small data perturbations can switch which one is chosen. This is problematic in genomics (correlated genes in a pathway) or NLP (synonyms).

**Grouping Effect of Elastic Net:**
For two perfectly correlated features: Lasso picks one; Elastic Net sets both to the same value. The L2 term's gradient `2λ(1−α)βⱼ` penalises any imbalance between correlated coefficients, driving them toward equality.

**Advantages:**
- Selects groups of correlated features
- Always selects ≤ n features (Lasso is limited to n when p > n)
- More numerically stable when p >> n

---

### 4.5 Polynomial & Basis Function Regression

```
Feature expansion: φ(x) = [1, x, x², ..., xᵖ]
ŷ = βᵀφ(x)   — still linear in parameters → apply OLS
```

**Universality of Basis Expansions:**
By the Weierstrass approximation theorem, any continuous function on a compact interval can be approximated arbitrarily closely by polynomials. The choice of basis encodes prior knowledge: polynomial bases are natural for smooth global behaviour but can oscillate wildly at boundaries (Runge's phenomenon). Spline bases are locally supported and avoid oscillation. Fourier bases are natural for periodic functions.

**More General Basis Functions:**
```
Radial Basis:  φⱼ(x) = exp(−||x−μⱼ||² / (2σ²))
Sigmoid:       φⱼ(x) = σ(wⱼᵀx + bⱼ)
Fourier:       φⱼ(x) = cos(ωⱼx + φⱼ)    — for periodic patterns
Cubic Splines: piecewise cubics with continuous 2nd derivatives at knots
```

**Smoothing Splines — Elegant Connection to Regularisation:**
A smoothing spline minimises:
```
Σᵢ (yᵢ − f(xᵢ))² + λ ∫ [f''(x)]² dx
```
The roughness penalty `∫[f'']²dx` penalises curvature. The solution is a natural cubic spline, and λ controls the smoothness-fit tradeoff. This connects basis function regression to non-parametric smoothing via regularisation — λ here plays the exact same role as λ in Ridge regression.

---

### 4.6 Bayesian Linear Regression

```
Prior:     β ~ N(μ₀, Σ₀)
Likelihood: y|X,β ~ N(Xβ, σ²I)

Posterior (conjugate update — closed form):
  Σₙ = (Σ₀⁻¹ + (1/σ²)XᵀX)⁻¹
  μₙ = Σₙ(Σ₀⁻¹μ₀ + (1/σ²)Xᵀy)

Predictive Distribution:
  y*|x*,X,y ~ N(μₙᵀx*, σ² + x*ᵀΣₙx*)
  ↑ prediction          ↑ aleatoric + epistemic uncertainty
```

**Two Sources of Uncertainty:**
- `σ²`: aleatoric uncertainty — irreducible noise in the process
- `x*ᵀΣₙx*`: epistemic uncertainty — our uncertainty about β, which decreases with more data

Near extrapolation regions, `x*ᵀΣₙx*` grows, correctly flagging unreliable predictions. This is the foundational difference from OLS which gives point predictions with no uncertainty beyond confidence intervals.

**Online Bayesian Updating via Woodbury:**
As new data `(xₙ₊₁, yₙ₊₁)` arrives:
```
Σₙ₊₁ = Σₙ − Σₙxₙ₊₁(σ² + xₙ₊₁ᵀΣₙxₙ₊₁)⁻¹xₙ₊₁ᵀΣₙ   (rank-1 update)
```
Each new point updates covariance with O(d²) work rather than O(d³) refit — efficient sequential learning.

---

## 5. Classification Algorithms

### 5.1 Logistic Regression

**Sigmoid & Log-Odds:**
```
σ(z) = 1/(1+e⁻ᶻ),   σ'(z) = σ(z)(1−σ(z))

Log-odds = logit(P) = log[P/(1−P)] = βᵀx
P(y=1|x) = σ(βᵀx) = 1/(1+exp(−βᵀx))
```

**Why the Logit Link?**
The logit function is the natural link for the Bernoulli distribution in the GLM framework. The Bernoulli distribution belongs to the exponential family with natural parameter `η = log(p/(1−p))` — the log-odds. The canonical link sets the linear predictor equal to the natural parameter: `βᵀx = logit(p)`. This gives logistic regression its statistical justification as an exponential family GLM.

**Binary Cross-Entropy Loss:**
```
L = −(1/n) Σᵢ [yᵢ log σ(βᵀxᵢ) + (1−yᵢ) log(1−σ(βᵀxᵢ))]
```

**Why Cross-Entropy, Not MSE?**
MSE with sigmoid creates a non-convex loss → many local minima. Cross-entropy is convex in β → guaranteed global minimum. MSE treats probability errors as Euclidean; cross-entropy respects the geometry of the probability simplex. MLE justification: minimising cross-entropy = maximising likelihood.

**Gradient — Elegant Closed Form:**
```
∂L/∂β = (1/n) Xᵀ(σ(Xβ) − y)
```

This elegant form arises from the exponential family structure: for any GLM, the gradient of the negative log-likelihood is `Xᵀ(μ − y)` where `μ = E[y|x]`. This unifies Poisson regression, linear regression, and logistic regression in a single framework.

**IRLS (Newton's Method):**
```
H = (1/n) XᵀWX,   W = diag[σ(xᵢ)(1−σ(xᵢ))]
β := β − H⁻¹∇L    (Iteratively Reweighted Least Squares)
```

Each Newton step is equivalent to a weighted least squares problem. The weights `wᵢ = σᵢ(1−σᵢ)` are larger near the decision boundary (most informative region) and smaller near the extremes. Newton's method converges in fewer iterations but each step costs O(np²).

**Multinomial Logistic (Softmax):**
```
P(y=k|x) = exp(βₖᵀx) / Σⱼ exp(βⱼᵀx)

Temperature scaling:
  P(y=k|x,T) = exp(βₖᵀx/T) / Σⱼ exp(βⱼᵀx/T)
  T→0: argmax (deterministic); T→∞: uniform distribution
```

**Softmax Non-Identifiability:**
Adding any constant vector c to all βₖ leaves probabilities unchanged (cancels in numerator and denominator). Fix: set β₁ = 0 (reference class), or use L2 regularisation which picks the minimum-norm solution.

---

### 5.2 K-Nearest Neighbours (KNN)

**Distance Metrics:**
```
Euclidean:    d_E = √Σᵢ(xᵢ−yᵢ)²
Manhattan:    d_M = Σᵢ|xᵢ−yᵢ|
Minkowski:    dₚ  = (Σᵢ|xᵢ−yᵢ|ᵖ)^{1/p}
Cosine:       d = 1 − xᵀy/(||x|| ||y||)
Mahalanobis:  d = √((x−y)ᵀΣ⁻¹(x−y))   — accounts for feature correlations
Hamming:      count of differing positions  — binary/categorical data
```

**Mahalanobis Distance — Why It Matters:**
Euclidean distance treats all features as equally important and uncorrelated. Mahalanobis distance accounts for correlation structure — it equals Euclidean distance after whitening. Without this, KNN is distorted by correlated features that effectively contribute twice to the distance.

**KNN as Non-Parametric Density Estimation:**
```
Classification: ŷ = argmax_c Σ_{xⱼ∈Nₖ(x)} 𝟙[yⱼ=c]
Regression:     ŷ = (1/k) Σ_{xⱼ∈Nₖ(x)} yⱼ
Weighted KNN:   ŷ = Σⱼ (1/dⱼ) yⱼ / Σⱼ (1/dⱼ)   (inverse distance weighting)
```

**Curse of Dimensionality:**
```
Vol(d-hypersphere) / Vol(d-hypercube) = πᵈ/² / (Γ(d/2+1) × 2ᵈ) → 0 as d → ∞
```

In high dimensions, all points become approximately equidistant. Ratio of max/min distance → 1 as d → ∞ → nearest neighbours are meaningless. To maintain the same neighbourhood density, n must grow exponentially in d. Practical remedy: dimensionality reduction (PCA, UMAP) before KNN.

**Complexity:**
```
Brute-force:   O(nd) per query
KD-Tree:       O(d log n) average, degrades for d > 20
Ball Tree:     better for high-d and non-Euclidean metrics
HNSW (approx): O(log n) — used in production vector databases
```

---

### 5.3 Naive Bayes

```
P(C|x) ∝ P(C) Πⱼ P(xⱼ|C)   (naive independence assumption)
ĉ = argmax_c [log P(C=c) + Σⱼ log P(xⱼ|C=c)]
```

**Variant Parameter Estimation:**
```
Gaussian NB:     P(xⱼ|C=c) = N(xⱼ; μⱼc, σ²ⱼc)

Multinomial NB:  P(wⱼ|C=c) = (N_cj + α)/(N_c + α|V|)   (Laplace smoothing)
  N_cj = count of word j in class c; |V| = vocabulary size

Bernoulli NB:    P(xⱼ|C=c) = pⱼc^xⱼ (1−pⱼc)^{1−xⱼ}
  Penalises for absence of terms — better for short texts

Complement NB:   uses complement class statistics — reduces bias for imbalanced classes
```

**When Does Naive Bayes Work Despite Wrong Assumptions?**
We need the decision boundary in the right place, not correct probability values. Analysis by Domingos & Pazzani (1997) shows the Naive Bayes decision boundary is correct whenever the dependency structure among features is the same across all classes — even if dependencies exist. Furthermore, Naive Bayes has only `O(d × K)` parameters — low variance, excellent for small datasets and high-dimensional inputs like text.

**Zero-Frequency Problem and Laplace Smoothing:**
Without smoothing, any unseen word makes the entire class probability zero — one unknown word overrides all other evidence. Laplace smoothing adds pseudo-count α to all counts:
```
P(wⱼ|C=c) = (count(wⱼ,c) + α) / (count(all words in c) + α×|V|)
```

**Calibration Note:** Despite often good accuracy, Naive Bayes probabilities are poorly calibrated (overconfident or underconfident). Apply Platt scaling or isotonic regression when calibrated probabilities matter.

---

### 5.4 Decision Trees

**Splitting Criteria:**
```
Entropy (Information Gain):    H(S) = −Σₖ pₖ log₂ pₖ
Gini Impurity:                 G(S) = 1 − Σₖ pₖ²
Misclassification Rate:        E(S) = 1 − max_k pₖ
```

**Relationship Between Criteria:**
Taylor expansion of `−p log p` around p=0.5 gives `−p log p ≈ p(1−p)`, so `H ≈ 2Σpₖ(1−pₖ) = 2G`. The two criteria are proportional for small impurity. Gini is computationally cheaper (no log); Entropy is slightly more sensitive to impurity. In practice, results are nearly identical — Gini is the default in most implementations.

**Information Gain:**
```
IG(S, A) = H(S) − Σᵥ (|Sᵥ|/|S|) H(Sᵥ)
```

**Gain Ratio (C4.5 — prevents bias toward high-cardinality features):**
```
GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)
SplitInfo(S, A) = −Σᵥ (|Sᵥ|/|S|) log₂(|Sᵥ|/|S|)
```
A feature like "unique customer ID" achieves perfect information gain (every leaf is pure). But this generalises terribly. Gain ratio normalises by the entropy of the split — a split into many equal-size branches has high SplitInfo, penalising the feature. This makes the criterion fair across features with different numbers of values.

**Regression Trees:**
```
Split: minimise Σᵥ (|Sᵥ|/|S|) Var(Sᵥ)
Leaf:  ŷ = mean(y in leaf)
```

**Cost-Complexity Pruning (α-pruning / CART):**
```
R_α(T) = R(T) + α|T|    (weighted misclassification rate + complexity)
```
Increase α → prune subtrees where cost reduction < α per leaf. This generates a nested sequence of trees `T₀ ⊃ T₁ ⊃ ... ⊃ T₁`. Cross-validation selects the optimal tree in this sequence.

**Bias-Variance in Decision Trees:**
A fully grown tree memorises the training set — zero training error, high variance. Pruning increases bias (coarser decision regions) while reducing variance. Cost-complexity pruning is a principled way to sweep the bias-variance curve.

---

### 5.5 Support Vector Machines (SVM)

**Hard-Margin (linearly separable data):**
```
Minimise:   (1/2)||w||²
Subject to: yᵢ(wᵀxᵢ+b) ≥ 1   ∀i

Margin: γ = 2/||w||   — maximum margin ↔ minimum ||w||²
```

**Why Maximum Margin?**
The VC dimension of maximum-margin classifiers is bounded by `O(R²/γ²)` where R is the data radius. Maximising the margin minimises this VC bound — the maximum-margin principle has a direct theoretical justification in structural risk minimisation.

**Lagrangian & Dual:**
```
Dual: max_α Σᵢαᵢ − (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼxᵢᵀxⱼ
      s.t. αᵢ ≥ 0, Σᵢαᵢyᵢ = 0

Primal-dual: w* = Σᵢ αᵢyᵢxᵢ   (only support vectors with αᵢ > 0 contribute)
```

**Why the Dual is Preferred:**
The dual depends on data only through inner products `xᵢᵀxⱼ` — enabling the kernel trick. For n << d, the dual is cheaper: O(n²) vs O(d²) variables. The dual is always a convex QP, guaranteeing global optimality.

**KKT Conditions:**
```
αᵢ[yᵢ(wᵀxᵢ+b) − 1] = 0  (complementary slackness)
```
αᵢ > 0 only for points exactly on the margin — the support vectors. All other points don't contribute to w*. This sparsity makes SVM predictions efficient at test time.

**Soft-Margin:**
```
Minimise:   (1/2)||w||² + C Σᵢ ξᵢ
Subject to: yᵢ(wᵀxᵢ+b) ≥ 1−ξᵢ,  ξᵢ ≥ 0
Dual:       max_α Σᵢαᵢ − (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼxᵢᵀxⱼ
            s.t. 0 ≤ αᵢ ≤ C, Σᵢαᵢyᵢ = 0
```

**Hinge Loss Interpretation:**
```
L_hinge(y, f(x)) = max(0, 1−yf(x))
SVM = (λ/2)||w||² + (1/n)Σᵢ max(0, 1−yᵢwᵀxᵢ)
```

**Kernel Trick:**
```
Replace xᵢᵀxⱼ → K(xᵢ, xⱼ) = φ(xᵢ)ᵀφ(xⱼ)
f(x) = sign(Σᵢ αᵢyᵢ K(xᵢ,x) + b)

Kernels:
  Linear:     K(x,z) = xᵀz
  Polynomial: K(x,z) = (γxᵀz+r)ᵈ
  RBF/Gauss:  K(x,z) = exp(−γ||x−z||²)
  Laplace:    K(x,z) = exp(−γ||x−z||₁)
  Chi-squared: K(x,z) = 1 − Σᵢ(xᵢ−zᵢ)²/(xᵢ+zᵢ)   (for histograms)

Mercer's Theorem: K is a valid kernel iff the Gram matrix K_{ij} = K(xᵢ,xⱼ) is PSD.
```

**Reproducing Kernel Hilbert Space (RKHS):**
A kernel K defines an RKHS where `φ(x) = K(x,·)` and `⟨φ(x),φ(z)⟩ = K(x,z)`. The kernel trick replaces potentially infinite-dimensional feature map computation with the kernel evaluation — O(d) work instead of O(∞). This is possible because the dual formulation depends only on inner products.

---

### 5.6 Discriminant Analysis

**LDA (Shared Covariance):**
```
Assumption: P(x|C=k) = N(μₖ, Σ)   — same Σ for all classes

Linear discriminant function:
  δₖ(x) = xᵀΣ⁻¹μₖ − (1/2)μₖᵀΣ⁻¹μₖ + log πₖ
  ĉ = argmax_k δₖ(x)

Parameter estimation:
  π̂ₖ = nₖ/n
  μ̂ₖ = (1/nₖ) Σᵢ∈Cₖ xᵢ
  Σ̂  = (1/(n−K)) Σₖ Σᵢ∈Cₖ (xᵢ−μ̂ₖ)(xᵢ−μ̂ₖ)ᵀ   (pooled within-class covariance)
```

**LDA as Dimensionality Reduction (Fisher's Criterion):**
```
Maximise: J(W) = |WᵀS_B W| / |WᵀS_W W|
S_W = Σₖ Σᵢ∈Cₖ (xᵢ−μₖ)(xᵢ−μₖ)ᵀ    (within-class scatter)
S_B = Σₖ nₖ(μₖ−μ)(μₖ−μ)ᵀ           (between-class scatter)
Solution: eigenvectors of S_W⁻¹S_B
Maximum: K−1 discriminant directions for K classes
```

**QDA (Class-Specific Covariance):**
```
Assumption: P(x|C=k) = N(μₖ, Σₖ)   — different Σₖ per class
δₖ(x) = −(1/2)log|Σₖ| − (1/2)(x−μₖ)ᵀΣₖ⁻¹(x−μₖ) + log πₖ
More flexible than LDA; needs more data to reliably estimate Σₖ.
```

**Theoretical Connection: LDA vs Logistic Regression:**
Both produce linear decision boundaries. LDA estimates parameters generatively (models P(x,y)) — optimal when Gaussian assumption holds, more efficient with small data. Logistic regression estimates discriminatively (models P(y|x) directly) — robust to non-Gaussian distributions. With large data and violated Gaussianity, prefer logistic regression.

---

## 6. Clustering & Unsupervised Learning

### 6.1 K-Means Clustering

**Objective:**
```
J = Σₖ Σᵢ∈Cₖ ||xᵢ−μₖ||²   (WCSS — Within-Cluster Sum of Squares)
```

**Lloyd's Algorithm:**
```
1. Initialise K centroids (random or K-means++)
2. Assignment: cᵢ = argmin_k ||xᵢ−μₖ||²
3. Update:     μₖ = (1/|Cₖ|) Σᵢ∈Cₖ xᵢ
4. Repeat until convergence
```

**K-means++ Initialisation:**
```
1. Pick x₁ uniformly at random
2. Compute D(x) = min_j d(x, xⱼ)² for all remaining points
3. Pick next centroid with P(x) ∝ D(x)   (far from existing centroids)
4. Repeat until K centroids chosen
Guarantee: E[WCSS] ≤ 8(ln K + 2) × WCSS_OPT
```

**Convergence Proof:** Each step (assignment + update) monotonically decreases J ≥ 0 → convergence in finite steps. However, local optima are common — run multiple restarts.

**K-means as Hard-Assignment EM on Spherical Gaussians:**
K-means is the hard-assignment limit of EM for a mixture of spherical Gaussians with equal, vanishing variances. As σ² → 0, soft GMM assignments become hard binary assignments, and EM updates reduce exactly to K-means. This connection explains why K-means assumes spherical equal-variance clusters.

**When K-means Fails:**
- Elongated clusters (use GMM with full covariance)
- Clusters of vastly different sizes
- Non-convex clusters like rings or crescents (use DBSCAN or spectral clustering)
- High-dimensional data where Euclidean distance is uninformative

**Choosing K:**
```
Elbow method:        plot WCSS vs K; choose inflection point
Silhouette:          s(i) = (b(i)−a(i))/max(a(i),b(i)); average s > 0.5 = good
Gap Statistic:       Gap(K) = E*[log WCSS_random] − log WCSS_data
Davies-Bouldin:      (1/K) Σₖ max_{k≠l} (σₖ+σₗ)/d(μₖ,μₗ)   (lower = better)
Calinski-Harabász:   tr(S_B)/(K−1) / tr(S_W)/(n−K)           (higher = better)
```

**Spectral Clustering:**
```
1. Build similarity matrix W: Wᵢⱼ = exp(−||xᵢ−xⱼ||²/(2σ²))
2. Normalised Laplacian: L_sym = I − D^{−1/2}WD^{−1/2}
3. Compute first k eigenvectors of L_sym
4. Embed each point as its row in the eigenvector matrix
5. Run K-means on the embedded points
```
Spectral clustering finds non-convex clusters because Laplacian eigenvectors encode graph connectivity, not Euclidean distance. The multiplicity of eigenvalue 0 of L equals the number of connected components.

---

### 6.2 DBSCAN

```
Core Point:   |N_ε(p)| ≥ minPts   (≥ minPts neighbours within radius ε)
Border Point: not core, but within ε of a core point
Noise Point:  neither core nor border

Algorithm:
  For each unvisited point p:
    If |N_ε(p)| < minPts: label noise
    Else: create new cluster, expand by density-reachability
  Border points assigned to nearest cluster

Complexity: O(n log n) with spatial indexing; O(n²) brute-force
```

**Parameter Selection:**
```
minPts: rule of thumb = 2×d (d = dimensions); minimum 3
ε: k-distance plot — sort k-NN distances ascending; choose at "elbow"
```

**Why DBSCAN is Robust:**
Unlike K-means, DBSCAN makes no assumptions about cluster shapes or number. Clusters are dense regions separated by sparse regions. Noise points are simply not assigned to any cluster — naturally robust to outliers. The key parameters ε and minPts define what "dense" means: setting minPts = 2×d is justified by the intrinsic dimensionality of most real datasets being lower than ambient dimensionality.

**HDBSCAN (Hierarchical DBSCAN):**
Builds a hierarchy over all density levels; extracts flat clustering via stability. More robust to varying density than standard DBSCAN.

---

### 6.3 Hierarchical Clustering

**Agglomerative (bottom-up):**
```
Start: n clusters (each point its own)
Each step: merge two closest clusters by linkage criterion
Result: dendrogram — cut at height h → flat clusters
```

**Linkage Criteria:**
```
Single (MIN):   d(A,B) = min_{a∈A,b∈B} d(a,b)    — chaining effect, elongated clusters
Complete (MAX): d(A,B) = max_{a∈A,b∈B} d(a,b)    — compact clusters, outlier sensitive
Average:        d(A,B) = mean_{a∈A,b∈B} d(a,b)   — compromise; widely used
Ward:           Δ(A,B) = (|A||B|/(|A|+|B|)) × ||μ_A − μ_B||²  — minimises variance increase
```

**Ward's Method — Theoretical Justification:**
Ward's linkage is equivalent to K-means with K decremented by 1 at each step. It produces compact, spherical clusters of similar size. Ward's method is deterministic and hierarchical, giving a complete nested structure. Best all-around linkage for most applications.

**Dendrogram Interpretation:**
The height at which two clusters merge is the dissimilarity at time of merge. The "ultrametric" property: `d(A,C) ≤ max(d(A,B), d(B,C))` — this stronger-than-triangle-inequality property is what makes hierarchical clusterings representable as trees.

**Cophenetic Correlation:** Measures how faithfully the dendrogram preserves pairwise distances. Good if > 0.75.

---

### 6.4 Gaussian Mixture Models (GMM)

```
P(x) = Σₖ πₖ N(x | μₖ, Σₖ),   Σₖ πₖ = 1
```

**EM Algorithm:**
```
E-step: γᵢₖ = πₖN(xᵢ|μₖ,Σₖ) / Σⱼ πⱼN(xᵢ|μⱼ,Σⱼ)   (soft assignments)

M-step: Nₖ = Σᵢγᵢₖ
        μₖ = (1/Nₖ) Σᵢγᵢₖxᵢ
        Σₖ = (1/Nₖ) Σᵢγᵢₖ(xᵢ−μₖ)(xᵢ−μₖ)ᵀ
        πₖ = Nₖ/n
```

**Why EM Works:**
EM maximises the ELBO — a lower bound on log-likelihood via Jensen's inequality. The E-step computes posterior assignments that make the bound tight at current θ. The M-step maximises the bound over θ, producing a new θ that is guaranteed to increase the log-likelihood. Repeating this converges to a local maximum.

**Model Selection:** BIC = −2ℓ + k log n. Choose K minimising BIC.

**Covariance Constraints:**
```
Full:      each Σₖ unconstrained — most flexible
Diagonal:  features independent within cluster
Spherical: Σₖ = σ²ₖI — equivalent to soft K-means
Tied:      all clusters share Σ — same structure as LDA
```

---

## 7. Dimensionality Reduction

### 7.1 Principal Component Analysis (PCA)

**Objective:** Find orthogonal directions (principal components) that maximise variance.

**Algorithm:**
```
1. Centre:         X ← X − μ   (subtract column means)
2. Covariance:     C = (1/n)XᵀX   ∈ ℝᵈˣᵈ
3. Eigendecompose: C = QΛQᵀ   (Q = eigenvectors, Λ = diagonal eigenvalues)
4. Sort by λ₁ ≥ λ₂ ≥ ... ≥ λd
5. Project:        Z = XQₖ   (n×k scores)
```

**Via SVD (numerically preferred):**
```
X = UΣVᵀ
Z = XVₖ = UₖΣₖ   (n×k scores)
Vₖ = first k columns of V = principal components (loadings)
```

**Variance Explained:**
```
PVE(k) = λₖ / Σᵢλᵢ
Reconstruction: X̂ = ZVₖᵀ + μ
Reconstruction error: ||X − X̂||²_F = Σⱼ>ₖ λⱼ
```

**Two Equivalent Characterisations of PCA:**
1. **Maximum Variance Projection:** The first PC `v₁` solves `max_{||v||=1} vᵀCv` — achieved by the leading eigenvector (variational characterisation of eigenvalues).
2. **Minimum Reconstruction Error:** The k-dimensional PCA projection minimises `||X − XVₖVₖᵀ||²_F = Σⱼ>ₖ λⱼ` — the Eckart-Young theorem applied to the covariance matrix.

Both give the same solution. This deep equivalence justifies PCA as both an analysis tool and a compression method.

**Whitening (Sphere Transform):**
```
Z_white = ZΛₖ^{−1/2}
— decorrelates AND normalises to unit variance
— used before ICA, some neural network training
```

**PCA Assumptions and When They Fail:**
PCA assumes linear relationships, variance = importance, and orthogonal components. These fail when data lies on a non-linear manifold (use UMAP, kernel PCA), low-variance directions contain discriminative information (PCA may remove what you need), or statistically independent components are desired (use ICA).

**Kernel PCA:**
```
Map data via kernel K; PCA in feature space φ(X)
Centred kernel: K̃ = K − 1ₙK − K1ₙ + 1ₙK1ₙ
Eigendecompose K̃ → project onto eigenvectors
```

**Independent Component Analysis (ICA):**
```
ICA model: x = As   where sᵢ are statistically independent sources
FastICA (fixed-point):
  w ← E[z g(wᵀz)] − E[g'(wᵀz)] w;  w ← w/||w||
  g(u) = u³ (kurtosis), tanh(u) (super-Gaussian), exp(−u²/2) (sub-Gaussian)
```
ICA maximises non-Gaussianity rather than variance. By the CLT, sums of independent components are more Gaussian than individuals — so maximising non-Gaussianity separates the mixture into original non-Gaussian sources.

---

### 7.2 t-SNE

```
High-d similarities:
  pⱼ|ᵢ = exp(−||xᵢ−xⱼ||²/(2σᵢ²)) / Σₖ≠ᵢ exp(−||xᵢ−xₖ||²/(2σᵢ²))
  pᵢⱼ = (pⱼ|ᵢ + pᵢ|ⱼ)/(2n)   (symmetrised)
  σᵢ set via binary search to match perplexity = 2^{H(pᵢ)}

Low-d similarities (Student-t, 1 dof — heavier tails):
  qᵢⱼ = (1+||yᵢ−yⱼ||²)⁻¹ / Σₖ≠ₗ(1+||yₖ−yₗ||²)⁻¹

Objective: min_{Y} KL(P||Q) = Σᵢⱼ pᵢⱼ log(pᵢⱼ/qᵢⱼ)

Gradient:
  ∂C/∂yᵢ = 4Σⱼ(pᵢⱼ−qᵢⱼ)(yᵢ−yⱼ)(1+||yᵢ−yⱼ||²)⁻¹
```

**Why Student-t Solves the Crowding Problem:**
In d dimensions, moderate distances span a wide range; in 2D, all must be compressed. A Gaussian similarity in low-d would crush moderate distances together. The Student-t distribution (1 dof) has much heavier tails — probability of large low-d distances decreases as `(1+||yᵢ−yⱼ||²)⁻¹` rather than exponentially. This "stretches" moderate distances in 2D, resolving the crowding problem.

**Gradient Interpretation:**
- When `pᵢⱼ > qᵢⱼ`: nearby in high-d, not in low-d → attractive force
- When `pᵢⱼ < qᵢⱼ`: nearby in low-d, not in high-d → repulsive force

**Practical Notes:** Perplexity 5–50 (typical: 30). Run 1000+ iterations. Results are non-deterministic. Distances and orientations between clusters are NOT meaningful — only local neighbourhood structure is preserved.

---

### 7.3 UMAP

**Core Idea (Riemannian Geometry + Fuzzy Simplicial Sets):**
```
1. Construct fuzzy topological representation in high-d:
   wᵢⱼ = exp(−(d(xᵢ,xⱼ) − ρᵢ)/σᵢ)   (membership strength)
   ρᵢ = distance to nearest neighbour   (local normalisation)

2. Optimise low-d positions to match this topology via cross-entropy:
   L = Σᵢⱼ [wᵢⱼ log(wᵢⱼ/ŵᵢⱼ) + (1−wᵢⱼ)log((1−wᵢⱼ)/(1−ŵᵢⱼ))]
   ŵᵢⱼ = (1 + a||yᵢ−yⱼ||^{2b})⁻¹

3. Optimise via SGD with negative sampling
```

**Local Normalisation — UMAP's Key Advantage Over t-SNE:**
The local normalisation via ρᵢ makes UMAP adaptive to varying density. Dense regions get tighter membership decay; sparse regions get wider decay. This allows UMAP to preserve both local and global structure simultaneously — unlike t-SNE which only preserves local structure reliably.

**UMAP vs t-SNE:**
```
Speed:          O(n log n) vs O(n²)
Global structure: UMAP preserves more
Determinism:     UMAP is more deterministic
Supervised mode: UMAP supports (t-SNE does not)
```

---

## 8. Ensemble Learning

### 8.1 Why Ensembles Work

**Variance Reduction Formula:**
```
If B models each with variance σ² and pairwise correlation ρ:
  Var(average) = ρσ² + (1−ρ)σ²/B

As B → ∞: Var → ρσ²  (irreducible correlation floor)
Reducing ρ (model diversity) is more effective than adding more models.
```

**Ambiguity Decomposition:**
```
(f̄(x) − y)² = (1/B)Σ(fᵦ(x)−y)² − (1/B)Σ(fᵦ(x)−f̄(x))²
error of ensemble = avg individual error − avg disagreement
```
To minimise ensemble error: (1) reduce individual errors, (2) increase disagreement (diversity). This formalises why ensembles benefit from diversity — accurate models that always agree provide no benefit from combination.

**Condorcet's Jury Theorem:**
If B independent classifiers each have P(correct) = p > 0.5:
```
P(majority correct) → 1 as B → ∞
```

---

### 8.2 Bagging

```
Bootstrap sample: draw n points with replacement
P(xᵢ not in bag) = (1−1/n)ⁿ → e⁻¹ ≈ 0.368
→ Each bag contains ~63.2% unique samples

Aggregation:
  Regression:     ŷ = (1/B) Σb hb(x)
  Classification: ŷ = mode{hb(x)}   (majority vote)

Out-Of-Bag (OOB) Error — free internal validation:
  OOB estimate ≈ (1/n) Σᵢ L(yᵢ, ĥ_OOB(xᵢ))
  where ĥ_OOB = average of trees that did NOT include xᵢ
  Asymptotically ≈ leave-one-out CV — no extra computation needed
```

---

### 8.3 Random Forest

```
At each node: subsample m features before finding best split
Classification: m = √p
Regression:     m = p/3

Feature Importance — Mean Decrease Impurity (MDI):
  FI(j) = (1/B) Σb Σ_{t:split on j} pₜ × ΔImpurity(t)

Feature Importance — Permutation Importance (more reliable):
  PI(j) = mean increase in OOB error when feature j is randomly permuted
  Unaffected by feature scale; detects non-linear relationships

Feature Importance — SHAP (most reliable):
  φⱼ = Σ_{S⊆F\{j}} [|S|!(|F|−|S|−1)!/|F|!] × [v(S∪{j}) − v(S)]
```

**Why Random Subsampling of Features Works:**
Subsampling m features per split decorrelates trees (reduces ρ in the variance formula) without significantly hurting individual tree accuracy. Trees trained on different feature subsets make different errors — the ensemble benefits from this diversity. This is why `m = √p` works so well: it provides enough features for good splits while ensuring different trees see different aspects of the data.

---

### 8.4 AdaBoost

```
1. wᵢ = 1/n for all i
2. For t = 1,...,T:
   a. Train hₜ on weighted data
   b. εₜ = Σᵢ wᵢ 𝟙[hₜ(xᵢ) ≠ yᵢ]   (weighted error)
   c. αₜ = (1/2)ln[(1−εₜ)/εₜ]        (weight for learner t)
   d. wᵢ ← wᵢ exp(−αₜ yᵢ hₜ(xᵢ)) / Zₜ   (reweight; misclassified get higher weight)
3. H(x) = sign(Σₜ αₜhₜ(x))

Training error bound: Train Error ≤ exp(−2Σₜ γₜ²),   γₜ = 0.5 − εₜ (edge)
```

**Properties:**
- αₜ > 0 iff εₜ < 0.5 (better than random)
- αₜ → ∞ as εₜ → 0 (near-perfect learner dominates)
- AdaBoost minimises Exponential Loss `L(y,f) = exp(−yf(x))`
- Equivalent to forward stagewise additive modelling with exponential loss

---

### 8.5 Gradient Boosting

```
Fₘ(x) = Fₘ₋₁(x) + η hₘ(x)

Pseudo-residuals (negative gradient of loss):
  rᵢₘ = −∂L(yᵢ, F(xᵢ))/∂F(xᵢ)|_{F=Fₘ₋₁}

Key pseudo-residuals:
  MSE loss:    rᵢ = yᵢ − Fₘ₋₁(xᵢ)       (ordinary residuals)
  MAE loss:    rᵢ = sign(yᵢ − Fₘ₋₁(xᵢ))
  Log-loss:    rᵢ = yᵢ − σ(Fₘ₋₁(xᵢ))
  Huber loss:  rᵢ = δ sign(rᵢ) if |rᵢ|>δ else rᵢ   (robust)
  Quantile(α): rᵢ = α if yᵢ>Fₘ₋₁ else α−1          (prediction intervals)
```

**Gradient Boosting as Functional Gradient Descent:**
Standard gradient descent operates in parameter space. Gradient boosting operates in *function space*: `Fₜ₊₁ = Fₜ + α hₜ` where `hₜ` is a new model fit to the negative functional gradient. By fitting a regression tree to these residuals, we find the best weak learner (restricted to the tree function class) pointing in the direction of steepest descent in function space. This perspective generalises immediately to any differentiable loss function.

---

### 8.6 XGBoost (Extreme Gradient Boosting)

```
Regularised objective:
  Obj = Σᵢ L(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)
  Ω(f) = γT + (λ/2)||w||²    (T = leaves, w = leaf scores)

Taylor expansion (2nd order):
  gᵢ = ∂L(yᵢ,ŷᵢ^{(t−1)})/∂ŷᵢ,   hᵢ = ∂²L/∂ŷᵢ²

Optimal leaf scores:
  wⱼ* = −Gⱼ/(Hⱼ+λ),   Gⱼ = Σᵢ∈Iⱼ gᵢ,   Hⱼ = Σᵢ∈Iⱼ hᵢ

Optimal split gain:
  Gain = (1/2)[Gₗ²/(Hₗ+λ) + Gᵣ²/(Hᵣ+λ) − (Gₗ+Gᵣ)²/(Hₗ+Hᵣ+λ)] − γ
```

**Why 2nd Order Outperforms Vanilla GBM:**
Standard GBM uses only first-order (gradient) information — like gradient descent. XGBoost uses second-order (Hessian) — like Newton's method. The Newton step accounts for curvature of the loss, giving more accurate step sizes and faster convergence. The optimal leaf score `wⱼ* = −Gⱼ/(Hⱼ+λ)` is the Newton step for each leaf — exactly optimal under the local quadratic approximation.

**Key Innovations:**
```
- Sparsity-aware splits: handles missing natively via learned default direction
- Approximate greedy with weighted quantile sketch (scalable to large data)
- Column subsampling per tree/level/node
- Cache-aware memory access (column blocks)
- Out-of-core computing for datasets > RAM
```

---

### 8.7 LightGBM

```
Histogram-based splitting:
  Bin features into B = 256 discrete bins
  Accumulate G, H stats into histograms in O(n) per feature
  Find best split: O(B) per feature → O(Bd) per level
  Speedup vs XGBoost exact: 8-16× on large datasets

GOSS (Gradient-based One-Side Sampling):
  Keep top-a% (large |gradient|) instances (high information)
  Sample b% from small-gradient instances, weight by (1−a)/b
  Preserves learning signal while reducing data size

EFB (Exclusive Feature Bundling):
  Bundle mutually exclusive sparse features (rarely non-zero simultaneously)
  Reduces effective feature count d' < d — speeds computation

Leaf-wise (best-first) growth:
  Always split the leaf with maximum delta_loss
  Grows deeper asymmetric trees → lower loss in fewer leaves
  Risk: overfit on small datasets → control with max_depth, min_data_in_leaf

Categorical feature handling:
  Sort by Σgᵢ/Σhᵢ per category → greedy binary split → O(k log k)
  No need for one-hot encoding
```

---

### 8.8 CatBoost

```
Ordered target statistics (prevents leakage):
  x̃ᵢ = (Σⱼ<σ(i) 𝟙[xⱼ=xᵢ]yⱼ + α·ȳ) / (Σⱼ<σ(i) 𝟙[xⱼ=xᵢ] + α)
  σ = random permutation; only past observations used → no leakage

Ordered boosting: M models trained on different orderings → prevents prediction shift

Symmetric trees (oblivious decision trees):
  All splits at same depth use same feature/threshold
  Tree = sequence of d (feature, threshold) pairs
  Evaluation: O(d) per sample — very fast inference
  Benefits: reduces overfit, fast inference, more interpretable
```

---

### 8.9 Stacking & Blending

```
k-Fold Stacking (avoids leakage):
  For each fold k:
    Train base models on D \ fold_k
    Predict on fold_k → meta-features Z[fold_k, :]
  Stack all folds → meta-feature matrix Z (n × M)
  Train meta-learner: fit(Z, y)
  Inference: average predictions from each fold's model

Meta-learner choice:
  Simple: Ridge/Lasso (interpretable, regularised)
  Complex: GBM (can learn non-linear combinations)
  Note: meta-learner trains on OOF predictions → fair, unbiased evaluation

Blending (holdout variant):
  Train (60%) → val (20%) → test (20%)
  Train base models on train, predict on val → meta-features
  Train meta-learner on val meta-features
  Faster but wastes data; susceptible to val set overfit
```

---

## 9. Time Series Analysis

### 9.1 Stationarity

```
Weak stationarity requires:
  E[Yₜ] = μ           (constant mean)
  Var(Yₜ) = σ²        (constant variance)
  Cov(Yₜ,Yₜ₋ₖ) = γₖ  (covariance depends only on lag k)

Achieving stationarity:
  First differencing:     ΔYₜ = Yₜ − Yₜ₋₁           (removes trend)
  Seasonal differencing:  ΔₛYₜ = Yₜ − Yₜ₋ₛ           (removes seasonality)
  Log transform:          Y'ₜ = log Yₜ               (stabilises variance)
  Log + differencing:     ΔlogYₜ ≈ % change

Unit Root Tests:
  ADF:   H₀: unit root (non-stationary); reject → evidence of stationarity
  KPSS:  H₀: stationarity; reject → non-stationarity
  Reject ADF AND fail to reject KPSS → strong stationarity evidence
```

**Why Stationarity is Required:**
The Wold decomposition theorem underpins all ARIMA modelling: every covariance-stationary process `{Yₜ}` can be written as `Yₜ = Σⱼ ψⱼ εₜ₋ⱼ + vₜ` — an MA(∞) plus a deterministic component. This decomposition does not apply to non-stationary series. Differencing transforms an I(d) series to an I(0) (stationary) series, restoring applicability.

**Spurious Regression Warning:**
If two independent non-stationary I(1) series are regressed, high R² and significant t-statistics appear with probability → 1 as n → ∞. This is spurious — statistical significance without any real relationship. The solution: test for cointegration before regressing non-stationary series.

---

### 9.2 ACF and PACF

```
ACF:  ρₖ = Cov(Yₜ, Yₜ₋ₖ) / Var(Yₜ)   (includes indirect effects)
PACF: φₖₖ = partial correlation at lag k (controlling for intervening lags)

95% confidence bounds: ±1.96/√n   (under H₀: white noise)

Model identification:
  MA(q): ACF cuts off at lag q; PACF decays geometrically
  AR(p): PACF cuts off at lag p; ACF decays geometrically
  ARMA:  both decay gradually
  Non-stationary: ACF decays very slowly (near unit root)
```

**ACF of AR(p) — Yule-Walker Equations:**
```
ρₖ = φ₁ρₖ₋₁ + φ₂ρₖ₋₂ + ... + φₚρₖ₋ₚ   for k ≥ 1
```
Solution: `ρₖ = Σⱼ Aⱼ λⱼᵏ` where λⱼ are roots of the characteristic polynomial. For stationary AR(p), all |λⱼ| < 1 → ACF decays geometrically. For a unit root (λⱼ = 1), ACF decays very slowly — the signature of non-stationarity.

---

### 9.3 ARIMA(p,d,q)

```
AR(p): Yₜ = c + φ₁Yₜ₋₁ + ... + φₚYₜ₋ₚ + εₜ
MA(q): Yₜ = μ + εₜ + θ₁εₜ₋₁ + ... + θqεₜ₋q
ARIMA(p,d,q): Φ(B)(1−B)ᵈ Yₜ = c + Θ(B)εₜ

Information Criteria (order selection):
  AIC  = −2ℓ + 2k
  AICc = AIC + 2k(k+1)/(n−k−1)   (small-sample correction — always prefer this)
  BIC  = −2ℓ + k log n            (consistent model selection for large n)
```

**AIC vs BIC:**
AIC is derived from minimising expected KL divergence — penalises complexity at 2 per parameter. BIC is derived from Bayesian model comparison — penalises at log(n) per parameter, favouring simpler models for large n. For large n, BIC tends to select the true model (consistency) while AIC does not. For small n or complex models, prefer AICc over AIC.

---

### 9.4 SARIMAX(p,d,q)(P,D,Q,s)

```
Full model:
  Φₚ(B)ΦP(Bˢ) ∇ᵈ ∇ˢᴰ Yₜ = c + Θq(B)ΘQ(Bˢ)εₜ + βXₜ

Example: SARIMA(1,1,1)(1,1,1)₁₂ for monthly data:
  (1−φ₁B)(1−Φ₁B¹²)(1−B)(1−B¹²)Yₜ = (1+θ₁B)(1+Θ₁B¹²)εₜ

Prophet (Facebook): decomposable model
  y(t) = g(t) + s(t) + h(t) + εₜ
  g(t): piecewise linear or logistic trend with changepoints
  s(t): Fourier series for seasonality
  h(t): user-supplied holiday effects
```

---

### 9.5 Exponential Smoothing (ETS)

```
SES (Simple):    ŷₜ₊₁ = α yₜ + (1−α) ŷₜ = α Σⱼ (1−α)ʲ yₜ₋ⱼ
Holt (Trend):    lₜ = α yₜ + (1−α)(lₜ₋₁+bₜ₋₁)
                 bₜ = β(lₜ−lₜ₋₁) + (1−β)bₜ₋₁
                 ŷₜ₊ₕ = lₜ + h bₜ
Holt-Winters (Multiplicative):
                 lₜ = α(yₜ/sₜ₋ₘ) + (1−α)(lₜ₋₁+bₜ₋₁)
                 sₜ = γ(yₜ/lₜ) + (1−γ)sₜ₋ₘ
                 ŷₜ₊ₕ = (lₜ+hbₜ)sₜ₋ₘ₊h

ETS (Error-Trend-Seasonality) state-space:
  Unified framework; AIC selects best combination
  Enables exact likelihood (Kalman filter) and calibrated prediction intervals
```

---

## 10. Deep Learning Foundations

### 10.1 Perceptron & MLP

**Forward Pass:**
```
Layer l:   zˡ = Wˡ aˡ⁻¹ + bˡ
           aˡ = g(zˡ)    (element-wise activation)
Output:    a^L = ŷ
```

**Universal Approximation Theorem (Cybenko 1989, Hornik 1991):**
A feedforward neural network with one hidden layer of N neurons and a non-polynomial activation function can approximate any continuous function `f: Kⁿ → ℝ` on a compact set K to arbitrary precision ε > 0, provided N is large enough.

**Depth vs Width:**
A depth-L ReLU network of width n partitions input space into O((n/d)^{d(L-1)} × nᵈ) linear regions — exponential in depth. Deep networks can represent some functions exponentially more compactly than shallow ones. The universal approximation theorem guarantees existence but not learnability — sample complexity grows with capacity.

**Why Non-Linearity is Essential:**
Without activation functions, a deep network `f(x) = W_L ... W_2 W_1 x` is equivalent to a single linear transformation (product of weight matrices). No matter how deep, a network of linear layers represents only linear functions. Non-linearity is what enables complex decision boundaries.

---

### 10.2 Activation Functions

```
Sigmoid: σ(z) = 1/(1+e⁻ᶻ)            range (0,1)     σ'(z) = σ(1−σ) ≤ 0.25
Tanh:    tanh(z) = (eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ)  range (−1,1)    tanh'(z) = 1−tanh²(z) ≤ 1
ReLU:    max(0,z)                       range [0,∞)     g'(z) = 𝟙[z>0]
Leaky:   max(αz,z), α≈0.01             range (−∞,∞)    g'(z) = 1 if z>0 else α
PReLU:   max(αz,z), α learnable
ELU:     z if z≥0; α(eᶻ−1) if z<0     smooth negative saturation → mean ≈ 0
SELU:    λ ELU(z, α)                   self-normalising: preserves mean=0, var=1
GELU:    z Φ(z) ≈ 0.5z(1+tanh(√(2/π)(z+0.044715z³)))   used in GPT, BERT
Swish:   z·σ(βz), β learnable          smooth, non-monotone; EfficientNet
Mish:    z·tanh(softplus(z))            smooth unbounded above, bounded below
Softmax: exp(zₖ)/Σⱼ exp(zⱼ)            multi-class output (valid probability distribution)
```

**Dying ReLU Problem:**
If a neuron's pre-activation is always negative (large negative bias), the gradient `g'(z) = 𝟙[z>0] = 0` for all inputs. The neuron permanently outputs 0 and receives no gradient — it is "dead." Causes: large learning rates, poor initialisation, aggressive regularisation. Solution: Leaky ReLU (g'(z) = α < 1 for z ≤ 0) allows a small gradient even for negative activations.

**GELU's Stochastic Interpretation:**
GELU(x) = x × Φ(x) where Φ is the standard normal CDF. This can be interpreted as: "keep the input with probability Φ(x), otherwise zero it." For large positive x: always kept. For large negative x: always dropped. This is a soft, input-dependent version of Dropout — the gating depends on the value itself, creating implicit regularisation in the attention mechanism.

**Choosing Activations:**
```
Hidden layers:  ReLU (default) → Leaky/ELU if dying ReLU → GELU for Transformers
Output:         Sigmoid (binary), Softmax (multi-class), Linear (regression)
Avoid:          Sigmoid/Tanh deep networks (vanishing gradient)
```

---

### 10.3 Backpropagation — Full Derivation

**Error Signals:**
```
δᴸ = ∂L/∂zᴸ = (ŷ − y) ⊙ g'(zᴸ)    (output layer)
δˡ = ((Wˡ⁺¹)ᵀ δˡ⁺¹) ⊙ g'(zˡ)      (hidden layers — chain rule)
```

**Parameter Gradients:**
```
∂L/∂Wˡ = δˡ (aˡ⁻¹)ᵀ    → shape: [nˡ × nˡ⁻¹]
∂L/∂bˡ = δˡ              → shape: [nˡ]
```

**Backpropagation as Reverse-Mode AD:**
Every operation in the forward pass is recorded in a computation graph. The backward pass applies the chain rule to every edge in reverse topological order. Each intermediate gradient is computed exactly once (memoisation) — total complexity O(forward pass) regardless of the number of parameters.

**Vanishing Gradient Analysis:**
```
||δˡ|| ≈ Π_{k=l}^{L−1} ||Wᵏ||·||g'(zᵏ)||

Sigmoid: max|g'| = 0.25. If ||W|| < 4 → product → 0 exponentially with depth.
ReLU:    g'(z) = 1 for z>0. Product bounded by ||W||^{L−l} — only exploding concern.
```

**Solutions to Vanishing/Exploding Gradients:**
```
Vanishing:  ReLU/GELU (non-saturating activations)
            Residual connections (gradient highway)
            Batch/Layer normalisation
            Careful initialisation (Xavier, He)
            LSTM/GRU (for sequences)

Exploding:  Gradient clipping: g := g × min(1, c/||g||)   (norm clipping)
            Spectral normalisation
```

---

### 10.4 Optimisation Algorithms

```
SGD:      θ := θ − α ∇L
Momentum: v := βv + ∇L;   θ := θ − αv
NAG:      v := βv + ∇L(θ−βv);   θ := θ − αv   (look-ahead gradient)

AdaGrad:  G += (∇L)²;   θ := θ − (α/√(G+ε)) ∇L
  Good for sparse gradients; G grows → lr → 0 (problem for long training)

RMSProp: G := βG + (1−β)(∇L)²;   θ := θ − (α/√(G+ε)) ∇L
  Fixes AdaGrad's diminishing lr with exponential moving average

Adam:
  m := β₁m + (1−β₁)∇L          (1st moment)
  v := β₂v + (1−β₂)(∇L)²       (2nd moment)
  m̂ = m/(1−β₁ᵗ);  v̂ = v/(1−β₂ᵗ)  (bias correction — critical at start)
  θ := θ − α m̂/(√v̂ + ε)
  Defaults: β₁=0.9, β₂=0.999, ε=1e-8, α=1e-3

AdamW:  θ := θ − α[m̂/(√v̂+ε) + λθ]
  Decouples weight decay from gradient adaptation
  Fixes L2 reg in Adam (they are NOT equivalent in adaptive methods)

Lion (2023): m := β₁m + (1−β₁)∇L;   θ := θ − α(sign(m) + λθ)
  Uses sign update — memory-efficient, strong performance on LLMs

Learning Rate Schedules:
  Step:          α(t) = α₀ γ^{floor(t/k)}
  Cosine:        α(t) = α_min + (α_max−α_min)(1+cos(πt/T))/2
  1Cycle:        increase to α_max, decrease to α_min/div_factor
  Linear warmup + cosine decay: standard for LLMs (warmup ~1000 steps)
```

**Why Adam Works (and Sometimes Doesn't):**
Adam adapts learning rates per parameter. Parameters with consistently large gradients get small effective rates; rare/small gradients get large rates — ideal for sparse NLP embeddings. Bias correction `(1−β₁ᵗ)` and `(1−β₂ᵗ)` are critical early: without them, m and v are heavily underestimated from zero initialisation, causing too-large early updates.

**Adam's known failure:** In some image models and convex problems, Adam converges to worse solutions than SGD+momentum. Adam's adaptive rates effectively change the geometry of the loss landscape — "generalisation-harming" sharp directions may receive large rates. AdamW partially addresses this via decoupled weight decay, but the fundamental geometry issue persists.

**Why Warmup is Needed:**
At the start of training, m and v estimates are noisy. A large initial learning rate combined with noisy gradients leads to catastrophic updates that destroy pre-trained weights (fine-tuning) or place parameters in a bad loss landscape region. Linear warmup gradually increases α, allowing m and v to stabilise before large updates.

---

### 10.5 Weight Initialisation

```
Problem: without careful init, activations explode or vanish across layers.

Xavier/Glorot (Tanh/Sigmoid):
  Var(W) = 2/(n_in + n_out)
  Derived: keep Var(activation) ≈ 1 through both forward AND backward pass

He/Kaiming (ReLU):
  Var(W) = 2/n_in
  The factor 2 compensates for ReLU zeroing half the activations (halves variance)

LeCun (SELU): Var(W) = 1/n_in

Orthogonal Init: W = random orthogonal matrix (via QR decomposition)
  Preserves gradient norms in linear networks; used for RNNs

Sparse Init: small number of non-zero weights per unit — avoids symmetry breaking
```

**The Exploding/Vanishing Variance Problem:**
For a deep ReLU network: `Var(zˡ) = (n_{l−1} × Var(W)/2) × Var(zˡ⁻¹)`. For variance to be preserved layer-to-layer: `n_{l−1} × Var(W)/2 = 1` → `Var(W) = 2/n_{l−1}`. This is exactly He initialisation. Without this, variances grow or shrink exponentially with depth → activations explode or vanish.

---

## 11. Convolutional Neural Networks (CNN)

### 11.1 The Convolution Operation

```
2D Cross-Correlation:
  (I ⋆ K)(i,j) = Σₘ Σₙ I(i+m, j+n) × K(m,n)

Output shape:
  H_out = ⌊(H_in + 2P − D(K−1) − 1) / S⌋ + 1
  W_out = ⌊(W_in + 2P − D(K−1) − 1) / S⌋ + 1
  (D = dilation; S = stride; P = padding; K = kernel size)

Parameter count per conv layer:
  (K × K × C_in + 1) × C_out   (bias per filter)
  Compare FC: H_in × W_in × C_in × H_out × W_out × C_out — massive

Inductive biases:
  Translation equivariance: f(T(x)) = T(f(x))  (shift input → shift output)
  Local connectivity: each output depends on a local region
  Weight sharing: same kernel applied everywhere → fewer params + regularisation
```

**Why Translation Equivariance is the Right Prior for Images:**
A cat is a cat regardless of where it appears in an image. Translation equivariance encodes this prior: the same filter detects features everywhere. Instead of learning a separate "cat detector" for each position (n_positions × n_filters parameters), we learn a single filter that works everywhere (n_filters parameters). This is enormous sample efficiency.

**Receptive Field:**
```
RF₁ = K₁
RFₗ = RFₗ₋₁ + (Kₗ−1) × Πⱼ<ₗ Sⱼ

Two 3×3 conv layers: same RF (5×5) as one 5×5, fewer params, two non-linearities.
Dilated (atrous) convolutions: effective K' = D(K−1)+1 → larger RF without extra params
```

**Convolution Types:**
```
Standard 2D:         K×K×C_in filters, C_out outputs
Depthwise:           K×K×1 per channel → C_in outputs (no cross-channel mixing)
Pointwise (1×1):     mix channels only, no spatial operation
Depthwise-separable: Depthwise + Pointwise → 8-9× fewer params
Transposed/deconv:   upsampling by reversing conv; used in U-Net decoder
Dilated/atrous:      insert zeros between kernel elements; larger RF, same params
```

---

### 11.2 Pooling & Normalisation

```
Max pooling:     y(i,j) = max over p×p window
Average pooling: y(i,j) = mean over p×p window
Global Average Pooling (GAP): y_c = mean of all spatial dims for channel c
  Replaces FC layers; forces feature maps to be class activations; reduces overfit
```

**Normalisation Layers:**
```
Batch Norm:
  μ_B = (1/m)Σxᵢ;   σ²_B = (1/m)Σ(xᵢ−μ_B)²
  x̂ᵢ = (xᵢ−μ_B)/√(σ²_B+ε);   yᵢ = γx̂ᵢ + β
  At test: use running statistics (EMA over training batches)

Layer Norm:      normalise over feature dim (not batch) — works for batch size=1, sequences
Instance Norm:   normalise per sample, per channel — style transfer
Group Norm:      divide channels into G groups, normalise within each — small batches
RMS Norm:        x̂ = x/√(Σxᵢ²/d + ε); y = γx̂    — simpler, used in LLaMA, Gemma
```

**Why Batch Normalisation Accelerates Training:**
Batch normalisation reduces internal covariate shift — the change in layer input distribution as previous layers' parameters update. More importantly, Santurkar et al. (2018) showed BN makes the loss function smoother (smaller Lipschitz constant of gradients) — the landscape has fewer sharp peaks and valleys, allowing larger learning rates, faster convergence, and reduced initialisation sensitivity.

**BN Training vs Inference:**
During training: each mini-batch computes its own statistics (μ_B, σ²_B) — introducing noise that acts as regularisation. During inference: use running statistics (EMA of training batch statistics) for deterministic, consistent predictions.

---

### 11.3 CNN Architectures

```
AlexNet (2012):  5 conv + 3 FC; ReLU; Dropout; 60M params; started the DL revolution
VGG (2014):      All 3×3 conv; 16/19 layers; 138M params; very regular structure
  Two 3×3 convs have same RF as one 5×5: 2×(3²C²) vs 5²C² params — cheaper + more non-linearity
GoogLeNet (2014): Inception modules (parallel 1×1, 3×3, 5×5); 22 layers; 5M params
  1×1 bottleneck: reduce C_in before expensive conv → saves compute
ResNet (2015):   Skip connections y = F(x) + x; 50/101/152 layers; 3.57% top-5 error
  Gradient: ∂L/∂x = ∂L/∂y(1 + ∂F/∂x); +1 prevents vanishing gradient
  Residual principle: easier to learn small deviation from identity than full mapping
DenseNet (2017): xˡ = Hˡ([x⁰,x¹,...,xˡ⁻¹]); every layer connected to all subsequent
  Strong gradient flow; feature reuse; fewer parameters than ResNet
EfficientNet(2019): Compound scaling depth/width/resolution: d=αᵠ, w=βᵠ, r=γᵠ; α·β²·γ²≈2
  Best accuracy/FLOPs tradeoff across all scales
ViT (2020):      Patch-based image Transformer (see Section 15.14)
```

---

## 12. Recurrent Neural Networks (RNN)

### 12.1 Vanilla RNN

```
hₜ = tanh(Wₕ hₜ₋₁ + Wₓ xₜ + bₕ)   (hidden state — depends on previous state)
ŷₜ = softmax(Wᵧ hₜ + bᵧ)            (output)

Shared parameters across all time steps: Wₕ, Wₓ, bₕ, Wᵧ, bᵧ
```

**BPTT (Backpropagation Through Time):**
```
∂hᵢ/∂hᵢ₋₁ = diag(tanh'(zᵢ)) Wₕ

Vanishing: ||Wₕ||·max|tanh'| < 1 → gradient exponentially decreases with sequence length
Exploding: ||Wₕ|| > 1 → gradient exponentially grows

Gradient clipping: g := g × min(1, c/||g||)   (preserves direction)
Truncated BPTT: backpropagate only k steps → O(k) memory, biased gradient
```

**The Vanishing Gradient is a Memory Problem:**
If the gradient from loss at time T with respect to hidden state at time k vanishes for large T-k, the network cannot learn dependencies spanning more than a few steps. In practice, vanilla RNNs struggle beyond 10-20 steps. LSTM solves this with the cell state highway where the gradient flows through element-wise multiplication rather than matrix multiplication.

---

## 13. Long Short-Term Memory (LSTM) & GRU

### 13.1 LSTM

```
Forget gate:   fₜ = σ(Wf[hₜ₋₁;xₜ] + bf)        — how much to forget cell state
Input gate:    iₜ = σ(Wᵢ[hₜ₋₁;xₜ] + bᵢ)        — how much to write
Candidate:     C̃ₜ = tanh(Wc[hₜ₋₁;xₜ] + bc)     — new candidate cell values
Cell update:   Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ       — forget old + add new
Output gate:   oₜ = σ(Wo[hₜ₋₁;xₜ] + bo)         — what to expose
Hidden state:  hₜ = oₜ ⊙ tanh(Cₜ)               — filtered cell state

Parameter count: 4 × (nₕ(nₓ + nₕ) + nₕ)   (4 gates)
```

**How LSTM Solves Vanishing Gradient:**
The gradient through the cell state:
```
∂Cₜ/∂Cₜ₋₁ = fₜ   (element-wise multiplication — no matrix multiplication!)
```
If `fₜ ≈ 1`: gradient flows unchanged → no vanishing over long sequences. If `fₜ ≈ 0`: old information selectively forgotten. The LSTM learns to set `fₜ ≈ 1` for time steps where long-range dependencies exist, and `fₜ ≈ 0` when a new pattern begins.

---

### 13.2 GRU

```
Reset gate:   rₜ = σ(Wr[hₜ₋₁;xₜ]+br)
Update gate:  zₜ = σ(Wz[hₜ₋₁;xₜ]+bz)
Candidate:    h̃ₜ = tanh(Wh[rₜ⊙hₜ₋₁;xₜ]+bh)
Hidden:       hₜ = (1−zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ

zₜ ≈ 1: new information overrides old (update gate open)
zₜ ≈ 0: keep old hidden state (update gate closed)
rₜ ≈ 0: ignore past hidden state (reset gate)

Parameter count: 3 × (nₕ(nₓ+nₕ) + nₕ)   (25% fewer than LSTM)
```

| Property | LSTM | GRU |
|---|---|---|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| State | Separate cell + hidden | Single hidden state |
| Parameters | 4× | 3× |
| Best for | Very long sequences | Shorter sequences, efficiency |

---

### 13.3 Sequence-to-Sequence & Attention

```
Encoder: {hₜ} = RNN(x₁,...,xT)   — source hidden states
Decoder: sₜ = RNN(sₜ₋₁, yₜ₋₁, cₜ);   ŷₜ = softmax(Wsₜ)

Bahdanau (Additive) Attention:
  eₜⱼ = vᵀ tanh(Wₛsₜ₋₁ + Wₕhⱼ)    (alignment score)
  αₜⱼ = exp(eₜⱼ) / Σⱼ' exp(eₜⱼ')   (softmax → probability distribution)
  cₜ = Σⱼ αₜⱼ hⱼ                    (weighted context vector)

Luong (Multiplicative) Attention:
  eₜⱼ = sₜᵀ Wₐ hⱼ   (or simply sₜᵀhⱼ for dot product)
```

**Why Fixed-Size Context Bottleneck Fails:**
In vanilla seq2seq, the encoder compresses the entire input into a single fixed-size vector — catastrophically information-lossy for long sequences. Attention solves this: the decoder at each step computes a dynamic weighted sum `cₜ = Σⱼ αₜⱼ hⱼ` over all encoder states. The alignment weights `αₜⱼ` learn which source positions are most relevant for generating each target token — the decoder has access to the entire source sequence.

**Attention as Soft Template Matching:**
The attention weight `αₜⱼ` is the probability that source position j aligns with target position t. High compatibility between decoder query state and encoder key → high attention weight → strong influence on context vector. This is differentiable soft template matching.

---

## 14. Advanced Deep Learning

### 14.1 Generative Models

**VAE (Variational Autoencoder):**
```
ELBO: L = E_q[log p(x|z)] − KL(q(z|x) || p(z))
           ← Reconstruction      ← Regularisation (KL to prior)

Reparameterisation: z = μ(x) + σ(x)⊙ε,  ε~N(0,I)   → enables backprop through sampling
β-VAE: L = Reconstruction − β·KL,  β > 1 → better disentanglement
```

**VAE ELBO — Information-Theoretic Interpretation:**
The reconstruction term maximises log-likelihood of data under the decoder. The KL term penalises the encoder for deviating from the prior — it regularises the latent space. Maximising ELBO is equivalent to minimising the gap between the approximate and true posterior. The variational approximation q(z|x) is the best Gaussian approximation to the true posterior.

**Posterior Collapse Warning:**
KL term can dominate, forcing q(z|x) → p(z). The decoder ignores z entirely. Solutions: β-VAE with small β, KL annealing from 0 to 1, or VQ-VAE (discrete latent space).

**GAN (Generative Adversarial Network):**
```
min_G max_D E[log D(x)] + E[log(1−D(G(z)))]
D: real/fake discriminator;  G: generator from noise z~p(z)

Non-saturating loss for G: use −E[log D(G(z))] instead of E[log(1−D(G(z)))]
  (avoids gradient saturation early in training)

Wasserstein GAN (more stable):
  min_G max_{||f||_L≤1} E[f(x)] − E[f(G(z))]
  Spectral normalisation: ||W||_σ = 1 → enforces Lipschitz constraint
```

**Diffusion Models (DDPM):**
```
Forward: q(xₜ|xₜ₋₁) = N(√(1−βₜ)xₜ₋₁, βₜI)   (gradually add noise)
Reverse: p_θ(xₜ₋₁|xₜ) = N(μ_θ(xₜ,t), Σ_θ)    (learned denoising)
Objective: ||ε − ε_θ(√ᾱₜx₀ + √(1−ᾱₜ)ε, t)||²   (predict noise)
ᾱₜ = Πₛ≤ₜ(1−βₛ)

Score matching connection:
  ∇_x log p_t(x) = −ε/√(1−ᾱₜ)   (score function = normalised noise)
  Training a denoiser ≡ training a score estimator
```

---

### 14.2 Regularisation

```
Dropout:
  Training: hᵢ = gᵢ × rᵢ,   rᵢ~Bernoulli(p)   (keep prob p)
  Inverted dropout (standard): divide by p during training → no test-time scaling
  Interpretation: ensemble of 2ⁿ thinned networks with shared weights

  Variants:
  DropConnect:   drop weights (not activations) — stronger
  Spatial Dropout: drop entire feature maps — better for conv layers
  DropPath:      drop entire residual blocks — regularises deep nets

Early Stopping:
  Monitor val loss; stop when no improvement for patience steps
  Equivalent to L2 regularisation under some conditions

Data Augmentation:
  Images: flip, rotation, random crop, colour jitter
          Mixup: xm = λxᵢ+(1−λ)xⱼ, ym = λyᵢ+(1−λ)yⱼ
          CutMix: paste region from another image
          AutoAugment, RandAugment: learned/simplified augmentation policies
  Text:   synonym replacement, back-translation, EDA

Label Smoothing:
  y_smooth = (1−ε)y_hard + ε/K
  Prevents overconfident logits; improves calibration
  Information-theoretically: adds entropy penalty preventing logits from → ∞
  Gradient: encourages all logits to be similar, not just the max

Spectral Normalisation: W_SN = W/σ₁(W) → Lipschitz constraint → stable training
```

---

## 15. Transformers — Complete Deep Dive

### 15.1 Motivation — Why Transformers?

```
RNN limitations:
  - Sequential computation → cannot parallelise training
  - Vanishing/exploding gradients over long sequences
  - Fixed-size context vector bottleneck (seq2seq)

CNN limitations:
  - Receptive field grows linearly/logarithmically with depth
  - Long-range dependencies require many layers

Transformer insight: replace sequential processing with
PARALLEL attention over all positions simultaneously.
O(n²d) but fully parallelisable → orders of magnitude faster on modern GPUs.
```

**The Fundamental Advantage: Parallelism Over Sequential Depth:**
RNNs process one token at a time — serial depth is T operations for sequence length T. Transformers compute all positions simultaneously via matrix operations — serial depth is just 1 (plus MLP depth). On modern GPUs with thousands of parallel cores, this difference is the key to Transformer scalability. This is why training a 175B parameter GPT-3 on thousands of GPUs is feasible.

---

### 15.2 Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V

Shapes:
  Q: (seq_len_q × dₖ)   — queries (what am I looking for?)
  K: (seq_len_k × dₖ)   — keys   (what do I contain?)
  V: (seq_len_v × dᵥ)   — values (what information do I return?)
  Output: (seq_len_q × dᵥ)

Score matrix: S = QKᵀ/√dₖ   shape: (seq_len_q × seq_len_k)
Attention weights: A = softmax(S, dim=−1)    — row-wise softmax → probability distribution
Output: Z = AV
```

**Why √dₖ Scaling?**
If q,k ~ N(0,1) i.i.d., then `qᵀk ~ N(0,dₖ)` — variance grows linearly with dₖ. Without scaling: large dₖ → large dot products → softmax saturates (near one-hot) → gradients ≈ 0 → no learning. Dividing by √dₖ normalises variance to 1 regardless of dimension.

**Attention as Soft Dictionary Lookup:**
A hard dictionary retrieves the value for an exact query match. Attention is differentiable: given query q, it computes a weighted sum of all values, weighted by similarity of query to each key. This allows retrieving a blend of information from multiple positions — unlike a hard dictionary which retrieves exactly one entry.

**Why Attention is O(n²):**
The score matrix `S = QKᵀ` has shape `(n × n)` — every query attends to every key. For n=1024 tokens: 1 million entries. For n=100,000 tokens: 10 billion entries (~40GB for float32). This quadratic scaling with sequence length is the core bottleneck of standard Transformers.

**Masked Attention (for autoregressive generation):**
```
M = upper triangular matrix with −∞
A = softmax((QKᵀ/√dₖ) + M)
→ each position can only attend to previous positions (causal mask)
```

**Cross-Attention:**
```
Q from decoder, K and V from encoder output
→ Queries (what target token needs) attends to Keys/Values (source information)
```

---

### 15.3 Multi-Head Attention (MHA)

```
For h = 1,...,H heads:
  headₕ = Attention(Q Wₕᴼ, K Wₕᴷ, V Wₕᵛ)
  Wₕᴼ ∈ ℝ^{d_model × dₖ},   dₖ = dᵥ = d_model/H

MultiHead(Q,K,V) = Concat(head₁,...,headₕ) Wᵒ
Wᵒ ∈ ℝ^{H·dᵥ × d_model}

Parameter count:
  3 × d_model²  (for Q,K,V projections across all heads)
+ d_model²       (output projection Wᵒ)
= 4 × d_model²
```

**Why Multiple Heads?**
Each head can attend to different representation subspaces simultaneously:
- Head 1: syntactic relationships (subject-verb agreement)
- Head 2: semantic relationships (coreference)
- Head 3: positional patterns (nearby tokens)
- Head 4: entity tracking across long distances

Multiple heads effectively create an ensemble of attention patterns — more expressive than a single attention mechanism.

---

### 15.4 Transformer Block

```
Pre-LN Transformer Block (GPT-2+, more stable training):
  x' = x + MultiHeadAttn(LayerNorm(x))
  x'' = x' + FFN(LayerNorm(x'))

Post-LN Transformer Block (original "Attention is All You Need"):
  x' = LayerNorm(x + MultiHeadAttn(x))
  x'' = LayerNorm(x' + FFN(x'))

Feed-Forward Network (FFN):
  FFN(x) = max(0, xW₁ + b₁)W₂ + b₂           (ReLU, original)
  FFN(x) = GELU(xW₁)W₂                          (GPT-2)
  FFN(x) = (SwiGLU(xW₁) ⊙ xW₃)W₂              (LLaMA, Mistral)
  Typical d_ff = 4 × d_model
  Parameter count: 2 × d_model × d_ff = 8 × d_model²

Total params per layer:
  MHA: 4d_model²   FFN: 8d_model²   LayerNorm: 4d_model
  ≈ 12d_model² per layer (dominant term)

L-layer Transformer:
  Total params ≈ L × 12d_model² + embedding (V × d_model) + output head
```

---

### 15.5 Positional Encoding

```
Sinusoidal (original — deterministic, no extra params):
  PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
  PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
  Property: PE(pos+k) is a linear function of PE(pos) → relative offsets encodable

Learnable Absolute (BERT, GPT):
  E_pos ∈ ℝ^{max_len × d_model}   (learned embedding table)
  Limited to max_len positions; does not extrapolate

RoPE (Rotary Position Embedding — LLaMA, GPT-NeoX):
  Rotate Q, K vectors by angle proportional to position:
  f(x, m) = x ⊗ cos(mθ) + rotate(x) ⊗ sin(mθ)
  Key property: (f(q,m))ᵀ(f(k,n)) depends only on (m−n) → true relative position
  Generalises to any sequence length; extrapolates well

ALiBi (Attention with Linear Biases — MPT):
  Score(i,j) = qᵢᵀkⱼ/√dₖ − m|i−j|    (subtract linear bias per head)
  m: head-specific slope; extrapolates to longer sequences at test time
```

---

### 15.6 Full Transformer Architecture

```
Encoder-Only (BERT family):
  Input: [CLS] + tokens + [SEP]
  Bidirectional: each token attends to all others
  Tasks: classification, NER, QA (span extraction)
  Training: MLM (15% masked) + NSP

Decoder-Only (GPT family — most modern LLMs):
  Causal (masked) self-attention only
  Autoregressive: P(x) = Πₜ P(xₜ|x₁,...,xₜ₋₁)
  Training: CLM (predict next token) — simplest and most scalable

Encoder-Decoder (T5, BART, original Transformer):
  Encoder: bidirectional (processes full source)
  Decoder: causal self-attn + cross-attn to encoder output
  Tasks: translation, summarisation, structured prediction
```

---

### 15.7 Pre-Training Objectives

```
MLM (BERT): Mask 15% of tokens → 80% [MASK], 10% random, 10% unchanged
  Predict original at masked positions; bidirectional context

CLM (GPT): L = −Σₜ log P(xₜ|x₁,...,xₜ₋₁)
  Simple; highly scalable; enables zero-shot generation

Span Corruption (T5):
  Replace contiguous token spans with sentinels [X], [Y]
  Decoder reconstructs original spans — efficient (shorter input)

ELECTRA (most sample-efficient):
  Generator (small BERT): performs MLM
  Discriminator (full model): predict replaced/original for ALL tokens
  Supervise every token, not just 15% → more efficient

Prefix LM (UniLM, GLM):
  Prefix: bidirectional attention; Continuation: causal
  Best of both worlds for conditional generation
```

---

### 15.8 Transformer Scaling Laws

**Chinchilla Scaling Laws (Hoffmann et al., 2022):**
```
L(N, D) = E + A/Nᵅ + B/Dᵝ    (irreducible + model + data terms)
α ≈ β ≈ 0.5

Optimal for compute budget C ≈ 6ND (FLOPs):
  N* ≈ G × C^{0.5},  D* ≈ G⁻¹ × C^{0.5}   (G ≈ 0.2)
  → Optimal: ~1 training token per parameter
  E.g., 70B model → train on ~1.4T tokens

Implications:
  Smaller models trained on more data often outperform larger under-trained models
  GPT-3 (175B) was significantly undertrained by Chinchilla standards
  LLaMA: trained Chinchilla-optimally at each scale
```

---

### 15.9 Efficient Attention

```
FlashAttention (Dao et al., 2022):
  Tiling: Q,K,V split into blocks; process one block at a time in SRAM
  Online softmax normalisation — update max(m) and sum(ℓ) per block
  Memory: O(n) instead of O(n²); 2-4× faster wall-clock; numerically exact
  Never materialises the full n×n attention matrix

FlashAttention-2 (2023): better work partitioning → 2× faster
FlashAttention-3 (2024): H100-optimised with overlapped compute/memory

Sparse Attention:
  Local window: attend to w neighbours → O(nw)
  Global + local: [CLS] attends globally; others locally (Longformer)

Multi-Query Attention (MQA):
  All heads share K, V; only Q is separate
  Saves KV cache memory by H×; slight quality drop

Grouped Query Attention (GQA — LLaMA-2, Mistral):
  G groups, each sharing K, V; H/G queries per group
  Interpolates between MHA (G=H) and MQA (G=1) — good speed/quality balance

Linear Attention:
  Replace softmax with kernel: K(q,k) ≈ φ(q)ᵀφ(k)
  Exploit associativity: Attention ≈ φ(Q)(φ(K)ᵀV) → O(nd²) instead of O(n²d)

Paged Attention (vLLM):
  KV cache managed in pages (like OS virtual memory)
  Non-contiguous allocation → less fragmentation; 3-24× throughput vs HuggingFace
```

---

### 15.10 Post-Training: Instruction Tuning & RLHF

```
Supervised Fine-Tuning (SFT):
  Fine-tune on (instruction, response) pairs
  L = −Σₜ log P(yₜ|y₁,...,yₜ₋₁, x)   (only on response tokens)

RLHF (Reinforcement Learning from Human Feedback):
  1. SFT: fine-tune base model on demonstrations
  2. Reward Model:
     Collect pairs (y_w, y_l) where humans prefer y_w
     L = −log σ(r_θ(x,y_w) − r_θ(x,y_l))   (Bradley-Terry model)
  3. RL (PPO):
     reward = r_θ(x,y) − β KL(π_θ(y|x) || π_ref(y|x))
     KL term prevents policy from diverging too far from SFT model
     PPO: clip(π/π_old, 1−ε, 1+ε) — prevents large destabilising updates

DPO (Direct Preference Optimisation — Rafailov et al., 2023):
  L_DPO = −E log σ(β log π_θ(y_w|x)/π_ref(y_w|x) − β log π_θ(y_l|x)/π_ref(y_l|x))
  Bypasses reward model; directly fine-tunes from preferences
  Simpler than PPO; widely adopted (Zephyr, Tulu)

Constitutional AI (Anthropic):
  Model critiques and revises own responses based on principles
  RLAIF: use AI feedback instead of human feedback

ORPO: combines SFT + preference learning in single step
KTO: unpaired feedback (good/bad labels only, no comparison needed)
```

---

### 15.11 Parameter-Efficient Fine-Tuning (PEFT)

```
Problem: fine-tuning 70B model needs 70B+ gradient params = >500GB → infeasible.

LoRA (Low-Rank Adaptation — Hu et al., 2021):
  For W ∈ ℝ^{d×k}: ΔW = BA,  B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k},  r << min(d,k)
  Forward: (W + BA)x   — no extra inference latency after merging
  Init: A ~ N(0,σ²), B = 0 → ΔW=0 at start (preserves pre-trained weights)
  Param savings: r=16, d=k=4096 → 0.78% of W's parameters
  Typical r: 4-64; most common: r=16

QLoRA (Dettmers et al., 2023):
  4-bit quantisation of base model (NF4 — Normal Float 4)
  LoRA adapters in float16/bfloat16
  Fine-tune 65B model on single 48GB GPU (vs >1TB otherwise)

Prefix Tuning: prepend K trainable "prefix" tokens to K and V at each layer (~0.1% params)
Prompt Tuning: prepend trainable soft tokens to input embeddings only — simpler
Adapter Layers: insert small bottleneck MLP (d → r → d) after attention and FFN — ~3-4% params
IA³: rescale K, V, and FFN activations with learned vectors — <0.1% params
```

---

### 15.12 Inference Optimisation & KV Cache

```
KV Cache:
  During autoregressive decoding: past K, V are cached and reused
  New token only needs new K, V + attention to cached states
  Memory: 2 × L × H × dₖ × n_tokens × bytes  — grows linearly with sequence length

Quantisation:
  FP32 → FP16/BF16: 2× memory, minimal quality loss (standard training)
  FP16 → INT8: 4× memory, small quality loss (LLM.int8(), SmoothQuant)
  INT8 → INT4: 8× memory, moderate quality loss (GPTQ, NF4, AWQ)
  AWQ: protect 1% salient weights (large activations); INT4 rest

Speculative Decoding:
  Small draft model generates k tokens quickly
  Large target model verifies all k tokens in parallel
  Speedup: 2-3× with good draft model sharing target's distribution

Parallelism:
  Tensor: split weight matrices across GPUs (column/row)
  Pipeline: split layers across GPUs with micro-batching
  Data: replicate model; split batch
```

---

### 15.13 Notable LLM Architectures

```
GPT-3 (2020):      175B; few-shot learning; emergent abilities
LLaMA-2 (2023):    7B-70B; GQA; 4k context; open + chat versions
LLaMA-3 (2024):    8B-70B-405B; 128k context; very strong open weights
Mistral-7B (2023): Sliding window attention; GQA; beats LLaMA-2-13B
Mixtral-8×7B (2023): Sparse MoE; 8 experts × 7B; activate 2/8 per token; 47B total / 12B active
Claude (Anthropic): Constitutional AI; strong on long context and safety
Gemini (Google):    Native multimodal; 1M+ context (Ultra)

Mixture of Experts (MoE):
  FFN replaced by E expert FFNs:
  y = Σₑ gₑ(x) FFNₑ(x)
  Router: g(x) = softmax(xWg); select top-K experts (usually K=2)
  Load balancing loss: auxiliary loss to prevent expert collapse
  Sparse MoE: only top-2 experts activated → cheap inference relative to model size
```

---

### 15.14 Vision Transformers (ViT)

```
ViT (Dosovitskiy et al., 2020):
  1. Divide image H×W×C into N = HW/P² patches of size P×P
  2. Linear projection: xₚ = PatchEmbed(patch) ∈ ℝ^{d_model}
  3. Add [CLS] token + learnable position embeddings
  4. Standard Transformer encoder: L layers of MHA + FFN
  5. Classification: linear head on [CLS] output

Patch embedding: xₚ = reshape(patch) × W_E,   W_E ∈ ℝ^{P²C × d}
Complexity: O(n²d) where n = HW/P² (for P=16, 224×224: n = 196 patches)

Variants:
  DeiT:  data-efficient ViT (training tricks + distillation token)
  Swin:  hierarchical; local shifted windows; O(n) not O(n²)
    Window attention: attend within w×w windows
    Shifted windows: alternate between two partitions for cross-window connections
  BEiT: BERT-style masked image modelling pre-training
  MAE:  mask 75% of patches; reconstruct pixels; very sample-efficient
  DINO: self-supervised ViT; teacher-student with EMA; global + local crops
  CLIP: contrastive image-text pre-training
    L = −Σᵢ log exp(τ·sᵢᵢ)/Σⱼ exp(τ·sᵢⱼ)   (image i matches text i)
    Enables zero-shot image classification via text prompts
```

---

## 16. LangChain & LLM Application Engineering

### 16.1 LangChain Overview

**Core Abstractions:**
```
Model I/O:     prompt templates, LLM/ChatModel wrappers, output parsers
Data:          document loaders, text splitters, vector stores, retrievers
Chains:        sequences of components (LCEL — LangChain Expression Language)
Agents:        LLMs that decide which tools to use based on reasoning
Memory:        persist state across interactions
Callbacks:     hooks for logging, streaming, monitoring (LangSmith)
```

**Installation:**
```bash
pip install langchain langchain-openai langchain-community
pip install langchain-anthropic langchain-google-genai
```

**Why RAG Instead of Just Fine-Tuning?**

| Dimension | Fine-Tuning | RAG |
|---|---|---|
| Knowledge update | Requires retraining | Update database only |
| Cost | High (GPU training) | Low (inference only) |
| Interpretability | Black box | Inspect retrieved docs |
| Best for | Learning new skills/style | Dynamic factual knowledge |
| Latency | Same as base | Higher (retrieval step) |

Modern systems often combine both: fine-tune for behaviour/style, RAG for factual knowledge.

---

### 16.2 LLM & Chat Model Wrappers

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# ChatOpenAI
chat = ChatOpenAI(model="gpt-4o", temperature=0)
messages = [
    SystemMessage(content="You are an expert data scientist."),
    HumanMessage(content="Explain gradient descent in 3 bullet points.")
]
response = chat.invoke(messages)

# Streaming
for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)

# Anthropic Claude
claude = ChatAnthropic(model="claude-opus-4-6", temperature=0)

# Local models via Ollama
from langchain_community.llms import Ollama
local_llm = Ollama(model="llama3")

# Batch inference
results = chat.batch([{"messages": [HumanMessage(content=q)]} for q in questions])
```

---

### 16.3 Prompt Templates

```python
from langchain_core.prompts import (
    PromptTemplate, ChatPromptTemplate,
    FewShotPromptTemplate, MessagesPlaceholder
)

# Simple template
prompt = PromptTemplate(
    input_variables=["topic", "audience"],
    template="Explain {topic} to a {audience} in simple terms."
)

# Chat prompt (recommended for chat models)
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer in {language}."),
    ("human", "{question}"),
])
messages = chat_prompt.format_messages(
    role="machine learning professor", language="English",
    question="What is overfitting?"
)

# With conversation history placeholder
conv_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Few-shot prompt
examples = [{"input": "2+2", "output": "4"}, {"input": "3×5", "output": "15"}]
example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)
few_shot_prompt = FewShotPromptTemplate(
    examples=examples, example_prompt=example_template,
    prefix="Solve the arithmetic problem:",
    suffix="Input: {query}\nOutput:", input_variables=["query"]
)
```

---

### 16.4 LCEL — LangChain Expression Language

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

# Basic chain: prompt | model | parser (pipe operator)
chain = chat_prompt | chat | StrOutputParser()
result = chain.invoke({"role": "tutor", "language": "English", "question": "What is SVM?"})

# All components implement Runnable: .invoke(), .stream(), .batch(), .ainvoke()

# Parallel execution
parallel = RunnableParallel(
    summary=prompt_summary | chat | StrOutputParser(),
    keywords=prompt_keywords | chat | StrOutputParser(),
)
results = parallel.invoke({"text": "Long document..."})

# Conditional routing
def route(info):
    if "urgent" in info["question"].lower():
        return priority_chain
    return standard_chain

full_chain = {"question": RunnablePassthrough()} | RunnableLambda(route)

# Retry and fallback
chain_with_retry = chain.with_retry(stop_after_attempt=3)
chain_with_fallback = primary_chain.with_fallbacks([backup_chain])

# Async
import asyncio
result = await chain.ainvoke({"question": "Explain backpropagation"})

# Batch with concurrency
results = chain.batch([{"question": q} for q in questions], config={"max_concurrency": 5})
```

---

### 16.5 Output Parsers

```python
from langchain_core.output_parsers import (
    StrOutputParser, JsonOutputParser, PydanticOutputParser
)
from pydantic import BaseModel, Field
from typing import List

# Pydantic structured output (most robust)
class ModelEvaluation(BaseModel):
    model_name: str = Field(description="Name of the ML model")
    accuracy: float = Field(description="Accuracy score 0-1")
    pros: List[str] = Field(description="List of advantages")
    cons: List[str] = Field(description="List of disadvantages")
    recommendation: str = Field(description="Final recommendation")

parser = PydanticOutputParser(pydantic_object=ModelEvaluation)
format_instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You evaluate ML models. {format_instructions}"),
    ("human", "Evaluate: {model}"),
]).partial(format_instructions=format_instructions)

chain = prompt | chat | parser
result: ModelEvaluation = chain.invoke({"model": "Random Forest"})
print(f"Accuracy: {result.accuracy}, Pros: {result.pros}")
```

---

### 16.6 Document Loaders & Text Splitters

```python
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, WebBaseLoader,
    DirectoryLoader, WikipediaLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, TokenTextSplitter
)

# Load PDF
loader = PyPDFLoader("ml_paper.pdf")
docs = loader.load()  # list of Document(page_content=..., metadata={...})

# Load from web
loader = WebBaseLoader("https://arxiv.org/abs/1706.03762")
docs = loader.load()

# Recommended splitter: recursive (tries \n\n, \n, " ", "" in order)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # target chunk size in chars
    chunk_overlap=200,      # overlap between chunks (preserves boundary context)
    add_start_index=True,   # track position in original document
)
chunks = splitter.split_documents(docs)

# Token-based (for tight context windows)
splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)

# Semantic chunking (split at content boundaries — best quality)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
semantic_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95
)
```

**Chunking Strategy Theory:**
The optimal chunk size trades retrieval precision against context completeness. Small chunks (200-400 tokens) retrieve precise passages but miss multi-sentence context. Large chunks (1500-2000 tokens) provide more context but include irrelevant text. Chunk overlap (100-200 tokens) ensures sentences near boundaries aren't split across chunks. Semantic chunking keeps semantically coherent units together — best quality but computationally expensive.

---

### 16.7 Embeddings & Vector Stores

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# OpenAI
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # 3072-dim

# Open-source (strong performance)
hf_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

# Vector Stores
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

# FAISS (in-memory, great for prototyping)
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings)

# Chroma (persistent)
vectorstore = Chroma.from_documents(
    documents=chunks, embedding=embeddings,
    persist_directory="./chroma_db", collection_name="ml_papers"
)

# Search methods
results = vectorstore.similarity_search("attention mechanism", k=4)
results_with_scores = vectorstore.similarity_search_with_score("attention", k=4)

# MMR (Maximal Marginal Relevance) — balances relevance + diversity
results = vectorstore.max_marginal_relevance_search(
    "attention", k=4, fetch_k=20, lambda_mult=0.5
)

# As retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "lambda_mult": 0.6}
)
```

**Vector Search Theory — HNSW:**
HNSW (Hierarchical Navigable Small World) graphs underpin most production approximate nearest-neighbour search. HNSW builds a multi-layer graph: Layer 0 contains all nodes connected to K nearest neighbours; higher layers contain random subsets (skip list style). Search starts at the top layer (few nodes, fast traversal) and greedily descends, refining at each layer. The multi-scale structure gives O(log n) search complexity with near-exact recall — the graph structure mirrors the intrinsic geometry of the embedding space.

---

### 16.8 RAG — Retrieval-Augmented Generation

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─── Basic RAG Chain ───
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert assistant. Answer using ONLY the provided context.
If the answer isn't in the context, say "I don't have enough information."

Context:
{context}"""),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source','')}]\n{d.page_content}" for d in docs
    )

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt | chat | StrOutputParser()
)
answer = rag_chain.invoke("What are the main contributions of the Transformer?")

# ─── Hybrid Search (dense + sparse BM25) ───
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever], weights=[0.4, 0.6]
)

# ─── Contextual Compression ───
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings, similarity_threshold=0.75
)
compressed_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, base_retriever=retriever
)

# ─── Multi-Query Retrieval ───
from langchain.retrievers.multi_query import MultiQueryRetriever
# Generates 3 query variations → more complete retrieval
multi_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=chat)

# ─── Self-Query (metadata filtering) ───
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(name="source", description="Paper source", type="string"),
    AttributeInfo(name="year", description="Publication year", type="integer"),
]
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=chat, vectorstore=vectorstore,
    document_contents="ML research papers",
    metadata_field_info=metadata_field_info
)
# Parses "papers about attention from 2017" → filters year=2017 + semantic search
```

**RAG Failure Modes and Mitigations:**

| Failure Mode | Description | Mitigation |
|---|---|---|
| Retrieval failure | Correct document not retrieved | Hybrid search (dense + BM25), query expansion |
| Context ignored | LLM ignores retrieved context | System prompt engineering, context formatting |
| Hallucination | LLM fabricates despite good context | Faithfulness evaluation (RAGAS), constrained decoding |
| Context overflow | Too many chunks exceed limit | Contextual compression, chunk size tuning |
| Vocabulary mismatch | Query and document use different terms | Keyword/BM25 backup, query rewriting |

---

### 16.9 Memory & Conversation Management

```python
from langchain.memory import (
    ConversationBufferMemory,          # keep all history (grows unbounded)
    ConversationBufferWindowMemory,    # keep last k turns (bounded)
    ConversationSummaryMemory,         # LLM compresses old turns (scalable)
    ConversationSummaryBufferMemory,   # recent verbatim + summary of old
)

# Modern approach: RunnableWithMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}  # session_id → ChatMessageHistory
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
response = chain_with_history.invoke(
    {"input": "What is gradient boosting?"},
    config={"configurable": {"session_id": "user_123"}}
)
# Same session_id → persistent conversation history
```

---

### 16.10 Agents & Tools

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, PythonREPLTool

# Define custom tools
@tool
def calculate_accuracy(y_true: list, y_pred: list) -> float:
    """Calculate classification accuracy given true and predicted labels."""
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

@tool
def train_linear_regression(features: list, targets: list) -> dict:
    """Train a linear regression model and return coefficients."""
    import numpy as np
    X = np.column_stack([np.ones(len(features)), features])
    coeffs = np.linalg.lstsq(X, targets, rcond=None)[0]
    return {"intercept": float(coeffs[0]), "coefficients": coeffs[1:].tolist()}

tools = [DuckDuckGoSearchRun(), PythonREPLTool(), calculate_accuracy, train_linear_regression]

# Build agent
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert ML engineer. Use tools to solve problems."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm=chat, tools=tools, prompt=agent_prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True,
    max_iterations=10, handle_parsing_errors=True, return_intermediate_steps=True
)

result = agent_executor.invoke({
    "input": "Search for LLaMA-3 benchmarks and compare them numerically.",
    "chat_history": []
})
```

**ReAct (Reasoning + Acting) Pattern:**
```python
from langchain.agents import create_react_agent
from langchain import hub

react_prompt = hub.pull("hwchase17/react")  # standard ReAct template
react_agent = create_react_agent(chat, tools, react_prompt)
executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)
# Agent alternates: Thought → Action → Observation → Thought → ...
```

---

### 16.11 LangGraph — Stateful Agent Workflows

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, List
import operator

# Define state type
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]   # messages accumulate
    retrieved_docs: List[str]
    final_answer: str

# Define nodes
def retrieve_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    return {"retrieved_docs": [d.page_content for d in docs]}

def generate_node(state: AgentState) -> AgentState:
    context = "\n".join(state["retrieved_docs"])
    response = chat.invoke([
        SystemMessage(content=f"Answer based on: {context}"),
        state["messages"][-1]
    ])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_conditional_edges("generate", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "generate")

# Compile with persistent checkpointing
from langgraph.checkpoint.memory import MemorySaver
app = workflow.compile(checkpointer=MemorySaver())

# Run with session persistence
result = app.invoke(
    {"messages": [HumanMessage("What is a Transformer?")]},
    config={"configurable": {"thread_id": "session_1"}}
)
# Resume same session — state is preserved
result2 = app.invoke(
    {"messages": [HumanMessage("How does it compare to RNN?")]},
    config={"configurable": {"thread_id": "session_1"}}
)
```

---

### 16.12 Production RAG Pipeline

```python
"""
Production RAG System with:
- Hybrid retrieval (dense + sparse BM25)
- Contextual compression
- Conversation memory with session management
- Streaming output
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

class MLKnowledgeBot:
    def __init__(self, documents, model="gpt-4o"):
        self.llm = ChatOpenAI(model=model, temperature=0, streaming=True)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.session_store = {}

        # Build hybrid retriever
        vectorstore = Chroma.from_documents(documents, self.embeddings)
        bm25 = BM25Retriever.from_documents(documents, k=5)
        dense = vectorstore.as_retriever(search_kwargs={"k": 5})
        hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.3, 0.7])

        # Compress retrieved docs to only relevant portions
        compression = EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=0.7)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compression, base_retriever=hybrid
        )

        # Build conversational RAG chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert ML/DL assistant.
Answer based on retrieved context. Include equations where relevant.
State clearly if context is insufficient.

Retrieved Context:
{context}
---"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        self.chain = (
            {
                "context": lambda x: self._format_docs(self.retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "history": lambda x: x.get("history", []),
            }
            | prompt | self.llm | StrOutputParser()
        )

        self.chain_with_memory = RunnableWithMessageHistory(
            self.chain, self._get_session_history,
            input_messages_key="question", history_messages_key="history",
        )

    def _format_docs(self, docs):
        return "\n\n---\n\n".join(
            f"**Source:** {d.metadata.get('source', 'Unknown')}\n{d.page_content}"
            for d in docs
        )

    def _get_session_history(self, session_id):
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
        return self.session_store[session_id]

    def chat(self, question: str, session_id: str = "default"):
        for chunk in self.chain_with_memory.stream(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        ):
            print(chunk, end="", flush=True)
        print()

# Usage
bot = MLKnowledgeBot(documents=chunks)
bot.chat("What is the difference between LSTM and GRU?", session_id="user_1")
bot.chat("Which one should I use for NLP tasks?", session_id="user_1")  # remembers context
```

---

### 16.13 LangChain Evaluation & Monitoring

```python
# LangSmith — observability platform
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls-..."
os.environ["LANGCHAIN_PROJECT"] = "ml-chatbot-prod"
# All chain invocations are now traced in LangSmith dashboard

# RAGAS — RAG evaluation metrics
# pip install ragas
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

eval_data = {
    "question": ["What is backpropagation?"],
    "answer": [generated_answer],
    "contexts": [["backprop computes gradients..."]],
    "ground_truth": ["Backpropagation is..."]
}
scores = evaluate(Dataset.from_dict(eval_data),
                  metrics=[faithfulness, answer_relevancy, context_precision, context_recall])

# LLM-as-judge
from langchain.evaluation import load_evaluator
eval_chain = load_evaluator("qa", llm=chat)
result = eval_chain.evaluate_strings(
    input="What is the kernel trick in SVM?",
    prediction=predicted_answer,
    reference=ground_truth_answer
)
print(result["score"])  # 0-1

# RAGAS Metric Descriptions:
# faithfulness:       Is the answer supported by the retrieved context?
# answer_relevancy:   Is the answer relevant to the question?
# context_precision:  Is the retrieved context focused on the topic?
# context_recall:     Does the context contain enough information to answer?
```

---

## 17. Handling Imbalanced Data

### 17.1 Understanding Class Imbalance

```
Imbalance Ratio (IR) = N_majority / N_minority
IR > 10:   mild imbalance
IR > 100:  severe imbalance
IR > 1000: extreme imbalance (fraud detection, rare disease)
```

**Why Accuracy Fails:**
With 99% negative examples, always predicting negative gives 99% accuracy — without learning anything about the positive class. Standard ERM minimises average loss `(1/n)Σᵢ L(yᵢ, ŷᵢ)`, which weights all samples equally. The loss is dominated by the majority class. Use instead: F1, AUC-ROC, AUC-PR (precision-recall), MCC, G-mean.

```
G-mean = √(Sensitivity × Specificity) = √(TPR × TNR)
Balanced Accuracy = (TPR + TNR)/2
```

---

### 17.2 Resampling Methods

```
SMOTE:
  x_syn = xᵢ + λ(x_nn − xᵢ),   λ~U(0,1),   x_nn = random KNN of xᵢ
  Generates synthetic minority samples along line segments between neighbours

ADASYN:
  rᵢ = Δᵢ/K   (fraction of majority among K neighbours)
  nᵢ = ⌈G × rᵢ/Σᵢ rᵢ⌉   samples for point i
  More samples for harder-to-learn minority instances (near decision boundary)

Borderline-SMOTE:
  Only oversample minority instances near decision boundary ("DANGER" zone)
  Better than SMOTE for well-separated classes

SMOTE-ENN:  SMOTE + Edited Nearest Neighbours (remove ambiguous samples)
SMOTE-Tomek: SMOTE + remove Tomek links (majority member of each boundary pair)

NearMiss-1: keep majority samples whose mean dist to K nearest minority is smallest
NearMiss-2: keep majority samples whose mean dist to K farthest minority is smallest

Focal Loss:
  FL(p) = −α(1−p)^γ log(p)
  (1−p)^γ: down-weights easy/confident predictions → focuses on hard examples
  γ=2, α=0.25 (typical for object detection); widely used for imbalanced classification

Class Weights:
  wₖ = n / (K × nₖ)   (inverse frequency weighting)
  Pass as class_weight param in sklearn; sample_weight in loss function
```

**Why Focal Loss is Theoretically Superior to Resampling:**
Resampling changes the training data distribution — a model trained on oversampled data learns the wrong posterior probabilities. Focal loss keeps the data distribution unchanged but modifies the loss to focus on hard examples. The factor `(1−p)^γ` dynamically down-weights easy examples (the majority, usually easy) and up-weights hard ones (minority, near the decision boundary). The model still learns the true posterior probability — just more efficiently.

---

## 18. Model Evaluation & Selection

### 18.1 Classification Metrics

```
Confusion matrix: TP, FP, TN, FN

Accuracy      = (TP+TN)/(TP+TN+FP+FN)   — misleading for imbalanced classes
Precision     = TP/(TP+FP)               — of what I called positive, how many were?
Recall/TPR    = TP/(TP+FN)               — of all actual positives, how many did I find?
Specificity   = TN/(TN+FP)              — recall for negative class
F1            = 2PR/(P+R)               — harmonic mean of precision and recall
Fβ            = (1+β²)PR/(β²P+R)        — β>1: recall-weighted; β<1: precision-weighted

MCC = (TP·TN−FP·FN)/√[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]   ∈ [−1,1]
  Balanced metric; correctly handles all four confusion matrix quadrants simultaneously
  MCC=+1 only when ALL four rates are simultaneously good → most honest single metric

Cohen's κ = (p_o − p_e)/(1−p_e)
  Accounts for agreement by chance; κ>0.8: excellent, 0.6-0.8: good

AUC-ROC: area under TPR vs FPR curve
  P(score(+) > score(−)) for random positive/negative pair
  AUC = 0.5: random; AUC = 1: perfect; threshold-invariant

AUC-PR (better for imbalanced):
  Average Precision = Σₙ(Rₙ−Rₙ₋₁)Pₙ
  ROC can be optimistic when TN >> FP; PR curve is more informative
```

**Why MCC is the Most Honest Single Metric:**
Accuracy can be maximised by predicting only the majority class. F1 can be maximised without caring about true negatives. MCC = (TP·TN − FP·FN) / √[...] correctly handles all four quadrants simultaneously. Chicco & Jurman (2020) demonstrated MCC is more informative than F1 and accuracy for binary classification with imbalanced classes. Use MCC when you want one number that honestly represents classifier quality.

**Calibration:**
```
Reliability diagram: plot mean predicted probability vs actual fraction positive per bin
  Perfectly calibrated: diagonal line

ECE = Σ_bins (n_b/n)|acc_b − conf_b|    (Expected Calibration Error)
Brier score = (1/n)Σ(pᵢ−yᵢ)²            (lower = better; combines calibration + refinement)
```

---

### 18.2 Regression Metrics

```
MAE  = (1/n)Σ|yᵢ−ŷᵢ|           — robust to outliers; same unit as y
MSE  = (1/n)Σ(yᵢ−ŷᵢ)²           — penalises large errors quadratically
RMSE = √MSE                       — same unit as y
MAPE = (100/n)Σ|yᵢ−ŷᵢ|/|yᵢ|    — percentage error; undefined for yᵢ=0
MSLE = (1/n)Σ(log(ŷᵢ+1)−log(yᵢ+1))²   — for exponential-scale targets
R²   = 1−SS_res/SS_tot            — proportion of variance explained
Adjusted R² = 1−(1−R²)(n−1)/(n−p−1)

Huber loss (combines MSE and MAE):
  L_δ(y,ŷ) = (1/2)(y−ŷ)²  if |y−ŷ| ≤ δ;   δ|y−ŷ| − (1/2)δ² otherwise

Quantile loss (for prediction intervals):
  L_q(y,ŷ) = q(y−ŷ) if y≥ŷ,   (1−q)(ŷ−y) if y<ŷ
  Minimise L_0.05 and L_0.95 for 90% prediction interval
```

**Why MSE Penalises Outliers Disproportionately:**
MSE = `(1/n)Σ(yᵢ−ŷᵢ)²`. An error of 10 contributes 100; an error of 1 contributes 1 — a 10× error contributes 100× to the loss. Single large errors can dominate the training signal, distorting the model. MAE penalises linearly — much more robust. Huber loss interpolates: MAE for large errors (robust), MSE for small errors (smooth gradient at 0).

---

### 18.3 Cross-Validation Strategies

```
k-Fold:          split into k folds; train on k−1, test on 1; repeat k times
  Standard k: 5 or 10. Larger k → lower bias, higher variance, more computation.

Stratified k-Fold: preserve class distribution in each fold (essential for imbalanced)

Repeated k-Fold:  run k-fold r times with different seeds; more stable estimates

Nested CV (required for unbiased model selection):
  Outer: k₁-fold for unbiased generalisation estimate
  Inner: k₂-fold for hyperparameter selection within each outer train fold
  Prevents optimistic bias from tuning on test data

Group k-Fold:    groups (e.g., patients) don't appear in both train and test
  Prevents data leakage from correlated samples

Time Series Split:
  Fold 1: train [1..t₁], test [t₁+1..t₂]
  Fold 2: train [1..t₂], test [t₂+1..t₃]
  Always test on FUTURE data; add gap between train/test if autocorrelation exists
```

**Why Nested CV is Required for Unbiased Model Selection:**
In non-nested CV, the same data is used to (1) select hyperparameters and (2) estimate test performance. The selected hyperparameters are those that happened to work best on that validation fold — an optimistic estimate. Nested CV separates these: inner loop selects hyperparameters; outer loop estimates performance on held-out data never used for hyperparameter selection. Non-nested CV performance estimates are commonly 5-15% optimistic on small datasets.

---

### 18.4 Hyperparameter Tuning

```
Grid Search:  O(Πᵢ|Hᵢ|) evaluations — exponential in number of params
Random Search: O(n_trials); 60× more efficient when few params matter (Bergstra 2012)
  Key insight: most parameters are irrelevant — random search covers important
  dimensions more thoroughly than a grid

Bayesian Optimisation:
  Surrogate: GP models f(hp) → performance
  Acquisition: α(hp) = Expected Improvement or UCB
  x_next = argmax α(hp) → balance explore vs exploit
  Each evaluation updates GP prior; tools: Optuna, Hyperopt, W&B sweeps

Hyperband:
  Combine random search + successive halving
  Allocate small budget to many configs → halve resources, keep top → repeat
  Asymptotically optimal exploration-exploitation

BOHB (Bayesian Optimisation + Hyperband):
  Use BO to sample good configs; Hyperband for early stopping
  Strong practical performance (Auto-sklearn, SMAC)

Population-Based Training (PBT):
  Train population in parallel; copy top performers' weights + perturb hyperparams
  Used by DeepMind for RL and DL models
```

---

## 19. Regularisation & Optimisation

### 19.1 Norms & Regularisation Theory

```
Geometry of regularisation:
  Constrained form: min_β L(β)  s.t. ||β||_p ≤ t
  Lagrangian:       min_β L(β) + λ||β||_p

L0: exact sparsity (NP-hard to solve directly)
L1: convex relaxation of L0 → sparse solutions (Lasso)
L2: shrinks all toward 0; differentiable everywhere (Ridge)
Elastic Net: αL1 + (1−α)L2 → grouped sparsity

Why L1 is sparse — subdifferential argument:
  Stationarity at βⱼ=0: 0 ∈ ∂L/∂βⱼ + λ∂|0|
  ∂|0| = [−1, 1]  (subdifferential of absolute value at 0)
  Solution is 0 when |∂L/∂βⱼ| ≤ λ → soft-thresholding: β* = sign(β_LS)max(|β_LS|−λ,0)

Nuclear norm: ||W||_* = Σᵢ σᵢ  (convex relaxation of matrix rank)
  Used in matrix completion, multi-task learning

Spectral norm: ||W||_σ = σ_max (largest singular value)
  Spectral normalisation: W_SN = W/σ_max → ||W_SN||_σ = 1 → Lipschitz constraint
```

**Implicit Regularisation by SGD:**
Beyond explicit regularisation, SGD itself has an implicit effect. For overparameterised linear models, gradient descent from zero initialisation converges to the minimum-norm solution (equivalent to L2 regularisation with λ → 0⁺). For neural networks: large learning rate → stronger implicit regularisation, preference for flat minima. Small batch size → noisier gradients → more exploration → often better generalisation.

**Flat vs Sharp Minima:**
Two solutions with the same training loss may generalise differently. A flat minimum has a wide basin — small perturbations in parameter space still give low loss. A sharp minimum has a narrow basin — small perturbations increase loss greatly. Flat minima generalise better because if the test distribution differs slightly from train (inevitable), a flat minimum stays in a low-loss region. SGD with large learning rates preferentially finds flat minima — the gradient noise effectively bounces out of sharp narrow basins.

---

### 19.2 Normalisation Layers (detailed comparison)

```
         | Normalise over  | Batch-dependent | Use case
---------|-----------------|-----------------|------------------
Batch    | (N, H, W)/C     | Yes             | Conv nets; large batches
Layer    | (C, H, W)/sample| No              | Transformers; RNNs; any batch size
Instance | (H, W)/each C   | No              | Style transfer
Group    | (C//G, H, W)    | No              | Small batches; detection
RMS      | all features    | No              | LLaMA, Gemma (simpler than LN)
```

**Normalisation and Optimisation Landscape:**
Santurkar et al. (2018) showed BN makes the loss function smoother — smaller Lipschitz constant of gradients. The landscape has fewer sharp peaks and valleys, allowing larger learning rates, faster convergence, and reduced sensitivity to initialisation. The regularisation effect is secondary to this smoothing effect.

---

## 20. Probabilistic Machine Learning

### 20.1 Gaussian Processes

```
f ~ GP(m(x), k(x,x'))
  m(x): mean function (often m≡0)
  k(x,x'): kernel — encodes all structural assumptions about f

Posterior given noisy observations y = f(X) + ε, ε~N(0,σ²I):
  f*|X,y,X* ~ N(μ*, Σ*)
    μ* = K*ᵀ(K+σ²I)⁻¹y              (posterior mean)
    Σ* = K** − K*ᵀ(K+σ²I)⁻¹K*      (posterior variance = Schur complement!)

Marginal likelihood:
  log P(y|X,θ) = −(1/2)yᵀ(K+σ²I)⁻¹y − (1/2)log|K+σ²I| − (n/2)log(2π)
  Maximise over kernel hyperparams θ → automatic model selection

Common kernels:
  RBF:       k(x,x') = σ²exp(−||x−x'||²/(2l²))       (infinitely differentiable)
  Matérn-5/2: k = σ²(1+√5r/l+5r²/3l²)exp(−√5r/l)    (twice differentiable)
  Periodic:  k = σ²exp(−2sin²(π||x−x'||/p)/l²)       (seasonal data)
  Linear:    k(x,x') = σ₀² + σ²xᵀx'                   (Bayesian linear regression)

Complexity: O(n³) training; O(n²) prediction
Sparse GPs (inducing points): O(nm² + m³) with m << n
```

**GP as Distribution Over Functions:**
A GP defines a prior distribution over functions: `f ~ GP(m, k)` means that for any finite set of inputs `{x₁,...,xₙ}`, the function values are jointly Gaussian with the specified mean and covariance. The kernel encodes all structural assumptions: RBF kernel → infinitely differentiable functions; periodic kernel → seasonal patterns. The posterior variance `Σ* = K** − K*ᵀ(K+σ²I)⁻¹K*` is the Schur complement — it correctly captures that predictions near training data are confident while extrapolations are uncertain.

---

### 20.2 Hidden Markov Models (HMM)

```
λ = (A, B, π)
  A: transition matrix, Aᵢⱼ = P(zₜ=j|zₜ₋₁=i)
  B: emission matrix, Bᵢ(x) = P(xₜ=x|zₜ=i)
  π: initial distribution

Three algorithms:
  1. Forward (Evaluation): P(x₁,...,xT|λ) in O(TN²)
     α₁(i) = πᵢBᵢ(x₁)
     αₜ(j) = [Σᵢ αₜ₋₁(i) Aᵢⱼ] Bⱼ(xₜ)

  2. Viterbi (Decoding): most likely state sequence in O(TN²)
     δₜ(j) = max_i[δₜ₋₁(i)Aᵢⱼ] Bⱼ(xₜ)
     Backtrack via ψₜ(j) = argmax_i[δₜ₋₁(i)Aᵢⱼ]

  3. Baum-Welch (Learning): EM for HMM parameters
     E: compute γₜ(i,j) via forward-backward algorithm
     M: re-estimate A, B, π from expected counts
```

**Why Three Different Algorithms:**
1. **Evaluation** (Forward): sum over N^T possible state sequences naively. DP factorisaton → O(TN²).
2. **Decoding** (Viterbi): max over N^T paths — replace sum with max in forward recursion → O(TN²).
3. **Learning** (Baum-Welch): standard MLE is intractable (latent state sequence). EM: E-step computes posterior expectations, M-step updates parameters via expected sufficient statistics.

---

### 20.3 Calibration

```
A model is calibrated if: P(Y=1 | P̂=p) = p  for all p ∈ [0,1]

Reliability diagram: group predictions into B bins;
  plot mean prediction vs fraction positive per bin
  Diagonal = perfectly calibrated

Calibration metrics:
  ECE = Σ_b (nᵦ/n)|acc(b) − conf(b)|
  Brier score = (1/n)Σ(pᵢ−yᵢ)²

Calibration methods:
  Platt scaling:        P̂_cal = σ(aP̂ + b)   (fit logistic regression on val set)
  Isotonic regression:  non-parametric monotonic fit (more flexible, overfit risk on small val set)
  Temperature scaling:  P̂_cal = softmax(z/T)
    T>1: softer distribution (less confident); T<1: sharper
    T optimised on val set via NLL minimisation; single scalar — simple and effective
  Beta calibration:     fit Beta distribution to predictions
```

**Why Neural Networks are Overconfident:**
Guo et al. (2017) showed modern neural networks are significantly overconfident — predicted probabilities are much higher than empirical accuracies. Causes: (1) large capacity enables memorisation with near-perfect confidence, (2) cross-entropy rewards higher probabilities without bound, (3) BN and weight decay have unintended calibration effects. Temperature scaling is the simplest fix — it does not change the ranking of classes (same top-1 accuracy), only the confidence values.

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

### Foundational Textbooks

- Bishop, C. M. — *Pattern Recognition and Machine Learning* (2006)
- Goodfellow, Bengio, Courville — *Deep Learning* (deeplearningbook.org)
- Hastie, Tibshirani, Friedman — *Elements of Statistical Learning* (free PDF)
- Murphy, K. P. — *Probabilistic Machine Learning* (2022, free PDF)
- Géron, A. — *Hands-On ML with Scikit-Learn, Keras & TensorFlow* (3rd ed.)
- Nielsen, M. — *Neural Networks and Deep Learning* (neuralnetworksanddeeplearning.com)
- Szeliski, R. — *Computer Vision: Algorithms and Applications* (2nd ed.)

### Landmark Papers

- Vaswani et al. (2017) — *Attention Is All You Need* — Transformer architecture
- Devlin et al. (2018) — *BERT* — Bidirectional Encoder Representations
- Radford et al. (2018–2020) — *GPT-1, GPT-2, GPT-3*
- He et al. (2015) — *Deep Residual Learning (ResNet)*
- Dosovitskiy et al. (2020) — *An Image is Worth 16×16 Words (ViT)*
- Brown et al. (2020) — *Language Models are Few-Shot Learners (GPT-3)*
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
- Santurkar et al. (2018) — *How Does Batch Normalisation Help Optimisation?*
- Krogh & Vedelsby (1995) — *Neural Network Ensembles, Cross Validation, and Active Learning*
- Domingos & Pazzani (1997) — *On the Optimality of the Simple Bayesian Classifier*
- Wei et al. (2022) — *Emergent Abilities of Large Language Models*
- Chicco & Jurman (2020) — *The advantages of the Matthews Correlation Coefficient*

### Online Resources

- [Papers With Code](https://paperswithcode.com) — SOTA benchmarks with code
- [Hugging Face](https://huggingface.co) — Models, datasets, demos
- [LangChain Docs](https://docs.langchain.com)
- [LangSmith](https://smith.langchain.com) — LLM observability
- [Distill.pub](https://distill.pub) — Visual ML explanations
- [3Blue1Brown Neural Networks](https://www.3blue1brown.com/topics/neural-networks)
- [Andrej Karpathy's Blog](https://karpathy.github.io)
- [Jay Alammar's Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Scikit-learn](https://scikit-learn.org) | [PyTorch](https://pytorch.org) | [TensorFlow](https://tensorflow.org)

---

*Theory → Math → Code → Production*
*by @ravidaliparthy*
