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

**Applications:** PCA, LSA (NLP), collaborative filtering, pseudo-inverse, matrix completion.

#### Covariance and Correlation

```
Covariance matrix:   Σ = (1/(n-1)) Xᵀ X     (X is mean-centred, n×d)
Σᵢⱼ = Cov(Xᵢ, Xⱼ) = E[(Xᵢ - μᵢ)(Xⱼ - μⱼ)]

Correlation matrix:  R = D⁻¹/² Σ D⁻¹/²     (D = diag(Σ))
Rᵢⱼ = Σᵢⱼ / (σᵢ σⱼ)   ∈ [-1, 1]
```

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

#### Chain Rule

```
Scalar:   dz/dx = (dz/dy)(dy/dx)

Vector:   ∂z/∂x = Jᵀ_y→x · (∂z/∂y)     (Jacobian-vector product)

General (backprop):
  If z = f(g(x)):  ∂z/∂xᵢ = Σⱼ (∂z/∂yⱼ)(∂yⱼ/∂xᵢ)
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

**Strong Convexity (μ-strongly convex):**
```
f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (μ/2)||y-x||²
Linear convergence: ||xₜ - x*||² ≤ (1 - μ/L)ᵗ ||x₀ - x*||²
Condition number κ = L/μ — smaller → faster convergence
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

**Implications:**
- **High Bias:** Model too simple (e.g., linear model for non-linear data). Both train and test error are high.
- **High Variance:** Model too complex (e.g., deep tree). Low train error, high test error.
- **Optimal complexity:** Minimises total expected error. Found via cross-validation.

**Double Descent Phenomenon (modern DL):**
Beyond the interpolation threshold (zero training error), error can *decrease again* as model size grows — classical bias-variance doesn't fully explain deep learning behaviour.

---

### 2.3 The No Free Lunch Theorem

**Statement:** For any two algorithms A₁ and A₂, averaged over all possible data-generating distributions:
```
Σ_f E[L(f, A₁, n)] = Σ_f E[L(f, A₂, n)]
```

**Implication:** No learning algorithm is universally superior. Every algorithm has inductive biases — assumptions baked in about the hypothesis class. Good algorithm selection requires domain knowledge.

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

**Sauer-Shelah Lemma (growth function):**
```
mH(n) ≤ Σᵢ₌₀^d C(n,i)   where d = VCdim(H)
```
For n > d: `mH(n) ≤ (en/d)^d` — polynomial, not exponential.

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

**Full Bayesian Inference** (goes beyond MAP):
```
P(θ|X) = P(X|θ)P(θ) / P(X)          (exact posterior)
P(x_new|X) = ∫ P(x_new|θ) P(θ|X) dθ  (posterior predictive)
```
MAP = mode of posterior. Full Bayes = full posterior distribution.

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
    x'(λ) = [(x+1)λ-1]/λ           if λ≠0, x≥0
            log(x+1)                if λ=0, x≥0
           -[(-x+1)^(2-λ)-1]/(2-λ) if λ≠2, x<0
           -log(-x+1)               if λ=2, x<0
  λ estimated by maximising log-likelihood of normality

Unit Vector Normalisation (per-sample, used in NLP/cosine similarity):
  x' = x / ||x||₂   — each sample becomes a unit vector
```

---

### 3.3 Encoding Categorical Variables

```
Label Encoding: [Low, Med, High] → [0, 1, 2]
  ONLY for ordinal variables. Implies magnitude relationship.

One-Hot Encoding: [Red, Green, Blue] → 3 binary columns
  Use k-1 columns (drop first) to avoid multicollinearity in linear models.
  Creates high-dimensional sparse feature space for high-cardinality categories.

Ordinal Encoding: maps to meaningful integers preserving order
  e.g. [Cold, Warm, Hot] → [1, 2, 3]

Target Encoding (Mean Encoding):
  x_encoded = E[y | x = category]  (with regularisation/smoothing)
  Smoothed: x_enc = (n_c × ȳ_c + α × ȳ) / (n_c + α)
    n_c = count of category c, ȳ_c = mean target for c, ȳ = global mean, α = smoothing factor
  Risk: target leakage if not done within CV folds

Frequency/Count Encoding:
  x_encoded = count(category) / total_count
  Useful when frequency correlates with target

Binary Encoding:
  Label encode → convert to binary → split bits into columns
  e.g. category 5 → 101 → [1, 0, 1]
  Reduces dimensionality vs one-hot for high-cardinality features

Hashing (Feature Hashing / Hashing Trick):
  x_encoded[h(category) mod m] += 1
  Fixed-size output; handles unseen categories; risk of collisions

Embedding Encoding (learned):
  Map category c to dense vector e_c ∈ ℝᵈ via lookup table
  Trained end-to-end in neural network (e.g., word embeddings)
  Dimensionality rule of thumb: d = min(50, ceil(cardinality/2))

Leave-One-Out Encoding:
  x_enc_i = (Σⱼ≠ᵢ [xⱼ=xᵢ] yⱼ) / (count(xᵢ) - 1)
  Avoids target leakage without requiring separate folds
```

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
Forward Selection:
  Start with ∅; greedily add feature that most improves CV score. Stop when no improvement.
  Complexity: O(p²) model fits for p features.

Backward Elimination:
  Start with all p features; greedily remove worst. Stop when removing hurts CV score.

Recursive Feature Elimination (RFE):
  1. Fit model on all features
  2. Rank features by |coefficient| or feature importance
  3. Remove k least important features
  4. Repeat until desired number of features reached

Sequential Floating (SFFS/SBFS): adds backward step in forward (or vice versa) — better but slower.
```

**Embedded Methods (regularisation-based):**
```
L1 (Lasso):       Drives coefficients of irrelevant features to exactly 0.
L2 (Ridge):       Shrinks all — no feature elimination.
Elastic Net:      L1 + L2 — sparse but handles correlated groups.
Tree-based:       Gini/entropy impurity decrease summed over all splits for each feature.
SHAP:             Game-theory-based importance — consistent, locally accurate.
```

---

### 3.5 Outlier Detection

```
Z-Score:          z = (x - μ) / σ;   flag |z| > 3
Modified Z-Score: Mᵢ = 0.6745(xᵢ - x̃) / MAD;  flag |M| > 3.5   (robust to outliers)
IQR (Tukey):      Lower = Q1 - 1.5×IQR;  Upper = Q3 + 1.5×IQR

Isolation Forest:
  Anomaly Score = 2^(-E[h(x)] / c(n))
  h(x) = path length;  c(n) = 2H(n-1) - 2(n-1)/n   (normalisation)
  Anomalies are isolated with fewer splits → shorter average path.

Local Outlier Factor (LOF):
  reach_dist_k(x,o) = max(k-dist(o), d(x,o))
  lrd_k(x) = 1 / (Σ_{o ∈ Nₖ(x)} reach_dist_k(x,o) / |Nₖ(x)|)
  LOF_k(x) = avg_{o ∈ Nₖ(x)} lrd_k(o) / lrd_k(x)
  LOF >> 1 → outlier (much sparser than neighbourhood)

DBSCAN-based: points labelled as noise = outliers.

One-Class SVM:
  Learn a hypersphere in feature space that encloses "normal" data.
  Outliers fall outside the sphere.
  min_{R,c,ξ} R² + (1/νn) Σᵢ ξᵢ  s.t. ||xᵢ-c||² ≤ R² + ξᵢ
  ν ≈ fraction of outliers (upper bound on outlier fraction)

Autoencoder-based:
  Train AE on normal data → high reconstruction error = outlier.
  Threshold on reconstruction loss: ||x - AE(x)||² > τ
```

---

### 3.6 Correlation Analysis

```
Pearson:     r = Σᵢ(xᵢ-x̄)(yᵢ-ȳ) / √[Σᵢ(xᵢ-x̄)² Σᵢ(yᵢ-ȳ)²]   (linear)
Spearman:    ρ = 1 - 6Σᵢdᵢ² / (n(n²-1))                        (monotonic)
Kendall:     τ = (concordant - discordant) / C(n,2)              (ordinal)
Point-biserial: r_pb = (ȳ₁-ȳ₀)/sᵧ × √(n₁n₀/n²)               (continuous vs binary)
Cramér's V:  V = √(χ²/(n × min(r-1,c-1)))                       (categorical vs categorical)

VIF (detects multicollinearity):
  VIF_j = 1 / (1 - R²_j)    where R²_j = R² of regressing xⱼ on all other features
  VIF > 5: moderate; VIF > 10: severe multicollinearity → feature removal or PCA
```

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

**Numerical Stability:**
```
Via QR decomposition: X = QR → β* = R⁻¹ Qᵀy     (O(nd²), more stable)
Via SVD: X = UΣVᵀ → β* = V Σ⁺ Uᵀy               (handles rank deficiency)
```

**Gradient Descent:**
```
∇L = (2/n) Xᵀ(Xβ - y) = 2(XᵀXβ - Xᵀy)
β := β - α∇L
```

**Gauss-Markov Theorem (OLS is BLUE):**
Under assumptions:
1. `E[ε|X] = 0` (zero conditional mean)
2. `Var(ε|X) = σ²I` (homoscedasticity + no autocorrelation)
3. No perfect multicollinearity

OLS is the Best Linear Unbiased Estimator (BLUE).

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

**Effective Degrees of Freedom:**
```
df(λ) = tr(X(XᵀX+λI)⁻¹Xᵀ) = Σᵢ σᵢ²/(σᵢ²+λ)
```
λ=0 → df=p (full OLS); λ→∞ → df→0.

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

**Why L1 Gives Sparsity (geometric argument):**
The constraint region `{β: ||β||₁ ≤ t}` is a polytope (diamond in 2D) with corners on coordinate axes. Loss contours (ellipses) typically first touch this region at a corner → one coordinate = 0.

**Lasso Solution Path:** As λ decreases from λ_max to 0:
- λ_max = max_j |Xⱼᵀy|/n — all coefficients zero
- Features enter the model one by one
- LARS (Least Angle Regression) efficiently computes the full path in O(np²) time

---

### 4.4 Elastic Net

```
L(β) = ||y-Xβ||² + λ[α||β||₁ + (1-α)||β||²]
```

**Naïve Elastic Net:**
```
β_EN = (1+λ(1-α)) × argmin_β ||y-Xβ||² + λα||β||₁ + (λ(1-α)/2)||β||²
```

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

**More general basis functions:**
```
Radial Basis:    φⱼ(x) = exp(-||x-μⱼ||² / (2σ²))
Sigmoid:         φⱼ(x) = σ(wⱼᵀx + bⱼ)
Fourier:         φⱼ(x) = cos(ωⱼx + φⱼ)   — for periodic patterns
Splines (cubic): piecewise cubic polynomials with continuous 2nd derivatives at knots
```

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

---

## 5. Classification Algorithms

### 5.1 Logistic Regression

**Sigmoid & Log-Odds:**
```
σ(z) = 1/(1+e⁻ᶻ),   σ'(z) = σ(z)(1-σ(z))

Log-odds = logit(P) = log[P/(1-P)] = β₀ + β₁x₁ + ... = βᵀx
P(y=1|x) = σ(βᵀx) = 1/(1+exp(-βᵀx))
```

**Binary Cross-Entropy Loss (Negative Log-Likelihood):**
```
L = -(1/n) Σᵢ [yᵢ log σ(βᵀxᵢ) + (1-yᵢ) log(1-σ(βᵀxᵢ))]
```

**Why Cross-Entropy, Not MSE?**
- MSE with sigmoid creates non-convex loss → many local minima
- Cross-entropy is convex in β → guaranteed global minimum
- MLE justification: minimising cross-entropy = maximising likelihood

**Gradient:**
```
∂L/∂β = (1/n) Xᵀ(σ(Xβ) - y)   — elegant: same form as linear regression
```

**Hessian (for Newton's method):**
```
H = (1/n) XᵀWX,   W = diag[σ(xᵢ)(1-σ(xᵢ))]
β := β - H⁻¹∇L   (IRLS — Iteratively Reweighted Least Squares)
```
Newton's method converges in fewer iterations than GD but each step costs O(np²).

**Multinomial Logistic (Softmax):**
```
P(y=k|x) = exp(βₖᵀx) / Σⱼ exp(βⱼᵀx)

Softmax properties:
  - Σₖ P(y=k|x) = 1  (valid probability distribution)
  - Temperature: P(y=k|x,T) = exp(βₖᵀx/T) / Σⱼ exp(βⱼᵀx/T)
    T→0: argmax (deterministic); T→∞: uniform

Cross-entropy loss: L = -(1/n) Σᵢ Σₖ yᵢₖ log P(y=k|xᵢ)
```

---

### 5.2 K-Nearest Neighbours (KNN)

**Distance Metrics:**
```
Euclidean:  d_E = √Σᵢ(xᵢ-yᵢ)²
Manhattan:  d_M = Σᵢ|xᵢ-yᵢ|
Minkowski:  dₚ  = (Σᵢ|xᵢ-yᵢ|ᵖ)^(1/p)
Cosine:     d_cos = 1 - xᵀy/(||x|| ||y||)
Mahalanobis: d_mah = √((x-y)ᵀΣ⁻¹(x-y))   — accounts for feature correlations
Hamming:    count of differing positions    — for binary/categorical

Classification: ŷ = argmax_c Σ_{xⱼ∈Nₖ(x)} 𝟙[yⱼ=c]
Regression:     ŷ = (1/k) Σ_{xⱼ∈Nₖ(x)} yⱼ
Weighted KNN:   ŷ = Σⱼ (1/dⱼ) yⱼ / Σⱼ (1/dⱼ)
```

**Curse of Dimensionality:**
```
Volume of d-hypersphere / volume of d-hypercube = πᵈ/² / (Γ(d/2+1) × 2ᵈ) → 0

Consequence: in high-d, all points are approximately equidistant.
Ratio of max/min distance → 1 as d → ∞ → nearest neighbours meaningless.

For KNN to work: n must grow exponentially in d to maintain same neighbourhood density.
Practical remedy: dimensionality reduction (PCA, UMAP) before KNN.
```

**Complexity:**
```
Brute-force: O(nd) per query
KD-Tree: O(d log n) average, degrades for d > 20
Ball Tree: better for high-d and non-Euclidean metrics
Approximate NN (HNSW, FAISS): O(log n) — used in production vector databases
```

---

### 5.3 Naive Bayes

```
P(C|x) ∝ P(x|C) P(C) = P(C) Πⱼ P(xⱼ|C)   (naive independence)
ĉ = argmax_c [log P(C=c) + Σⱼ log P(xⱼ|C=c)]
```

**Variant Parameter Estimation:**
```
Gaussian NB:      P(xⱼ|C=c) = N(xⱼ; μⱼc, σ²ⱼc)
                  μⱼc = (Σᵢ∈c xᵢⱼ)/nᵢ,  σ²ⱼc = variance within class c

Multinomial NB:   P(xⱼ|C=c) = (N_cj + α)/(N_c + α|V|)   (Laplace smoothing)
  N_cj = count of word j in class c documents
  N_c = total word count in class c
  |V| = vocabulary size

Bernoulli NB:     P(xⱼ|C=c) = pⱼc^xⱼ (1-pⱼc)^(1-xⱼ)
  Penalises for absence of terms — often better for short texts

Complement NB:    Uses complement class statistics — reduces bias for imbalanced classes
  θ̂ⱼc = (α + Σᵢ∉c xᵢⱼ) / (α|V| + Σᵢ∉c Σⱼ xᵢⱼ)
```

**Calibration:** Despite often good accuracy, Naive Bayes probabilities are poorly calibrated — apply Platt scaling or isotonic regression if probabilities matter.

---

### 5.4 Decision Trees

**Splitting Criteria:**
```
Entropy:       H(S) = -Σₖ pₖ log₂ pₖ
Gini:          G(S) = 1 - Σₖ pₖ²
Misclass rate: E(S) = 1 - max_k pₖ
```

**Relationship:**
- All three are 0 when pure, maximal when uniform
- Entropy is most sensitive to impurity (sharper)
- Gini is computationally cheaper (no log)
- In practice: Gini ≈ Entropy results, both >> Misclassification rate

**Information Gain:**
```
IG(S, A) = H(S) - Σᵥ (|Sᵥ|/|S|) H(Sᵥ)
```

**Gain Ratio (C4.5 — prevents bias toward many-valued attributes):**
```
GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)
SplitInfo(S, A) = -Σᵥ (|Sᵥ|/|S|) log₂(|Sᵥ|/|S|)
```

**Regression Trees:**
```
Split: minimise Σᵥ (|Sᵥ|/|S|) Var(Sᵥ)
Leaf: ŷ = mean(y in leaf)
Regression impurity: MSE or MAE
```

**Cost-Complexity Pruning (α-pruning / CART):**
```
R_α(T) = R(T) + α|T|
```
- R(T) = weighted misclassification rate, |T| = number of leaves
- Increase α → prune subtrees where cost reduction < α per leaf
- Produces a sequence of nested trees T₀ ⊃ T₁ ⊃ ... ⊃ T₁ (root only)
- Choose optimal α by cross-validation

**CART Splitting Algorithm:**
```
For each feature j, for each threshold t:
  Gini_gain(j,t) = Gini(S) - (|S_L|/|S|)Gini(S_L) - (|S_R|/|S|)Gini(S_R)
Choose (j*, t*) = argmax_{j,t} Gini_gain(j,t)
```

**Tree Complexity:** A depth-d binary tree has at most 2^d leaves. Full tree with n points has depth log₂n.

---

### 5.5 Support Vector Machines (SVM)

**Hard-Margin:**
```
Minimise:   (1/2)||w||²
Subject to: yᵢ(wᵀxᵢ+b) ≥ 1   ∀i
```

**Margin:** γ = 2/||w||. Maximum margin ↔ minimum ||w||².

**Lagrangian & Dual:**
```
Primal: L = (1/2)||w||² - Σᵢ αᵢ[yᵢ(wᵀxᵢ+b)-1]
Dual:   max_α Σᵢαᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼxᵢᵀxⱼ
        s.t. αᵢ ≥ 0, Σᵢαᵢyᵢ = 0
Primal-dual: w* = Σᵢ αᵢyᵢxᵢ   (only support vectors with αᵢ > 0 contribute)
```

**KKT Conditions:**
```
αᵢ ≥ 0
yᵢ(wᵀxᵢ+b) - 1 ≥ 0
αᵢ[yᵢ(wᵀxᵢ+b) - 1] = 0  → αᵢ>0 only for support vectors on the margin
```

**Soft-Margin:**
```
Minimise:   (1/2)||w||² + C Σᵢ ξᵢ
Subject to: yᵢ(wᵀxᵢ+b) ≥ 1-ξᵢ,  ξᵢ ≥ 0
Dual:       max_α Σᵢαᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼxᵢᵀxⱼ
            s.t. 0 ≤ αᵢ ≤ C, Σᵢαᵢyᵢ = 0
```

**Hinge Loss Interpretation:**
```
L_hinge(y, f(x)) = max(0, 1-yf(x))
SVM primal = (λ/2)||w||² + (1/n)Σᵢ max(0, 1-yᵢwᵀxᵢ)
```

**Kernel Trick:**
```
Replace xᵢᵀxⱼ → K(xᵢ, xⱼ) = φ(xᵢ)ᵀφ(xⱼ)
f(x) = sign(Σᵢ αᵢyᵢ K(xᵢ,x) + b)

Kernels:
  Linear:     K(x,z) = xᵀz
  Polynomial: K(x,z) = (γxᵀz+r)ᵈ
  RBF/Gauss:  K(x,z) = exp(-γ||x-z||²) = exp(-||x-z||²/(2σ²))
  Laplace:    K(x,z) = exp(-γ||x-z||₁)
  Sigmoid:    K(x,z) = tanh(γxᵀz+r)   [not always PSD]
  Chi-squared: K(x,z) = 1 - Σᵢ(xᵢ-zᵢ)²/(xᵢ+zᵢ)  (for histograms)
  String:     K(s,t) = |{substrings shared by s and t}|  (for text/sequences)

Mercer's Theorem: K is a valid kernel iff the Gram matrix K_{ij} = K(xᵢ,xⱼ) is PSD.
```

---

### 5.6 Discriminant Analysis

**LDA (Shared Covariance):**
```
Assumption: P(x|C=k) = N(μₖ, Σ)   — same Σ for all classes

Linear discriminant function:
  δₖ(x) = xᵀΣ⁻¹μₖ - (1/2)μₖᵀΣ⁻¹μₖ + log πₖ
  ĉ = argmax_k δₖ(x)

Parameter estimation:
  π̂ₖ = nₖ/n
  μ̂ₖ = (1/nₖ) Σᵢ∈Cₖ xᵢ
  Σ̂ = (1/(n-K)) Σₖ Σᵢ∈Cₖ (xᵢ-μ̂ₖ)(xᵢ-μ̂ₖ)ᵀ  (pooled within-class covariance)
```

**LDA as Dimensionality Reduction:**
```
Maximise: J(W) = |WᵀS_B W| / |WᵀS_W W|   (Fisher's criterion)
S_W = Σₖ Σᵢ∈Cₖ (xᵢ-μₖ)(xᵢ-μₖ)ᵀ          (within-class scatter)
S_B = Σₖ nₖ(μₖ-μ)(μₖ-μ)ᵀ                (between-class scatter)
Solution: eigenvectors of S_W⁻¹S_B
Maximum discriminant directions: K-1 (for K classes)
```

**QDA (Class-Specific Covariance):**
```
Assumption: P(x|C=k) = N(μₖ, Σₖ)   — different Σₖ per class
Quadratic decision boundary:
  δₖ(x) = -(1/2)log|Σₖ| - (1/2)(x-μₖ)ᵀΣₖ⁻¹(x-μₖ) + log πₖ
More flexible than LDA; needs more data to estimate Σₖ reliably.
```

**Regularised Discriminant Analysis (RDA):**
```
Σₖ(α) = αΣₖ + (1-α)Σ   (interpolate between QDA and LDA)
α ∈ [0,1] tuned by cross-validation
```

---

## 6. Clustering & Unsupervised Learning

### 6.1 K-Means Clustering

**Objective:**
```
J = Σₖ Σᵢ∈Cₖ ||xᵢ-μₖ||²   (WCSS — Within-Cluster Sum of Squares)
```

**Lloyd's Algorithm:**
```
1. Initialise K centroids (random or K-means++)
2. Assignment: cᵢ = argmin_k ||xᵢ-μₖ||²
3. Update:     μₖ = (1/|Cₖ|) Σᵢ∈Cₖ xᵢ
4. Repeat until convergence (assignments don't change)
```

**K-means++ Initialisation:**
```
1. Pick x₁ uniformly at random from data
2. Compute D(x) = min_j d(x, xⱼ)² for all remaining points
3. Pick next centroid with P(x) ∝ D(x)
4. Repeat until K centroids chosen
Guarantee: E[WCSS] ≤ 8(ln K + 2) × WCSS_OPT
```

**Convergence Proof:** Each step (assignment + update) monotonically decreases J, and J is bounded below by 0 → convergence in finite steps. However, local optima are common — run multiple restarts.

**Complexity:** O(n·K·d·I) per iteration for I iterations. Typically converges in 10–100 iterations.

**Choosing K:**
```
Elbow method: plot WCSS vs K; choose inflection point
Silhouette: s(i) = (b(i)-a(i))/max(a(i),b(i))
  a(i) = mean intra-cluster dist;  b(i) = mean nearest-cluster dist
  s ∈ [-1,1]; average s > 0.5 = good structure
Gap Statistic: Gap(K) = E*[log WCSS_random] - log WCSS_data
  Choose K where Gap(K) - Gap(K+1) + se(K+1) ≥ 0
Davies-Bouldin Index = (1/K) Σₖ max_{k≠l} (σₖ+σₗ)/d(μₖ,μₗ)   (lower = better)
Calinski-Harabasz Index = tr(S_B)/(K-1) / tr(S_W)/(n-K)         (higher = better)
```

---

### 6.2 DBSCAN

```
Core Point:   |N_ε(p)| ≥ minPts   (≥ minPts neighbours within radius ε)
Border Point: not core, but within ε of a core point
Noise Point:  neither core nor border

Cluster: maximal set of density-connected points
Density-connected: p and q connected via chain of core points

Algorithm:
  For each unvisited point p:
    If |N_ε(p)| < minPts: label noise (tentative)
    Else: create new cluster, expand by density-reachability
  Border points assigned to nearest cluster

Complexity: O(n log n) with R*-tree indexing; O(n²) brute-force
```

**Choosing Parameters:**
```
minPts: rule of thumb = 2×d (d = dimensions); minimum 3
ε: k-distance plot — plot k-NN distances sorted ascending; choose at "elbow"
```

**HDBSCAN (Hierarchical DBSCAN):**
Builds a hierarchy over all density levels; extracts flat clustering via stability — more robust to varying density.

---

### 6.3 Hierarchical Clustering

**Agglomerative (bottom-up):**
```
Start: n clusters (each point its own)
At each step: merge two closest clusters according to linkage criterion
Result: dendrogram — cut at height h → clusters

Divisive (top-down): start with all points in one cluster; recursively split.
```

**Linkage Criteria:**
```
Single (MIN):    d(A,B) = min_{a∈A,b∈B} d(a,b)   — chaining effect; elongated clusters
Complete (MAX):  d(A,B) = max_{a∈A,b∈B} d(a,b)   — compact clusters; sensitive to outliers
Average (UPGMA): d(A,B) = (1/|A||B|) ΣΣ d(a,b)   — compromise; widely used
Ward:            d(A,B) = Δ variance when merging  — minimises within-cluster variance
  Ward update: d(A∪B,C) = √[(|A|+|C|)d²(A,C)+(|B|+|C|)d²(B,C)-|C|d²(A,B)] / √(|A|+|B|+|C|)
Centroid:        d(A,B) = ||μ_A - μ_B||            — can produce inversions in dendrogram
```

**Cophenetic Correlation:** Measures how faithfully dendrogram preserves pairwise distances. Good if > 0.75.

---

### 6.4 Gaussian Mixture Models (GMM)

```
P(x) = Σₖ πₖ N(x | μₖ, Σₖ),   Σₖ πₖ = 1

E-M Algorithm:
  E-step: γᵢₖ = πₖN(xᵢ|μₖ,Σₖ) / Σⱼ πⱼN(xᵢ|μⱼ,Σⱼ)   (soft assignment)
  M-step: Nₖ = Σᵢγᵢₖ
          μₖ = (1/Nₖ)Σᵢγᵢₖxᵢ
          Σₖ = (1/Nₖ)Σᵢγᵢₖ(xᵢ-μₖ)(xᵢ-μₖ)ᵀ
          πₖ = Nₖ/n
```

**EM Convergence:** Each iteration increases log-likelihood. Guaranteed convergence to a local maximum.

**Model Selection:** BIC = -2ℓ + k log n. Choose K minimising BIC.

**Covariance Constraints:**
```
Full:        each Σₖ unconstrained — most flexible, most parameters
Diagonal:    Σₖ = diag(σ²ₖ₁,...,σ²ₖd) — features independent within cluster
Spherical:   Σₖ = σ²ₖI — K-means with soft assignments (same as K-means when σ→0)
Tied Full:   all clusters share Σ (same as LDA structure)
```

---

## 7. Dimensionality Reduction

### 7.1 Principal Component Analysis (PCA)

**Objective:** Find orthogonal directions (principal components) that maximise variance.

**Algorithm:**
```
1. Centre: X ← X - μ   (subtract column means)
2. Covariance: C = (1/n)XᵀX   ∈ ℝᵈˣᵈ
3. Eigendecompose: C = QΛQᵀ   (Q = eigenvectors, Λ = diag eigenvalues)
4. Sort by λ₁ ≥ λ₂ ≥ ... ≥ λd
5. Project: Z = XQₖ   (n×k scores)
```

**Via SVD (numerically preferred):**
```
X = UΣVᵀ
Z = UₖΣₖ = XVₖ   (n×k)
Vₖ = first k columns of V = principal components (loadings)
```

**Variance Explained:**
```
PVE(k) = λₖ / Σᵢλᵢ = σₖ² / Σᵢσᵢ²
Cumulative PVE = Σⱼ≤ₖ λⱼ / Σᵢ λᵢ
Reconstruction: X̂ = ZVₖᵀ + μ
Reconstruction error: ||X - X̂||²_F = Σⱼ>ₖ λⱼ
```

**Whitening (Sphere Transform):**
```
Z_white = ZΛₖ⁻¹/² = XVₖΛₖ⁻¹/²
— decorrelates AND normalises to unit variance
— used before ICA, some neural network training
```

**PCA Limitations:**
- Linear only (see kernel PCA for non-linear)
- Sensitive to scaling → always standardise first
- Sensitive to outliers → use Robust PCA
- Components may not be interpretable

**Kernel PCA:**
```
Map data via kernel K; PCA in feature space φ(X)
Centred kernel: K̃ = K - 1ₙK - K1ₙ + 1ₙK1ₙ
Eigendecompose K̃ → project onto eigenvectors
```

---

### 7.2 t-SNE

```
High-d similarities:
  pⱼ|ᵢ = exp(-||xᵢ-xⱼ||²/(2σᵢ²)) / Σₖ≠ᵢ exp(-||xᵢ-xₖ||²/(2σᵢ²))
  pᵢⱼ = (pⱼ|ᵢ + pᵢ|ⱼ)/(2n)   (symmetrised)
  σᵢ set via binary search to match perplexity = 2^H(pᵢ)

Low-d similarities (Student-t, 1 dof):
  qᵢⱼ = (1+||yᵢ-yⱼ||²)⁻¹ / Σₖ≠ₗ(1+||yₖ-yₗ||²)⁻¹

Objective: min_{Y} KL(P||Q) = Σᵢⱼ pᵢⱼ log(pᵢⱼ/qᵢⱼ)

Gradient:
  ∂C/∂yᵢ = 4Σⱼ(pᵢⱼ-qᵢⱼ)(yᵢ-yⱼ)(1+||yᵢ-yⱼ||²)⁻¹
```

**Crowding Problem:** In d dimensions, moderate distances take many values; in 2D, all must be compressed. Student-t distribution (heavier tails than Gaussian) gives more space to moderate distances.

**Practical Notes:**
- Perplexity 5–50; typical choice 30
- Run 1000+ iterations with momentum
- Results are non-deterministic — run multiple times
- Distances/orientations between clusters NOT meaningful — only local structure preserved

---

### 7.3 UMAP

Based on Riemannian geometry and fuzzy simplicial sets.

**Core Idea:**
```
1. Construct a fuzzy topological representation in high-d:
   wᵢⱼ = exp(-(d(xᵢ,xⱼ) - ρᵢ)/σᵢ)   (membership strength)
   ρᵢ = distance to nearest neighbour of xᵢ

2. Cross-entropy between high-d and low-d fuzzy sets:
   L = Σᵢⱼ [wᵢⱼ log(wᵢⱼ/ŵᵢⱼ) + (1-wᵢⱼ)log((1-wᵢⱼ)/(1-ŵᵢⱼ))]
   where ŵᵢⱼ = (1 + a||yᵢ-yⱼ||^{2b})⁻¹   (a,b fit to min_dist parameter)

3. Optimise low-d positions {yᵢ} via SGD with negative sampling
```

**vs t-SNE:**
- Faster (O(n log n) vs O(n²))
- Preserves more global structure
- More deterministic
- Supports supervised/semi-supervised mode

---

## 8. Ensemble Learning

### 8.1 Why Ensembles Work

**Variance Reduction (Bagging):**
```
If B models each with variance σ² and pairwise correlation ρ:
  Var(average) = ρσ² + (1-ρ)σ²/B

As B→∞: Var → ρσ² (irreducible correlation)
Reducing ρ (model diversity) reduces variance more than adding more models.
```

**Bias-Variance of Ensemble:**
```
Bias(F̄) = Bias(f)   — same as individual
Var(F̄)  = ρσ² + (1-ρ)σ²/B  < σ²  if ρ < 1
```

**Condorcet's Jury Theorem:** If B independent classifiers each have P(correct) = p > 0.5:
```
P(majority correct) = Σⱼ>B/2 C(B,j) pʲ(1-p)^{B-j} → 1  as B → ∞
```

---

### 8.2 Bagging

```
Bootstrap sample: draw n points with replacement
P(xᵢ not in bag) = (1-1/n)ⁿ → e⁻¹ ≈ 0.368
∴ each bag contains ~63.2% unique samples

Aggregation:
  Regression:     ŷ = (1/B) Σb hb(x)
  Classification: ŷ = mode{hb(x)}   (majority vote)

Out-Of-Bag (OOB) Error — free internal validation:
  OOB estimate ≈ (1/n) Σᵢ L(yᵢ, ĥ_OOB(xᵢ))
  where ĥ_OOB(xᵢ) = average of trees that did NOT include xᵢ
  Asymptotically ≈ leave-one-out CV
```

---

### 8.3 Random Forest

```
At each node: subsample m features (not all p) before finding best split
Classification: m = √p
Regression:     m = p/3  or  max(1, floor(log₂p+1))

Benefits:
  1. Decorrelates trees (lower ρ → lower ensemble variance)
  2. Faster than bagging (fewer features evaluated per split)
  3. Robust to irrelevant features

Feature Importance — Mean Decrease Impurity (MDI):
  FI(j) = (1/B) Σb Σ_{t:split on j} pₜ × ΔImpurity(t)
  pₜ = fraction of samples reaching node t

Feature Importance — Permutation Importance (more reliable):
  PI(j) = mean increase in OOB error when feature j is randomly permuted
  Unaffected by feature scale; detects non-linear relationships

Feature Importance — SHAP (most reliable):
  φⱼ = Σ_{S⊆F\{j}} |S|!(|F|-|S|-1)!/|F|! × [v(S∪{j}) - v(S)]
  Unique: efficient, consistent, locally accurate
```

---

### 8.4 AdaBoost

```
Algorithm:
  1. wᵢ = 1/n for all i
  2. For t = 1,...,T:
     a. Train hₜ on weighted data
     b. εₜ = Σᵢ wᵢ 𝟙[hₜ(xᵢ) ≠ yᵢ]
     c. αₜ = (1/2)ln[(1-εₜ)/εₜ]   (weight for learner t)
     d. wᵢ ← wᵢ exp(-αₜ yᵢ hₜ(xᵢ)) / Zₜ   (normalise)
  3. H(x) = sign(Σₜ αₜhₜ(x))

Properties:
  αₜ > 0 iff εₜ < 0.5 (better than random)
  αₜ → ∞ as εₜ → 0 (near-perfect learner dominates)
  Misclassified points get higher weight → focus on hard examples

Training error bound:
  Train Error ≤ exp(-2Σₜ γₜ²),   γₜ = 0.5 - εₜ (edge)

AdaBoost minimises Exponential Loss: L(y,f) = exp(-yf(x))
Equivalent to forward stagewise additive modelling with exp loss.
```

---

### 8.5 Gradient Boosting

```
Fₘ(x) = Fₘ₋₁(x) + η hₘ(x)

Pseudo-residuals (negative gradient of loss):
  rᵢₘ = -∂L(yᵢ, F(xᵢ))/∂F(xᵢ)|_{F=Fₘ₋₁}

Key pseudo-residuals:
  MSE loss:      rᵢ = yᵢ - Fₘ₋₁(xᵢ)       (ordinary residuals)
  MAE loss:      rᵢ = sign(yᵢ - Fₘ₋₁(xᵢ))
  Log-loss:      rᵢ = yᵢ - σ(Fₘ₋₁(xᵢ))
  Huber loss:    rᵢ = δ sign(rᵢ) if |rᵢ|>δ else rᵢ   (robust to outliers)
  Quantile(α):   rᵢ = α if yᵢ>Fₘ₋₁ else α-1          (for prediction intervals)

Subsampling (Stochastic GBM):
  Randomly sample fraction s of data each iteration (s ≈ 0.5)
  Reduces correlation between trees; reduces overfitting; speeds training

Line Search (optimal step size):
  η_m = argmin_η Σᵢ L(yᵢ, Fₘ₋₁(xᵢ) + η hₘ(xᵢ))
```

---

### 8.6 XGBoost (Extreme Gradient Boosting)

```
Regularised objective:
  Obj = Σᵢ L(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)
  Ω(f) = γT + (λ/2)||w||²    (T = leaves, w = leaf scores)

Taylor expansion (2nd order):
  Obj^(t) ≈ Σᵢ [gᵢ fₜ(xᵢ) + (1/2)hᵢ fₜ²(xᵢ)] + Ω(fₜ) + const
  gᵢ = ∂L(yᵢ,ŷᵢ^(t-1))/∂ŷᵢ,   hᵢ = ∂²L/∂ŷᵢ²

Optimal leaf scores:
  wⱼ* = -Gⱼ/(Hⱼ+λ),   Gⱼ = Σᵢ∈Iⱼ gᵢ,   Hⱼ = Σᵢ∈Iⱼ hᵢ

Optimal split gain:
  Gain = (1/2)[Gₗ²/(Hₗ+λ) + Gᵣ²/(Hᵣ+λ) - (Gₗ+Gᵣ)²/(Hₗ+Hᵣ+λ)] - γ

Why 2nd order? More accurate curvature estimate → better step size → faster convergence.
Generalises to any twice-differentiable loss function.

Key innovations:
  - Sparsity-aware splits (handles missing natively via default direction)
  - Approximate greedy with weighted quantile sketch (scalable)
  - Column (feature) subsampling per tree/level/node
  - Cache-aware memory access (column blocks)
  - Out-of-core computing for datasets > RAM
```

---

### 8.7 LightGBM

```
Histogram-based splitting:
  Bin features into B = 256 discrete bins
  For each feature: accumulate G, H stats into histograms in O(n)
  Find best split: O(B) per feature → total O(Bd) per level
  Speedup vs XGBoost (exact): 8-16× on large datasets

GOSS (Gradient-based One-Side Sampling):
  Keep top-a% (large |gradient|) instances
  Sample b% from small-gradient instances, weight by (1-a)/b
  Preserves learning signal while reducing data

EFB (Exclusive Feature Bundling):
  Features that rarely have non-zero values simultaneously can be bundled
  Bundle = combine into single feature using offset trick
  Reduces effective feature count d' < d

Leaf-wise (best-first) growth:
  Always split leaf with max delta_loss (vs level-wise in most others)
  Grows deeper, asymmetric trees → lower loss in fewer leaves
  Risk: overfit on small datasets → use max_depth, min_data_in_leaf

Categorical feature handling:
  Optimal split: find partition of categories into two groups
  Uses Fisher's algorithm: sort by Σgᵢ/Σhᵢ, then greedy split → O(k log k)
```

---

### 8.8 CatBoost

```
Ordered target statistics (prevents target leakage):
  x̃ᵢ = (Σⱼ<σ(i) 𝟙[xⱼ=xᵢ]yⱼ + α·ȳ) / (Σⱼ<σ(i) 𝟙[xⱼ=xᵢ] + α)
  σ = random permutation; use only past observations → no leakage

Ordered boosting:
  M different models trained on different random orderings
  Prevents prediction shift (biased gradient estimation)

Symmetric trees (oblivious decision trees):
  All splits at the same depth use the same feature/threshold
  Tree = sequence of d (feature, threshold) pairs
  Evaluation: O(d) per sample (just d comparisons)
  Benefits: reduces overfit, fast inference, more interpretable

Handling categoricals natively:
  Automatic ordered statistics; handles high-cardinality well
  No need for manual encoding
```

---

### 8.9 Stacking & Blending

```
k-Fold Stacking (avoids leakage):
  For each fold k:
    Train base models on D \ fold_k
    Predict on fold_k → meta-features Z[fold_k, :]
  Stack all folds → meta-feature matrix Z (n × M)
  Train meta-learner: θ_meta = fit(Z, y)
  Inference: Z_test = (1/k)Σ predictions from each fold's model

Meta-learner choice:
  Simple: Ridge/Lasso (interpretable, regularised)
  Complex: GBM (can learn non-linear combos)
  Note: meta-learner trains on OOF (out-of-fold) predictions → fair evaluation

Blending (holdout variant):
  Split data: train (60%) → val (20%) → test (20%)
  Train base models on train
  Predict on val → meta-features
  Train meta-learner on val meta-features
  Inference: predict on test with base models + meta-learner
  Faster but wastes data; susceptible to val set overfit
```

---

## 9. Time Series Analysis

### 9.1 Stationarity

```
Strict stationarity: F(yₜ₁,...,yₜₖ) = F(yₜ₁+τ,...,yₜₖ+τ)  ∀k, τ
Weak stationarity:
  E[Yₜ] = μ           (constant mean)
  Var(Yₜ) = σ²        (constant variance, i.e., no heteroscedasticity)
  Cov(Yₜ,Yₜ₋ₖ) = γₖ  (covariance depends only on lag k, not t)

Achieving stationarity:
  First differencing:       ΔYₜ = Yₜ - Yₜ₋₁                (removes trend)
  Second differencing:      Δ²Yₜ = ΔYₜ - ΔYₜ₋₁              (removes quadratic trend)
  Seasonal differencing:    ΔₛYₜ = Yₜ - Yₜ₋ₛ                (removes seasonality)
  Log transform:            Y'ₜ = log Yₜ                       (stabilises variance)
  Log + differencing:       ΔlogYₜ = log(Yₜ/Yₜ₋₁) ≈ % change

Unit Root Tests:
  ADF (Augmented Dickey-Fuller): H₀: unit root (non-stationary)
    ADF test statistic = τ = β̂/(SE(β̂)) on ΔYₜ = α + βYₜ₋₁ + Σφⱼ ΔYₜ₋ⱼ + εₜ
  KPSS: H₀: stationarity (opposite to ADF)
  PP (Phillips-Perron): non-parametric correction for serial correlation
  Reject ADF H₀ AND fail to reject KPSS H₀ → strong evidence of stationarity
```

---

### 9.2 ACF and PACF

```
Autocovariance:  γₖ = Cov(Yₜ, Yₜ₋ₖ)
ACF:             ρₖ = γₖ/γ₀ = Corr(Yₜ, Yₜ₋ₖ)
PACF:            φₖₖ = Corr(Yₜ - P(Yₜ|Yₜ₋₁,...,Yₜ₋ₖ₊₁), Yₜ₋ₖ - P(Yₜ₋ₖ|Yₜ₋₁,...,Yₜ₋ₖ₊₁))
  (partial out intermediate lags)

95% confidence bounds for ACF: ±1.96/√n   (under H₀: white noise)

Signature identification:
  MA(q): ACF cuts off sharply after lag q; PACF decays geometrically
  AR(p): ACF decays geometrically; PACF cuts off sharply after lag p
  ARMA(p,q): both decay gradually; harder to identify directly
  Non-stationary: ACF decays very slowly (near unit root)
```

---

### 9.3 ARIMA(p,d,q)

```
AR(p): Yₜ = c + φ₁Yₜ₋₁ + ... + φₚYₜ₋ₚ + εₜ
MA(q): Yₜ = μ + εₜ + θ₁εₜ₋₁ + ... + θqεₜ₋q
ARIMA(p,d,q): Φ(B)(1-B)ᵈ Yₜ = c + Θ(B)εₜ

Invertibility (MA): roots of Θ(B) outside unit circle → MA can be written as infinite AR
Causality (AR):    roots of Φ(B) outside unit circle → stationary

Information criteria for order selection:
  AIC  = -2ℓ + 2k
  AICc = AIC + 2k(k+1)/(n-k-1)   (small-sample correction)
  BIC  = -2ℓ + k log n            (stronger complexity penalty)
  k = p + q + intercept; BIC tends to select simpler models
```

---

### 9.4 SARIMAX(p,d,q)(P,D,Q,s)

```
Full model:
  Φₚ(B)ΦP(Bˢ) ∇ᵈ ∇ˢᴰ Yₜ = c + Θq(B)ΘQ(Bˢ)εₜ + βXₜ

Seasonal AR: ΦP(Bˢ) = 1-Φ₁Bˢ-Φ₂B²ˢ-...-ΦₚBᴾˢ
Seasonal MA: ΘQ(Bˢ) = 1+Θ₁Bˢ+...+ΘQBQˢ
Seasonal differencing: ∇ˢ = (1-Bˢ)

Example: SARIMA(1,1,1)(1,1,1)₁₂ for monthly data:
  (1-φ₁B)(1-Φ₁B¹²)(1-B)(1-B¹²)Yₜ = (1+θ₁B)(1+Θ₁B¹²)εₜ

Prophet (Facebook): decomposable trend + seasonality + holidays
  y(t) = g(t) + s(t) + h(t) + εₜ
  g(t): piecewise linear or logistic trend with changepoints
  s(t): Fourier series for seasonality
  h(t): holiday effects (user-supplied)
```

---

### 9.5 Exponential Smoothing

```
SES:         ŷₜ₊₁ = α yₜ + (1-α) ŷₜ = α Σⱼ (1-α)ʲ yₜ₋ⱼ
Holt:        lₜ = α yₜ + (1-α)(lₜ₋₁+bₜ₋₁)
             bₜ = β(lₜ-lₜ₋₁) + (1-β)bₜ₋₁
             ŷₜ₊ₕ = lₜ + h bₜ
Holt-Winters (multiplicative):
             lₜ = α(yₜ/sₜ₋ₘ) + (1-α)(lₜ₋₁+bₜ₋₁)
             bₜ = β(lₜ-lₜ₋₁) + (1-β)bₜ₋₁
             sₜ = γ(yₜ/lₜ) + (1-γ)sₜ₋ₘ
             ŷₜ₊ₕ = (lₜ+hbₜ)sₜ₋ₘ₊h
ETS (Error-Trend-Seasonality):
  Unified state-space framework; AIC selects best combination
  Error: Additive/Multiplicative
  Trend: None/Additive/Additive Damped/Multiplicative
  Season: None/Additive/Multiplicative
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

**Universal Approximation Theorem (Cybenko, 1989; Hornik, 1991):**
A feedforward neural network with one hidden layer of N neurons and a non-polynomial activation function can approximate any continuous function f: Kⁿ → ℝ on a compact set K to arbitrary precision ε > 0, provided N is large enough.

**Depth vs Width:**
Deep networks can represent some functions exponentially more compactly than shallow ones (Montufar et al., 2014). A depth-L ReLU network of width n partitions input space into O((n/d)^{d(L-1)} × n^d) linear regions — exponential in depth.

---

### 10.2 Activation Functions

```
Sigmoid: σ(z) = 1/(1+e⁻ᶻ)           range (0,1)    σ'(z) = σ(z)(1-σ(z)) ≤ 0.25
Tanh:    tanh(z) = (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ)  range (-1,1)   tanh'(z) = 1-tanh²(z) ≤ 1
ReLU:    max(0,z)                      range [0,∞)    derivative = 𝟙[z>0] (dead for z<0)
Leaky:   max(αz,z), α≈0.01            range (-∞,∞)   αz for z<0 (prevents dying)
PReLU:   max(αz,z), α learnable
ELU:     z if z≥0; α(eᶻ-1) if z<0    negative saturation → mean activation ≈ 0
SELU:    λ ELU(z, α)                  self-normalising: preserves mean=0, var=1
GELU:    z Φ(z) ≈ 0.5z(1+tanh(√(2/π)(z+0.044715z³)))  used in GPT, BERT
Swish:   z·σ(βz), β learnable         smooth, non-monotone; used in EfficientNet
Mish:    z·tanh(softplus(z))          smooth unbounded above, bounded below
Softmax: exp(zₖ)/Σⱼ exp(zⱼ)          for multi-class output
Softplus: log(1+eᶻ)                   smooth approximation to ReLU

Choosing:
  Hidden layers: ReLU (default) → Leaky ReLU/ELU if dying ReLU problem → GELU for Transformers
  Output:        Sigmoid (binary), Softmax (multi-class), Linear (regression)
  Avoid:         Sigmoid/Tanh in deep networks (vanishing gradient)
```

---

### 10.3 Backpropagation — Full Derivation

**Error Signals:**
```
δᴸ = ∂L/∂zᴸ = (ŷ - y) ⊙ g'(zᴸ)   (output layer, MSE + linear: δᴸ = ŷ-y)
δˡ = ((Wˡ⁺¹)ᵀ δˡ⁺¹) ⊙ g'(zˡ)     (hidden layers)
```

**Parameter Gradients:**
```
∂L/∂Wˡ = δˡ (aˡ⁻¹)ᵀ    →  shape: [nˡ × nˡ⁻¹]
∂L/∂bˡ = δˡ              →  shape: [nˡ]
```

**Vanishing Gradient Analysis:**
```
||δˡ|| ≈ Π_{k=l}^{L-1} ||Wᵏ||·||g'(zᵏ)||

Sigmoid: max|g'| = 0.25. If ||W|| < 4 → product → 0 exponentially.
ReLU:    g'(z) = 1 for z>0. Product bounded by ||W||^(L-l) — only exploding concern.

Solutions:
  - Use ReLU/GELU (non-saturating)
  - Residual connections (gradient highway)
  - Batch/Layer normalisation
  - Gradient clipping (for exploding)
  - LSTM/GRU (for sequences)
  - Careful initialisation (Xavier, He)
```

---

### 10.4 Optimisation Algorithms

```
SGD:       θ := θ - α ∇L
Momentum:  v := βv + ∇L;   θ := θ - αv
NAG:       v := βv + ∇L(θ-βv);   θ := θ - αv   (look-ahead gradient)

AdaGrad:   G += (∇L)²;   θ := θ - (α/√(G+ε)) ∇L
  — per-parameter learning rate; good for sparse gradients; G grows → lr→0

RMSProp:   G := βG + (1-β)(∇L)²;   θ := θ - (α/√(G+ε)) ∇L
  — fixes AdaGrad's diminishing lr with exponential moving average

Adam:
  m := β₁m + (1-β₁)∇L          (1st moment)
  v := β₂v + (1-β₂)(∇L)²       (2nd moment)
  m̂ = m/(1-β₁ᵗ);  v̂ = v/(1-β₂ᵗ)  (bias correction — important at start)
  θ := θ - α m̂/(√v̂ + ε)
  Defaults: β₁=0.9, β₂=0.999, ε=1e-8, α=1e-3

AdamW:   θ := θ - α[m̂/(√v̂+ε) + λθ]
  Decouples weight decay from gradient adaptation
  Fixes L2 reg in Adam (they're not equivalent in adaptive methods)

Lion (2023):   m := β₁m + (1-β₁)∇L;   θ := θ - α(sign(m) + λθ)
  Uses sign update — memory-efficient, strong on LLMs

Learning Rate Schedules:
  Step:     α(t) = α₀ γ^{floor(t/k)}
  Cosine:   α(t) = α_min + (α_max-α_min)(1+cos(πt/T))/2
  1cycle:   increase to α_max, decrease to α_min/div_factor
  Linear warmup + cosine decay: standard for LLMs
    - Warmup: linearly increase α for first ~1000 steps
    - Prevents instability at start when m,v estimates are noisy
```

---

### 10.5 Weight Initialisation

```
Problem: without careful init, activations explode or vanish across layers.

Xavier/Glorot (Tanh/Sigmoid):
  Var(W) = 2/(n_in + n_out)
  W ~ U[-√(6/(n_in+n_out)), √(6/(n_in+n_out))]
  Derived by: keep Var(activation) ≈ 1 through forward AND backward pass

He/Kaiming (ReLU):
  Var(W) = 2/n_in
  W ~ N(0, 2/n_in)
  Factor 2: ReLU zeros half the activations → halves variance → compensate

LeCun (SELU):
  Var(W) = 1/n_in

Orthogonal Init:
  W = random orthogonal matrix (via QR decomposition)
  Preserves gradient norms in linear networks; used for RNNs

Sparse Init: small number of non-zero weights per unit — avoids symmetry breaking issues.
```

---

## 11. Convolutional Neural Networks (CNN)

### 11.1 The Convolution Operation

```
2D Cross-Correlation (called convolution in DL):
  (I ⋆ K)(i,j) = Σₘ Σₙ I(i+m, j+n) × K(m,n)
  True convolution (K flipped): (I * K)(i,j) = Σₘ Σₙ I(i-m, j-n)K(m,n)

Output shape:
  H_out = ⌊(H_in + 2P - D(K-1) - 1) / S⌋ + 1
  W_out = ⌊(W_in + 2P - D(K-1) - 1) / S⌋ + 1
  D = dilation rate; S = stride; P = padding; K = kernel size

Parameter count per conv layer:
  (K × K × C_in + 1) × C_out   (bias per filter)
  Compare FC: H_in × W_in × C_in × H_out × W_out × C_out — massive

Inductive biases of convolution:
  Translation equivariance: f(T(x)) = T(f(x))   (shift input → shift output)
  Local connectivity: each output depends only on a local region
  Weight sharing: same kernel applied everywhere → fewer params + regularisation
```

**Receptive Field:**
```
RF₁ = K₁
RFₗ = RFₗ₋₁ + (Kₗ-1) × Πⱼ<ₗ Sⱼ

For L conv layers (stride=1, no dilation): RF = L(K-1)+1
Dilated convolutions: effective K' = D(K-1)+1 → larger RF without extra params
```

**Convolution Types:**
```
Standard 2D conv:       K×K×C_in filters, C_out outputs
Depthwise:              K×K×1 per channel → C_in outputs (no cross-channel mixing)
Pointwise (1×1 conv):   1×1×C_in filters → mix channels, no spatial
Depthwise-separable:    Depthwise + Pointwise → 8-9× fewer params than standard
Transposed/deconv:      upsampling by reversing conv; used in U-Net decoder
Dilated/atrous conv:    insert zeros between kernel elements; larger RF, same params
Grouped conv:           split channels into g groups, convolve independently
```

---

### 11.2 Pooling & Normalisation

```
Max pooling:    y(i,j) = max over p×p window starting at (is,js)
Average pooling: y(i,j) = mean over p×p window
Global Average Pooling (GAP): y_c = mean of spatial dims for channel c
  Replaces FC layers; forces feature maps to be class activations; reduces overfit

Batch Normalisation:
  μ_B = (1/m)Σxᵢ;   σ²_B = (1/m)Σ(xᵢ-μ_B)²
  x̂ᵢ = (xᵢ-μ_B)/√(σ²_B+ε);   yᵢ = γx̂ᵢ + β
  At test: use running statistics (EMA over training batches)
  Benefits: reduces internal covariate shift, allows higher lr, mild regularisation

Layer Norm: normalise over feature dimension (not batch); used in Transformers
  μ = (1/d)Σxⱼ;   σ² = (1/d)Σ(xⱼ-μ)²;   x̂ = (x-μ)/√(σ²+ε)
  Independent of batch size — works for batch size = 1, sequences

Instance Norm: normalise per sample, per channel (style transfer)
Group Norm:    divide channels into G groups, normalise within each group
RMS Norm:      x̂ = x/RMS(x),  RMS(x) = √(Σxᵢ²/d)   — used in LLaMA, simpler than LayerNorm
```

---

### 11.3 CNN Architectures

```
AlexNet (2012):     5 conv + 3 FC; ReLU; Dropout; Local Response Norm; 60M params; 15.3% top-5 error
VGG (2014):         All 3×3 conv; deeper nets (16/19 layers); 138M params; very regular
  Two 3×3 convs: same RF as one 5×5 but 2(3²C²) vs 5²C² params (cheaper) + more non-linearity
GoogLeNet (2014):   Inception modules (parallel 1×1, 3×3, 5×5); 22 layers; 5M params
  1×1 bottleneck: reduce C_in before expensive 3×3, 5×5 → saves compute
ResNet (2015):      Skip connections; 50/101/152 layers; 3.57% top-5 error; SOTA
  y = F(x, {Wᵢ}) + x;   if optimal is identity: F→0, easier to learn than small perturbation
  Gradient: ∂L/∂x = ∂L/∂y(1 + ∂F/∂x); +1 prevents vanishing gradient
  Bottleneck block: 1×1 (reduce) → 3×3 (convolve) → 1×1 (expand)
DenseNet (2017):    xˡ = Hˡ([x⁰,x¹,...,xˡ⁻¹]); every layer connected to all subsequent
  Feature reuse; stronger gradient flow; fewer parameters than ResNet
EfficientNet (2019): Compound scaling: d=αᵠ, w=βᵠ, r=γᵠ; α·β²·γ²≈2
  Jointly scales depth/width/resolution; best accuracy/flops tradeoff
ViT (2020):         Patch-based image Transformer (see Section 15)
```

---

## 12. Recurrent Neural Networks (RNN)

### 12.1 Vanilla RNN

```
hₜ = tanh(Wₕ hₜ₋₁ + Wₓ xₜ + bₕ)   (hidden state)
ŷₜ = softmax(Wᵧ hₜ + bᵧ)            (output)

Computational graph: unrolled T steps → T copies of same parameters
Shared parameters: Wₕ, Wₓ, bₕ, Wᵧ, bᵧ (all time steps)
```

**BPTT (Backpropagation Through Time):**
```
∂L/∂Wₓ = Σₜ Σₖ≤ₜ (∂Lₜ/∂hₜ)(Πᵢ=k+1..t ∂hᵢ/∂hᵢ₋₁)(∂hₖ/∂Wₓ)
∂hᵢ/∂hᵢ₋₁ = diag(tanh'(zᵢ)) Wₕ

Vanishing: ||Wₕ||·max|tanh'| < 1 → gradient exponentially decreases with sequence length
Exploding: ||Wₕ|| > 1 → gradient exponentially grows → use gradient clipping:
  g := g × min(1, c/||g||)   (norm clipping, preserves direction)

Truncated BPTT: backpropagate only k steps at a time → O(k) memory, biased gradient
```

---

## 13. Long Short-Term Memory (LSTM) & GRU

### 13.1 LSTM

```
Forget gate:  fₜ = σ(Wf[hₜ₋₁;xₜ] + bf)        — how much to forget cell state
Input gate:   iₜ = σ(Wᵢ[hₜ₋₁;xₜ] + bᵢ)        — how much to write to cell
Candidate:    C̃ₜ = tanh(Wc[hₜ₋₁;xₜ] + bc)      — new candidate cell values
Cell update:  Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ        — element-wise: forget old + add new
Output gate:  oₜ = σ(Wo[hₜ₋₁;xₜ] + bo)         — what to expose from cell
Hidden state: hₜ = oₜ ⊙ tanh(Cₜ)               — filtered cell state

Gradient through cell: ∂Cₜ/∂Cₜ₋₁ = fₜ   (no matrix multiplication!)
If fₜ ≈ 1: gradient flows unchanged → no vanishing over long sequences
If fₜ ≈ 0: old information forgotten → selective memory

Parameter count: 4 × (nₕ(nₓ + nₕ) + nₕ)   (4 gates × Wₓ + Wₕ + b)
```

---

### 13.2 GRU

```
Reset gate:  rₜ = σ(Wr[hₜ₋₁;xₜ]+br)
Update gate: zₜ = σ(Wz[hₜ₋₁;xₜ]+bz)
Candidate:   h̃ₜ = tanh(Wh[rₜ⊙hₜ₋₁;xₜ]+bh)
Hidden:      hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ

zₜ ≈ 1: new information overrides old (update gate open)
zₜ ≈ 0: keep old hidden state (update gate closed)
rₜ ≈ 0: ignore past hidden state (reset to start fresh)

Parameter count: 3 × (nₕ(nₓ+nₕ) + nₕ)   (25% fewer than LSTM)
```

---

### 13.3 Sequence-to-Sequence & Attention

```
Encoder: {hₜ} = RNN(x₁,...,xT)   — source hidden states
Decoder: sₜ = RNN(sₜ₋₁, yₜ₋₁, cₜ);   ŷₜ = softmax(Wsₜ)

Bahdanau (Additive) Attention:
  eₜⱼ = vᵀ tanh(Wₛsₜ₋₁ + Wₕhⱼ)    (alignment score)
  αₜⱼ = exp(eₜⱼ) / Σⱼ' exp(eₜⱼ')  (softmax → weights)
  cₜ = Σⱼ αₜⱼ hⱼ                   (weighted context)

Luong (Multiplicative) Attention:
  eₜⱼ = sₜᵀ Wₐ hⱼ
  or dot: eₜⱼ = sₜᵀ hⱼ

Attention is O(TₛTₜ) — quadratic in sequence length pair.
Trains both encoder and decoder jointly via cross-entropy on target tokens.
```

---

## 14. Advanced Deep Learning

### 14.1 VAE & Generative Models

```
VAE ELBO:
  L = E_q[log p(x|z)] - KL(q(z|x) || p(z))
  = Reconstruction - Regularisation

Reparameterisation: z = μ(x) + σ(x)⊙ε,  ε~N(0,I)  → backprop through sampling

β-VAE: L = Reconstruction - β·KL,  β > 1 → better disentanglement

GAN:
  min_G max_D E[log D(x)] + E[log(1-D(G(z)))]
  D: real/fake discriminator;  G: generator from noise z~p(z)
  Gradient of G: E[log(1-D(G(z)))] saturates → use -E[log D(G(z))] instead
  Training instability: mode collapse, vanishing gradient
  Wasserstein GAN: use Wasserstein-1 distance → more stable gradients:
    min_G max_{||f||_L≤1} E[f(x)] - E[f(G(z))]
  Spectral normalisation: ||W||_σ = 1 → enforces Lipschitz constraint

Diffusion Models (DDPM):
  Forward: q(xₜ|xₜ₋₁) = N(√(1-βₜ)xₜ₋₁, βₜI)   (gradually add noise)
  Reverse: p_θ(xₜ₋₁|xₜ) = N(μ_θ(xₜ,t), Σ_θ(xₜ,t))   (denoise)
  Objective: ||ε - ε_θ(√ᾱₜx₀ + √(1-ᾱₜ)ε, t)||²   (predict noise)
  ᾱₜ = Πₛ≤ₜ(1-βₛ);  as t→T: xₜ ≈ N(0,I)
  DDIM: deterministic sampling → fewer steps, same quality
  Score matching: ∇_x log p_t(x) = -ε/√(1-ᾱₜ)
```

---

### 14.2 Regularisation

```
Dropout:
  During training: hᵢ = gᵢ × rᵢ,   rᵢ~Bernoulli(p)   (keep prob p)
  During test:     h = g   (no dropout; E[h]=p·g → scale by p at test OR /p at train)
  Inverted dropout (standard): divide by p during training → no test-time scaling

  Interpretation: ensemble of 2^n thinned networks; weight sharing
  Where: FC layers (p=0.5), conv (p=0.1-0.2), embedding (p=0.1)
  DropConnect: drop weights (not activations); stronger regularisation
  Spatial Dropout: drop entire feature maps → better for conv layers
  DropPath (Stochastic Depth): randomly drop entire residual blocks → regularises deep nets

Early Stopping:
  Monitor val loss; stop when val loss stops improving for patience steps
  Save checkpoint at best val loss
  Equivalent to L2 regularisation under some conditions

Data Augmentation:
  Images: horizontal flip, rotation (±10°), random crop, colour jitter,
          Mixup (xm=λxᵢ+(1-λ)xⱼ, ym=λyᵢ+(1-λ)yⱼ),
          CutMix (paste region from another image),
          AutoAugment (learned augmentation policy),
          RandAugment (simplified, equally good)
  Text:   synonym replacement, random insertion/deletion/swap, back-translation
          EDA (Easy Data Augmentation), SimCSE (contrastive)
  Tabular: SMOTE, Gaussian noise, feature subsampling

Label Smoothing:
  y_smooth = (1-ε)y_hard + ε/K
  Prevents overconfident logits; improves calibration; regularises softmax
  Gradient: encourages all logits to be similar (not just max)

Spectral Normalisation: W_SN = W/σ₁(W)  → Lipschitz constraint → stable training
```

---

## 15. 🔥 Transformers — Complete Deep Dive

### 15.1 Motivation — Why Transformers?

```
RNN limitations:
  - Sequential computation → cannot parallelise during training
  - Vanishing/exploding gradients over long sequences
  - Fixed-size context vector bottleneck (seq2seq)
  - O(n) sequential operations → slow

CNN limitations:
  - Receptive field grows linearly/logarithmically with depth
  - Long-range dependencies require many layers
  - Positional information must be encoded explicitly

Transformer key insight: replace sequential processing with
PARALLEL attention over all positions simultaneously.
O(n²d) attention but fully parallelisable → faster on modern hardware.
```

---

### 15.2 Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V

Shapes:
  Q: (seq_len_q × dₖ)   — queries (what am I looking for?)
  K: (seq_len_k × dₖ)   — keys   (what do I contain?)
  V: (seq_len_v × dᵥ)   — values (what do I return?)
  Output: (seq_len_q × dᵥ)

Score matrix: S = QKᵀ/√dₖ   shape: (seq_len_q × seq_len_k)
Attention weights: A = softmax(S, dim=-1)    — row-wise softmax
Output: Z = AV

Why √dₖ scaling?
  If q,k ~ N(0,1) iid, then qᵀk ~ N(0,dₖ)
  Without scaling: variance dₖ → large dot products → softmax saturates (near one-hot)
  → gradients ≈ 0 → no learning
  √dₖ normalises variance to 1

Masked Attention (for autoregressive generation):
  M = upper triangular matrix with -∞ (or -1e9)
  A = softmax((QKᵀ/√dₖ) + M)
  → each position can only attend to previous positions (causal mask)

Cross-Attention:
  Q from decoder, K and V from encoder output
  Queries (target) attend to keys and values (source)
```

---

### 15.3 Multi-Head Attention (MHA)

```
For h = 1,...,H heads:
  headₕ = Attention(QWₕᴼ, KWₕᴷ, VWₕᵛ)

  Wₕᴼ ∈ ℝ^{d_model × dₖ},   dₖ = dᵥ = d_model/H

MultiHead(Q,K,V) = Concat(head₁,...,headₕ)Wᵒ
Wᵒ ∈ ℝ^{H·dᵥ × d_model}

Why multiple heads?
  Each head can attend to different representation subspaces
  Head 1: syntactic (e.g., subject-verb agreement)
  Head 2: semantic (e.g., coreference)
  Head 3: positional (nearby tokens)
  Effectively: an ensemble of attention patterns

Parameter count (MHA):
  Wₕᴼ, Wₕᴷ, Wₕᵛ: H × 3 × d_model × (d_model/H) = 3 × d_model²
  Wᵒ: d_model × d_model
  Total: 4 × d_model²
```

---

### 15.4 Transformer Block

```
Pre-LN Transformer Block (used in GPT-2+, more stable training):
  x' = x + MultiHeadAttn(LayerNorm(x))
  x'' = x' + FFN(LayerNorm(x'))

Post-LN Transformer Block (original "Attention is All You Need"):
  x' = LayerNorm(x + MultiHeadAttn(x))
  x'' = LayerNorm(x' + FFN(x'))

Feed-Forward Network:
  FFN(x) = max(0, xW₁ + b₁)W₂ + b₂        (ReLU, original)
  FFN(x) = GELU(xW₁)W₂                      (GPT-2)
  FFN(x) = (SwiGLU(xW₁) ⊙ xW₃)W₂          (LLaMA, Mistral — SwiGLU gate)
  Typical d_ff = 4 × d_model
  Parameter count: 2 × d_model × d_ff = 8 × d_model²

Total params per layer (approx):
  MHA: 4d_model²   FFN: 8d_model²   LayerNorm: 4d_model
  ≈ 12d_model² per layer

L-layer Transformer:
  Total params ≈ L × 12d_model²  + embedding (V × d_model) + output head
```

---

### 15.5 Positional Encoding

```
Absolute (Sinusoidal — original):
  PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
  PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
  input = token_embedding + PE   (added, not concatenated)
  Property: PE(pos+k) can be expressed as linear function of PE(pos) → relative offsets

Learnable Absolute (BERT, GPT):
  E_pos ∈ ℝ^{max_len × d_model}   — learned embedding table
  Limited to max_len positions; does not extrapolate

Relative Position (Shaw et al., 2018):
  Score(q, k) = (q + a_{q,k})ᵀ(k + a_{q,k}) / √dₖ
  a_{q,k}: learnable relative position bias

Rotary Position Embedding (RoPE — LLaMA, GPT-NeoX):
  Rotate Q, K vectors by angle proportional to position:
  f(x, m) = x ⊗ cos(mθ) + rotate(x) ⊗ sin(mθ)
  Key property: (f(q,m))ᵀ(f(k,n)) depends only on (m-n) → true relative position
  Generalises to any sequence length; extrapolates well

ALiBi (Attention with Linear Biases — MPT):
  Score(i,j) = qᵢᵀkⱼ/√dₖ - m|i-j|    (subtract linear bias)
  m: head-specific slope; trained implicitly; extrapolates to longer sequences

YaRN, LongRoPE: extend RoPE for much longer contexts (e.g., 128k+ tokens).
```

---

### 15.6 Full Transformer Architecture

```
Encoder-Only (BERT family):
  Input: [CLS] + tokens + [SEP]
  Bidirectional: each token attends to all others
  Output: contextualised representations for each token
  Tasks: classification, NER, QA (span extraction)
  Training: MLM (Masked Language Modelling) + NSP (Next Sentence Prediction)

Decoder-Only (GPT family — most modern LLMs):
  Causal (masked) self-attention only
  Each token attends only to previous tokens
  Autoregressive generation: P(x) = Πₜ P(xₜ|x₁,...,xₜ₋₁)
  Training: CLM (Causal Language Modelling) — predict next token

Encoder-Decoder (T5, BART, original Transformer):
  Encoder: bidirectional (processes full source)
  Decoder: causal self-attn + cross-attn to encoder output
  Tasks: translation, summarisation, structured prediction
  Training: denoising (BART) or seq-to-seq (T5)
```

---

### 15.7 Pre-Training Objectives

```
Masked Language Modelling (MLM — BERT):
  Mask 15% of input tokens: 80% → [MASK], 10% → random, 10% → unchanged
  Predict original token at masked positions: L = -Σ log P(xᵢ|x̃)
  Bidirectional context; NOT autoregressive

Causal Language Modelling (CLM — GPT):
  L = -Σₜ log P(xₜ|x₁,...,xₜ₋₁) = -Σₜ log softmax(e_{xₜ}ᵀ h_t)
  Simple; highly scalable; enables zero-shot generation

Span Corruption (T5):
  Replace contiguous spans of tokens with sentinels [X], [Y], ...
  Decoder must reconstruct original spans
  More efficient: input shorter; output focuses on corrupted parts

Prefix LM (UniLM, GLM):
  Prefix: bidirectional attention (like BERT)
  Continuation: causal attention (like GPT)
  Best of both worlds for conditional generation

ELECTRA:
  Generator (small BERT): MLM replacement
  Discriminator (full model): predict replaced/original for ALL tokens
  More sample-efficient: supervise every token, not just 15%
```

---

### 15.8 Transformer Scaling Laws

**Chinchilla Scaling Laws (Hoffmann et al., 2022):**
```
Optimal training: N_params ~ N_tokens (parameters ≈ training tokens)
L(N, D) = E + A/N^α + B/D^β    (irreducible + model + data terms)
α ≈ β ≈ 0.5 (empirically)

Compute budget C ≈ 6ND (FLOPs)
Optimal: N* = G × C^a,  D* = G⁻¹ × C^b   (G ≈ 0.2, a ≈ b ≈ 0.5)
E.g., 70B parameters should train on ~1.4T tokens

Implications:
  Smaller models trained on more data often outperform larger under-trained models
  LLaMA: trained Chinchilla-optimally at each scale
  GPT-3 (175B) was significantly undertrained per Chinchilla
```

---

### 15.9 Efficient Attention

```
Problem: standard attention is O(n²d) in memory and time.

FlashAttention (Dao et al., 2022):
  Recompute attention softmax on-the-fly during backward pass (materialise only blocks)
  Tiling: Q,K,V split into blocks; process one block at a time in SRAM
  Memory: O(n) instead of O(n²); 2-4× faster wall-clock; numerically exact
  Trick: online softmax normalisation — max(m,m') and sum(ℓ,ℓ') updated per block

FlashAttention-2 (2023): better work partitioning, 2× faster than FA1
FlashAttention-3 (2024): H100-optimised with overlapped compute/memory

Sparse Attention:
  Local window: each token attends to w neighbours → O(nw)
  Strided: attend to every k-th token
  Global + local: [CLS] attends globally; others attend locally (Longformer)

Linear Attention (Katharopoulos et al., 2020):
  Replace softmax with kernel: K(q,k) ≈ φ(q)ᵀφ(k)
  Attention(Q,K,V) = φ(Q)(φ(K)ᵀV) — O(nd²) via associativity
  No softmax → loses expressive power but O(n) instead of O(n²)

Multi-Query Attention (MQA — Shazeer 2019):
  All heads share same K, V; only Q is separate
  Saves KV cache memory by H×; slight quality drop

Grouped Query Attention (GQA — Google 2023):
  G groups, each sharing K, V; H/G queries per group
  Interpolates between MHA (G=H) and MQA (G=1)
  LLaMA-2, Mistral use GQA — good speed/quality balance

Sliding Window Attention:
  Attend to w tokens in each direction → O(nw)
  Mistral-7B: window=4096, effective context >> window via stacking

Paged Attention (vLLM):
  KV cache managed in pages (like OS virtual memory)
  Enables efficient batching with variable-length sequences
```

---

### 15.10 Post-Training: Instruction Tuning & RLHF

```
Supervised Fine-Tuning (SFT):
  Fine-tune LLM on (instruction, response) pairs
  L = -Σₜ log P(yₜ|y₁,...,yₜ₋₁, x)   (only on response tokens)
  InstructGPT, Alpaca, Vicuna, etc.

RLHF (Reinforcement Learning from Human Feedback):
  1. SFT: fine-tune base model on demonstrations
  2. Reward Model: train r_θ to predict human preference
     - Collect pairs (y_w, y_l) where humans prefer y_w over y_l
     - L = -log σ(r_θ(x,y_w) - r_θ(x,y_l))   (Bradley-Terry model)
  3. RL (PPO): optimise policy π_θ via:
     reward = r_θ(x,y) - β KL(π_θ(y|x) || π_ref(y|x))
     KL term prevents policy from diverging too far from SFT model
     PPO: clip(π/π_old, 1-ε, 1+ε) — prevents large updates

DPO (Direct Preference Optimisation — Rafailov et al., 2023):
  Bypasses reward model; direct fine-tuning from preferences:
  L_DPO = -E[(y_w,y_l)] log σ(β log π_θ(y_w|x)/π_ref(y_w|x) - β log π_θ(y_l|x)/π_ref(y_l|x))
  Simpler than PPO; often competitive; widely adopted (Zephyr, Tulu)

Constitutional AI (Anthropic):
  Model critiques and revises its own responses based on principles
  Reduces need for human labelling of harmful content
  RLAIF: use AI feedback instead of human feedback

ORPO (Odds Ratio Preference Optimisation):
  Combines SFT and preference learning in single step
  No separate reference model needed

KTO (Kahneman-Tversky Optimisation):
  Uses unpaired feedback (good/bad labels, no comparison needed)
  Based on prospect theory — humans are loss-averse
```

---

### 15.11 Parameter-Efficient Fine-Tuning (PEFT)

```
Problem: fine-tuning 70B model needs 70B+ grad params = >500GB → infeasible.

LoRA (Low-Rank Adaptation — Hu et al., 2021):
  For a weight matrix W ∈ ℝ^{d×k}:
  ΔW = BA,   B ∈ ℝ^{d×r},  A ∈ ℝ^{r×k},  r << min(d,k)
  Forward: Wx + BAx = (W + BA)x
  Only A, B trained (r(d+k) << dk parameters)
  Init: A ~ N(0,σ²), B = 0 → ΔW=0 at start (preserves pre-trained weights)
  Typical r: 4-64; common: r=16
  Param savings: for d=k=4096, r=16: 2×4096×16/4096² = 0.78% of W params

QLoRA (Dettmers et al., 2023):
  4-bit quantisation of base model (NF4 — Normal Float 4)
  LoRA adapters in float16/bfloat16
  Double quantisation: quantise quantisation constants
  Fine-tune 65B model on single 48GB GPU (vs >1TB otherwise)

Prefix Tuning:
  Prepend K trainable "prefix" tokens to K and V at each layer
  Only prefix parameters trained (~0.1% of params)

Prompt Tuning:
  Prepend trainable "soft prompt" tokens to input embeddings only
  Simpler than prefix tuning; competitive for large models

Adapter Layers:
  Insert small bottleneck MLP (d → r → d) after attention and FFN
  Only adapter params trained; ~3-4% extra params

IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations):
  Rescale K, V, and FFN activations with learned vectors
  Very few params (<0.1%); strong performance on few-shot

(IA)³ vs LoRA: IA³ fewer params; LoRA more flexible; QLoRA most practical
```

---

### 15.12 Inference Optimisation & KV Cache

```
KV Cache:
  During autoregressive decoding, past K, V matrices are cached
  New token only needs to attend to cached K, V + new K, V
  Memory: 2 × L × H × dₖ × n_tokens × bytes_per_element
  For LLaMA-2-70B: 2×80×(4096)×8192×n bytes — grows linearly with sequence length

Quantisation:
  FP32 → FP16/BF16: 2× memory, minimal quality loss (standard for training)
  FP16 → INT8: 4× memory, small quality loss (LLM.int8(), SmoothQuant)
  INT8 → INT4: 8× memory, moderate quality loss (GPTQ, NF4, AWQ)
  GPTQ: minimize ||WX - ŴX||² greedily; optimal for each weight
  AWQ (Activation-aware): protect 1% salient weights (large activations); INT4 rest
  Bitsandbytes: in-place quantisation with mixed precision matmuls

Speculative Decoding:
  Small draft model generates k tokens quickly
  Large target model verifies all k tokens in parallel
  Accept tokens matching target's distribution; reject & resample otherwise
  Speedup: 2-3× with good draft model (same distribution)

Tensor Parallelism: split weight matrices across GPUs (column/row split)
Pipeline Parallelism: split layers across GPUs; micro-batching
Data Parallelism: replicate model; split batch

vLLM / PagedAttention:
  KV cache pages (blocks of tokens) managed like OS pages
  Non-contiguous allocation → less fragmentation
  High throughput serving (3-24× vs HuggingFace)
```

---

### 15.13 Notable LLM Architectures

```
GPT-2 (2019): 1.5B; decoder-only; byte-pair encoding; pre-LN; no RLHF
GPT-3 (2020): 175B; few-shot learning; emergent abilities
InstructGPT (2022): SFT + RLHF; follows instructions; 1.3B beats 175B base
ChatGPT (2022): GPT-3.5 + RLHF; conversational tuning
GPT-4 (2023): ~1T MoE; vision; stronger reasoning; RLHF + RLAIF
LLaMA-1 (2023): 7B-65B; open weights; competitive with GPT-3 at smaller scale
LLaMA-2 (2023): 7B-70B; GQA; 4k→2× context; open + chat versions
LLaMA-3 (2024): 8B-70B-405B; 128k context; tiktoken-style BPE; very strong
Mistral-7B (2023): sliding window attn; GQA; outperforms LLaMA-2-13B
Mixtral-8×7B (2023): Sparse MoE; 8 experts × 7B; activate 2/8; 47B params, 12B active
Claude (Anthropic): Constitutional AI; strong on long context and instruction following
Gemini (Google): native multimodal; 1M+ context (Ultra)
Falcon: RefinedWeb; multiquery attn; strong open model

MoE (Mixture of Experts):
  FFN replaced by E expert FFNs:
  y = Σₑ gₑ(x) FFNₑ(x)
  Router: g(x) = softmax(xWg); select top-K experts
  Expert capacity: fraction of tokens routed to each expert
  Load balancing loss: auxiliary loss to prevent expert collapse
  Sparse MoE: only top-2 experts activated per token → cheap inference
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

Patch embedding:
  xₚ = reshape(patch) × W_E,   W_E ∈ ℝ^{P²C × d}

Complexity: O(n²d) where n = HW/P²
  P=16, H=W=224: n = 14×14 = 196 patches

Variants:
  DeiT: data-efficient ViT (training tricks + distillation token)
  Swin Transformer: hierarchical; local shifted windows; O(n) not O(n²)
    - Window partition: attend within w×w windows
    - Shifted window: alternate between two partitions for cross-window connections
    - Patch merging: halve spatial dims, double channels (like pooling)
  BEiT: BERT-style masked image modelling pre-training
  MAE (Masked Autoencoder): mask 75% of patches; reconstruct pixels; very efficient
  DINO: self-supervised ViT; global + local crops; teacher-student with EMA
  CLIP: train image encoder + text encoder with contrastive loss on image-text pairs
    L = -Σᵢ log exp(τ·sᵢᵢ)/Σⱼ exp(τ·sᵢⱼ)   (image i matches text i)
    τ: learnable temperature; s = cosine similarity
    Enables zero-shot image classification via text prompts
```

---

## 16. 🦜 LangChain & LLM Application Engineering

### 16.1 LangChain Overview

```
LangChain is a framework for building applications with LLMs.
Core abstractions:
  Model I/O:     prompt templates, LLM/ChatModel wrappers, output parsers
  Data:          document loaders, text splitters, vector stores, retrievers
  Chains:        sequences of components (LCEL)
  Agents:        LLMs that decide which tools to use
  Memory:        persist state across interactions
  Callbacks:     hooks for logging, streaming, monitoring

Installation:
  pip install langchain langchain-openai langchain-community
  pip install langchain-anthropic langchain-google-genai   # other providers
```

---

### 16.2 LLM & Chat Model Wrappers

```python
# Basic LLM (completion-style)
from langchain_openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7)
result = llm.invoke("What is the capital of France?")

# Chat Model (message-based)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

chat = ChatOpenAI(model="gpt-4o", temperature=0)
messages = [
    SystemMessage(content="You are an expert data scientist."),
    HumanMessage(content="Explain gradient descent in 3 bullet points.")
]
response = chat.invoke(messages)
print(response.content)

# Streaming
for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)

# Anthropic (Claude)
from langchain_anthropic import ChatAnthropic
claude = ChatAnthropic(model="claude-opus-4-6", temperature=0)

# Local models via Ollama
from langchain_community.llms import Ollama
local_llm = Ollama(model="llama3")
```

---

### 16.3 Prompt Templates

```python
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder
)

# Simple template
prompt = PromptTemplate(
    input_variables=["topic", "audience"],
    template="Explain {topic} to a {audience} in simple terms."
)
formatted = prompt.format(topic="backpropagation", audience="10-year-old")

# Chat prompt template (recommended for chat models)
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer in {language}."),
    ("human", "{question}"),
])
messages = chat_prompt.format_messages(
    role="machine learning professor",
    language="English",
    question="What is overfitting?"
)

# With conversation history placeholder
conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Few-shot prompt
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "3×5", "output": "15"},
]
example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Solve the arithmetic problem:",
    suffix="Input: {query}\nOutput:",
    input_variables=["query"]
)
```

---

### 16.4 LCEL — LangChain Expression Language

```python
# LCEL: pipe operator | chains components
# All components implement Runnable interface: invoke, stream, batch, ainvoke

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Basic chain: prompt | model | parser
chain = chat_prompt | chat | StrOutputParser()
result = chain.invoke({"role": "tutor", "language": "English", "question": "What is SVM?"})

# Parallel execution
parallel = RunnableParallel(
    summary=prompt_summary | chat | StrOutputParser(),
    keywords=prompt_keywords | chat | StrOutputParser(),
)
results = parallel.invoke({"text": "Long document..."})

# Branching with RunnableLambda
from langchain_core.runnables import RunnableLambda

def route(info):
    if "urgent" in info["question"].lower():
        return priority_chain
    return standard_chain

full_chain = {"question": RunnablePassthrough()} | RunnableLambda(route)

# Retry and fallback
chain_with_retry = chain.with_retry(stop_after_attempt=3)
chain_with_fallback = primary_chain.with_fallbacks([backup_chain])

# Batching
results = chain.batch([{"question": q} for q in questions], config={"max_concurrency": 5})

# Async
import asyncio
result = await chain.ainvoke({"question": "..."})
```

---

### 16.5 Output Parsers

```python
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    CommaSeparatedListOutputParser
)
from pydantic import BaseModel, Field
from typing import List

# String parser (most common)
str_parser = StrOutputParser()

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

# JSON output parser
json_parser = JsonOutputParser()
chain = prompt | chat | json_parser   # returns dict

# Custom parser with RunnableLambda
from langchain_core.runnables import RunnableLambda
extract_numbers = RunnableLambda(lambda x: [float(n) for n in re.findall(r'\d+\.?\d*', x)])
chain = prompt | chat | StrOutputParser() | extract_numbers
```

---

### 16.6 Document Loaders & Text Splitters

```python
# Document Loaders
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, WebBaseLoader,
    DirectoryLoader, UnstructuredHTMLLoader,
    WikipediaLoader, ArxivLoader
)

# Load PDF
loader = PyPDFLoader("ml_paper.pdf")
docs = loader.load()  # list of Document(page_content=..., metadata={...})

# Load directory of files
loader = DirectoryLoader("./docs/", glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()

# Web scraping
loader = WebBaseLoader("https://arxiv.org/abs/1706.03762")
docs = loader.load()

# Text Splitters
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    SentenceTransformersTokenTextSplitter
)

# Recommended: recursive (tries \n\n, \n, " ", "" in order)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,         # target chunk size in chars
    chunk_overlap=200,       # overlap between chunks (preserves context)
    length_function=len,     # char-based; use tiktoken for token-based
    add_start_index=True,    # track position in original doc
)
chunks = splitter.split_documents(docs)

# Token-based (for tight context windows)
splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)

# Semantic chunking (split on meaning change)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
semantic_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",   # or "standard_deviation"
    breakpoint_threshold_amount=95
)
```

---

### 16.7 Embeddings & Vector Stores

```python
# Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import (
    HuggingFaceEmbeddings, OllamaEmbeddings
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # 3072-dim
hf_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",  # strong open-source model
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

# Vector Stores
from langchain_community.vectorstores import (
    FAISS, Chroma, Pinecone, Weaviate, Milvus, Qdrant
)

# FAISS (in-memory, great for prototyping)
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings)

# Chroma (persistent, easy setup)
from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="ml_papers"
)

# Similarity search
results = vectorstore.similarity_search("What is attention mechanism?", k=4)
results_with_scores = vectorstore.similarity_search_with_score("attention", k=4)

# MMR (Maximal Marginal Relevance) — balances relevance + diversity
results = vectorstore.max_marginal_relevance_search("attention", k=4, fetch_k=20, lambda_mult=0.5)

# As retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",                    # or "similarity", "similarity_score_threshold"
    search_kwargs={"k": 5, "lambda_mult": 0.6}
)
```

---

### 16.8 RAG — Retrieval-Augmented Generation

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─── Basic RAG chain ───
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert assistant. Answer using ONLY the provided context.
If the answer isn't in the context, say "I don't have enough information."

Context:
{context}"""),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(f"[Source: {d.metadata.get('source','')}]\n{d.page_content}" for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | chat
    | StrOutputParser()
)
answer = rag_chain.invoke("What are the main contributions of the Transformer architecture?")

# ─── Advanced RAG: with citations ───
from langchain.chains import RetrievalQAWithSourcesChain

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# ─── Hybrid Search (dense + sparse) ───
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6]   # BM25 40%, semantic 60%
)

# ─── Contextual Compression ───
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter

# LLM-based: extract relevant portions
compressor = LLMChainExtractor.from_llm(chat)
# Embedding-based (cheaper): filter by similarity to query
embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)

compressed_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=retriever
)

# ─── Multi-Query Retrieval ───
from langchain.retrievers.multi_query import MultiQueryRetriever
multi_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=chat)
# Generates 3 query variations → more complete retrieval

# ─── Self-Query Retrieval (metadata filtering) ───
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

---

### 16.9 Memory & Conversation Management

```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    VectorStoreRetrieverMemory
)

# Buffer (keep all history — grows unbounded)
memory = ConversationBufferMemory(return_messages=True, memory_key="history")

# Window (keep last k turns — bounded)
memory = ConversationBufferWindowMemory(k=5, return_messages=True)

# Summary (LLM compresses old turns — scalable)
memory = ConversationSummaryMemory(llm=chat, return_messages=True)

# Summary + Buffer (recent turns verbatim + summary of old ones)
memory = ConversationSummaryBufferMemory(
    llm=chat, max_token_limit=2000, return_messages=True
)

# Vector store memory (semantic retrieval of relevant past context)
memory = VectorStoreRetrieverMemory(retriever=vectorstore.as_retriever(k=3))

# ─── Modern approach: use RunnableWithMessageHistory ───
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
```

---

### 16.10 Agents & Tools

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool, StructuredTool
from langchain_community.tools import (
    DuckDuckGoSearchRun, WikipediaQueryRun, PythonREPLTool
)

# ─── Define Custom Tools ───
@tool
def calculate_accuracy(y_true: list, y_pred: list) -> float:
    """Calculate classification accuracy given true and predicted labels."""
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)

@tool
def train_linear_regression(features: list, targets: list) -> dict:
    """Train a linear regression model and return coefficients."""
    import numpy as np
    X = np.array(features)
    y = np.array(targets)
    coeffs = np.linalg.lstsq(
        np.column_stack([np.ones(len(X)), X]), y, rcond=None
    )[0]
    return {"intercept": float(coeffs[0]), "coefficients": coeffs[1:].tolist()}

# ─── Agent with Tools ───
tools = [
    DuckDuckGoSearchRun(),
    PythonREPLTool(),
    calculate_accuracy,
    train_linear_regression,
]

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert ML engineer. Use tools to solve problems."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm=chat, tools=tools, prompt=agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

result = agent_executor.invoke({
    "input": "Search for the latest benchmarks of LLaMA-3 and compare them numerically.",
    "chat_history": []
})

# ─── ReAct Agent (Reasoning + Acting) ───
from langchain.agents import create_react_agent
from langchain import hub
react_prompt = hub.pull("hwchase17/react")  # standard ReAct template
react_agent = create_react_agent(chat, tools, react_prompt)
executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)

# ─── Structured Chat Agent ───
from langchain.agents import create_structured_chat_agent
structured_prompt = hub.pull("hwchase17/structured-chat-agent")
structured_agent = create_structured_chat_agent(chat, tools, structured_prompt)
```

---

### 16.11 LangGraph — Stateful Agent Workflows

```python
# LangGraph: build controllable, stateful multi-agent systems
# pip install langgraph

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, List
import operator

# ─── Define State ───
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]   # messages accumulate
    retrieved_docs: List[str]
    final_answer: str
    iteration: int

# ─── Define Nodes ───
def retrieve_node(state: AgentState) -> AgentState:
    """RAG retrieval node."""
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    return {"retrieved_docs": [d.page_content for d in docs]}

def generate_node(state: AgentState) -> AgentState:
    """LLM generation node."""
    context = "\n".join(state["retrieved_docs"])
    response = chat.invoke([
        SystemMessage(content=f"Answer based on: {context}"),
        state["messages"][-1]
    ])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """Routing function."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END

# ─── Build Graph ───
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_conditional_edges("generate", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "generate")

app = workflow.compile()
result = app.invoke({"messages": [HumanMessage(content="Explain BERT's pre-training")]})

# ─── Checkpointing (persistent state) ───
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
result = app.invoke(
    {"messages": [HumanMessage("What is a Transformer?")]},
    config={"configurable": {"thread_id": "session_1"}}
)
# Resume conversation — state is preserved
result2 = app.invoke(
    {"messages": [HumanMessage("How does it compare to RNN?")]},
    config={"configurable": {"thread_id": "session_1"}}
)
```

---

### 16.12 Building a Production ML Chatbot Pipeline

```python
"""
Production RAG System with:
- Hybrid retrieval (dense + sparse)
- Contextual compression
- Conversation memory
- Structured output
- Streaming
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

        # Compress retrieved docs
        compression = EmbeddingsFilter(
            embeddings=self.embeddings, similarity_threshold=0.7
        )
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compression, base_retriever=hybrid
        )

        # Build chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert ML/DL assistant.
Answer based on the retrieved context. Be precise and include equations where relevant.
If context is insufficient, state that clearly.

Retrieved Context:
{context}

---"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        self.chain = (
            {
                "context": lambda x: self._format_docs(
                    self.retriever.invoke(x["question"])
                ),
                "question": lambda x: x["question"],
                "history": lambda x: x.get("history", []),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        self.chain_with_memory = RunnableWithMessageHistory(
            self.chain,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="history",
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
        """Stream response."""
        for chunk in self.chain_with_memory.stream(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        ):
            print(chunk, end="", flush=True)
        print()

# Usage
bot = MLKnowledgeBot(documents=chunks)
bot.chat("What is the difference between LSTM and GRU?", session_id="user_1")
bot.chat("Which one should I use for NLP tasks?", session_id="user_1")
```

---

### 16.13 LangChain Evaluation & Monitoring

```python
# ─── LangSmith (LangChain's observability platform) ───
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls-..."
os.environ["LANGCHAIN_PROJECT"] = "ml-chatbot-prod"
# All chain invocations are now traced in LangSmith dashboard

# ─── RAGAS (RAG evaluation) ───
# pip install ragas
from ragas import evaluate
from ragas.metrics import (
    faithfulness,           # is answer faithful to context?
    answer_relevancy,       # is answer relevant to question?
    context_precision,      # is retrieved context precise?
    context_recall,         # does context contain answer?
)
from datasets import Dataset

eval_data = {
    "question": ["What is backpropagation?"],
    "answer": [answer],
    "contexts": [["backprop computes gradients..."]],
    "ground_truth": ["backpropagation is..."]
}
dataset = Dataset.from_dict(eval_data)
scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])

# ─── Custom evaluator ───
from langchain.evaluation import load_evaluator
eval_chain = load_evaluator("qa", llm=chat)
result = eval_chain.evaluate_strings(
    input="What is the kernel trick in SVM?",
    prediction=predicted_answer,
    reference=ground_truth_answer
)
print(result["score"])  # 0-1

# ─── LLM-as-judge ───
judge_prompt = ChatPromptTemplate.from_messages([
    ("system", """Rate the quality of this ML answer on a scale 1-5.
Criteria: accuracy, completeness, clarity, correct use of math.
Return JSON: {{"score": int, "reason": str}}"""),
    ("human", "Question: {question}\nAnswer: {answer}")
])
judge_chain = judge_prompt | chat | JsonOutputParser()
```

---

## 17. Handling Imbalanced Data

### 17.1 Understanding Class Imbalance

```
Imbalance Ratio (IR) = N_majority / N_minority
IR > 10:   mild imbalance
IR > 100:  severe imbalance
IR > 1000: extreme imbalance (e.g., fraud detection, rare disease)

Why accuracy fails: if 99% are majority → always predict majority → 99% accuracy, 0% minority recall
Use instead: F1, AUC-ROC, AUC-PR (precision-recall), MCC, G-mean

G-mean = √(Sensitivity × Specificity) = √(TPR × TNR)
Balanced Accuracy = (TPR + TNR)/2
```

---

### 17.2 Resampling Methods

```
SMOTE:
  x_syn = xᵢ + λ(x_nn - xᵢ),   λ~U(0,1),   x_nn = random KNN of xᵢ
  Generates synthetic minority samples along line segments between neighbours

ADASYN:
  rᵢ = Δᵢ/K   (fraction of majority class among K neighbours of xᵢ)
  Generate more samples for harder-to-learn minority instances (higher rᵢ)
  nᵢ = ⌈G × rᵢ/Σᵢ rᵢ⌉   samples for point i

Borderline-SMOTE:
  Only oversample minority instances near decision boundary ("DANGER" zone)
  Better than SMOTE for well-separated classes

SMOTE-ENN: SMOTE + Edited Nearest Neighbours (remove ambiguous samples)
SMOTE-Tomek: SMOTE + remove Tomek links

Tomek Links:
  Pair (xᵢ, xⱼ) is a Tomek link if yᵢ≠yⱼ and ∄xₖ: d(xₖ,xᵢ)<d(xᵢ,xⱼ) or d(xₖ,xⱼ)<d(xᵢ,xⱼ)
  Remove majority class member of each Tomek link → cleaner boundary

NearMiss:
  NearMiss-1: keep majority samples whose mean dist to K nearest minority is smallest
  NearMiss-2: keep majority samples whose mean dist to K farthest minority is smallest
  NearMiss-3: for each minority sample, keep M nearest majority samples

Focal Loss:
  FL(p) = -α(1-p)^γ log(p)
  (1-p)^γ: down-weights easy/confident predictions → focuses on hard examples
  γ=2, α=0.25 (typical); used in RetinaNet, heavily imbalanced classification

Class Weights:
  wₖ = n / (K × nₖ)   (inverse frequency weighting)
  Pass as class_weight param in sklearn; sample_weight in loss function
```

---

## 18. Model Evaluation & Selection

### 18.1 Classification Metrics

```
Confusion matrix: TP, FP, TN, FN

Accuracy      = (TP+TN)/(TP+TN+FP+FN)   — misleading for imbalanced classes
Precision     = TP/(TP+FP)               — "of what I called positive, how many were?"
Recall/TPR    = TP/(TP+FN)               — "of all actual positives, how many did I find?"
Specificity   = TN/(TN+FP)              — recall for negative class
F1            = 2PR/(P+R)               — harmonic mean of precision & recall
Fβ            = (1+β²)PR/(β²P+R)        — β>1: recall-weighted; β<1: precision-weighted

MCC           = (TP·TN-FP·FN)/√[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]   ∈ [-1,1]
  Balanced metric; works well for imbalanced classes; +1 = perfect

Cohen's κ     = (p_o - p_e)/(1-p_e)
  Accounts for agreement by chance; κ>0.8: excellent, 0.6-0.8: good

AUC-ROC: area under TPR vs FPR curve
  Probabilistic interpretation: P(score(+) > score(-)) for random positive/negative pair
  AUC = 0.5: random; AUC = 1: perfect; scale-invariant; threshold-invariant

AUC-PR: area under Precision-Recall curve
  Better for highly imbalanced data (ROC can be optimistic when TN >> FP)
  Average Precision = Σₙ(Rₙ-Rₙ₋₁)Pₙ

Calibration: P(y=1|P̂=p) = p for all p?
  Reliability diagram; Brier score = (1/n)Σ(pᵢ-yᵢ)² (lower=better)
  Expected Calibration Error (ECE) = Σ_bins (n_b/n)|acc_b - conf_b|
```

---

### 18.2 Regression Metrics

```
MAE  = (1/n)Σ|yᵢ-ŷᵢ|           — robust to outliers; same unit as y
MSE  = (1/n)Σ(yᵢ-ŷᵢ)²           — penalises large errors heavily
RMSE = √MSE                       — same unit as y; penalises outliers
MAPE = (100/n)Σ|yᵢ-ŷᵢ|/|yᵢ|    — percentage error; undefined for yᵢ=0
sMAPE = (200/n)Σ|yᵢ-ŷᵢ|/(|yᵢ|+|ŷᵢ|)   — symmetric MAPE; bounded
MSLE = (1/n)Σ(log(ŷᵢ+1)-log(yᵢ+1))²   — for exponential-scale targets
R²   = 1-SS_res/SS_tot           — proportion of variance explained
Adjusted R² = 1-(1-R²)(n-1)/(n-p-1)

Huber loss (combines MSE and MAE):
  L_δ(y,ŷ) = (1/2)(y-ŷ)²            if |y-ŷ| ≤ δ
              δ|y-ŷ| - (1/2)δ²       otherwise
Quantile loss (for prediction intervals):
  L_q(y,ŷ) = q(y-ŷ) if y≥ŷ,   (1-q)(ŷ-y) if y<ŷ
  Minimise L_0.05 and L_0.95 for 90% prediction interval
```

---

### 18.3 Cross-Validation Strategies

```
k-Fold: split into k folds; train on k-1, test on 1; repeat k times; average
  Standard k: 5 or 10. Larger k → lower bias, higher variance (computation)

Stratified k-Fold: preserve class distribution in each fold (use for imbalanced)

LOOCV (Leave-One-Out): k=n; nearly unbiased; high variance; expensive

Repeated k-Fold: run k-fold r times with different random seeds; more stable estimate

Nested CV (double CV): inner loop for hyperparameter tuning; outer loop for evaluation
  Outer: k₁-fold for unbiased generalisation estimate
  Inner: k₂-fold for hyperparameter selection within each outer train fold
  Prevents optimistic bias from tuning on test data

Group k-Fold: groups (e.g., patients) don't appear in both train and test
  Prevents data leakage from correlated samples

Time Series Split:
  Fold 1: train [1..t₁], test [t₁+1..t₂]
  Fold 2: train [1..t₂], test [t₂+1..t₃]
  Always test on future data. Use gap between train/test if leakage risk.

Blocked k-Fold (time series): preserve temporal blocks to avoid autocorrelation
```

---

### 18.4 Hyperparameter Tuning

```
Grid Search: O(Π|Hᵢ|) evaluations — exponential in number of params
Random Search: O(n_trials); 60x more efficient when few params matter (Bergstra 2012)

Bayesian Optimisation:
  Surrogate: Gaussian Process models f(hp) → performance
  Acquisition: α(hp) = expected improvement or UCB
  x_next = argmax α(hp)
  Each evaluation updates GP; balance explore vs exploit
  Optuna, Hyperopt, Ax/BoTorch, Weights & Biases sweeps

Hyperband:
  Combine random search + successive halving
  Allocate small budget to many configs; halve resources, keep top configs; repeat
  Asymptotically optimal exploration-exploitation tradeoff

BOHB (Bayesian Optimisation + Hyperband):
  Use BO to sample good configs; Hyperband for early stopping
  Strong practical performance (Auto-sklearn, SMAC)

Population-Based Training (PBT):
  Train population in parallel; periodically copy top performers' weights + perturb hyperparams
  Used by DeepMind for RL and DL models

Neural Architecture Search (NAS):
  DARTS: differentiable architecture; gradient descent over architecture
  ENAS: shared weights; controller selects paths via RL
```

---

## 19. Regularisation & Optimisation

### 19.1 Norms & Regularisation Theory

```
Geometry of regularisation:
  Constrained form: min_{β} L(β)   s.t. ||β||_p ≤ t
  Lagrangian:       min_{β} L(β) + λ||β||_p
  Equivalent for correct λ(t)

L0: ||β||₀ = count(βⱼ≠0) → exact sparsity (NP-hard to solve)
L1: ||β||₁ = Σ|βⱼ| → convex relaxation of L0 → sparse solutions
L2: ||β||₂² = Σβⱼ² → shrinks all toward 0; differentiable everywhere
Elastic Net: αL1 + (1-α)L2 → grouped sparsity

Why L1 is sparse:
  Subdifferential of |β| at 0: ∂|0| = [-1,1]
  Stationarity at 0: 0 ∈ ∂L/∂β + λ[-1,1]
  → solution is 0 unless |∂L/∂β| > λ
  → soft-thresholding: β* = sign(β_LS)max(|β_LS|-λ,0)

Nuclear norm (matrix):  ||W||_* = Σᵢ σᵢ  (sum of singular values)
  Convex relaxation of matrix rank
  Used in matrix completion (Netflix prize), multi-task learning

Spectral norm: ||W||_σ = σ_max  (largest singular value)
  Controls Lipschitz constant of linear layer
  Spectral normalisation: W_SN = W/σ_max → ||W_SN||_σ = 1

Dropout as regularisation:
  Equivalent to L2 regularisation on activations (Wager et al., 2013)
  In linear models: dropout on input xᵢ with Bernoulli(p) → shrinkage
```

---

### 19.2 Normalisation Layers

```
Batch Norm: normalise over (N, H, W) per channel — fast, requires large batch
Layer Norm: normalise over (C, H, W) per sample — independent of batch size; Transformers
Instance Norm: normalise over (H, W) per sample per channel — style transfer
Group Norm: normalise over (C//G, H, W) per group per sample — small batches, detection
RMS Norm:   x̂ = x/RMS(x) = x/√(Σxᵢ²/d + ε);  y = γx̂  — simpler; LLaMA, Gemma

BN variance at test time (running stats via EMA):
  μ_run = (1-m)μ_run + m·μ_batch
  σ²_run = (1-m)σ²_run + m·σ²_batch
  m = momentum (default 0.1 in PyTorch)

Layer scale (DeiT, CaiT):
  x := x + α × sublayer(x),  α learnable, init 1e-4 to 1e-6
  Stabilises training of deep ViTs
```

---

## 20. Probabilistic Machine Learning

### 20.1 Gaussian Processes

```
f ~ GP(m(x), k(x,x'))
  m(x): mean function (often m≡0)
  k(x,x'): kernel — encodes assumptions about f

Posterior given noisy observations y = f(X) + ε, ε~N(0,σ²I):
  [y, f*]ᵀ ~ N(0, [[K+σ²I, K*ᵀ],[K*, K**]])
  f*|X,y,X* ~ N(μ*, Σ*)
    μ* = K*ᵀ(K+σ²I)⁻¹y
    Σ* = K** - K*ᵀ(K+σ²I)⁻¹K*

Marginal likelihood (for hyperparameter optimisation):
  log P(y|X,θ) = -(1/2)yᵀ(K+σ²I)⁻¹y - (1/2)log|K+σ²I| - (n/2)log(2π)
  Maximise over kernel hyperparams θ

Common kernels:
  RBF:  k(x,x') = σ²exp(-||x-x'||²/(2l²))         (infinitely differentiable)
  Matérn-5/2: k = σ²(1+√5r/l+5r²/3l²)exp(-√5r/l)  (twice differentiable)
  Periodic: k = σ²exp(-2sin²(π||x-x'||/p)/l²)       (for seasonal data)
  Linear:  k(x,x') = σ₀² + σ²xᵀx'                  (Bayesian linear regression)

Complexity: O(n³) training (Cholesky), O(n²) prediction
Sparse GPs (inducing points): O(nm²) with m << n; SVGP, FITC
```

---

### 20.2 Hidden Markov Models (HMM)

```
λ = (A, B, π)
  A: transition matrix, Aᵢⱼ = P(zₜ=j|zₜ₋₁=i)
  B: emission matrix, Bᵢ(x) = P(xₜ=x|zₜ=i)
  π: initial distribution

Three algorithms:
  1. Forward (Evaluation): P(x₁,...,xₜ|λ) in O(TN²)
     αₜ(i) = P(x₁,...,xₜ,zₜ=i|λ)
     α₁(i) = πᵢBᵢ(x₁)
     αₜ(j) = [Σᵢ αₜ₋₁(i) Aᵢⱼ] Bⱼ(xₜ)

  2. Viterbi (Decoding): most likely state sequence in O(TN²)
     δₜ(j) = max_{z₁,...,zₜ₋₁} P(z₁,...,zₜ=j,x₁,...,xₜ|λ)
     δₜ(j) = max_i[δₜ₋₁(i)Aᵢⱼ] Bⱼ(xₜ)
     Backtrack via ψₜ(j) = argmax_i[δₜ₋₁(i)Aᵢⱼ]

  3. Baum-Welch (Learning): EM for HMM parameters
     E: compute γₜ(i,j) = P(zₜ=i,zₜ₊₁=j|X,λ) via forward-backward
     M: re-estimate A, B, π from expected counts
```

---

### 20.3 Calibration

```
A model is calibrated if: P(Y=1 | P̂=p) = p  for all p ∈ [0,1]

Reliability diagram: group predictions into B bins;
  plot mean prediction vs fraction positive per bin
  Diagonal = perfectly calibrated

Calibration metrics:
  ECE = Σ_b (nᵦ/n)|acc(b) - conf(b)|
  MCE = max_b |acc(b) - conf(b)|
  Brier score = (1/n)Σ(pᵢ-yᵢ)²   (combines calibration + refinement)

Calibration methods:
  Platt scaling: P̂_cal = σ(aP̂ + b)   (fit logistic regression on val set)
  Isotonic regression: non-parametric monotonic fit; overfit risk on small val set
  Temperature scaling (neural nets): P̂_cal = softmax(z/T)
    T>1: softer distribution (less confident)
    T<1: sharper (more confident)
    T optimised on val set via NLL minimisation; one scalar — simple, works well
  Beta calibration: fit Beta distribution to predictions
```

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
