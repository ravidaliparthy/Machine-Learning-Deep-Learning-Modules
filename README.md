# Machine-Learning-Deep-Learning-Modules

# 🤖 Machine Learning — The Ultimate Beginner to Advanced Reference

> A **complete, theory-first** repository covering every major area of Machine Learning and Deep Learning — from mathematical foundations to state-of-the-art algorithms. Every algorithm includes intuition, step-by-step derivation, mathematical formulas, and practical notes.

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
13. [Long Short-Term Memory (LSTM) & GRU](#13-long-short-term-memory-lstm--gru)
14. [Advanced Deep Learning](#14-advanced-deep-learning)
15. [Handling Imbalanced Data](#15-handling-imbalanced-data)
16. [Model Evaluation & Selection](#16-model-evaluation--selection)
17. [Regularisation & Optimisation](#17-regularisation--optimisation)
18. [Probabilistic Machine Learning](#18-probabilistic-machine-learning)
19. [Repository Structure](#19-repository-structure)

---

## 1. Mathematical Foundations

### 1.1 Linear Algebra

**Vectors and Norms**
```
L1 norm:  ||x||₁ = Σᵢ |xᵢ|
L2 norm:  ||x||₂ = √(Σᵢ xᵢ²)
Lp norm:  ||x||ₚ = (Σᵢ |xᵢ|ᵖ)^(1/p)
```

**Matrix Operations**
```
Transpose:        (AB)ᵀ = BᵀAᵀ
Inverse:          AA⁻¹ = I     (only for square, non-singular)
Trace:            tr(A) = Σᵢ Aᵢᵢ = Σᵢ λᵢ
Determinant:      det(A) = Πᵢ λᵢ   (product of eigenvalues)
Frobenius norm:   ||A||_F = √(Σᵢ Σⱼ Aᵢⱼ²) = √tr(AᵀA)
```

**Eigendecomposition**
```
Av = λv           (v = eigenvector, λ = eigenvalue)
A = Q Λ Qᵀ        (symmetric A: Q = orthogonal matrix of eigenvectors,
                   Λ = diagonal matrix of eigenvalues)
```

**Singular Value Decomposition (SVD)**
```
A = U Σ Vᵀ
```
Where U (m×m) and V (n×n) are orthogonal; Σ (m×n) is diagonal with singular values σᵢ ≥ 0.
Relationship to eigenvalues: `σᵢ = √λᵢ(AᵀA)`

**Covariance Matrix**
```
Σ = (1/(n-1)) Xᵀ X     (X is mean-centred)
Σᵢⱼ = Cov(Xᵢ, Xⱼ) = E[(Xᵢ - μᵢ)(Xⱼ - μⱼ)]
```

### 1.2 Calculus & Optimisation

**Gradient** (vector of partial derivatives):
```
∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```

**Hessian Matrix** (matrix of second-order partial derivatives):
```
H(f)ᵢⱼ = ∂²f / (∂xᵢ ∂xⱼ)
```
- H is positive definite → local minimum
- H is negative definite → local maximum
- H is indefinite → saddle point

**Chain Rule (scalar)**
```
dz/dx = (dz/dy)(dy/dx)
```

**Chain Rule (vector — Jacobian)**
```
∂z/∂x = (∂y/∂x)ᵀ (∂z/∂y)
```

**Taylor Series Expansion** (used in optimisation proofs):
```
f(x + δ) ≈ f(x) + ∇f(x)ᵀ δ + (1/2) δᵀ H(x) δ + ...
```

### 1.3 Probability & Statistics

**Bayes' Theorem**
```
P(A|B) = P(B|A) P(A) / P(B)
```

**Expectation and Variance**
```
E[X] = Σₓ x P(X=x)              (discrete)
E[X] = ∫ x f(x) dx              (continuous)
Var(X) = E[X²] - (E[X])²
Var(aX + b) = a² Var(X)
```

**Common Distributions**

| Distribution | PMF/PDF | Mean | Variance |
|---|---|---|---|
| Bernoulli(p) | p^x (1-p)^(1-x) | p | p(1-p) |
| Binomial(n,p) | C(n,x) pˣ(1-p)^(n-x) | np | np(1-p) |
| Gaussian(μ,σ²) | (1/√(2πσ²)) exp(-(x-μ)²/2σ²) | μ | σ² |
| Exponential(λ) | λ e^(-λx) | 1/λ | 1/λ² |
| Categorical(π) | Πₖ πₖ^(xₖ) | πₖ | πₖ(1-πₖ) |

**Entropy and Information Theory**
```
Entropy:           H(X) = -Σₓ P(x) log₂ P(x)
Joint Entropy:     H(X,Y) = -Σₓ Σᵧ P(x,y) log P(x,y)
Conditional:       H(Y|X) = H(X,Y) - H(X)
Mutual Info:       I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
KL Divergence:     D_KL(P||Q) = Σₓ P(x) log(P(x)/Q(x))   ≥ 0 always
Cross-Entropy:     H(P,Q) = H(P) + D_KL(P||Q) = -Σₓ P(x) log Q(x)
```

---

## 2. Core Machine Learning Theory

### 2.1 The Learning Problem

**Goal:** Find a hypothesis `h ∈ H` that best approximates the unknown target function `f: X → Y`.

**Empirical Risk Minimisation (ERM):**
```
ĥ = argmin_{h ∈ H} (1/n) Σᵢ L(h(xᵢ), yᵢ)
```

**Expected Risk (True Objective):**
```
R(h) = E_{(x,y) ~ P} [L(h(x), y)]
```

**Generalisation Gap:**
```
R(h) - R̂(h) ≤ ε   with high probability
```

### 2.2 Bias-Variance Trade-off

For a squared loss setting:
```
E[(y - ĥ(x))²] = Bias²(ĥ(x)) + Var(ĥ(x)) + σ²_noise
```
Where:
```
Bias(ĥ(x))  = E[ĥ(x)] - f(x)          (systematic error)
Var(ĥ(x))   = E[(ĥ(x) - E[ĥ(x)])²]    (sensitivity to training data)
σ²_noise     = irreducible noise
```

- **High Bias (Underfitting):** Model too simple; misses underlying patterns.
- **High Variance (Overfitting):** Model too complex; memorises noise.
- **Goal:** Find the "sweet spot" — optimal model complexity.

### 2.3 The No Free Lunch Theorem

No single learning algorithm is universally superior. Every algorithm has the same average performance over ALL possible problems. This means domain knowledge and algorithm selection matter enormously.

### 2.4 VC Dimension & PAC Learning

**VC Dimension (Vapnik-Chervonenkis):** The largest set of points that a hypothesis class H can shatter (correctly classify for all possible labellings).

- Linear classifiers in ℝᵈ: VC dim = d + 1
- Neural networks: VC dim ≈ O(W log W) where W = number of weights

**PAC Learning Bound:**
```
n ≥ (1/ε) [ln|H| + ln(1/δ)]
```
With `n` samples, we can guarantee with probability ≥ 1-δ that:
```
R(h) - R̂(h) ≤ ε
```

### 2.5 Maximum Likelihood Estimation (MLE)

Given observations `X = {x₁, ..., xₙ}` i.i.d. from `P(x|θ)`:
```
θ_MLE = argmax_θ  L(θ) = argmax_θ  Σᵢ log P(xᵢ | θ)
```
Taking log converts products to sums (numerically stable).

### 2.6 Maximum A Posteriori (MAP) Estimation

Incorporates a prior `P(θ)`:
```
θ_MAP = argmax_θ  log P(θ|X)
       = argmax_θ  [Σᵢ log P(xᵢ|θ) + log P(θ)]
```
MAP with Gaussian prior = MLE with L2 regularisation.
MAP with Laplace prior = MLE with L1 regularisation.

---

## 3. Data Preprocessing & Feature Engineering

### 3.1 Handling Missing Values

**Types of Missingness:**
- **MCAR** (Missing Completely At Random): Missingness is unrelated to any variable.
- **MAR** (Missing At Random): Missingness depends on observed variables.
- **MNAR** (Missing Not At Random): Missingness depends on the missing value itself.

**Imputation Methods:**

Mean/Median Imputation:
```
x_imputed = mean(x_observed)    (continuous — use median for skewed)
x_imputed = mode(x_observed)    (categorical)
```

Regression Imputation — Predict missing values using other features:
```
x_missing = β₀ + β₁x₁ + β₂x₂ + ... + ε
```

KNN Imputation — Replace with weighted average of K nearest complete neighbours:
```
x_imputed = Σₖ wₖ xₖ / Σₖ wₖ,    wₖ = 1/d(x, xₖ)
```

Multiple Imputation (MICE — Multivariate Imputation by Chained Equations):
Iteratively regress each variable with missing values on all others.

### 3.2 Feature Scaling

**Min-Max Normalisation (scales to [0,1]):**
```
x' = (x - x_min) / (x_max - x_min)
```

**Z-Score Standardisation (mean=0, std=1):**
```
x' = (x - μ) / σ
```

**Robust Scaling (uses IQR — robust to outliers):**
```
x' = (x - median) / IQR,    IQR = Q3 - Q1
```

**Log Transform (for right-skewed distributions):**
```
x' = log(x + 1)    (+1 handles x=0)
```

**Box-Cox Transform:**
```
x'(λ) = (xλ - 1) / λ,  if λ ≠ 0
        log(x),          if λ = 0
```
λ is estimated by maximising the log-likelihood of normality.

**Yeo-Johnson Transform** (extends Box-Cox to negative values):
```
x'(λ) = [(x+1)λ - 1] / λ,        if λ ≠ 0, x ≥ 0
         log(x+1),                  if λ = 0, x ≥ 0
        -[(-x+1)^(2-λ)-1]/(2-λ),  if λ ≠ 2, x < 0
        -log(-x+1),                 if λ = 2, x < 0
```

### 3.3 Encoding Categorical Variables

**Label Encoding:** Integer encoding. Only appropriate for ordinal variables.
```
[Small, Medium, Large] → [0, 1, 2]
```

**One-Hot Encoding:** Binary column per category. Creates k-1 columns to avoid multicollinearity.
```
[Red, Green, Blue] → [1,0,0], [0,1,0], [0,0,1]
```

**Target Encoding (Mean Encoding):** Replace category with the mean target value.
```
x_encoded = E[y | x = category]  (smoothed to avoid overfitting)
```

**Frequency Encoding:**
```
x_encoded = count(category) / total_count
```

**Hashing (Feature Hashing):** Map categories to fixed-size vector via hash function. Handles unseen categories.

### 3.4 Feature Selection

**Filter Methods:**
- Correlation coefficient (remove highly correlated features)
- Chi-squared test (for categorical features vs categorical target)
- ANOVA F-test (for continuous features vs categorical target)
- Mutual Information: `I(X; Y) = H(X) - H(X|Y)`

**Wrapper Methods:**
- **Forward Selection:** Start with empty set, add best feature one at a time.
- **Backward Elimination:** Start with all features, remove worst one at a time.
- **Recursive Feature Elimination (RFE):** Fit model, rank features by importance, remove least important, repeat.

**Embedded Methods (built-in to model):**
- L1 Regularisation (Lasso): drives coefficients of irrelevant features to exactly 0.
- Tree-based feature importance: measures impurity decrease.

### 3.5 Outlier Detection

**Z-Score Method:**
```
z = (x - μ) / σ
Flag as outlier if |z| > 3
```

**IQR Method (Tukey's Fences):**
```
Lower fence = Q1 - 1.5 × IQR
Upper fence = Q3 + 1.5 × IQR
```

**Isolation Forest:** Randomly partition feature space. Anomalies are isolated faster (shorter path length in the tree).
```
Anomaly Score = 2^(-E[h(x)] / c(n))
```
Where `h(x)` = path length, `c(n) = 2H(n-1) - 2(n-1)/n` (average path length).

**Local Outlier Factor (LOF):**
```
LOF(x) = avg_neighbourhood(reach_density(neighbour)) / reach_density(x)
```
LOF >> 1 → outlier (sparser than its neighbours).

### 3.6 Correlation Analysis

**Pearson Correlation (linear relationships):**
```
r = Σᵢ(xᵢ - x̄)(yᵢ - ȳ) / √[Σᵢ(xᵢ - x̄)² Σᵢ(yᵢ - ȳ)²]
```

**Spearman Rank Correlation (monotonic, non-parametric):**
```
ρ = 1 - 6Σᵢ dᵢ² / (n(n²-1)),    dᵢ = rank(xᵢ) - rank(yᵢ)
```

**Point-Biserial (continuous vs binary):**
```
r_pb = (ȳ₁ - ȳ₀)/s_y × √(n₁n₀/n²)
```

**Variance Inflation Factor (VIF) — detects multicollinearity:**
```
VIF_j = 1 / (1 - R²_j)
```
VIF > 10 → severe multicollinearity for feature j.

---

## 4. Regression Algorithms

### 4.1 Linear Regression

**Model:**
```
ŷ = β₀ + β₁x₁ + ... + βₙxₙ = Xβ    (matrix form)
```

**Ordinary Least Squares (OLS) — minimise MSE:**
```
L(β) = ||y - Xβ||² = (y - Xβ)ᵀ(y - Xβ)
```

**Closed-Form Solution (Normal Equation):**
```
∂L/∂β = -2Xᵀ(y - Xβ) = 0
β* = (XᵀX)⁻¹ Xᵀy
```
Requires `XᵀX` to be invertible. Complexity: O(n d² + d³).

**Gradient Descent:**
```
β := β - α × (2/n) Xᵀ(Xβ - y)
```

**Gauss-Markov Theorem:** OLS estimator is BLUE (Best Linear Unbiased Estimator) when:
1. Linearity: `y = Xβ + ε`
2. Exogeneity: `E[ε|X] = 0`
3. No perfect multicollinearity
4. Homoscedasticity: `Var(ε|X) = σ²I`

**R-squared:**
```
R² = 1 - SS_res/SS_tot = 1 - Σ(yᵢ-ŷᵢ)² / Σ(yᵢ-ȳ)²
```

**Adjusted R² (penalises extra features):**
```
R²_adj = 1 - (1-R²)(n-1)/(n-p-1)
```

### 4.2 Ridge Regression (L2 Regularisation)

Adds L2 penalty to prevent overfitting:
```
L(β) = ||y - Xβ||² + λ||β||²
```

**Closed-Form Solution:**
```
β_ridge = (XᵀX + λI)⁻¹ Xᵀy
```
`λI` ensures the matrix is always invertible (even with multicollinearity).

**Effect:** Shrinks all coefficients towards zero; never sets them to exactly zero.

**Bayesian Interpretation:** MAP estimation with Gaussian prior `β ~ N(0, σ²/λ I)`.

### 4.3 Lasso Regression (L1 Regularisation)

```
L(β) = ||y - Xβ||² + λ||β||₁ = ||y - Xβ||² + λΣⱼ|βⱼ|
```

No closed-form solution — uses subgradient methods or coordinate descent.

**Key Property:** Produces **sparse solutions** — drives some coefficients to exactly 0, performing automatic feature selection.

**Why L1 causes sparsity:** The L1 ball has corners at the axes; the loss contours tend to touch these corners first.

**Coordinate Descent Update (Lasso):**
```
βⱼ = S(ρⱼ, λ) / Σᵢ xᵢⱼ²
```
Where `S` is the soft-threshold operator:
```
S(ρ, λ) = sign(ρ) max(|ρ| - λ, 0)
ρⱼ = Σᵢ xᵢⱼ (yᵢ - ŷᵢ^{(-j)})   (partial residual)
```

### 4.4 Elastic Net

Combines L1 and L2:
```
L(β) = ||y - Xβ||² + λ₁||β||₁ + λ₂||β||²
     = ||y - Xβ||² + λ[α||β||₁ + (1-α)||β||²]
```
- `α=1` → Lasso; `α=0` → Ridge; `0<α<1` → Elastic Net
- Selects groups of correlated features (unlike Lasso which picks one).

### 4.5 Polynomial Regression

Extends linear regression with polynomial feature expansion:
```
ŷ = β₀ + β₁x + β₂x² + ... + βₚxᵖ
```
Create new features: `[x, x², ..., xᵖ]`, then apply standard linear regression.

**Underfitting/Overfitting:** Degree p too low → underfits; too high → overfits. Use cross-validation.

### 4.6 Bayesian Linear Regression

**Prior:** `β ~ N(μ₀, Σ₀)`
**Likelihood:** `y|X,β ~ N(Xβ, σ²I)`
**Posterior:**
```
P(β|X,y) ∝ P(y|X,β) P(β)
β|X,y ~ N(μₙ, Σₙ)

Σₙ = (Σ₀⁻¹ + (1/σ²)XᵀX)⁻¹
μₙ = Σₙ(Σ₀⁻¹μ₀ + (1/σ²)Xᵀy)
```
Provides uncertainty estimates (full posterior distribution over β).

---

## 5. Classification Algorithms

### 5.1 Logistic Regression

**Sigmoid Function:**
```
σ(z) = 1/(1 + e⁻ᶻ),   σ'(z) = σ(z)(1-σ(z))
```

**Log-Odds (Logit):**
```
log[P/(1-P)] = β₀ + β₁x₁ + ... + βₙxₙ
```

**Binary Cross-Entropy Loss:**
```
L = -(1/n) Σᵢ [yᵢ log σ(βᵀxᵢ) + (1-yᵢ) log(1-σ(βᵀxᵢ))]
```

**Why not use MSE for classification?** Cross-entropy is a strictly proper scoring rule — minimising it directly maximises the likelihood of the data. MSE creates a non-convex loss surface for probabilities.

**Gradient:**
```
∂L/∂β = (1/n) Xᵀ(σ(Xβ) - y)
```

**Newton-Raphson Update (faster convergence):**
```
β := β - H⁻¹ ∇L
H = (1/n) XᵀWX,   W = diag[σ(xᵢ)(1-σ(xᵢ))]
```
This is the IRLS (Iteratively Reweighted Least Squares) algorithm.

**Multi-class — Softmax Regression:**
```
P(y=k|x) = exp(βₖᵀx) / Σⱼ exp(βⱼᵀx)
```
Cross-entropy loss:
```
L = -(1/n) Σᵢ Σₖ yᵢₖ log P(y=k|xᵢ)
```

### 5.2 K-Nearest Neighbours (KNN)

**Euclidean Distance:**
```
d_E(x,y) = √[Σᵢ(xᵢ-yᵢ)²]
```

**Manhattan Distance:**
```
d_M(x,y) = Σᵢ|xᵢ-yᵢ|
```

**Minkowski Distance:**
```
d_p(x,y) = (Σᵢ|xᵢ-yᵢ|ᵖ)^(1/p)
```

**Cosine Similarity:**
```
cos(x,y) = (xᵀy)/(||x|| ||y||)
```

**Classification:** `ŷ = argmax_c Σ_{xⱼ ∈ Nₖ(x)} 𝟙[yⱼ = c]`

**Weighted KNN:** Weight by inverse distance:
```
ŷ = argmax_c Σⱼ (1/dⱼ) 𝟙[yⱼ = c]
```

**Curse of Dimensionality:** In high dimensions, all points become equidistant. The volume of a hypersphere relative to its bounding box:
```
V_sphere / V_cube = πᵈ/² / (Γ(d/2+1) × 2ᵈ) → 0  as d → ∞
```

**Choosing K:**
- Odd K avoids ties in binary classification.
- K=√n is a common rule of thumb.
- Use cross-validation; plot K vs accuracy.

**Time Complexity:** O(n·d) per query. Use KD-Trees (O(d log n)) or Ball Trees for speedup.

### 5.3 Naive Bayes

**Bayes' Theorem Applied:**
```
P(C|x) = P(x|C) P(C) / P(x)
```

**Naive Independence Assumption:**
```
P(x|C) = Πⱼ P(xⱼ|C)
```

**Decision Rule:**
```
ĉ = argmax_c log P(C=c) + Σⱼ log P(xⱼ|C=c)
```
(log is taken for numerical stability)

**Gaussian Naive Bayes (continuous features):**
```
P(xⱼ|C=c) = (1/√(2πσ²ⱼc)) exp(-(xⱼ-μⱼc)²/(2σ²ⱼc))
```

**Multinomial Naive Bayes (count features — text):**
```
P(xⱼ|C=c) = (N_cj + α) / (N_c + α|V|)
```

**Bernoulli Naive Bayes (binary features):**
```
P(xⱼ|C=c) = pⱼc^xⱼ × (1-pⱼc)^(1-xⱼ)
```

**Laplace Smoothing (prevents zero probability):**
```
P(xⱼ = v|C=c) = (count(xⱼ=v, C=c) + α) / (count(C=c) + α|V|)
```

**Why "Naive" works despite strong assumption?**
Even when features are correlated, the decision boundary (argmax) often remains correct even if the probabilities are miscalibrated.

### 5.4 Decision Trees

**Entropy:**
```
H(S) = -Σₖ pₖ log₂ pₖ
```
H = 0 (pure node), H = log₂K (completely impure, K classes).

**Information Gain (ID3, C4.5):**
```
IG(S, A) = H(S) - Σᵥ (|Sᵥ|/|S|) H(Sᵥ)
```

**Gain Ratio (C4.5 — normalises by split info to penalise many-valued attributes):**
```
GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)
SplitInfo(S, A) = -Σᵥ (|Sᵥ|/|S|) log₂(|Sᵥ|/|S|)
```

**Gini Impurity (CART):**
```
Gini(S) = 1 - Σₖ pₖ²
```
Minimum 0 (pure), maximum 1-1/K.

**Gini Gain:**
```
GiniGain(S, A) = Gini(S) - Σᵥ (|Sᵥ|/|S|) Gini(Sᵥ)
```

**MSE for Regression Trees:**
```
Split criterion: minimise Σᵥ (|Sᵥ|/|S|) Var(Sᵥ)
Leaf prediction: ŷ = mean(y in leaf)
```

**Cost-Complexity Pruning (α-pruning):**
```
R_α(T) = R(T) + α|T|
```
Where R(T) = misclassification rate, |T| = number of leaves.
Larger α → simpler tree.

**Depth vs Overfitting:**
- Depth 1 = "Decision Stump" (weakest learner — used in boosting)
- No limit = memorises training data
- Cross-validate to find optimal depth

### 5.5 Support Vector Machines (SVM)

**Hard-Margin SVM (linearly separable):**
```
Minimise:   (1/2)||w||²
Subject to: yᵢ(wᵀxᵢ + b) ≥ 1,  ∀i
```

**Geometric Margin:** `γ = 2/||w||`

**Lagrangian Dual (used to derive kernel trick):**
```
Primal:   min_w,b (1/2)||w||² - Σᵢ αᵢ[yᵢ(wᵀxᵢ+b)-1]
Dual:     max_α  Σᵢαᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼxᵢᵀxⱼ
          s.t.   αᵢ ≥ 0,  Σᵢ αᵢyᵢ = 0
```

**KKT Conditions:**
```
αᵢ[yᵢ(wᵀxᵢ+b)-1] = 0  →  αᵢ > 0 only for support vectors (on margin boundary)
```

**Decision function (only depends on support vectors):**
```
f(x) = sign(Σᵢ αᵢ yᵢ K(xᵢ, x) + b)
```

**Soft-Margin SVM (slack variables ξᵢ):**
```
Minimise:   (1/2)||w||² + C Σᵢ ξᵢ
Subject to: yᵢ(wᵀxᵢ+b) ≥ 1 - ξᵢ,  ξᵢ ≥ 0
```
- C large → narrow margin, few violations (may overfit)
- C small → wide margin, allows more violations (may underfit)

**Hinge Loss interpretation:**
```
SVM objective = (1/n) Σᵢ max(0, 1-yᵢf(xᵢ)) + (λ/2)||w||²
```

**Kernel Functions:**

| Kernel | Formula | When to Use |
|--------|---------|-------------|
| Linear | `K(x,z) = xᵀz` | Linearly separable, high-d text |
| Polynomial | `(γ xᵀz + r)ᵈ` | Explicit feature interactions |
| RBF/Gaussian | `exp(-γ||x-z||²)` | General purpose, default choice |
| Laplace | `exp(-γ||x-z||₁)` | Robust to outliers |
| Sigmoid | `tanh(γ xᵀz + r)` | Neural network-like |

**Mercer's Condition:** A kernel K is valid iff the kernel matrix K is positive semi-definite for any set of points.

**Kernel Trick:** We never need to explicitly compute φ(x); we only need K(x,z) = φ(x)ᵀφ(z). This allows infinite-dimensional feature spaces.

### 5.6 Discriminant Analysis

**Linear Discriminant Analysis (LDA):**

Assumes Gaussian class-conditionals with shared covariance:
```
P(x|C=k) = N(μₖ, Σ)   (shared Σ)
```

Decision boundary is linear:
```
δₖ(x) = xᵀΣ⁻¹μₖ - (1/2)μₖᵀΣ⁻¹μₖ + log πₖ
```
Assign x to class k with highest δₖ(x).

**Quadratic Discriminant Analysis (QDA):** Each class has its own covariance Σₖ → quadratic boundary.

**LDA as Dimensionality Reduction:** Projects data to maximise between-class scatter relative to within-class scatter:
```
J(W) = Wᵀ S_B W / Wᵀ S_W W

S_W = Σₖ Σᵢ∈Cₖ (xᵢ-μₖ)(xᵢ-μₖ)ᵀ   (within-class scatter)
S_B = Σₖ nₖ(μₖ-μ)(μₖ-μ)ᵀ          (between-class scatter)
```
Solution: Eigenvectors of `S_W⁻¹ S_B`.

---

## 6. Clustering & Unsupervised Learning

### 6.1 K-Means Clustering

**Objective (WCSS):**
```
J = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²
```

**Lloyd's Algorithm:**
1. Initialise K centroids `{μₖ}` (randomly or K-means++)
2. **Assignment:** `cᵢ = argmin_k ||xᵢ - μₖ||²`
3. **Update:** `μₖ = (1/|Cₖ|) Σᵢ∈Cₖ xᵢ`
4. Repeat 2-3 until convergence

**K-means++ Initialisation (better starting centroids):**
1. Pick first centroid uniformly at random.
2. Pick next centroid with probability ∝ d(x, nearest centroid)².
3. Repeat until K centroids.

Guarantees E[cost] ≤ 8(ln K + 2) × OPT.

**Elbow Method:**
```
Plot WCSS vs K. Choose K at the "elbow" where
marginal reduction in WCSS flattens.
```

**Silhouette Score (measures cluster quality):**
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
Where:
- `a(i)` = mean distance to all points in same cluster
- `b(i)` = min mean distance to points in other clusters
- s(i) ∈ [-1, 1]; higher is better

**Limitations:** Assumes spherical clusters; sensitive to outliers; requires K upfront.

### 6.2 DBSCAN

**Density-Based Spatial Clustering of Applications with Noise**

Parameters: ε (radius), minPts (minimum neighbours)

**Definitions:**
- **Core Point:** Has ≥ minPts points within distance ε.
- **Border Point:** Not core, but within ε of a core point.
- **Noise Point:** Neither core nor border.

**Clusters** = maximal sets of density-connected points.

**Advantages:** Finds arbitrarily shaped clusters; automatically finds K; robust to outliers (marks them as noise).

**Complexity:** O(n log n) with spatial indexing.

### 6.3 Hierarchical Clustering

Builds a tree (dendrogram) of clusters. No need to specify K in advance.

**Agglomerative (bottom-up):** Start with each point as its own cluster; merge closest pairs.

**Linkage Criteria:**

| Method | Distance between clusters A and B |
|---|---|
| Single | min d(a,b) for a∈A, b∈B |
| Complete | max d(a,b) for a∈A, b∈B |
| Average (UPGMA) | (1/|A||B|) ΣΣ d(a,b) |
| Ward | Increase in total within-cluster variance |

**Ward Linkage:**
```
d(A∪B, C) = √[(|A|+|C|)d²(A,C) + (|B|+|C|)d²(B,C) - |C|d²(A,B)] / √(|A|+|B|+|C|)
```

### 6.4 Gaussian Mixture Models (GMM)

Assumes data is generated from K Gaussian distributions:
```
P(x) = Σₖ πₖ N(x | μₖ, Σₖ)
```
Where `πₖ` are mixing weights, `Σₖ πₖ = 1`.

**E-M Algorithm:**

**E-step (compute responsibilities):**
```
γᵢₖ = πₖ N(xᵢ|μₖ,Σₖ) / Σⱼ πⱼ N(xᵢ|μⱼ,Σⱼ)
```

**M-step (update parameters):**
```
Nₖ = Σᵢ γᵢₖ
μₖ = (1/Nₖ) Σᵢ γᵢₖ xᵢ
Σₖ = (1/Nₖ) Σᵢ γᵢₖ (xᵢ-μₖ)(xᵢ-μₖ)ᵀ
πₖ = Nₖ/n
```

**Relationship to K-Means:** K-means is a special case of GMM with Σₖ = σ²I and hard assignments (γᵢₖ ∈ {0,1}).

---

## 7. Dimensionality Reduction

### 7.1 Principal Component Analysis (PCA)

**Goal:** Find orthogonal directions of maximum variance.

**Algorithm:**
1. Centre data: `X ← X - μ`
2. Compute covariance: `C = (1/n)XᵀX`
3. Eigendecompose: `C = V Λ Vᵀ`
4. Sort by eigenvalue descending.
5. Project: `Z = XV_k` (top-k eigenvectors)

**Via SVD (numerically preferred):**
```
X = U Σ Vᵀ
Z = U_k Σ_k = X V_k
```

**Proportion of Variance Explained:**
```
PVE_k = λₖ / Σᵢ λᵢ
```

**Scree Plot:** Plot eigenvalues vs component index; keep components before the "elbow".

**Reconstruction Error:**
```
||X - X_reconstructed||²_F = Σᵢ>k λᵢ
```

**PCA for Compression:**
```
Compressed: Z = X Vₖ   (n × k matrix)
Reconstruct: X̂ = Z Vₖᵀ = X Vₖ Vₖᵀ
```

**Limitations of PCA:** Linear only; sensitive to scaling (always standardise first); components may not be interpretable.

### 7.2 t-SNE (t-Distributed Stochastic Neighbour Embedding)

Preserves local structure for visualisation (2D/3D).

**Step 1:** Compute pairwise similarities in high-d space:
```
pⱼ|ᵢ = exp(-||xᵢ-xⱼ||²/(2σᵢ²)) / Σₖ≠ᵢ exp(-||xᵢ-xₖ||²/(2σᵢ²))
pᵢⱼ = (pⱼ|ᵢ + pᵢ|ⱼ) / (2n)
```

**Step 2:** Compute similarities in 2D space using Student-t (heavier tails):
```
qᵢⱼ = (1 + ||yᵢ-yⱼ||²)⁻¹ / Σₖ≠ₗ (1 + ||yₖ-yₗ||²)⁻¹
```

**Step 3:** Minimise KL divergence:
```
KL(P||Q) = Σᵢ Σⱼ pᵢⱼ log(pᵢⱼ/qᵢⱼ)
```

**Why t-distribution?** Heavier tails prevent the "crowding problem" — moderate distances in high-d space map to larger distances in 2D.

**Perplexity:** Controls effective number of neighbours (~5–50). Analogous to σᵢ.

### 7.3 UMAP (Uniform Manifold Approximation and Projection)

Based on Riemannian geometry and algebraic topology. Faster than t-SNE and preserves more global structure.

**Core Idea:** Model data as a fuzzy topological structure and find a low-dimensional representation that has a similar structure.

---

## 8. Ensemble Learning

### 8.1 Why Ensembles Work

**Variance Reduction (Bagging):**
```
Var(x̄) = σ²/B + (1-1/B) ρ σ²
```
Where ρ = pairwise correlation between models. Reducing ρ (diversity) reduces variance.

**Bias Reduction (Boosting):** Sequentially correct errors; each model focuses on what previous ones got wrong.

**Decomposition:**
```
Error = Bias² + Variance + Noise
Bagging:   reduces Variance
Boosting:  reduces Bias (and sometimes Variance)
Stacking:  reduces both through meta-learning
```

### 8.2 Bagging

**Bootstrap Sample:** Draw n samples with replacement. ~63.2% unique samples per bag.
```
P(sample included) = 1 - (1-1/n)ⁿ → 1 - e⁻¹ ≈ 0.632
```

**Aggregation:**
```
Regression:     ŷ = (1/B) Σ_b hb(x)
Classification: ŷ = majority_vote{hb(x)}
```

### 8.3 Random Forest

Extends Bagging with feature randomness:
- At each split, only `m = √p` (classification) or `m = p/3` (regression) features considered.
- This decorrelates trees, reducing variance.

**Out-of-Bag (OOB) Error:**
```
OOB error ≈ (1/n) Σᵢ L(yᵢ, ĥ_OOB(xᵢ))
```
Where `ĥ_OOB(xᵢ)` = prediction from trees that did NOT include xᵢ in training.

**Feature Importance (Mean Decrease in Impurity):**
```
FI(j) = (1/B) Σ_b Σ_{t: split on j} nₜ × ΔImpurity(t) / nₜ_root
```

**Permutation Importance (more reliable):**
Permute feature j in OOB samples; measure increase in error:
```
PI(j) = (1/B) Σ_b [Error_b(permuted j) - Error_b(original)]
```

### 8.4 AdaBoost

**Algorithm:**
1. Initialise weights: `wᵢ = 1/n`
2. For t = 1, ..., T:
   a. Train weak learner `hₜ` minimising weighted error.
   b. Compute error: `εₜ = Σᵢ wᵢ 𝟙[hₜ(xᵢ) ≠ yᵢ]`
   c. Compute learner weight: `αₜ = (1/2) ln[(1-εₜ)/εₜ]`
   d. Update weights:
      ```
      wᵢ ← wᵢ exp(-αₜ yᵢ hₜ(xᵢ))
      wᵢ ← wᵢ / Σⱼ wⱼ
      ```
3. Final: `H(x) = sign(Σₜ αₜ hₜ(x))`

**Note:** `αₜ > 0` when εₜ < 0.5 (better than random). `αₜ` is larger when error is smaller.

**Training Error Bound:**
```
Training Error ≤ Πₜ 2√(εₜ(1-εₜ)) = exp(-2Σₜ γₜ²)
```
Where `γₜ = 0.5 - εₜ` is the edge of learner t.

**AdaBoost as Stagewise Additive Modelling:**
Minimises exponential loss `L(y, f) = exp(-yf)` via greedy forward stagewise:
```
(αₜ, hₜ) = argmin_{α,h} Σᵢ exp(-yᵢ(Fₜ₋₁(xᵢ) + αh(xᵢ)))
```

### 8.5 Gradient Boosting

**General Framework:** Fit each new model to the negative gradient of the loss:
```
Fₘ(x) = Fₘ₋₁(x) + η × hₘ(x)
```

**Pseudo-residuals (negative gradient):**
```
rᵢₘ = -[∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ)]_{F=Fₘ₋₁}
```

**Loss Function → Pseudo-residuals:**

| Loss | L(y,F) | Pseudo-residual rᵢ |
|------|--------|-------------------|
| MSE (Regression) | (y-F)²/2 | y - F(x) |
| MAE (Regression) | |y-F| | sign(y-F) |
| Log-loss (Classification) | log(1+e^{-yF}) | y - σ(F(x)) |
| Deviance | -2[yF - log(1+eᶠ)] | y - σ(F(x)) |

**Shrinkage (learning rate η):** Trades off speed vs generalisation. Smaller η → better generalisation, needs more trees.

**Stochastic Gradient Boosting:** Subsample fraction of training data at each step → reduces variance, speeds training.

### 8.6 XGBoost

**Regularised Objective:**
```
Obj = Σᵢ L(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)
Ω(f) = γT + (1/2)λ||w||²
```

**Second-Order Taylor Approximation:**
```
Obj ≈ Σᵢ [gᵢ fₜ(xᵢ) + (1/2)hᵢ fₜ²(xᵢ)] + Ω(fₜ) + const
```
Where `gᵢ = ∂L/∂ŷᵢ`, `hᵢ = ∂²L/∂ŷᵢ²`.

**Optimal Leaf Score:**
```
wⱼ* = -Gⱼ / (Hⱼ + λ),    Gⱼ = Σᵢ∈Iⱼ gᵢ,  Hⱼ = Σᵢ∈Iⱼ hᵢ
```

**Optimal Gain for a Split:**
```
Gain = (1/2)[Gₗ²/(Hₗ+λ) + Gᵣ²/(Hᵣ+λ) - (Gₗ+Gᵣ)²/(Hₗ+Hᵣ+λ)] - γ
```

**Key Innovations of XGBoost:**
- Sparsity-aware split finding (handles missing values natively)
- Approximate greedy algorithm with weighted quantile sketch
- Column block structure for parallel tree learning
- Cache-aware access patterns

### 8.7 LightGBM

**Histogram-Based Splitting:** Bucket continuous features into discrete bins (256 bins). Trades slight accuracy loss for 8–16× speedup.

**GOSS (Gradient-based One-Side Sampling):** Keep all large-gradient instances; randomly sample small-gradient instances.
```
Ñ_new = top-a% large gradient + random-b% small gradient × (1-a)/b
```

**EFB (Exclusive Feature Bundling):** Bundle mutually exclusive features (features that rarely take non-zero values simultaneously) to reduce feature count.

**Leaf-wise (best-first) vs Level-wise Growth:**
- LightGBM: leaf-wise → splits the leaf with max gain → lower loss, risk of overfit
- XGBoost: level-wise → all leaves at same depth → more stable

### 8.8 CatBoost

**Key Innovation:** Ordered Target Statistics to prevent target leakage:
```
x̃ᵢ = (Σⱼ<i 𝟙[xⱼ=xᵢ] yⱼ + α ȳ) / (Σⱼ<i 𝟙[xⱼ=xᵢ] + α)
```
Uses only past observations (in a random permutation) to encode category → no leakage.

**Symmetric Trees:** All splits at the same level use the same feature/threshold → reduces overfitting, fast inference.

### 8.9 Voting Classifiers

**Hard Voting:**
```
ŷ = mode{ŷ₁, ŷ₂, ..., ŷₘ}
```

**Soft Voting (weighted probability averaging):**
```
ŷ = argmax_k Σₘ wₘ Pₘ(y=k|x)
```

**Error Probability of Majority Vote (K independent classifiers, each error prob. ε):**
```
P(majority wrong) = Σⱼ>K/2 C(K,j) εʲ(1-ε)^{K-j}
```
If ε < 0.5, this is less than ε → ensembles are better than individuals.

### 8.10 Stacking

**Level-0 (Base Learners):** Diverse models (e.g., Ridge, RF, SVM, GBM).

**Level-1 (Meta-Learner):** Trained on out-of-fold predictions from level-0.

**k-Fold Stacking Procedure:**
```
For each fold k:
  Train all base models on folds ≠ k
  Get predictions on fold k: [ĥ₁(xᵢ), ..., ĥₘ(xᵢ)]

Stack predictions → new feature matrix Z (n × M)
Train meta-learner on Z with labels y
```

**Blending (simpler variant):** Use holdout set (not CV) to generate meta-features. Faster but wastes data.

---

## 9. Time Series Analysis

### 9.1 Stationarity

A time series `{Yₜ}` is **strictly stationary** if its joint distribution is time-invariant.

**Weak (Covariance) Stationarity:**
```
E[Yₜ] = μ           (constant mean)
Var(Yₜ) = σ²        (constant variance)
Cov(Yₜ, Yₜ₋ₖ) = γₖ (covariance depends only on lag k)
```

**Achieving Stationarity:**
- Differencing: `Y'ₜ = Yₜ - Yₜ₋₁`
- Log transform: `Y'ₜ = log Yₜ`
- Seasonal differencing: `Y'ₜ = Yₜ - Yₜ₋s`

### 9.2 ACF and PACF

**Autocorrelation Function (ACF):**
```
ρₖ = γₖ/γ₀ = Cov(Yₜ, Yₜ₋ₖ) / Var(Yₜ)
```

**Partial Autocorrelation (PACF):**
```
φₖₖ = Corr(Yₜ - PL(Yₜ|Yₜ₋₁,...,Yₜ₋ₖ₊₁), Yₜ₋ₖ - PL(Yₜ₋ₖ|Yₜ₋₁,...,Yₜ₋ₖ₊₁))
```
Where PL = linear projection.

**Model Identification via ACF/PACF:**

| Pattern | Model |
|---------|-------|
| ACF cuts off at lag q; PACF decays | MA(q) |
| ACF decays; PACF cuts off at lag p | AR(p) |
| Both decay gradually | ARMA(p,q) |
| ACF does not decay | Non-stationary → difference |

### 9.3 ARIMA(p,d,q)

**AR(p) component:**
```
Yₜ = c + φ₁Yₜ₋₁ + φ₂Yₜ₋₂ + ... + φₚYₜ₋ₚ + εₜ
Φ(B) Yₜ = c + εₜ
Φ(B) = 1 - φ₁B - φ₂B² - ... - φₚBᵖ
```
Stationary if all roots of Φ(B) lie outside the unit circle.

**MA(q) component:**
```
Yₜ = μ + εₜ + θ₁εₜ₋₁ + ... + θqεₜ₋q
Yₜ = μ + Θ(B)εₜ
Θ(B) = 1 + θ₁B + ... + θqBq
```

**ARIMA(p,d,q):**
```
Φ(B)(1-B)ᵈ Yₜ = c + Θ(B) εₜ
```

**Maximum Likelihood Estimation:**
```
L(φ,θ,σ²|Y) = (2πσ²)^(-n/2) |P₁|^(1/2) exp(-1/(2σ²) Yᵀ P₁⁻¹ Y)
```
Where P₁ is the theoretical covariance matrix.

**Information Criteria for Order Selection:**
```
AIC = -2 log L̂ + 2k
AICc = AIC + 2k(k+1)/(n-k-1)   (corrected for small n)
BIC = -2 log L̂ + k log n
```
Minimise these; BIC penalises complexity more → tends to select simpler models.

### 9.4 SARIMAX(p,d,q)(P,D,Q,s)

**Full Model:**
```
Φₚ(B) Φₚ(Bˢ) ∇ᵈ ∇ˢᴰ Yₜ = c + Θq(B) Θq(Bˢ) εₜ + β Xₜ
```

**Seasonal AR operator:**
```
Φₚ(Bˢ) = 1 - Φ₁Bˢ - Φ₂B²ˢ - ... - ΦₚBᴾˢ
```

**Example (1,1,1)(1,1,1)₁₂ — monthly data:**
```
(1-φ₁B)(1-Φ₁B¹²)(1-B)(1-B¹²)Yₜ = (1+θ₁B)(1+Θ₁B¹²)εₜ
```

**Exogenous Variables (X):** External predictors added as linear regressors (e.g., holiday indicator, temperature, price).

### 9.5 Exponential Smoothing

**Simple (SES):**
```
ŷₜ₊₁ = α yₜ + (1-α) ŷₜ = α Σⱼ (1-α)ʲ yₜ₋ⱼ
```

**Holt's Double Exponential (trend):**
```
lₜ = α yₜ + (1-α)(lₜ₋₁ + bₜ₋₁)
bₜ = β(lₜ - lₜ₋₁) + (1-β)bₜ₋₁
ŷₜ₊ₕ = lₜ + h bₜ
```

**Holt-Winters Triple Exponential (trend + seasonality):**
```
lₜ = α(yₜ/sₜ₋ₘ) + (1-α)(lₜ₋₁+bₜ₋₁)
bₜ = β(lₜ-lₜ₋₁) + (1-β)bₜ₋₁
sₜ = γ(yₜ/lₜ) + (1-γ)sₜ₋ₘ
ŷₜ₊ₕ = (lₜ + hbₜ)sₜ₋ₘ₊h
```
(Multiplicative seasonality; additive version replaces divisions/products with +/-)

---

## 10. Deep Learning Foundations

### 10.1 Perceptron

The building block of neural networks:
```
z = Σⱼ wⱼ xⱼ + b = wᵀx + b
ŷ = g(z)
```

**Perceptron Learning Rule:**
```
wⱼ ← wⱼ + α(y - ŷ) xⱼ
```

**Limitation:** Can only learn linearly separable functions (XOR problem → multi-layer needed).

### 10.2 Multi-Layer Perceptron (MLP)

**Forward Propagation:**
```
Layer l:   zˡ = Wˡ aˡ⁻¹ + bˡ
           aˡ = g(zˡ)
Output:    a^L = ŷ
```

**Universal Approximation Theorem:**
A feedforward network with at least one hidden layer with a sufficient number of neurons can approximate any continuous function on a compact subset of ℝⁿ to arbitrary precision.

### 10.3 Activation Functions

**Sigmoid:**
```
σ(z) = 1/(1+e⁻ᶻ),     σ'(z) = σ(z)(1-σ(z))    ∈ (0,0.25]
```
Problems: Saturates (gradients → 0), not zero-centred.

**Tanh:**
```
tanh(z) = (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ) = 2σ(2z)-1,   tanh'(z) = 1-tanh²(z)  ∈ (0,1]
```
Zero-centred; still saturates.

**ReLU:**
```
f(z) = max(0,z),   f'(z) = 𝟙[z>0]
```
No saturation for z>0; sparse activation; fast. **Dying ReLU problem:** Neurons with z<0 always output 0.

**Leaky ReLU:**
```
f(z) = max(αz, z),  α ≈ 0.01,   f'(z) = α if z<0 else 1
```

**Parametric ReLU (PReLU):** α is a learnable parameter.

**ELU (Exponential Linear Unit):**
```
f(z) = z           if z ≥ 0
       α(eᶻ-1)     if z < 0

f'(z) = 1             if z ≥ 0
        f(z) + α      if z < 0
```
Negative saturation region pushes mean activations toward 0.

**GELU (Gaussian Error Linear Unit — used in Transformers/BERT):**
```
GELU(z) = z × Φ(z) = z × P(X ≤ z),   X ~ N(0,1)
         ≈ 0.5z(1 + tanh(√(2/π)(z + 0.044715z³)))
```

**Swish:**
```
f(z) = z × σ(z) = z/(1+e⁻ᶻ)
```

**Mish:**
```
f(z) = z × tanh(ln(1 + eᶻ))
```

### 10.4 Backpropagation — Full Derivation

**Goal:** Compute `∂L/∂Wˡ` and `∂L/∂bˡ` for all layers.

**Define error signal:**
```
δˡ = ∂L/∂zˡ   (how loss changes with pre-activation)
```

**Output layer (MSE, single output):**
```
δᴸ = ∂L/∂zᴸ = (∂L/∂aᴸ)(∂aᴸ/∂zᴸ) = (ŷ-y) × g'(zᴸ)
```

**Hidden layer (backprop step):**
```
δˡ = ((Wˡ⁺¹)ᵀ δˡ⁺¹) ⊙ g'(zˡ)
```

**Gradients:**
```
∂L/∂Wˡ = δˡ (aˡ⁻¹)ᵀ    →   Wˡ := Wˡ - α ∂L/∂Wˡ
∂L/∂bˡ = δˡ             →   bˡ := bˡ - α ∂L/∂bˡ
```

**Vanishing Gradient:** For sigmoid/tanh with many layers:
```
||δˡ|| ≈ Πₖ>ˡ ||Wᵏ|| × ||g'(zᵏ)||
```
Since σ'(z) ≤ 0.25 and tanh'(z) ≤ 1, the product shrinks → early layers learn extremely slowly.

**Solutions:** ReLU, Batch Norm, Residual connections, LSTM gates.

### 10.5 Optimisation Algorithms

**SGD:**
```
θ := θ - α ∇_θ L(θ; xᵢ, yᵢ)
```

**Mini-batch SGD:** Average gradient over batch of B samples.

**Momentum:**
```
v := βv + ∇L
θ := θ - α v
```
Accelerates in consistent gradient directions; dampens oscillations.

**Nesterov Accelerated Gradient (NAG):**
```
v := βv + ∇L(θ - βv)   (look ahead)
θ := θ - α v
```

**AdaGrad:**
```
G := G + (∇L)²
θ := θ - (α/√(G+ε)) ∇L
```
Adapts learning rate per parameter. Problem: G grows monotonically → learning rate → 0.

**RMSProp:**
```
G := β G + (1-β)(∇L)²     (exponential moving average)
θ := θ - (α/√(G+ε)) ∇L
```

**Adam (Adaptive Moment Estimation):**
```
m := β₁m + (1-β₁)∇L                          (1st moment)
v := β₂v + (1-β₂)(∇L)²                       (2nd moment)
m̂ := m/(1-β₁ᵗ);  v̂ := v/(1-β₂ᵗ)           (bias correction)
θ := θ - α m̂/(√v̂ + ε)
```
Defaults: β₁=0.9, β₂=0.999, ε=10⁻⁸, α=0.001

**AdamW (Adam with Weight Decay):**
```
θ := θ - α [m̂/(√v̂+ε) + λθ]
```
Decouples weight decay from gradient adaptation (fixes L2 regularisation in Adam).

**Learning Rate Schedules:**
```
Step decay:       α(t) = α₀ × γ^{floor(t/k)}
Exponential:      α(t) = α₀ × e^{-λt}
Cosine annealing: α(t) = αₘᵢₙ + (αₘₐₓ-αₘᵢₙ)(1+cos(πt/T))/2
Warmup + decay:   increase for first W steps, then decay
```

### 10.6 Weight Initialisation

**Problem:** Wrong scale → vanishing or exploding gradients from the start.

**Xavier/Glorot Initialisation (for Tanh/Sigmoid):**
```
W ~ U[-√(6/(nᵢₙ+nₒᵤₜ)), √(6/(nᵢₙ+nₒᵤₜ))]
Var(W) = 2/(nᵢₙ+nₒᵤₜ)
```

**He Initialisation (for ReLU):**
```
W ~ N(0, 2/nᵢₙ)
Var(W) = 2/nᵢₙ
```
Factor of 2 compensates for ReLU zeroing out half the activations.

---

## 11. Convolutional Neural Networks (CNN)

### 11.1 The Convolution Operation

**2D Discrete Convolution:**
```
(I * K)(i,j) = Σₘ Σₙ I(i+m, j+n) × K(m,n)
```
(In practice, deep learning uses cross-correlation — no kernel flip — but calls it convolution.)

**Output Spatial Dimensions:**
```
H_out = floor((H_in + 2P - K) / S) + 1
W_out = floor((W_in + 2P - K) / S) + 1
```
Where H=height, W=width, P=padding, K=kernel size, S=stride.

**Number of Parameters:**
```
Params_conv = (K × K × C_in + 1) × C_out
```

**Receptive Field:**
```
RF_l = RF_{l-1} + (Kₗ - 1) × Πₖ<ₗ Sₖ
```

### 11.2 Pooling

**Max Pooling:**
```
y(i,j) = max_{0≤m,n<p} x(i×s+m, j×s+n)
```
Provides translation invariance; reduces spatial size.

**Average Pooling:**
```
y(i,j) = (1/p²) Σₘ Σₙ x(i×s+m, j×s+n)
```

**Global Average Pooling (GAP):**
```
y_c = (1/H×W) Σᵢ Σⱼ x_c(i,j)
```
Replaces fully connected layers → drastically reduces parameters; reduces overfitting.

### 11.3 Batch Normalisation

Applied before activation:
```
μ_B = (1/m) Σᵢ xᵢ
σ²_B = (1/m) Σᵢ (xᵢ-μ_B)²
x̂ᵢ = (xᵢ-μ_B) / √(σ²_B+ε)
yᵢ = γ x̂ᵢ + β
```
γ and β are learnable. At test time, uses running mean/variance.

**Why it works:**
- Reduces internal covariate shift
- Acts as regulariser (reduces need for Dropout)
- Allows higher learning rates

### 11.4 Notable CNN Architectures

**AlexNet (2012):**
- 5 conv layers + 3 FC; 60M params
- First use of ReLU and Dropout in CNNs
- Top-5 error: 15.3% on ImageNet

**VGG-16 (2014):**
- 13 conv + 3 FC; 138M params; all 3×3 kernels
- Two 3×3 convs have same RF as one 5×5 but fewer params and more non-linearity
- Params: 2×(3²×C) < 5²×C

**GoogLeNet / Inception (2014):**
- Inception module: parallel 1×1, 3×3, 5×5 convs + MaxPool
- 1×1 convolutions for dimensionality reduction (bottleneck)
- 22 layers, only 5M params (vs AlexNet 60M)

**ResNet (2015):**
- **Residual/Skip Connection:**
  ```
  y = F(x, {Wᵢ}) + x
  ```
  If identity is optimal, F(x) → 0 (easier to learn).
- Enables 150+ layer networks.
- Gradient flows directly through skip: `∂L/∂x = ∂L/∂y (1 + ∂F/∂x)` — the "+1" prevents vanishing.

**DenseNet (2017):**
- Every layer connected to all subsequent layers:
  ```
  xˡ = Hˡ([x⁰, x¹, ..., xˡ⁻¹])
  ```
- Alleviates vanishing gradient; feature reuse; fewer params.

**EfficientNet (2019):**
- Compound scaling of depth/width/resolution simultaneously:
  ```
  depth: d = α^φ,  width: w = β^φ,  resolution: r = γ^φ
  s.t. α×β²×γ² ≈ 2
  ```

### 11.5 Object Detection Architectures

**YOLO (You Only Look Once):** Single forward pass; divides image into S×S grid; each cell predicts B bounding boxes + confidence + C class probabilities.
```
Confidence = P(Object) × IoU(pred, truth)
IoU = Area(A∩B) / Area(A∪B)
```

**Loss (YOLO):**
```
L = λ_coord Σ 1_{obj}[(x-x̂)² + (y-ŷ)²]
  + λ_coord Σ 1_{obj}[(√w-√ŵ)² + (√h-√ĥ)²]
  + Σ 1_{obj}(C-Ĉ)² + λ_noobj Σ 1_{noobj}(C-Ĉ)²
  + Σ 1_{obj} Σ_c (p(c)-p̂(c))²
```

---

## 12. Recurrent Neural Networks (RNN)

### 12.1 Vanilla RNN

**Forward Pass:**
```
hₜ = tanh(Wₕ hₜ₋₁ + Wₓ xₜ + bₕ)
ŷₜ = softmax(Wᵧ hₜ + bᵧ)
```

**One-to-One:** Standard feedforward.
**One-to-Many:** Image captioning (single image → sequence of words).
**Many-to-One:** Sentiment analysis (sequence → single label).
**Many-to-Many (equal):** POS tagging.
**Many-to-Many (unequal):** Machine translation (encoder-decoder).

### 12.2 Backpropagation Through Time (BPTT)

**Unroll RNN for T steps; apply backprop:**
```
∂L/∂Wₓ = Σₜ ∂Lₜ/∂Wₓ = Σₜ Σₖ≤ₜ (∂Lₜ/∂hₜ)(Πᵢ=k+1..t ∂hᵢ/∂hᵢ₋₁)(∂hₖ/∂Wₓ)
```

**Jacobian of hidden state:**
```
∂hₜ/∂hₜ₋₁ = diag(tanh'(zₜ)) Wₕ
```

**For long sequences:**
```
||Πᵢ ∂hᵢ/∂hᵢ₋₁|| ≤ (||Wₕ|| × max tanh'(z))^T
```
- If `||Wₕ|| × max tanh'(z) < 1` → **vanishing gradients** (no long-range dependencies)
- If `> 1` → **exploding gradients** → use **gradient clipping**:
  ```
  g := g × min(1, threshold/||g||)
  ```

**Truncated BPTT:** Only backpropagate k steps to limit memory/computation.

---

## 13. Long Short-Term Memory (LSTM) & GRU

### 13.1 LSTM — Full Equations

**Input concatenation:**
```
concat = [hₜ₋₁; xₜ]   (row vector of size nₕ + nₓ)
```

**Forget Gate** (what to erase from cell):
```
fₜ = σ(Wf [hₜ₋₁; xₜ] + bf)
```

**Input Gate** (what new info to add):
```
iₜ = σ(Wᵢ [hₜ₋₁; xₜ] + bᵢ)
```

**Candidate Cell State:**
```
C̃ₜ = tanh(Wc [hₜ₋₁; xₜ] + bc)
```

**Cell State Update:**
```
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
```

**Output Gate:**
```
oₜ = σ(Wo [hₜ₋₁; xₜ] + bo)
hₜ = oₜ ⊙ tanh(Cₜ)
```

**Parameter Count:**
```
Total params = 4 × (nₕ × nₓ + nₕ × nₕ + nₕ)   (4 gates, each with W and b)
```

**Why LSTMs Solve Vanishing Gradient:**
The gradient through the cell state:
```
∂Cₜ/∂Cₜ₋₁ = fₜ   (element-wise — no matrix multiplication)
```
If fₜ ≈ 1 (forget gate open), gradient flows unchanged → no vanishing over long sequences.
Compare RNN: `∂hₜ/∂hₜ₋₁ = diag(tanh') Wₕ` — repeated matrix multiplication → vanish/explode.

### 13.2 GRU (Gated Recurrent Unit)

Simpler than LSTM with 2 gates instead of 3:

**Reset Gate:**
```
rₜ = σ(Wr [hₜ₋₁; xₜ] + br)
```

**Update Gate:**
```
zₜ = σ(Wz [hₜ₋₁; xₜ] + bz)
```

**Candidate Hidden State:**
```
h̃ₜ = tanh(Wh [rₜ ⊙ hₜ₋₁; xₜ] + bh)
```

**Hidden State Update:**
```
hₜ = (1-zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
```

**Interpretation:**
- Update gate zₜ interpolates between old and new state
- Reset gate rₜ allows forgetting past hidden state when irrelevant
- Fewer parameters than LSTM; comparable performance on many tasks

**Parameter Count:**
```
Total params = 3 × (nₕ × nₓ + nₕ × nₕ + nₕ)   (3 gates)
```

### 13.3 Bidirectional RNN

Processes sequence in both directions:
```
→hₜ = RNN_forward(xₜ, →hₜ₋₁)
←hₜ = RNN_backward(xₜ, ←hₜ₊₁)
hₜ = [→hₜ; ←hₜ]   (concatenate)
```
Captures both past and future context. Used in BERT, NER, etc.

### 13.4 Encoder-Decoder (Seq2Seq)

**Encoder:**
```
hₜ = f(hₜ₋₁, xₜ);   context vector c = hₜ_final
```

**Decoder:**
```
sₜ = g(sₜ₋₁, yₜ₋₁, c);   ŷₜ = softmax(Vs sₜ)
```

**Bottleneck Problem:** Entire source sentence compressed into single vector c.

**Solution — Bahdanau Attention:**
```
eₜⱼ = vᵀ tanh(Wₛ sₜ₋₁ + Wₕ hⱼ)      (alignment score)
αₜⱼ = softmax(eₜ)_j                   (attention weight)
cₜ = Σⱼ αₜⱼ hⱼ                        (context vector — weighted sum)
```
Decoder can attend to any encoder hidden state → no bottleneck.

---

## 14. Advanced Deep Learning

### 14.1 Attention Mechanism & Transformers

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
```
- Q (Queries), K (Keys), V (Values) are linear projections of input
- `√dₖ` scaling prevents dot products from growing large → softmax saturation

**Multi-Head Attention:**
```
MultiHead(Q,K,V) = Concat(head₁,...,headₕ) Wₒ
headᵢ = Attention(Q Wᵢᴼ, K Wᵢᴷ, V Wᵢᵛ)
```
Each head learns different aspects of relationships.

**Transformer Block:**
```
x' = LayerNorm(x + MultiHeadAttn(x,x,x))     (self-attention + residual)
x'' = LayerNorm(x' + FFN(x'))                 (feedforward + residual)
FFN(x) = max(0, xW₁+b₁)W₂ + b₂
```

**Positional Encoding (sinusoidal):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 14.2 Autoencoders

**Architecture:**
```
Encoder: z = f(x) = g(Wₑ x + bₑ)   (compress to latent space)
Decoder: x̂ = h(z) = g(Wᵈ z + bᵈ)   (reconstruct)
Loss:    L = ||x - x̂||²             (reconstruction error)
```

**Variational Autoencoder (VAE):**
Learns a distribution over the latent space:
```
Encoder outputs: μ(x), σ(x)
z ~ N(μ(x), σ²(x))
```

**ELBO Loss:**
```
L = E[log P(x|z)] - KL(q(z|x) || P(z))
  = Reconstruction loss - KL divergence
```

**Reparameterisation Trick (allows backprop through sampling):**
```
z = μ + σ ⊙ ε,   ε ~ N(0, I)
```

### 14.3 Regularisation Techniques

**Dropout:**
```
hᵢ = gᵢ × rᵢ,   rᵢ ~ Bernoulli(p)
```
At test time: `h = p × g` (scale by keep probability).

**Inverted Dropout (scale during training):**
```
h = (g × r) / p   (scale during training, no scaling at test)
```

**L1 Regularisation:**
```
L_total = L_data + λ Σⱼ |wⱼ|
∂L_total/∂wⱼ = ∂L_data/∂wⱼ + λ sign(wⱼ)
```

**L2 Regularisation:**
```
L_total = L_data + (λ/2) Σⱼ wⱼ²
∂L_total/∂wⱼ = ∂L_data/∂wⱼ + λ wⱼ
```

**Early Stopping:**
Monitor validation loss; stop when it starts increasing. Keeps model at the minimum of validation error.
```
Best epoch = argmin_t L_val(t)
```

**Data Augmentation:**
- Image: flip, rotate, crop, colour jitter, cutout
- Text: synonym replacement, back-translation
- Audio: time stretch, pitch shift, add noise

**Label Smoothing:**
```
y_smooth = (1-ε) y_true + ε/K
```
Prevents overconfident predictions; improves calibration.

---

## 15. Handling Imbalanced Data

### 15.1 Understanding Class Imbalance

**Imbalance Ratio:**
```
IR = N_majority / N_minority
```
IR > 10 → mildly imbalanced; IR > 100 → severely imbalanced.

**Why standard accuracy fails:**
- 99% majority class → predict always majority → 99% accuracy, 0% recall on minority!
- Use: F1, AUC-ROC, AUC-PR, MCC.

### 15.2 SMOTE and Variants

**SMOTE:**
```
x_synthetic = xᵢ + λ(xⱼ - xᵢ),   λ ~ Uniform(0,1)
```
xⱼ = randomly chosen KNN of xᵢ.

**SMOTE-NC (Nominal and Continuous):** Modified distance for mixed data types.

**ADASYN (Adaptive Synthetic Sampling):**
```
Generate more synthetic samples for minority instances that are harder to classify (i.e., surrounded by majority class neighbours).
rᵢ = Δᵢ / K   (fraction of majority neighbours)
nᵢ = rᵢ / Σ rᵢ × G   (number of synthetic samples for xᵢ)
```

**SMOTE-ENN (cleaning step):** Apply SMOTE then remove ambiguous samples using Edited Nearest Neighbours.

**Borderline-SMOTE:** Only oversample minority instances near the decision boundary.

### 15.3 Undersampling Methods

**Tomek Links:** Two samples (one minority, one majority) form a Tomek link if no third sample lies between them. Remove majority sample.

**NearMiss-1:** Keep majority samples whose average distance to K nearest minority samples is smallest.

**NearMiss-2:** Keep majority samples whose average distance to K farthest minority samples is smallest.

**Condensed Nearest Rule:** Iteratively remove majority samples that don't affect the decision boundary.

### 15.4 Cost-Sensitive Learning

**Modified Loss with Class Weights:**
```
L = -Σᵢ wᵧᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

**Inverse Frequency Weighting:**
```
wₖ = n / (K × nₖ)
```

**Focal Loss (for extreme imbalance — used in RetinaNet):**
```
FL(p) = -α(1-p)^γ log(p)
```
- α: class weight; γ: focusing parameter (γ=2 typical)
- (1-p)^γ down-weights easy negatives → focuses on hard positives

### 15.5 Threshold Optimisation

Default threshold 0.5 is not optimal for imbalanced data. Choose threshold to optimise desired metric:

**Precision-Recall Trade-off:**
```
For threshold τ:
  ŷ = 1 if P(y=1|x) ≥ τ, else 0
```

**F1-Optimal Threshold:**
```
τ* = argmax_τ F1(τ)
```

**Youden's J Statistic:**
```
J = Sensitivity + Specificity - 1 = TPR - FPR
τ* = argmax_τ J(τ)
```

---

## 16. Model Evaluation & Selection

### 16.1 Classification Metrics

**Confusion Matrix:**
```
              Predicted
              Pos    Neg
Actual  Pos | TP  |  FN |
        Neg | FP  |  TN |
```

**Core Metrics:**
```
Accuracy      = (TP+TN) / (TP+TN+FP+FN)
Precision     = TP / (TP+FP)
Recall        = TP / (TP+FN)
Specificity   = TN / (TN+FP)
F1            = 2×Precision×Recall / (Precision+Recall)
F-β           = (1+β²)×P×R / (β²×P + R)
```

**Matthews Correlation Coefficient (balanced metric):**
```
MCC = (TP×TN - FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]
```
Ranges [-1,+1]; +1 = perfect, 0 = random.

**Cohen's Kappa (accounts for chance agreement):**
```
κ = (pₒ - pₑ) / (1 - pₑ)
```
Where pₒ = observed agreement, pₑ = expected agreement by chance.

**ROC Curve:**
```
TPR (Recall)  = TP/(TP+FN)   vs   FPR = FP/(FP+TN)  at varying thresholds
```

**AUC (Area Under ROC Curve):**
- AUC = 1.0: perfect classifier
- AUC = 0.5: random classifier
- Probabilistic interpretation: P(score of random positive > score of random negative)

**Precision-Recall Curve:** Better than ROC for highly imbalanced data.

**Average Precision:**
```
AP = Σₙ (Rₙ - Rₙ₋₁) Pₙ
```

### 16.2 Regression Metrics

```
MAE  = (1/n) Σ|yᵢ - ŷᵢ|
MSE  = (1/n) Σ(yᵢ - ŷᵢ)²
RMSE = √MSE
MAPE = (100/n) Σ|yᵢ-ŷᵢ|/|yᵢ|
R²   = 1 - SS_res/SS_tot = 1 - Σ(yᵢ-ŷᵢ)² / Σ(yᵢ-ȳ)²
R²_adj = 1 - (1-R²)(n-1)/(n-p-1)
```

**Huber Loss (robust to outliers):**
```
L_δ(y, ŷ) = (1/2)(y-ŷ)²            if |y-ŷ| ≤ δ
             δ|y-ŷ| - (1/2)δ²       otherwise
```

### 16.3 Cross-Validation

**k-Fold CV:**
```
CV_score = (1/K) Σₖ metric(model_k, fold_k)
```

**Stratified k-Fold:** Preserves class ratio in each fold.

**Leave-One-Out CV (LOOCV):** K = n. Nearly unbiased but high variance; expensive.

**Repeated k-Fold:** Run k-fold multiple times with different random splits; average.

**Time Series CV (Walk-Forward):**
```
Train: [1..t]      Test: [t+1]
Train: [1..t+1]    Test: [t+2]
...  (never test on past data)
```

### 16.4 Hyperparameter Tuning

**Grid Search:** Exhaustive search over all combinations.
```
Complexity: O(Πᵢ |Hᵢ|)    (product of hyperparameter grid sizes)
```

**Random Search:** Random combinations. More efficient than grid search when some hyperparams matter more than others (Bergstra & Bengio, 2012).

**Bayesian Optimisation:**
```
Surrogate model (Gaussian Process): P(f | observations)
Acquisition function: α(x) = E[max(0, f(x) - f*)]  (Expected Improvement)
Next evaluation: x_next = argmax_x α(x)
```

**Hyperband:** Combines random search with successive halving — early stop bad configurations.

### 16.5 Bias-Variance Diagnostic

| Training Error | Test Error | Diagnosis | Fix |
|---|---|---|---|
| High | High | High Bias (underfitting) | More complex model; more features |
| Low | High | High Variance (overfitting) | Regularisation; more data; simpler model |
| Low | Low | Good fit | — |
| High | Low | Data mismatch | Check data distributions |

**Learning Curves:** Plot train/val loss vs training set size.
- High bias: both curves plateau at high error.
- High variance: large gap between train and val curves.

---

## 17. Regularisation & Optimisation

### 17.1 The Norm Zoo

| Norm | Formula | Effect in Regularisation |
|---|---|---|
| L0 | count(wⱼ ≠ 0) | Exact sparsity (NP-hard) |
| L1 (Lasso) | Σ|wⱼ| | Sparse solution; feature selection |
| L2 (Ridge) | Σwⱼ² | Shrinks all weights; no sparsity |
| Elastic Net | αL1 + (1-α)L2 | Sparse + stable with correlated features |

**Geometric Interpretation:**
```
Constrained form: min L(β) s.t. ||β||ₚ ≤ t
Lagrangian form:  min L(β) + λ||β||ₚ
```
L1 ball has corners → solutions at corners have zero coordinates (sparse).
L2 ball is smooth → solutions on smooth surface (no sparsity).

### 17.2 Batch Normalisation Deep Dive

**Inference:** Use running statistics (exponential moving average):
```
μ_running = β × μ_running + (1-β) × μ_batch
σ²_running = β × σ²_running + (1-β) × σ²_batch
```

**Layer Normalisation (used in Transformers — normalises over features, not batch):**
```
x̂ᵢ = (xᵢ - μ_layer) / √(σ²_layer + ε),   μ_layer = (1/d) Σⱼ xⱼ
```

**Instance Normalisation (per-sample, per-channel — used in style transfer):**
```
Normalise over spatial dimensions H,W for each sample and channel.
```

**Group Normalisation (divides channels into groups; good for small batches):**
```
Normalise over (group_channels × H × W) for each sample.
```

### 17.3 Gradient Clipping

**Value Clipping:**
```
g := clip(g, -c, c)
```

**Norm Clipping (preserves direction):**
```
g := g × min(1, c/||g||)
```

---

## 18. Probabilistic Machine Learning

### 18.1 Gaussian Processes (GP)

A GP is a distribution over functions:
```
f(x) ~ GP(m(x), k(x, x'))
```
Where m(x) is the mean function and k(x,x') is the covariance (kernel) function.

**Posterior (given observations y at X):**
```
f* | X, y, X* ~ N(μ*, Σ*)
μ* = K*X (KXX + σ²I)⁻¹ y
Σ* = K** - K*X (KXX + σ²I)⁻¹ KX*
```
Provides uncertainty estimates (predictive variance).

### 18.2 Hidden Markov Models (HMM)

**States:** `z₁, z₂, ..., zₜ` (latent, discrete)
**Observations:** `x₁, x₂, ..., xₜ` (observed)

**Transition probabilities:** `P(zₜ | zₜ₋₁) = Aᵢⱼ`
**Emission probabilities:** `P(xₜ | zₜ) = Bᵢ(xₜ)`
**Initial distribution:** `P(z₁) = πᵢ`

**Three Problems:**
1. **Evaluation** (likelihood): Forward Algorithm → O(TN²)
2. **Decoding** (most likely states): Viterbi Algorithm → O(TN²)
3. **Learning** (parameters): Baum-Welch (special case of EM)

### 18.3 Calibration

A probabilistic model is **calibrated** if:
```
P(y=1 | ŷ = p) = p   for all p ∈ [0,1]
```
I.e., when the model says 70% confidence, it should be correct 70% of the time.

**Reliability Diagram:** Plot mean predicted probability vs actual fraction of positives per bin.

**Calibration Methods:**
- **Platt Scaling:** Fit sigmoid to model outputs.
- **Isotonic Regression:** Non-parametric monotonic calibration.
- **Temperature Scaling (for neural nets):** `P(y=k|x) = softmax(z/T)` — tune scalar T.

---

## 19. Repository Structure

```
📦 ravidaliparthy/
 ┃
 ┣━━ 📂 Regression
 ┃    ┣━━ 📓 ANN for regression 3.ipynb
 ┃    ┗━━ 📓 house pricing 2.ipynb
 ┃
 ┣━━ 📂 Classification
 ┃    ┣━━ 📓 KNN_Practice (1).ipynb
 ┃    ┣━━ 📓 Naive Bayes.ipynb
 ┃    ┣━━ 📓 diabetes prediction 1.ipynb
 ┃    ┣━━ 📓 credit_card_fraud_detection 3.ipynb
 ┃    ┗━━ 📓 email spam 5.ipynb
 ┃
 ┣━━ 📂 Ensemble Learning
 ┃    ┗━━ 📓 Machine Learning & Majority Voting.ipynb
 ┃
 ┣━━ 📂 Deep Learning — ANN
 ┃    ┣━━ 📓 ANN for Mnist 3.ipynb
 ┃    ┗━━ 📓 ANN for regression 3.ipynb
 ┃
 ┣━━ 📂 Deep Learning — CNN
 ┃    ┣━━ 📓 CNN for Mnist 3.ipynb
 ┃    ┣━━ 📓 cifar-cnn 1.ipynb
 ┃    ┗━━ 📓 imageprocessing.ipynb
 ┃
 ┣━━ 📂 Deep Learning — RNN/LSTM
 ┃    ┗━━ 📓 LSTM 1.ipynb
 ┃
 ┣━━ 📂 Time Series
 ┃    ┣━━ 📓 Arima & Sarimax 1.ipynb
 ┃    ┗━━ 📓 Temperature-Forecasting 1.ipynb
 ┃
 ┣━━ 📂 Preprocessing & Balancing
 ┃    ┣━━ 📓 Oversampling & undersampling.ipynb
 ┃    ┣━━ 📓 heartdisease Data balancing 1.ipynb
 ┃    ┗━━ 📓 correlation.ipynb
 ┃
 ┣━━ 📂 Computer Vision
 ┃    ┗━━ 🐍 body_detection.py
 ┃
 ┣━━ 📂 Recommendation & NLP
 ┃    ┗━━ 📓 movie recommendation 2.ipynb
 ┃
 ┣━━ 📂 Miscellaneous
 ┃    ┣━━ 📓 car details.ipynb
 ┃    ┣━━ 📓 updated_pollution.ipynb
 ┃    ┗━━ 📓 train.ipynb
 ┃
 ┗━━ 📄 README.md
```

---

## 📚 References

- Bishop, C. M. — *Pattern Recognition and Machine Learning* (2006)
- Goodfellow, I., Bengio, Y., Courville, A. — *Deep Learning* (deeplearningbook.org)
- Géron, A. — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3rd ed.)
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning* (free PDF)
- Murphy, K. P. — *Probabilistic Machine Learning* (2022, free PDF)
- Nielsen, M. — *Neural Networks and Deep Learning* (neuralnetworksanddeeplearning.com)
- Szeliski, R. — *Computer Vision: Algorithms and Applications*
- [Scikit-learn](https://scikit-learn.org) | [TensorFlow](https://tensorflow.org) | [PyTorch](https://pytorch.org) | [Keras](https://keras.io)
- [Papers With Code](https://paperswithcode.com) | [Arxiv ML](https://arxiv.org/list/cs.LG/recent)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat)
![XGBoost](https://img.shields.io/badge/XGBoost-orange?style=flat)

---

*Made with ❤️ by [@ravidaliparthy](https://github.com/ravidaliparthy) — Star ⭐ this repo if it helped you!*
