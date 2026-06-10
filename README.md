# 🤖 Machine Learning, Deep Learning & LLM Engineering
### The Most Complete Beginner-to-Production Reference — Zero to Expert

> **Every algorithm. Every derivation. Every formula. Every intuition.**
> Written so that ANYONE — even with zero prior knowledge — can understand.
> Theory → Intuition → Math → Visualisation → Code → Production.

---

## 🧭 How to Use This Guide

**If you are completely new to machine learning:**
- Read every "💡 What this means in plain English" block first
- Follow the ASCII diagrams to build visual intuition
- Don't skip the analogies — they are the foundation
- The math will start making sense once you have the intuition

**If you already have some background:**
- Use the detailed derivations and proofs
- Focus on the "Why it works" sections
- The production notes and failure modes are especially useful

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
15. [Transformers — Complete Deep Dive](#15-transformers—complete-deep-dive)
16. [Natural Language Processing (NLP)](#16-Natural-Language-Processing-(NLP))
17. [LangChain & LLM Application Engineering](#17-LangChain-&-LLM-Application-Engineering)
18. [Handling Imbalanced Data](#18Handling-Imbalanced-Data)



---

## 1. Mathematical Foundations

> 💡 **Why math?** Think of math as the language machines use to learn. Just like you can't read a French novel without knowing French, you can't truly understand ML without a little math. But don't worry — we'll explain every symbol and every idea from scratch, building up your intuition step by step.

---

### 1.1 Linear Algebra

> 💡 **What is Linear Algebra?** Linear algebra is the study of vectors and matrices — tools for organising and transforming large amounts of numbers efficiently. In ML, your dataset of 1,000 customers each described by 50 features is just a matrix of numbers. All the learning happens through mathematical operations on these matrices.

---

#### Vectors and Norms

> 💡 **What is a vector?** A vector is simply an ordered list of numbers. For example, `[height=170, weight=65, age=25]` is a 3-dimensional vector describing a person. In machine learning, every data point (a customer, a patient, an image) is represented as a vector of numbers.

```
Person A = [170, 65, 25]   ← a 3-dimensional vector (height, weight, age)
Person B = [180, 80, 30]   ← another 3-dimensional vector

A house could be: [bedrooms=3, bathrooms=2, area=1500, price=300000]
An image (28×28 pixels): [pixel1=0.2, pixel2=0.8, pixel3=0.1, ..., pixel784=0.5]
```

> 💡 **What is a norm?** A norm measures the "size" or "length" of a vector. Just like you can measure the length of an arrow on paper, you can measure the "magnitude" of a data vector. Different norms measure size differently — like measuring the same journey "as the crow flies" (straight line) vs "along city streets" (sum of blocks walked).

**Why norms matter for Machine Learning:**
When training a model, we often want to keep the model's parameters (weights) small to prevent overfitting. We add a "penalty" to the loss function that penalises large weights. The *type* of penalty we use (which norm) dramatically changes the behaviour of the model — specifically, whether the model sets some weights to exactly zero (sparse) or just makes them all small.

**Visualising different norms — the unit ball in 2D:**

```
The "unit ball" is the set of all points with norm ≤ 1.
The SHAPE of this ball determines how the model behaves.

         β₂
          |
    ┌─────┼─────┐   ← L∞ norm ball (square): max(|β₁|,|β₂|) ≤ 1
    │  ╭──┼──╮  │
    │ ╱   │   ╲ │   ← L2 norm ball (circle): β₁²+β₂² ≤ 1
    │╱    │    ╲│
────┼─────┼─────┼──── β₁
    │╲    │    ╱│
    │ ╲   │   ╱ │
    │  ╰──┼──╯  │
    │    ╱│╲    │   ← L1 norm ball (diamond): |β₁|+|β₂| ≤ 1
    └───╱─┼─╲───┘
       ╱  │  ╲
      ●   │   ●    ← corners of L1 diamond are ON the axes!
```

**Why the L1 diamond creates sparse solutions:**
```
Imagine minimising a loss function (elliptical contours) subject to a norm constraint.
The optimal solution is where the contour first TOUCHES the constraint region:

   L1 constraint (diamond):      L2 constraint (circle):
        β₂                              β₂
         │      ╭──╮                    │      ╭──╮
         │ ╲   ╱    ╲                   │  ╲  ╱    ╲
         ●←─╲╱      ╲                   │   ╲╱      ╲
         │   ╲◎       ╲                 │  ╭╱●        ╲
    ─────┼────╲─────────  β₁       ─────┼──╱────────────  β₁
         │     ╲                        │ ╱
         │                              │

    ◎ hits a CORNER (β₂=0)!          ● hits the smooth curve
    → one weight is exactly ZERO!     → weights are shrunk but nonzero
    → SPARSE model (Lasso)            → Ridge regression
```

The L1 ball has sharp **corners** exactly on the coordinate axes. When your loss function's elliptical contours expand outward, they almost always hit one of these corners first — making one or more coefficients **exactly zero**. This is the geometric reason L1 regularisation (Lasso) creates sparse models, selecting only the most important features.

```
L0 norm:  ||x||₀ = count(xᵢ ≠ 0)             — number of non-zero elements
L1 norm:  ||x||₁ = Σᵢ |xᵢ|                   — Manhattan distance
L2 norm:  ||x||₂ = √(Σᵢ xᵢ²)                 — Euclidean distance
L∞ norm:  ||x||∞ = maxᵢ |xᵢ|                  — Chebyshev distance (max absolute value)
Lp norm:  ||x||ₚ = (Σᵢ |xᵢ|ᵖ)^(1/p)
```

> 💡 **Manhattan distance (L1)** is called that because in Manhattan (NYC), you can only walk along grid streets — you can't cut diagonally through buildings. Your total walking distance = blocks east/west + blocks north/south = sum of absolute differences.

> 💡 **Euclidean distance (L2)** is the straight-line "as the crow flies" distance. It's what most people mean by "distance." The Pythagorean theorem gives us this: √(Δx² + Δy²) in 2D.

> 💡 **Chebyshev distance (L∞)** is just the largest single-coordinate difference. Example: to move a chess king from (0,0) to (3,5), it takes max(3,5)=5 moves (it can move diagonally). This is the L∞ distance.

**Concrete example with numbers:**
```
Vector x = [3, -4]  (a 2D vector)

L0 norm:  ||x||₀ = 2        (both components are non-zero)
L1 norm:  ||x||₁ = |3|+|-4| = 7
L2 norm:  ||x||₂ = √(9+16)  = √25 = 5  ← the hypotenuse!
L∞ norm:  ||x||∞ = max(3,4)  = 4
```

**Triangle Inequality (foundation of all distance reasoning):**
```
||x + y||ₚ ≤ ||x||ₚ + ||y||ₚ   (holds for all p ≥ 1)
||x - z||ₚ ≤ ||x - y||ₚ + ||y - z||ₚ  (metric property)
```

> 💡 **In plain English:** The shortcut is never longer than the detour. Going from city A to city C directly is never longer than going A→B→C via some intermediate city B. This is fundamental to all distance-based algorithms (KNN clustering, etc.) — without it, "nearest neighbour" wouldn't make consistent sense.

**Dual Norms:**
```
The dual norm of ||·||ₚ is ||·||_q  where 1/p + 1/q = 1
Dual of L1 = L∞
Dual of L2 = L2  (self-dual)

Hölder's inequality:  |xᵀy| ≤ ||x||ₚ · ||y||_q
Cauchy-Schwarz (p=q=2):  |xᵀy| ≤ ||x||₂ · ||y||₂
```

> 💡 **Cauchy-Schwarz** says the dot product of two vectors is at most the product of their lengths. Equality holds when the vectors point in the same direction. This is why cosine similarity (dot product divided by both lengths) is always between -1 and +1.

The dual norm appears in subgradient analysis. The subdifferential of `||x||₁` at a non-zero point is `sign(x)`; at zero it is the entire dual ball `{v : ||v||∞ ≤ 1}`. This is why coordinate descent on L1-penalised objectives leads to soft-thresholding operators — a direct algebraic consequence of the geometry.

---

#### Matrix Operations

> 💡 **What is a matrix?** A matrix is a rectangular grid of numbers. Your entire dataset IS a matrix: rows are data points (customers, patients, images), columns are features (age, income, pixel values). When a machine learning algorithm "learns from data," it is doing matrix algebra operations on this grid of numbers.

```
Example dataset as a matrix (3 people, 4 features):

         Age  Income  Score  Approved?
Alice  [  25   50000    720     1    ]
Bob    [  30   80000    680     0    ]   ← Data matrix X (3×4)
Carol  [  22   35000    750     1    ]

Last column (Approved?) is the target vector y:
y = [1, 0, 1]
```

**Key matrix operations and what they mean:**

```
Transpose (Aᵀ):          Flip rows and columns
  A (3×4) → Aᵀ (4×3)
  
  A = [1  2  3]       Aᵀ = [1  4]
      [4  5  6]             [2  5]
                            [3  6]
  Use case: XᵀX gives the "feature correlation" matrix

Inverse (A⁻¹):           Like dividing for matrices: A × A⁻¹ = I
  Only square, non-singular (full-rank) matrices have inverses
  Use case: solving systems of equations (linear regression: β = (XᵀX)⁻¹Xᵀy)

Pseudo-inverse (A⁺):      Works for any matrix (not just square)
  A⁺ = VΣ⁺Uᵀ             via SVD decomposition
  Use case: minimum-norm solution when system is overdetermined

Trace: tr(A) = Σᵢ Aᵢᵢ     Sum of diagonal elements
  A = [3  1]    tr(A) = 3 + 5 = 8
      [2  5]
  tr(A) = sum of eigenvalues
  Use case: effective degrees of freedom = tr(H) for projection matrix H

Determinant: det(A) = Πᵢ λᵢ    Product of eigenvalues
  det ≠ 0 → matrix is invertible
  det = 0 → matrix is singular (features are perfectly correlated → problem!)
  Use case: detecting multicollinearity

Frobenius norm: ||A||_F = √(Σᵢ Σⱼ Aᵢⱼ²)    "length" of a matrix
  Like the L2 norm but for matrices — square root of sum of all squared elements
  Use case: measuring how much weights changed, nuclear norm regularisation

Nuclear norm: ||A||_* = Σᵢ σᵢ    Sum of singular values
  Convex relaxation of matrix rank — used to enforce low-rank structure
  Use case: matrix completion (Netflix recommendations), multi-task learning
```

**Matrix Calculus Identities (essential for deriving ML algorithms):**

> 💡 **Why matrix calculus?** To train any ML model, we need to minimise a loss function over a matrix of weights. Instead of differentiating one weight at a time (which would take forever for a 1M-parameter model), matrix calculus gives us compact formulas that compute all derivatives simultaneously.

```
tr(AB) = tr(BA)                        (cyclic property — very useful in proofs)

∂/∂A tr(AB)      = Bᵀ                 (derivative of trace w.r.t. matrix)
∂/∂A tr(AᵀA)     = 2A                 (derivative of squared Frobenius norm)
∂/∂x (xᵀAx)      = (A + Aᵀ)x = 2Ax   (derivative of quadratic form; 2Ax if A symmetric)
∂/∂x (aᵀx)       = a                  (derivative of linear form = the coefficient vector)
∂/∂A det(A)       = det(A) A⁻ᵀ        (derivative of determinant)
```

**Worked example — OLS linear regression gradient:**
```
Loss: L(β) = ||y - Xβ||² = (y-Xβ)ᵀ(y-Xβ)
           = yᵀy - 2βᵀXᵀy + βᵀXᵀXβ

∂L/∂β = -2Xᵀy + 2XᵀXβ   (using ∂/∂β (βᵀAβ) = 2Aβ for symmetric A=XᵀX)
       = 2Xᵀ(Xβ - y)

Set to zero: XᵀXβ = Xᵀy  → β* = (XᵀX)⁻¹Xᵀy  ← the Normal Equations!
```

**Woodbury Matrix Identity (critical for efficient computation):**
```
(A + UCV)⁻¹ = A⁻¹ − A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹VA⁻¹
```

> 💡 **The problem this solves:** Inverting an n×n matrix costs O(n³) — for n=10,000, that's 10¹² operations, which takes hours. But sometimes we start with an easy-to-invert A and need the inverse of A plus a small "update" UCV. Woodbury lets us compute the correction without re-inverting from scratch — just invert a small matrix (the size of the update). Used in: kernel ridge regression, Gaussian process updates, online learning.

```
Concrete example: Gaussian process update
When a new data point arrives, instead of re-inverting the entire (n+1)×(n+1) kernel matrix,
Woodbury gives us a rank-1 update formula that costs only O(n²) instead of O(n³).
```

**Schur Complement:**
```
For block matrix M = [[A, B],[C, D]]:
  M/A = D − CA⁻¹B   (Schur complement of A in M)
  det(M) = det(A) · det(M/A)
```

> 💡 **The Schur complement** is the covariance of Y given X in a joint Gaussian. If (X,Y) are jointly Gaussian with covariance [[Σ_XX, Σ_XY],[Σ_YX, Σ_YY]], then Var(Y|X) = Σ_YY - Σ_YX Σ_XX⁻¹ Σ_XY — the Schur complement. This is the exact formula used in Gaussian Process regression for the posterior variance.

---

#### Eigendecomposition

> 💡 **What are eigenvalues and eigenvectors?** Imagine a transformation that stretches and rotates space (like a matrix multiplication). Most vectors change both their direction and magnitude. But a few special vectors only get SCALED — they keep their direction. These special vectors are **eigenvectors**. The scale factor is the **eigenvalue**.

```
Visualising eigenvectors:

Before: applying matrix A to various vectors    After:

  ↑                                             ↑
  │  →                                          │    →→   ← v₂ scaled by λ₂=2
  │ ↗                                           │  ↗
  │╱                                            ↑ ↗      ← v₁ scaled by λ₁=3
  └──────                                       └──────

Most vectors ROTATE (change direction)
Eigenvectors ONLY scale — v₁ gets 3× longer, v₂ gets 2× longer
```

```
Av = λv            (v = eigenvector, λ = eigenvalue)
A = Q Λ Qᵀ         (symmetric A: Q = orthogonal eigenvectors, Λ = diagonal eigenvalues)

Decomposition visualised:
A = Q × Λ × Qᵀ
    ↑     ↑   ↑
  rotate scale un-rotate
  
This says: "To apply A, first rotate to the eigenvector basis,
then scale each dimension by the eigenvalue, then rotate back."
```

**Properties:**
- Symmetric matrices always have real eigenvalues and orthogonal eigenvectors (Spectral Theorem)
- Positive Definite (PD): all λᵢ > 0 — unique minimum in quadratic optimisation
- Positive Semi-Definite (PSD): all λᵢ ≥ 0 — required for valid covariance matrices and kernels
- Condition number κ = λ_max / λ_min — large κ → ill-conditioned system

> 💡 **Positive Definite matrix** means it defines a "bowl" in every direction — you can always go uphill from the minimum. When the Hessian of a loss function is positive definite, gradient descent is guaranteed to find a unique global minimum.

> 💡 **Condition number κ** measures "how stretched" the matrix is. κ=1 is a perfect sphere (easy). κ=10,000 is an extremely elongated ellipsoid (hard — gradient descent bounces). In neural networks, adaptive optimisers like Adam work because they implicitly reduce the effective condition number.

**Why Ill-Conditioning Hurts Gradient Descent:**
```
Well-conditioned loss landscape (κ≈1):    Ill-conditioned (κ>>1):
                                           
     ●start                                ●start
      \                                     │
       \  ● ← converges quickly             │←→←→←→ gradient bounces
        \ │                                 │    back and forth
         ↓│                                 │    for thousands
         ●minimum                           │    of steps
                                            ●minimum (eventually)

"Round bowl" → fast descent              "Thin valley" → slow zigzag
```

When minimising `f(x) = (1/2)xᵀAx` with condition number κ, gradient descent with optimal step size converges at rate:
```
||xₜ − x*||² ≤ ((κ−1)/(κ+1))^{2t} ||x₀ − x*||²
```
When κ = 10,000, the contraction factor ≈ 0.9998 — thousands of iterations to converge. Preconditioning (dividing each coordinate by its eigenvalue) transforms the landscape to a sphere. This is exactly what Adam does adaptively: normalising gradients by their approximate second moment achieves implicit diagonal preconditioning.

**Spectral Theorem:**
Every real symmetric matrix `A ∈ ℝⁿˣⁿ` decomposes as `A = QΛQᵀ = Σᵢ λᵢ qᵢqᵢᵀ` — a weighted sum of rank-1 projection matrices. PCA is the direct application: eigenvectors are directions of maximum variance, eigenvalues quantify how much variance lies in each direction.

**Matrix Functions via Eigendecomposition:**
```
Aᵏ      = Q Λᵏ Qᵀ          (raise eigenvalues to the k-th power)
exp(A)  = Q diag(exp(λᵢ)) Qᵀ  (exponentiate each eigenvalue)
A^{1/2} = Q diag(√λᵢ) Qᵀ   (square root each eigenvalue; only if A ≻ 0)
log(A)  = Q diag(log λᵢ) Qᵀ   (log each eigenvalue)
```

> 💡 **Matrix functions** let you apply scalar functions to matrices by working in the "eigenbasis" — rotate to eigenvectors, apply the function to eigenvalues, rotate back. Matrix square roots appear in whitening transforms (PCA preprocessing); matrix exponentials appear in continuous-time dynamical systems (neural ODEs).

---

#### Singular Value Decomposition (SVD)

> 💡 **SVD is the Swiss Army knife of linear algebra.** It works on ANY matrix (not just square ones) and finds the hidden structure — the most important patterns. Think of it as finding the fundamental "ingredients" in your data, ranked by importance.

```
Analogy: SVD is like making smoothies

Your dataset A is like a complex smoothie.
SVD finds:
  U = the "flavour directions" in output space
  Σ = how strong each flavour is (singular values, largest first)
  V = the "ingredient combinations" in input space

Just like you can describe any smoothie as "mostly mango (σ₁=8.5),
a bit of pineapple (σ₂=2.1), trace of ginger (σ₃=0.3)",
SVD describes any matrix in terms of its most important components.
```

```
A = U Σ Vᵀ

Shapes for A (m×n):
  U: (m×m)  — "output patterns"    (left singular vectors)
  Σ: (m×n)  — diagonal, σ₁≥σ₂≥...≥0  (singular values, importance ranking)
  V: (n×n)  — "input combinations"  (right singular vectors)
```

**Concrete example: Movie rating matrix**
```
         Action SciFi Romance Comedy
Alice  [  5      4     1       2   ]
Bob    [  4      5     2       1   ]
Carol  [  1      2     5       4   ]
Dave   [  2      1     4       5   ]

SVD discovers hidden "taste dimensions":
  σ₁ = 8.5 → "Genre preference: Action/SciFi vs Romance/Comedy"
  σ₂ = 2.1 → "Preference: Action vs SciFi within action fans"
  σ₃ ≈ 0   → noise
  σ₄ ≈ 0   → noise

Alice & Bob have high U₁ (love action/scifi)
Carol & Dave have low U₁ (prefer romance/comedy)
```

**Truncated (rank-k) SVD:**
```
Aₖ = Uₖ Σₖ Vₖᵀ       (best rank-k approximation — Eckart-Young theorem)
||A − Aₖ||_F² = Σᵢ>k σᵢ²

Image compression example:
  Original image: 1000×1000 = 1M numbers to store
  Keep top 50 singular values:
    50 × (1000 + 1000 + 1) ≈ 100K numbers → 10× compression!
  Reconstruction quality: Σᵢ>50 σᵢ² / Σᵢ σᵢ² ≈ 1% of total variance lost
  
  Full SVD:          Rank-50 approx:        Rank-10 approx:
  ████████████       ████████████           ████▒▒▒▒▒▒▒▒
  ████████████  →    ████████████     →     ████▒▒▒▒▒▒▒▒
  (perfect)          (98% quality)          (blurry but recognisable)
```

**Eckart-Young Theorem:** Among all rank-k matrices B, the truncated SVD minimises `||A − B||_F`. The proof: any rank-k matrix B can be written as a sum of k rank-1 terms, and the Frobenius norm decomposes as `||A − B||_F² = Σᵢ(σᵢ − σ̃ᵢ)²` when B shares singular vectors with A. Setting `σ̃ᵢ = σᵢ` for i ≤ k and 0 otherwise is optimal. This underpins PCA (best linear dimensionality reduction), collaborative filtering, and data compression.

**SVD and the Four Fundamental Subspaces:**
```
Column space of A   = span(U₁,...,Uᵣ)       (r = rank(A))
Left null space     = span(Uᵣ₊₁,...,Uₘ)
Row space of A      = span(V₁,...,Vᵣ)
Right null space    = span(Vᵣ₊₁,...,Vₙ)
```

> 💡 **Applications of SVD in ML:** PCA (dimensionality reduction), LSA (topic modelling in NLP), collaborative filtering (Netflix recommendations), pseudo-inverse (minimum-norm least squares), matrix completion, image compression, data denoising.

---

#### Covariance and Correlation

> 💡 **Covariance** measures how two variables move together. If tall people tend to also be heavier, then height and weight have positive covariance. If increased study hours correspond to decreased TV hours, those have negative covariance. Covariance of zero means the variables are uncorrelated (though not necessarily independent!).

```
Positive covariance (both go up together):
  Height: [160, 170, 180, 190]
  Weight: [ 55,  65,  75,  85]
  As height ↑, weight ↑ → Cov > 0

  Weight
  85 |            ●
  75 |        ●
  65 |    ●
  55 |●
     +--+--+--+-- Height
     160 170 180 190

Negative covariance (one up, other down):
  Hours of exercise: [1, 2, 3, 4]
  Hours of TV:       [4, 3, 2, 1]
  As exercise ↑, TV ↓ → Cov < 0
```

```
Covariance matrix:   Σ = (1/(n−1)) Xᵀ X     (X is mean-centred, n×d)
Σᵢⱼ = Cov(Xᵢ, Xⱼ)  = E[(Xᵢ − μᵢ)(Xⱼ − μⱼ)]

For 3 features [height, weight, age]:
       height    weight    age
Σ = [ Var(h)   Cov(h,w)  Cov(h,a) ]
    [ Cov(w,h)  Var(w)   Cov(w,a) ]
    [ Cov(a,h)  Cov(a,w)  Var(a)  ]

Diagonal = individual variances
Off-diagonal = how features co-vary

Correlation matrix:  R = D^{−1/2} Σ D^{−1/2}   (D = diag(Σ))
Rᵢⱼ = Σᵢⱼ / (σᵢ σⱼ)   ∈ [−1, 1]
```

> 💡 **Correlation normalises covariance** to be between -1 and +1, making it comparable across features with different scales. Cov(height_cm, weight_kg) has different units than Cov(height_inches, weight_lbs). Correlation is scale-free: Corr(height, weight) = 0.7 regardless of units.

**Geometry of Covariance — The Ellipsoid View:**

```
When Σ = σ²I (spherical):        Large off-diagonal covariance:
data spreads equally in all       data mainly along one direction
directions

    ● ● ●                              ●
   ● ● ● ● ●                        ● ● ●
  ● ● ● ● ● ●                      ● ● ● ● ●
   ● ● ● ● ●                         ● ● ●
    ● ● ●                               ●

→ KNN works well (distance      → KNN distorted! Need Mahalanobis
  is meaningful)                  distance to account for correlation
```

The covariance matrix defines an ellipsoid `{x : xᵀΣ⁻¹x ≤ 1}` — the Mahalanobis unit ball. Eigenvectors of Σ are the axes of this ellipsoid; axis lengths are `√λᵢ`.

**Sample vs Population Covariance:**
```
Population: (1/n) Σᵢ (xᵢ−μ)(xᵢ−μ)ᵀ   (biased — divides by n)
Sample:     (1/(n−1)) Σᵢ (xᵢ−x̄)(xᵢ−x̄)ᵀ  (Bessel's correction — unbiased)
```

> 💡 **Why n−1 not n?** When you estimate the mean from the same data you're computing variance for, the sample mean is always slightly pulled toward the data points, making the deviations look slightly smaller than they really are. Dividing by n−1 instead of n corrects for this bias — it "inflates" the variance estimate just enough to make it unbiased. This is called **Bessel's correction**.

---

### 1.2 Calculus & Optimisation

> 💡 **Why calculus for ML?** Training a machine learning model means finding the best possible parameter values (weights) that minimise prediction error. Calculus tells us: at the current parameter values, which direction should we move to reduce the error? This directional information — the gradient — is the engine behind virtually all ML training via gradient descent.

---

#### Gradient, Jacobian, Hessian

> 💡 **The gradient is your compass.** If you're blindfolded on a hilly landscape and want to find the valley (minimum loss), you need to know which direction is "most downhill." The gradient points toward the steepest ASCENT. So to minimise, you move in the OPPOSITE direction (negative gradient = steepest descent).

```
Analogy: Finding the lowest point in a hilly landscape (blindfolded!)

Strategy: Feel the slope under your feet.
  - The gradient = direction of steepest UPHILL
  - Move OPPOSITE to the gradient (downhill)
  - Take a small step, re-measure slope, repeat
  - Eventually reach a valley (local/global minimum)

       ⛰️Mountain              
      /         \             Loss
     /           \             │
    /      ↑grad  \            │╲             ↗
   /   (go DOWN!) \           │  ╲           ╱
  /                 \         │   ╲_________╱
         valley                    ← minimum
```

```
Gradient of a scalar function f: ℝⁿ → ℝ:
  ∇f(x) = [∂f/∂x₁, ..., ∂f/∂xₙ]ᵀ   ∈ ℝⁿ

Example: f(x₁, x₂) = x₁² + 2x₂²
  ∂f/∂x₁ = 2x₁
  ∂f/∂x₂ = 4x₂
  ∇f = [2x₁, 4x₂]ᵀ
  At point (3, 1): ∇f = [6, 4]ᵀ → move in direction [-6, -4] to go downhill

Jacobian of a vector function f: ℝⁿ → ℝᵐ:
  J(f)ᵢⱼ = ∂fᵢ/∂xⱼ                   ∈ ℝᵐˣⁿ
  (the gradient generalised to vector-valued functions)

Hessian of a scalar function f: ℝⁿ → ℝ:
  H(f)ᵢⱼ = ∂²f / (∂xᵢ ∂xⱼ)            ∈ ℝⁿˣⁿ
  (matrix of all second-order partial derivatives)
  = "curvature" — how fast the gradient changes
```

> 💡 **Jacobian** — if your function maps a vector to another vector (e.g., a neural network layer maps activations to the next layer's activations), the Jacobian is the matrix of all partial derivatives. It tells you how much each output changes when each input changes. The backpropagation algorithm chains Jacobians together via the chain rule.

> 💡 **Hessian** — the second derivative tells you the "curvature" of the loss. Positive Hessian = bowl-shape (good — local minimum). Negative Hessian = dome-shape (bad — local maximum). Mixed Hessian = saddle point (mixed — not a minimum or maximum). In practice, computing the full Hessian costs O(n²) memory and O(n³) computation for n parameters — infeasible for large networks. Methods like Adam approximate the diagonal of the Hessian instead.

**Critical points — what the Hessian tells you:**
```
H ≻ 0  (all eigenvalues positive)  → local MINIMUM     ← we want these
H ≺ 0  (all eigenvalues negative)  → local MAXIMUM
H indefinite (mixed signs)          → SADDLE POINT      ← common in deep networks
H ⪰ 0  (non-negative eigenvalues)  → degenerate case

3D visualisation:

Local minimum:          Local maximum:         Saddle point:
     ↑ loss                ●                      │↑ goes up
    ╱│╲                  ╱│╲                   ───●───
   ╱ ● ╲               ╱ │ ╲                      │↓ goes down →
  (bowl pointing up)  (dome)                   (goes up one way,
                                                down another)
```

**Convexity — the property that makes optimisation tractable:**

A function f is convex if for all x, y and λ ∈ [0,1]:
```
f(λx + (1−λ)y) ≤ λf(x) + (1−λ)f(y)
```

> 💡 **Convexity in plain English:** A function is convex if the "line" (chord) connecting any two points on the function lies ABOVE the function. Picture a bowl: take any two points on the rim, the straight line between them is above the surface of the bowl. This means there is exactly ONE minimum — the bottom of the bowl. Gradient descent is guaranteed to find it.

```
Convex function (bowl):             Non-convex function (mountain range):

  f(x)                               f(x)
   │    ●───────────●                 │  ●   ●   ●
   │   ╱             ╲               │ ╱ ╲ ╱ ╲ ╱ ╲
   │  ╱               ╲              │╱   ●   ●   ╲
   │ ╱                 ╲             └─────────────
   │╱___________________╲          
   └────────────────────           Gradient descent might get stuck
     ONE global minimum            in any of several local minima!

The chord (dashed line) is ABOVE      Same chord? It can go BELOW
the function everywhere.              the function — not convex.
```

Key properties of convex functions:
- **Every local minimum is a global minimum** → gradient descent always finds the best solution
- **The gradient is zero only at the minimum** → if we find where ∇f=0, that's the answer
- Logistic regression loss is convex → guaranteed to converge to the global optimum
- Neural network losses are NOT convex → but in practice, most local minima are "good enough"

**First-Order Convexity Condition:**
```
f is convex ⟺ f(y) ≥ f(x) + ∇f(x)ᵀ(y−x)  ∀x,y

In English: the tangent hyperplane at any point is a GLOBAL LOWER BOUND.
The function always lies ABOVE its tangent plane.

Visualisation:
         f(y)●                    The tangent line at x (dashed)
            │╲                    lies below f everywhere.
            │  ╲                  
tangent ----│---●---- f(x) + ∇f(x)(y-x)
            │    ╲                
            │     ●minimum
```

**Second-Order Convexity Condition:**
```
f is convex ⟺ H(x) ⪰ 0  ∀x

(The Hessian is positive semi-definite everywhere — no downward curvature)
```

---

#### Chain Rule

> 💡 **The chain rule is the mathematical foundation of backpropagation** — the algorithm that trains every deep neural network. It tells you how to compute the derivative of a composition of functions (a chain of operations). In a neural network, the loss is a long chain: input → layer1 → layer2 → ... → layerL → loss. The chain rule unravels this.

```
Simple analogy: How does heating a room affect your electricity bill?

Temperature → Thermostat settings → Furnace gas usage → Electricity bill

dBill/dTemperature = (dBill/dGas) × (dGas/dThermostat) × (dThermostat/dTemp)
                   = each "link" in the chain multiplied together

This IS the chain rule.
```

```
Scalar case:   dz/dx = (dz/dy) × (dy/dx)

Example: z = (x²+1)⁵
  Let y = x²+1, z = y⁵
  dz/dy = 5y⁴
  dy/dx = 2x
  dz/dx = 5y⁴ × 2x = 10x(x²+1)⁴  ← chain rule

Vector case:   ∂z/∂x = Jᵀ_{y→x} · (∂z/∂y)     (Jacobian-vector product)

General (backpropagation in a neural network):
  If z = f(g(x)):  ∂z/∂xᵢ = Σⱼ (∂z/∂yⱼ)(∂yⱼ/∂xᵢ)
```

**How backpropagation uses the chain rule:**
```
Neural network: x → W₁ → σ → W₂ → σ → W₃ → ŷ → Loss L

FORWARD PASS (computing predictions):
x →[W₁]→ z₁ →[σ]→ a₁ →[W₂]→ z₂ →[σ]→ a₂ →[W₃]→ ŷ → L
  (store all intermediate values!)

BACKWARD PASS (chain rule in reverse):
∂L/∂W₃ ← ∂L/∂ŷ                   (output gradient)
∂L/∂a₂ ← ∂L/∂ŷ × ∂ŷ/∂a₂         (chain through W₃)
∂L/∂z₂ ← ∂L/∂a₂ × ∂a₂/∂z₂        (chain through σ)
∂L/∂W₂ ← ∂L/∂z₂ × ∂z₂/∂W₂        (gradient for W₂)
... (continue back to W₁)

Each arrow = multiply by the local derivative (Jacobian).
All parameter gradients computed in ONE backward pass!
```

The key insight of reverse-mode automatic differentiation: computing `∂L/∂θ` for ALL parameters simultaneously costs only O(1) times the cost of a forward pass, regardless of parameter count. This is why training networks with billions of parameters is computationally feasible.

**Forward-Mode vs Reverse-Mode AD:**
```
Forward-mode: "push" derivatives from inputs to output
  Efficient when: few inputs, many outputs
  Cost: 1 pass per input variable

Reverse-mode: "pull" derivatives from output back to inputs
  Efficient when: few outputs (just the loss!), many inputs (millions of params)
  Cost: 1 backward pass computes ALL parameter gradients simultaneously
  → This is BACKPROPAGATION

Why reverse-mode for ML?
  A network has: ~1 output (loss value) and millions of inputs (parameters)
  Forward-mode would need ~1M passes to get all gradients
  Reverse-mode needs just 1 pass → ~1M× more efficient!
```

---

#### Taylor Series & Optimisation Proofs

> 💡 **Taylor series** lets us approximate complicated functions with simpler polynomial ones near a point. In optimisation, we use the first two terms (first-order = gradient, second-order = Hessian/curvature) to build a "local quadratic model" of the loss, then minimise that model to get our next parameter update. This is the mathematical justification for gradient descent and Newton's method.

```
f(x + δ) ≈ f(x) + ∇f(x)ᵀδ + (1/2)δᵀH(x)δ + O(||δ||³)
                  ↑               ↑
           1st order term    2nd order term
           (linear)          (curvature)

Example: f(x) = x² near x=2
  f(2 + δ) ≈ f(2) + f'(2)δ + (1/2)f''(2)δ²
            = 4 + 4δ + (1/2)(2)δ²
            = 4 + 4δ + δ²
  Exact: f(2+δ) = (2+δ)² = 4 + 4δ + δ²  ← perfect for quadratics!
```

**Gradient Descent — First-Order Taylor Approximation:**
```
Using only the first-order term:
f(x + δ) ≈ f(x) + ∇f(x)ᵀδ

To DECREASE f, choose δ = −α∇f(x) (move opposite the gradient)
→ f(x − α∇f(x)) ≈ f(x) − α||∇f(x)||²  (always decreases for small α!)

This is gradient descent:  x_{t+1} = xₜ − α ∇f(xₜ)
                                      ↑
                                  learning rate
```

**Newton's Method — Second-Order Taylor Approximation:**
```
Using both terms:
f(x + δ) ≈ f(x) + ∇f(x)ᵀδ + (1/2)δᵀH(x)δ

Minimise over δ (take derivative w.r.t. δ, set to zero):
  ∇f(x) + H(x)δ = 0
  δ* = −H⁻¹∇f(x)   ← the Newton step!

x_{t+1} = xₜ − H⁻¹ ∇f(xₜ)
```

> 💡 **Newton's method is smarter than gradient descent.** Instead of always taking a fixed step in the gradient direction, Newton accounts for curvature. In a direction where the function is very curved (steep), take a small step. In a direction where it's flat, take a big step. Result: converges in many fewer iterations BUT each step costs O(n³) to invert the Hessian — infeasible for large n. Adam approximates this cheaply using only the diagonal of H.

**Gradient Descent Convergence (L-smooth f):**
```
If ||∇²f|| ≤ L (Lipschitz gradient), then GD with α = 1/L:
  f(xₜ) − f* ≤ (L||x₀ − x*||²) / (2t)    O(1/t) sublinear convergence
```

**Derivation Sketch:** From L-smoothness: `f(y) ≤ f(x) + ∇f(x)ᵀ(y−x) + (L/2)||y−x||²`. Setting `y = x − (1/L)∇f(x)` gives descent of at least `(1/2L)||∇f(x)||²` per step. Summing over t steps yields the O(1/t) rate.

**Strong Convexity (μ-strongly convex):**
```
f(y) ≥ f(x) + ∇f(x)ᵀ(y−x) + (μ/2)||y−x||²
Linear convergence: ||xₜ − x*||² ≤ (1 − μ/L)ᵗ ||x₀ − x*||²
Condition number κ = L/μ — smaller → faster convergence
```

> 💡 **Strong convexity** gives EXPONENTIAL convergence instead of O(1/t). The error shrinks by a factor of (1 - μ/L) each step. For κ=10: factor = 0.9, so error is 0.9^100 ≈ 0.00003 after 100 steps. Compare O(1/t): after 100 steps, error is still 1/100 = 0.01. This is why strongly convex problems (like ridge regression) converge so much faster.

**Conjugate Gradient Method:**
```
For Ax = b — minimises (1/2)xᵀAx − bᵀx:
  Generates A-conjugate directions: pᵢᵀApⱼ = 0 for i≠j
  Exact convergence in n steps (theoretically)
  Convergence rate: O((√κ−1)/(√κ+1))^t — faster than GD's O((κ−1)/(κ+1))^t
```

---

### 1.3 Probability & Statistics

> 💡 **Why probability for ML?** Real-world data is noisy and uncertain. A patient with certain symptoms might have disease A with 70% probability — not 100%. Probability theory gives us the language to reason about this uncertainty mathematically, and statistics gives us tools to learn distributions from finite (noisy) data.

---

#### Bayes' Theorem & Its Geometry

> 💡 **Bayes' theorem is the most important equation in machine learning.** It tells you how to update your beliefs when you see new evidence. It's the mathematical formalisation of learning — you start with a prior belief, see data, and update to a posterior belief. This is literally what every ML algorithm does.

```
Intuitive example — The Rare Disease Test:

Facts:
  Disease is rare: 1 in 100 people have it → P(disease) = 0.01
  Test is accurate: 99% sensitivity → P(positive | disease) = 0.99
  Test has 1% false positives → P(positive | healthy) = 0.01

Question: You test positive. What's the chance you're actually sick?

Most people answer "99%"... Let's use Bayes:

Imagine 10,000 people are tested:
  ┌─────────────────────────────────────────┐
  │  10,000 people                          │
  │    ├── 100 have disease (1%)            │
  │    │     └── 99 test positive ✓         │ ← true positives
  │    │           1 test negative (missed) │
  │    └── 9,900 are healthy (99%)          │
  │          └── 99 test positive ✗         │ ← false positives!
  │              9,801 test negative        │
  └─────────────────────────────────────────┘
  
  Total positives: 99 + 99 = 198
  Of these, truly sick: 99/198 = 50%!
  
  Even with a 99% accurate test, a positive result only means 50% chance of disease
  WHY? Because the disease is so rare, false positives outnumber true positives.
```

```
P(A|B) = P(B|A) P(A) / P(B)

In medical terms:
  P(disease | positive test) = P(positive | disease) × P(disease) / P(positive)
                              = 0.99 × 0.01 / (0.99×0.01 + 0.01×0.99)
                              = 0.0099 / 0.0198 = 0.50

In ML terms:
  Prior:      P(θ)           — your belief about parameters BEFORE seeing data
  Likelihood: P(data|θ)      — probability of the data IF parameters are θ
  Posterior:  P(θ|data)      — updated belief AFTER seeing data
  Evidence:   P(data) = ∫ P(data|θ) P(θ) dθ  — normalisation constant (often intractable)
```

**Bayes as Information Update (Gaussian example):**
```
Scenario: Estimating a person's true IQ

Prior: "Average person has IQ ~ N(μ₀=100, τ²=225)" (std=15 points)
After observing n test scores x₁,...,xₙ with noise σ²:

Posterior: IQ | data ~ N(μₙ, τₙ²)
  μₙ  = (τ⁻² μ₀ + nσ⁻² x̄) / (τ⁻² + nσ⁻²)   (precision-weighted average)
  τₙ⁻² = τ⁻² + nσ⁻²                           (precisions add up)

Intuition (precision = 1/variance = "how confident"):

  n=0:    posterior = prior (no data → stick with prior belief)
  n=1:    posterior leans slightly toward observed score
  n=10:   posterior mostly determined by data
  n=∞:    posterior = MLE estimate (data overwhelms prior)
  
  Prior confidence + Data confidence → Combined confidence
  (Like two people arguing: the more confident one wins)
```

> 💡 **The posterior mean is a precision-weighted average** of the prior mean and the data mean. If you have a strong prior (high precision τ⁻²) and few data points (small nσ⁻²), the posterior stays close to the prior — this IS regularisation. The prior is mathematically equivalent to an L2 penalty.

**Sequential Bayesian Updating:**
```
Bayes' theorem is naturally incremental:

Start: Prior P(θ)
See data point 1: P(θ|x₁) ∝ P(x₁|θ) P(θ)    ← Posterior 1
See data point 2: P(θ|x₁,x₂) ∝ P(x₂|θ) P(θ|x₁)  ← use Posterior 1 as new prior!
...

This is online learning! Each new data point updates the belief.
The order doesn't matter — the final posterior is the same.
```

---

#### Expectation, Variance, Covariance

> 💡 **Expectation** is the "centre of mass" of a distribution — what value would you get on average if you repeated the random experiment infinitely many times. For a fair die: E[X] = (1+2+3+4+5+6)/6 = 3.5. You never actually roll 3.5, but it's the long-run average.

> 💡 **Variance** measures spread or unpredictability. A stock that moves ±50% daily has high variance. A savings account earning 2% annually has near-zero variance. High variance = less predictable = more uncertainty.

```
E[X] = Σₓ x P(X=x)           (discrete: weighted average of all values)
E[X] = ∫ x f(x) dx            (continuous: integral version)
E[aX + bY] = aE[X] + bE[Y]   (LINEARITY — holds for ANY X, Y, even dependent!)

Example: E[2·Height + 3·Weight]
  = 2·E[Height] + 3·E[Weight]  ← linearity is incredibly useful

Var(X) = E[(X−μ)²] = E[X²] − (E[X])²

Computational formula: Var(X) = E[X²] − (E[X])²
  For die roll: E[X²] = (1+4+9+16+25+36)/6 = 91/6 ≈ 15.17
                E[X]² = (3.5)² = 12.25
                Var(X) = 15.17 - 12.25 = 2.92

Var(aX + b) = a² Var(X)    ← scaling by a multiplies variance by a²
Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)

Cov(X,Y) = E[(X−μₓ)(Y−μᵧ)] = E[XY] − E[X]E[Y]
```

> 💡 **Important pitfall:** `Cov(X,Y) = 0` means uncorrelated, BUT NOT necessarily independent. Example: X ~ Uniform(-1,1), Y = X². Cov(X,Y) = 0 (they're uncorrelated), but knowing X perfectly determines Y — they're completely dependent! Correlation only captures LINEAR relationships.

**Law of Total Expectation and Total Variance:**
```
Law of Total Expectation: E[X] = E[E[X|Y]]

Example: Average height of a human
  E[Height] = E[E[Height | Sex]]
            = P(male) × E[Height|male] + P(female) × E[Height|female]
            = 0.5 × 175 + 0.5 × 162 = 168.5 cm  ✓

Law of Total Variance: Var(X) = E[Var(X|Y)] + Var(E[X|Y])
                               = "within-group variance" + "between-group variance"

Example: Variance in exam scores
  = (Average variance WITHIN each school) + (Variance of school AVERAGES)
  ← How much students vary         ← How much schools differ
     within their own school           from each other
```

> 💡 **The law of total variance** is why ensemble methods work. Each model captures different patterns (between-group variance). Their average is much more stable (reduces within-group variance). This is the mathematical reason 100 trees in a Random Forest beats any single tree.

---

#### Key Distributions

> 💡 **Distributions** are mathematical descriptions of different types of random phenomena. Choosing the right distribution to model your data is crucial — it determines what kind of model, loss function, and metrics are appropriate.

```
Visual gallery of key distributions:

Bernoulli (single coin flip, p=0.3):   Binomial (n=10 flips, p=0.3):
  P(X)                                    P(X)
  0.7│●                                   0.28│   ●
  0.3│    ●                               0.23│ ●   ●
     ├────┤                               0.12│●       ●
     0    1                                  ├─────────────
                                             0 1 2 3 4 5 6

Gaussian N(μ=0,σ=1):                   Poisson (λ=3):
  P(x)                                    P(X)
  0.40│    ●●                             0.22│  ●●
  0.24│  ●    ●                           0.15│●    ●
  0.05│●        ●                         0.04│         ●
     ├────────────                            ├─────────────
    -3 -2 -1 0 1 2 3                          0 1 2 3 4 5 6

Exponential (λ=1):                     Uniform (a=0,b=1):
  P(x)                                    P(x)
  1.0│●                                   1.0│████████████
  0.4│  ●                                    │████████████
  0.1│    ●●●                               ├────────────
     ├────────────                          0.0          1.0
     0  1  2  3  4
```

| Distribution | PMF/PDF | Mean | Variance | Use in ML |
|---|---|---|---|---|
| Bernoulli(p) | pˣ(1−p)^(1−x) | p | p(1−p) | Binary classification output |
| Binomial(n,p) | C(n,x)pˣ(1−p)^(n−x) | np | np(1−p) | Count data |
| Gaussian(μ,σ²) | (1/√(2πσ²))exp(−(x−μ)²/2σ²) | μ | σ² | Continuous output, weight prior |
| Categorical(π) | Πₖ πₖ^xₖ | πₖ | πₖ(1−πₖ) | Multi-class output |
| Dirichlet(α) | ∝ Πₖ xₖ^(αₖ−1) | αₖ/α₀ | — | Prior over Categorical |
| Laplace(μ,b) | (1/2b)exp(−\|x−μ\|/b) | μ | 2b² | L1 prior, robust regression |
| Beta(α,β) | ∝ x^{α−1}(1−x)^{β−1} | α/(α+β) | — | Prior over Bernoulli |
| Exponential(λ) | λe^{−λx} | 1/λ | 1/λ² | Time between events |
| Poisson(λ) | λˣe^{−λ}/x! | λ | λ | Count of rare events per unit time |

**Why the Gaussian Appears Everywhere — The Central Limit Theorem:**

```
The Central Limit Theorem (CLT) — perhaps the most important theorem in statistics:

Take ANY distribution (doesn't matter what shape):

   Dice roll distribution:        Distribution of AVERAGES of 30 dice rolls:
   (flat/uniform)                 (bell-shaped!)
                                  
   P│███████████                  P│     ●●●
    │███████████                   │   ●●●●●●●
    │███████████                   │ ●●●●●●●●●●●
    └──────────                    └────────────
     1  2  3  4  5  6              2.5  3.0  3.5  4.0  4.5

No matter what distribution you start with:
  average of n samples → Gaussian as n → ∞!
  (with mean = μ_original, variance = σ²_original / n)
```

This justifies:
- Modelling residuals (prediction errors) as Gaussian — they're sums of many small errors, so CLT applies
- Using Mean Squared Error (MSE) as the standard regression loss — it's the MLE under Gaussian noise
- Gaussian confidence intervals for parameter estimates

**Maximum Entropy Property:** Among all distributions with fixed mean μ and variance σ², the Gaussian maximises differential entropy `H(X) = −∫ p(x) log p(x) dx`. It is the "least informative" distribution given only first two moments — a principled default when no additional structural knowledge is available. When in doubt, use a Gaussian prior.

---

#### Exponential Family (Unified Framework)

> 💡 **The exponential family** is a large "super-family" of distributions — including Gaussian, Bernoulli, Poisson, Exponential, Beta, and many others — that all share the same mathematical structure. Once you recognise a distribution as belonging to this family, you automatically get: closed-form MLE, conjugate priors, and a unified framework for generalised linear models (GLMs). This is why logistic regression, linear regression, and Poisson regression all look so similar mathematically.

```
General exponential family form:
  p(x|η) = h(x) exp(ηᵀT(x) − A(η))

Components:
  η (eta):     natural parameters — the "settings" of the distribution
  T(x):        sufficient statistics — what you need to know from data
  A(η):        log-partition function — ensures probabilities sum to 1
  h(x):        base measure — "default" over x ignoring parameters

Example — Bernoulli as exponential family:
  p(x|p) = pˣ(1-p)^(1-x)
           = exp(x·log(p/(1-p)) + log(1-p))  ← exponential family form!
  
  η = log(p/(1-p))     ← natural parameter = log-odds!
  T(x) = x             ← sufficient statistic = the observation itself
  A(η) = log(1+eᶑ)     ← log-partition function (ensures sums to 1)
  
  This is why logistic regression uses log-odds as its linear predictor!
```

**Key Cumulant Property:**
```
∂A/∂η  = E[T(x)]        (first derivative of log-partition = MEAN)
∂²A/∂η² = Var[T(x)]     (second derivative = VARIANCE)

This means: if you know A(η), you know the mean and variance for free!
For Gaussian: A(η) = μ²/(2σ²), so ∂A/∂η = μ → mean is μ ✓
```

**Why the Exponential Family Matters for ML:**

1. **Sufficient Statistics:** T(x) captures ALL information about η from the data — you don't need the full raw data to estimate parameters. For a Gaussian, just knowing (x̄, s²) is equivalent to knowing all n data points for parameter estimation.

2. **MLE has Closed Form:** Set mean sufficient statistics equal to empirical mean: `∂A/∂η = (1/n)Σᵢ T(xᵢ)`. This gives closed-form MLE for any exponential family member.

3. **Conjugate Priors:** Every exponential family has a conjugate prior (in the same family) that gives closed-form posterior updates — no approximation needed.

4. **Generalised Linear Models (GLMs):** All GLMs assume the response belongs to an exponential family with a "link function" connecting the linear predictor βᵀx to the natural parameter η. This unifies linear regression (Gaussian), logistic regression (Bernoulli), and Poisson regression (Poisson) in one framework.

---

#### Information Theory

> 💡 **Information theory** measures uncertainty and information mathematically. Developed by Claude Shannon in 1948 to quantify how much data can be transmitted through a noisy channel, it turns out to be deeply connected to machine learning: cross-entropy loss, KL divergence, and mutual information all come from information theory.

**Entropy — measuring surprise/uncertainty:**

```
High entropy = high uncertainty:      Low entropy = low uncertainty:
P(heads) = P(tails) = 0.5            P(heads) = 0.99, P(tails) = 0.01

  0.5│●  ●                            0.99│●
     │                                    │
  0.0└────                            0.01│   ●
       H  T                               └────
                                            H  T
H = -(0.5 log₂ 0.5 + 0.5 log₂ 0.5)  H = -(0.99 log₂ 0.99 + 0.01 log₂ 0.01)
  = 1 bit (maximum for binary)          ≈ 0.08 bits (near zero)
  "Completely unpredictable"            "Almost always heads — low surprise"
```

```
Entropy (uncertainty in X):
  H(X) = −Σₓ P(x) log₂ P(x)    ≥ 0
  (maximum for uniform distribution; zero when one outcome is certain)

Conditional Entropy (uncertainty in Y given we know X):
  H(Y|X) = H(X,Y) − H(X)
  (how much uncertainty remains in Y after knowing X)

Mutual Information (how much X and Y share):
  I(X;Y) = H(X) − H(X|Y) = H(Y) − H(Y|X) = H(X) + H(Y) − H(X,Y)   ≥ 0
  (zero if X and Y are independent; maximum when one determines the other)
  
  Used in: feature selection (choose features with high MI with target),
           information bottleneck, tree-building (information gain)
```

**KL Divergence — measuring how different two distributions are:**

```
D_KL(P||Q) = Σₓ P(x) log(P(x)/Q(x))  ≥ 0   (= 0 only if P = Q)

Intuition: You designed a code for distribution Q (optimal for Q).
           But the true distribution is P.
           D_KL(P||Q) = extra bits you waste by using the wrong code.

Example:
  True weather P: [sunny=0.7, rainy=0.3]
  Your model Q:   [sunny=0.5, rainy=0.5]
  
  D_KL(P||Q) = 0.7·log(0.7/0.5) + 0.3·log(0.3/0.5)
             = 0.7·(0.485) + 0.3·(-0.737)
             ≈ 0.118 bits wasted per prediction
```

```
Cross-Entropy (total "wrongness" of using Q when truth is P):
  H(P,Q) = H(P) + D_KL(P||Q) = −Σₓ P(x) log Q(x)

This is the standard loss function for classification!
When training a classifier:
  P = true labels (one-hot: [0, 0, 1, 0] for class 3)
  Q = model's predicted probabilities ([0.1, 0.2, 0.6, 0.1])
  Minimising H(P,Q) = minimising D_KL(P||Q) = maximising log-likelihood
  → All three are the SAME objective! (They differ only by constants)

Jensen-Shannon Divergence (symmetric, bounded [0,1]):
  JSD(P||Q) = (1/2)D_KL(P||M) + (1/2)D_KL(Q||M),   M = (P+Q)/2
  √JSD is a proper metric (symmetric, satisfies triangle inequality)
  Used in: GANs (JS divergence measures generator quality)
```

**Forward KL vs Reverse KL — a critical distinction:**

```
True distribution P (bimodal — two separate clusters):
         ●●                    ●●
        ●●●●●               ●●●●●●
       ●●●●●●●             ●●●●●●●●
  ─────────────────────────────────
      mode 1              mode 2

FORWARD KL (P||Q): minimise mismatch from P's perspective
  → Q must "cover" all of P's mass → Q spreads out to cover both modes
  
       ●●●●●●●●●●●●●●●●●●●●●
      ●●●●●●●●●●●●●●●●●●●●●●●
  → Q is BLURRY (mean-seeking), covering both modes but not sharp

REVERSE KL (Q||P): minimise mismatch from Q's perspective
  → Q avoids regions where P=0 → Q focuses on just ONE mode
  
           ●●●
          ●●●●●●●
  → Q is SHARP (mode-seeking), picks one mode and ignores the other
```

> 💡 **This distinction matters enormously in practice.** Classification training (cross-entropy = forward KL) gives calibrated probabilities over all classes. VAE decoders (reverse KL) tend to collapse to a single mode. GAN training tries to match distributions. Knowing which KL you're minimising determines what kind of outputs your model produces.

**Data Processing Inequality:**
```
If X → Y → Z is a Markov chain (X causes Y causes Z):
  I(X;Z) ≤ I(X;Y)

In English: Processing data can only DESTROY information, never CREATE it.

Implications for ML:
  → Raw data is the most valuable resource
  → Every preprocessing step risks losing information
  → Feature engineering must be justified: does it preserve task-relevant information?
  → Good representations COMPRESS while preserving mutual information with the target
```

This is the theoretical basis for the Information Bottleneck principle: a good representation Z of input X for predicting Y should maximise I(Z;Y) while minimising I(X;Z).

---

## 2. Core Machine Learning Theory

> 💡 **What IS machine learning?** Machine learning is the science of making computers learn from data without being explicitly programmed. Instead of writing rules by hand ("if email contains FREE then spam"), we show the computer thousands of examples ("here are 10,000 emails labelled spam/not-spam") and let it discover the rules itself. This section gives you the mathematical foundations of HOW and WHY this works.

---

### 2.1 The Learning Problem

> 💡 **The fundamental setup:** You have data. You want a model. You train the model on data. You deploy it on new, unseen data. The big question: will the model work on new data that it has never seen before? This is the generalisation problem — the central question of all of machine learning.

**Formal Setup:**
```
The ML ingredients:
  Input space X:   the set of all possible inputs (e.g., all possible images)
  Output space Y:  the set of all possible outputs (e.g., {cat, dog, bird})
  True function f: the unknown "truth" we're trying to learn
  
Training data D = {(x₁,y₁), ..., (xₙ,yₙ)} drawn i.i.d. from P(X,Y)
  ↑
  Each (xᵢ, yᵢ) is an input-output pair: (image_i, "cat")
  i.i.d. = each drawn independently from the same distribution

Hypothesis class H:  the "family" of models we consider
  e.g., H = {all linear classifiers}
       or H = {all neural networks with 3 layers and ReLU activation}
  (We can only find models within H — our algorithm can't invent new architectures)

Loss function L(ŷ, y):  measures how wrong a prediction ŷ is
  Squared loss:   L(ŷ, y) = (ŷ-y)²         ← regression
  0-1 loss:       L(ŷ, y) = 𝟙[ŷ ≠ y]        ← classification (1 if wrong, 0 if right)
  Cross-entropy:  L(ŷ, y) = -Σ y log(ŷ)     ← probabilistic classification
```

**Empirical Risk Minimisation (ERM) — the core algorithm:**
```
ĥ = argmin_{h ∈ H}  R̂(h) = (1/n) Σᵢ L(h(xᵢ), yᵢ)

Translation:
  ĥ = the model we choose
  argmin = "the h that minimises"
  R̂(h) = training error = average loss on training data
  
In English: find the model in H that makes the smallest total error on the training data.
This is what gradient descent does — it minimises the training loss.
```

**Expected (True) Risk:**
```
R(h) = E_{(x,y)~P} [L(h(x), y)]

This is the TRUE error we care about — the expected loss on NEW, UNSEEN data.
We can't compute this directly (we don't know P), so we estimate it with test data.
```

**The Generalisation Gap — the central problem:**
```
Generalisation gap = |R(h) - R̂(h)|
                   = |test error - training error|

Good model:          Overfitting:          Underfitting:
R̂(h) ≈ 0.15        R̂(h) ≈ 0.01          R̂(h) ≈ 0.40
R(h)  ≈ 0.17        R(h)  ≈ 0.45          R(h)  ≈ 0.42
gap   ≈ 0.02        gap   ≈ 0.44          gap   ≈ 0.02
(good!)             (BAD — memorised       (bad — model too
                     training data)         simple to learn)
```

**The Fundamental Tension in Learning:**
```
Two ways a model can fail:

UNDERFITTING (High Bias):                 OVERFITTING (High Variance):
Model too simple — can't capture          Model too complex — fits noise
the true pattern in the data              as if it were signal

True function: wavy curve                True function: wavy curve
Model: straight line                      Model: wiggly line through every point

     ●   ●                                    ●   ●
     |●  ●  ●                               /  ● ↗ ●
     |  ●   ●                              / ● ↗   ↘●
     |-------- (model)                    /●↗       ↘●
     
Train error: high                         Train error: near 0
Test error: high                          Test error: high
Problem: model too simple                 Problem: model memorised noise

FIX: use more complex model               FIX: regularise, get more data,
     or better features                        use simpler model
```

**Rademacher Complexity (modern generalisation bound):**
```
R̂ₙ(H) = E_σ[ sup_{h∈H} (1/n) Σᵢ σᵢ h(xᵢ) ]
  σᵢ ∈ {−1,+1} i.i.d. uniform (random ±1 labels)

Intuition:
  Give the model RANDOM labels (±1) for each training point.
  How well can it fit these random labels?
  → High Rademacher complexity = model can memorise ANY labelling
  → Low Rademacher complexity = model has limited fitting power

Generalisation bound:
  R(ĥ) ≤ R̂(ĥ) + 2Rₙ(H) + √(log(1/δ)/(2n))
  
  True error ≤ training error + model complexity + confidence correction
  
  To ensure small generalisation gap:
    Use simpler models (smaller Rₙ(H))
    Get more training data (larger n)
```

---

### 2.2 Bias-Variance Trade-off

> 💡 **The bias-variance tradeoff is the most important concept in all of practical machine learning.** It explains WHY models fail, HOW to fix them, and WHAT the fundamental limits of learning are. Every model selection decision comes down to managing this tradeoff.

**The Decomposition:**

For squared loss, the expected error of any estimator ĥ decomposes as:
```
E[(y − ĥ(x))²] = Bias²(ĥ(x)) + Var(ĥ(x)) + σ²_noise
```

Where:
```
Bias(ĥ(x))  = E[ĥ(x)] − f(x)
             = "systematic error" — the gap between what we predict on average
               and the true value
             = caused by model being TOO SIMPLE

Var(ĥ(x))   = E[(ĥ(x) − E[ĥ(x)])²]
             = "sensitivity to training data" — how much predictions change
               if you retrain on a different sample of the same size
             = caused by model being TOO COMPLEX

σ²_noise     = Var(y|x)
             = "irreducible noise" — the randomness in the data itself
             = can NEVER be reduced, even with a perfect model
```

**Full Derivation — why these three terms:**
```
MSE = E[(y − ĥ)²]

Add and subtract f(x) and E[ĥ]:
    = E[((y−f) + (f−Eĥ) + (Eĥ−ĥ))²]

Square out — cross-terms vanish:
    = E[(y−f)²]   + (f−Eĥ)²        + E[(Eĥ−ĥ)²]
    = σ²_noise   +   Bias²          +   Variance

Why cross-terms vanish:
  E[(y−f)(f−Eĥ)] = (f−Eĥ)·E[(y−f)] = (f−Eĥ)·0 = 0
  (because noise y-f has mean zero)
  
  E[(y−f)(Eĥ−ĥ)] = 0
  (because noise is independent of the training data used to fit ĥ)
```

**Visualising the Tradeoff:**
```
Total error vs model complexity:

Error
  │╲                          
  │  ╲  ← total error          
  │    ╲___________            
  │         ╲      ╲           
  │          ╲      ╲  variance grows
  │           ╲______●── variance
  │ bias²┄┄┄┄┄┄╲___             
  │  ↓         ╲    ╲            
  │  bias²→0    ╲    ╲           
  │              ╲    ╲          
  │               ╲               
  └─────────────────────────────
   simple                complex
   (underfitting)        (overfitting)
   
   ● = optimal complexity (minimum total error)
   
   Left of ●: reducing complexity raises bias faster than it lowers variance
   Right of ●: increasing complexity raises variance faster than it lowers bias
```

**Concrete example with KNN:**
```
KNN with different k values:

k=1 (most complex):               k=n (simplest):
  Each test point gets the         Predict the global average for everyone
  label of its NEAREST neighbor    
  
  Bias ≈ 0 (exact interpolation)   Bias = large (ignores local patterns)
  Var = large (sensitive to        Var ≈ 0 (same prediction always)
        which neighbors are near)  

k=1 decision boundary:            k=15 decision boundary:
  Very jagged:                      Smooth:
  ┌──────────────┐                 ┌──────────────┐
  │○○●●○○●●○○●●│                 │○○○○○○○○○○○○│
  │○●○●○●○●○●○│                 │○○○○○○│●●●●●│
  │○○●●○○●●○○●│                 │○○○○○○│●●●●●│
  │●○○●●○○●●○○│                 │      ●●●●●●│
  └──────────────┘                 └──────────────┘
  (fits training data perfectly)   (smooth, generalises better)
```

**Double Descent (modern deep learning phenomenon):**

```
Classical understanding:               Modern deep learning:

Error                                  Error
  │╲                                     │╲
  │  ╲________                           │  ╲_____        
  │   ↑ optimal                          │        ╲       ╱╲
  │                                      │         ╲_____╱  ╲___
  │                                      │               ↑       ↑
  └─────────────────                     └────────────────────────
   simple      complex                    ← classical → interpolation → "benign overfit"
   
Beyond the "interpolation threshold" (zero training error),
error can DECREASE AGAIN as model size grows!
Reason: over-parameterised models find smooth interpolations via implicit regularisation
(SGD prefers flat minima; large networks have many equivalent low-error solutions)
```

---

### 2.3 The No Free Lunch Theorem

> 💡 **The No Free Lunch Theorem** says there's no universally best algorithm. Every algorithm has implicit assumptions baked in. When those assumptions match your problem, it works great. When they don't, it fails. Understanding your data's structure is essential for choosing the right algorithm.

**Statement:** For any two algorithms A₁ and A₂, averaged over ALL possible data-generating distributions:
```
Σ_f E[L(f, A₁, n)] = Σ_f E[L(f, A₂, n)]

No algorithm beats random guessing on AVERAGE over all possible problems.
```

**What this really means for ML practitioners:**
```
Misreading: "All algorithms are equal" → WRONG!
Correct reading: "Every algorithm makes ASSUMPTIONS. It wins where assumptions are right,
                  loses where they're wrong. On AVERAGE over ALL tasks, they tie."

In practice: we care about SPECIFIC tasks, not averages over all possible tasks.

CNNs assume: local spatial structure (images have nearby pixel correlations)
  → WIN on images, FAIL on tabular data

RNNs assume: sequential temporal structure
  → WIN on time series/text, FAIL on independent data

Decision trees assume: axis-aligned decision boundaries
  → WIN on certain tabular data, FAIL on diagonal boundaries

"The 'free lunch' we pay for is the inductive bias we build in."
```

> 💡 **Practical lesson:** The NFL theorem tells you why domain knowledge is so valuable. A data scientist who knows that house prices are influenced by location should use spatial features. An ML engineer who knows customer behaviour is sequential should use temporal models. The assumptions (inductive biases) you build in should match the real structure of your data.

---

### 2.4 VC Dimension & PAC Learning

> 💡 **VC dimension** measures the "capacity" or "complexity" of a model class — how many different patterns it can fit. A model with high VC dimension is flexible but needs more data to generalise. A model with low VC dimension is limited but works with less data. This gives us principled sample size requirements.

**VC Dimension Definition:**
```
A hypothesis class H has VC dimension d if:
  ∃ a set of d points that H can SHATTER (correctly classify ALL 2ᵈ possible labellings)
  AND no set of d+1 points can be shattered

"Shattering" means: for every possible way of labelling n points ○/●,
there exists some h ∈ H that achieves that exact labelling.

Example with linear classifiers in 2D:

3 points CAN be shattered by a line:    4 points CANNOT be shattered by a line:
                                         (XOR configuration cannot be separated)
  ● ○      ○ ●      ● ●      ● ○        
  ● ●      ● ●      ○ ●      ○ ●           ● ○
   ...all 8 labellings achievable...        ○ ●
                                          This labelling impossible with a line!
→ VCdim(linear classifiers in ℝ²) = 3   → VCdim = 3 (can't shatter 4)
```

**Examples:**
```
Linear classifiers in ℝᵈ:    VCdim = d + 1
  (d+1 points in general position can always be shattered)

Circles in ℝ²:                VCdim = 3
Convex polygons in ℝ²:        VCdim = ∞  (can shatter any finite set!)
Neural networks (W weights):  VCdim ≈ O(W log W)

Why linear classifiers in ℝᵈ have VCdim = d+1:
  Any d+1 points in "general position" (no d lying on a (d-1)-dimensional hyperplane)
  can be shattered by a hyperplane.
  For d+2 points: Radon's theorem guarantees a partition where convex hulls intersect
  → no hyperplane can separate this partition.
```

**PAC Learning Bound (how much data do you need?):**
```
With probability ≥ 1−δ:
  R(ĥ) − R̂(ĥ) ≤ √(2 VCdim(H) log(en/VCdim(H)) / n) + √(log(1/δ)/(2n))

Simplified sample complexity:
  n ≥ (1/ε²)[VCdim(H) log(1/ε) + log(1/δ)]

Translation:
  ε = acceptable generalisation gap (e.g., 0.05 = within 5% of perfect)
  δ = failure probability (e.g., 0.05 = 95% confidence)
  n = how many training examples you need
  
  Example: Linear classifier in 100 dimensions (VCdim=101)
    For ε=0.05, δ=0.05: n ≥ (1/0.0025)[101·3 + 3] ≈ 122,000 samples
```

> 💡 **Practical interpretation:** High-dimensional models (many features, complex architecture) need EXPONENTIALLY more data to generalise. This is why you can't train a 1-billion parameter model on 1,000 examples. The VC dimension gives you the theoretical minimum sample size requirement.

---

### 2.5 Maximum Likelihood Estimation (MLE)

> 💡 **MLE is the most fundamental approach to fitting models to data.** The idea is simple: find the parameter values that make the observed data as likely as possible. If you saw 7 heads in 10 coin flips, MLE says the best estimate for P(heads) is 0.7 — because that's the value that maximises the probability of seeing exactly what you saw.

**The Setup:**
```
You have data x₁, x₂, ..., xₙ (all drawn i.i.d. from distribution p(x|θ))
You want to estimate θ.

Likelihood function: L(θ) = p(x₁,...,xₙ|θ) = Πᵢ p(xᵢ|θ)
                           = "how likely is this data, given parameters θ?"

MLE: θ_MLE = argmax_θ L(θ) = argmax_θ Πᵢ p(xᵢ|θ)
           = argmax_θ ℓ(θ) = argmax_θ Σᵢ log p(xᵢ|θ)
```

> 💡 **Why use log?** Products of many small probabilities can underflow (become zero numerically). Log converts products to sums, which are numerically stable. Also, log is monotone so argmax is the same. The log-likelihood is much easier to differentiate too.

**Concrete example — Gaussian MLE:**
```
Data: exam scores [75, 82, 68, 90, 71] assumed ~ N(μ, σ²)

Log-likelihood: ℓ(μ,σ²) = Σᵢ log N(xᵢ; μ, σ²)
              = -n/2 log(2πσ²) - 1/(2σ²) Σᵢ(xᵢ-μ)²

∂ℓ/∂μ = 1/σ² Σᵢ(xᵢ-μ) = 0  →  μ̂_MLE = x̄ = (75+82+68+90+71)/5 = 77.2

∂ℓ/∂σ² = -n/(2σ²) + 1/(2σ⁴)Σᵢ(xᵢ-μ)² = 0  →  σ̂²_MLE = (1/n)Σᵢ(xᵢ-μ̂)²

MLE recovers sample mean and variance — not surprising, but now it's principled!
```

**MLE as KL Minimisation:**
```
θ_MLE = argmin_θ D_KL(p_data || p_θ)
       = argmin_θ H(p_data, p_θ)

Proof:
  D_KL(p_data || p_θ) = E_data[log p_data(x)] - E_data[log p_θ(x)]
  First term: constant w.r.t. θ
  Second term: -E_data[log p_θ(x)] ≈ -(1/n)Σᵢ log p_θ(xᵢ) = -ℓ(θ)/n

So: minimise KL = maximise log-likelihood = minimise cross-entropy
These are three different names for the SAME optimisation problem!
```

**Score Function and Fisher Information:**
```
Score:    s(θ) = ∇_θ ℓ(θ) = Σᵢ ∇_θ log p(xᵢ|θ) = 0   (at MLE)
  The "score" measures how much the log-likelihood changes with θ.
  At the maximum, it must be zero (gradient condition for maximum).

Fisher Information: I(θ) = E[−∇²_θ log p(x|θ)] = E[s(θ)s(θ)ᵀ]
  Measures the "information content" of observations about θ.
  High Fisher information → single observation gives lots of info about θ.
  
Cramér-Rao Bound: Var(θ̂) ≥ I(θ)⁻¹
  No unbiased estimator can have SMALLER variance than 1/Fisher Information.
  MLE achieves this bound asymptotically → MLE is the most efficient estimator!
```

**Natural Gradient:**
```
Standard gradient descent: θ ← θ - α ∇_θ ℓ
  Uses Euclidean geometry in parameter space.
  
Natural gradient: θ ← θ - α I(θ)⁻¹ ∇_θ ℓ
  Uses the intrinsic Riemannian geometry of probability distributions.
  Makes updates invariant to reparameterisation.
  Converges much faster for probabilistic models.
```

**Asymptotic Properties of MLE:**
```
√n (θ_MLE − θ*) →_d N(0, I(θ*)⁻¹)

As n → ∞:
  1. Consistency: θ_MLE → θ* (converges to true value)
  2. Asymptotic normality: the estimator becomes Gaussian
  3. Efficiency: achieves Cramér-Rao bound (minimum possible variance)
  → MLE is the "best" large-sample estimator
```

---

### 2.6 Maximum A Posteriori (MAP) Estimation

> 💡 **MAP adds prior knowledge to MLE.** Instead of only asking "what parameters make the data most likely?", MAP asks "what parameters are most likely given BOTH the data AND my prior beliefs?" This is Bayesian learning. The key insight: **MAP = MLE + regularisation**. Choosing a prior over parameters is mathematically equivalent to adding a regularisation term to the loss.

```
MAP = argmax_θ P(θ|data) = argmax_θ P(data|θ) P(θ)

Taking log:
  θ_MAP = argmax_θ [Σᵢ log P(xᵢ|θ) + log P(θ)]
                   ↑                    ↑
           log-likelihood            log-prior = regularisation!
```

**Prior → Regularisation Correspondence (the key insight):**

```
Gaussian prior β ~ N(0, σ²/λ I):

  log P(β) = -λ/(2σ²) ||β||²  + const
  
  MAP objective: Σᵢ log P(yᵢ|xᵢ,β) + log P(β)
               = -1/(2σ²)||y-Xβ||² - λ/(2σ²)||β||²
               ∝ ||y-Xβ||² + λ||β||²   ← this is RIDGE REGRESSION!
  
Laplace prior β ~ Laplace(0, b):
  log P(β) = -1/b ||β||₁  + const
  MAP → ||y-Xβ||² + λ||β||₁   ← this is LASSO!

Spike-and-slab prior:
  MAP → L0 sparsity (exact subset selection, NP-hard)

PRIOR STRENGTH ↔ REGULARISATION STRENGTH:
  Strong prior (small variance) → large λ → heavy regularisation
  Weak prior (large variance)   → small λ → light regularisation
  No prior (uniform)            → λ=0     → plain MLE (no regularisation)
```

**Why L2 Regularisation = Gaussian Prior (detailed):**
```
Under Gaussian prior β ~ N(0, τ²I) and Gaussian likelihood y|β ~ N(Xβ, σ²I):

log P(β|X,y) ∝ log P(y|X,β) + log P(β)
             = -1/(2σ²)||y − Xβ||² - 1/(2τ²)||β||²  + const

Setting λ = σ²/τ² gives exactly the Ridge objective:
  min_β ||y − Xβ||² + λ||β||²

Larger τ² (weaker prior, less regularisation) → smaller λ
Smaller τ² (stronger prior, more regularisation) → larger λ
τ² → ∞ (no prior) → λ=0 → plain OLS
```

**Full Bayesian Inference vs MAP:**
```
MAP:  θ_MAP = mode of P(θ|X)    ← single point estimate
  Advantages: computationally cheap, clear interpretation
  Disadvantages: ignores posterior shape, no uncertainty

Full Bayes:
  P(θ|X) = P(X|θ)P(θ) / P(X)             ← full posterior distribution
  P(x_new|X) = ∫ P(x_new|θ) P(θ|X) dθ   ← posterior predictive

  Advantages: full uncertainty quantification, calibrated predictions
  Disadvantages: integral is often intractable, computationally expensive

When does MAP fail?
  Posterior is multimodal: MAP picks ONE mode, ignores others
  Posterior is highly asymmetric: mode ≠ mean ≠ median
  
  Example:
  P(θ|X):     ●●●●●●●●       ●●●
              ─────────────────────
                MAP here ↑    ← also has large probability mass here!
              
  MAP gives a misleading point estimate — full Bayes captures the whole picture.
```

---

## 3. Data Preprocessing & Feature Engineering

> 💡 **Garbage in, garbage out.** No matter how sophisticated your model, it can't compensate for poor quality data. Data preprocessing — cleaning, transforming, and enriching your data — is often what separates a working ML system from a failed one. Experienced ML practitioners spend 70-80% of their time on data, not modelling.

---

### 3.1 Handling Missing Values

> 💡 **Missing data is everywhere.** Survey respondents skip questions, sensors malfunction, forms are partially filled. How you handle missing values can make or break your model — wrong choices introduce bias and inflate uncertainty estimates.

**Types of Missingness (Rubin's Classification, 1976):**
```
Understanding WHY data is missing is crucial:

MCAR (Missing Completely At Random):
  The fact that data is missing has NO relationship to any values (observed or missing)
  Example: A form was accidentally shredded — no pattern in which forms went missing
  P(missing | data) = P(missing)   ← missingness is pure chance
  
  MCAR test: Little's MCAR test checks if missingness is independent of observed values
  Safe to: drop missing rows (complete case analysis)
  
MAR (Missing At Random):
  Missingness depends on OBSERVED variables, but not on the missing value itself
  Example: Men are less likely to report their weight, but given their height, 
           the probability of reporting is the same regardless of actual weight
  P(missing | data) = P(missing | observed only)
  
  Safe to: impute using observed variables
  Dangerous to: drop rows (creates biased sample)

MNAR (Missing Not At Random):  ← the most dangerous!
  Missingness depends on the MISSING VALUE ITSELF
  Example: Sicker patients don't show up for follow-up appointments
           → missing health measurements for those who got sicker
  P(missing | data) = P(missing | missing value)
  
  Cannot handle purely statistically! Need:
    • Model the missingness mechanism explicitly
    • Sensitivity analysis
    • Domain knowledge
    
Visualisation:
  MCAR: missing values scattered randomly:     MNAR: missing values concentrated
  █ █ █ _ █ _ █ █ _ █ █ _                     in specific regions:
  _ █ █ █ _ █ █ _ █ █ _ █                     █ █ █ █ █ █ _ _ _ _ _ _
  (random gaps)                                 (gaps have a pattern!)
```

**Practical Guidance:**
- **MCAR** → safe to drop rows (complete case analysis won't bias results)
- **MAR** → use imputation (simple to sophisticated)
- **MNAR** → most dangerous; need a model for the missingness mechanism itself

**Why Complete Case Analysis Under MAR is Biased:**
```
Example: Survey of income and education level.
  Less educated people are less likely to answer income questions (MAR).
  
  Complete case analysis (drop missing):
    → Sample becomes biased toward more educated people
    → Estimated average income is too HIGH
    
  Correct approach (imputation):
    → Use education level to predict likely income for non-responders
    → Imputed values are unbiased estimates
```

**Imputation Methods (from simple to sophisticated):**
```
1. Mean/Median/Mode Imputation:
   x_imputed = mean(x_observed)
   
   Example: height column has some missing
   Heights: [170, 165, _, 180, 175, _]
   Mean: (170+165+180+175)/4 = 172.5
   Imputed: [170, 165, 172.5, 180, 175, 172.5]
   
   ✓ Simple, fast
   ✗ Reduces variance (artificially makes distribution narrower)
   ✗ Distorts correlations (imputed values all have same value)
   ✗ Use median for skewed distributions (robust to outliers)
   ✗ Use mode for categorical features

2. Regression Imputation:
   x_missing = β₀ + β₁x₁ + β₂x₂ + ε
   Predict the missing value from other features using a regression model
   
   ✓ Uses relationships between features
   ✗ Still underestimates variance (residual ε often omitted)

3. KNN Imputation:
   Find k most similar complete observations, take weighted average
   x_imputed = Σₖ wₖ xₖ / Σₖ wₖ,   wₖ = 1/d(x, xₖ)ᵖ
   
   Example: Patient with missing cholesterol, similar patients:
   Neighbor 1 (most similar): cholesterol=195, weight=1/1.2=0.83
   Neighbor 2: cholesterol=210, weight=1/2.1=0.48
   Imputed = (195×0.83 + 210×0.48) / (0.83+0.48) = 200
   
   ✓ Preserves local data structure
   ✗ Computationally expensive for large datasets
   ✗ Sensitive to the distance metric chosen

4. Multiple Imputation by Chained Equations (MICE):
   The gold standard for missing data handling:
   
   Step 1: Initial imputation (mean imputation as starting point)
   Step 2: For each variable xⱼ with missing values:
     a. Regress xⱼ on ALL other variables using observed cases
     b. SAMPLE from the predictive distribution (don't just take mean!)
     c. Fill in missing xⱼ values with samples
   Step 3: Repeat step 2 for all variables → one "imputation"
   Step 4: Repeat steps 2-3 until convergence → complete dataset
   Step 5: Repeat whole process m times → m complete datasets
   Step 6: Fit your model on each, combine results with Rubin's rules:
   
   Q̄ = (1/m)Σ Qₘ            ← mean of m estimates
   T = W̄ + (1 + 1/m)B       ← total variance
   W̄ = within-imputation variance
   B = between-imputation variance (uncertainty from missing data)
   
   ✓ Correctly propagates uncertainty from missing data
   ✓ Unbiased under MAR
   ✓ Appropriate confidence intervals
   ✗ Computationally expensive

Why Multiple Imputation Outperforms Single Imputation:
  Single imputation: treats imputed values as if they were observed
    → artificially reduces standard errors (we're more "confident" than we should be)
    → we have fake precision!
  
  Multiple imputation: creates m different datasets with different imputed values
    → the SPREAD between the m estimates captures our uncertainty about missing values
    → Rubin's rules propagate this uncertainty into final estimates
    → confidence intervals are correctly wider to reflect missing data uncertainty
```

**Missing Indicator Variables:**
```
Strategy: add a binary column indicating which rows were missing
  x_age_missing = [0, 0, 1, 0, 1, 0]  ← 1 means age was missing, then imputed

Why? Sometimes MISSINGNESS ITSELF is informative!
  Example: If a patient's lab result is missing, it might be because
           they were too sick to complete the test → high risk!
  
  By adding the missing indicator, the model can learn:
  "When x_age_missing=1, the patient tends to be higher risk"
  — information that would be lost with just imputation alone.
```

---

### 3.2 Feature Scaling

> 💡 **Why does scaling matter?** Many ML algorithms measure "distance" or use "weights" that are sensitive to the scale of features. Without scaling, a feature measured in thousands (annual salary: 50,000) would dominate a feature measured in ones (number of children: 2), even if the children feature is more important for the prediction. Scaling puts all features on a level playing field.

**Which algorithms NEED scaling, and why:**
```
MUST scale:
  • SVMs — maximise margin 2/||w||, which depends on feature scale
  • PCA — eigenvectors dominated by features with large variance
  • KNN — distance computation meaningless with unscaled features
  • Neural networks — gradient descent converges poorly with different scales
  • Regularised regression (Ridge/Lasso) — penalty applies equally to all β

DON'T need scaling (scale-invariant):
  • Decision trees — only use threshold comparisons (is x₁ > 5?)
  • Random forests — (same reason as trees)
  • XGBoost, LightGBM — (same reason)
  • These find optimal splits regardless of scale
```

```
Min-Max Normalisation — scales to [0,1]:
  x' = (x − x_min) / (x_max − x_min)
  
  Example: heights [150, 160, 170, 180, 190]
    x_min=150, x_max=190, range=40
    150 → 0.00
    160 → 0.25
    170 → 0.50
    180 → 0.75
    190 → 1.00
  
  ✓ Bounded output [0,1] → good for neural networks with sigmoid output
  ✗ VERY sensitive to outliers!
    If one person has height=250: now 170 → 0.20/1.00 = 0.20 (squashed!)

Z-Score Standardisation — mean=0, std=1:
  x' = (x − μ) / σ
  
  Same heights: μ=170, σ=15.8
    150 → -1.27   (1.27 standard deviations below average)
    160 → -0.63
    170 →  0.00   (exactly average)
    180 →  0.63
    190 →  1.27
  
  ✓ Robust to outliers compared to Min-Max
  ✓ The gold standard for PCA, SVMs, neural networks
  ✗ Output is unbounded (an outlier at 250 → z=5.06, which is fine!)

Robust Scaling — median/IQR (resistant to outliers):
  x' = (x − median) / IQR,    IQR = Q3 − Q1
  
  ✓ Best for data with outliers
  ✓ Appropriate when outliers are real (not errors) and you don't want to clip them

Log Transform — right-skewed data:
  x' = log(x + 1)       (the +1 handles x=0, since log(0) = -∞)
  
  When to use: income, population, prices — data with long right tails
  
  Before:              After log transform:
  │●                   │   ●●●●
  │ ●●                 │ ●●●●●●●●
  │   ●●●●●●●●        │●●●●●●●●●●●●
  └─────────────       └─────────────
  (right-skewed)       (more symmetric)

Box-Cox Transform — approximate Gaussian:
  x'(λ) = (xλ−1)/λ  if λ≠0;  log(x) if λ=0   (requires x > 0)
  λ estimated by maximising log-likelihood of normality
  
  λ=1: no transform (x)
  λ=0: log transform
  λ=-1: reciprocal (1/x)
  λ=0.5: square root
  The optimal λ is found via MLE or grid search.

Yeo-Johnson Transform — extends Box-Cox to x ≤ 0:
  Works for negative values too (no requirement x > 0)
  x'(λ) = [(x+1)λ−1]/λ           if λ≠0, x≥0
           log(x+1)                if λ=0, x≥0
          −[(−x+1)^{2−λ}−1]/(2−λ) if λ≠2, x<0
          −log(−x+1)               if λ=2, x<0

Unit Vector Normalisation — per-sample:
  x' = x / ||x||₂    (each SAMPLE becomes a unit vector)
  
  Used for: text data (TF-IDF vectors), cosine similarity computation
  Distinct from feature scaling — this normalises each ROW, not each COLUMN
```

---

### 3.3 Encoding Categorical Variables

> 💡 **Machine learning algorithms work with numbers, not categories.** You can't feed "red", "green", "blue" directly into a regression. You need to convert categories to numbers. But HOW you encode them matters enormously — wrong encoding can introduce false orderings or explode dimensionality.

```
The encoding problem:

Category      | What you can't do     | Why it's wrong
─────────────────────────────────────────────────────────
[Low, Med, High]    | Just use 0,1,2      | Implies High-Low=2 and Med-Low=1,
                    |                     | spacing might not be meaningful
[Red,Green,Blue]    | Use 1,2,3           | Implies Blue > Green > Red — nonsense!
[City names]        | Use 1 to 50,000     | Implies NYC and LA have numeric distance
```

```
Label Encoding: [Low, Med, High] → [0, 1, 2]
  ✓ ONLY correct for ORDINAL variables (inherent ordering exists)
  ✗ NEVER for nominal variables (colours, cities, countries)
  
One-Hot Encoding: [Red, Green, Blue] → 3 binary columns
  Red   Green  Blue
  [1,    0,     0]   ← "Red"
  [0,    1,     0]   ← "Green"
  [0,    0,     1]   ← "Blue"
  
  ✓ Correct encoding for nominal categories
  ✓ No false ordering implied
  ✗ Drop one column in linear models (to avoid perfect multicollinearity)!
    With k categories, use k-1 columns.
    "Red" is identified by Green=0 AND Blue=0 — the dropped category
  ✗ Creates very high-dimensional sparse vectors for high-cardinality categories
    (50,000 cities → 49,999 columns, mostly zeros)

Target Encoding (Mean Encoding):
  Replace each category with the mean target value for that category
  + smoothing to handle rare categories:
  
  x_enc = (n_c × ȳ_c + α × ȳ) / (n_c + α)
  n_c = count of category c
  ȳ_c = mean target for category c  
  α = smoothing factor (typically 10-100)
  ȳ = global mean target
  
  Example: encoding "City" for predicting house price
    New York: 150 houses, avg price $800k → encoded as ~800,000
    Paris:    200 houses, avg price $600k → encoded as ~600,000
    Tiny Village: 2 houses, avg price $100k
      → with smoothing: (2×100k + 10×600k) / (2+10) ≈ 517k
        (pulled toward global mean because we have little data for this city)
  
  ✓ Single dimension (vs thousands for one-hot)
  ✓ Captures relationship between category and target
  ⚠️ DANGER: Target Leakage! (see below)

⚠️ THE TARGET LEAKAGE PROBLEM:
  If you compute ȳ_c using the SAME rows that you train on:
    • The feature directly encodes the target → circular reasoning!
    • Model will "learn" that high target-encoded feature → high target
    • This appears to work on train but FAILS at test time (data leakage)
  
  CORRECT: Compute target statistics only on out-of-fold training data
    Never include the current validation/test fold in the statistics

Frequency/Count Encoding:
  x_encoded = count(category) / total_count
  Useful when the frequency of a category is itself informative

Binary Encoding:
  Label encode → convert integer to binary bits → split into columns
  100 categories → only 7 columns (vs 100 for one-hot)
  Category 5 → binary 000101 → [0, 0, 0, 1, 0, 1]

Hashing (Feature Hashing / Hashing Trick):
  x_encoded[h(category) mod m] += 1
  
  ✓ Fixed output size (m), no matter how many categories
  ✓ Handles UNSEEN categories at test time
  ✗ Hash collisions (different categories map to same bucket)
  Used in: very large-scale text classification (web spam detection)

Embedding Encoding (learned in neural networks):
  Map each category to a dense vector e_c ∈ ℝᵈ via a learnable lookup table
  Rule of thumb: d = min(50, ceil(cardinality/2))
  
  Example: 50,000 cities → each city gets a learned 50-dimensional vector
  Similar cities end up with similar vectors (if the model sees enough examples)
  
  ✓ Captures semantic similarity (cities in same region have similar embeddings)
  ✓ Much lower dimension than one-hot (50 vs 50,000)
  ✓ Dense — no wasted zeros
  ✗ Requires enough training data to learn good representations
  ✗ Need to handle unseen categories at test time (OOV problem)
```

---

### 3.4 Feature Selection

> 💡 **More features is NOT always better.** Irrelevant features add noise, increase training time, make models harder to interpret, and can actually hurt performance (the curse of dimensionality). Feature selection finds the subset of features that are most useful for predicting the target.

**Why Feature Selection Helps:**
```
Scenario: Predicting house prices. You have 500 features including:
  Useful: bedrooms, bathrooms, area, location, year built, school rating
  Useless: house color, owner's birthday, street number, phase of moon

With all 500 features:
  ✗ Model might latch onto spurious correlations (phase of moon appeared correlated in training)
  ✗ Overfitting: more degrees of freedom → easier to fit noise
  ✗ Slow: more features = more computation
  ✗ Hard to interpret: what does "phase of moon" mean for house prices?

With just the 6 useful features:
  ✓ Better generalisation (no noise features to overfit)
  ✓ Faster training and inference
  ✓ Interpretable: we understand what drives price
  ✓ Lower data requirements (fewer features = fewer parameters to estimate)
```

**Filter Methods (model-independent, fast — good for initial screening):**
```
Pearson Correlation:   r = Cov(x,y) / (σₓ σᵧ)      ∈ [-1, 1]
  Measures LINEAR relationship between feature x and target y
  r=+1: perfect positive linear correlation
  r=-1: perfect negative linear correlation
  r=0: no LINEAR relationship (might still have nonlinear!)
  
  Use for: continuous features, continuous target
  Limitation: misses nonlinear relationships

Spearman Rank Correlation: ρ = 1 − 6Σdᵢ² / (n(n²−1))
  Measures MONOTONIC relationship (not just linear)
  Convert values to ranks, then compute correlation of ranks
  
  Example: 
    x: [1, 10, 100, 1000]  → ranks: [1, 2, 3, 4]
    y: [2,  4,   8,   16]  → ranks: [1, 2, 3, 4]
    Spearman ρ = 1.0 (perfect monotone) even though not linear!
  
  Use for: ordinal data or nonlinear but monotone relationships

Mutual Information: I(X;Y) = H(X) − H(X|Y)
  Measures ANY statistical relationship (linear or nonlinear)
  I=0: X and Y are independent
  I>0: knowing X reduces uncertainty about Y
  
  ✓ Detects all types of dependencies
  ✗ Requires density estimation (more complex, more data needed)

Chi-Squared test (χ²): for categorical features vs categorical target
  χ² = Σ (Observed − Expected)²/Expected
  High χ² → feature and target are DEPENDENT (feature is useful)
  Use p-value threshold (e.g., p<0.05 → keep feature)

ANOVA F-statistic: for continuous features vs categorical target
  F = (variance BETWEEN groups) / (variance WITHIN groups)
  High F → feature values differ significantly across target classes

Variance threshold: remove features with near-zero variance
  A constant feature (always=5) contains zero information → remove
  threshold_var = Var(feature) < ε → drop feature
```

**Wrapper Methods (use a model to score feature subsets):**
```
Forward Selection (greedy):
  Start with empty set ∅
  Repeat:
    Test adding each remaining feature
    Add the one that most improves CV score
  Stop when no improvement or max features reached
  
  ┌──────────────────────────────────────────────────────────┐
  │ ∅ → {x₂} → {x₂,x₅} → {x₂,x₅,x₁} → {x₂,x₅,x₁,x₇} → stop │
  └──────────────────────────────────────────────────────────┘
  
  ✓ Guaranteed to add only beneficial features
  ✗ Greedy (might miss optimal combo): {x₁,x₃} together is great,
     but x₁ alone is mediocre, so forward selection never discovers this

Backward Elimination (greedy):
  Start with full feature set
  Repeat:
    Test removing each feature
    Remove the one that least hurts CV score
  Stop when removal hurts too much
  
  ✓ Considers all feature interactions (starts with full model)
  ✗ Expensive: O(p²) model fits

Recursive Feature Elimination (RFE):
  1. Fit model with all features
  2. Rank features by importance (e.g., by coefficient magnitude)
  3. Remove the worst-ranked feature(s)
  4. Repeat until desired number of features
  
  Very popular with SVMs and linear models.
  RFE-CV: use cross-validation to select the optimal number of features.
  
  Complexity: O(p²) model fits (expensive but often worth it)
```

**Embedded Methods (feature selection as part of model training):**
```
L1 (Lasso): drives irrelevant coefficients to EXACTLY zero
  After training: features with β=0 are eliminated
  ✓ Fast (single model fit)
  ✓ Principled (selects the globally optimal sparse model)
  ✗ Handles correlated features poorly (arbitrarily picks one)

L2 (Ridge): shrinks all coefficients but never to exactly zero
  → NOT feature selection! All features retained (just smaller weights)

Elastic Net: αL1 + (1−α)L2 → sparse but handles correlated groups better

Tree-based importance (Random Forest, XGBoost):
  Feature importance = total reduction in impurity across all splits that use feature j
  Features that appear in many splits early in the trees = important
  
  ⚠️ Limitation: biased toward high-cardinality features (more possible split points)
  Better alternative: permutation importance

Permutation Importance (model-agnostic, reliable):
  1. Train model, record baseline performance
  2. For each feature j:
     a. Randomly shuffle feature j (break any relationship with target)
     b. Re-evaluate model performance
     c. Importance = how much performance DROPPED
  
  PI(j) = Baseline score - Score with feature j permuted
  
  ✓ Directly measures what each feature contributes to prediction
  ✓ Works for any model
  ✓ Detects nonlinear relationships
  ✗ Correlated features: removing one feature might not change performance
     if another correlated feature carries the same information

SHAP Values — Theoretical Foundation (the gold standard):
  Based on Shapley values from cooperative game theory:
  
  φⱼ = Σ_{S⊆F\{j}} [|S|!(|F|−|S|−1)!/|F|!] × [v(S∪{j}) − v(S)]
  
  Idea: Imagine features are "players" in a game (predicting y).
  Each player's "contribution" (Shapley value) is the average marginal contribution
  across all possible orderings in which players join the team.
  
  SHAP is the UNIQUE attribution satisfying 4 axioms:
  1. Efficiency: Σⱼ φⱼ = f(x) − E[f(X)]  (explains full prediction)
  2. Symmetry:  Interchangeable features get equal attribution
  3. Dummy:     Non-contributing features get zero attribution
  4. Additivity: For model ensembles, SHAP values add linearly
  
  Example: House price prediction, one house = $400,000 (model says $420,000)
  Global mean prediction = $350,000
  
  SHAP values:
    Bedrooms=4:     +$35,000  (contributes positively, above average)
    Location=good:  +$40,000  (very valuable location)
    Area=small:     -$20,000  (below average area, hurts price)
    Year=old:       -$15,000  (older house, hurts price)
    SHAP sum: $350k + 35k + 40k - 20k - 15k = $390k (close to model's $420k)
```

---

### 3.5 Outlier Detection

> 💡 **Outliers are data points that are very different from the rest.** They can be errors (someone entered age=999) or genuine anomalies (a fraudulent transaction, a faulty sensor reading, a patient with a rare disease). Whether to remove or keep them depends on the context — errors should be removed, but genuine anomalies might be what you're trying to detect!

**Simple Statistical Methods:**
```
Z-Score:
  z = (x − μ)/σ
  Flag |z| > 3 as outlier (beyond 3 standard deviations from mean)
  
  Normal distribution: 99.7% of data within 3 std devs
  So z>3 has only 0.3% probability under normality
  
  Example:
  Heights: [165, 170, 175, 168, 172, 300]
  μ=191.7, σ=49.1
  z(300) = (300-191.7)/49.1 = 2.21 ← NOT flagged (below 3)!
  
  Problem: mean and std are themselves affected by the outlier (non-robust)

Modified Z-Score (Robust):
  Mᵢ = 0.6745(xᵢ − x̃)/MAD   where MAD = median(|xᵢ − median(x)|)
  Flag |M| > 3.5
  
  Uses MEDIAN and MAD (median absolute deviation) instead of mean and std
  → Not affected by the outlier itself!
  
  Same example:
  Median = 171.5, MAD = 5.0
  M(300) = 0.6745 × (300-171.5)/5.0 = 17.35 ← clearly flagged!

IQR (Tukey's Fences — industry standard):
  Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 - Q1
  Lower fence = Q1 − 1.5×IQR
  Upper fence = Q3 + 1.5×IQR
  
  Example: [165, 168, 170, 172, 175, 300]
  Q1=168, Q3=175, IQR=7
  Lower = 168 - 10.5 = 157.5
  Upper = 175 + 10.5 = 185.5
  300 → OUTLIER (above 185.5) ✓
  
  For heavy-tailed distributions: use 3×IQR instead of 1.5×IQR
```

**Machine Learning Based Detection:**
```
Isolation Forest:
  Key insight: outliers are "few and different" — they're isolated in sparse regions.
  A random tree needs few splits to isolate an outlier (it's alone),
  but many splits to isolate a normal point (surrounded by others).
  
  Algorithm:
  1. Randomly sample a small subset of data
  2. Randomly select a feature and a split value
  3. Recursively partition until each point is isolated
  4. Repeat to build a forest of isolation trees
  5. Anomaly score = 2^{−E[h(x)] / c(n)}
     h(x) = average path length to isolate x
     Short path = anomaly; Long path = normal point
  
  Visualisation:
  Dense region (normal):                Sparse region (anomaly):
  ●●●●●●●●                                        ●
  ●●●●●●●●  ← needs many cuts         single point → isolated in 2 cuts!
  ●●●●●●●●
  
  ✓ O(n log n) — very fast even for large datasets
  ✓ Works well in high dimensions
  ✓ No distance metric needed
  ✗ Less interpretable than statistical methods

Local Outlier Factor (LOF):
  Key insight: an outlier is a point that is in a much SPARSER neighbourhood
               than its nearest neighbours themselves.
  
  Algorithm:
  1. For each point x, find its k nearest neighbours Nₖ(x)
  2. Compute local reachability density:
     lrd_k(x) = 1 / (mean of reach_dist_k(x, neighbours))
  3. LOF score = ratio of neighbour densities to x's own density:
     LOF_k(x) = mean_{o ∈ Nₖ(x)} lrd_k(o) / lrd_k(x)
  
  LOF ≈ 1: similar density to neighbours (normal)
  LOF >> 1: much sparser than neighbours (outlier!)
  
  Example:
  Dense cluster:                  Outlier between two clusters:
  ●●●●                              ●●●●      ●
  ●●●●  ← LOF≈1.0                  ●●●●    ● ← LOF≈3.5
  ●●●●                              ●●●●     ●●●●●●
                                              ●●●●●●
  
  ✓ Detects LOCAL anomalies (not globally extreme but anomalous in their neighbourhood)
  ✗ Computationally expensive for large n (need to compute k-NN)

Autoencoder-based:
  Train autoencoder (encoder-decoder neural network) on NORMAL data only
  → Network learns to reconstruct normal patterns
  → When given an anomaly, it CAN'T reconstruct well (it's never seen this pattern!)
  
  Anomaly score: ||x − Autoencoder(x)||² > threshold
  
  ✓ Learns complex high-dimensional patterns (great for images, sensor data)
  ✓ No manual feature engineering needed
  ✗ Requires enough normal training data
  ✗ Threshold selection can be tricky
```

---

### 3.6 Correlation Analysis

> 💡 **Understanding how features relate to each other and to the target** is essential for both feature engineering and avoiding pitfalls. High correlation between features (multicollinearity) can destroy linear model performance. Understanding these relationships helps you select the right models and features.

```
Pearson:     r = Σ(xᵢ−x̄)(yᵢ−ȳ) / √[Σ(xᵢ−x̄)² Σ(yᵢ−ȳ)²]   (linear relationships)
Spearman:    ρ = 1 − 6Σdᵢ² / (n(n²−1))                       (monotonic, non-parametric)
Kendall:     τ = (concordant − discordant) / C(n,2)            (ordinal, robust)
Point-biserial: r_pb = (ȳ₁−ȳ₀)/sᵧ × √(n₁n₀/n²)             (continuous vs binary)
Cramér's V:  V = √(χ²/(n × min(r−1,c−1)))                     (categorical vs categorical)

Comparison of correlation types:
Feature type    | Target type    | Use
─────────────────────────────────────────────────────────────────────────
Continuous      | Continuous     | Pearson (linear) or Spearman (monotone)
Continuous      | Binary         | Point-biserial correlation
Ordinal         | Any            | Spearman rank correlation
Categorical     | Categorical    | Cramér's V (based on chi-squared)
Any             | Any            | Mutual Information (model-free)
```

**Variance Inflation Factor (VIF) — detecting multicollinearity:**
```
VIF_j = 1 / (1 − R²_j)
  R²_j = R² from regressing feature j on ALL other features

VIF = 1:        Feature j is uncorrelated with all others
VIF 1-5:        Mild multicollinearity — usually acceptable
VIF 5-10:       Moderate — consider removing or combining features
VIF > 10:       Severe — feature j is nearly a linear combination of others

Example:
  VIF(bedrooms) = 8.5 → moderately collinear with other features
  (bedrooms is correlated with both area and bathrooms)
  
  Solution options:
    1. Remove the feature with highest VIF, recompute
    2. Combine correlated features (e.g., average of bedrooms and bathrooms)
    3. Use Ridge regression (handles multicollinearity via λI term)
    4. Use PCA to decorrelate features before modelling
```

**Why Multicollinearity Destroys Linear Models:**
```
Perfect multicollinearity: x₂ = 2x₁ (second feature is exactly twice the first)

  Loss landscape (β₁, β₂ space):
                   β₂
                   │  ↗ this entire line gives the same MSE!
                   │↗
                 ──●──────  β₁
                   │╲
                   │  ╲
  
  Infinitely many optimal solutions → β̂ is undefined!
  XᵀX is singular (det=0) → cannot be inverted → normal equations have no unique solution
  
  Near-collinearity (x₂ ≈ 2x₁):
  XᵀX is nearly singular → (XᵀX)⁻¹ has very large entries
  Var(β̂) = σ²(XᵀX)⁻¹ → explodes!
  
  Interpretation: with correlated features, the data can't tell which one
  is "really" responsible for the outcome — coefficients become unreliable.
  
  Fix: Ridge regression adds λI to XᵀX:
    β_ridge = (XᵀX + λI)⁻¹ Xᵀy
    (XᵀX + λI) is ALWAYS invertible (λI guarantees all eigenvalues ≥ λ > 0)
```

---

## 4. Regression Algorithms

> 💡 **What is regression?** Regression is predicting a *continuous number* — not a category. Examples: predicting house price ($342,000), forecasting tomorrow's temperature (22.5°C), estimating a patient's blood pressure. The word comes from "regressing toward the mean" — early statisticians noticed that tall parents tend to have children *closer* to average height.

---

### 4.1 Linear Regression

> 💡 **The core idea:** Draw the best possible straight line through your data points. "Best" means the line minimises the total squared distance between the predicted values and the actual values. Simple, elegant, and the foundation of almost all other regression methods.

**The Model:**
```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

In plain English:
  ŷ       = our prediction
  β₀      = intercept (the baseline value when all features are zero)
  β₁...βₙ = weights (how much each feature influences the prediction)
  x₁...xₙ = the input features

Example — predicting house price:
  Price = 50,000 + 300×(area_sqft) + 10,000×(bedrooms) − 5,000×(age_years)
  
  A 1500 sqft, 3 bedroom, 10 year old house:
  Price = 50,000 + 300×1500 + 10,000×3 − 5,000×10
        = 50,000 + 450,000 + 30,000 − 50,000
        = $480,000

Matrix form (compact notation for ALL data points at once):
  ŷ = Xβ
  X is the n×(p+1) design matrix (n data points, p features + intercept column of 1s)
  β is the (p+1)×1 vector of weights we want to learn
```

**Visualising what "best fit line" means:**
```
Without the right β (bad fit):           With the optimal β (best fit):

  Price │                                  Price │        /
  500k  │    ●                             500k  │    ●  /
  400k  │  ●   ●  ← data points           400k  │  ● ● /  ← fitted line
  300k  │●   ●                             300k  │●  ● /
  200k  │−−−−−−−−  ← wrong line           200k  │  /
        └──────────── sqft                       └──────────── sqft
  
  Residuals (errors) are LARGE              Residuals are MINIMISED
  (gaps between dots and line)
  
  Each residual eᵢ = yᵢ − ŷᵢ  (actual minus predicted)
  We want to minimise:  Σᵢ eᵢ²  (sum of SQUARED residuals = RSS)
```

**The OLS (Ordinary Least Squares) Loss Function:**
```
L(β) = ||y − Xβ||² = Σᵢ (yᵢ − ŷᵢ)²

      = (y − Xβ)ᵀ(y − Xβ)     ← matrix form

Expanding:
  = yᵀy − 2βᵀXᵀy + βᵀXᵀXβ

Why SQUARED errors?
  1. Negative and positive errors don't cancel out
  2. Large errors are penalised more heavily than small ones (quadratic penalty)
  3. The math becomes beautifully clean — the derivative has a closed form
  4. MLE under Gaussian noise gives exactly this loss (see Section 2.5)
```

**Deriving the Normal Equations (closed-form solution):**
```
Take derivative of L(β) with respect to β, set to zero:

  ∂L/∂β = −2Xᵀy + 2XᵀXβ = 0

  XᵀXβ = Xᵀy         ← the "Normal Equations"

  β* = (XᵀX)⁻¹ Xᵀy  ← the OLS estimator

What does each part mean?
  Xᵀy = "correlation between features and target"
  XᵀX = "how features correlate with each other" (feature covariance)
  (XᵀX)⁻¹ = "inverse of feature covariance" (decorrelating the features)

In words: the optimal weights are found by computing
  how much each feature correlates with the target,
  then adjusting for how features correlate with each other.
```

**Why (XᵀX)⁻¹ can fail — and what to do:**
```
(XᵀX) is NOT invertible when:
  1. Perfect multicollinearity: one feature = linear combination of others
     Example: x₃ = 2x₁ + x₂ → XᵀX is singular
  2. More features than samples: p > n
     Example: 100 patients, 200 gene expression features → underdetermined system
  3. Constant feature (zero variance) → no information → can't invert

Solutions:
  • Ridge Regression: add λI → (XᵀX + λI) is ALWAYS invertible
  • PCA preprocessing: remove near-collinear directions first
  • Feature selection: remove redundant features
```

**Numerical Alternatives (more stable than directly inverting):**
```
Via QR decomposition:
  X = QR  (Q = orthogonal matrix, R = upper triangular)
  β* = R⁻¹ Qᵀy
  Cost: O(np²) — more numerically stable than forming XᵀX directly

Via SVD (most robust, handles rank deficiency):
  X = UΣVᵀ
  β* = V Σ⁺ Uᵀy    (Σ⁺ = pseudo-inverse: replace each σᵢ with 1/σᵢ, zeros stay zero)
  Works even when X is rank-deficient (underdetermined systems)
```

**Geometric Interpretation — OLS as Projection:**
```
OLS projects y onto the COLUMN SPACE of X.

Column space of X = all possible predictions Xβ for any β

Imagine y as a vector in n-dimensional space.
The column space of X is a p-dimensional subspace within that space.
OLS finds the CLOSEST point in that subspace to y — the projection.

        y (actual values)
        ●
       /│
      / │  ← residual vector e = y − ŷ (perpendicular to col space)
     /  │
    ●───●  ← ŷ = Hŷ (projection onto column space of X)
    (prediction lies in column space)

The Hat Matrix H = X(XᵀX)⁻¹Xᵀ:
  ŷ = Hy              (projects y → predictions)
  H² = H              (idempotent: projecting twice = projecting once)
  Hᵀ = H              (symmetric)
  tr(H) = p+1         (effective degrees of freedom = number of parameters)
  Hᵢᵢ ∈ [0,1]         (leverage of data point i)

Residuals e = (I−H)y are orthogonal to the column space:
  Xᵀe = 0             ← proof that the OLS solution is exact
```

**Understanding Leverage and Influence:**
```
Leverage (Hᵢᵢ):  
  Measures how "unusual" point i is in feature space.
  High leverage (Hᵢᵢ near 1) = point is far from the others in X space.
  High leverage points can dramatically change the regression line.
  
  Normal range: Hᵢᵢ ∈ (0,1), average = (p+1)/n
  High leverage: Hᵢᵢ > 2(p+1)/n  ← rule of thumb

Cook's Distance (measures INFLUENCE = leverage × residual):
  Dᵢ = eᵢ² × Hᵢᵢ / (p × MSE × (1−Hᵢᵢ)²)

  High leverage + small residual = interesting but not influential
  High leverage + large residual = VERY influential (can distort the entire model!)
  
  Visualisation:
  
  Case 1: High leverage,        Case 2: High leverage,     Case 3: Low leverage,
          small residual                 large residual             large residual
  
  ●───────────────●             ●──────────────          ●────────────
                  ↕ small                      ● ← far   ↕ large but
                                               from line  doesn't move line
  
  Cook's D ≈ small              Cook's D ≈ LARGE            Cook's D ≈ small
```

**Gauss-Markov Theorem — Why OLS is the Best:**
```
Under these assumptions:
  1. E[ε|X] = 0           (errors have zero mean given features)
  2. Var(ε|X) = σ²I       (errors have constant variance — homoscedasticity)
  3. No perfect multicollinearity
  
OLS is the Best Linear Unbiased Estimator (BLUE):
  ✓ Unbiased: E[β*] = β_true    (correct on average)
  ✓ Best:     minimum variance among all linear unbiased estimators

IMPORTANT: "Best" among unbiased estimators.
  Ridge regression is BIASED but can have LOWER MSE by reducing variance.
  Gauss-Markov doesn't say OLS has the lowest MSE overall — just lowest variance
  among the class of linear unbiased estimators.
```

**Statistical Inference — Testing If Coefficients Are Real:**
```
Variance of the estimator:
  Var(β*) = σ² (XᵀX)⁻¹
  
Standard error of each coefficient:
  SE(βⱼ) = √[σ² × (XᵀX)⁻¹ⱼⱼ]

t-statistic for testing H₀: βⱼ = 0 (feature j has NO effect):
  tⱼ = βⱼ / SE(βⱼ)   ~  t(n−p−1)
  
  |tⱼ| > 2: coefficient is roughly "significant" at α=0.05
  p-value < 0.05: strong evidence that feature j matters

R² (proportion of variance explained):
  R² = 1 − SS_res/SS_tot = 1 − Σ(yᵢ−ŷᵢ)² / Σ(yᵢ−ȳ)²
  
  R²=0:   model explains nothing (no better than predicting the mean)
  R²=0.7: model explains 70% of variance in y
  R²=1:   perfect fit (usually means overfitting!)
  
  ⚠️ R² always increases with more features — even junk features.
     Use Adjusted R² to penalise complexity:
  
  Adjusted R² = 1 − (1−R²)(n−1)/(n−p−1)
  
  F-statistic (tests ALL coefficients jointly):
  F = (R²/p) / ((1−R²)/(n−p−1))  ~  F(p, n−p−1)
  Large F → at least one feature is genuinely predictive
```

**Gradient Descent for Linear Regression (when n is huge):**
```
When n is millions, forming XᵀX (p×p matrix) and inverting it is too slow.
Use gradient descent instead:

Gradient of MSE loss:
  ∇L = (2/n) Xᵀ(Xβ − y)
  
  = (2/n) Xᵀ(predictions − actuals)
  = (2/n) × (feature matrix)ᵀ × (error vector)
  
Update rule:
  β := β − α × ∇L
  
  α = learning rate (step size, typically 0.001–0.1)
  
Convergence visualisation:
  
  Loss │╲
       │  ╲
       │    ╲___         ← gradient descent trace
       │        ╲____
       │              ╲___
       └──────────────────── iterations
  
  Each step: β moves in the direction that most reduces the loss.
  Step too large (α too big): overshoots, oscillates.
  Step too small (α too small): takes forever to converge.
```

---

### 4.2 Ridge Regression (L2 Regularisation)

> 💡 **The problem Ridge solves:** When features are correlated (multicollinear), OLS coefficients become wildly unstable — tiny changes in the data flip the signs of large coefficients. Ridge adds a penalty for large weights, shrinking them all toward zero and dramatically stabilising the solution. The "Ridge" name comes from the ridge added to the diagonal of XᵀX.

**The Ridge Loss Function:**
```
L(β) = ||y − Xβ||² + λ||β||²₂

      = Σᵢ(yᵢ − ŷᵢ)²  +  λ Σⱼ βⱼ²
        ↑                    ↑
        fit to data           penalty for large weights

λ (lambda) = regularisation strength:
  λ=0: pure OLS (no penalty, can overfit)
  λ→∞: all weights forced to zero (total underfitting)
  λ=optimal: found via cross-validation
```

**Geometric Intuition — The Ridge Ball:**
```
Ridge solves: minimise loss  subject to  ||β||² ≤ t

  β₂
   │          ╭─── loss contours (ellipses)
   │      ╭──╯
   │  ╭──╯
   │─╯
   ●───────────────── β₁
   ↑
   Sphere constraint ||β||² ≤ t (no corners!)
   
The loss contours first touch the sphere at a smooth point
→ both coefficients are SHRUNK but neither is exactly zero
→ Ridge NEVER produces exact zeros (unlike Lasso)
```

**The Ridge Closed-Form Solution:**
```
β_ridge = (XᵀX + λI)⁻¹ Xᵀy

Key difference from OLS: add λI to XᵀX before inverting.

Why λI always makes it invertible:
  XᵀX has eigenvalues σᵢ² ≥ 0 (can be zero for collinear features)
  (XᵀX + λI) has eigenvalues σᵢ² + λ > 0 (all strictly positive!)
  → ALWAYS invertible for any λ > 0
  → Solves the multicollinearity problem
```

**Deep Insight via SVD — What Ridge Really Does:**
```
Let X = UΣVᵀ (SVD). Then:

  β_ridge = V diag(σᵢ²/(σᵢ²+λ)) Uᵀy
  
Compare to OLS:
  β_OLS   = V diag(1/σᵢ) Uᵀy  = V diag(σᵢ/σᵢ²) Uᵀy

Each component βᵢ is scaled by a "shrinkage factor":
  d_i = σᵢ²/(σᵢ²+λ)
  
  When σᵢ >> λ (well-determined direction):  d_i ≈ 1  (kept intact)
  When σᵢ << λ (near-collinear direction):   d_i ≈ 0  (heavily shrunk)

Visualisation:
  
  Shrinkage factor d_i
  1.0 │────────────╮
      │             ╲
  0.5 │              ╲
      │               ╲──────
  0.0 │                       ──────────────────
      └─────────────────────────────────────────── σᵢ²
            λ
  
  Ridge is like a smoothly adjusting dimmer switch:
  It "dims" directions with low variance (unreliable) 
  while leaving high-variance directions (reliable) alone.
```

**Bias and Variance of Ridge:**
```
Bias(β_ridge) = −λ(XᵀX + λI)⁻¹ β_true   (non-zero — Ridge is BIASED!)
Var(β_ridge)  = σ² (XᵀX+λI)⁻¹ XᵀX (XᵀX+λI)⁻¹   (strictly < OLS variance)

MSE decomposition:
  MSE(β_ridge) = Bias² + Variance
  MSE(β_OLS)   = 0     + Variance_OLS  (higher variance, zero bias)
  
There ALWAYS exists λ > 0 where MSE(β_ridge) < MSE(β_OLS)!

  Error
    │╲
    │  ╲ OLS (unbiased but high variance)
    │   ╲_______●
    │           │╲
    │      Ridge │ ╲
    │       MSE  │  ╲──────────────
    └────────────────────────────── λ
         ↑
         optimal λ

Why this makes sense: Even if Ridge introduces a small bias,
if it drastically reduces variance, the net MSE decreases.
It's better to be consistently slightly wrong than wildly random.
```

**Effective Degrees of Freedom:**
```
df(λ) = tr(H_ridge) = Σᵢ σᵢ²/(σᵢ²+λ)

  λ=0:   df = p  (full OLS, all degrees of freedom used)
  λ=∞:  df = 0  (intercept only)
  
This measures "how many effective parameters" Ridge is using.
Selecting λ via cross-validation simultaneously selects the complexity level.
```

---

### 4.3 Lasso Regression (L1 Regularisation)

> 💡 **Lasso's superpower: automatic feature selection.** While Ridge shrinks all coefficients toward zero, Lasso sets many of them to *exactly* zero — eliminating irrelevant features entirely. This is like Ridge doing the shrinkage but also deciding "this feature adds nothing, I'll remove it completely." The name stands for **L**east **A**bsolute **S**hrinkage and **S**election **O**perator.

**The Lasso Loss Function:**
```
L(β) = ||y − Xβ||² + λ Σⱼ |βⱼ|

      = Σᵢ(yᵢ − ŷᵢ)²  +  λ ||β||₁
        ↑                    ↑
        fit to data           L1 penalty (sum of ABSOLUTE values)

Key difference from Ridge: |βⱼ| instead of βⱼ²
  The L1 penalty has a KINK at zero (non-differentiable)
  This kink is what creates exact zeros — the mathematical heart of Lasso
```

**Why L1 Creates Exact Zeros (Geometric Argument):**
```
Lasso constraint: ||β||₁ ≤ t  (a DIAMOND in 2D)

  β₂
   │      ╭── loss contours (ellipses)
   │  ╭──╯
   ●  ← CORNER of L1 diamond (β₁=0, β₂>0)
   │╲
   │  ╲  ← diamond edges
   │    ╲
   └──────●──────── β₁
         ↑
         CORNER (β₁>0, β₂=0)
   
The loss contours (ellipses) ALMOST ALWAYS first touch the diamond
at a CORNER — where one coordinate is exactly zero.
→ Lasso solution has β₁=0 or β₂=0 → SPARSE!

In d dimensions: the L1 ball has corners on ALL coordinate axes.
For p=100 features: 200 corners (one per axis, positive & negative direction).
The loss contour is overwhelmingly likely to touch one of these corners first.
```

**Why Lasso Has No Closed-Form Solution:**
```
The L1 norm |βⱼ| is NOT differentiable at βⱼ=0.

  d|β|/dβ = sign(β)  for β ≠ 0
           = undefined at β = 0

We can't set ∂L/∂β = 0 and solve algebraically.
Instead, we use the SUBDIFFERENTIAL (generalised derivative):

  Subdifferential of |β| at β=0: any value in [−1, +1]
  
This leads to the concept of "stationarity conditions":
  if |∂(fit)/∂βⱼ| ≤ λ  →  βⱼ = 0 is optimal  (feature excluded)
  if |∂(fit)/∂βⱼ| > λ  →  βⱼ ≠ 0  (feature included, shrunk)
```

**Coordinate Descent — Solving Lasso One Weight at a Time:**
```
Algorithm: cycle through features, update each one while fixing all others.

For feature j, define the partial residual:
  ρⱼ = Σᵢ xᵢⱼ (yᵢ − Σₖ≠ⱼ xᵢₖ βₖ)
     = "how much does feature j correlate with the unexplained residual?"

The update for βⱼ is the SOFT-THRESHOLD OPERATOR:
  βⱼ ← S(ρⱼ, λ/2) / Σᵢ xᵢⱼ²

  Where the soft-threshold function S(ρ, λ) is:
  
  S(ρ, λ) = sign(ρ) × max(|ρ| − λ, 0)

Visualising the soft-threshold:
  
  βⱼ (output)
     │          /  ← β = ρ − λ (shrunk toward 0)
     │         /
  ───┼────────/─────────── ρ (input)
     │       /
     │      / 
     │     dead zone     /
     │◄──────────────────►
         |ρ| ≤ λ → β=0 exactly!

  • If |ρⱼ| ≤ λ: βⱼ = 0 (not enough signal to include feature j)
  • If ρⱼ > λ:  βⱼ = (ρⱼ − λ)/Σxᵢⱼ² > 0 (positive, shrunk)
  • If ρⱼ < −λ: βⱼ = (ρⱼ + λ)/Σxᵢⱼ² < 0 (negative, shrunk)
```

**The Lasso Solution Path:**
```
As λ decreases from λ_max toward 0:
  λ_max = max_j |Xⱼᵀy|/n  ← all coefficients are zero here
  
  Features enter the model ONE BY ONE as λ decreases.
  
  β
   │
β₅ │─────────────────────────────────────────
   │                       ╱─────────────────  β₂ enters
β₃ │                      ╱
   │                 ─────╱────────────────── β₃ enters  
   │          ──────╱
β₁ │─────────╱───────────────────────────────  β₁ enters first (most correlated)
   │─────────────────────────────────────────
   0 = = = = = = = = = = = = = = = = = = = ─
   └──────────────────────────────────────── λ
  λ_max                                    0
  (all zero)                            (OLS)
  
  The "optimal" λ is the one where cross-validation score peaks.
  Features that enter early (high λ) are the most important.

LARS (Least Angle Regression) computes the full path efficiently in O(np²).
```

**When Lasso Fails — Correlated Features:**
```
If x₁ and x₂ are highly correlated (say, correlation = 0.99):
  Lasso arbitrarily picks ONE and sets the other to zero.
  Which one it picks depends on tiny data perturbations.
  
  Problem in practice:
  Gene 1 and Gene 2 are 99% correlated.
  Lasso picks Gene 1 in one dataset, Gene 2 in another.
  This is UNSTABLE and scientifically misleading.
  
  Solution: Elastic Net (next section)
```

---

### 4.4 Elastic Net

> 💡 **Elastic Net = Ridge + Lasso.** It combines the best of both worlds: the grouping property of Ridge (keeping correlated features together) and the sparsity of Lasso (setting irrelevant features to exactly zero). Named "elastic" because it stretches like a net — flexible enough to handle both individual feature selection AND correlated groups.

**The Elastic Net Loss Function:**
```
L(β) = ||y−Xβ||² + λ[α||β||₁ + (1−α)||β||²]

         ↑              ↑            ↑
         MSE fit    L1 penalty   L2 penalty
                   (sparsity)   (grouping)

Parameters:
  λ = overall regularisation strength
  α ∈ [0,1] = mixing ratio
    α=1: pure Lasso
    α=0: pure Ridge
    α=0.5: equal mix (most common default)

The constraint region in 2D:
  
         β₂
          │    ╭──── Elastic net constraint
          │ ╭─╮│     (between diamond and circle)
          │╱   ╲│
    ───────●─────● ───── β₁
          │╲   ╱│
          │ ╰─╯│
          │
          
  Not as sharp as L1 (fewer corners) but still has corners
  → still produces some zeros (sparsity)
  → but the corners are "rounded" by L2 → handles correlation better
```

**The Grouping Property — Elastic Net's Key Advantage:**
```
For two perfectly correlated features x₁ and x₂ (correlation = 1.0):

  Lasso:         picks ONE, sets other to zero (arbitrary selection)
  Ridge:         sets both to equal values (grouping, but no sparsity)
  Elastic Net:   sets BOTH to equal values AND can set both to zero

  Mathematical proof (for α∈(0,1)):
  If x₁ = x₂, then in the optimal solution: β₁ = β₂
  The L2 term's gradient (2(1−α)λβⱼ) penalises any imbalance between β₁ and β₂,
  driving them toward equality.
  
Real-world impact:
  Genomics: Genes in the same pathway are correlated.
    Lasso: randomly picks one gene from the pathway.
    Elastic Net: includes ALL genes in the pathway with equal weights.
    → Elastic Net gives a more biologically meaningful result!
```

**When to Use What — Decision Guide:**
```
Feature type                    → Best choice
─────────────────────────────────────────────────────────────────
Many features, low correlation  → Lasso (clean sparse solution)
Few features, high correlation  → Ridge (stable solution)
Many features, high correlation → Elastic Net (groups + sparsity)
p >> n (more features than data)→ Elastic Net (Lasso limited to n features)
All features genuinely matter   → Ridge (no elimination needed)
Feature selection is the goal   → Lasso or Elastic Net

Practical rule: when in doubt, start with Elastic Net α=0.5
  and tune α via cross-validation.
```

**Additional Note — Lasso Cannot Select More Than n Features:**
```
When p > n (more features than samples):
  Lasso can select AT MOST n non-zero features.
  Proof: the active set of a linear programming problem has at most n constraints active.
  
  Example: 100 patients, 10,000 genes.
  Lasso: can identify at most 100 relevant genes.
  Elastic Net: no such limit — can identify hundreds of relevant genes.
  → Elastic Net is essential for genomics, NLP bag-of-words, and similar settings.
```

---

### 4.5 Polynomial & Basis Function Regression

> 💡 **The fundamental insight:** Real relationships are rarely perfectly linear. A plant doesn't grow linearly with water — too little and it dies, too much and it drowns. Polynomial regression captures curves while still using the linear regression framework. The trick: create new features that are powers or transformations of the original features. The model is still "linear in parameters" even if it fits curves.

**Polynomial Feature Expansion:**
```
Original feature: x = [house age in years]

Polynomial features up to degree 3:
  φ(x) = [1,  x,  x²,   x³  ]
         [1, age, age², age³]

Model becomes:
  ŷ = β₀ + β₁x + β₂x² + β₃x³

Still LINEAR in the βs! So we can use OLS.
But it fits CURVES in x.

Visualisation:
  
  Price │    ●                    Price │    ●    ← cubic fits better
  500k  │●     ●                  500k  │●    ●
  400k  │  ●     ●                400k  │  ● / ●
  300k  │    ●                    300k  │   ●
        └──────────── age               └──────────── age
  
  Straight line can't capture                Cubic polynomial captures
  the U-shape (new houses                     the true U-shape
  and very old houses are                    (value dips then rises)
  both expensive)
```

**The Universal Approximation Property of Polynomials:**
```
Weierstrass Approximation Theorem:
  Any continuous function f on a closed interval [a,b] can be approximated
  arbitrarily closely by a polynomial of sufficiently high degree.
  
In other words: given enough polynomial terms, you can fit ANY smooth curve.
This justifies polynomial regression as a universal tool.

BUT: high-degree polynomials OSCILLATE wildly near the boundaries!

Runge's Phenomenon (degree 9 polynomial):
  
  f(x) │    ●   ●    ←──── data points
       │●       ●●
       │           ●
  ─────┼───────────────
       │         ↓
       │    WILD oscillations ← polynomial trying to pass through all points
       │  ↑
       │ Degree 9 polynomial
  
  Solution: use SPLINES instead (piecewise polynomials)
```

**Spline Regression — Piecewise Polynomials:**
```
Instead of one high-degree polynomial, use low-degree polynomials
in "pieces" (intervals separated by KNOTS).

Cubic spline with knots at k₁, k₂, k₃:
  
  Piece 1:  β₀ + β₁x + β₂x² + β₃x³           for x < k₁
  Piece 2:  different polynomial                 for k₁ ≤ x < k₂
  Piece 3:  different polynomial                 for k₂ ≤ x < k₃
  Piece 4:  different polynomial                 for x ≥ k₃

Smoothness constraints at each knot:
  • Function is continuous (no jumps)
  • First derivative is continuous (no kinks)
  • Second derivative is continuous (no sharp bends)
  
Result: a smooth curve that fits data well without wild oscillations.

Natural cubic spline adds:
  • Second derivative = 0 at the boundary knots (linear tails outside data range)
  → More stable extrapolation
```

**Smoothing Splines — The Elegant Regularised Version:**
```
A smoothing spline minimises:

  Σᵢ (yᵢ − f(xᵢ))² + λ ∫ [f''(x)]² dx
  ↑                       ↑
  goodness of fit          penalty for CURVATURE (roughness)

  f''(x) = second derivative = curvature of the function
  ∫[f'']²dx = total curvature = roughness penalty

λ controls the bias-variance tradeoff:
  λ=0:    f interpolates every point (perfect fit, zero smoothing)
  λ=∞:    f is a straight line (maximum smoothing, linear fit)
  λ=optimal: found via cross-validation

This is IDENTICAL in spirit to Ridge regression:
  OLS term + penalty term
  The penalty here penalises function curvature instead of coefficient magnitude.
  → Smoothing splines are Ridge regression in function space.
```

**Other Basis Functions — Choosing the Right Shape:**
```
Different basis functions capture different patterns:

Radial Basis (RBF):
  φⱼ(x) = exp(−||x−μⱼ||² / (2σ²))
  Peaked around centre μⱼ, decays with distance.
  Used in: kernel methods, neural networks (RBF networks), spatial interpolation.
  
  φⱼ(x)
     │    ●
     │  ●   ●
     │●       ●
     │           ●   ●
     └──────────────────── x
           μⱼ ↑ (centre)

Sigmoid Basis (single neural network neuron!):
  φⱼ(x) = σ(wⱼᵀx + bⱼ)   where σ(z) = 1/(1+e⁻ᶻ)
  Smooth step function. Sum of these = neural network.

Fourier Basis (periodic patterns):
  φⱼ(x) = cos(ωⱼx + φⱼ)
  Perfect for: seasonal data, audio, any repeating signal.
  
  φ(x)
     │  ●       ●
     │●   ●   ●   ●
     │      ●       ●
     └────────────────── x
```

---

### 4.6 Bayesian Linear Regression

> 💡 **The key difference from OLS:** Instead of finding a single "best" set of weights β, Bayesian regression maintains a full *distribution* over all possible weights. Before seeing data, we have a prior belief about β. After seeing data, we update to a posterior belief. The result: not just a prediction, but a *probability distribution* over predictions — built-in uncertainty quantification.

**Setting Up the Bayesian Model:**
```
Prior belief about weights:
  β ~ N(μ₀, Σ₀)
  "Before seeing data, I believe weights are near μ₀ with spread Σ₀"
  
  Simple case: μ₀=0, Σ₀=(1/λ)I  → Gaussian prior with equal variance per weight
                                  → this corresponds to Ridge regularisation!

Likelihood (how data is generated):
  y|X,β ~ N(Xβ, σ²I)
  "Given weights β, each observation = linear prediction + Gaussian noise"

These are conjugate: Gaussian prior × Gaussian likelihood = Gaussian posterior.
→ The posterior is also Gaussian (closed-form update!).
```

**The Posterior Update (Closed Form):**
```
After observing data (X, y), the posterior is:

  β|X,y ~ N(μₙ, Σₙ)

Where:
  Σₙ = (Σ₀⁻¹ + (1/σ²)XᵀX)⁻¹
     ↑ posterior precision = prior precision + data precision

  μₙ = Σₙ(Σ₀⁻¹μ₀ + (1/σ²)Xᵀy)
     ↑ precision-weighted average of prior mean and MLE estimate

Interpretation:
  Σ₀⁻¹μ₀     = "prior information"  (your belief before data)
  (1/σ²)Xᵀy   = "data information"  (what data says)
  Σₙ(sum)      = weighted average based on confidence in each source
  
  With lots of data (large n):     Σₙ ≈ σ²(XᵀX)⁻¹ (data dominates, prior ignored)
  With little data (small n):      μₙ ≈ μ₀ (prior dominates, like strong regularisation)
```

**The Predictive Distribution (the real payoff):**
```
For a NEW test point x*:
  y*|x*,X,y ~ N(μₙᵀx*,  σ² + x*ᵀΣₙx*)
                  ↑              ↑
             prediction     total uncertainty
             
  σ²         = aleatoric uncertainty (irreducible noise in the data itself)
  x*ᵀΣₙx*    = epistemic uncertainty (our uncertainty about β)
              = SHRINKS as we get more data
              = GROWS when x* is far from training data!

Visualisation of prediction uncertainty:
  
  y │         95% CI gets WIDER here
    │     ●  ●  ●      ────╮         ← few training points in this region
    │   ●  ●      ───────╯│
    │ ●  ●  ─────────────╮│          ← uncertainty band
    │─────────────        ╰───────── 
    │     ●  ●  ●  (training data)   ← narrow CI where data is dense
    └────────────────────────────── x
       (training range)    (extrapolation)
    
  Bayesian regression correctly says: "I'm less sure here — I have no data!"
  OLS gives no such signal.

This is the KEY advantage: automatic uncertainty calibration.
OLS gives a point prediction everywhere with equal (false) confidence.
Bayesian regression gives WIDER confidence intervals where data is sparse.
```

**Online Bayesian Updating — Sequential Learning:**
```
New data point (x_{n+1}, y_{n+1}) arrives. Update via Woodbury identity:

  Σₙ₊₁ = Σₙ − Σₙxₙ₊₁(σ² + xₙ₊₁ᵀΣₙxₙ₊₁)⁻¹xₙ₊₁ᵀΣₙ   ← rank-1 update

Cost: O(d²) per new point (vs O(d³) to refit from scratch)

This enables true online/streaming learning:
  • Collect a measurement
  • Update the posterior distribution
  • Repeat
  
At any time, the posterior captures EVERYTHING learned from all data so far.
```

---

## 5. Classification Algorithms

> 💡 **What is classification?** Instead of predicting a continuous number, we predict a *category*. Is this email spam or not? Which digit (0-9) is in this image? Does this tumour image show malignant or benign tissue? The goal is to learn a *decision boundary* — a dividing line (or surface) that separates different classes.

---

### 5.1 Logistic Regression

> 💡 **Why not just use linear regression for classification?** Say you predict spam=1, not-spam=0. Linear regression can predict values outside [0,1] like 1.7 or −0.3 — these are meaningless as probabilities! Logistic regression squashes the output through a sigmoid function, guaranteeing outputs in (0,1) that can be interpreted as probabilities.

**The Sigmoid (Logistic) Function:**
```
σ(z) = 1/(1+e⁻ᶻ)

Properties:
  σ(0)    = 0.5   (boundary: 50-50)
  σ(+∞)   → 1.0   (very confident positive)
  σ(−∞)   → 0.0   (very confident negative)
  σ'(z)   = σ(z)(1−σ(z))  ← elegant self-referential derivative

Visualisation:
  
  P(y=1)
  1.0 │                     ─────────────
      │                 ╱
  0.5 │────────────────●─────────────────
      │            ╱
  0.0 │──────────────────────────────────
      │         ╱
     -5        0        +5        z = βᵀx
      
  At z=0: exactly 50% probability
  z positive → probability > 50% → predict class 1
  z negative → probability < 50% → predict class 0
```

**Why the Log-Odds (Logit) Link?**
```
The logistic model is:
  P(y=1|x) = σ(βᵀx) = 1/(1+exp(−βᵀx))

Equivalently:
  log[P(y=1)/P(y=0)] = βᵀx
  ↑ log-odds (logit)
  
In English: the LOG-ODDS of being class 1 is a LINEAR function of features.

This arises naturally from the exponential family:
  Bernoulli distribution has natural parameter η = log(p/(1−p))
  Setting βᵀx = η is the "canonical link" for Bernoulli GLM.
  → Logistic regression is EXACTLY the canonical GLM for binary outcomes.

Interpretation of coefficients:
  βⱼ = change in LOG-ODDS per unit increase in xⱼ
  exp(βⱼ) = ODDS RATIO — how much the odds multiply per unit increase
  
  Example: β_age = 0.05 for heart disease
    Each year of age multiplies the odds by e^0.05 ≈ 1.05 (5% increase per year)
```

**Why Cross-Entropy Loss (not MSE):**
```
With a sigmoid output, using MSE creates problems:

MSE with sigmoid is NON-CONVEX:
  
  Loss │    ●
       │  ●   ●   ← local minima!
       │●       ●
       │
       └─────────── β
  (gradient descent gets stuck)
  
Cross-entropy is CONVEX:
  
  Loss │●
       │  ●
       │    ●
       │      ●
       │        ●──────────
       └─────────────────── β
  (gradient descent always finds the global minimum)

Binary Cross-Entropy Loss:
  L = −(1/n) Σᵢ [yᵢ log σ(βᵀxᵢ) + (1−yᵢ) log(1−σ(βᵀxᵢ))]

When yᵢ=1 (true positive):
  L = −log σ(βᵀxᵢ) = −log(predicted probability of correct class)
  High probability prediction → low loss ✓
  Low probability prediction → high loss (log of small number → large negative → negate → large positive)

When yᵢ=0 (true negative):
  L = −log(1−σ(βᵀxᵢ)) = −log(predicted probability of negative class)
```

**The Gradient — Elegant and Interpretable:**
```
∂L/∂β = (1/n) Xᵀ(σ(Xβ) − y)
                ↑           ↑
          predictions   actual labels
        
= (1/n) × features × (prediction_error)

This beautifully says:
  "Adjust weights proportionally to the prediction error,
   scaled by the feature values."
   
  • If prediction = 0.9 but truth = 0 (confidently wrong!): large gradient update
  • If prediction = 0.55 but truth = 0 (slightly wrong): small gradient update
  • If prediction ≈ truth: near-zero gradient (already good)

This form is IDENTICAL to linear regression gradient! 
The only difference is what "predictions" means:
  Linear: predictions = Xβ  (linear)
  Logistic: predictions = σ(Xβ)  (sigmoid-transformed)
This unification happens because both are GLMs.
```

**Newton's Method / IRLS for Logistic Regression:**
```
Newton's method applies 2nd-order optimisation (uses curvature):
  
  Hessian of logistic loss:
  H = (1/n) XᵀWX
  W = diag[σ(xᵢ)(1−σ(xᵢ))]  ← weight matrix
  
  Update: β := β − H⁻¹∇L

The weights W have an elegant interpretation:
  w_i = σ(xᵢ)(1−σ(xᵢ)) = variance of Bernoulli(σ(xᵢ))
  
  σ=0.5 (boundary): w_i=0.25 (MAXIMUM weight, most informative)
  σ=0 or 1 (certain): w_i≈0 (near zero — already classified correctly)
  
  → Newton's method focuses learning effort on UNCERTAIN points near the boundary.
  This is called IRLS (Iteratively Reweighted Least Squares) — each Newton step
  is equivalent to solving a weighted least squares problem.
```

**Multinomial Logistic (Softmax) for Multiple Classes:**
```
For K classes:
  P(y=k|x) = exp(βₖᵀx) / Σⱼ exp(βⱼᵀx)   ← softmax

In words: assign exponential score to each class, then normalise to probabilities.

Temperature Scaling:
  P(y=k|x,T) = exp(βₖᵀx/T) / Σⱼ exp(βⱼᵀx/T)
  
  T→0: one class gets all the probability (argmax, completely confident)
  T=1: standard softmax
  T→∞: uniform distribution (completely uncertain)
  
  Temperature is used in LLM sampling to control "randomness" of generation.

Non-Identifiability:
  Adding any constant vector c to ALL βₖ leaves probabilities unchanged.
  Fix: set β_K = 0 (reference class), use L2 regularisation, or
       use softmax with K-1 free parameters.
```

---

### 5.2 K-Nearest Neighbours (KNN)

> 💡 **KNN is the simplest ML algorithm imaginable.** To classify a new point, find the K training points closest to it and take a vote. No training phase, no parameters to learn — just store all training data and compute distances at prediction time. Deceptively simple, surprisingly effective, and a fantastic baseline.

**The Algorithm:**
```
1. Store all (xᵢ, yᵢ) training pairs.

2. To classify new point x_new:
   a. Compute distance from x_new to ALL training points
   b. Find K nearest neighbours (closest K points)
   c. Take majority vote among their labels

Visualisation (K=3):

  ● ○ ●         x_new = ?
  ● ● ○ ○         │
      ○ ⭐ ← x_new │
       ○ ●         │
                   ↓
  3 nearest: ○ ○ ●  → 2 circles, 1 dot
  Majority vote: ○  → classify as ○ !
```

**Distance Metrics — Choosing the Right Geometry:**
```
Euclidean (L2):   d = √Σᵢ(xᵢ−yᵢ)²
  The "as the crow flies" distance.
  ✓ Most intuitive, default choice.
  ✗ Sensitive to feature scale → ALWAYS standardise first!
  
Manhattan (L1):   d = Σᵢ|xᵢ−yᵢ|
  "City block" distance (only horizontal/vertical moves).
  ✓ More robust to outliers than Euclidean.
  ✓ Better in high dimensions (less affected by curse of dimensionality).
  
Cosine:           d = 1 − xᵀy/(||x|| ||y||)
  Measures angle between vectors, ignores magnitude.
  ✓ Perfect for text (TF-IDF vectors) — document length doesn't matter.
  ✓ For NLP: "king" and "kings" should be close regardless of count.
  
Mahalanobis:      d = √((x−y)ᵀΣ⁻¹(x−y))
  Accounts for feature correlation AND scale simultaneously.
  ✓ Equivalent to Euclidean after whitening the data.
  ✓ Essential when features have very different scales AND are correlated.
  
  Visualisation:
  
  Euclidean:       Mahalanobis (elongated data):
  ●●●●●●          ●●●●●●●●●●●●
  ●●●●●●          ●●●●●●●●●●●
  ●●●●●●          ●●●●●●●●●
  (circles = iso-  (ellipses align with
  distance curves)  data covariance)
```

**The Curse of Dimensionality:**
```
In HIGH dimensions, KNN breaks down completely.

Mathematical fact:
  Vol(d-sphere of radius r) / Vol(d-cube of side 2r) = πᵈ/² / (2ᵈ Γ(d/2+1)) → 0 as d → ∞

In English: in high dimensions, almost ALL the volume of a hypercube
  is in its CORNERS, not near the centre.
  
  d=2 (2D):        d=10:            d=100:
  
  ████████         only corners:    ALL corners, no centre
  ██ ●● ██         ●             ●
  ██ ●● ██                         ●
  ████████         ●             ●
  
  Sphere/cube       Ratio = 0.24     Ratio ≈ 0.000000000001

Practical consequence: With n=10,000 points in d=100 dimensions,
  points are so spread out that "nearest neighbours" are essentially
  as far as random points → neighbourhood becomes meaningless.

The ratio of max-to-min distance:
  max_dist/min_dist → 1 as d → ∞

→ ALL points become "equally far" in very high dimensions!

Solutions:
  • Dimensionality reduction (PCA, UMAP) before KNN
  • Feature selection (keep only informative features)
  • Use cosine distance (less affected) for high-d text data
  • Switch to other algorithms for d > 20-30
```

**Bias-Variance in KNN:**
```
K=1 (most complex):
  Decision boundary follows every point exactly.
  Bias ≈ 0, Variance = HIGH (sensitive to noise)
  
K=n (simplest):
  Everything gets the majority class label.
  Bias = HIGH, Variance ≈ 0
  
K=optimal:
  Found via cross-validation, typically K=√n as a starting point.
  
  ┌─────────────────────────────────────────────────────────────────┐
  │ K=1 boundary      K=5 boundary        K=15 boundary           │
  │ (overfit)          (balanced)           (underfit)              │
  │                                                                  │
  │ ████░░░███░       ██████░░░░░          ███████████             │
  │ ██░░░░░███        █████░░░░░░          ███████████             │
  │ █░░░░░░░██        ████░░░░░░░          ████░░░░░░░             │
  │ ░░░░░░░░░█        ████░░░░░░░          ░░░░░░░░░░░             │
  │ ░░░░░░░░░░        ░░░░░░░░░░░          ░░░░░░░░░░░             │
  └─────────────────────────────────────────────────────────────────┘
  
  K=1: jagged, complex        K=15: smooth, simple
  boundary (high variance)     boundary (higher bias)
```

**Weighted KNN:**
```
Instead of equal votes, weight by inverse distance:
  ŷ = Σⱼ (1/dⱼ) yⱼ / Σⱼ (1/dⱼ)

Intuition: closer neighbours should have more say.
  A point 0.1 away has 10× the vote of a point 1.0 away.
  
Generalised: weight = 1/dⱼᵖ where p controls how fast influence decays.
```

---

### 5.3 Naive Bayes

> 💡 **Naive Bayes is built on one assumption that sounds absurd but works surprisingly well.** The "naive" part is assuming that ALL features are INDEPENDENT given the class label. For text classification: given that an email is spam, the word "FREE" appearing is independent of the word "MONEY" appearing. That's clearly not true in reality, but the classifier still works well in practice. Why? Because we only need the *decision boundary* to be correct, not the *probabilities*.

**The Model:**
```
We want: P(class | features) = P(class) × P(features | class) / P(features)

Naive independence assumption:
  P(x₁, x₂, ..., xₙ | class) = P(x₁|class) × P(x₂|class) × ... × P(xₙ|class)

Classification rule:
  ĉ = argmax_c [log P(C=c) + Σⱼ log P(xⱼ|C=c)]
                ↑              ↑
            log-prior    sum of log-likelihoods for each feature
  
Using logs: products of small probabilities → sums (numerically stable)

Example — spam detection:
  P("FREE"|spam) = 0.30   P("FREE"|ham) = 0.01
  P("MONEY"|spam) = 0.20  P("MONEY"|ham) = 0.005
  P(spam) = 0.40          P(ham) = 0.60
  
  Email contains "FREE" and "MONEY":
  
  Log score for spam:
    log(0.40) + log(0.30) + log(0.20) = −0.916 − 1.204 − 1.609 = −3.729
  
  Log score for ham:
    log(0.60) + log(0.01) + log(0.005) = −0.511 − 4.605 − 5.298 = −10.414
  
  Spam score (−3.73) >> Ham score (−10.41) → Classified as SPAM ✓
```

**Three Variants of Naive Bayes:**
```
1. Gaussian Naive Bayes (continuous features):
   P(xⱼ|C=c) = N(xⱼ; μⱼc, σ²ⱼc)
   = (1/√(2πσ²ⱼc)) × exp(−(xⱼ−μⱼc)²/(2σ²ⱼc))
   
   Learn: sample mean μⱼc and variance σ²ⱼc for each feature j in each class c
   Use for: continuous features (age, height, pixel intensity)

2. Multinomial Naive Bayes (word counts):
   P(wⱼ|C=c) = (N_cj + α)/(N_c + α|V|)  ← Laplace smoothed!
   N_cj = count of word j in class c
   N_c = total words in class c
   |V| = vocabulary size
   α = smoothing parameter (usually α=1)
   
   Use for: text classification, bag-of-words features

3. Bernoulli Naive Bayes (binary features):
   P(xⱼ|C=c) = pⱼc^xⱼ × (1−pⱼc)^(1−xⱼ)
   
   Treats each feature as binary (word present/absent).
   Key difference from Multinomial: PENALISES absent words!
   P(word absent|class) is explicitly modelled.
   
   Use for: short texts, binary feature vectors
```

**The Zero-Frequency Problem and Laplace Smoothing:**
```
Problem: 
  Word "neuroscience" never appears in training spam emails.
  P("neuroscience"|spam) = 0/total = 0
  
  Now: P(email with "neuroscience" | spam) = ... × 0 × ... = 0
  
  ONE unseen word makes the ENTIRE probability zero!
  Any email with an unseen word gets probability 0 for all classes.
  → Classifier breaks completely on new vocabulary.

Solution — Laplace Smoothing (add-1 smoothing):
  P(wⱼ|C=c) = (count(wⱼ,c) + 1) / (total_words_in_c + |V|)
  
  "Pretend every word appeared at least once" → no more zeros.
  
  Generalised (add-α smoothing):
  P(wⱼ|C=c) = (count(wⱼ,c) + α) / (total_words_in_c + α×|V|)
  
  α=1: uniform prior (equal pseudocounts for all words)
  α<1: sub-linear smoothing (less aggressive)

Why This Works:
  Adding α is equivalent to a Dirichlet(α) prior on word probabilities.
  MAP estimation with this prior = add-α smoothing.
  → Laplace smoothing is a form of Bayesian regularisation.
```

**Why Naive Bayes Works Despite the Independence Lie:**
```
Key insight from Domingos & Pazzani (1997):
  We don't need correct probabilities — we need the correct DECISION.
  
  NB is correct if:
    arg max_c P(C=c|x) is the same whether features are dependent or not.
    
  This holds when:
    The dependency structure among features is the same across all classes.
    (Equal dependencies cancel out in the ratio P(spam|x)/P(ham|x))
    
  In practice: NB works amazingly well for text because:
    • d × K parameters (not d² × K) — VERY low variance
    • Low variance → works well with small datasets
    • Even if probabilities are wrong, rankings are often right
    • Text features ARE relatively independent (words appear independently of order)

Calibration Warning:
  NB probabilities are often poorly calibrated (overconfident).
  P̂(spam) = 0.9999 when true probability might be 0.7.
  Fix: apply Platt scaling or isotonic regression after getting NB scores.
```

---

### 5.4 Decision Trees

> 💡 **Decision trees are like a flowchart of yes/no questions.** Starting from the root, at each node you ask a question about a feature ("Is income > $50,000?") and follow the appropriate branch. At the end (leaf node) you get a prediction. The algorithm learns which questions to ask and in what order to best separate the classes.

**The Tree Structure:**
```
A decision tree for predicting loan default:

                    Income > $40k?
                   /              \
                YES                NO
               /                    \
       Credit > 700?            Income > $25k?
       /        \                /           \
     YES          NO           YES             NO
      │            │            │               │
  [No default]  [Default]  [No default]     [Default]
   (Leaf)         (Leaf)      (Leaf)           (Leaf)

Each internal node: a SPLIT on a feature
Each leaf: a PREDICTION (class label or probability)
Each path: a RULE

Example rule:
  IF income > 40k AND credit > 700 THEN no default
  IF income ≤ 40k AND income > 25k THEN no default
  IF income ≤ 40k AND income ≤ 25k THEN DEFAULT
```

**Splitting Criteria — How to Choose the Best Question:**
```
At each node, we have a set S of training examples.
We want to split S into subsets {Sᵥ} that are as PURE as possible
(each subset contains mostly one class).

IMPURITY MEASURES:

1. Gini Impurity:
   G(S) = 1 − Σₖ pₖ²
   pₖ = fraction of class k in S
   
   Example: S = {5 spam, 5 ham}:  G = 1 − (0.5² + 0.5²) = 0.5  (maximum impurity)
            S = {9 spam, 1 ham}:  G = 1 − (0.9² + 0.1²) = 0.18  (low impurity)
            S = {10 spam, 0 ham}: G = 1 − (1.0²) = 0  (pure!)
   
   Range: [0, 0.5] for binary; [0, 1−1/K] for K classes

2. Information Gain (Entropy-based):
   H(S) = −Σₖ pₖ log₂ pₖ
   
   Same example: S = {5,5}: H = −(0.5×log₂0.5 + 0.5×log₂0.5) = 1 bit (maximum)
                 S = {9,1}: H = −(0.9×log₂0.9 + 0.1×log₂0.1) = 0.47 bits
                 S = {10,0}: H = 0 bits (pure — no surprise)
   
   Information Gain of a split A:
   IG(S, A) = H(S) − Σᵥ (|Sᵥ|/|S|) H(Sᵥ)
             ↑        ↑
         before split  weighted average entropy AFTER split
   
   Choose the split A that MAXIMISES information gain.

3. Misclassification Rate:
   E(S) = 1 − max_k pₖ
   
   Rarely used for building trees (less sensitive to purity changes).

Gini vs Entropy:
  For binary classification, Gini and Entropy give nearly identical trees.
  Both peak at 0.5 (equal classes), zero at 1.0 (pure).
  Gini is faster (no log computation) → default in most implementations.
  Entropy is slightly more sensitive to changes near pure nodes.
```

**Why Gain Ratio Prevents Bias Toward High-Cardinality Features:**
```
Problem with plain Information Gain:
  Feature "CustomerID" achieves PERFECT information gain:
    Every leaf has exactly 1 customer → perfectly pure!
    But it completely overfits and has ZERO generalisation.
  
  More generally: any feature with many distinct values tends to have higher IG.

Solution — Gain Ratio (C4.5 algorithm):
  GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)
  SplitInfo(S, A) = −Σᵥ (|Sᵥ|/|S|) log₂(|Sᵥ|/|S|)
  
  SplitInfo measures how EVENLY the split divides the data:
    Many small equal-size branches → high SplitInfo → heavily penalised
    Few large branches → low SplitInfo → less penalised
  
  CustomerID: SplitInfo = −(1/n × log₂(1/n)) × n = log₂(n) ≈ huge
              GainRatio = IG/log₂(n) ≈ tiny → correctly penalised
```

**Regression Trees:**
```
For regression (predicting numbers instead of classes):

Split criterion: minimise VARIANCE of target in each child:
  min Σᵥ (|Sᵥ|/|S|) Var(yᵢ ∈ Sᵥ)
  
  = minimise within-group variance
  = maximise between-group variance (explained variance)
  
Leaf prediction: mean of all training samples in that leaf
  ŷ = (1/|leaf|) Σᵢ∈leaf yᵢ

Visualisation:
  
  y     ─────●●●●  ← leaf 1: ŷ=8
  8     │
  6     │●●●       ← leaf 2: ŷ=5.5  
  5     │                   
  3 ●●● │           ← leaf 3: ŷ=2.5
        └────────── x₁
             └── split point
```

**Cost-Complexity Pruning — Preventing Overfitting:**
```
Fully grown tree: zero training error, but massive overfitting.
Need to prune back to a simpler tree.

Cost-Complexity objective:
  R_α(T) = R(T) + α × |T|
          ↑         ↑    ↑
       misclassification  penalty     number of leaves
       rate on training             (complexity)

For small α: prefer complex trees (low R(T))
For large α: prefer simple trees (small |T|)

Algorithm:
  1. Grow full tree T_max
  2. For each internal node t: compute "weakest link" = (R(T_pruned)−R(T)) / (|T|−|T_pruned|)
  3. Prune the subtree with smallest weakest link
  4. Repeat → sequence of nested trees T_max ⊃ T₁ ⊃ T₂ ⊃ ... ⊃ T_root
  5. Use cross-validation to select the best tree in this sequence

Why pruning works:
  Unpruned: memorises noise, high variance, low bias
  Pruned:   simpler rules, higher bias, LOWER variance
  → Cross-validated pruning finds the bias-variance sweet spot
```

---

### 5.5 Support Vector Machines (SVM)

> 💡 **SVMs find the safest possible decision boundary.** Imagine you're placing a road between two groups of houses. You could put the road anywhere in the gap — but you'd choose the route that keeps the maximum distance from both sides. That's exactly what SVMs do: they find the hyperplane that maximises the MARGIN between classes, making the classifier as "safe" as possible against small perturbations.

**The Hard-Margin SVM (Linearly Separable Data):**
```
Two classes: y ∈ {−1, +1}

Decision boundary: wᵀx + b = 0

Classification rule: ŷ = sign(wᵀx + b)
  wᵀx + b > 0 → predict +1
  wᵀx + b < 0 → predict −1

The margin:
  Distance from any point xᵢ to the hyperplane = |wᵀxᵢ + b| / ||w||
  For correctly classified points: yᵢ(wᵀxᵢ + b) ≥ 1
  
  Margin width = 2/||w||  (distance between the two margin hyperplanes)
  
  Visualisation:
  
  ●  ●                 Margin = 2/||w||
  ●  ●  ║  ○  ○        ←───────────────→
  ●  ●  ║  ○  ○
     ●  ║  ○
    ━━━━━┃━━━━━    ← decision boundary (wᵀx + b = 0)
  −−−−−−┃−−−−−−   ← support vectors touch these margins
  wᵀx+b=−1 │ wᵀx+b=+1
  
  Support vectors: the data points closest to the boundary
                   (the ones sitting exactly ON the margin lines)
  These are the ONLY points that determine the solution!
  All other points don't matter.

Optimisation:
  Minimise: (1/2)||w||²         ← makes margin 2/||w|| large
  Subject to: yᵢ(wᵀxᵢ+b) ≥ 1  ← all points correctly classified with margin
```

**The Lagrangian and Dual Formulation:**
```
Primal problem: minimise (1/2)||w||² subject to constraints.

Lagrangian:
  L(w, b, α) = (1/2)||w||² − Σᵢ αᵢ[yᵢ(wᵀxᵢ+b) − 1]
  αᵢ ≥ 0 = Lagrange multipliers (one per data point)

Taking derivatives and setting to zero:
  ∂L/∂w = 0 →  w = Σᵢ αᵢyᵢxᵢ  (solution is a linear combination of training points!)
  ∂L/∂b = 0 →  Σᵢ αᵢyᵢ = 0

Substituting back → DUAL problem:
  Maximise: Σᵢαᵢ − (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼ xᵢᵀxⱼ
  Subject to: αᵢ ≥ 0, Σᵢαᵢyᵢ = 0

Key insight: data only appears as INNER PRODUCTS xᵢᵀxⱼ!
→ This enables the KERNEL TRICK (see below).

KKT complementary slackness condition:
  αᵢ[yᵢ(wᵀxᵢ+b) − 1] = 0
  
  αᵢ > 0 only for points ON the margin → these are SUPPORT VECTORS.
  All other αᵢ = 0 → those points don't influence w at all!
  
  Sparsity: typically only a small fraction of training points become support vectors.
```

**Soft-Margin SVM (Handling Non-Separable Data):**
```
Real data is never perfectly separable. Allow some violations:

  Minimise: (1/2)||w||² + C Σᵢ ξᵢ
  Subject to: yᵢ(wᵀxᵢ+b) ≥ 1 − ξᵢ,  ξᵢ ≥ 0

  ξᵢ = slack variable (how much point i violates the margin)
  C = regularisation parameter:
    Large C: penalise violations heavily → smaller margin, fewer errors
    Small C: tolerate violations → larger margin, more errors
    
  Visualisation:
  
  ●  ●  ║  ○  ○
  ●  ●  ║× ○  ○   ← × is a support vector violation (ξ > 0)
  ●  ×  ║  ○
  ━━━━━━┃━━━━━━━   ← decision boundary
     violation ↑ ξᵢ > 0

  C=∞: hard margin (no violations allowed)
  C=1: moderate regularisation (some violations OK)
  C=0: no penalty (all points can be violations)

Hinge Loss Perspective:
  SVM objective = λ||w||² + (1/n)Σᵢ max(0, 1−yᵢ(wᵀxᵢ+b))
                                      ↑
                                  hinge loss
  
  Hinge loss is 0 when point is correctly classified WITH margin.
  It grows linearly when the point is inside the margin or misclassified.
  
  Hinge loss vs 0-1 loss:
  Loss│     ╲
     │  0-1: ╲──────────────
     │  ╲
     │   ╲ hinge: ╲
     │         ╲───────────
     └─────────────────────── yf(x)
          0    1
```

**The Kernel Trick — Implicitly Operating in Infinite Dimensions:**
```
The SVM dual depends on data ONLY through inner products xᵢᵀxⱼ.

Kernel function K(xᵢ, xⱼ) = φ(xᵢ)ᵀφ(xⱼ)

We can REPLACE xᵢᵀxⱼ with K(xᵢ, xⱼ) without explicitly computing φ!

Classification becomes:
  f(x) = sign(Σᵢ αᵢyᵢ K(xᵢ,x) + b)

Common kernels:
  Linear:       K(x,z) = xᵀz              (standard SVM)
  Polynomial:   K(x,z) = (γxᵀz + r)ᵈ      (degree-d polynomial features)
  RBF/Gaussian: K(x,z) = exp(−γ||x−z||²)  (infinite-dimensional feature space!)
  Sigmoid:      K(x,z) = tanh(γxᵀz + r)

RBF Kernel Intuition:
  K(x,z) = exp(−γ||x−z||²)
  
  As x→z: K→1 (identical points are maximally similar)
  As ||x−z||→∞: K→0 (distant points are unrelated)
  
  γ controls the "reach" of each support vector:
  
  γ large:         γ small:
  ◎ ← tight bubble  ○ ← wide bubble
  
  Large γ: complex boundary (each support vector affects small region)
  Small γ: smooth boundary (each support vector affects large region)

Mercer's Theorem:
  K is a valid kernel ↔ the n×n Gram matrix K_{ij} = K(xᵢ,xⱼ) is PSD.
  Ensures the dual problem remains convex (solvable to global optimum).
```

---

### 5.6 Discriminant Analysis

> 💡 **LDA and QDA are "generative" classifiers.** Instead of directly learning the decision boundary (like logistic regression), they model the *distribution of each class* and then use Bayes' theorem to classify. This is like learning "what spam emails look like" and "what ham emails look like," then asking "which type does this new email look more like?"

**Linear Discriminant Analysis (LDA):**
```
Assumption: Each class has a Gaussian distribution WITH THE SAME covariance:
  P(x|C=k) = N(x; μₖ, Σ)   ← shared Σ across all classes

By Bayes' theorem:
  P(C=k|x) ∝ P(x|C=k) × P(C=k)
            = N(x; μₖ, Σ) × πₖ

Taking log and keeping only terms that depend on k:
  δₖ(x) = xᵀΣ⁻¹μₖ − (1/2)μₖᵀΣ⁻¹μₖ + log πₖ

Classification: ĉ = argmax_k δₖ(x)

Why is this called LINEAR?
  δₖ(x) is LINEAR in x (it's xᵀ[Σ⁻¹μₖ] + constant)
  → Decision boundaries are HYPERPLANES (linear)
  → Shared covariance → shared "shape" of classes, only centres differ

Estimating parameters from training data:
  π̂ₖ = nₖ/n                         (class frequencies)
  μ̂ₖ = (1/nₖ) Σᵢ∈Cₖ xᵢ              (class means)
  Σ̂  = (1/(n−K)) Σₖ Σᵢ∈Cₖ (xᵢ−μ̂ₖ)(xᵢ−μ̂ₖ)ᵀ  (pooled within-class covariance)
```

**LDA as Dimensionality Reduction (Fisher's Linear Discriminant):**
```
Equivalent view: project data onto directions that BEST SEPARATE classes.

Find projection W that maximises:
  J(W) = |WᵀS_B W| / |WᵀS_W W|
  ↑ ratio of BETWEEN-class scatter to WITHIN-class scatter

  S_W = Σₖ Σᵢ∈Cₖ (xᵢ−μₖ)(xᵢ−μₖ)ᵀ    ← within-class scatter
  S_B = Σₖ nₖ(μₖ−μ)(μₖ−μ)ᵀ           ← between-class scatter
  μ = overall mean

Solution: eigenvectors of S_W⁻¹S_B

Maximum number of discriminant directions: K−1 (for K classes)
  Binary (K=2): 1 direction (project to a LINE)
  3 classes: 2 directions (project to a PLANE)

Visualisation (2 classes, 2D → 1D):

  x₂ │    ●●●●
     │  ●●●●  ○○○○  ← original 2D data
     │          ○○○○
     └──────────────── x₁
     
  Project onto best direction:
  
  ●●●●●○○○○○○   ← 1D projection (classes well separated)
  ───────────────
         ↑ decision boundary in 1D
```

**Quadratic Discriminant Analysis (QDA):**
```
Relaxes LDA's shared covariance assumption:
  P(x|C=k) = N(x; μₖ, Σₖ)   ← SEPARATE covariance per class

Discriminant function (NOW QUADRATIC in x):
  δₖ(x) = −(1/2)log|Σₖ| − (1/2)(x−μₖ)ᵀΣₖ⁻¹(x−μₖ) + log πₖ

The (x−μₖ)ᵀΣₖ⁻¹(x−μₖ) term is quadratic in x → elliptical decision boundaries.

LDA vs QDA:

  LDA (linear boundaries):          QDA (quadratic/curved boundaries):
  ┌──────────────────────┐          ┌──────────────────────┐
  │●●●●│○○○○            │          │●●●●  ╭──╮  ○○○○   │
  │●●●●│  ○○○○          │          │●●●  ╱    ╲  ○○○○  │
  │●●●●│    ○○○○        │          │●●  │ ●●●● │   ○○  │
  └──────────────────────┘          └──────────────────────┘
  Shared Σ → straight line           Different Σ → curved boundary

LDA vs Logistic Regression:
  Both: linear decision boundaries for binary classification
  
  LDA: models P(x|y) generatively (assumes Gaussian features)
    → Better when Gaussian assumption holds (small data)
    → More efficient if assumptions are correct (fewer params needed)
  
  Logistic: models P(y|x) discriminatively (no feature distribution assumed)
    → Better when features are NOT Gaussian (robust to violations)
    → Preferred for large datasets with non-Gaussian features
```

---

## 6. Clustering & Unsupervised Learning

> 💡 **What is unsupervised learning?** We have data but NO labels. We want to find hidden structure — natural groupings, patterns, or organisation in the data. Clustering is the most common form: find groups (clusters) of similar data points, without being told what the groups should be.

---

### 6.1 K-Means Clustering

> 💡 **K-Means is the "Hello World" of clustering.** Divide n points into K groups, where each group is as tight as possible (minimises within-group distance). Repeat until stable. It's simple, fast, and a great starting point for any clustering task.

**The Objective:**
```
WCSS (Within-Cluster Sum of Squares):
  J = Σₖ Σᵢ∈Cₖ ||xᵢ − μₖ||²
  
  Where μₖ = centroid (mean) of cluster k

Minimise J over:
  • Cluster assignments: which points go in which cluster
  • Centroids: where the cluster centres are

Both are HARD to optimise simultaneously (NP-hard exactly).
Lloyd's algorithm gives a local minimum efficiently.
```

**Lloyd's Algorithm (the Standard K-Means):**
```
Step 1: INITIALISE K centroids (random or K-means++)

Step 2: ASSIGN each point to nearest centroid:
  cᵢ = argmin_k ||xᵢ − μₖ||²   ← find which centroid is closest

Step 3: UPDATE centroids to be the mean of assigned points:
  μₖ = (1/|Cₖ|) Σᵢ∈Cₖ xᵢ        ← mean of all points in cluster k

Step 4: REPEAT steps 2-3 until assignments don't change.

Convergence Proof:
  Each step (assign + update) monotonically DECREASES J:
  
  • Assignment step: each point moved to NEARER centroid → J decreases
  • Update step: centroid = mean MINIMISES sum of squared distances → J decreases
  
  J ≥ 0 and decreasing → must converge in finite steps!
  BUT: converges to LOCAL minimum, not global minimum.
  
  Solution: run multiple random restarts, keep best solution.

Visualisation of iterations:

  Iter 0 (random init):    Iter 1 (assign+update):   Iter 2 (converged):
  
  ● ● ×  ● ×  ○           ●●●   │  ●○●○  ●○         ●●●│ ●●●○  ○○
  ● ●    ●    ○ ○          ●●   │  ●  ○  ○ ○         ●●│   ●  ○ ○
  ●  ○   ● × ○ ○            ●  │   ●   ○  ○            ●│    ●  ○
       ×  ○                    │  ○  ○  ○                │ ○  ○  ○
  × = initial centroids      centroids moved            stable!
```

**K-Means++ Initialisation (Smarter Starting Points):**
```
Problem with random initialisation:
  Bad luck → centroids start far from optimal → slow convergence or poor solution

K-means++ algorithm:
  1. Choose first centroid uniformly at random
  2. For each remaining centroid:
     a. Compute D(x) = min distance from x to any chosen centroid
     b. Choose next centroid with probability ∝ D(x)²
     c. → Points FAR from existing centroids are MORE LIKELY to be chosen
  3. Continue until K centroids are chosen

Intuition: we SPREAD out the initial centroids:
  
  After 1st centroid:        After 2nd centroid:
  
  ●●●●  ●●●●  ●●●●           ●●●●  ●●●●  ●●●●
  ●●×●  ●●●●  ●●●●           ●●×●  ●●●●  ●●×●
  ●●●●  ●●●●  ●●●●           ↑              ↑
    ↑ 1st centroid         chose far away (high D²)

Theoretical guarantee:
  E[WCSS with K-means++] ≤ 8(ln K + 2) × WCSS_optimal
  → At most 8(ln K + 2) times the optimal → much better than random init in practice
```

**When K-Means Fails:**
```
K-Means assumes: SPHERICAL clusters of SIMILAR SIZE.

Fails for:
  1. Non-convex shapes (crescents, rings):
  
  Data has ring structure:       K-Means wrongly splits:
  ○○○○○○○                        ●●●○○○○
  ○●●●●○○                        ●●●●○○○
  ○●●×●○○   ← × = centroid       ●×●●○×○○
  ○●●●●○○                        ●●○○○○○
  ○○○○○○○                        (centroid in the hole!)
  
  Fix: DBSCAN or Spectral Clustering
  
  2. Very different cluster sizes:
  
  Data:                          K-Means splits large cluster:
  ●●●●●●●●●●    ●               ●●●●●│●●●●●    ●
  ●●●●●●●●●●    ● ●             ●●●●●│●●●●●    ●●
  (large)       (small)          incorrectly bisects large cluster!
  
  Fix: GMM (soft assignments adapt to cluster shape)
```

**Choosing the Number of Clusters K:**
```
Elbow Method (WCSS vs K):
  
  WCSS
  3000│●
  2000│  ●
  1000│    ●
   500│      ●──────────────   ← elbow at K=4
   300│               ● ● ●
      └──────────────────────── K
              3   4   5  6  7
  
  Choose K at the "elbow" — where additional clusters give diminishing WCSS reduction.
  Problem: often no clear elbow in real data.

Silhouette Score (measures cluster quality):
  For each point i:
    a(i) = mean distance to other points in SAME cluster (cohesion)
    b(i) = mean distance to points in NEAREST OTHER cluster (separation)
    
    s(i) = (b(i) − a(i)) / max(a(i), b(i))
  
  s(i) range: [−1, +1]
    s(i) ≈ +1: point is well-matched to its cluster, far from others (good)
    s(i) ≈ 0:  point is on the border between clusters
    s(i) ≈ −1: point is mis-clustered (closer to another cluster)
  
  Choose K that maximises average silhouette score.
  Rule of thumb: average s > 0.5 indicates reasonable clustering.

Gap Statistic (comparison to random data):
  Gap(K) = E*[log WCSS_random] − log WCSS_data
  
  Compare your clustering to K clusters in UNIFORM random data.
  If Gap(K) is large: your clustering is much better than random.
  Choose K where Gap(K) is maximised.
  
  This is the most statistically principled method.
```

**Spectral Clustering — For Non-Convex Shapes:**
```
Handles rings, crescents, and other non-convex clusters.
Key idea: use the GRAPH STRUCTURE of data instead of Euclidean distance.

Algorithm:
  1. Build similarity graph W:
     Wᵢⱼ = exp(−||xᵢ−xⱼ||²/(2σ²))   ← Gaussian kernel similarity
  
  2. Compute normalised graph Laplacian:
     D = diag(Σⱼ Wᵢⱼ)   (degree matrix)
     L_sym = I − D^{−1/2} W D^{−1/2}
  
  3. Find first k eigenvectors of L_sym → embed each point as a row
  
  4. Run K-Means on the embedded points
  
Why it works:
  Eigenvalues of L_sym near 0 correspond to CONNECTED COMPONENTS.
  Multiplicity of eigenvalue 0 = number of connected components.
  The embedding "unrolls" the graph structure, making non-convex clusters
  linearly separable in the embedding space.
  
  Data (two rings):          Spectral embedding:
  ○○○○○○○                    ●●●●●●  ○○○○○○○
  ○●●●●○○                         (now linearly separable!)
  ○●●●●○○
  ○○○○○○○
```

---

### 6.2 DBSCAN

> 💡 **DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds clusters of ARBITRARY shape by looking at density.** Instead of asking "which centroid is closest?", it asks "am I in a dense region?" Points in dense regions form clusters. Points in sparse regions are labelled as noise (outliers). No need to specify K in advance — the algorithm discovers the number of clusters!

**Core Concepts:**
```
Two parameters:
  ε (epsilon): neighbourhood radius — how far to look for neighbours
  minPts: minimum points required to form a dense region (typically 2×d, minimum 3)

Three types of points:

  Core Point: has ≥ minPts neighbours within distance ε
    ● ← core point     
   ●●●                  ε
  ●●●●    ← at least minPts=4 other points within radius ε
   ●●

  Border Point: NOT a core point, but within ε of a core point
    ●───→border point (ε away from a core point but has <minPts neighbours itself)
   ●● ●
  ●●
  ●

  Noise Point: not core, not border — completely isolated
    ●  ← noise (outlier) — far from everything
    
    (lots of empty space)

Algorithm:
  For each unvisited point p:
    If |N_ε(p)| < minPts: label p as NOISE (tentatively)
    Else: create new cluster C, expand:
      Add all ε-neighbours of p to C
      For each new core point in C: recursively add its ε-neighbours
  Border points get assigned to the cluster they're adjacent to.

Visualisation:
  
  ●●●●           ●●●     ← 3 clusters found automatically
  ●●●●    ●      ●●●
  ●●●●            ●●
     ↑ noise
  
  Dense blobs = clusters (any shape!)
  Sparse points = noise
  K determined automatically!
```

**Parameter Selection:**
```
How to choose ε:
  1. Compute k-distance for each point (distance to k-th nearest neighbour)
     (rule of thumb: k = minPts − 1)
  2. Sort k-distances in ascending order
  3. Plot the k-distance graph:
  
  k-dist
  3.0 │         ╭──────── ← set ε here (at the "elbow")
  2.0 │        ╱
  1.0 │──────╱
  0.5 │────╱
      └─────────────────── points sorted by k-distance
      
  The elbow represents the transition from "dense core" points to "sparse" points.
  Setting ε at the elbow captures dense clusters while ignoring noise.

How to choose minPts:
  Rule of thumb: minPts = 2 × d (where d = number of dimensions)
  Minimum: minPts = 3
  
  Justification: in d-dimensional space, the intrinsic dimensionality of most
  datasets is lower than d. Setting minPts ∝ d ensures robustness to noise
  proportional to the number of free dimensions.
```

**Why DBSCAN is Robust:**
```
No assumptions about cluster shape:
  K-Means: must be roughly spherical
  DBSCAN: any shape defined by connectivity between dense regions
  
Handles noise naturally:
  K-Means: all points get assigned to a cluster (even outliers)
  DBSCAN: outliers labelled as NOISE, not forced into any cluster
  
No K to specify:
  K-Means: must set K beforehand
  DBSCAN: discovers K from data density
  
Weakness: doesn't handle varying-density clusters well.
  If cluster A is dense and cluster B is sparse, the same ε either:
    • Correctly clusters A but misses B (ε too small for B)
    • Correctly clusters B but merges A into one giant cluster
  
  Solution: HDBSCAN (hierarchical version) handles varying densities better.
```

---

### 6.3 Hierarchical Clustering

> 💡 **Hierarchical clustering gives you a TREE of possible clusterings.** Unlike K-Means which gives one answer for a fixed K, hierarchical clustering builds a dendrogram (tree diagram) that shows you how clusters merge or split at every level. You then "cut" the tree at any height to get the number of clusters you want — exploring multiple K values from a single computation.

**Agglomerative (Bottom-Up) Clustering:**
```
Algorithm:
  Start: n clusters (each point is its own cluster)
  
  Repeat until 1 cluster remains:
    Find the two MOST SIMILAR clusters
    Merge them into one new cluster
  
  Result: a dendrogram recording all merges

Visualisation (4 points → dendrogram):

  Points:                     Dendrogram:
  
  A  B    C  D                Height
                               3 │         ╔═══════╗
  A●  ●B   C●  ●D             2 │    ╔══╗ ╟───────╢
                               1 │ ╔╗ ╟──╢        ╟
  A, B are close                 │ AB CD             
  C, D are close                 A  B  C  D
  (AB) and (CD) are farther
  
  Read: A and B merge first (height=1), C and D merge (height=1.5),
  then (AB) and (CD) merge at height=3.

Cutting the dendrogram:
  Cut at height=2.5: 2 clusters {A,B} and {C,D}
  Cut at height=1.2: 3 clusters {A,B}, {C}, {D}
  Cut at height=0.5: 4 clusters {A},{B},{C},{D} (no merging)
```

**Linkage Criteria — Defining "Distance Between Clusters":**
```
When two clusters A and B have multiple points each,
how do we define d(A, B)?

Single Linkage (Minimum):
  d(A,B) = min_{a∈A, b∈B} d(a,b)   ← nearest pair
  
  Effect: "chaining" — elongated, snake-like clusters
  
  ●─●─●─●─●─●─●  ← all connected via single links
  (one long chain!)

Complete Linkage (Maximum):
  d(A,B) = max_{a∈A, b∈B} d(a,b)   ← farthest pair
  
  Effect: compact, roughly equal-sized clusters
  Sensitive to outliers (one outlier far away inflates distance)

Average Linkage (UPGMA):
  d(A,B) = mean_{a∈A, b∈B} d(a,b)  ← average of all pairwise distances
  
  Effect: compromise between single and complete
  Often the best general-purpose choice

Ward's Linkage (MINIMISES VARIANCE INCREASE):
  Δ(A,B) = (|A||B|/(|A|+|B|)) × ||μ_A − μ_B||²
  
  "How much does WCSS increase when we merge A and B?"
  Merges clusters that cause the SMALLEST increase in total variance.
  
  Effect: compact, spherical clusters of similar size (most like K-Means)
  Best all-around choice for most applications
  
  Theoretical connection: Ward's method = K-means with K decremented by 1 each step!

Visual comparison:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Single: chains             Complete: compact    Ward: equal spheres    │
  │                                                                         │
  │ ●─●─●─●─●─●             ●●●  ○○○              ●●●  ○○○              │
  │            ●─●─○ ○      ●●●  ○○○              ●●●  ○○○              │
  │                ○─○─○   ●●   ○○                ●●   ○○               │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

### 6.4 Gaussian Mixture Models (GMM)

> 💡 **GMM is K-Means with probability.** Instead of hard assignments ("point X is in cluster 3"), GMM gives soft assignments ("point X is 70% likely in cluster 3, 25% in cluster 1, 5% in cluster 2"). This captures uncertainty and handles overlapping clusters gracefully. GMM also allows clusters of different shapes and orientations (not just spheres).

**The Model:**
```
Data is modelled as a mixture of K Gaussian distributions:
  P(x) = Σₖ πₖ N(x | μₖ, Σₖ)
  
  πₖ = mixing weight (probability of belonging to cluster k), Σₖ πₖ = 1
  μₖ = mean of cluster k
  Σₖ = covariance matrix of cluster k

This model can represent:
  • Spherical clusters (Σₖ = σₖ²I) ← like soft K-Means
  • Elongated clusters (Σₖ diagonal with different entries)
  • Tilted/correlated clusters (full Σₖ)

Visualisation:
  
  ┌──────────────────────────────────────────────────────┐
  │  Contours of GMM density:                           │
  │                                                      │
  │          ╭──╮                                       │
  │      ╭──╯    ╰──╮                                  │
  │     │  cluster 1 │  ╭────╮                         │
  │      ╰──╮    ╭──╯  │cluster│                       │
  │          ╰──╯       │  2   │                       │
  │                      ╰────╯                         │
  │                                    ╭──────╮         │
  │                                   │cluster│         │
  │                                   │   3   │         │
  │                                    ╰──────╯         │
  └──────────────────────────────────────────────────────┘
  Each ellipse = one Gaussian component (can have different shape/orientation!)
```

**The EM Algorithm for GMM:**
```
We can't directly maximise likelihood (sum inside log is intractable).
EM (Expectation-Maximisation) iterates between:

E-Step (Expectation): Compute SOFT ASSIGNMENTS (responsibilities):
  γᵢₖ = P(component k | xᵢ) = πₖN(xᵢ|μₖ,Σₖ) / Σⱼ πⱼN(xᵢ|μⱼ,Σⱼ)
  
  γᵢₖ ∈ (0,1): "how responsible is component k for generating point i?"
  Σₖ γᵢₖ = 1   (responsibilities sum to 1 for each point)
  
  Example for 3 clusters:
  Point x₁: γ₁₁=0.7, γ₁₂=0.2, γ₁₃=0.1  (mostly in cluster 1)
  Point x₂: γ₂₁=0.05, γ₂₂=0.9, γ₂₃=0.05 (clearly in cluster 2)

M-Step (Maximisation): Update parameters using weighted MLE:
  Nₖ = Σᵢ γᵢₖ                             (effective number in cluster k)
  μₖ = (1/Nₖ) Σᵢ γᵢₖ xᵢ                   (weighted mean)
  Σₖ = (1/Nₖ) Σᵢ γᵢₖ (xᵢ−μₖ)(xᵢ−μₖ)ᵀ    (weighted covariance)
  πₖ = Nₖ/n                                 (updated mixing weights)

Repeat E and M steps until log-likelihood converges.

Why EM is Guaranteed to Converge:
  EM maximises a LOWER BOUND (ELBO) on the log-likelihood via Jensen's inequality.
  Each E-step makes the bound tight (best bound at current parameters).
  Each M-step increases the bound (and thus the log-likelihood).
  Log-likelihood is bounded above (by 0 for proper probabilities).
  → Must converge.
  
  (But to a LOCAL maximum — try multiple random initialisations!)
```

**GMM vs K-Means:**
```
┌─────────────────────────────────────────────────────────────────────┐
│              K-Means                GMM                             │
├─────────────────────────────────────────────────────────────────────┤
│ Assignments    Hard (0 or 1)         Soft (probabilities)          │
│ Cluster shape  Spherical only        Any Gaussian shape            │
│ Cluster size   Implicitly equal      Different (via πₖ)            │
│ Missing data   No natural handling   Posterior gives uncertainty   │
│ Complexity     O(nKd) per iter       O(nKd²) per iter              │
│ Metric         WCSS (distances)      Log-likelihood (probabilistic) │
│ K selection    Elbow, silhouette     BIC: −2ℓ + k log n            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Dimensionality Reduction

> 💡 **The curse of dimensionality strikes again.** Real datasets have hundreds or thousands of features. Many are redundant (correlated with others). Many are noisy. Dimensionality reduction finds a compact, lower-dimensional representation that preserves the essential structure of the data — useful for visualisation, denoising, compression, and preprocessing before other algorithms.

---

### 7.1 Principal Component Analysis (PCA)

> 💡 **PCA finds the "most informative" directions in your data.** If you had to describe your dataset using only a few numbers per point, which directions would capture the most information? PCA says: the direction with the HIGHEST VARIANCE is most informative, then the direction with the next highest variance (perpendicular to the first), and so on. These are the principal components.

**Intuitive Setup:**
```
Imagine 2D data shaped like a tilted ellipse:

  x₂
   │        ● ● ●
   │      ●  ●  ●
   │    ●  ●   ●        ← data is spread mainly along this diagonal direction
   │  ●  ●  ●
   │●  ●  ●
   └─────────────── x₁

  There's a "natural axis" of the data (the long axis of the ellipse).
  Points don't vary much PERPENDICULAR to this axis.
  → Project onto the long axis → capture most information in 1D!

PC1: the direction of MAXIMUM VARIANCE (the long axis)
PC2: the direction of second most variance (perpendicular to PC1)
PC3: ... and so on
```

**PCA Algorithm (Step by Step):**
```
Given: Data matrix X (n×d), n points in d dimensions.

Step 1: CENTRE the data (subtract mean from each feature):
  X̄ = X − 1ₙ μᵀ    where μⱼ = (1/n)Σᵢ Xᵢⱼ
  
  Why? PCA finds directions of variance. If data isn't centred,
  the mean dominates the first PC (just points toward the mean).

Step 2: Compute COVARIANCE MATRIX:
  C = (1/n) X̄ᵀX̄    shape: (d×d)
  
  Cᵢⱼ = covariance between feature i and feature j

Step 3: EIGENDECOMPOSE the covariance matrix:
  C = Q Λ Qᵀ
  Q = columns are eigenvectors (the principal components!)
  Λ = diagonal with eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λd
  
  λₖ = variance explained by principal component k

Step 4: SORT eigenvectors by decreasing eigenvalue.
  PC1 = eigenvector with largest λ (most variance)
  PC2 = eigenvector with second largest λ

Step 5: PROJECT data onto first k components:
  Z = X̄ Qₖ    shape: (n×k)
  Z = the "compressed" representation using k dimensions

Step 6: RECONSTRUCT (optional, for compression):
  X̂ = Z Qₖᵀ + μ = X̄ Qₖ Qₖᵀ + μ
  Reconstruction error = ||X − X̂||²_F = Σⱼ>ₖ λⱼ
```

**Two Equivalent Characterisations of PCA:**
```
Characterisation 1 — MAXIMUM VARIANCE:
  PC1 = v₁ = argmax_{||v||=1} vᵀ C v
  
  The first principal component is the direction that MAXIMISES the variance
  of the projected data.
  
  Proof: variance of projection = vᵀ(X̄ᵀX̄/n)v = vᵀCv
  Maximise subject to ||v||=1 → solution is eigenvector of C with largest λ.

Characterisation 2 — MINIMUM RECONSTRUCTION ERROR:
  The first k PCs minimise: ||X̄ − X̄VₖVₖᵀ||²_F = Σⱼ>ₖ λⱼ
  
  The Eckart-Young theorem: the truncated SVD gives the BEST rank-k approximation
  to a matrix in Frobenius norm.

Both give the SAME solution! PCA is simultaneously:
  • The projection that maximises variance
  • The compression that minimises reconstruction error
  
  This beautiful duality explains why PCA is so fundamental.
```

**How Much Variance Is Retained?**
```
Proportion of Variance Explained (PVE):
  PVE(k) = Σⱼ₌₁ᵏ λⱼ / Σⱼ₌₁ᵈ λⱼ
  
  = fraction of total variance captured by first k PCs

Scree plot (helps choose k):

  PVE
  0.50│●
  0.25│  ●
  0.15│    ●──────────────────────   ← flattens out here
  0.05│            ● ● ● ● ● ● ●
      └──────────────────────────── PC number
         1  2  3  4  5  6  7  8  9
         
  Choose k at the "elbow" — where adding more PCs gives diminishing returns.
  
  Cumulative PVE:
  1.00│                    ──────
  0.90│              ─────
  0.75│          ────
  0.50│      ────
  0.25│  ────
      └──────────────────── k
  
  Common targets: retain 90% or 95% of variance.
```

**Whitening Transform:**
```
After PCA projection Z, apply whitening:
  Z_white = Z × Λₖ^{−1/2}
  
  Each PC is divided by its standard deviation → all PCs have unit variance.
  PCs are DECORRELATED AND STANDARDISED.
  
  The covariance of Z_white = I (identity matrix).
  
  Use case: some algorithms (ICA, neural network training) work better
  when inputs are decorrelated and have unit variance. Whitening is the
  "perfect" preprocessing that achieves this.
```

**When PCA Fails:**
```
1. Non-linear structure:
   
   Data lies on a CURVED manifold:
   
   ● ● ●            If you project this
     ●   ●           onto a line, you lose
       ●   ●         all the structure!
         ● ● ●
   (data on a "Swiss roll")
   
   Fix: Kernel PCA, UMAP, t-SNE (non-linear methods)

2. Variance ≠ Importance:
   
   Feature A: 0.001 ± 0.0001 (low variance, but perfectly predicts y)
   Feature B: 100 ± 50 (high variance, random noise)
   
   PCA keeps B (high variance), drops A (low variance) → disaster!
   
   Fix: Use supervised PCA or LDA instead for classification tasks.

3. Non-Gaussian distributions:
   
   PCA finds linear structure. Data with multi-modal or skewed distributions
   may have important structure ORTHOGONAL to the direction of max variance.
   
   Fix: ICA (Independent Component Analysis) for non-Gaussian sources.
```

---

### 7.2 t-SNE (t-Distributed Stochastic Neighbour Embedding)

> 💡 **t-SNE is for VISUALISATION, not compression.** Unlike PCA which gives you a mathematically principled linear projection, t-SNE creates a 2D (or 3D) scatter plot that preserves the NEIGHBOURHOOD structure of high-dimensional data. Points that are close in high dimensions end up close in 2D, and clusters in high-D appear as clusters in the visualisation. The "t" and "stochastic" are the tricks that make it work.

**The Core Idea:**
```
Step 1: In HIGH-d space, convert distances to PROBABILITIES:
  "What's the probability that xᵢ would choose xⱼ as its neighbour?"
  
  pⱼ|ᵢ = exp(−||xᵢ−xⱼ||²/(2σᵢ²)) / Σₖ≠ᵢ exp(−||xᵢ−xₖ||²/(2σᵢ²))
  
  σᵢ: bandwidth for point i (set to match the specified perplexity)
  Symmetrise: pᵢⱼ = (pⱼ|ᵢ + pᵢ|ⱼ) / (2n)
  
  Close points: high p (likely neighbours)
  Distant points: low p (unlikely neighbours)

Step 2: In LOW-d space (2D), also compute probabilities:
  qᵢⱼ = (1 + ||yᵢ−yⱼ||²)⁻¹ / Σₖ≠ₗ (1 + ||yₖ−yₗ||²)⁻¹
  
  Using STUDENT-t distribution (1 degree of freedom) instead of Gaussian!

Step 3: MINIMISE KL divergence between P and Q:
  C = KL(P || Q) = Σᵢⱼ pᵢⱼ log(pᵢⱼ/qᵢⱼ)
  
  Gradient:
  ∂C/∂yᵢ = 4Σⱼ (pᵢⱼ − qᵢⱼ)(yᵢ−yⱼ)(1 + ||yᵢ−yⱼ||²)⁻¹
  
  Interpretation:
  pᵢⱼ > qᵢⱼ: neighbours in high-d but not in 2D → ATTRACTION (pull together)
  pᵢⱼ < qᵢⱼ: neighbours in 2D but not in high-d → REPULSION (push apart)
```

**Why Student-t Solves the "Crowding Problem":**
```
The crowding problem:
  In d=100 dimensions, moderate distances span a HUGE range.
  Trying to map these to 2D: you can't fit them all!
  
  With a Gaussian in 2D:
    All moderate-distance pairs get CRUSHED together (no room for them all)
    → clusters merge, structure lost
  
  With a Student-t (heavier tails) in 2D:
    (1 + ||yᵢ−yⱼ||²)⁻¹ decays as 1/r² instead of exponentially fast
    
    Student-t tail:     Gaussian tail:
    
    P │╲                P │╲
      │  ╲                │  ╲
      │    ──────────      │    ╲────────────
      │                    │    (too fast!)
      └─────── distance    └─────── distance
      
    The "heavy tail" stretches out moderate distances → more room for structure.
    CLOSE points still map to close, but MEDIUM-distance points get more spread out.

Intuition: t-SNE "blows up" the low-dimensional space at moderate distances,
creating room for cluster structure that Gaussian kernels would crush.
```

**Practical Notes for Using t-SNE:**
```
Perplexity (key hyperparameter):
  ≈ effective number of neighbours each point considers
  
  Typical range: 5–50. Default: 30.
  
  Perplexity=5:           Perplexity=30:          Perplexity=100:
  Very local structure     Balanced                Global structure
  Many tiny clusters       (usually best)          Large, merged clusters
  
  Too low: noisy, fragmented    Too high: clusters merge

What t-SNE preserves:
  ✓ Local neighbourhood structure (nearby points remain nearby)
  ✓ Cluster membership (points in same cluster end up together)
  
What t-SNE does NOT preserve:
  ✗ Global distances (cluster A being far from cluster B in 2D means NOTHING)
  ✗ Cluster sizes (a big cluster in 2D doesn't mean a big cluster in original space)
  ✗ Cluster orientations or shapes

Critical warnings:
  ✗ NOT deterministic: run again → different plot
  ✗ NOT a linear projection: can't apply the "transform" to new points easily
  ✗ Distances between clusters are MEANINGLESS
  ✓ Use for VISUALISATION ONLY, not for downstream ML features
```

---

### 7.3 UMAP (Uniform Manifold Approximation and Projection)

> 💡 **UMAP does what t-SNE does but better and faster.** Based on Riemannian geometry and algebraic topology (fancy mathematics, but the intuition is simple), UMAP preserves both local neighbourhood structure AND some global structure — something t-SNE fails at. It's also much faster (O(n log n) vs t-SNE's O(n²)), can be applied to new points (unlike t-SNE), and can work in supervised mode.

**The Core Idea (Simplified):**
```
Step 1: Build a "fuzzy topological representation" of the data.
  For each point xᵢ, find its nearest neighbours.
  Define membership strength (probability of connection):
  
  wᵢⱼ = exp(−(d(xᵢ,xⱼ) − ρᵢ)/σᵢ)
  
  ρᵢ = distance to nearest neighbour (local normalisation!)
  
  The LOCAL NORMALISATION (ρᵢ) is UMAP's key innovation:
    Each point's neighbourhood is measured relative to its nearest neighbour.
    Dense regions: σᵢ is small → tight connections only
    Sparse regions: σᵢ is large → wider connections
    → Adapts to LOCAL density → preserves both local AND global structure!

Step 2: Optimise a 2D layout to match this fuzzy structure:
  Binary cross-entropy loss:
  L = Σᵢⱼ [wᵢⱼ log(wᵢⱼ/ŵᵢⱼ) + (1−wᵢⱼ)log((1−wᵢⱼ)/(1−ŵᵢⱼ))]
  
  ŵᵢⱼ = (1 + a||yᵢ−yⱼ||^{2b})⁻¹  (low-d membership)
  
  Optimised via SGD with negative sampling (much faster than t-SNE's gradient!).
```

**UMAP vs t-SNE:**
```
┌─────────────────────────────────────────────────────────────────────┐
│ Property              t-SNE              UMAP                      │
├─────────────────────────────────────────────────────────────────────┤
│ Speed                 O(n²) — SLOW       O(n log n) — FAST         │
│ Global structure      Poor               Better preserved           │
│ Local structure       Excellent          Excellent                  │
│ Determinism           Non-deterministic  More deterministic         │
│ New points           Can't transform    CAN transform (transform)  │
│ Supervised mode       No                 YES (can use labels)       │
│ Typical runtime       10 min/100k pts    1 min/100k pts            │
│ Meaningful distances  No (both)          Slightly more meaningful  │
└─────────────────────────────────────────────────────────────────────┘

Supervised UMAP:
  Can incorporate label information to create better separations!
  Points from same class are pulled together during optimisation.
  Semi-supervised: use labels for some points, unsupervised for rest.
```

---

## 8. Ensemble Learning

> 💡 **"Two heads are better than one" — but only if they disagree.** Ensemble methods combine multiple models to create predictions better than any individual model. The key insight: if you combine models that make DIFFERENT kinds of errors, their errors cancel out. A model that's sometimes wrong but in random ways, averaged with another model that's wrong in different random ways, produces much better predictions.

---

### 8.1 Why Ensembles Work — The Theory

**Variance Reduction Formula:**
```
If you average B models, each with:
  • Variance σ² per model
  • Pairwise correlation ρ between models

Then variance of the average:
  Var(f̄) = ρσ² + (1−ρ)σ²/B
            ↑        ↑
        irreducible   reducible term
        floor         (goes to 0 as B→∞)

As B → ∞:    Var(f̄) → ρσ²

Two key insights:
  1. If ρ=1 (models are identical): no benefit from averaging!
     Var(f̄) = σ² (same as a single model)
     
  2. If ρ=0 (models are independent): massive benefit!
     Var(f̄) = σ²/B → 0 as B→∞
     
  3. ρ is MORE IMPORTANT than B:
     Reducing correlation between models beats adding more models.
     → This is why Random Forest subsamples features!
```

**The Ambiguity Decomposition:**
```
(error of ensemble)² = (avg individual error)² − (avg disagreement)²

                           ↑                        ↑
                    want to minimise          want to maximise

For ensemble to be better than average individual:
  Accurate models (small individual error) + Diverse models (high disagreement) = win!

Example:
  3 models each with 20% error rate.
  If errors are IDENTICAL: ensemble error = 20% (no improvement)
  If errors are INDEPENDENT: ensemble error = 1−P(at least 2 wrong)
    = 1 − (3×0.2²×0.8 + 0.2³) = 1 − (0.096 + 0.008) = 89.6% accuracy!
  
  Independent errors → ensemble is much better!
```

**Condorcet's Jury Theorem:**
```
If each of B independent classifiers has accuracy p > 0.5:

  P(majority vote correct) → 1 as B → ∞

Proof sketch:
  Majority vote = at least ⌈B/2⌉ classifiers correct.
  By Law of Large Numbers, fraction correct → p > 0.5 as B → ∞.
  
  For finite B: P(majority correct) = Σₖ₌⌈B/2⌉^B C(B,k) pᵏ(1−p)^{B−k}
  
  Example: p=0.7, B=11 classifiers:
  P(majority correct) = 0.910  → 91% accuracy from 70% classifiers!
  
  This is the mathematical foundation of why voting ensembles work.
  
  CRUCIAL ASSUMPTION: classifiers must be INDEPENDENT (or at least diverse).
  Correlated classifiers don't get this benefit.
```

---

### 8.2 Bagging (Bootstrap Aggregating)

> 💡 **Bagging creates diversity by training each model on a slightly different dataset.** It repeatedly samples with replacement from the training data (bootstrap), trains a model on each sample, then averages predictions. The key: models trained on different bootstrap samples will be trained on different "versions" of the data, making them diverse enough to benefit from combination.

**Bootstrap Sampling:**
```
Original dataset: {x₁, x₂, x₃, x₄, x₅} (n=5 points)

Bootstrap sample (draw n WITH REPLACEMENT):
  Sample 1: {x₁, x₁, x₃, x₅, x₂}  ← x₁ appears twice, x₄ missing
  Sample 2: {x₂, x₃, x₃, x₁, x₅}  ← x₃ appears twice, x₄ missing
  Sample 3: {x₄, x₂, x₅, x₃, x₂}  ← x₂ appears twice, x₁ missing
  
  Each bootstrap sample ≈ 63.2% UNIQUE points.
  P(point xᵢ NOT in a sample) = (1−1/n)ⁿ → e⁻¹ ≈ 0.368 as n→∞
  
  The remaining ~36.8% = "out-of-bag" (OOB) examples for that model.

Bagging prediction:
  Regression:      f̄(x) = (1/B) Σb fb(x)           ← average
  Classification:  ŷ = majority vote of {fb(x)}     ← mode
```

**Out-of-Bag (OOB) Error — Free Internal Validation:**
```
Each training point appears in about 63.2% of bootstrap samples.
It's EXCLUDED from the remaining 36.8%.

For prediction on xᵢ: use only models trained WITHOUT xᵢ:
  ŷᵢ^OOB = average of {fb(xᵢ) : xᵢ ∉ bootstrap sample b}

OOB error = (1/n) Σᵢ L(yᵢ, ŷᵢ^OOB)

Properties:
  ✓ Uses ALL training data for training (no validation split needed)
  ✓ Each point evaluated on models it hasn't been trained on → unbiased
  ✓ Asymptotically equivalent to leave-one-out cross-validation
  ✓ Zero extra computation cost!
  
  Essentially gives you a free cross-validated estimate of generalisation error.
```

---

### 8.3 Random Forest

> 💡 **Random Forest = Bagging + Random Feature Subsampling.** Bagging creates diversity by using different data samples. Random Forest adds another source of diversity: at each split, only a RANDOM SUBSET of features is considered. This further decorrelates the trees, dramatically improving the ensemble's variance reduction.

**The Extra Randomisation — Why It Helps:**
```
In a standard decision tree, if feature x₃ is very predictive,
EVERY tree will split on x₃ first → trees are highly correlated!

Random Forest fix: at each split, only consider m randomly chosen features
  Classification: m = √p   (square root of total features)
  Regression:     m = p/3  (third of total features)

Effect on correlation between trees:
  
  Standard bagged trees:     Random forest:
  All consider x₃ first      Some trees can't "see" x₃
  ρ ≈ 0.7 (high)             ρ ≈ 0.2 (low)
  
  Lower ρ → lower ensemble variance → better generalisation!
  
  Recall: Var(ensemble) = ρσ² + (1−ρ)σ²/B
  ρ=0.7 vs ρ=0.2 with σ²=1, B=100:
    Bagging:  0.7×1 + 0.3×1/100 = 0.703
    RF:       0.2×1 + 0.8×1/100 = 0.208
    → Random Forest: 3× lower variance!
```

**Feature Importance in Random Forests:**
```
Method 1: Mean Decrease Impurity (MDI):
  FI(j) = (1/B) Σb Σ_{t: node t splits on feature j} p_t × ΔImpurity(t)
  
  p_t = fraction of training samples reaching node t
  ΔImpurity(t) = impurity reduction from the split
  
  ✓ Fast (computed during training)
  ✗ Biased toward high-cardinality features (more split options → higher chance selected)

Method 2: Permutation Importance (More Reliable):
  1. Train forest, record OOB accuracy A₀
  2. For each feature j:
     a. Randomly SHUFFLE feature j in OOB samples (break relationship with y)
     b. Re-predict → new accuracy Aⱼ
     c. PI(j) = A₀ − Aⱼ  (how much accuracy dropped from breaking feature j)
  
  ✓ Works with any metric
  ✓ Not biased by cardinality
  ✓ Captures all types of relationships (nonlinear, interactions)
  ✗ Slightly more expensive to compute

Method 3: SHAP Values (see Section 3.4 — most theoretically rigorous):
  Applies Shapley game theory values to individual predictions.
  More computation, but guaranteed to satisfy consistency axioms.
```

---

### 8.4 AdaBoost (Adaptive Boosting)

> 💡 **Boosting is sequential learning — each new model focuses on the mistakes of the previous ones.** Where bagging trains models independently and hopes their random differences make them diverse, boosting DELIBERATELY trains each new model to be better on the examples the current ensemble gets wrong. The result: a "committee" that collectively handles hard cases by having specialist models for each type of difficulty.

**The Algorithm:**
```
1. Initialise equal weights: wᵢ = 1/n for all training points

2. For t = 1, 2, ..., T:
   a. Train weak learner hₜ on WEIGHTED training data (points with higher weight get more attention)
   
   b. Compute weighted error:
      εₜ = Σᵢ wᵢ × 𝟙[hₜ(xᵢ) ≠ yᵢ]
           ↑ fraction of total weight on misclassified points
   
   c. Compute learner weight:
      αₜ = (1/2) ln[(1−εₜ)/εₜ]
      
      εₜ = 0.5 (random): αₜ = 0 (learner is useless, ignore it)
      εₜ = 0.1 (good):   αₜ = 1.1 (learner has high weight)
      εₜ → 0 (perfect):  αₜ → ∞ (this learner dominates)
   
   d. Update weights (emphasise mistakes):
      wᵢ ← wᵢ × exp(−αₜ × yᵢ × hₜ(xᵢ))
      
      If hₜ got xᵢ RIGHT (yᵢ = hₜ(xᵢ)): exp(−αₜ) < 1 → weight DECREASES
      If hₜ got xᵢ WRONG (yᵢ ≠ hₜ(xᵢ)): exp(+αₜ) > 1 → weight INCREASES
      
      Normalise so Σwᵢ = 1.

3. Final prediction: H(x) = sign(Σₜ αₜ hₜ(x))
```

**Weight Evolution Visualisation:**
```
Iteration 1:         Iteration 2:           Iteration 3:
All equal weight     Mistakes get bigger     New mistakes get bigger

●  ●  ●  ●  ●       ●  ●  ◎  ●  ◎         ●  ◎  ◉  ◉  ◎
                         ↑ misclassified          ↑ still hard
                         weights increase         weights increase more

Legend: ● = small weight, ◎ = medium, ◉ = large

Each model specialises in different "hard" regions.
The final ensemble handles all regions well.
```

**Training Error Bound — Why Boosting Works:**
```
With T weak learners each with edge γₜ = 0.5 − εₜ > 0 (better than random):

  Training Error ≤ Πₜ √(1 − 4γₜ²)  ≤  exp(−2 Σₜ γₜ²)

If γₜ ≥ γ > 0 for all t:
  Training Error ≤ exp(−2γ²T) → 0 as T → ∞

Remarkable result: even if each weak learner is only SLIGHTLY better than random
(γ=0.01), boosting drives training error to 0 exponentially fast!

AdaBoost minimises EXPONENTIAL LOSS:
  L_exp(y, f(x)) = exp(−y f(x))
  
  y = +1 and f(x) large positive: exp(−large) → 0 (correct, low loss)
  y = +1 and f(x) large negative: exp(+large) → ∞ (wrong, huge loss!)
  
  Exponential loss HEAVILY penalises confident wrong predictions.
  This is why AdaBoost is sensitive to outliers and label noise.
```

---

### 8.5 Gradient Boosting

> 💡 **Gradient Boosting is the most powerful and widely-used ensemble method.** Like AdaBoost, it builds models sequentially. But instead of reweighting misclassified points, it fits each new tree to the RESIDUALS (errors) of the current ensemble. The name "gradient" comes from the insight that residuals are actually the negative gradient of the loss function — so this is gradient descent in function space.

**The Core Idea — Gradient Descent in Function Space:**
```
Standard gradient descent (parameter space):
  θₜ₊₁ = θₜ − α ∇_θ L(θ)
  (move parameters in direction that reduces loss)

Gradient Boosting (function space):
  Fₘ(x) = Fₘ₋₁(x) + η hₘ(x)
  (add a new function that reduces loss)

"Pseudo-residuals" = negative gradient of loss w.r.t. current predictions:
  rᵢₘ = − ∂L(yᵢ, F(xᵢ))/∂F(xᵢ)|_{F=Fₘ₋₁}

We fit a regression tree hₘ to these pseudo-residuals.
The tree is the "best available approximation" to the gradient direction.
Adding hₘ is taking a step in function space toward the loss minimum.
```

**Pseudo-Residuals for Different Losses:**
```
MSE Loss L = (1/2)(y−F)²:
  rᵢ = −∂L/∂F = yᵢ − F(xᵢ)   ← just the ordinary residuals!
  
  This is why for regression with MSE, GB simply fits trees to residuals
  (exactly like what intuition suggests).

MAE Loss L = |y−F|:
  rᵢ = sign(yᵢ − F(xᵢ))
  Pseudo-residuals are just ±1 (direction, not magnitude).
  → More robust to outliers than MSE-based boosting.

Log-Loss L = −[y log σ(F) + (1−y)log(1−σ(F))]:
  rᵢ = yᵢ − σ(F(xᵢ))   ← residual in probability space
  = actual label − predicted probability
  
  This is why gradient boosted trees for classification fit residuals that look
  like probability errors.

Huber Loss (robust regression):
  rᵢ = δ sign(yᵢ−F) if |yᵢ−F| > δ
  rᵢ = yᵢ − F        if |yᵢ−F| ≤ δ
  
  MSE-like for small errors (smooth gradient), MAE-like for large errors (bounded gradient).
  Less sensitive to outliers than MSE, smoother than MAE.
```

**The Full Gradient Boosting Algorithm:**
```
1. Initialise with constant prediction:
   F₀(x) = argmin_γ Σᵢ L(yᵢ, γ)   (e.g., mean for MSE)

2. For m = 1 to M (number of trees):
   
   a. Compute pseudo-residuals:
      rᵢₘ = −[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]_{F=Fₘ₋₁}
   
   b. Fit regression tree hₘ to {(xᵢ, rᵢₘ)}
      (tree trained to predict the pseudo-residuals)
   
   c. Find optimal leaf weights:
      γⱼₘ = argmin_γ Σᵢ∈Rⱼₘ L(yᵢ, Fₘ₋₁(xᵢ) + γ)
      (for each leaf j, find the best step size)
   
   d. Update:
      Fₘ(x) = Fₘ₋₁(x) + η × hₘ(x)
      η = learning rate (shrinkage), typically 0.01–0.1

3. Final prediction: F_M(x)

Hyperparameter Notes:
  Lower η → need more trees M → more regularisation
  Typical: η=0.01–0.1, M=100–1000 trees
  Early stopping: monitor validation loss, stop when it stops decreasing
```

---

### 8.6 XGBoost (Extreme Gradient Boosting)

> 💡 **XGBoost is gradient boosting done right, at scale.** It adds second-order Taylor expansion (like Newton's method vs gradient descent), explicit regularisation of tree structure, missing value handling, and dozens of engineering optimisations for speed and memory. It dominated Kaggle competitions for years and remains a top algorithm for tabular data.

**Second-Order Taylor Expansion — The Key Insight:**
```
Standard gradient boosting: fits trees to first-order gradient (pseudo-residuals).
  Like gradient descent: uses slope only, ignores curvature.

XGBoost: uses second-order Taylor expansion of the loss:
  L(yᵢ, Fₘ₋₁(xᵢ) + f(xᵢ)) ≈ L(yᵢ, Fₘ₋₁) + gᵢf(xᵢ) + (1/2)hᵢf(xᵢ)²
  
  gᵢ = ∂L/∂F|_{Fₘ₋₁}    (first derivative = gradient)
  hᵢ = ∂²L/∂F²|_{Fₘ₋₁}  (second derivative = Hessian)

Adding regularisation:
  Obj = Σᵢ [gᵢf(xᵢ) + (1/2)hᵢf(xᵢ)²] + Ω(f)
  Ω(f) = γT + (λ/2)Σⱼ wⱼ²
         ↑        ↑
       #leaves penalty   leaf weight penalty
  
  T = number of leaves
  γ = minimum gain to split (tree complexity penalty)
  λ = L2 regularisation on leaf weights

Optimal leaf weights (closed form!):
  wⱼ* = −Gⱼ/(Hⱼ + λ)
  
  Gⱼ = Σᵢ∈Iⱼ gᵢ   (sum of gradients in leaf j)
  Hⱼ = Σᵢ∈Iⱼ hᵢ   (sum of Hessians in leaf j)

This is the NEWTON STEP for leaf j: accounting for curvature
makes the step more accurate → fewer iterations to converge.

Optimal split gain formula:
  Gain = (1/2)[Gₗ²/(Hₗ+λ) + Gᵣ²/(Hᵣ+λ) − (Gₗ+Gᵣ)²/(Hₗ+Hᵣ+λ)] − γ
          ↑                                                               ↑
       reduction in objective from the split                  complexity cost
  
  Only split if Gain > 0 (otherwise tree is pruned via the γ term).
```

---

### 8.7 LightGBM

> 💡 **LightGBM is XGBoost that's been engineered for speed.** On large datasets (millions of rows), XGBoost can be slow because it sorts data and scans all feature values. LightGBM uses histograms (grouping values into bins) and smart sampling to achieve 10-100× speedups with minimal accuracy loss.

**Histogram-Based Splitting:**
```
XGBoost exact: considers all n unique values as potential split points
  For n=10M, feature x₃: 10M potential splits to evaluate!

LightGBM histogram: groups values into B=256 bins
  x₃ range [0, 1000] → 256 bins of width ~4 each
  
  For each bin, store: Gⱼ = ΣᵢG_i, Hⱼ = ΣᵢH_i (gradient sums)
  
  Finding best split:
  Scan 256 bins instead of 10M values → 40,000× speedup!
  
  Memory savings:
  Store bin index (uint8 = 1 byte) instead of float64 (8 bytes) → 8× compression
  
  Quality loss: minimal — split points are rounded to bin boundaries.
  In practice, no measurable accuracy difference from exact split.
```

**GOSS (Gradient-based One-Side Sampling):**
```
Key insight: points with small gradients are already well-learned.
  Large gradient → large error → still learning from this point
  Small gradient → small error → nearly "done" with this point
  
  Keep ALL large-gradient points (informative for learning).
  SAMPLE small-gradient points (mostly redundant).
  
GOSS algorithm:
  1. Sort training instances by |gradient|
  2. Keep top-a% (large |gradient|) → always included
  3. Randomly sample b% from remaining (small |gradient|)
  4. Upweight sampled small-gradient instances by factor (1−a)/b
     (so they still represent the full small-gradient population)
  
  Data reduction: (a + b)% instead of 100%
  
  Example: a=10%, b=10% → use 20% of data, yet barely lose accuracy!
  Speedup: 5×, Quality loss: usually < 0.1% accuracy

EFB (Exclusive Feature Bundling):
  Sparse features that are never non-zero simultaneously can be bundled:
  
  Feature A: [3, 0, 0, 7, 0, 0, 2]
  Feature B: [0, 4, 0, 0, 5, 0, 0]
  Feature C: [0, 0, 6, 0, 0, 8, 0]
  
  A, B, C are mutually exclusive → combine into one feature:
  Bundle: [3, 4+A_offset, 6+B_offset, 7, 5+A_offset, 8+B_offset, 2]
  
  Reduces effective number of features → fewer histograms to build → faster.
```

---

### 8.8 Stacking and Blending

> 💡 **Stacking is "learning to combine models."** Instead of averaging or voting (fixed combination rules), stacking trains a META-LEARNER to figure out the best way to combine base model predictions. The meta-learner sees which models are right in which situations and learns to combine them accordingly.

**Stacking Architecture:**
```
Level 0 (Base Models):              Level 1 (Meta-Learner):
  
  x → Model 1 → ŷ₁                  [ŷ₁, ŷ₂, ŷ₃] → Meta-Model → final ŷ
  x → Model 2 → ŷ₂
  x → Model 3 → ŷ₃

  Base models: diverse (RF, XGBoost, SVM, Linear, KNN, ...)
  Meta-model: often Ridge Regression or simple GBM

The meta-learner learns:
  "When Model 1 says high and Model 2 says low, trust Model 1 for examples
   with feature pattern X, but trust Model 2 for examples with pattern Y."
```

**k-Fold Stacking (Preventing Leakage):**
```
Problem: if we train base models on all training data and make predictions on
the same training data for the meta-learner, we LEAK information.
The meta-learner will see that each base model perfectly fits its training data
→ it learns to trust overfitted predictions → fails at test time.

Solution: Generate "out-of-fold" (OOF) predictions:

For each fold k (say 5-fold):
  1. Train base models on ALL folds EXCEPT fold k
  2. Predict on fold k with these models → fill in the meta-feature for fold k
  3. Repeat for all folds

Result: meta-features for ALL n training points,
        but each point's prediction came from a model that didn't train on it!
        → No leakage

Visualisation (5-fold with 3 base models):

  Training data (5 folds):
  ┌──┬──┬──┬──┬──┐
  │F1│F2│F3│F4│F5│
  └──┴──┴──┴──┴──┘
  
  Fold 1: train on F2-F5, predict F1 → fill meta-features[F1]
  Fold 2: train on F1,F3-F5, predict F2 → fill meta-features[F2]
  ...
  Fold 5: train on F1-F4, predict F5 → fill meta-features[F5]
  
  Final meta-feature matrix Z: (n × M) where M = number of base models
  Train meta-learner on (Z, y).
  
  For test data: average predictions from all 5 fold models.
```

---

## 9. Time Series Analysis

> 💡 **Time series data is special because observations are NOT independent.** Yesterday's temperature affects today's. Last quarter's sales affect this quarter's. The standard ML assumption (i.i.d. data) is VIOLATED. Time series methods explicitly model temporal dependence, trends, seasonality, and autocorrelation.

---

### 9.1 Stationarity

> 💡 **Stationarity is the foundation of time series modelling.** A stationary process has CONSTANT statistical properties over time — the same mean, variance, and autocorrelation structure at all time points. Most classical time series methods require stationarity. Transforming non-stationary data to stationary is the first critical step.

**Definition of Stationarity:**
```
Weak (covariance) stationarity requires ALL THREE:

  1. E[Yₜ] = μ  (constant mean — doesn't trend up or down over time)
  
  2. Var(Yₜ) = σ²  (constant variance — no expansion or contraction of fluctuations)
  
  3. Cov(Yₜ, Yₜ₋ₖ) = γₖ  (covariance depends only on LAG k, not on time t)

Visualisation:

  Stationary:               Non-stationary (trend):   Non-stationary (variance):
  
  y │~~~~~~~~~~~~~~~~~~~~~  y │           ╱           y │    ╱╲    ╱╲    ╱╲╲
    │~~~~~~~~~~~~~~~~~~~~~    │         ╱               │  ╱╲╱  ╲╱╲╱╲╱╲╱  ╲╱╲
    │~~~~~~~~~~~~~~~~~~~~~    │       ╱                 │╱╲╱
    └─────────────────── t    └─────────────── t        └─────────────── t
    
    (wiggles around a fixed    (drifts upward)          (variance grows over time)
     mean — stationary!)
```

**Achieving Stationarity — Transformations:**
```
Problem 1: TREND (non-constant mean)
  Solution: Differencing
  ΔYₜ = Yₜ − Yₜ₋₁   (first difference)
  
  If Yₜ = trend + noise:
  ΔYₜ = (trend_t − trend_{t-1}) + (noise_t − noise_{t-1})
       ≈ constant + noise   ← stationary!
  
  Example: Stock price (random walk with drift):
    Price: 100, 102, 105, 103, 108, ...  (non-stationary, drifts up)
    Returns (ΔPrice): 2, 3, −2, 5, ...   (stationary! random around mean)

Problem 2: SEASONALITY (periodic non-constant mean)
  Solution: Seasonal differencing
  ΔₛYₜ = Yₜ − Yₜ₋ₛ   (subtract same period last year/week/etc.)
  
  Example: Monthly sales with annual seasonality (s=12):
    ΔₛYₜ = Yₜ − Yₜ₋₁₂   (sales this month minus same month last year)

Problem 3: VARIANCE grows over time (heteroscedasticity)
  Solution: Log transform
  Y'ₜ = log(Yₜ + 1)   (+ 1 to handle zeros)
  
  GDP, stock prices, population all grow multiplicatively.
  Log makes multiplicative growth into additive:
  log(Yₜ × (1+rₜ)) = log(Yₜ) + log(1+rₜ) ≈ log(Yₜ) + rₜ
  → variance of log returns is roughly constant even when price grows

Combined (very common in practice):
  Log + difference: log(Yₜ) − log(Yₜ₋₁) ≈ % change
  → removes both trend AND variance growth simultaneously
```

**Unit Root Tests:**
```
ADF (Augmented Dickey-Fuller) Test:
  H₀: series has a UNIT ROOT (non-stationary)
  H₁: series is stationary
  
  If p-value < 0.05: reject H₀ → evidence of stationarity
  If p-value > 0.05: fail to reject → may be non-stationary
  
KPSS (Kwiatkowski-Phillips-Schmidt-Shin) Test:
  H₀: series IS STATIONARY
  H₁: series has unit root (non-stationary)
  
  ← NOTE: Opposite null hypothesis!
  If p-value < 0.05: reject stationarity → may be non-stationary

Gold standard: use both!
  Reject ADF AND fail to reject KPSS → strong evidence of stationarity ✓

Why Stationarity Matters — The Spurious Regression Warning:
  Two INDEPENDENT non-stationary processes:
  
  Process A: 10, 12, 11, 14, 13, 15, 17, 16, ...  (random walk going up)
  Process B: 5,  6,  7,  5,  7,  8,  9,  10, ...  (random walk going up)
  
  Regressing B on A: R² ≈ 0.95, t-statistic huge!
  BUT they're INDEPENDENT — this is completely spurious!
  
  Non-stationary series trend together by chance → fake high R².
  Solution: first-difference both series, then regress.
  Or: test for cointegration first (Engle-Granger, Johansen tests).
```

---

### 9.2 ACF and PACF

> 💡 **The ACF and PACF are diagnostic tools that tell you WHICH TIME SERIES MODEL to use.** Before fitting ARIMA, always plot both — they contain a fingerprint of the underlying temporal structure.

**Autocorrelation Function (ACF):**
```
ACF at lag k:
  ρₖ = Cov(Yₜ, Yₜ₋ₖ) / Var(Yₜ) = γₖ/γ₀   ∈ [−1, +1]

Measures: correlation between Yₜ and Yₜ₋ₖ DIRECTLY.
Includes BOTH direct and indirect effects of intervening time steps.

Example: Daily temperatures
  ρ₁ = 0.9   (today strongly correlated with yesterday)
  ρ₂ = 0.8   (today correlated with 2 days ago — mostly via yesterday)
  ρ₇ = 0.6   (weekly seasonality — same day last week)
  ρ₃₆₅ = 0.7  (annual seasonality)
```

**Partial Autocorrelation Function (PACF):**
```
PACF at lag k:
  φₖₖ = partial correlation between Yₜ and Yₜ₋ₖ
         AFTER REMOVING the effect of lags 1, 2, ..., k−1

"What's the DIRECT effect of the k-day-ago value on today,
 once I account for everything in between?"

Example:
  If Yₜ = 0.8 Yₜ₋₁ + εₜ (AR(1)):
  
  PACF(1) = 0.8  ← direct effect at lag 1
  PACF(2) = 0    ← no direct effect at lag 2 (only via lag 1)
  PACF(3) = 0    ← no direct effect at lag 3
```

**Model Identification Rules:**
```
Look at the pattern of ACF and PACF to determine p, q for ARIMA(p,d,q):

Pattern                     ACF                    PACF
──────────────────────────────────────────────────────────────────────
Pure AR(p)                  Decays geometrically    CUTS OFF at lag p
Pure MA(q)                  CUTS OFF at lag q       Decays geometrically
ARMA(p,q)                   Decays gradually        Decays gradually
Non-stationary              Decays VERY slowly      Large spike at lag 1

Visualisation:
  
  AR(2) model:              MA(1) model:
  
  ACF:                       ACF:
  1.0│●                      1.0│●
  0.5│  ●                    0.5│  ●
  0.3│    ●                  0.0│    ───────────────
  0.2│      ●──────────           0   1   2   3   lag
  0.1│
      0  1  2  3  lag          PACF:
                                1.0│●
  PACF:                         0.5│  ●
  1.0│●                         0.0│    ●──────────
  0.5│  ●                      -0.2│      ●
  0.0│    ───────────           -0.3│        ●─────
      0  1  2  3  lag
  
  PACF cuts off at lag 2!     ACF cuts off at lag 1!
  → AR(2) signature           → MA(1) signature
```

---

### 9.3 ARIMA(p,d,q)

> 💡 **ARIMA is the classical workhorse of time series forecasting.** AR = Autoregressive (past values predict present), I = Integrated (differencing for stationarity), MA = Moving Average (past errors predict present). Most stationary, non-seasonal time series can be well-modelled by ARIMA.

**The Three Components:**
```
AR(p) — Autoregressive:
  Yₜ = c + φ₁Yₜ₋₁ + φ₂Yₜ₋₂ + ... + φₚYₜ₋ₚ + εₜ
  
  "Today's value = constant + weighted sum of p past values + noise"
  
  Intuition: stock prices, temperatures, and many economic series
             are "sticky" — tomorrow tends to be near today.
  
  Stationarity condition: roots of φ(B) = 1 − φ₁B − ... − φₚBᵖ must lie outside unit circle

MA(q) — Moving Average:
  Yₜ = μ + εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + ... + θqεₜ₋q
  
  "Today's value = mean + current shock + weighted sum of past shocks"
  
  Intuition: when you receive information that affects decisions,
             and that information takes q periods to fully work its way through the system.
  
  Note: MA does NOT mean "rolling average"! It's about ERROR terms.

ARIMA(p,d,q):
  Φₚ(B)(1−B)ᵈ Yₜ = c + Θq(B)εₜ
  
  d = order of differencing needed to achieve stationarity
  
  B = backshift operator: B Yₜ = Yₜ₋₁, B² Yₜ = Yₜ₋₂
  (1−B) Yₜ = Yₜ − Yₜ₋₁   (first difference)
  (1−B)² Yₜ = Yₜ − 2Yₜ₋₁ + Yₜ₋₂  (second difference)

Common special cases:
  ARIMA(0,1,0): random walk (Yₜ = Yₜ₋₁ + εₜ)
  ARIMA(1,0,0): AR(1) — stationary autoregression
  ARIMA(0,0,1): MA(1)
  ARIMA(1,1,0): first-order AR on first-differenced series
```

**Model Selection via Information Criteria:**
```
AIC (Akaike Information Criterion):
  AIC = −2ℓ + 2k
  ℓ = log-likelihood of fitted model
  k = number of parameters
  
  Lower is better. Balances fit (ℓ) vs complexity (k).
  Derived from minimising expected KL divergence to true model.

AICc (corrected for small samples — ALWAYS use this):
  AICc = AIC + 2k(k+1)/(n−k−1)
  
  Adds extra penalty for small n (regular AIC can overfit with few observations).

BIC (Bayesian Information Criterion):
  BIC = −2ℓ + k log(n)
  
  Penalises complexity more strongly as n grows (log n > 2 for n > 8).
  Favours simpler models for large datasets.
  Statistically CONSISTENT: will select the true model as n→∞ (under certain conditions).

When to use which:
  Small sample: prefer AICc
  Large n, want parsimony: prefer BIC
  Prediction focus: AIC/AICc (minimises prediction error)
  Model identification focus: BIC (finds the true model)
```

---

### 9.4 Seasonal Models and ETS

**SARIMA — Adding Seasonality:**
```
SARIMA(p,d,q)(P,D,Q)ₛ:
  Seasonal AR order P, differencing D, MA order Q, with period s

Example: SARIMA(1,1,1)(1,1,1)₁₂ for monthly data:
  (1−φ₁B)(1−Φ₁B¹²)(1−B)(1−B¹²)Yₜ = (1+θ₁B)(1+Θ₁B¹²)εₜ
  
  (1−B): removes trend (annual growth)
  (1−B¹²): removes annual seasonality (month-of-year effects)
  Remaining: ARMA structure in the double-differenced series

When to use: data with clear seasonality (monthly sales, quarterly GDP, daily temperatures)
s=12: monthly data with annual seasonality
s=7:  daily data with weekly seasonality
s=4:  quarterly data with annual seasonality
```

**Exponential Smoothing (ETS):**
```
The simplest approach: smooth weighted average of past values.

Simple Exponential Smoothing (SES) — no trend, no seasonality:
  ŷₜ₊₁ = α yₜ + (1−α) ŷₜ
  
  Expanding: ŷₜ₊₁ = α Σⱼ (1−α)ʲ yₜ₋ⱼ
  = exponentially DECAYING weights on past observations!
  
  α = 0.9: heavy weight on recent, almost forget old
  α = 0.1: slow to adapt, gives old observations lots of weight
  
  Weights visualised (α=0.3):
  yₜ    : 0.30  (most recent)
  yₜ₋₁  : 0.21
  yₜ₋₂  : 0.147
  yₜ₋₃  : 0.103
  ...exponentially decaying...

Holt's Method — adds trend:
  lₜ = α yₜ + (1−α)(lₜ₋₁ + bₜ₋₁)     (level estimate)
  bₜ = β(lₜ−lₜ₋₁) + (1−β)bₜ₋₁       (trend estimate)
  ŷₜ₊ₕ = lₜ + h bₜ                     (h-step ahead forecast)

Holt-Winters — adds multiplicative seasonality:
  lₜ = α(yₜ/sₜ₋ₘ) + (1−α)(lₜ₋₁+bₜ₋₁)   (level, seasonality-adjusted)
  bₜ = β(lₜ−lₜ₋₁) + (1−β)bₜ₋₁            (trend)
  sₜ = γ(yₜ/lₜ) + (1−γ)sₜ₋ₘ              (seasonal factor)
  ŷₜ₊ₕ = (lₜ + hbₜ) × sₜ₋ₘ₊ₕ             (forecast)
  
  m = seasonal period (12 for monthly, 7 for daily)
```

---

## 10. Deep Learning Foundations

> 💡 **Deep learning is the art of stacking multiple layers of transformations.** Each layer learns increasingly abstract representations. The first layer of a CNN might detect edges. The second combines edges into shapes. The third combines shapes into object parts. The final layer combines parts into full objects. This hierarchy of features is learned automatically from data.

---

### 10.1 The Perceptron and MLP

**From Neuron to Network:**
```
Biological neuron:             Artificial neuron (perceptron):
  
  Dendrites ──→ ──→ ──→        x₁ ──w₁──→
  (inputs)  ──→ Cell ──→ axon  x₂ ──w₂──→  [Σwᵢxᵢ+b] → g(·) → output
             ──→ body          x₃ ──w₃──→
                               
  Threshold activation          g(z) = activation function
  (fire or don't fire)         (non-linear "squashing")

Multi-Layer Perceptron (MLP) = stacked layers of neurons:

  Input    Hidden Layer 1   Hidden Layer 2    Output
  x₁ ─┬──→ h₁₁ ─┬──→ h₂₁ ─┬──→
  x₂ ─┼──→ h₁₂ ─┼──→ h₂₂ ─┼──→  ŷ
  x₃ ─┘──→ h₁₃ ─┘──→ h₂₃ ─┘──→
  
  Each arrow has a weight. Each neuron applies:
    zˡⱼ = Σᵢ Wˡⱼᵢ aˡ⁻¹ᵢ + bˡⱼ   (weighted sum + bias)
    aˡⱼ = g(zˡⱼ)                  (activation function)

Matrix form for a whole layer:
  zˡ = Wˡ aˡ⁻¹ + bˡ    (matrix multiplication for all neurons at once)
  aˡ = g(zˡ)            (element-wise activation)
```

**Universal Approximation Theorem:**
```
Statement (Cybenko 1989, Hornik 1991):
  A neural network with ONE hidden layer and ENOUGH neurons
  can approximate ANY continuous function on a compact set
  to ARBITRARY precision.

Formally:
  For any continuous f: Kⁿ → ℝ (K compact, n = input dimension)
  and any ε > 0,
  ∃ network N with one hidden layer such that:
    max_{x∈K} |f(x) − N(x)| < ε

What this DOES say:
  Neural networks are universal function approximators.
  In principle, any function can be learned.
  
What this DOESN'T say:
  ✗ How many neurons you need (could be exponential!)
  ✗ That gradient descent will find the right weights
  ✗ That a single layer is efficient (deep networks can approximate more compactly)
  ✗ Anything about generalisation to unseen data

Depth vs Width:
  A depth-L ReLU network partitions input space into O((n/d)^{d(L-1)} × nᵈ) linear regions.
  This grows EXPONENTIALLY in depth — deep networks are exponentially more expressive
  than wide shallow ones for certain functions.
```

---

### 10.2 Activation Functions

> 💡 **Activation functions are what make neural networks nonlinear.** Without them, stacking layers is pointless — a product of linear functions is still linear. The choice of activation function profoundly affects training speed, gradient flow, and what patterns can be learned.

**Complete Gallery:**
```
SIGMOID:
  σ(z) = 1/(1+e⁻ᶻ)   range: (0,1)   derivative: σ(z)(1−σ(z))
  
  P │         ─────────
  1 │      ╱
  0.5─────●──────────────   (symmetric around z=0, value=0.5)
  0 │╲____________
    -5    0    +5    z
    
  ✓ Outputs in (0,1) → good for probabilities
  ✗ Saturates (derivative → 0) for |z| > 3 → VANISHING GRADIENT
  ✗ Not zero-centred → zig-zag gradient updates
  Use: output layer for binary classification ONLY

TANH:
  tanh(z) = (eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ)   range: (−1,1)   derivative: 1−tanh²(z)
  
  1 │         ─────────
  0 │─────────●───────────  (ZERO-CENTRED — better than sigmoid!)
  -1│╲_________________
    -5    0    +5    z
  
  ✓ Zero-centred (better gradient flow than sigmoid)
  ✗ Still saturates for |z| > 2 → still has vanishing gradient problem

ReLU (Rectified Linear Unit):
  ReLU(z) = max(0, z)   range: [0,∞)   derivative: 𝟙[z>0]
  
  a │      /
    │     /
    │    /
    │───/──────────────  z
        0
  
  ✓ No saturation for z > 0 → gradients flow freely!
  ✓ Very fast to compute (just a threshold)
  ✓ Sparse activations (many zeros = efficient)
  ✗ DYING ReLU: if z < 0 always, gradient = 0, neuron is "dead"
  Use: default choice for hidden layers in most networks

LEAKY ReLU:
  f(z) = z if z > 0 else αz   (α ≈ 0.01)   
  
  Keeps a small gradient for z < 0: α × 1 instead of 0.
  → Solves the dying ReLU problem.
  
  a │      /
    │     /
    │────/──────────────  z
      ╱ ← tiny negative slope (α=0.01)

ELU (Exponential Linear Unit):
  ELU(z) = z if z≥0 else α(eᶻ−1)
  
  Smooth for z < 0 (exponential approach to −α).
  Output mean ≈ 0 → better gradient flow than ReLU.
  Slightly more expensive (exp computation).

GELU (Gaussian Error Linear Unit):
  GELU(z) = z × Φ(z)   where Φ = standard normal CDF
  ≈ 0.5z(1 + tanh(√(2/π)(z + 0.044715z³)))
  
  Stochastic interpretation: keep z with probability Φ(z), else drop it.
  → Input-dependent "soft Dropout"
  ✓ Used in: BERT, GPT, Transformers (best for attention mechanisms)
  ✓ Smooth, non-monotone (allows inhibitory effects)

SOFTMAX (output layer only):
  softmax(z)ₖ = exp(zₖ) / Σⱼ exp(zⱼ)
  
  Converts K logits to K probabilities summing to 1.
  
  Example: z = [2.0, 1.0, 0.1]
  exp(z) = [7.39, 2.72, 1.11]
  sum = 11.22
  softmax = [0.659, 0.242, 0.099]  → these are class probabilities!
  
  Temperature-scaled softmax:
  P(k) = exp(zₖ/T) / Σⱼ exp(zⱼ/T)
  T→0: approaches argmax (confident), T→∞: approaches uniform (uncertain)

Activation Function Summary Table:
┌────────────┬──────────┬─────────────────┬──────────────────────────────────┐
│ Function   │ Range    │ Vanishing Grad? │ Best Use                        │
├────────────┼──────────┼─────────────────┼──────────────────────────────────┤
│ Sigmoid    │ (0,1)    │ YES (z>>0)      │ Output: binary classification    │
│ Tanh       │ (-1,1)   │ YES (z>>0)      │ Rarely (prefer ReLU in hidden)  │
│ ReLU       │ [0,∞)    │ No (z>0 side)   │ Default hidden layer choice     │
│ Leaky ReLU │ (-∞,∞)   │ No              │ When dying ReLU is a problem     │
│ ELU        │ (-α,∞)   │ No              │ When zero-mean activations matter│
│ GELU       │ (-∞,∞)   │ No              │ Transformers (BERT, GPT)         │
│ Softmax    │ (0,1)ᴷ   │ N/A             │ Output: multi-class              │
└────────────┴──────────┴─────────────────┴──────────────────────────────────┘
```

**The Dying ReLU Problem — Detailed:**
```
Scenario: neuron with weights w, bias b receives inputs x.
  Pre-activation: z = wᵀx + b

If the weight update during training pushes wᵀx + b < 0 for ALL training examples:
  ReLU output = 0 for all inputs
  Gradient of ReLU = 0 (d(max(0,z))/dz = 0 when z < 0)
  Backprop: gradient × 0 = 0 → weights receive ZERO gradient
  Weights NEVER CHANGE AGAIN → neuron is permanently "dead"

Detection: look at fraction of neurons with activation=0.
  Normal: ~50% of ReLU neurons off at any given input
  Problem: >80% neurons permanently off (zero gradient always)

Causes:
  • Too-large learning rate: one big update pushes all activations negative
  • Bad initialisation: initial weights push all pre-activations negative
  • Aggressive regularisation or L2 weight decay

Solutions:
  • Leaky ReLU: g(z) = max(αz, z), α=0.01 → always has gradient α or 1
  • He initialisation: specifically designed for ReLU networks
  • Lower learning rate
  • Use GELU or ELU (no hard zero)
```

---

### 10.3 Backpropagation — Complete Derivation

> 💡 **Backpropagation is how neural networks learn.** It efficiently computes the gradient of the loss with respect to EVERY weight in the network using the chain rule. The magic: it computes all gradients in a single backward pass through the network, in the same time as a forward pass.

**Forward Pass — Computing Predictions:**
```
For layer l = 1, 2, ..., L:
  zˡ = Wˡ aˡ⁻¹ + bˡ    (pre-activation: weighted sum)
  aˡ = g(zˡ)             (post-activation: apply activation function)

Store ALL intermediate values! They're needed for the backward pass.

Example (3-layer network):
  a⁰ = x                    (input)
  z¹ = W¹ x + b¹            (layer 1 pre-activation)
  a¹ = ReLU(z¹)             (layer 1 output)
  z² = W² a¹ + b²           (layer 2 pre-activation)
  a² = ReLU(z²)             (layer 2 output)
  z³ = W³ a² + b³            (output pre-activation)
  ŷ = softmax(z³)            (final prediction)
  L  = cross-entropy(ŷ, y)   (loss)
```

**Backward Pass — Computing Gradients:**
```
Define ERROR SIGNALS (δ) for efficient computation:
  δˡ = ∂L/∂zˡ   (gradient of loss w.r.t. pre-activation at layer l)

Output layer (l=L):
  δᴸ = (∂L/∂aᴸ) ⊙ g'(zᴸ)
     = (ŷ − y) ⊙ g'(zᴸ)    (for cross-entropy + softmax, simplifies to ŷ−y)

Hidden layer l (working backwards):
  δˡ = ((Wˡ⁺¹)ᵀ δˡ⁺¹) ⊙ g'(zˡ)
        ↑                    ↑
   backpropagated gradient   activation derivative

  In words: "the error signal at layer l comes from backpropagating the error at
             layer l+1 through the weight matrix Wˡ⁺¹, then multiplying by the
             local derivative of the activation function."

Parameter gradients:
  ∂L/∂Wˡ = δˡ (aˡ⁻¹)ᵀ    (outer product of error signal and previous activations)
  ∂L/∂bˡ = δˡ              (error signal directly gives bias gradient)
```

**Why Backpropagation is O(forward pass) — Not O(parameters):**
```
Naive approach: compute ∂L/∂wᵢ separately for EACH weight wᵢ.
  Cost: O(n_params) forward passes → infeasible for millions of weights!

Backpropagation with chain rule:
  Uses DYNAMIC PROGRAMMING — each error signal δˡ is computed once
  and reused for ALL weights in layer l.
  
  Computation graph:    x → z¹ → a¹ → z² → a² → z³ → ŷ → L
  
  Forward pass:  compute and STORE each intermediate value.  Cost: O(1 forward pass)
  Backward pass: propagate gradients in REVERSE.             Cost: O(1 forward pass)
  
  Total: O(2 × forward pass) regardless of number of weights!
  
  This is the key insight of reverse-mode automatic differentiation.
  Training a 1B parameter model still only requires 2 forward passes per step.
```

**Vanishing and Exploding Gradients:**
```
Error signal at layer l:
  δˡ = Πₖ₌ₗ^{L} [diag(g'(zᵏ)) Wᵏ] × δᴸ

For deep networks (L−l large), this product can:

VANISH if ||Wᵏ|| × max|g'| < 1:
  Sigmoid max derivative = 0.25.
  If ||W|| = 1: gradient shrinks by 0.25 per layer.
  After 10 layers: 0.25^10 ≈ 0.0000001 (essentially zero!)
  Early layers receive nearly ZERO gradient → they can't learn.

EXPLODE if ||Wᵏ|| > 1:
  With ||W|| = 2: gradient grows by 2 per layer.
  After 10 layers: 2^10 = 1024 (HUGE)
  Unstable training, NaN losses.

Visualisation:
  
  Vanishing gradient:             Exploding gradient:
  
  Layer 10 (output): δ = 1.0     Layer 10: δ = 1.0
  Layer  9: δ = 0.25             Layer  9: δ = 2.0
  Layer  8: δ = 0.06             Layer  8: δ = 4.0
  Layer  7: δ = 0.015            Layer  7: δ = 8.0
  Layer  6: δ = 0.003            Layer  6: δ = 16.0
  Layer  5: δ ≈ 0.0009           Layer  5: δ = 32.0
  Layer  4: δ ≈ 0               Layer  4: δ = 64.0
       ↑                               ↑
  No learning for early layers    NaN/divergence

Solutions — Vanishing:
  1. ReLU activations (g'=1 for z>0, doesn't shrink gradient)
  2. Skip connections / residual connections (bypass multiplication)
  3. Batch/Layer normalisation (re-normalises gradients)
  4. LSTM/GRU (highway for gradient through time)
  
Solutions — Exploding:
  1. Gradient clipping: g := g × min(1, c/||g||)
     Scale gradient DOWN if it exceeds threshold c. Preserves direction.
  2. Spectral normalisation: ||W|| ≤ 1
  3. Careful initialisation (Xavier, He)
```

---

### 10.4 Optimisation Algorithms

> 💡 **Gradient descent is the engine of deep learning, but the basic version is slow and finicky.** Modern optimisers add momentum (to overcome local flat regions), adaptive learning rates per parameter (to handle different-scale gradients), and bias correction (to avoid bad early updates). These are not just tweaks — they make the difference between a model that trains in hours vs days.

**Gradient Descent Variants:**
```
Batch Gradient Descent: use ALL n examples per update
  ∇L = (1/n) Σᵢ ∇Lᵢ(θ)
  ✓ Exact gradient → stable convergence
  ✗ n=millions → one step takes forever
  ✗ Stuck in local minima (no noise to escape)

Stochastic Gradient Descent (SGD): use 1 example per update
  ∇L ≈ ∇Lᵢ(θ)  for random i
  ✓ Very fast updates, can process streaming data
  ✓ Noise helps escape sharp local minima → better generalisation!
  ✗ Very noisy → erratic convergence

Mini-batch SGD: use batch of 32-512 examples
  ∇L ≈ (1/B) Σᵢ∈batch ∇Lᵢ(θ)
  ✓ Best of both worlds: stable but fast
  ✓ GPU-parallelisable → fast in practice
  ← DEFAULT for deep learning
```

**Momentum:**
```
Problem: gradient descent in a narrow valley oscillates side-to-side
         while making slow progress along the valley floor.

  ↓   ↑   ↓   ↑   ← oscillating sideways
   ↘   ↙   ↘   ↙   ← slowly drifting forward

Momentum adds "inertia" — accumulate velocity in consistent directions:
  vₜ = βvₜ₋₁ + ∇Lₜ
  θₜ₊₁ = θₜ − α vₜ
  
  β ≈ 0.9: 90% of previous velocity carried forward
  
  Effect: consistent directions (toward minimum) build up speed.
          Oscillating directions cancel out.
  
  With momentum:   →→→→→→→→→  (smooth, fast progress along valley)

NAG (Nesterov Accelerated Gradient) — look-ahead version:
  Compute gradient at the LOOKAHEAD position (θ − βv), not current position:
  vₜ = βvₜ₋₁ + ∇L(θₜ − βvₜ₋₁)
  θₜ₊₁ = θₜ − α vₜ
  
  Advantage: "corrects" for overshoot before it happens.
  Converges faster for convex functions.
```

**Adaptive Learning Rates:**
```
Problem: different parameters need different learning rates.
  Rare features (e.g., word "neuroscience" in NLP embedding): small gradient,
  should update more aggressively (large lr for this parameter).
  Frequent features (e.g., "the"): large gradient, should update conservatively.

AdaGrad — accumulate squared gradients:
  G += (∇L)²        (accumulated squared gradient, per parameter)
  θ := θ − (α/√(G+ε)) ∇L
  
  Large G (frequent updates): effective lr = α/√G → SMALL (conservative)
  Small G (rare updates): effective lr = α/√G → LARGE (aggressive)
  
  ✓ Great for sparse features (NLP, recommendation)
  ✗ G keeps growing → lr → 0 → learning stops! (fatal for deep networks)

RMSProp — exponential moving average of squared gradients:
  G := βG + (1−β)(∇L)²   ← "forget" old squared gradients
  θ := θ − (α/√(G+ε)) ∇L
  β ≈ 0.99
  
  ✓ Fixes AdaGrad's decaying lr problem
  ✓ Adapts to non-stationary gradients
```

**Adam (Adaptive Moment Estimation) — The Gold Standard:**
```
Combines momentum (first moment) + RMSProp (second moment):

  mₜ = β₁mₜ₋₁ + (1−β₁)∇L         (first moment: EMA of gradient)
  vₜ = β₂vₜ₋₁ + (1−β₂)(∇L)²      (second moment: EMA of squared gradient)
  
  Bias correction (CRITICAL at the start!):
  m̂ₜ = mₜ/(1−β₁ᵗ)                  (corrects for initialisation at zero)
  v̂ₜ = vₜ/(1−β₂ᵗ)
  
  θₜ₊₁ = θₜ − α × m̂ₜ/(√v̂ₜ + ε)
  
  Default hyperparameters (almost never need tuning):
    β₁ = 0.9   (momentum decay)
    β₂ = 0.999  (RMSProp decay)
    ε  = 1e-8   (numerical stability)
    α  = 1e-3   (learning rate — sometimes tuned)

Why bias correction matters:
  At t=1 with β₁=0.9: m₁ = 0.9×0 + 0.1×g₁ = 0.1×g₁  (10× too small!)
  Without correction: update = 0.1g₁/(√(0.01g₁²)+ε) ≈ 0.1/0.1 = 1 (OK)
  
  Actually: m̂₁ = 0.1g₁/(1−0.9) = g₁ (correct!)
  The correction inflates early estimates back to their true scale.

AdamW — Fixing Adam's Weight Decay:
  Observation: L2 regularisation (add λ||θ||² to loss) ≠ weight decay for Adam!
  
  With Adam:
    ∇L_reg = ∇L + λθ
    Update: θ := θ − α × (m̂/√v̂+ε) where m, v include the λθ gradient
    → The L2 gradient gets SCALED by the adaptive rates → inconsistent regularisation
  
  AdamW explicitly SEPARATES weight decay from gradient adaptation:
    θₜ₊₁ = θₜ − α[m̂ₜ/(√v̂ₜ+ε) + λθₜ]
    
    The λθₜ term is NOT scaled by adaptive rates.
    → True weight decay, independent of gradient history.
    → AdamW is the preferred optimiser for Transformers and LLMs.
```

---

### 10.5 Weight Initialisation

> 💡 **How you initialise weights dramatically affects training.** Too small: activations shrink to zero layer by layer (vanishing). Too large: activations explode to infinity (exploding). The right initialisation preserves signal magnitude through every layer at the start of training.

**The Problem — Cascading Variance:**
```
Consider a ReLU network. At initialisation, for layer l:
  Var(zˡ) ≈ nˡ₋₁ × Var(W) × Var(aˡ⁻¹)
  
  For ReLU: only half the neurons are active (z > 0), so:
  Var(aˡ) ≈ (1/2) nˡ₋₁ × Var(W) × Var(aˡ⁻¹)
  
  For variance to be PRESERVED layer to layer:
  (1/2) nˡ₋₁ × Var(W) = 1
  Var(W) = 2/nˡ₋₁
  
  This is HE INITIALISATION!

Without He initialisation:
  If Var(W) = 1/nˡ₋₁:  Var(a) = (1/2) × 1 × Var(prev) → shrinks by 0.5 each layer
  After 100 layers:     Var(a) = 0.5^100 ≈ 10^{-30}  → VANISHED!
  
  If Var(W) = 4/nˡ₋₁:  Var(a) grows by 2 each layer
  After 100 layers:     Var(a) = 2^100 ≈ 10^{30}  → EXPLODED!
```

**Initialisation Methods:**
```
Xavier/Glorot (for Tanh/Sigmoid):
  Var(W) = 2/(n_in + n_out)
  
  Derived by requiring that variance is PRESERVED in BOTH:
  • Forward pass (signal propagation)
  • Backward pass (gradient propagation)
  
  Accounts for both the number of inputs (n_in) and outputs (n_out) to the layer.

He/Kaiming (for ReLU):
  Var(W) = 2/n_in
  
  The factor of 2 compensates for ReLU zeroing negative activations:
  E[ReLU(z)²] = E[z² × 𝟙(z>0)] = (1/2) E[z²]   (half the neurons active)
  → Need 2× larger initial variance to compensate.

LeCun (for SELU):
  Var(W) = 1/n_in
  
  SELU (Scaled ELU) is self-normalising: it preserves mean=0 and var=1 through layers
  when combined with LeCun initialisation.

In practice:
  Default: He initialisation with ReLU
  Transformers: often Xavier with scaling √(2/d) for each sub-layer
  RNNs: orthogonal initialisation (W is an orthogonal matrix)
```

---

## 11. Convolutional Neural Networks (CNN)

> 💡 **CNNs exploit the structure of images.** Images have two key properties: (1) nearby pixels are related — the pixel at position (100,100) tells you something about (100,101). (2) Useful patterns (edges, textures, objects) can appear ANYWHERE in an image. CNNs are specifically designed to exploit both properties: local connectivity exploits (1), and weight sharing exploits (2).

---

### 11.1 The Convolution Operation

**What Convolution Computes:**
```
Cross-Correlation (what CNNs actually compute, called "convolution" by convention):
  (I ⋆ K)(i,j) = Σₘ Σₙ I(i+m, j+n) × K(m,n)

I = input (image)
K = kernel (filter)

Visualisation (3×3 kernel sliding over input):

Input (5×5):           Kernel (3×3):          Output (3×3):
┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│ 1  2  3  4  5│        │ 0  1  0 │        │        │
│ 6  7  8  9 10│     ×  │ 1 -4  1 │   →    │   O₁₁  │
│11 12 13 14 15│        │ 0  1  0 │        │        │
│16 17 18 19 20│        └─────────────┘        └─────────────┘
│21 22 23 24 25│
└─────────────┘

Step: kernel "slides" over input, computing dot product at each position.

O₁₁ = 1×0 + 2×1 + 3×0 + 6×1 + 7×(−4) + 8×1 + 11×0 + 12×1 + 13×0
     = 0 + 2 + 0 + 6 − 28 + 8 + 0 + 12 + 0
     = 0  ← This particular kernel detects "Laplacian" (edges)

Output size formula:
  H_out = ⌊(H_in + 2P − D(K−1) − 1) / S⌋ + 1

Where:
  H_in = input height
  P    = padding (zeros added around border)
  D    = dilation (spacing between kernel elements, usually 1)
  K    = kernel size
  S    = stride (how many pixels to move each step)
```

**Why Weight Sharing is Genius:**
```
Fully Connected layer: each output neuron connects to every input pixel.
  ImageNet image: 224×224×3 = 150,528 inputs
  Hidden layer of 4096 neurons: 150,528 × 4096 = 617 MILLION weights per layer!

Convolutional layer: ALL output positions SHARE the same kernel weights.
  3×3×3 kernel: only 27 weights!
  Regardless of image size, same 27 weights learn to detect the same pattern.
  
  Why is this valid?
  An edge detector in the top-left corner of an image is the same edge detector
  as in the bottom-right corner — the underlying pattern doesn't change with location!
  
  Weight sharing inductive bias:
  "Important patterns are TRANSLATION INVARIANT — the same features
   appear regardless of WHERE in the image they occur."
  
  For images: this assumption is correct → massive parameter reduction!

Translation Equivariance:
  If f(x) is the feature map after convolution:
  f(T(x)) = T(f(x))   where T = translation
  
  "Shift the input → shift the output by the same amount."
  Moving a cat 10 pixels right → cat detection feature map shifts 10 pixels right.
  This allows a single learned filter to detect a pattern ANYWHERE in the image.
```

**Receptive Field — What Does Each Neuron "See"?**
```
The receptive field of a neuron = the region of input that influences its output.

Layer 1 (3×3 conv): each neuron sees a 3×3 patch of input.
Layer 2 (3×3 conv): each neuron sees a 5×5 region of original input!
Layer 3 (3×3 conv): 7×7 region!

General formula:
  RF_l = RF_{l-1} + (K_l − 1) × Π_{j<l} S_j

  K_l = kernel size at layer l
  S_j = stride at layer j

Two 3×3 layers vs one 5×5 layer:
  Same receptive field: 5×5
  Two 3×3 layers: 2 × (3×3×C²) = 18C² parameters
  One 5×5 layer:  1 × (5×5×C²) = 25C² parameters
  → 28% fewer parameters for same receptive field!
  → PLUS: two non-linearities (more expressive) vs one
  → This is why ALL modern CNNs stack small (3×3) kernels!

Dilated Convolutions — Large RF, Same Parameters:
  Insert gaps of D between kernel elements:
  Effective kernel size: K' = D(K−1) + 1
  
  D=1 (no dilation):  ● ● ●    receptive field = 3
  D=2:                ● × ●    receptive field = 5 (× = skipped input)
  D=4:                ● × × × ●  receptive field = 9
  
  Exponentially growing receptive field with NO extra parameters!
  Used in: WaveNet (audio), DeepLab (segmentation), temporal CNNs.
```

---

### 11.2 Pooling and Normalisation

**Pooling Layers:**
```
Max Pooling (2×2, stride 2):
  
  Input:                        Output:
  ┌────────────────┐            ┌────────────┐
  │  1   3   2   4 │            │   3   4    │   max(1,3,5,7)=7? Wait...
  │  5   7   8   6 │   →        │   8   9    │
  │  1   2   3   4 │            └────────────┘
  │  5   6   7   9 │
  └────────────────┘
  
  Top-left 2×2 block {1,3,5,7}: max = 7
  Top-right 2×2 block {2,4,8,6}: max = 8
  ...and so on.
  
  Effect: reduces spatial dimensions by 2× (stride=2).
          Retains the STRONGEST activations.

Why Max Pooling?
  ✓ Reduces computation (smaller feature maps)
  ✓ Provides translation invariance: a feature detected anywhere in a 2×2 region
    contributes the same max value → small shifts don't change output
  ✓ Focuses on "did this feature appear anywhere in this region?" (presence detection)

Global Average Pooling (GAP):
  For each feature map of shape H×W: output = average of ALL H×W values.
  C feature maps → C output values.
  
  Use: replaces fully connected layers in modern CNNs (GoogLeNet, ResNet).
  Benefits:
  ✓ No parameters to learn (vs FC: C × H × W × n_classes parameters)
  ✓ More robust to translations (average is more stable than max)
  ✓ Forces each feature map to represent a class-specific concept
  ✓ Works for any input size (no fixed spatial dimensions needed)
```

**Batch Normalisation:**
```
Problem: as training proceeds, the distribution of each layer's input changes
         as previous layers' weights change.
         This "internal covariate shift" makes each layer constantly adapt to
         a shifting input distribution → slow training.

Batch Normalisation:
  For a mini-batch of activations x of size (B, C, H, W):
  
  For each channel c:
    μ_c = (1/BHW) Σᵦ Σₕ Σw x_{b,c,h,w}        (batch mean)
    σ²_c = (1/BHW) Σᵦ Σₕ Σw (x−μ_c)²           (batch variance)
    
    x̂_{b,c,h,w} = (x_{b,c,h,w} − μ_c) / √(σ²_c + ε)   (normalise)
    y_{b,c,h,w} = γ_c × x̂ + β_c                          (scale and shift)
    
    γ, β: learnable parameters (learned scale and shift per channel)
    ε: small constant for numerical stability (1e-5)

Training vs Inference difference:
  Training: use batch statistics (μ_B, σ²_B) — introduces noise (regularisation!)
  Inference: use running averages (EMA over all training batches) → deterministic
    μ_running = 0.9 × μ_running + 0.1 × μ_B  (exponential moving average)
  
  This is why model.train() and model.eval() MATTER in PyTorch!

Why BN Accelerates Training (Santurkar et al., 2018):
  Popular belief: BN reduces internal covariate shift.
  Actual finding: BN makes the loss landscape SMOOTHER.
  
  Smoothness: smaller Lipschitz constant of the gradient
  → gradients don't change drastically with parameter changes
  → larger learning rates are safe → faster convergence
  → less sensitive to initialisation

Normalisation Methods Comparison:
  
  Input: (Batch=4, Channels=2, Height=2, Width=2)
  
  Batch Norm:   normalise across N,H,W for each C → each channel has mean=0,std=1 in batch
  Layer Norm:   normalise across C,H,W for each N → each sample normalised across features
  Instance Norm: normalise across H,W for each N,C → each sample,channel normalised spatially
  Group Norm:   normalise across C//G channels within each group for each N
  
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Method      │ Normalise over  │ Batch-dep │ Best use                │
  ├─────────────────────────────────────────────────────────────────────┤
  │ Batch       │ N, H, W         │ YES       │ CNNs, large batches     │
  │ Layer       │ C, H, W         │ NO        │ Transformers, RNNs      │
  │ Instance    │ H, W            │ NO        │ Style transfer          │
  │ Group       │ C//G, H, W      │ NO        │ Small batches, detection│
  │ RMS Norm    │ All features    │ NO        │ LLaMA, Gemma (simpler)  │
  └─────────────────────────────────────────────────────────────────────┘
  
  RMS Norm (used in modern LLMs):
    x̂ = x / √(Σxᵢ²/d + ε)   output = γ × x̂   (no mean subtraction!)
    Simpler, faster, empirically works just as well as Layer Norm.
```

---

### 11.3 CNN Architecture Evolution

**The ResNet Revolution — Skip Connections:**
```
Problem before ResNet: deeper networks were HARDER to train than shallow ones.
  Counterintuitive! More layers = more capacity, so shouldn't it be better?
  
  Reason: vanishing gradients. Gradient magnitude:
  ||δˡ|| ≈ Πₖ ||Wᵏ|| × ||g'(zᵏ)||
  
  For 50+ layers: product of many <1 values → ≈ 0 at the early layers.
  
  He et al. (2015) solution: RESIDUAL CONNECTIONS (skip connections):
  
  Normal block:          Residual block:
  
  x → [F] → y           x ─┬→ [F] → +output
                            └──────────┘
                                   (x + F(x) = y)
  
  y = F(x)               y = F(x) + x
  
  Key insight: F(x) now represents the RESIDUAL (deviation from identity).
  The network learns F(x) = y − x (small correction) instead of y (whole output).
  
  Why is this easier?
  If optimal output ≈ input, then optimal F ≈ 0.
  Learning F=0 is easy (just drive all weights to zero).
  Learning a complex transformation from scratch is hard.
  → "Residual learning" is an easier optimisation problem.

Gradient flow with skip connections:
  ∂L/∂x = ∂L/∂y × (1 + ∂F/∂x)
  
  The +1 ensures gradients of at least 1 flow back through each block!
  Early layers receive MUCH larger gradients → can learn effectively.
  
  This enabled ResNet-152 (152 layers!) to outperform 8-layer AlexNet.

Architecture Timeline:
  
  2012 AlexNet: 5 conv + 3 FC, 60M params → won ImageNet, started the revolution
  2014 VGG:     All 3×3 convs, 16-19 layers, 138M params → very regular, easy to use
  2014 GoogLeNet: Inception modules, 22 layers, only 5M params (efficiency focus)
  2015 ResNet:  Skip connections, 50-152 layers, 3.57% top-5 error
  2017 DenseNet: Every layer connected to all subsequent layers (dense connections)
  2019 EfficientNet: Compound scaling of depth/width/resolution simultaneously
  2020 ViT:     Transformers for images (see Section 15.14)
```

---

## 12. Recurrent Neural Networks (RNN)

> 💡 **RNNs process sequences by maintaining a "memory state."** At each time step, the RNN takes the current input AND its previous hidden state, producing a new hidden state. This hidden state acts as a compressed memory of everything seen so far. This enables processing text, speech, time series — any data where order matters.

---

### 12.1 Vanilla RNN

**The Architecture:**
```
Traditional feedforward:    RNN:
  x → [NN] → output          At each time step t:
  (processes one item)        hₜ = tanh(Wₕ hₜ₋₁ + Wₓ xₜ + b)
                              ŷₜ = softmax(Wᵧ hₜ + bᵧ)
                              ↑
                              same weights Wₕ, Wₓ used at EVERY step!

Unrolled through time (for sequence x₁, x₂, x₃, x₄):

  x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃ → [RNN] → h₄ → ŷ
         ↑ shared         ↑ shared         ↑ shared
         Wₕ, Wₓ           Wₕ, Wₓ           Wₕ, Wₓ

  h₀: initial hidden state (usually zeros)
  Each state h_t: compressed summary of x₁, ..., xₜ
  
  Parameters: Wₕ, Wₓ, bₕ, Wᵧ, bᵧ (shared across all time steps!)
  Much fewer parameters than a feedforward network processing fixed-length sequences.
```

**BPTT (Backpropagation Through Time):**
```
To compute ∂L/∂Wₓ and ∂L/∂Wₕ, we "unroll" the RNN through time and apply chain rule:

∂hᵢ/∂hᵢ₋₁ = diag(tanh'(zᵢ)) × Wₕ

∂L/∂Wₓ = Σₜ ∂Lₜ/∂hₜ × Πₖ₌ᵢ^{ₜ} (∂hₖ/∂hₖ₋₁) × ∂h₁/∂Wₓ

The product Πₖ (∂hₖ/∂hₖ₋₁) is the problematic term:
  = Πₖ diag(tanh'(zₖ)) Wₕ
  
  tanh max derivative = 1 (at z=0), but typically ≈ 0.3-0.7
  If ||Wₕ|| × 0.5 < 1: vanishing gradient (can't learn long-term dependencies)
  If ||Wₕ|| > 1: exploding gradient (unstable)

Practical consequence:
  Standard RNNs can only learn dependencies spanning ~10-20 time steps.
  For "The cat, which ate the fish that was bought at the store that..., was fat."
  → The dependency between "cat" and "was fat" spans too many steps → can't learn it.

Solutions:
  Gradient clipping: g := g × min(1, c/||g||)   (for exploding)
  LSTM/GRU: designed specifically for long-term dependencies (for vanishing)
  Truncated BPTT: only backprop k steps → O(k) memory, approximate gradient
```

---

## 13. Long Short-Term Memory (LSTM) & GRU

---

### 13.1 LSTM — Solving Long-Term Dependencies

> 💡 **LSTM adds a "cell state" — a highway for information to flow through time without being multiplied through weights repeatedly.** The key insight: instead of the hidden state (which goes through tanh and weight multiplication at every step), LSTM has a SEPARATE cell state that flows through mostly unchanged, with small ADDITIONS and REMOVALS controlled by "gates."

**The Architecture:**
```
Four components:

1. FORGET GATE — decide what to erase from cell state:
   fₜ = σ(Wf[hₜ₋₁; xₜ] + bf)
   fₜ ∈ (0,1) per element: 0 = completely forget, 1 = completely keep
   
   Example: reading a new sentence → forget the previous sentence's subject

2. INPUT GATE — decide what new info to add:
   iₜ = σ(Wᵢ[hₜ₋₁; xₜ] + bᵢ)   ← how much to write
   C̃ₜ = tanh(Wc[hₜ₋₁; xₜ] + bc)  ← what to write (candidate values)
   
   Example: encountering a new subject → update the "current subject" memory

3. CELL STATE UPDATE — combine forget and add:
   Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
         ↑            ↑
   forget old info    add new info
   
   The cell state flows through with only ELEMENT-WISE multiplication and addition!
   No matrix multiplication that could cause gradient vanishing.

4. OUTPUT GATE — decide what to expose from cell state:
   oₜ = σ(Wo[hₜ₋₁; xₜ] + bo)
   hₜ = oₜ ⊙ tanh(Cₜ)
   
   Only exposes RELEVANT parts of cell state as hidden state.

Memory diagram:
  
  Past cell state Cₜ₋₁                    Current cell state Cₜ
  ────────────────────────────────────────────────────────────────→
  ↑ unchanged except for:
  │  × fₜ (forget)   + iₜC̃ₜ (write)
  │
  └── gradient flows through with minimal vanishing!
```

**Why LSTM Solves Vanishing Gradient:**
```
Gradient through the cell state:
  ∂Cₜ/∂Cₜ₋₁ = fₜ   (element-wise multiplication, NOT matrix multiplication!)
  
  Compare to vanilla RNN:
  ∂hₜ/∂hₜ₋₁ = diag(g'(zₜ)) Wₕ   (matrix multiplication → eigenvalue issues)
  
  LSTM: gradient = product of forget gate values
  If fₜ ≈ 1: gradient flows completely unimpeded!
  If fₜ ≈ 0: this information was forgotten (intentionally)
  
  The LSTM LEARNS when to let gradients flow (f≈1 = remember)
  and when to cut them off (f≈0 = forget).
  
  This gives CONTROL over the gradient, eliminating both vanishing and explosion.

Practical memory span: 100s to 1000s of steps (vs ~10-20 for vanilla RNN).
```

---

### 13.2 GRU (Gated Recurrent Unit)

> 💡 **GRU is LSTM simplified.** It achieves comparable performance with 25% fewer parameters by merging the forget and input gates into a single "update gate" and eliminating the separate cell state. For most tasks, GRU and LSTM perform similarly, but GRU trains faster.

**GRU Architecture:**
```
Two gates instead of three:

1. RESET GATE — how much to ignore past hidden state:
   rₜ = σ(Wr[hₜ₋₁; xₜ] + br)
   rₜ ≈ 0: completely ignore past → start fresh
   rₜ ≈ 1: use all of past hidden state

2. UPDATE GATE — how much to update the hidden state:
   zₜ = σ(Wz[hₜ₋₁; xₜ] + bz)
   zₜ ≈ 1: replace old with new info (large update)
   zₜ ≈ 0: keep old hidden state (small update)

3. Candidate state:
   h̃ₜ = tanh(Wh[rₜ ⊙ hₜ₋₁; xₜ] + bh)
   (reset gate controls how much past to incorporate into candidate)

4. Final hidden state:
   hₜ = (1−zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
         ↑                   ↑
    keep old state       use new candidate
    (scaled by 1−zₜ)    (scaled by zₜ)

This is the COMPLEMENTARY structure:
  If zₜ=1: completely replace with new info
  If zₜ=0: keep old state unchanged (perfect memory!)
```

**LSTM vs GRU — When to Use Which:**
```
┌─────────────────────────────────────────────────────────────────────┐
│               LSTM                       GRU                       │
├─────────────────────────────────────────────────────────────────────┤
│ Gates         3 (forget, input, output)  2 (reset, update)         │
│ State         Separate cell + hidden     Single hidden state        │
│ Parameters    4 × (n_h(n_x+n_h) + n_h)  3 × (n_h(n_x+n_h) + n_h) │
│ Complexity    Higher                     25% less                   │
│ Long seqs     Slightly better            Comparable                 │
│ Speed         Slower                     ~33% faster                │
│ Recommendation: try GRU first; use LSTM if performance matters more │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 14. Advanced Deep Learning

---

### 14.1 Generative Models

> 💡 **Generative models learn to PRODUCE new data that looks like the training data.** Instead of learning "which category does this belong to?", they learn "what does this type of data look like?" This enables generating new images, synthesising speech, creating text, and many creative applications.

**Variational Autoencoder (VAE):**
```
Architecture:
  Encoder: x → q(z|x) = N(μ(x), σ²(x))    (maps input to distribution over latent z)
  Decoder: z → p(x|z)                        (maps latent z back to data space)

The ELBO (Evidence Lower BOund) objective:
  L = E_q[log p(x|z)] − KL(q(z|x) || p(z))
       ↑                    ↑
  Reconstruction term    Regularisation term
  
  (Maximise: predictions should be good)   (Minimise: posterior should be near prior)
  
  p(z) = N(0, I): standard Gaussian prior (latent space should be "nice")
  
Why "Evidence LOWER Bound"?
  log p(x) = ELBO + KL(q(z|x) || p(z|x)) ≥ ELBO
  
  The true log-likelihood is intractable (p(z|x) requires integrating over all z).
  ELBO is a tractable lower bound that we maximise instead.
  Maximising ELBO = simultaneously:
    1. Maximising likelihood of reconstruction
    2. Minimising KL between approximate and true posterior

The Reparameterisation Trick (enables backprop through sampling):
  Problem: sampling z ~ q(z|x) = N(μ,σ²) is NOT differentiable.
           Can't backpropagate through a random sample!
  
  Solution: z = μ(x) + σ(x) ⊙ ε,  ε ~ N(0,I)
  
  The randomness is SEPARATED into ε (not a function of θ).
  Gradients flow through μ(x) and σ(x) normally.
  
  Before trick:   After trick:
  x → μ,σ → sample z  x → μ,σ → z = μ + σε, ε~N(0,1)
  (can't differentiate)   (differentiable w.r.t. μ,σ!)
```

**GAN (Generative Adversarial Network):**
```
Two networks in competition:

  Generator G:      z (random noise) → G(z) (fake data)
  Discriminator D:  x (real or fake) → D(x) ∈ [0,1] (probability of being real)

Minimax game:
  min_G max_D E[log D(x)] + E[log(1−D(G(z)))]
  
  D wants: log D(x) high (correctly identify real as real)
           log(1−D(G(z))) high (correctly identify fake as fake)
  
  G wants: log(1−D(G(z))) LOW (fool D into thinking fake is real)
           Equivalent: G maximises log D(G(z)) (non-saturating loss)
  
  Training alternates:
    1. Fix G, train D: distinguish real from fake
    2. Fix D, train G: generate more realistic fakes
  
  Equilibrium (Nash): G produces samples indistinguishable from real data.
  D(x) = 0.5 everywhere (can't tell real from fake).
  
  At equilibrium: p_G = p_data (generator has learned true distribution!)

Visualisation:
  
  Iteration 0:       Iteration 100:      Iteration 1000:
  G: random noise    G: blurry shapes    G: realistic images
  D: easy task       D: harder           D: barely better than chance
  
  ●●●○○●●○ →        ●●●●○○●● →         ●●●●●●●●
  (D easily wins)   (getting harder)    (D can't tell anymore)

Training challenges:
  Mode collapse: G always produces the same output (exploits D's weakness)
  Training instability: D becomes too strong → G receives no gradient
  Solution: Wasserstein GAN (uses different distance metric, more stable training)
```

**Diffusion Models:**
```
The highest-quality generative model paradigm (as of 2024):

FORWARD PROCESS — gradually destroy the image with noise:
  q(xₜ|xₜ₋₁) = N(√(1−βₜ)xₜ₋₁, βₜI)
  
  x₀ = clean image
  x₁ = slightly noisy image
  x₂ = more noisy
  ...
  xT = pure Gaussian noise N(0,I)
  
  Schedule: β₁ < β₂ < ... < βT (noise increases each step)
  Closed form: xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε,  ε~N(0,I)
               ᾱₜ = Πₛ≤ₜ(1−βₛ)

REVERSE PROCESS — learn to denoise:
  p_θ(xₜ₋₁|xₜ) = N(μ_θ(xₜ,t), Σ_θ(xₜ,t))
  
  Train a neural network ε_θ to predict the noise added at step t:
  L = E[||ε − ε_θ(√ᾱₜ x₀ + √(1−ᾱₜ)ε, t)||²]
  
  Simple MSE loss! Predict which noise was added to get xₜ.

Sampling (generation):
  Start with xT ~ N(0,I) (pure noise)
  For t = T, T−1, ..., 1:
    Predict noise ε_θ(xₜ, t)
    Remove predicted noise to get xₜ₋₁
  Result: x₀ ≈ clean generated image!

Why diffusion models work so well:
  Training objective: simple denoising (MSE) — easy to optimise!
  No adversarial training instability (vs GANs)
  No posterior collapse (vs VAEs)
  Score function connection: ε_θ ∝ ∇_x log p_t(x) (score matching)
  → Training a denoiser ≡ learning the score function of the noisy distribution
```

---

### 14.2 Regularisation Techniques

**Dropout:**
```
During training: randomly set each neuron output to 0 with probability (1−p).
  p = "keep probability" (typically 0.5-0.8)
  
  Effectively trains an ENSEMBLE of 2^n "thinned" networks sharing weights.
  
  At each forward pass, a different random subset of neurons is active.
  
  Batch 1: ● ● ○ ● ○ ● ← activated neurons
  Batch 2: ● ○ ● ○ ● ● ← different neurons dropped
  Batch 3: ○ ● ● ● ○ ● ← different again
  
  Inverted Dropout (standard implementation):
  During training: multiply kept activations by 1/p (scale UP to compensate for zeros)
  At test time: use ALL neurons (no scaling needed — outputs already at right scale)
  
  Why it works:
  1. Ensemble interpretation: averaging 2^n thinned networks ≈ ensemble learning
  2. Reduces co-adaptation: neurons can't rely on specific other neurons (randomly absent)
  3. Adds noise: regularises by making training harder → reduces overfitting

Data Augmentation (implicit regularisation via bigger effective dataset):

  Images:
    Flip (horizontal):   ████ → ████ (mirror)
    Rotation:            ████ → ████ (rotated 15°)
    Random crop:         select random 224×224 from 256×256 image
    Color jitter:        randomly change brightness, contrast, saturation
    Mixup:               x_mix = λxᵢ + (1−λ)xⱼ,  y_mix = λyᵢ + (1−λ)yⱼ
                         Creates INTERPOLATED training examples!
    CutMix:              paste a patch from one image into another
  
  Text:
    Synonym replacement: swap words with synonyms from WordNet
    Back-translation:    English → German → English (paraphrase)
    Random deletion/insertion/swap of words (EDA)

Label Smoothing:
  y_smooth = (1−ε) × y_hard + ε/K
  
  Instead of one-hot [0,0,1,0] (100% confident):
  Use smooth label [0.025, 0.025, 0.9, 0.025] (ε=0.1, K=4)
  
  Why?
  Cross-entropy with one-hot: forces softmax logits to grow unboundedly (logit → ∞ for correct class).
  Label smoothing: finite optimal logits → prevents overconfident predictions.
  
  Information-theoretic view: adds entropy penalty
    H(y_smooth, ŷ) = (1−ε)H(y_hard, ŷ) + ε × H(uniform, ŷ)
  The second term maximises entropy → models are less "peaked."
  Result: better calibration, slightly better accuracy, especially for knowledge distillation.
```
# 🤖 Machine Learning, Deep Learning, NLP & LLM Engineering
### The Most Complete Beginner-to-Production Reference — Sections 15 Onwards

> **Every algorithm. Every derivation. Every formula. Every intuition.**
> Written so that ANYONE — even with zero prior knowledge — can understand.
> Theory → Intuition → Math → Visualisation → Code → Production.

> 💡 **How to use this guide:** If you are new, read every plain-English block first. Follow the ASCII diagrams. Don't skip analogies — they are the foundation. The math will make sense once you have the intuition. If you have background, use the derivations, proofs, and production notes.

---

## 📑 Table of Contents (This Document)

15. [Transformers — Complete Deep Dive](#15-transformers--complete-deep-dive)
16. [Natural Language Processing (NLP)](#16-natural-language-processing-nlp)
17. [LangChain & LLM Application Engineering](#17-langchain--llm-application-engineering)
18. [Handling Imbalanced Data](#18-handling-imbalanced-data)
19. [Model Evaluation & Selection](#19-model-evaluation--selection)
20. [Regularisation & Optimisation](#20-regularisation--optimisation)
21. [Probabilistic Machine Learning](#21-probabilistic-machine-learning)

---

## 15. Transformers — Complete Deep Dive

> 💡 **Transformers are the architecture behind GPT, BERT, T5, LLaMA, DALL-E, Whisper, and virtually every state-of-the-art AI model today.** The key innovation: replace sequential RNN processing with PARALLEL attention over all positions simultaneously. This one change — "look at everything at once instead of one step at a time" — made it possible to train models with hundreds of billions of parameters on trillions of words. This section explains every component from absolute scratch.

---

### 15.1 Why Transformers? — The Problem with Everything Before

> 💡 **Before Transformers, the best sequence models were RNNs and LSTMs.** They worked by reading one word at a time, left to right, updating a "memory" at each step. This created three fundamental problems that Transformers solve completely.

**Problem 1: Sequential Computation (Cannot Parallelise)**
```
RNN processes one word at a time:

  Word 1 → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃ → ... → hₙ → output
              ↑ must wait      ↑ must wait      ↑ must wait
              for h₁           for h₂           for h₃

  Step 100 CANNOT be computed until steps 1-99 are done.
  
  Modern GPUs have ~10,000 parallel cores → 99% of them sit IDLE!
  
  For a sequence of 1,000 words: 1,000 sequential steps — no matter
  how powerful your hardware is, you can't speed this up.

Transformer processes ALL words simultaneously:

  [Word 1]  [Word 2]  [Word 3]  ...  [Word n]    ← all at once!
      ↓          ↓          ↓              ↓
  [Attention over ALL positions — fully parallel matrix operations]
      ↓          ↓          ↓              ↓
  [Output 1] [Output 2] [Output 3] ... [Output n]
  
  ALL GPU cores busy simultaneously → training is 10-100× faster!
```

**Problem 2: The Vanishing Memory Problem**
```
RNN compresses everything into a single vector:

  "The cat, which had been living in the old barn at the edge of the
   farm for the past three winters, eating mice and sleeping in the
   hay, was finally fat."
   ↑                                                              ↑
   "cat" mentioned here                              "was fat" refers to "cat"
   
   Distance: ~30 words apart
   
   RNN: by the time it reaches "was fat", the memory of "cat" has been
        overwritten ~30 times. The gradient back to "cat" vanishes.
   
   Memory state after 30 updates:
   h₁ = [info about cat]
   h₅ = [muddy mix of cat + barn + farm + ...]
   h₁₅ = [mostly farm + winter + ...]
   h₃₀ = [mostly hay + sleeping + fat ← cat info mostly GONE]

Transformer: every word attends to EVERY other word directly:

   "fat" → looks directly at → "cat" (full strength, no dilution)
   "fat" → looks directly at → "barn" (low weight, irrelevant)
   "fat" → looks directly at → "was" (medium weight)
   
   The path from "fat" to "cat" is always LENGTH 1, regardless of distance.
   No forgetting. No dilution. Direct access.
```

**Problem 3: The Bottleneck**
```
Seq2Seq RNN (used for translation):

  Input sentence → [Encoder RNN] → ONE VECTOR → [Decoder RNN] → Output

  Entire meaning of "How are you doing on this fine Thursday morning?"
  must fit into ONE fixed-size vector (e.g., 512 numbers).
  
  For long sentences: this vector is like trying to describe the contents
  of an entire library using only 512 words — information is lost.

Transformer: no bottleneck. Decoder attends to ALL encoder positions.
  For each output word, the decoder looks at ALL input words and
  decides which are relevant — full-resolution access to the input.
```

---

### 15.2 Scaled Dot-Product Attention — The Core Operation

> 💡 **Attention is the heart of everything.** The idea: when processing a word, instead of being limited to a fixed memory state, LOOK at all the other words and gather information from the relevant ones. The "query" is what you're looking for. The "keys" are labels for what each position contains. The "values" are the actual information to retrieve.

**The Perfect Analogy — A Smart Library:**
```
Imagine a library where every book has a KEY (title/topic) on the cover.

You walk in with a QUERY: "I want information about cat behaviour"

The library compares your query against EVERY book's key:
  "Feline Psychology"      → HIGH match score → show this book prominently
  "Dog Training Manual"    → MEDIUM match → show a little
  "Car Engine Repair"      → LOW match → barely show
  "Ancient Roman History"  → ZERO match → ignore completely

But instead of giving you ONE book (hard lookup), the library creates a
custom summary: 80% "Feline Psychology" + 15% "Dog Training" + 5% others.

This weighted combination IS the attention output.

In neural network terms:
  QUERY   (Q) = "what am I looking for?"
  KEY     (K) = "what does each position label itself as?"
  VALUE   (V) = "what actual information does each position hold?"
  
  Score between query i and key j = how much position i wants to
                                     "borrow" information from position j
```

**The Mathematical Formula:**
```
Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) × V

Let's break this down piece by piece:

Step 1 — Compute similarity scores:
  S = QKᵀ    shape: (n × n)   where n = sequence length
  
  Sᵢⱼ = qᵢ · kⱼ = dot product of query i with key j
       = how much position i "wants" information from position j
  
  Example (4-word sentence: "The cat sat mat"):
        The   cat   sat   mat
  The [ 0.1  0.8   0.3   0.2 ]  ← "The" wants mostly "cat" info
  cat [ 0.9  0.1   0.5   0.3 ]  ← "cat" wants mostly "The" (article agreement)
  sat [ 0.2  0.7   0.1   0.6 ]  ← "sat" wants "cat" (subject) and "mat" (object)
  mat [ 0.1  0.3   0.8   0.1 ]  ← "mat" wants "sat" (what happened to mat?)

Step 2 — Why divide by √dₖ?
  
  If q, k ~ N(0,1) independently, then q·k ~ N(0, dₖ)
  Variance grows with dimension! For dₖ = 512: std ≈ 22.6
  
  WITHOUT scaling:
  Scores like [−15, +23, −8, +19] → softmax → [≈0, ≈1, ≈0, ≈0]
  Nearly one-hot: one position gets ALL the attention → no information mixing
  Gradient of near-one-hot softmax ≈ 0 → no learning possible!
  
  WITH scaling by √dₖ:
  Divide by √512 ≈ 22.6 → scores now [−0.66, 1.02, −0.35, 0.84]
  Softmax → [0.15, 0.40, 0.20, 0.25] → nice distribution → learning works!

Step 3 — Softmax converts scores to probabilities:
  A = softmax(S / √dₖ)   each ROW sums to 1
  
  Aᵢⱼ = attention weight: "what fraction of position j's information
         does position i incorporate into its output?"
  
  Aᵢⱼ ∈ (0, 1),   Σⱼ Aᵢⱼ = 1 for each i

Step 4 — Weighted sum of values:
  Z = AV    shape: (n × dᵥ)
  
  Zᵢ = Σⱼ Aᵢⱼ × Vⱼ    (weighted average of value vectors)
     = position i's new representation, enriched with info from all positions
```

**Visualising the Attention Matrix:**
```
For the sentence "The cat sat on the mat":

Attention weights A (each row sums to 1):

        The   cat   sat   on   the   mat
  The  [0.50  0.25  0.10  0.05  0.05  0.05]  ← "The" attends mostly to itself
  cat  [0.40  0.30  0.15  0.05  0.05  0.05]  ← "cat" attends to article "The"
  sat  [0.05  0.40  0.20  0.10  0.05  0.20]  ← "sat" attends to "cat" and "mat"
  on   [0.10  0.10  0.20  0.30  0.10  0.20]  ← "on" attends to preposition context
  the  [0.05  0.05  0.05  0.35  0.40  0.10]  ← "the" attends to "on" (follows it)
  mat  [0.05  0.15  0.25  0.15  0.10  0.30]  ← "mat" attends to "sat" and itself

Darker = higher attention:

        The   cat   sat   on   the   mat
  The  [████  ██    █     ░    ░     ░  ]
  cat  [███   ██    █     ░    ░     ░  ]
  sat  [░     ████  ██    █    ░     ██ ]
  on   [█     █     ██    ███  █     ██ ]
  the  [░     ░     ░     ███  ████  █  ]
  mat  [░     █     ██    █    █     ███]

This shows the model learning syntactic structure:
  "sat" correctly identifies "cat" (subject) and "mat" (prepositional object)
  "the" correctly clusters with "on" (they form "on the mat")
```

**Causal (Masked) Attention for Language Generation:**
```
When GENERATING text (GPT-style), we must NOT look at future tokens.
If we did, the model would "cheat" by seeing the answer while generating it.

Solution: MASK future positions with −∞ before softmax.

For "The cat sat mat":

       The   cat   sat   mat
  The [ 0    −∞    −∞    −∞ ]  ← "The" can only see itself
  cat [ ✓     0    −∞    −∞ ]  ← "cat" sees "The" and itself
  sat [ ✓     ✓     0    −∞ ]  ← "sat" sees "The", "cat", itself
  mat [ ✓     ✓     ✓     0  ]  ← "mat" sees everything before it

After softmax: exp(−∞) = 0, so future positions get exactly zero weight.
The model CAN ONLY use past and present when predicting the next word.

This is how language models learn to predict: "given everything so far,
what comes next?" — without ever cheating by peeking at the answer.
```

---

### 15.3 Multi-Head Attention (MHA) — Multiple Perspectives Simultaneously

> 💡 **A single attention head can only focus on one TYPE of relationship at a time.** But language has many simultaneous relationships. Multi-head attention runs H separate attention operations in parallel, each specialising in different patterns. It's like having H different experts reading the same sentence and each reporting what they noticed.

**Why One Head Is Not Enough:**
```
Consider: "John gave Mary the book because she needed it."

Relationship types simultaneously present:
  Syntactic:    "gave" connects to "John" (subject) and "Mary" (indirect object)
  Semantic:     "she" → "Mary" (pronoun coreference)
  Positional:   "the book" is close to "gave" (noun phrase clustering)
  Causal:       "because" links two clauses
  Dependency:   "needed" connects to "she" and "it"

One attention head can specialise in ONE of these.
8 or 16 heads can capture ALL of these simultaneously.

Visualisation:
  Input: "John gave Mary the book because she needed it"
  
  Head 1 (syntactic):    John ←→ gave ←→ Mary        (subject-verb-object)
  Head 2 (coreference):  she ────────────→ Mary      (pronoun resolution)
  Head 3 (local):        clusters nearby words       (phrase detection)
  Head 4 (long-range):   because ←→ needed           (causal link)
  Head 5 (article):      the → book                  (determiner-noun)
  ...
  
  Final output: concatenate all heads' insights → richer representation!
```

**The Multi-Head Attention Formula:**
```
For each head h = 1, 2, ..., H:
  headₕ = Attention(Q Wₕᴼ, K Wₕᴷ, V Wₕᵛ)
  
  Where:
    Wₕᴼ ∈ ℝ^{d_model × dₖ}   ← "head h's query projection"
    Wₕᴷ ∈ ℝ^{d_model × dₖ}   ← "head h's key projection"
    Wₕᵛ ∈ ℝ^{d_model × dᵥ}   ← "head h's value projection"
    
    dₖ = dᵥ = d_model / H    ← split model dimension equally across heads

Intuition behind projections:
  The SAME input (Q, K, V) is projected into H DIFFERENT subspaces.
  Head 1 projects to a "syntax subspace" → syntax-focused attention.
  Head 2 projects to a "coreference subspace" → pronoun-focused attention.
  Each head learns its own projection matrices during training.

Concatenate all head outputs:
  MultiHead(Q,K,V) = Concat(head₁, head₂, ..., headH) × Wᵒ
  
  Concat produces: (n × H×dᵥ) = (n × d_model)   ← back to original dimension
  Final projection Wᵒ ∈ ℝ^{d_model × d_model}: mixes information across heads

Parameter Count:
  Each head: 3 projection matrices of size (d_model × dₖ)
  All heads combined: 3 × d_model × (H × dₖ) = 3 × d_model²
  Output projection Wᵒ: d_model²
  Total MHA parameters: 4 × d_model²   (same as ONE big attention regardless of H!)
  
  Multi-head attention does NOT add parameters compared to single-head!
  It just uses the same parameters more effectively (richer, multi-perspective attention).
```

**Dimension Walkthrough (Concrete Example):**
```
Say d_model = 512, H = 8 heads, so dₖ = dᵥ = 512/8 = 64.
Sequence length n = 10 (10 words).

Input Q, K, V: each (10 × 512)

For head 1:
  Q₁ = Q × W₁ᴼ     → (10 × 512) × (512 × 64) = (10 × 64)
  K₁ = K × W₁ᴷ     → (10 × 64)
  V₁ = V × W₁ᵛ     → (10 × 64)
  
  Score: Q₁K₁ᵀ/√64  → (10 × 10)   (attention matrix)
  Attn:  softmax(·)   → (10 × 10)
  head₁: Attn × V₁   → (10 × 64)

Repeat for all 8 heads → 8 tensors of shape (10 × 64).
Concatenate: [head₁;...;head₈] → (10 × 512)   ← back to d_model
Final: multiply by Wᵒ → (10 × 512)

Each position is now enriched with information gathered from all other
positions, from 8 different "perspectives" (attention heads).
```

---

### 15.4 The Full Transformer Block

> 💡 **A Transformer block is a standardised module that stacks two operations: attention (to gather information from other positions) and a feedforward network (to process that gathered information). Both are wrapped with residual connections and layer normalisation for stable training.**

**Block Architecture (Modern Pre-LayerNorm Variant):**
```
Input tensor x  (shape: batch × seq_len × d_model)
      │
      ├──────────────────────────────────────┐
      │                                      │  (residual / skip connection)
      ↓                                      │
  LayerNorm(x)                               │
      ↓                                      │
  Multi-Head Self-Attention                  │
      ↓                                      │
      + ←────────────────────────────────────┘
      │
      ├──────────────────────────────────────┐
      │                                      │  (residual / skip connection)
      ↓                                      │
  LayerNorm(x')                              │
      ↓                                      │
  Feed-Forward Network (FFN)                 │
      ↓                                      │
      + ←────────────────────────────────────┘
      │
Output tensor x''  (same shape as input)

Formula:
  x'  = x + MultiHeadAttn(LayerNorm(x))
  x'' = x' + FFN(LayerNorm(x'))
```

**Why Residual Connections Are Critical:**
```
Without residual connections (plain stacking):
  
  x → Layer1 → Layer2 → Layer3 → ... → Layer96 → output
  
  Gradient must flow back through all 96 layers:
  ∂L/∂x ≈ Π₉₆ (local Jacobian) → product of 96 matrices
  
  Even if each Jacobian has norm slightly < 1: 0.99^96 ≈ 0.38 → vanishing!
  Even if slightly > 1: 1.01^96 ≈ 2.60 → and for deeper layers, exploding!

With residual connections:
  
  x → [Layer1] → + → [Layer2] → + → ... → output
       ↑___________↑  ↑___________↑
       skip             skip
  
  Gradient through the residual path:
  ∂L/∂x = ∂L/∂output × (1 + ∂Layer/∂x)
  
  The +1 term ensures gradient magnitude is at LEAST 1, even if the layer
  contributes nothing. Gradients flow freely to the earliest layers.
  
  This is why GPT-3 can have 96 layers and still train stably!
  
  Intuition: the network only needs to LEARN THE CORRECTION (residual),
  not the entire transformation from scratch. Small corrections are easier
  to learn than full transformations.
```

**The Feed-Forward Network (FFN):**
```
FFN(x) = W₂ × g(W₁ × x + b₁) + b₂

  W₁ ∈ ℝ^{d_model × d_ff}   ← expand from 512 to 2048 dimensions
  W₂ ∈ ℝ^{d_ff × d_model}   ← contract back from 2048 to 512

  d_ff = 4 × d_model  (standard "4× expansion")

Activation function g:
  Original Transformer: g = ReLU
  GPT-2:                g = GELU    (smoother, better for transformers)
  LLaMA/Mistral:        g = SwiGLU  (gated — even better)

Why 4× expansion?
  "Expand and mix" pattern: expand to high dimension (allows complex mixing),
  apply non-linearity (compute non-linear combinations), contract back.
  
  Conceptual role: while attention GATHERS information from other positions,
  the FFN PROCESSES that gathered information using position-wise computation.
  Recent research shows FFN layers act as "key-value memory" — they store
  factual associations learned during training.

Parameters per layer:
  MHA:         4 × d_model²                    (e.g., 4 × 512² = 1,048,576)
  FFN:         2 × d_model × d_ff = 8×d_model²  (e.g., 8 × 512² = 2,097,152)
  LayerNorms:  4 × d_model                     (negligible)
  Total/layer: ≈ 12 × d_model²
  
  L layers total: L × 12 × d_model² + V × d_model (embeddings)
  
  GPT-3 check: L=96, d_model=12288
  96 × 12 × 12288² ≈ 173.6 billion ≈ 175B ✓
```

**Layer Normalisation:**
```
LayerNorm(x) normalises ACROSS the feature dimension (not across the batch):

  For a single vector x ∈ ℝ^d:
  μ = (1/d) Σᵢ xᵢ           (mean of all features for this position)
  σ² = (1/d) Σᵢ (xᵢ−μ)²     (variance of all features for this position)
  x̂ᵢ = (xᵢ − μ) / √(σ² + ε)  (normalise each feature)
  yᵢ = γᵢ x̂ᵢ + βᵢ             (scale and shift with LEARNABLE γ, β)

Why LayerNorm (not BatchNorm) for Transformers?
  BatchNorm normalises across the BATCH dimension.
  For sequences: different positions have different "natural" statistics.
  With batch size = 1 (common at inference): BatchNorm breaks (can't compute stats).
  LayerNorm operates PER SAMPLE — works for any batch size, any sequence length.
  → LayerNorm is sequence-length and batch-size agnostic.

Pre-norm (modern) vs Post-norm (original):
  Post-norm (original "Attention is All You Need"):
    x' = LayerNorm(x + Attn(x))
    Problem: gradient scale varies layer to layer → sensitive initialisation needed.
  
  Pre-norm (GPT-2+, most modern models):
    x' = x + Attn(LayerNorm(x))
    Advantage: gradient always has a clean path through the residual stream.
    More stable training for very deep models.
```

---

### 15.5 Positional Encoding — Teaching the Model About Order

> 💡 **Attention itself is "position-blind."** If you shuffle the words in a sentence, the attention operation (being a set operation) would still produce the same result — it has no idea about order! Positional encodings inject information about WHERE each token is in the sequence, so the model can learn to use position when needed.

**The Intuition:**
```
Without positional encoding:
  "Dog bites man" ≡ "Man bites dog" ← same tokens, different meaning!
  Attention would treat these identically → model learns nothing about word order.

With positional encoding:
  Each token gets a unique "position fingerprint" added to its embedding.
  
  "Dog" at position 0 → embedding + PE(0)
  "bites" at position 1 → embedding + PE(1)
  "man" at position 2 → embedding + PE(2)
  
  Now "Dog bites man" and "Man bites dog" have DIFFERENT representations!
  The model can learn: "at position 0 + 'Dog' → subject; at position 2 + 'Dog' → object"
```

**Sinusoidal Positional Encoding (Original — No Parameters):**
```
PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

Where:
  pos = position in sequence (0, 1, 2, ...)
  i   = dimension index (0, 1, ..., d_model/2 − 1)

Different dimensions oscillate at different frequencies:

  Dimension 0 (i=0):   period = 2π × 1      ← changes every ~6 positions
  Dimension 2 (i=1):   period = 2π × 100    ← changes every ~628 positions
  Dimension 512 (i=256): period = 2π × 10000 ← very slow, global position signal

Visualisation for d_model=4:

  Position 0:  [sin(0/1),   cos(0/1),   sin(0/100),   cos(0/100)]
             = [0.000,      1.000,      0.000,        1.000]
  
  Position 1:  [sin(1/1),   cos(1/1),   sin(1/100),   cos(1/100)]
             = [0.841,      0.540,      0.010,        0.9999]
  
  Position 2:  [sin(2/1),   cos(2/1),   sin(2/100),   cos(2/100)]
             = [0.909,     -0.416,      0.020,        0.9998]

Each position gets a UNIQUE fingerprint of sin/cos values!

Like a binary clock where:
  Lowest bit flips every step (fast oscillation, local position info)
  Next bit flips every 2 steps
  Next bit flips every 4 steps
  ...
  Highest bit flips rarely (slow oscillation, global position info)

Key property: PE(pos + k) is a LINEAR FUNCTION of PE(pos):
  [sin(pos+k)/[cos(pos+k)] = rotation matrix × [sin(pos)]/[cos(pos)]
  
  This means the model can learn RELATIVE positions (how far apart two tokens are)
  just from the positional encodings — without needing to see absolute positions.
```

**Rotary Position Embedding (RoPE) — Modern Standard:**
```
Used in: LLaMA, Mistral, GPT-NeoX, Falcon, and most modern open-source LLMs.

Key idea: instead of adding position info to embeddings, ROTATE query and key
vectors by an angle proportional to their position.

For a 2D example (one pair of dimensions):
  f(x, pos) = x × exp(i × pos × θ)   (multiply by complex rotation)
           = [x₁cos(pos×θ) − x₂sin(pos×θ),
              x₁sin(pos×θ) + x₂cos(pos×θ)]

The magical property of RoPE:
  (f(q, m))ᵀ (f(k, n)) = qᵀk × function(m−n)
  
  The inner product between a rotated query and rotated key depends on:
  1. The CONTENT of q and k (the actual word meanings)
  2. The RELATIVE POSITION (m−n) (how far apart they are)
  
  → Purely relative positional encoding!
  → Works for any sequence length (no fixed max position)
  → Extrapolates to longer sequences than seen during training

RoPE vs Sinusoidal:
  Sinusoidal: adds to embeddings BEFORE attention → absolute position only
  RoPE:       applied to Q and K vectors INSIDE attention → true relative position
  
  RoPE empirically trains faster and generalises better → adopted by all major LLMs.

ALiBi (Attention with Linear Biases — MPT, BLOOM):
  Even simpler: subtract a linear penalty for distance from attention scores.
  Score(i,j) = qᵢᵀkⱼ/√dₖ − m × |i−j|
  m: head-specific slope (different heads penalise distance differently)
  
  Advantages: even simpler than RoPE, extrapolates well to long sequences.
```

---

### 15.6 Complete Transformer Architectures

> 💡 **There are three main flavours of Transformer architecture, each best suited for different tasks.** Understanding which to use and why is critical for practical ML engineering.

**Encoder-Only (BERT Family) — Understanding Text:**
```
Architecture:
  Input:  [CLS] + tokens + [SEP]
  Attention: BIDIRECTIONAL — each position attends to ALL positions
  Output: contextualised embeddings for every token

Why bidirectional?
  For UNDERSTANDING (not generating), we WANT to see the full context.
  "Bank" in "river bank" vs "bank account" — you need both sides to disambiguate.
  BERT can see left AND right context simultaneously.

Pre-training:
  Task 1: Masked Language Modelling (MLM)
    Input:  "The [MASK] sat on the [MASK]"
    Target: "cat", "mat"
    15% of tokens masked → predict original → forces learning context
    Masking strategy: 80% [MASK], 10% random word, 10% unchanged
    (This prevents the model from only learning to handle [MASK] tokens)
  
  Task 2: Next Sentence Prediction (NSP)
    Input A: "The cat sat on the mat."
    Input B: "It was a comfortable mat." (50%: true next, 50%: random)
    Predict: IsNextSentence or NotNextSentence
    (Teaches understanding of sentence relationships)

Fine-tuning for downstream tasks:
  
  Classification: add linear head on [CLS] token
    [CLS] → linear → sigmoid/softmax → label
  
  Token classification (NER): add linear head on each token
    [tok₁] [tok₂] [tok₃] → B-PER I-PER O → entity labels
  
  Q&A (SQuAD): predict start and end positions of answer span
    [CLS] question [SEP] context [SEP] → start_logits, end_logits

Best for: classification, NER, Q&A, sentence similarity, information extraction.
NOT for: text generation (bidirectional means no causal masking for generation).

Popular variants:
  RoBERTa:   removes NSP, larger batches, more data → better than BERT
  ALBERT:    parameter sharing across layers → smaller, faster
  DeBERTa:   disentangled attention for position and content → SOTA for many tasks
  ELECTRA:   replaced token detection instead of MLM → 4× more sample efficient
```

**Decoder-Only (GPT Family) — Generating Text:**
```
Architecture:
  Input:  tokens x₁, x₂, ..., xₙ
  Attention: CAUSAL — position t can only attend to positions 1,...,t
  Output: next-token probabilities P(xₙ₊₁ | x₁,...,xₙ)

Why causal?
  Language generation is sequential. Predicting word t should only use
  words 1,...,t-1 (available at generation time). If it could see t+1
  during training, it would learn to cheat → useless for generation.

Pre-training (Causal Language Modelling, CLM):
  L = −(1/T) Σₜ log P(xₜ | x₁, ..., xₜ₋₁)
  
  Just predict the next word, every word, every step.
  With enough data and parameters, this FORCES learning:
    - Grammar (some sequences are grammatical, others aren't)
    - Facts (some facts follow naturally from context)
    - Reasoning (logical conclusions follow from premises)
    - Style (different domains have different patterns)
  
  100% of positions generate gradients (vs 15% for MLM) → 6.7× more efficient!

Autoregressive generation:
  Start: [BOS] "The weather today is"
  Step 1: predict → "sunny" (top probability token)
  Step 2: input "The weather today is sunny", predict → "and"
  Step 3: input + "and", predict → "warm"
  ...continue until [EOS] token or max length reached.

Sampling strategies:
  Greedy:        always pick argmax → deterministic, repetitive, boring
  Temperature:   P(x) ∝ exp(logit/T)
                 T=0: argmax, T=1: standard, T=2: very random
  Top-k:         sample from k most likely tokens → avoids very unlikely tokens
  Top-p (nucleus): sample from tokens totalling probability ≥ p
                   e.g., p=0.9: smallest set of tokens with cumulative prob ≥ 90%
                   Adapts to distribution shape (narrow or wide)
  Min-p:         newer method, simpler and often better than top-p

Best for: text generation, chat, code generation, creative writing.
Modern universal model: GPT-4, LLaMA-3, Claude (all decoder-only or similar).
```

**Encoder-Decoder (T5 Family) — Understanding and Generating:**
```
Architecture:
  Encoder: processes full input with BIDIRECTIONAL attention
           (sees all of the input simultaneously)
  Decoder: generates output CAUSALLY, with two attention mechanisms:
    1. Self-attention (causal): attends to previously generated tokens
    2. Cross-attention: attends to ENCODER output (reads the input!)

Cross-Attention (the key difference from decoder-only):
  Q from decoder: "what do I need from the input for this output token?"
  K, V from encoder: "here's everything in the input"
  
  This enables precise input-conditional generation:
    Translation: "generate German word attending to relevant English words"
    Summarisation: "generate next summary word attending to relevant source sentences"

T5 ("Text-to-Text Transfer Transformer"):
  Unifies ALL tasks as text-to-text:
    Translation:    "translate English to German: The cat sat" → "Die Katze saß"
    Summarisation:  "summarise: [long article]" → "[summary]"
    Q&A:            "question: What is 2+2? context: [math facts]" → "4"
    Classification: "sentiment: This movie is great!" → "positive"
  
  Same model, same loss function (next-token prediction) for everything!
  Just change the input prefix.

Best for: translation, summarisation, structured generation, tasks requiring
          full bidirectional understanding of input + generative output.
```

---

### 15.7 Pre-Training Objectives Compared

```
                MLM (BERT)          CLM (GPT)          Span Corruption (T5)
────────────────────────────────────────────────────────────────────────────
What's masked   15% tokens           0% (predict next)  Contiguous spans
Tokens trained  15% per batch        100% per batch      span tokens only
Context         Bidirectional        Left-only            Both sides for enc.
Parallelism     Full                 Full                 Full
Sample eff.     Lower                Higher               Medium
Architecture    Encoder-only         Decoder-only         Enc-Dec
Best for        Understanding        Generation           Conditional gen.

ELECTRA (most sample-efficient pre-training):
  Small generator (like BERT, runs MLM) creates corrupted tokens.
  Large discriminator predicts: "is this token original or replaced?"
  
  e.g., Input: "The chef [MASK] the sauce."
  Generator: "The chef [tasted] the sauce."
  Discriminator: for EVERY token: original/replaced?
  
  Advantage: ALL tokens generate signal (not just 15%) → 4× more efficient!
  With same compute, ELECTRA outperforms GPT and BERT.
```

---

### 15.8 Chinchilla Scaling Laws — How to Train Efficiently

> 💡 **Scaling laws tell you the optimal way to spend your compute budget.** Before Chinchilla (2022), researchers kept making models BIGGER while using the same amount of data. Chinchilla showed this was wrong — you should scale DATA and MODEL SIZE equally for optimal performance.**

**The Scaling Law Formula:**
```
L(N, D) = E + A/Nᵅ + B/Dᵝ

Where:
  L = cross-entropy loss (lower is better)
  N = number of model parameters (e.g., 7 billion)
  D = number of training tokens (e.g., 1 trillion)
  E ≈ 1.69 = irreducible loss (entropy of natural language — can't do better)
  A, B = constants fit to empirical data
  α ≈ β ≈ 0.5 (approximately!)

The last point (α ≈ β) is the KEY INSIGHT:
  Doubling model size reduces loss by the same amount as doubling data!
  → You should ALWAYS scale both together.

Concrete formula for compute-optimal training:
  Total compute C ≈ 6 × N × D   (each token requires ~6 multiplications per param)
  
  Optimal N*(C) ≈ (C / (6 × 20))^0.5   → N scales as √C
  Optimal D*(C) ≈ 20 × N*(C)            → train on ~20 tokens per parameter
```

**What This Means in Practice:**
```
Pre-Chinchilla thinking (2020-2021):
  "Let's make GPT-3 bigger: 175B parameters trained on 300B tokens"
  
  Chinchilla-optimal for 175B params:
  D* = 20 × 175B = 3,500 billion tokens needed!
  GPT-3 was trained on only 300B → SEVERELY undertrained by 10×!

Post-Chinchilla models:
  Chinchilla (70B params, 1.4T tokens): outperforms GPT-3 despite 2.5× fewer params!
  
  LLaMA-2 (7B params, 2T tokens):
    D* = 20 × 7B = 140B tokens needed (minimum compute-optimal)
    Actually used 2T = 14× more than minimum → overtrained but better inference!
    A smaller model trained on more data = same loss, cheaper to RUN.

  LLaMA-3 (8B params, 15T tokens):
    Using 15T tokens: roughly 1875 tokens per parameter.
    Far beyond compute-optimal training → deliberately overtrained for inference efficiency.

The inference efficiency insight:
  Compute-optimal ≠ Deployment-optimal.
  If you will run the model 1 TRILLION times, it's worth extra training cost
  to make the model smaller (cheaper to run) but equally good.
  
  Rule of thumb: for production deployment, train 10-100× more tokens
  than compute-optimal.

Scaling law visualisation:
  
  Loss
    │╲                            ← larger N, same D (model-only scaling)
    │  ╲                          
    │    ╲                        ← both scale (Chinchilla-optimal)
    │      ╲
    │        ╲___________________  ← same N, more D (data-only scaling)
    └─────────────────────────────
      1B    10B    100B    1T     N (parameters)
  
  The diagonal line (scale both) achieves the best loss per FLOP.
```

---

### 15.9 Efficient Attention Mechanisms

> 💡 **Standard attention has O(n²) memory cost** — for n=100,000 tokens, the attention matrix has 10 billion entries (~40GB). This makes long-context models impossible with standard attention. These techniques make it feasible.

**FlashAttention — The Engineering Breakthrough:**
```
Problem: GPU memory hierarchy matters enormously.
  HBM (GPU RAM):  80 GB, bandwidth ~2 TB/s    ← slow, large
  SRAM (on-chip): 20 MB, bandwidth ~20 TB/s   ← fast, tiny, 10× faster!
  
  Standard attention: writes full n×n matrix to HBM → bottlenecked by HBM bandwidth.
  Even though FLOPs are the same, it's 10× slower than necessary!

FlashAttention insight:
  Never materialise the full n×n attention matrix in HBM.
  Instead: compute attention in TILES that fit in SRAM.
  
  Algorithm:
  Divide Q into blocks Q₁, Q₂, ..., Qₜ  (each fits in SRAM)
  Divide K, V into blocks K₁, K₂, ..., Kₜ
  
  For each Q block:
    Load Q block into SRAM
    For each K, V block:
      Load K, V block into SRAM
      Compute partial attention scores in SRAM
      Update running softmax normalisation (online softmax trick)
      Accumulate partial output
    Write final output back to HBM
  
  Key: the n×n matrix never exists in HBM. Only tiles exist in SRAM.
  
  Result:
  Memory: O(n) instead of O(n²)  → can handle n=100K tokens!
  Speed:  2-4× faster wall-clock  → same math, less data movement
  Quality: EXACTLY equivalent     → not an approximation, mathematically identical!

Online Softmax Trick (the clever math that makes it work):
  
  Standard softmax requires two passes:
  Pass 1: compute max value for numerical stability
  Pass 2: compute exp and normalise
  
  FlashAttention uses online (one-pass) normalisation:
  When processing block j, maintain running:
    m_j = max score seen so far
    ℓ_j = sum of exp(score - m_j) seen so far
  
  When new block comes:
    m_new = max(m_j, new_max)
    ℓ_new = exp(m_j - m_new) × ℓ_j + exp(new_scores - m_new).sum()
  
  This correctly accumulates the softmax normalisation without ever storing
  the full attention matrix.

FlashAttention versions:
  FA-1 (2022): Original. 2-4× speedup.
  FA-2 (2023): Better work partitioning. 2× faster than FA-1.
  FA-3 (2024): H100-specific. Overlaps computation and memory. Another 2×.
  
  → FA-3 is ~8× faster than standard attention at long context.
  → Now standard in ALL serious LLM training and inference.
```

**Multi-Query Attention (MQA) and Grouped-Query Attention (GQA):**
```
Problem: KV Cache memory at inference.

During generation, we cache K and V for all past tokens:
  Memory per token = 2 × L × H × dₖ × sizeof(float16)
  GPT-3 (L=96, H=96, dₖ=128): 2 × 96 × 96 × 128 × 2 bytes ≈ 4.7 MB per token!
  For 10K context: 47 GB — just for ONE user's conversation!

Multi-Query Attention (MQA):
  Instead of separate K, V per head: ALL heads SHARE the same K, V.
  Only Q is separate per head.
  
  Standard MHA: Q_h, K_h, V_h for each h ∈ {1,...,H}
  MQA:          Q_h for each h, but K and V are shared (same for all heads)
  
  KV cache reduction: H× smaller → 96× reduction for GPT-3!
  
  Cost: some quality degradation (heads can't attend to different things as independently).

Grouped-Query Attention (GQA) — the compromise:
  Group H heads into G groups.
  Each group shares ONE K, V.
  
  G=1:  MQA (maximum efficiency, some quality loss)
  G=H:  MHA (full quality, maximum memory)
  G=8:  GQA (intermediate, used by LLaMA-2 70B, Mistral 7B)
  
  Example: LLaMA-2 70B: H=64 heads, G=8 groups
    8 groups × 1 K,V each = 8× memory reduction vs MHA
    Quality: nearly identical to full MHA in practice!

Visualisation:
  
  Standard MHA (H=4):       GQA (H=4, G=2):         MQA (H=4, G=1):
  Q₁ K₁ V₁                  Q₁ ─── K₁ V₁            Q₁ ──┐
  Q₂ K₂ V₂                  Q₂ ─┘                   Q₂   │
  Q₃ K₃ V₃                  Q₃ ─── K₂ V₂            Q₃   ├─ K₁ V₁
  Q₄ K₄ V₄                  Q₄ ─┘                   Q₄ ──┘
  
  4 KV pairs                 2 KV pairs               1 KV pair
  (full quality)             (good compromise)         (cheapest)
```

**Sparse and Local Attention:**
```
Standard attention: every token attends to every other token → O(n²)
Not all attention is useful! Most attention weights are near zero anyway.

Sliding Window / Local Attention:
  Each token only attends to the w nearest tokens (window of size w).
  
  ● ● ● ● ● ● ●
       ↑ this token only sees these 3
  
  Complexity: O(n × w) instead of O(n²)
  
  For w=512 and n=32768: 64× speedup!
  
  Problem: tokens far apart can't directly interact.
  Solution: use some "global" tokens (e.g., [CLS]) that attend to everything.

Longformer combines:
  - Local attention (window) for most tokens
  - Global attention for special tokens ([CLS], task-relevant tokens)
  - O(n × (w + k)) where k = number of global tokens

Sliding Window + Dilated Attention (Mistral):
  Layer 0-15:  sliding window, size 4096
  Remainder:   standard attention
  Some layers use dilated windows (attend to every other token in window)
  
  Allows full n×n attention in some layers for global reasoning while being
  efficient overall.
```

---

### 15.10 Post-Training: Making LLMs Helpful and Safe

> 💡 **Pre-training creates a model that predicts text. Post-training makes it HELPFUL, HONEST, and HARMLESS.** The pre-trained model might complete "How do I make a bomb?" with a news article about explosions — it just predicts likely text. Post-training reshapes its behaviour to be a helpful assistant instead.

**Supervised Fine-Tuning (SFT):**
```
Collect a dataset of (instruction, ideal response) pairs:
  Instruction: "Explain quantum entanglement simply."
  Response:    "Quantum entanglement is when two particles become linked..."
  
  Instruction: "Write a Python function to sort a list."
  Response:    "def sort_list(lst):\n    return sorted(lst)"

Fine-tune the pre-trained LLM on this data:
  Loss = − Σₜ log P(response_token_t | instruction + previous_response_tokens)
  
  Note: loss computed on RESPONSE tokens only (not instruction tokens).
  The model learns to FOLLOW INSTRUCTIONS, not just predict text.

Problem: This requires human expert labellers. Expensive and doesn't scale.
Even with millions of SFT examples, it's hard to cover all cases.
The model also can't EVALUATE which responses are better vs worse.
→ Need a way to give feedback without labelling every possible response.
```

**RLHF — Reinforcement Learning from Human Feedback:**
```
The full pipeline (used for InstructGPT, ChatGPT, Claude):

PHASE 1: SFT (as above)
  Start with pre-trained model, fine-tune on instruction-response pairs.
  This gives a baseline helpful model.

PHASE 2: Reward Model Training
  
  Step 1: Generate multiple responses to the same prompt:
    Prompt: "What is the best programming language to learn first?"
    Response A: "Python is great for beginners because..."
    Response B: "It depends on your goals. For web: JavaScript..."
    Response C: "Machine language is most fundamental..."
  
  Step 2: Human labellers RANK responses: A > B > C
  
  Step 3: Train a reward model r_θ to predict human preference:
    r_θ(prompt, response_A) > r_θ(prompt, response_B)
    
    Loss: L = −log σ(r_θ(x, y_w) − r_θ(x, y_l))
    
    y_w = preferred (winner) response
    y_l = less preferred (loser) response
    σ = sigmoid function
    
    This is the Bradley-Terry model: probability preferred choice wins =
    σ(score_winner − score_loser).
  
  Result: reward model r_θ that scores any response's human preference.

PHASE 3: RL Fine-Tuning (PPO)
  
  Objective: maximise reward while staying close to SFT baseline:
    reward(x, y) = r_θ(x, y) − β × KL(π_θ(y|x) || π_ref(y|x))
    
    r_θ(x, y) = human preference reward from reward model
    β × KL(...)  = penalty for diverging too far from SFT policy
    
    The KL penalty prevents the model from "gaming" the reward model:
    Without KL: model learns to output "The best! Amazing! Perfect!" everywhere
    (the reward model gives high scores to flattering language)
    With KL: model must stay reasonably close to original SFT distribution.
  
  PPO (Proximal Policy Optimisation) update:
    Clip(π_θ/π_old, 1−ε, 1+ε)  ← never update too much in one step
    Prevents catastrophic policy updates.

RLHF Full Pipeline Diagram:
  
  ┌─────────────────────────────────────────────────────────────────┐
  │  PHASE 1: SFT                                                   │
  │  Pre-trained LLM → fine-tune on (prompt, response) pairs        │
  │  → SFT Model                                                    │
  ├─────────────────────────────────────────────────────────────────┤
  │  PHASE 2: Reward Model                                          │
  │  SFT Model generates responses → humans rank them              │
  │  → Train reward model r_θ to predict rankings                  │
  ├─────────────────────────────────────────────────────────────────┤
  │  PHASE 3: RL                                                    │
  │  SFT Model (now "policy") generates responses                   │
  │  Reward model scores responses                                  │
  │  PPO updates policy to increase reward while staying near SFT   │
  │  → Final RLHF Model (helpful, harmless, honest)                 │
  └─────────────────────────────────────────────────────────────────┘
```

**DPO — Direct Preference Optimisation (Simpler Alternative):**
```
RLHF requires training a separate reward model + RL training → complex and unstable.

DPO insight: you can DIRECTLY fine-tune on preference pairs without a reward model!

DPO Loss:
  L_DPO = −E log σ(β log[π_θ(y_w|x)/π_ref(y_w|x)] − β log[π_θ(y_l|x)/π_ref(y_l|x)])

Breaking this down:
  π_θ(y_w|x)/π_ref(y_w|x) = "how much more does the fine-tuned model like the
                               preferred response compared to the reference model?"
  
  π_θ(y_l|x)/π_ref(y_l|x) = "how much more does it like the less-preferred response?"
  
  We want: ratio for y_w > ratio for y_l
  i.e., the model should increase probability of preferred responses relative to
  the base model, and decrease probability of rejected responses.

Why DPO works without a reward model:
  There is a mathematical equivalence: the optimal policy π* for the RLHF objective
  has a closed form: π*(y|x) ∝ π_ref(y|x) × exp(r*(x,y)/β)
  
  Rearranging: r*(x,y) = β × log[π*(y|x)/π_ref(y|x)] + log Z(x)
  
  The REWARD is implicitly defined by the ratio of policies!
  DPO directly optimises the policy without explicitly computing the reward.
  
  Practical advantages:
  ✓ No separate reward model (saves GPU memory and training time)
  ✓ No RL stability issues (standard supervised training)
  ✓ Simpler implementation
  ✓ Competitive quality with RLHF (sometimes better)
  
  Used by: Zephyr, Tulu-2, Mistral-Instruct, and many open-source models.
```

---

### 15.11 Parameter-Efficient Fine-Tuning (PEFT)

> 💡 **Full fine-tuning of a 70B parameter model requires 70B gradients, which means ~560GB of GPU memory just for gradients — far beyond any single GPU.** PEFT methods fine-tune only a tiny fraction of parameters, making adaptation of large models practical on consumer hardware.

**Why Standard Fine-Tuning Is Impractical:**
```
A 7B parameter model in float16:
  Model weights:  7B × 2 bytes = 14 GB
  Gradients:      7B × 2 bytes = 14 GB  (one gradient per weight)
  Optimiser state (Adam): 7B × 8 bytes = 56 GB  (m and v per weight)
  Activations:    variable, ~10-30 GB for reasonable batch size
  
  Total: 14 + 14 + 56 + 20 = ~104 GB minimum
  
  A single A100 GPU has 80 GB → doesn't fit!
  You'd need 2-4 A100s at ~$30,000 each → not accessible.

PEFT goal: fine-tune effectively using <1% of the parameters.
LoRA can fine-tune a 7B model on a single 24GB consumer GPU!
```

**LoRA — Low-Rank Adaptation:**
```
Core insight from linear algebra:
  
  Weight matrices in neural networks are highly REDUNDANT.
  Typical weight matrix W ∈ ℝ^{d×k} has rank d (or k if d>k).
  But during fine-tuning, the CHANGE in weights (ΔW) is often LOW-RANK.
  
  Low-rank = the update lives in a much smaller subspace than the full matrix.

LoRA approach:
  Instead of updating W directly, learn a LOW-RANK approximation of ΔW:
  
  ΔW = B × A
  B ∈ ℝ^{d×r},  A ∈ ℝ^{r×k},  where r << min(d, k)
  
  The effective update is BA, but we only store B and A!
  
  Forward pass:
  y = (W + ΔW)x = Wx + BAx = Wx + B(Ax)
  ↑ original, frozen   ↑ small LoRA update

Parameter savings example:
  W ∈ ℝ^{4096 × 4096}:   original = 16,777,216 parameters
  LoRA with r=16:          B = 4096×16 = 65,536 params
                           A = 16×4096 = 65,536 params
                           LoRA params = 131,072
  
  Savings: 131,072 / 16,777,216 = 0.78%  (train 0.78% of original parameters!)

Initialisation:
  A initialised with random Gaussian
  B initialised to ZERO → ΔW = BA = 0 at start → identical to pretrained model at init
  This ensures we start from the pretrained weights, not random behaviour.

Scaling:
  y = Wx + (α/r) × BAx    (scale by α/r to control adaptation strength)
  α is a hyperparameter (often set = r for simplicity)

Which weights to apply LoRA to?
  Typically: Q, K, V, O projections in attention + FFN weight matrices
  Applying to Q and V only often sufficient (fewer parameters, similar quality)

After training:
  Merge: W' = W + BA   (just add the small update)
  Now inference uses W' with NO extra computation!
  LoRA adds zero inference overhead when merged.

```

**QLoRA — Quantise the Base Model:**
```
Problem: even a frozen 7B model in float16 uses 14 GB — might not fit with LoRA adapters.

QLoRA idea: quantise base model to 4-bit precision, train LoRA adapters in float16.

4-bit NormalFloat (NF4):
  Regular 4-bit quantisation: 16 levels between min and max of each layer.
  NF4: 16 levels chosen so that data is EQUALLY DISTRIBUTED across quantisation buckets.
  
  For normally distributed weights: NF4 minimises quantisation error.
  
  NF4 values: [-1, -0.693, -0.509, -0.395, -0.308, -0.233, -0.164, -0.083,
                0,  0.083,  0.164,  0.233,  0.308,  0.395,  0.509,  0.693]  (normalised)

Double Quantisation:
  Even the scaling factors (used to dequantise) take memory.
  QLoRA quantises THESE quantisation constants too.
  Saves another ~0.5 bits per parameter.

Memory savings:
  Standard fine-tuning of 65B model: ~780 GB → needs 10+ A100s
  QLoRA of 65B model: ~48 GB → fits on a single 80GB A100!
  QLoRA of 7B model: ~6 GB → fits on a consumer RTX 3080!

Quality: QLoRA achieves near-identical results to full 16-bit fine-tuning.
  The 4-bit base model introduces quantisation error, but LoRA adapters
  (in 16-bit) compensate. Net result: ~0.1% quality loss vs full fine-tuning.
```

**Other PEFT Methods:**
```
Prefix Tuning:
  Add K trainable "prefix" tokens to K and V at EACH attention layer.
  Input: [prefix₁, prefix₂, ..., prefix_L, x₁, x₂, ..., xₙ]
  
  The prefix tokens act as "soft prompts" that steer the model's behaviour.
  Only prefix parameters (~0.1% of model) are trained.
  
  Problem: prefix tokens "use up" context window space.
  For a 4096-token context with 100-token prefix: only 3996 tokens for content.

Prompt Tuning:
  Even simpler: only add trainable tokens to the INPUT EMBEDDING (not every layer).
  Fewer parameters than prefix tuning.
  Only effective for very large models (>10B params).

Adapter Layers:
  Insert small bottleneck MLP layers BETWEEN existing layers:
    FFN → Adapter → Next Layer
  
  Adapter architecture:
    x → Linear(d_model → r) → GELU → Linear(r → d_model) + x
    (project down, activate, project up, add residual)
  
  r = bottleneck dimension (e.g., 64)
  Parameters ≈ 2 × d_model × r × L layers ≈ 2-5% of model
  Inference adds small overhead (adapter layers must be run).

LoRA Comparison:
  ┌───────────────────────────────────────────────────────────────┐
  │ Method         │ Params │ Inference │ Best For              │
  ├───────────────────────────────────────────────────────────────┤
  │ Full FT        │ 100%   │ None      │ Unlimited compute      │
  │ Adapters       │ 2-5%   │ Small     │ Many task variants     │
  │ Prefix Tuning  │ 0.1%   │ None*     │ Task prompting         │
  │ LoRA           │ 0.1-1% │ None†     │ General fine-tuning   │
  │ QLoRA          │ 0.1-1% │ None†     │ Consumer GPU tuning   │
  │ Prompt Tuning  │ <0.01% │ None*     │ Very large models only │
  └───────────────────────────────────────────────────────────────┘
  * Small overhead from prefix tokens in context
  † Zero overhead after merging LoRA into base weights
```

---

### 15.12 Inference Optimisation & The KV Cache

> 💡 **Inference is where LLMs are used in production, and it's often the bottleneck.** Generating text token-by-token is inherently slow. These techniques make it 10-100× faster without changing the model's output quality.

**The KV Cache — Fundamental Optimisation:**
```
During autoregressive generation, at each step we need attention over ALL past tokens.

Naive approach: recompute K, V for all past tokens at every step.
  At step t, recompute K, V for tokens 1, 2, ..., t-1 → O(t²) total computation!

KV Cache: STORE the computed K, V and REUSE them.
  
  Step 1: generate token 1
    Compute Q₁, K₁, V₁ from token 1
    Store K₁, V₁ in cache
    Attention over {K₁, V₁}
    Output: next token prediction
  
  Step 2: generate token 2
    Input: new token 2 only
    Compute Q₂, K₂, V₂ from token 2
    Append K₂, V₂ to cache: {K₁, K₂}, {V₁, V₂}
    Attention: Q₂ attends over {K₁, K₂}, {V₁, V₂}
    Output: next token prediction
    (No need to recompute K₁, V₁ — they're cached!)
  
  At step t: only compute Q, K, V for the NEW token. O(t) total computation!

KV Cache memory formula:
  Memory = 2 × L × H × dₖ × n_tokens × bytes_per_float
  
  Example: LLaMA-2-7B (L=32, H=32, dₖ=128, bfloat16)
  Per token: 2 × 32 × 32 × 128 × 2 bytes = 524,288 bytes = 512 KB
  For 4096 tokens: 512 KB × 4096 = 2 GB per user session!
  
  For 100 concurrent users: 200 GB just for KV caches → expensive!

GQA reduces KV cache proportionally:
  H=32 heads, G=8 groups: KV cache reduces 4×
  Per token: 512 KB / 4 = 128 KB (much more manageable!)
```

**Speculative Decoding — 2-3× Faster Generation:**
```
Problem: autoregressive generation generates ONE token per forward pass.
Even with batching, you need T forward passes for T tokens.

Key insight:
  A small "draft" model is much faster than the large target model.
  If the draft model often agrees with the target, use it to generate
  multiple tokens at once, then verify with the target model in PARALLEL.

Algorithm:
  1. Use small model (e.g., 7B) to generate k=5 candidate tokens rapidly.
     "The weather today is" → draft: ["sunny", "and", "warm", "with", "clouds"]
  
  2. Run large model (e.g., 70B) on ALL k tokens simultaneously (one forward pass!).
     70B model computes: P(sunny|context), P(and|context+sunny), ...
  
  3. For each token, compare draft vs target probabilities:
     If target probability ≥ draft probability: ACCEPT the token.
     Otherwise: REJECT and sample from a corrected distribution. Stop here.
  
  4. For every accepted token, ONE large model step was "saved."
     On average: accept 3-4 tokens per large model step → 3-4× speedup!

Why this is exact (not approximate):
  The rejection sampling step ensures the final output distribution is
  IDENTICAL to just using the large model alone.
  Speculative decoding is a lossless acceleration — same quality, faster.

Requirements: draft model must match target model's vocabulary and produce
good approximations (usually a smaller version of the same model family).

Typical speedup: 2-3× on GPU for auto-regressive generation.
```

**Quantisation for Inference:**
```
Training uses float32 or bfloat16 (16-bit) for numerical stability.
Inference can use lower precision without much quality loss.

INT8 Quantisation:
  Store weights as 8-bit integers (instead of 16-bit floats).
  Memory: 2× reduction.  Speed: 1.5-2× on hardware with INT8 support.
  Quality: typically < 0.5% accuracy drop.
  
  Method: for each weight tensor, compute scale = max(|W|) / 127
          W_int8 = round(W / scale)
          At inference: W_float = W_int8 × scale  (dequantise on the fly)
  
  LLM.int8() (bitsandbytes): uses mixed-precision:
    Outlier channels (>0.5% of weights): kept in float16 (these matter!)
    Non-outlier channels: quantised to int8
    Result: near-lossless int8 quantisation for LLMs.

INT4 Quantisation:
  4-bit weights: 4× memory reduction vs float16.
  Requires careful quantisation or models degrade significantly.
  
  GPTQ (most popular):
    Layer-by-layer quantisation that minimises reconstruction error.
    For each layer: find INT4 weights minimising ||W x − W_q x||²
    Uses second-order (Hessian) information for precise quantisation.
    
  AWQ (Activation-aware Weight Quantisation):
    Insight: 1% of weights ("salient") are disproportionately important.
    Protect these weights (keep in higher precision).
    Quantise the remaining 99% aggressively.
    Outperforms GPTQ with less computational overhead.

Quantisation quality comparison:
  float16 (baseline):  MMLU score = 70.0
  INT8 (LLM.int8()):   MMLU score = 69.8   (-0.2%)
  INT4 (GPTQ):         MMLU score = 68.5   (-1.5%)
  INT4 (AWQ):          MMLU score = 69.5   (-0.5%)   ← best INT4 method
  INT2:                MMLU score = 45.0   (-35%)    ← too lossy for most uses
```

---

### 15.13 Notable LLM Architectures Comparison

```
Model          │ Params │ Context │ Key Innovations                    │ Open?
───────────────────────────────────────────────────────────────────────────────
GPT-3          │  175B  │   4K    │ Scale, few-shot learning           │  No
ChatGPT        │  ~20B? │   4K    │ RLHF, instruction following        │  No
GPT-4          │  ~1.8T?│ 128K   │ Multimodal, MoE (rumoured)         │  No
Claude-3.5     │   ?    │ 200K   │ Constitutional AI, long context     │  No
LLaMA-2        │ 7-70B  │   4K   │ GQA, open weights, chat model      │  YES
LLaMA-3        │ 8-405B │ 128K   │ GQA, improved training, 15T tokens  │  YES
Mistral-7B     │   7B   │  32K   │ Sliding window attention, GQA       │  YES
Mixtral-8×7B   │  47B   │  32K   │ Sparse MoE, 8 experts, 2 active    │  YES
Gemma-2        │ 9-27B  │   8K   │ Alternating global/local attention  │  YES
Phi-3          │ 3.8B   │ 128K   │ Textbook data quality, tiny model   │  YES
Qwen-2         │ 7-72B  │ 128K   │ Multilingual, strong math/code      │  YES

Mixture of Experts (MoE) — how Mixtral works:
  
  Standard FFN: ALL neurons compute for every token.
  MoE FFN: route each token to just 2 of 8 expert FFNs.
  
  Architecture:
  x → Router → softmax → [select top-2 experts] → weighted average → output
  
  Router: small linear layer that produces 8 scores per token.
  Top-2 selection: pick the 2 experts with highest scores.
  Weighted average: output = g₁ × Expert₁(x) + g₂ × Expert₂(x)
  
  Mixtral 8×7B:
  Total params: 8 × 7B = 47B (one set of non-MoE weights + 8 expert FFNs)
  Active params per token: ~12B (only 2 experts active)
  
  Quality: comparable to LLaMA-2-70B (which has ALL 70B active)
  Inference cost: comparable to 12B model
  → MoE gives "70B quality at 12B cost" at inference time!
  
  Load balancing loss:
  Without regularisation, all tokens route to 1-2 experts (collapse).
  Add auxiliary loss: Σᵢ fᵢ × Pᵢ where fᵢ = fraction using expert i, Pᵢ = mean probability
  This penalises unequal load and encourages all experts to be used.
```

---

### 15.14 Vision Transformers (ViT)

> 💡 **ViT applies the Transformer architecture to images by treating image PATCHES as tokens.** Just as NLP Transformers process sequences of word tokens, ViT processes sequences of image patches. This allows image models to leverage the same architecture and pre-training techniques as language models.

**How ViT Works:**
```
Step 1: Divide image into patches.
  Input image: 224×224×3 (RGB)
  Patch size P: 16×16 (standard)
  Number of patches: N = (224/16)² = 196 patches
  
  Each patch: 16×16×3 = 768 pixel values → flatten to a 768-dim vector.
  
  Visualisation:
  ┌──────────────────────────────────────────┐
  │ ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┤
  │ │P₁│P₂│P₃│P₄│P₅│P₆│P₇│P₈│P₉│P₁₀│..│..│
  │ ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
  │ │P₁₄│ ...                          │
  │ └─────────────────────────────────────┘
  Each cell = one 16×16 patch → one "token"
  224×224 image → 196 patches → 196 tokens

Step 2: Embed each patch.
  Flatten patch: 16×16×3 = 768 numbers
  Linear projection: 768 → d_model (e.g., 768)
  E_patch = Flatten(patch) × W_E   (learned embedding matrix)

Step 3: Add [CLS] token and position embeddings.
  [CLS] token: a learnable embedding prepended to the sequence.
  Its final representation is used for classification (like BERT's [CLS]).
  
  Input to Transformer: [[CLS], patch₁, patch₂, ..., patch₁₉₆]
                         + positional embeddings (learnable, one per position)
  
  Total sequence length: 1 + 196 = 197 tokens.

Step 4: Run through L Transformer encoder layers.
  Standard bidirectional attention (no causal mask — for classification, not generation).
  All patches attend to all other patches + [CLS] token.
  
  This is where global reasoning happens:
  Patch in top-left can directly attend to patch in bottom-right.
  (CNN would need many layers for this — ViT does it in every layer!)

Step 5: Classification.
  Output of [CLS] token → Linear → class probabilities.
  
  Why [CLS] for classification?
  [CLS] can attend to ALL 196 patches → naturally aggregates global info.
  Its final representation is a "summary" of the entire image.

Computational complexity:
  Number of patches n = (H×W/P²)
  Attention: O(n²) = O((H×W)²/P⁴)
  
  For P=16, 224×224: n=196, O(196²) = O(38,416) — manageable.
  For P=8 (finer): n=784, O(784²) = O(614,656) — 16× more compute!
  
  ViT is more computationally expensive than CNN at the same resolution,
  but scales better with data and model size.

ViT variants:
  ViT-B/16: 12 layers, d_model=768,  H=12, 86M params
  ViT-L/16: 24 layers, d_model=1024, H=16, 307M params
  ViT-H/14: 32 layers, d_model=1280, H=16, 632M params (patch size 14!)
  
  Naming: B/16 = Base model, patch size 16.
```

**CLIP — Connecting Vision and Language:**
```
CLIP (Contrastive Language-Image Pre-training) trains a vision encoder
and a text encoder to align their representations.

Training data: 400 million (image, caption) pairs from the internet.

Training objective: contrastive learning.
  
  For a batch of N (image, text) pairs:
  - Compute image embeddings: v₁, v₂, ..., vₙ (using ViT)
  - Compute text embeddings: t₁, t₂, ..., tₙ (using Transformer)
  - Compute similarity matrix: Sᵢⱼ = τ × vᵢ · tⱼ  (τ = learned temperature)
  
  Loss: make diagonal entries (matched pairs) have high similarity,
        off-diagonal entries (mismatched) have low similarity.
  
  L = (1/N) Σᵢ [−log softmax(Sᵢᵢ over rows) − log softmax(Sᵢᵢ over columns)] / 2
  
  Visualisation (4×4 similarity matrix):
        text₁  text₂  text₃  text₄
  img₁ [████    ░░     ░░     ░░  ]  ← (img₁, text₁) matched → HIGH
  img₂ [ ░░    ████    ░░     ░░  ]  ← (img₂, text₂) matched → HIGH
  img₃ [ ░░     ░░    ████    ░░  ]  ← all matched pairs: diagonal
  img₄ [ ░░     ░░     ░░    ████ ]  ← off-diagonal: push to LOW

After training:
  For ANY text description, CLIP can find matching images (zero-shot retrieval).
  For ANY image, CLIP can find matching text descriptions.
  
  Zero-shot classification:
  1. Create text embeddings for class names:
     "a photo of a cat", "a photo of a dog", "a photo of a car"
  2. Compute image embedding for test image.
  3. Find the closest text embedding → predicted class!
  
  CLIP achieves ImageNet accuracy comparable to supervised ResNet-50,
  trained on internet data with NO manual ImageNet labels!
  
  This is the foundation of DALL-E, Stable Diffusion (use CLIP text encoder),
  GPT-4V, LLaVA, and all major vision-language models.
```

---

## 16. Natural Language Processing (NLP)

> 💡 **NLP is the field of teaching machines to understand, interpret, and generate human language.** It's one of the oldest and richest areas of AI, ranging from simple rule-based systems to modern LLMs. This section covers both the classical foundations AND modern neural approaches — understanding the history makes the present much clearer.

---

### 16.1 Text Preprocessing — Turning Raw Text Into Numbers

> 💡 **Computers don't understand words — they understand numbers.** Before any NLP algorithm can run, text must be converted into numerical representations. The pipeline: raw text → tokenisation → numericisation → optional embedding → model input.

**The Full Text Preprocessing Pipeline:**
```
Raw text: "Natural Language Processing is amazing! It's not easy, though..."

Step 1: Lowercasing
  → "natural language processing is amazing! it's not easy, though..."
  (reduces vocabulary size: "The" and "the" become the same)

Step 2: Punctuation handling
  → "natural language processing is amazing it's not easy though"
  (depends on task: for sentiment, "!" is informative → maybe keep it)

Step 3: Tokenisation (split into tokens)
  Word tokenisation: ["natural", "language", "processing", "is", "amazing",
                      "it's", "not", "easy", "though"]
  
  Problem: "it's" — is this one token or two ("it" + "'s")?
  Problem: "don't", "can't", "shouldn't" — contractions
  Problem: "New York" — should be one token (proper noun)
  
  Better: subword tokenisation (see below)

Step 4: Stop word removal (optional, task-dependent)
  Remove: ["is", "it's", "not", "though"] (common words with low information)
  Keep: ["natural", "language", "processing", "amazing", "easy"]
  
  ⚠️ Warning: for sentiment analysis, "not" is CRITICAL!
  "Not amazing" becomes "amazing" if you remove stop words → wrong!

Step 5: Stemming / Lemmatisation
  Stemming (crude, rule-based):
    "processing" → "process"
    "amazing"    → "amaz"     ← sometimes breaks words
    "easier"     → "easi"     ← not a real word!
  
  Lemmatisation (dictionary-based, slower but better):
    "processing" → "process"
    "amazing"    → "amazing"
    "easier"     → "easy"
    "was"        → "be"       ← knows the lemma of irregular verbs

Step 6: Numericisation
  Vocabulary: {0: "<UNK>", 1: "natural", 2: "language", 3: "processing", ...}
  "natural language processing" → [1, 2, 3]
```

**Subword Tokenisation — The Modern Standard:**
```
Problem with word tokenisation:
  - Unknown words: "transformerised" → not in vocabulary → <UNK> (useless!)
  - Huge vocabulary: English has ~170,000 words → 170,000-dim one-hot vector
  - Languages like German create compound words: "Donaudampfschifffahrtsgesellschaft"

Problem with character tokenisation:
  - Too granular: "cat" → ['c', 'a', 't'] → sequences become very long
  - Characters alone carry little meaning: 'c' doesn't mean much

Subword tokenisation — the sweet spot:
  Break rare words into meaningful subunits, keep common words as-is.

BPE (Byte-Pair Encoding) — used by GPT-2, GPT-3, GPT-4, LLaMA:
  
  Algorithm:
  1. Start with character vocabulary + special tokens
  2. Find the most frequently occurring pair of adjacent tokens
  3. Merge that pair into a new token
  4. Repeat until vocabulary size reaches target (e.g., 50,000)
  
  Example training process:
  Start: corpus = "low lower lowest"
  Character tokens: {l, o, w, e, r, s, t}
  
  Iteration 1: most frequent pair = (l,o) → 4 occurrences → merge to "lo"
  Now: "lo w lo w er lo w est"
  
  Iteration 2: most frequent pair = (lo,w) → 3 occurrences → merge to "low"
  Now: "low low er low est"
  
  Iteration 3: "er" → merge (appear together often)
  And so on...
  
  Final vocabulary includes: low, lower, lowest, er, est, etc.
  
  Tokenising "lowest" with learned vocab:
  "lowest" → ["low", "est"] → IDs [423, 91]
  
  Tokenising "superlowest" (new word, not in vocab):
  → ["super", "low", "est"] → IDs [28, 423, 91]
  NEVER gives <UNK>! Always can represent any string.

WordPiece — used by BERT:
  Similar to BPE, but maximises likelihood of training data under the tokeniser model.
  Subwords that aren't at the start of a word get "##" prefix:
  "amazing" → ["amazing"]
  "processing" → ["process", "##ing"]
  "transformerised" → ["transform", "##er", "##ised"]

SentencePiece — used by LLaMA, T5, many multilingual models:
  Works directly on raw Unicode characters (no pre-tokenisation required).
  Language-agnostic: works equally well for Chinese, Arabic, English.
  Handles spaces as explicit tokens (▁ represents a space before a word):
  "natural language" → ["▁natural", "▁language"]

Comparison:
  Method        │ Vocab Size │ OOV? │ Used By
  ──────────────────────────────────────────────
  Word          │ ~170K+     │ YES  │ older systems
  Character     │ ~256       │ NO   │ some older NLP
  BPE           │ 32K-100K   │ NO   │ GPT family
  WordPiece     │ 30K        │ NO   │ BERT, ELECTRA
  SentencePiece │ 32K-256K   │ NO   │ LLaMA, T5, mBERT
```

---

### 16.2 Classical NLP Representations

> 💡 **Before neural networks dominated NLP, classical methods achieved impressive results.** Understanding them is essential — they're still useful in production, fast, interpretable, and sometimes competitive with neural methods for specific tasks.

**Bag of Words (BoW):**
```
The simplest possible text representation: just count the words!

Document 1: "the cat sat on the mat"
Document 2: "the dog sat on the log"

Vocabulary: {cat, dog, log, mat, on, sat, the}

BoW representation:
         cat  dog  log  mat  on  sat  the
  Doc 1: [ 1    0    0    1    1   1    2  ]
  Doc 2: [ 0    1    1    0    1   1    2  ]

This is a VECTOR of word counts.

✓ Simple, fast, interpretable
✓ Works surprisingly well for topic classification
✗ Loses word ORDER completely:
  "The cat killed the dog" ≡ "The dog killed the cat"
  → Same BoW representation!
✗ High-dimensional and sparse (mostly zeros)
✗ No notion of word similarity ("car" and "automobile" are unrelated)
✗ Common words dominate ("the", "and" have very high counts)
```

**TF-IDF (Term Frequency-Inverse Document Frequency):**
```
Problem with raw word counts: "the" appears in every document → uninformative.
We want to upweight DISTINCTIVE words, not universal ones.

TF (Term Frequency):
  TF(t, d) = count(t in d) / total_words_in_d
  = what fraction of document d is made up of term t?

IDF (Inverse Document Frequency):
  IDF(t) = log(N / df(t))
  
  N = total number of documents
  df(t) = number of documents containing term t
  
  "the" appears in ALL N documents: IDF = log(N/N) = log(1) = 0
  "quantum" appears in 10 of 1,000,000 docs: IDF = log(100,000) = 11.5
  
  High IDF = rare across documents = distinctive = informative!

TF-IDF:
  TFIDF(t, d) = TF(t, d) × IDF(t)
  
  Example: document about quantum computing, corpus of 1M general documents.
    "quantum": TF=0.02, IDF=11.5 → TFIDF = 0.23 (HIGH → important)
    "the":     TF=0.08, IDF=0.0  → TFIDF = 0.00 (LOW → common word, ignored)
    "computer": TF=0.01, IDF=5.2 → TFIDF = 0.05 (medium)

TF-IDF vectors:
  Each document → vector of TF-IDF scores, one per vocabulary term.
  Similar documents have similar TF-IDF vectors (same distinctive words).
  
  Application: cosine similarity between TF-IDF vectors = document similarity!
  cosine_sim(doc1, doc2) = (tfidf₁ · tfidf₂) / (||tfidf₁|| × ||tfidf₂||)
  
  ✓ Better than raw counts (filters out stop words automatically)
  ✓ Used in: search engines (BM25 is TF-IDF variant), document retrieval
  ✗ Still bag-of-words (no word order)
  ✗ No word similarity (vocabulary must be identical for comparison)
```

**N-Grams — Capturing Word Order:**
```
Bigrams (n=2): consecutive pairs of words.
  "the cat sat on the mat"
  Bigrams: (the,cat), (cat,sat), (sat,on), (on,the), (the,mat)

Trigrams (n=3): consecutive triplets.
  Trigrams: (the,cat,sat), (cat,sat,on), (sat,on,the), (on,the,mat)

BoW of bigrams captures SOME word order:
  "cat ate" and "ate cat" have different bigram representations!
  
  Doc: "cats eat dogs" → bigrams: {cats_eat: 1, eat_dogs: 1}
  Doc: "dogs eat cats" → bigrams: {dogs_eat: 1, eat_cats: 1}
  Completely different representations → order preserved!

Trade-off:
  Unigrams: vocabulary V → V features
  Bigrams:  vocabulary V → V² potential bigrams (but sparse)
  Trigrams: V³ potential (even more sparse)
  
  N-gram models work well for:
    Spell checking (does this bigram appear often in English?)
    Autocomplete (given "I want to", what word comes next?)
    Language model perplexity (how surprising is this sequence?)

N-gram Language Model:
  P("cat sat on mat") = P(cat) × P(sat|cat) × P(on|sat) × P(mat|on)
  
  Bigram probability: P(wₜ | wₜ₋₁) = count(wₜ₋₁, wₜ) / count(wₜ₋₁)
  
  Problem: data sparsity. "cat sat" might appear in training, but "purple sat" never does.
  → P("purple sat on mat") = 0 (zero probability for valid sentence)
  
  Fix: smoothing (add-k, Kneser-Ney, Good-Turing)
  Kneser-Ney smoothing (best for n-gram LMs):
    P_KN(w|context) = max(count(context, w) − d, 0) / count(context)
                     + λ(context) × P_KN(w | shorter_context)
    d = discount (e.g., 0.75)
    Recursively backs off to shorter contexts when longer context has no data.
```

---

### 16.3 Word Embeddings — Distributed Representations

> 💡 **Word embeddings are the revolution that made modern NLP possible.** Instead of treating each word as an isolated symbol (one-hot encoding), embeddings represent each word as a dense vector where SIMILAR words have SIMILAR vectors. "King" and "queen" are nearby. "Dog" and "cat" are nearby. "King − man + woman ≈ queen" — the famous analogy!**

**The Problem with One-Hot Encoding:**
```
With vocabulary V = {king, queen, man, woman, cat, dog}, one-hot:

  king:  [1, 0, 0, 0, 0, 0]
  queen: [0, 1, 0, 0, 0, 0]
  man:   [0, 0, 1, 0, 0, 0]
  woman: [0, 0, 0, 1, 0, 0]
  cat:   [0, 0, 0, 0, 1, 0]
  dog:   [0, 0, 0, 0, 0, 1]

Distance between ANY two different words = √2 (identical distance).
"king" and "queen" are just as "different" as "king" and "cat".
The model has NO information about word relationships from the encoding.

Word embeddings:
  king:  [0.92, 0.15, 0.34, 0.88, ...]   (dense, real-valued)
  queen: [0.89, 0.17, 0.28, 0.92, ...]   (very similar to "king"!)
  man:   [0.88, 0.18, -0.16, -0.35, ...] (similar to "king", different in gender)
  woman: [0.86, 0.20, -0.22, 0.40, ...]  (similar to "queen", different in gender)
  cat:   [0.12, 0.88, 0.75, -0.22, ...]  (very different from royalty!)
  dog:   [0.14, 0.85, 0.72, -0.18, ...]  (similar to "cat"!)

cosine_similarity(king, queen) = 0.97   ← very similar!
cosine_similarity(king, cat)   = 0.23   ← very different!
cosine_similarity(cat, dog)    = 0.94   ← similar animals!
```

**Word2Vec — Learning Embeddings from Context:**
```
Core hypothesis (the Distributional Hypothesis):
  "You shall know a word by the company it keeps." — J.R. Firth, 1957
  
  Words that appear in SIMILAR CONTEXTS have SIMILAR MEANINGS.
  "I bought a ____." → likely: car, house, ticket, coffee
  "I petted the ____." → likely: cat, dog, hamster
  
  Words in similar contexts → should have similar embeddings.

Two architectures:

CBOW (Continuous Bag Of Words):
  Given the context words, predict the TARGET word.
  
  Context: ["The", "cat", "on", "mat"]
  Target: "sat"  (the middle word)
  
  [The]  →   ↘
  [cat]  →    → Average → Linear → Softmax → P("sat")
  [on]   →   ↗           ↑ this is the embedding of context words
  [mat]  →  ↗
  
  Training: maximise P(target | context)

Skip-gram:
  Given the TARGET word, predict the CONTEXT words.
  (Opposite direction — often works better for rare words)
  
  Target: "sat"
  Predict: P("The"|"sat"), P("cat"|"sat"), P("on"|"sat"), P("mat"|"sat")
  
  [sat] → Linear → Softmax → P(each context word)

Training objective (Skip-gram):
  Maximise: Σ Σ log P(wₒ | wᵢ)    (over all words and their context windows)
  
  P(wₒ | wᵢ) = exp(vₒ · vᵢ) / Σⱼ exp(vⱼ · vᵢ)
  
  The denominator sums over ALL vocabulary words → expensive!
  
  Fix: Negative Sampling
  Instead of normalising over all words, just distinguish real context
  from randomly sampled "negative" (non-context) words.
  
  L = log σ(vₒ · vᵢ) + Σₖ E[log σ(−vₖ · vᵢ)]
      ↑                   ↑
  real context           k negative samples
  (push similar)         (push apart)
  
  k = 5-20 negative samples per positive → ~10-20× cheaper than full softmax!

The Famous Analogy:
  After training, the embedding space has remarkable geometric structure:
  
  king − man + woman ≈ queen
  
  This means:
  vec("king") − vec("man") = direction vector for "royalty" (gender-neutral)
  Adding vec("woman") to this: "royalty" + "female" = "queen"!
  
  Other analogies that work:
  Paris − France + Germany ≈ Berlin    (capital relationship)
  walked − walk + run ≈ ran           (past tense relationship)
  biggest − big + cold ≈ coldest      (superlative relationship)
  
  The embedding space has LEARNED to encode semantic relationships
  as GEOMETRIC directions — without any explicit linguistic knowledge!
  Pure statistics from co-occurrence patterns.
```

**GloVe (Global Vectors for Word Representation):**
```
Word2Vec: local context windows → learns from individual (word, context) pairs.
GloVe: global co-occurrence statistics → learns from the entire corpus matrix.

Build co-occurrence matrix X:
  Xᵢⱼ = number of times word i and word j appear within a window together.
  
  For corpus "the cat sat on the mat the cat ate":
        the  cat  sat  on   mat  ate
  the  [ 0    4    1    1    1    1 ]
  cat  [ 4    0    1    0    0    1 ]
  sat  [ 1    1    0    1    0    0 ]
  ...

GloVe objective:
  Learn vectors u and v such that: uᵢ · vⱼ ≈ log(Xᵢⱼ)
  
  Weighted objective (weight rare co-occurrences less):
  L = Σᵢⱼ f(Xᵢⱼ) × (uᵢ · vⱼ + bᵢ + bⱼ − log Xᵢⱼ)²
  
  f(x) = (x/x_max)^α if x < x_max, else 1   (downweight very common pairs)
  
  Final embeddings: average word and context vectors: w = (u + v) / 2

GloVe vs Word2Vec:
  ✓ GloVe uses global statistics → generally better on analogy tasks
  ✓ Faster to train (one pass over co-occurrence matrix)
  Word2Vec ✓ Better on semantic similarity
  Both produce high-quality 300-dimensional embeddings.
  In practice, pre-trained GloVe (840B token, 300d) is a common baseline.
```

**FastText — Handling Rare Words and Morphology:**
```
Problem with Word2Vec/GloVe: "transformerised" is not in the vocabulary → <UNK>

FastText (Bojanowski et al., 2017):
  Represent each word as a BAG OF CHARACTER N-GRAMS.
  
  "where" with n=3: <wh, whe, her, ere, re>  (<  = word boundary)
  
  Word embedding = sum of character n-gram embeddings:
  embed("where") = embed(<wh) + embed(whe) + embed(her) + embed(ere) + embed(re>)
  
  Benefits:
  ✓ Rare/unseen words: compose from character n-grams (always available)
  ✓ Morphological generalisation: "running", "runner", "runs" share n-grams "run"
  ✓ Subword information: "un-" prefix learned across "unfair", "unkind", "untrue"
  ✓ Works for agglutinative languages (Turkish, Finnish, Arabic) where many word forms
  
  "transformerised" (unseen):
  = embed(<tr) + embed(tra) + ... + embed(ise) + embed(sed) + ...
  Approximate meaningful embedding from known character n-grams!
```

---

### 16.4 Sequence Models for NLP — Before and During Transformers

**Named Entity Recognition (NER) — Labelling Each Token:**
```
Task: identify and classify entities in text.
  "Apple was founded by Steve Jobs in Cupertino in 1976."
  Apple [ORG] was founded by Steve Jobs [PER] in Cupertino [LOC] in 1976 [DATE].

BIO Tagging Scheme:
  B-XXX: Beginning of entity of type XXX
  I-XXX: Inside (continuation of) entity type XXX
  O:     Outside (not an entity)
  
  Token:  Apple  was  founded  by  Steve  Jobs  in  Cupertino  in  1976
  Label:  B-ORG   O      O     O   B-PER  I-PER  O   B-LOC     O   B-DATE

Classical approach: CRF (Conditional Random Field)
  
  CRF models the probability of the ENTIRE label sequence given input:
  P(y | x) = (1/Z(x)) × exp(Σᵢ Σₖ λₖ fₖ(yᵢ₋₁, yᵢ, x, i))
  
  Features fₖ: "current word is capitalised", "next word is a city name",
               "previous label was B-PER" (transition feature), etc.
  
  The crucial difference from simple per-token classification:
  CRF JOINTLY predicts all labels, respecting label dependencies.
  
  E.g., "I-PER" cannot follow "B-ORG" (can't continue a person entity
  that started as an organisation entity).
  
  Viterbi algorithm finds the globally best label sequence in O(n|Y|²).

Modern approach: BiLSTM-CRF or BERT + CRF
  BiLSTM: reads sequence forward AND backward → rich contextual embeddings.
  CRF on top: ensures globally valid label sequences.
  BERT: bidirectional Transformer → even richer context.
```

**Text Classification — Document-Level Prediction:**
```
Input: entire document or sentence.
Output: one label (sentiment, topic, spam/not-spam, etc.)

Approaches (roughly chronological):

1. TF-IDF + Logistic Regression / SVM
   Fast, interpretable, often surprisingly competitive.
   Works when vocabulary matters more than word order.

2. FastText (simple but powerful):
   Average word embeddings → linear classifier.
   Trick: also use bigram/trigram embeddings to capture local order.
   Trains in seconds on millions of documents.
   Often competitive with deep learning for simple classification.

3. TextCNN (Kim 2014):
   1D convolutions over word embeddings with multiple kernel sizes.
   
   Input: sequence of word embeddings, shape (n × d)
   
   Convolutions with widths [2, 3, 4] (bigram, trigram, 4-gram patterns):
   - Width-2 conv: looks at pairs → detects bigram features
   - Width-3 conv: trigrams → local phrase features
   - Width-4 conv: 4-grams → slightly longer phrases
   
   Max-pooling: take max over each filter → "did this feature appear anywhere?"
   Concatenate all max-pooled features → dense vector → classification.
   
   ✓ Captures local phrase features
   ✓ Fast (convolutions are parallel)
   ✓ Better than BoW for order-sensitive tasks

4. BERT for Classification (current SOTA):
   [CLS] + sentence → BERT encoder → [CLS] final embedding → Linear → label
   Fine-tune ALL BERT weights + linear head end-to-end.
   
   Pre-trained BERT already understands language deeply.
   Fine-tuning just teaches it the specific classification task.
   
   Few hundred labelled examples often sufficient!
```

---

### 16.5 Sequence-to-Sequence Tasks

> 💡 **Seq2Seq tasks take one sequence as input and produce a DIFFERENT sequence as output.** Translation, summarisation, Q&A, dialogue, code generation — all are seq2seq problems. Understanding the architecture evolution here explains why Transformers became dominant.

**Machine Translation — Historical Evolution:**
```
PHASE 1: Rule-Based Machine Translation (1950s-1980s)
  Linguists hand-wrote grammar rules and bilingual dictionaries.
  "Der Hund beißt den Mann" → look up each word in dictionary → "The dog bites the man"
  
  Problems: exceptions to every rule, idioms, ambiguity.
  "Time flies like an arrow" — does "flies" mean the verb or the insect?
  Rule systems couldn't handle ambiguity.

PHASE 2: Statistical Machine Translation (1990s-2015)
  IBM Models 1-5: probabilistic word alignment.
  P(French | English) = P(alignment) × P(words given alignment)
  
  Phrase-based SMT (Moses, the dominant system 2005-2015):
  1. Segment English into phrases.
  2. Translate each phrase using phrase table (learned from parallel corpus).
  3. Reorder translated phrases using language model.
  
  Each component trained separately. Complex pipeline, many moving parts.

PHASE 3: Neural Machine Translation (2014-present)
  Seq2Seq RNN (2014): encode entire source sentence → single vector → decode.
  Attention (2015): decoder dynamically attends to all source positions.
  Transformer (2017): replaces RNN with self-attention entirely.
  
  Performance: Transformers achieve >30 BLEU on WMT translation benchmarks
               where rule-based systems scored <10 and phrase-based ~25.

How Transformer Translation Works:
  
  Input: "The cat sat on the mat." (English)
  
  Encoder: reads entire English sentence with bidirectional attention.
  Creates contextualised embeddings for each English word.
  
  Decoder: generates German word by word:
  Step 1: [START] → cross-attend to "The cat sat on the mat" → predict "Die"
  Step 2: [START, Die] → cross-attend → predict "Katze"
  Step 3: [START, Die, Katze] → cross-attend → predict "saß"
  ...
  
  Cross-attention at each decoder step: "which English words are most relevant
  for generating the current German word?"
  
  For generating "saß" (sat): high attention to "sat" in English source.
  For generating "die" (the, feminine): high attention to "cat" (knowing cats are feminine in German).
  
  The model learns these alignment patterns from millions of parallel sentence pairs!
```

**BLEU Score — Evaluating Translation Quality:**
```
BLEU (Bilingual Evaluation Understudy):
  Compares machine translation to one or more human reference translations.
  Counts n-gram overlaps between hypothesis and references.

Modified n-gram precision:
  For each n-gram in hypothesis: count how many appear in any reference.
  (Modified: clip count to max frequency in any reference)
  
  pₙ = (count of matching n-grams) / (total n-grams in hypothesis)

BLEU score:
  BLEU = BP × exp(Σₙ wₙ log pₙ)
  
  BP (Brevity Penalty) = min(1, exp(1 − r/c))
    c = hypothesis length, r = reference length
    Prevents trivially short translations with high precision.
  
  wₙ = 1/N typically (uniform weights, N=4 for BLEU-4)

Example:
  Reference: "The cat sat on the mat"
  Hypothesis: "The cat is on the mat"
  
  Unigram matches: The, cat, on, the, mat → 5/6 correct
  Bigram matches: The cat, on the, the mat → 3/5 correct
  Trigram matches: on the mat → 1/4 correct
  4-gram matches: 0/3 correct
  
  BLEU-4 ≈ exp(0.25 × (log(5/6) + log(3/5) + log(1/4) + log(ε)))
  
  Interpretation:
  BLEU ≈ 0: no overlap (terrible translation)
  BLEU ≈ 0.3: acceptable human translation quality
  BLEU ≈ 0.5+: better than average human translation on some metrics
  BLEU = 1.0: perfect match (nearly impossible)
  
  Caveats:
  ✗ BLEU doesn't capture meaning — just surface overlap.
  ✗ Multiple valid translations score poorly.
  ✗ Doesn't evaluate fluency.
  
  Modern alternatives:
  METEOR: considers synonyms and morphological variations
  BERTScore: use BERT embeddings to compare meaning (not just surface form)
  COMET: neural model trained to predict human judgements
```

**Summarisation:**
```
Two types:

Extractive summarisation: select EXISTING sentences from the document.
  "Which 3 sentences from this 10-sentence document are most important?"
  Output: copy-paste selection — grammatically perfect, may miss connections.
  
  Classic method: TextRank (graph-based)
    Build graph where sentences are nodes.
    Edge weight = similarity between sentences (cosine of TF-IDF).
    Run PageRank → sentences with highest "centrality" are most important.
    Select top-K sentences as summary.

Abstractive summarisation: generate NEW sentences that capture the meaning.
  Like how a human would summarise — paraphrase, combine, condense.
  More challenging but produces more coherent, readable summaries.
  
  Modern approach: fine-tune BART or T5 on (document, summary) pairs.
  BART (Bidirectional and Auto-Regressive Transformer):
    Encoder: encodes full document with BERT-like bidirectional attention.
    Decoder: generates summary autoregressively.
    Pre-trained by corrupting text and learning to reconstruct it.
    Fine-tuned on CNN/DailyMail, XSum, etc.

ROUGE Score (for evaluating summarisation):
  Recall-Oriented Understudy for Gisting Evaluation.
  Measures n-gram recall (did the summary cover the reference's content?)
  
  ROUGE-1: unigram overlap
  ROUGE-2: bigram overlap
  ROUGE-L: longest common subsequence (captures sentence structure)
  
  ROUGE-N = matched n-grams / total n-grams in reference
  (Recall-focused, unlike BLEU's precision-focus)
```

---

### 16.6 Information Retrieval and Semantic Search

> 💡 **Information retrieval (IR) is the foundation of search engines.** Given a query, find the most relevant documents from a large corpus. Modern "semantic search" goes beyond keyword matching to understand the MEANING of queries.

**Sparse Retrieval — Keyword Based:**
```
BM25 (Best Match 25) — the gold standard sparse retrieval:
  
  Score(d, q) = Σ_{t ∈ q} IDF(t) × TF(t, d) × (k₁+1) / (TF(t,d) + k₁(1−b+b×|d|/avgdl))
  
  Where:
  IDF(t) = log((N−df(t)+0.5) / (df(t)+0.5) + 1)
  k₁ = term frequency saturation parameter (typically 1.2-2.0)
  b   = length normalisation parameter (typically 0.75)
  |d| = document length, avgdl = average document length
  
  Intuition:
  - TF contributes, but with SATURATION (diminishing returns for repeated terms)
    Without saturation: mentioning "cat" 100 times → 100× score? No!
    k₁ controls saturation: TF saturates at roughly k₁ × (1-b+b|d|/avgdl)
  
  - IDF: rare query terms get higher weight (more discriminative)
  - Length normalisation: longer documents naturally have higher TF → normalise by |d|
  
  Practical performance:
  ✓ Very fast (inverted index lookup, milliseconds for billions of documents)
  ✓ No GPU needed
  ✓ Robust, interpretable
  ✗ No synonyms: "car" doesn't match "automobile" → missed relevant docs
  ✗ No semantic understanding

Inverted Index (how BM25 scales to billions of documents):
  
  Forward index: document → list of words
  Inverted index: word → list of documents containing that word
  
  Query: "cat mat"
  1. Look up "cat" → [doc3, doc7, doc15, doc234, ...]  (milliseconds)
  2. Look up "mat" → [doc7, doc15, doc88, doc234, ...]
  3. Intersect/union → [doc7, doc15, doc234, ...]
  4. Score each by BM25 → rank
  
  Google's index: ~130 trillion web pages, lookup in <100ms. This is how.
```

**Dense Retrieval — Semantic Understanding:**
```
Neural approach: encode BOTH query and documents as dense vectors.
Find nearest neighbours in vector space = find semantically similar documents.

Bi-Encoder (Dual Encoder):
  query  → [BERT-like encoder] → q_vec ∈ ℝᵈ
  document → [BERT-like encoder] → d_vec ∈ ℝᵈ
  
  score(q, d) = cosine_similarity(q_vec, d_vec) = q_vec · d_vec / (||q|| ||d||)
  
  Key advantage: document vectors can be pre-computed offline!
  At query time: encode query (fast) + compare to cached document vectors (fast).
  With HNSW index: search 1M documents in <10ms on CPU.
  
  Training:
  Contrastive loss on (query, positive_doc, negative_docs) triplets:
  L = −log(exp(q·d⁺/τ) / (exp(q·d⁺/τ) + Σⱼ exp(q·dⱼ⁻/τ)))
  
  In-batch negatives: treat other queries' documents as negatives → efficient!

Cross-Encoder (for re-ranking):
  Input: [CLS] query [SEP] document [SEP]
  Output: single relevance score
  
  Pros: models INTERACTION between query and document terms → much more accurate
  Cons: must run for every (query, candidate) pair → slow for large corpora
  
  Two-stage retrieval pipeline (industry standard):
  Stage 1: Bi-encoder retrieves top-1000 candidates fast (ms)
  Stage 2: Cross-encoder re-ranks top-1000 to get top-10 precisely (seconds)
  
  This combines speed of bi-encoder with accuracy of cross-encoder!

Popular dense retrieval models:
  DPR (Dense Passage Retrieval): uses dual BERT encoders, trained on Natural Questions
  E5 (EmbEddings from bidirEctional Encoder rEpresentations): strong general embeddings
  BGE (BAAI General Embedding): best open-source embedding model for many tasks
  text-embedding-3-large (OpenAI): excellent commercial embedding model (3072 dims)
  all-MiniLM-L6-v2 (sentence-transformers): fast, small, good quality
```

---

### 16.7 Question Answering

> 💡 **Question answering (QA) is one of the most direct measures of language understanding.** Given a question and possibly a context, output the answer. This ranges from simple fact lookup to complex multi-hop reasoning.**

**Types of QA:**
```
Extractive QA (SQuAD-style):
  Context: "Albert Einstein was born in Ulm, Germany in 1879."
  Question: "Where was Einstein born?"
  Answer: "Ulm, Germany" ← extract span from context
  
  Model outputs: start_position, end_position in the context.
  
  BERT for extractive QA:
  Input: [CLS] question [SEP] context [SEP]
  Output: two vectors (logits over all context positions):
    start_logits = W_start × all_token_embeddings
    end_logits = W_end × all_token_embeddings
  
  Answer span = token[argmax(start_logits)] to token[argmax(end_logits)]

Open-Domain QA:
  No context provided — model must know the answer from training.
  "Who invented the telephone?"
  Two-stage:
    1. Retriever: find relevant documents from Wikipedia/web
    2. Reader: extract answer from retrieved documents
  
  Retriever-Reader Pipeline (Retrieval-Augmented Generation in disguise):
  Question → [Dense Retriever] → top-K documents → [Reader model] → Answer

Closed-Book QA:
  The LLM answers from parametric memory only (no retrieval).
  GPT-4, Claude: often answer factual questions correctly from training.
  Limitation: can hallucinate (confident wrong answers), can't update with new facts.

Multi-Hop QA:
  "What is the capital of the country where Einstein was born?"
  
  Requires chaining:
  Hop 1: "Einstein was born in Germany"
  Hop 2: "The capital of Germany is Berlin"
  Answer: "Berlin"
  
  Simple extractive QA cannot handle this — needs reasoning.
  Chain-of-Thought prompting helps LLMs solve multi-hop problems.

F1 Score for QA evaluation:
  Compare predicted answer to gold answer at token level.
  
  Predicted: "Ulm, Germany"
  Gold:      "Ulm"
  
  Precision = overlap tokens / predicted tokens = 1/2 = 0.5
  Recall    = overlap tokens / gold tokens     = 1/1 = 1.0
  F1        = 2PR/(P+R) = 2×0.5×1.0/1.5 = 0.67
  
  (Partial credit for partial matches — better than exact match only)
```

---

### 16.8 Text Generation and Controllable Generation

**Decoding Strategies:**
```
Given a trained language model, how do you generate text?

Greedy Decoding:
  At each step, pick the single most likely token.
  P(token | context) → argmax → append → next step.
  
  ✓ Deterministic, fast
  ✗ Repetitive and degenerate: "The cat sat on the the the the..."
  The model falls into loops — each token makes the next token more likely.

Temperature Sampling:
  Divide logits by temperature T before softmax.
  P(token) ∝ exp(logit / T)
  
  T=0.7:  sharper distribution (less likely tokens nearly impossible)
  T=1.0:  standard distribution
  T=1.5:  flatter distribution (more creative, more random)
  T=2.0:  very flat (often incoherent)
  
  Choosing T:
  Creative writing: T=0.8-1.2 (interesting, still coherent)
  Factual Q&A:      T=0.1-0.4 (confident, consistent)
  Code generation:  T=0.2-0.6 (correct syntax, not too creative)

Top-k Sampling:
  At each step, keep only the k most likely tokens, re-normalise, sample.
  k=50: sample from 50 most likely tokens (ignore very unlikely ones)
  
  Problem: k=50 works differently for different contexts.
  If distribution is NARROW: 50 choices is too many, includes garbage tokens.
  If distribution is WIDE: 50 choices is too few, misses valid options.

Top-p (Nucleus) Sampling:
  Keep the SMALLEST set of tokens whose total probability ≥ p.
  
  Narrow distribution: [0.7, 0.2, 0.05, 0.03, 0.02] → top-p=0.9: keep first 2 tokens
  Wide distribution: [0.1, 0.09, 0.08, 0.07, ...] → top-p=0.9: keep first ~12 tokens
  
  Adapts to the distribution shape → better than top-k.
  p=0.9-0.95 is a common default.

Min-p Sampling (newer, often better):
  Minimum probability threshold as a fraction of the top token's probability.
  min_p = 0.1: keep tokens with probability ≥ 0.1 × P(top_token)
  
  If top token has P=0.5: threshold = 0.05 → keep tokens with P ≥ 0.05
  If top token has P=0.1: threshold = 0.01 → keep tokens with P ≥ 0.01
  
  More principled than top-p: adapts to both distribution width AND confidence.

Beam Search:
  Maintain k hypotheses (beams) simultaneously.
  At each step: expand all k beams, keep top-k total sequences.
  
  k=4 beams for "The cat":
  Beam 1: "The cat sat..." (score -1.2)
  Beam 2: "The cat ate..." (score -1.4)
  Beam 3: "The cat slept..." (score -1.6)
  Beam 4: "The cat ran..." (score -1.7)
  
  ✓ Finds higher-probability sequences than greedy
  ✗ Still tends to produce generic, boring output
  ✗ 4× more compute than greedy
  
  Good for: translation, summarisation (where there's a "right" answer)
  Bad for: creative generation (produces bland text)

Repetition Penalty:
  Reduce logits for tokens that have already appeared:
  logit(t) = logit(t) / penalty   if t appeared before (penalty > 1)
  
  Prevents the model from repeating "The cat sat on the the the..."
  Common penalty: 1.1-1.3 (10-30% reduction per repetition)
```

**Chain-of-Thought Prompting:**
```
The breakthrough insight (Wei et al., 2022):
  Adding "Let's think step by step" to a prompt causes LLMs to
  produce intermediate reasoning steps → massive improvement on complex tasks.

Without CoT:
  Q: "A store has 15 apples. They sell 7 and get a new shipment of 12. How many apples?"
  A: "27"  ← just gives the number, might be wrong for complex problems

With CoT:
  Q: "... Let's think step by step."
  A: "The store starts with 15 apples. They sell 7, so they have 15 - 7 = 8 apples.
      Then they get 12 more: 8 + 12 = 20 apples. The answer is 20."

Why does this work?
  1. Forces the model to generate intermediate computations explicitly.
  2. Each reasoning step is easier (smaller logical leap) than the full problem.
  3. The model "debugs" its own reasoning as it goes.
  4. The context window provides "working memory" for multi-step problems.

Few-Shot CoT (providing examples with reasoning):
  Q1: "John has 3 cats. He gives 1 away and buys 2 more."
  A1: "John starts with 3 cats. 3 - 1 = 2 cats. 2 + 2 = 4 cats. Answer: 4."
  
  Q2: [new problem]
  A2: [model follows the same reasoning pattern]
  
  Few-shot CoT dramatically outperforms zero-shot CoT on complex reasoning.

Zero-Shot CoT triggers:
  "Let's think step by step." (original)
  "Let's work through this carefully."
  "First, let me understand what's being asked..."
  "Step 1:" (models continue with more steps)

Self-Consistency (improved CoT):
  1. Sample multiple CoT reasoning paths (temperature > 0)
  2. Take the MAJORITY vote among final answers
  
  3 independent reasoning paths might give: 20, 20, 18.
  Majority vote: 20 → more reliable than any single path.
  Works because errors in reasoning are diverse; correct reasoning converges.

ReAct (Reasoning + Acting — for agents):
  Interleave reasoning (Thought) and action (Act) steps:
  
  Thought: "I need to find the capital of France."
  Act:     "Search[capital of France]"
  Obs:     "Paris is the capital of France."
  Thought: "The answer is Paris."
  Act:     "Finish[Paris]"
  
  Allows LLMs to use tools (search, code execution, databases) while reasoning.
  Foundation of modern AI agents.
```

---

### 16.9 Hallucination — The Critical Problem in NLP/LLMs

> 💡 **Hallucination is when a model confidently states something that is false.** "Einstein won the Nobel Prize for relativity" (wrong — it was for the photoelectric effect). This is perhaps the most important practical challenge in deploying LLMs today.

**Types of Hallucination:**
```
1. Factual Hallucination:
   "Barack Obama was born in Kenya." (False — born in Hawaii)
   Model confidently states wrong facts from training data noise or confusion.

2. Contradictory Hallucination:
   Within the same response, contradicts itself.
   "The Eiffel Tower is 300 meters tall. ... As I mentioned, at 330 meters..."

3. Faithfulness Hallucination (RAG/Summarisation):
   Context says: "The report shows 15% growth."
   Model says: "The report shows 25% growth."
   Model generates information not in the source.

4. Entity Confusion:
   "Marie Curie discovered penicillin." (No — that was Alexander Fleming)
   Confuses similar-sounding or associated concepts.

5. Citation Fabrication:
   "According to Smith et al. (2019)..." — paper doesn't exist!
   Generates plausible-sounding but fake references.
```

**Why Hallucinations Happen:**
```
Root cause 1: Training objective mismatch
  LLMs are trained to produce FLUENT, LIKELY next tokens.
  Fluency ≠ truthfulness.
  A confident-sounding wrong answer may be MORE likely than a hedged right answer
  based on training data patterns.

Root cause 2: Knowledge limitations
  Training data has a cutoff date.
  Not all facts are equally represented in training data.
  The model extrapolates when it doesn't know → plausible-sounding guesses.

Root cause 3: Lack of grounding
  Without retrieval, the model has no external reference to check against.
  It only knows what it "remembers" from training — imperfect and approximate.

Root cause 4: RLHF overconfidence
  Human raters often prefer confident-sounding answers.
  RLHF inadvertently trains models to sound confident even when uncertain.

Root cause 5: Temperature and sampling
  Higher temperature → more creative → more likely to produce rare/wrong tokens.
  The model explores the long tail of its distribution → hallucinations live there.
```

**Mitigating Hallucination:**
```
1. Retrieval-Augmented Generation (RAG):
   Don't rely on parametric memory. Retrieve facts from an authoritative source.
   Ground every factual claim in retrieved documents.
   → Reduces factual hallucination significantly (but not faithfulness hallucination)

2. Citation and Attribution:
   Force the model to cite the specific passage supporting each claim.
   "According to [retrieved document, paragraph 2]: ..."
   Easy to verify manually or automatically.

3. Constitutional AI and Self-Correction:
   After generating an answer, prompt the model to critique it:
   "Review your answer. Is everything factually accurate? Correct any errors."
   Models can catch some of their own hallucinations.

4. Temperature = 0 for Factual Tasks:
   Greedy decoding minimises randomness → fewer hallucinations.
   Trade-off: less creative, more repetitive.

5. Uncertainty Calibration:
   Train models to say "I don't know" or "I'm not sure" when uncertain.
   Selective prediction: abstain rather than guess on uncertain questions.
   Models with well-calibrated uncertainty are much safer to deploy.

6. FACTSCORE and Evaluation:
   Decompose answers into atomic facts.
   Check each atomic fact against a knowledge base.
   FActScore = fraction of atomic facts that are supported.
   
   Answer: "Einstein won the Nobel Prize in 1921 for the photoelectric effect."
   Fact 1: "Einstein won Nobel Prize" ✓ (supported)
   Fact 2: "in 1921" ✓ (supported)
   Fact 3: "for photoelectric effect" ✓ (supported)
   FActScore = 3/3 = 1.0 (perfect for this sentence)

7. Retrieval + Verification Pipeline:
   Generate claim → retrieve evidence → verify consistency → output with confidence.
   Used in: perplexity.ai, bing chat, google's SGE.
```

---

### 16.10 Modern NLP Benchmarks

```
Benchmark          │ Task Type             │ What It Tests
────────────────────────────────────────────────────────────────────────────
GLUE               │ Various              │ General language understanding (9 tasks)
SuperGLUE          │ Various              │ Harder version of GLUE
SQuAD 1.1/2.0      │ Extractive QA        │ Reading comprehension
HotpotQA           │ Multi-hop QA         │ Reasoning across multiple documents
MMLU               │ Multiple choice      │ Massive Multitask Language Understanding
                   │                      │ 57 subjects from elementary to professional
GSM8K              │ Math word problems   │ Grade-school math (requires multi-step)
MATH               │ Math problems        │ Competition-level mathematics
HumanEval          │ Code generation      │ Python function generation from docstrings
MBPP               │ Code generation      │ Mostly Basic Programming Problems
HellaSwag          │ Commonsense          │ Complete a story plausibly
WinoGrande         │ Coreference          │ Pronoun resolution in tricky contexts
ARC                │ Science QA           │ Grade-school science (easy & challenge)
TruthfulQA         │ Factual accuracy     │ Avoid truthful-sounding but false answers
BIG-bench          │ Various (200+ tasks) │ Diverse difficult tasks beyond GPT-3
MT-Bench           │ Multi-turn chat      │ Conversation quality (GPT-4 as judge)
AlpacaEval         │ Instruction following│ Win rate vs GPT-4 (human or GPT-4 judge)

MMLU Example:
  Question: "A patient presents with fever, stiff neck, and photophobia. The most
             likely diagnosis is:"
  A) Migraine  B) Bacterial meningitis  C) Tension headache  D) Encephalitis
  Answer: B) Bacterial meningitis

  Covers 57 subjects: medicine, law, mathematics, history, chemistry, ethics...
  A perfect general-purpose AI assistant should excel on all.

GPT-4 performance on key benchmarks (2023):
  MMLU:     86.4%  (human expert = ~89%)
  GSM8K:    92.0%  (was 57% for GPT-3!)
  HumanEval: 67.0% (was 28% for GPT-3.5)
  
LLaMA-3-70B performance (2024, open-source):
  MMLU:     82.0%  (competitive with GPT-4!)
  GSM8K:    93.0%
  HumanEval: 81.7%
  
  Open-source models are rapidly catching up to proprietary models.
```

---

## 17. LangChain & LLM Application Engineering

> 💡 **LangChain is a framework for building applications with LLMs.** It provides abstractions for the most common LLM application patterns: chaining operations, retrieving documents, managing conversation history, and building agents that can use tools. Think of it as the "Express.js for LLM apps" — opinionated structure, lots of batteries included.

---

### 17.1 The Core Abstractions

**Why LangChain Exists:**
```
Building an LLM app from scratch involves:
  1. Formatting prompts (templates with variables)
  2. Calling the LLM API (handling rate limits, retries, streaming)
  3. Parsing the output (JSON, structured data, etc.)
  4. Retrieving relevant documents (vector search, keyword search)
  5. Managing conversation history (truncating, summarising)
  6. Chaining operations (output of one step → input of next)
  7. Building agents (LLM decides what tool to use next)
  
LangChain provides clean abstractions for ALL of these.
Change from OpenAI to Anthropic: change ONE line of code, everything else works.
Add retrieval to a chain: add ONE retriever component, chain is augmented.

Core primitives (everything implements "Runnable"):
  .invoke(input) → output          (single call)
  .stream(input) → iterator        (streaming tokens)
  .batch(inputs) → list[output]    (parallel calls)
  .ainvoke(input) → awaitable      (async version)
```

**Installation and Setup:**
```python
# Install
pip install langchain langchain-openai langchain-anthropic
pip install langchain-community langchain-chroma faiss-cpu

# Environment
import os
os.environ["OPENAI_API_KEY"] = "sk-..."      # OpenAI
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..." # Anthropic

# LangSmith (optional but recommended for debugging/monitoring)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls-..."
os.environ["LANGCHAIN_PROJECT"] = "my-rag-app"
```

**LLM and Chat Model Wrappers:**
```python
from langchain_openai import ChatOpenAI, OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama  # local models

# Chat models (recommended — structured messages)
gpt4 = ChatOpenAI(
    model="gpt-4o",
    temperature=0,          # 0 = deterministic, consistent
    max_tokens=2048,        # max output length
    timeout=30,             # API timeout in seconds
)

claude = ChatAnthropic(
    model="claude-opus-4-6",
    temperature=0.7,        # some creativity
)

# Local models via Ollama (free, private, runs locally)
local_llm = Ollama(
    model="llama3",
    temperature=0,
)

# Message types
from langchain_core.messages import (
    SystemMessage,   # sets the assistant's persona/behaviour
    HumanMessage,    # user's message
    AIMessage,       # assistant's previous response (for history)
)

# Simple call
messages = [
    SystemMessage(content="You are an expert ML engineer."),
    HumanMessage(content="Explain attention mechanisms."),
]
response = gpt4.invoke(messages)
print(response.content)  # the text response

# Streaming (for real-time output)
for chunk in gpt4.stream(messages):
    print(chunk.content, end="", flush=True)

# Batch (parallel calls)
questions = ["What is BERT?", "What is GPT?", "What is T5?"]
results = gpt4.batch([
    [HumanMessage(content=q)] for q in questions
])
```

---

### 17.2 Prompt Templates — Structured Prompts**

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,  # for inserting chat history
)

# ─── Chat Prompt Template ───
template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert {domain} tutor.
    Explain concepts clearly with examples.
    Use analogies appropriate for a {level} student."""),
    ("human", "{question}"),
])

# Fill in variables
messages = template.format_messages(
    domain="machine learning",
    level="beginner",
    question="What is gradient descent?"
)

# ─── Partial Templates (fill in some variables now, some later) ───
partial_template = template.partial(domain="machine learning")
# Now only need level and question at call time
messages = partial_template.format_messages(
    level="expert",
    question="Derive the Adam update rule."
)

# ─── Few-Shot Prompting ───
examples = [
    {
        "question": "What is 2+2?",
        "thinking": "Simple addition: 2 plus 2 equals 4.",
        "answer": "4"
    },
    {
        "question": "What is 15 × 8?",
        "thinking": "15 × 8 = 15 × 8. I can compute: 10×8=80, 5×8=40, total=120.",
        "answer": "120"
    },
]

example_prompt = PromptTemplate(
    input_variables=["question", "thinking", "answer"],
    template="Question: {question}\nThinking: {thinking}\nAnswer: {answer}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Solve math problems step by step:",
    suffix="Question: {query}\nThinking:",
    input_variables=["query"],
)

# ─── With Conversation History ───
conversation_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),  # insert past messages here
    ("human", "{input}"),
])
```

---

### 17.3 LCEL — The Pipe Operator for Chaining**

> 💡 **LCEL (LangChain Expression Language) uses the `|` (pipe) operator to chain components together.** `A | B | C` means: run A, pass output to B, pass that output to C. Like Unix pipes, but for LLM operations.

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# ─── Basic Chain ───
chain = template | gpt4 | StrOutputParser()
# template formats the prompt
# gpt4 calls the API and returns AIMessage
# StrOutputParser extracts the text content from AIMessage

result = chain.invoke({
    "domain": "deep learning",
    "level": "intermediate",
    "question": "How does backpropagation work?"
})
print(result)  # plain string response

# ─── Streaming ───
for chunk in chain.stream({
    "domain": "NLP",
    "level": "beginner",
    "question": "What is tokenisation?"
}):
    print(chunk, end="", flush=True)

# ─── Parallel Chains ───
analysis_chain = RunnableParallel(
    # Run two chains in PARALLEL (simultaneously)
    summary = (template | gpt4 | StrOutputParser()),
    keywords = (keyword_template | gpt4 | StrOutputParser()),
)

results = analysis_chain.invoke({"text": "Long article about neural networks..."})
print(results["summary"])    # Summary from chain 1
print(results["keywords"])   # Keywords from chain 2

# ─── Conditional Routing ───
from langchain_core.runnables import RunnableLambda

def route_question(info):
    """Route to different chains based on content."""
    if "code" in info["question"].lower():
        return code_chain
    elif "math" in info["question"].lower():
        return math_chain
    else:
        return general_chain

full_chain = (
    {"question": RunnablePassthrough()}
    | RunnableLambda(route_question)
)

# ─── Error Handling ───
safe_chain = chain.with_retry(
    stop_after_attempt=3,  # retry up to 3 times
    wait_exponential_jitter=True,  # exponential backoff with jitter
)

# Fallback: if primary fails, try backup
robust_chain = primary_chain.with_fallbacks([
    backup_chain_gpt35,  # try cheaper model if gpt4 fails
    local_llm_chain,     # then try local model
])

# ─── Async Support ───
import asyncio

async def answer_questions(questions):
    tasks = [chain.ainvoke({"question": q}) for q in questions]
    return await asyncio.gather(*tasks)

results = asyncio.run(answer_questions([
    "What is BERT?", "What is GPT?", "What is T5?"
]))
```

---

### 17.4 Output Parsers — Structured Outputs**

```python
from langchain_core.output_parsers import (
    StrOutputParser,      # plain text
    JsonOutputParser,     # JSON dict
    PydanticOutputParser, # validated structured output
)
from pydantic import BaseModel, Field
from typing import List, Optional

# ─── Pydantic Structured Output (Best Practice) ───
class MLModelAnalysis(BaseModel):
    """Structured analysis of an ML model."""
    model_name: str = Field(description="Name of the ML model")
    model_type: str = Field(description="Type: classification/regression/generative")
    key_innovations: List[str] = Field(description="List of key technical innovations")
    strengths: List[str] = Field(description="Model's main strengths")
    weaknesses: List[str] = Field(description="Model's main weaknesses")
    best_use_cases: List[str] = Field(description="Ideal applications")
    paper_year: Optional[int] = Field(description="Year of original paper", default=None)
    confidence: float = Field(description="Confidence in analysis 0-1", ge=0, le=1)

parser = PydanticOutputParser(pydantic_object=MLModelAnalysis)

analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an ML expert. Analyse the given model.
    {format_instructions}
    Return ONLY valid JSON, no markdown code blocks."""),
    ("human", "Analyse this ML model: {model_name}"),
]).partial(format_instructions=parser.get_format_instructions())

# Chain with structured output
analysis_chain = analysis_prompt | gpt4 | parser

# Returns a validated MLModelAnalysis object
result: MLModelAnalysis = analysis_chain.invoke({"model_name": "BERT"})
print(f"Model: {result.model_name}")
print(f"Innovations: {result.key_innovations}")
print(f"Confidence: {result.confidence}")

# ─── With GPT-4's native structured output (more reliable) ───
structured_gpt4 = gpt4.with_structured_output(MLModelAnalysis)
result = structured_gpt4.invoke("Analyse BERT for NLP tasks")
```

---

### 17.5 RAG — Retrieval-Augmented Generation

> 💡 **RAG is the most important LLM application pattern for production systems.** Instead of relying purely on the model's training knowledge (which may be outdated or incomplete), RAG retrieves relevant documents and includes them in the context. "Look it up before you answer" — making LLMs factual and updatable.

**The RAG Pipeline:**
```
User Query: "What were the main contributions of the Attention is All You Need paper?"

Step 1: RETRIEVE
  Convert query to vector: embed("What were the main contributions...")
  Search vector database for similar chunks from your document corpus
  Return top-K most relevant chunks from the paper.
  
  Retrieved chunks:
  [1] "We propose a new simple network architecture, the Transformer, based solely 
      on attention mechanisms, dispensing with recurrence and convolutions entirely."
  [2] "The Transformer achieves 28.4 BLEU on the WMT 2014 English-to-German translation 
      task, improving over the existing best results..."
  [3] "Multi-head attention allows the model to jointly attend to information from 
      different representation subspaces at different positions."

Step 2: AUGMENT  
  Insert retrieved chunks into the prompt:
  "Answer the question using ONLY the provided context:
   Context: [chunk 1] [chunk 2] [chunk 3]
   Question: What were the main contributions?"

Step 3: GENERATE
  LLM reads the context + question and generates a grounded answer.
  Answer is based on retrieved documents → less hallucination!
```

**Building a Complete RAG System:**
```python
from langchain_community.document_loaders import (
    PyPDFLoader,      # PDF files
    TextLoader,       # .txt files
    WebBaseLoader,    # web pages
    DirectoryLoader,  # entire directory of files
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ─── Step 1: Load Documents ───
# Load a PDF paper
loader = PyPDFLoader("attention_is_all_you_need.pdf")
docs = loader.load()
# Each doc has: doc.page_content (text), doc.metadata (source, page, etc.)

# Load a web page
web_loader = WebBaseLoader("https://arxiv.org/abs/1706.03762")
web_docs = web_loader.load()

# Load all .txt files in a directory
dir_loader = DirectoryLoader("./papers/", glob="**/*.txt", loader_cls=TextLoader)
all_docs = dir_loader.load()

# ─── Step 2: Split Into Chunks ───
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # target size in characters
    chunk_overlap=200,    # overlap between adjacent chunks
    # Overlap preserves context at chunk boundaries:
    # Without overlap: "...end of concept A | beginning of concept B..."
    # The boundary sentence might be cut in half!
    # With overlap: both chunks contain the boundary sentence.
    
    separators=["\n\n", "\n", ". ", " ", ""],
    # Try to split at paragraph breaks first, then newlines, then sentences...
    # Graceful degradation to smaller units if needed.
    
    add_start_index=True,  # track where in original doc each chunk starts
)

chunks = splitter.split_documents(docs)
print(f"Split {len(docs)} documents into {len(chunks)} chunks")
print(f"Example chunk: {chunks[0].page_content[:200]}...")

# ─── Step 3: Create Embeddings and Vector Store ───
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# Each text chunk → 3072-dimensional vector
# Similar chunks → nearby vectors in 3072-dimensional space

# Create and persist the vector database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",  # saves to disk for reuse
    collection_name="ml_papers",
)

# Load existing database
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
    collection_name="ml_papers",
)

# ─── Step 4: Create Retriever ───
retriever = vectorstore.as_retriever(
    search_type="mmr",   # Maximal Marginal Relevance
    # MMR balances relevance (similar to query) AND diversity (different from each other)
    # Avoids returning 5 nearly-identical chunks!
    search_kwargs={
        "k": 5,            # return 5 chunks
        "lambda_mult": 0.7 # 0=max diversity, 1=max relevance, 0.7=good balance
    }
)

# Test retrieval
retrieved = retriever.invoke("How does multi-head attention work?")
for doc in retrieved:
    print(f"Source: {doc.metadata['source']}, Page: {doc.metadata.get('page', '?')}")
    print(f"Content: {doc.page_content[:300]}...")
    print("---")

# ─── Step 5: Build the RAG Chain ───
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert AI assistant. Answer questions using ONLY 
    the provided context. If the context doesn't contain enough information, 
    say "I don't have enough information to answer this question."
    
    Always cite which part of the context supports your answer.
    
    Context:
    {context}
    """),
    ("human", "{question}")
])

def format_docs(docs):
    """Format retrieved documents for inclusion in the prompt."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        formatted.append(
            f"[Document {i} — Source: {source}, Page: {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)

rag_chain = (
    {
        "context": retriever | format_docs,  # retrieve and format docs
        "question": RunnablePassthrough(),    # pass question through unchanged
    }
    | rag_prompt
    | gpt4
    | StrOutputParser()
)

# Use it!
answer = rag_chain.invoke(
    "What is the computational complexity of multi-head attention?"
)
print(answer)

# Streaming version
for chunk in rag_chain.stream(
    "Explain the scaled dot-product attention formula."
):
    print(chunk, end="", flush=True)
```

**Advanced RAG Techniques:**
```python
# ─── Hybrid Search (Dense + BM25) ───
# Dense search: finds semantically similar docs (catches synonyms)
# BM25 search: finds keyword matches (catches exact terms)
# Hybrid: best of both worlds

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# BM25 requires all documents in memory
bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Combine with weighted ensemble
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6],  # 40% keyword, 60% semantic
)

# ─── Multi-Query Retrieval ───
# Problem: your query might be phrased poorly for the retriever.
# Solution: generate multiple reformulations of the query.

from langchain.retrievers.multi_query import MultiQueryRetriever

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=dense_retriever,
    llm=gpt4,
    # Internally, the LLM generates 3 paraphrases of your query.
    # Retrieve docs for all 3 queries, deduplicate.
    # More diverse set of relevant documents!
)

# ─── Contextual Compression ───
# Problem: retrieved chunks may be long with irrelevant parts.
# Solution: LLM compresses each chunk to only the relevant portion.

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(gpt4)
# For each retrieved chunk, the compressor extracts only the relevant sentences.

compressed_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=dense_retriever,
)

# ─── Self-Query (Filter by Metadata) ───
# Natural language queries that filter metadata automatically
# "papers about attention from 2017" → filters year=2017 + semantic search

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(name="source", description="Filename of the source paper", type="string"),
    AttributeInfo(name="page", description="Page number in the paper", type="integer"),
    AttributeInfo(name="year", description="Publication year of the paper", type="integer"),
]

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=gpt4,
    vectorstore=vectorstore,
    document_contents="Machine learning research papers",
    metadata_field_info=metadata_field_info,
    verbose=True,  # see the generated structured query
)

# This will automatically filter: semantic search for "attention" AND filter year=2017
results = self_query_retriever.invoke("papers about attention from 2017")
```

---

### 17.6 Memory and Conversation Management**

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ─── Session-Based Memory ───
# Store conversation history per session (per user, per conversation)

session_store = {}  # In production: use Redis, DynamoDB, etc.

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Retrieve or create conversation history for a session."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# Wrap any chain with memory
chain_with_memory = RunnableWithMessageHistory(
    chain,                              # your base chain
    get_session_history,               # function to get history
    input_messages_key="input",        # which key is the human's message
    history_messages_key="history",    # which key to insert history into template
)

# Use it — same session_id = persistent conversation
response1 = chain_with_memory.invoke(
    {"input": "My name is Alice."},
    config={"configurable": {"session_id": "alice_session_1"}}
)
response2 = chain_with_memory.invoke(
    {"input": "What is my name?"},  # model remembers!
    config={"configurable": {"session_id": "alice_session_1"}}
)
# response2 will say "Your name is Alice."

# Different session = fresh start
response3 = chain_with_memory.invoke(
    {"input": "What is my name?"},  # different session, no memory
    config={"configurable": {"session_id": "bob_session_1"}}
)
# response3: "I don't know your name yet."

# ─── Memory with Summarisation (for long conversations) ───
# When conversation history grows too long for the context window:
# Summarise old messages, keep recent ones verbatim.

from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=gpt4,              # model used for summarisation
    max_token_limit=1000,  # keep last 1000 tokens verbatim
    return_messages=True,  # return as message objects
)
# When memory exceeds max_token_limit, oldest messages get summarised.
# "Previous conversation summary: User asked about BERT, we explained it was
#  a bidirectional encoder-only Transformer pre-trained with MLM..."
```

---

### 17.7 Agents — LLMs That Use Tools**

> 💡 **Agents are LLMs that can decide WHICH TOOLS to use to answer a question.** Instead of a fixed pipeline (retrieve → generate), an agent can: search the web, run code, query a database, call an API — whatever is needed. The LLM acts as the "brain" that decides the sequence of actions.

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun

# ─── Define Custom Tools ───
@tool
def calculate_model_parameters(
    num_layers: int,
    d_model: int,
    vocab_size: int
) -> str:
    """
    Calculate the total parameters in a Transformer model.
    
    Args:
        num_layers: Number of Transformer layers
        d_model: Model dimension (hidden size)
        vocab_size: Vocabulary size
    
    Returns:
        String describing parameter count breakdown
    """
    attention_params = 4 * d_model ** 2 * num_layers
    ffn_params = 8 * d_model ** 2 * num_layers
    embedding_params = vocab_size * d_model
    total = attention_params + ffn_params + embedding_params
    
    return (f"Transformer Parameters:\n"
            f"  Attention: {attention_params:,} ({attention_params/1e9:.2f}B)\n"
            f"  FFN:       {ffn_params:,} ({ffn_params/1e9:.2f}B)\n"
            f"  Embedding: {embedding_params:,} ({embedding_params/1e9:.2f}B)\n"
            f"  Total:     {total:,} ({total/1e9:.2f}B)")

@tool
def search_arxiv(query: str) -> str:
    """Search ArXiv for recent ML papers. Use for finding papers on specific topics."""
    import arxiv
    search = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
    results = []
    for paper in search.results():
        results.append(f"Title: {paper.title}\nSummary: {paper.summary[:300]}...\n")
    return "\n---\n".join(results) if results else "No papers found."

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return the output. Good for calculations and data analysis."""
    import subprocess
    result = subprocess.run(
        ["python", "-c", code],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        return result.stdout
    else:
        return f"Error: {result.stderr}"

# ─── Build the Agent ───
tools = [
    DuckDuckGoSearchRun(),      # web search
    WikipediaQueryRun(),         # Wikipedia search
    calculate_model_parameters,  # custom: parameter calculation
    search_arxiv,               # custom: paper search
    run_python_code,            # custom: code execution
]

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert ML engineer and researcher.
    Use your tools to answer questions accurately.
    
    Always verify facts with search when unsure.
    Show calculations when doing math.
    Break complex tasks into multiple tool calls.
    
    After gathering information, provide a comprehensive, well-structured answer."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    # agent_scratchpad = the agent's internal reasoning + tool outputs
])

agent = create_tool_calling_agent(
    llm=gpt4,
    tools=tools,
    prompt=agent_prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,        # shows reasoning steps
    max_iterations=10,   # prevent infinite loops
    handle_parsing_errors=True,  # recover from malformed outputs
    return_intermediate_steps=True,  # inspect tool calls
)

# ─── Use the Agent ───
result = agent_executor.invoke({
    "input": "Compare the parameter counts of BERT-base, GPT-3, and LLaMA-3-70B. Also search for the latest benchmark comparing these model families.",
    "chat_history": [],
})

print("Final Answer:", result["output"])
print("\nTool calls made:")
for action, observation in result["intermediate_steps"]:
    print(f"  Tool: {action.tool}")
    print(f"  Input: {action.tool_input}")
    print(f"  Output: {str(observation)[:200]}...")
```

**ReAct Pattern — The Agent's Reasoning Loop:**
```
ReAct = Reasoning + Acting

The agent alternates between THINKING and ACTING:

Thought 1: "The user wants to compare BERT, GPT-3, and LLaMA-3-70B parameters.
            I should calculate these using the calculate_model_parameters tool."

Act 1: calculate_model_parameters(num_layers=12, d_model=768, vocab_size=30000)
Obs 1: "BERT-base total: 110M parameters"

Thought 2: "Now I need GPT-3 parameters."
Act 2: calculate_model_parameters(num_layers=96, d_model=12288, vocab_size=50000)
Obs 2: "GPT-3 total: 175B parameters"

Thought 3: "LLaMA-3-70B has GQA and different architecture. Let me search for exact specs."
Act 3: DuckDuckGoSearch("LLaMA-3 70B exact parameter count architecture")
Obs 3: "LLaMA-3-70B: 70.6B parameters, 80 layers, d_model=8192, GQA..."

Thought 4: "Now let me search for recent benchmark comparisons."
Act 4: search_arxiv("BERT GPT LLaMA comparison benchmark 2024")
Obs 4: [paper summaries...]

Thought 5: "I have all the information. Let me provide a comprehensive answer."
Final Answer: [complete comparison with sources]

This is NOT just predefined steps — the agent DECIDES what to do next
based on what it has learned so far. If a tool fails, it tries another.
If it needs more information, it searches more. True adaptive reasoning!
```

---

### 17.8 LangGraph — Stateful Workflows**

> 💡 **LangGraph builds on LangChain to create stateful, multi-agent workflows.** While LangChain chains are linear (A→B→C), LangGraph creates graphs with loops, branches, and persistent state — enabling complex agentic behaviours like "try this approach, if it fails try another."

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, List
import operator

# ─── Define the Graph State ───
# State is shared and updated throughout the workflow
class ResearchState(TypedDict):
    question: str                                    # original question
    messages: Annotated[List, operator.add]          # conversation history (accumulates)
    retrieved_docs: List[str]                        # documents found
    draft_answer: str                                # initial answer
    final_answer: str                                # polished final answer
    iteration_count: int                             # loop counter
    
# ─── Define Nodes (steps in the workflow) ───
def retrieve_node(state: ResearchState) -> dict:
    """Search for relevant documents."""
    docs = retriever.invoke(state["question"])
    return {"retrieved_docs": [doc.page_content for doc in docs]}

def draft_answer_node(state: ResearchState) -> dict:
    """Generate an initial draft answer."""
    context = "\n".join(state["retrieved_docs"])
    response = gpt4.invoke([
        SystemMessage(content=f"Answer based on context:\n{context}"),
        HumanMessage(content=state["question"])
    ])
    return {"draft_answer": response.content}

def critique_node(state: ResearchState) -> dict:
    """Critique the draft and decide if it needs improvement."""
    response = gpt4.invoke([
        SystemMessage(content="Critique this answer. Is it complete and accurate?"),
        HumanMessage(content=f"Question: {state['question']}\nAnswer: {state['draft_answer']}")
    ])
    # Append the critique as a message
    return {
        "messages": [AIMessage(content=f"Critique: {response.content}")],
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def refine_node(state: ResearchState) -> dict:
    """Refine the answer based on critique."""
    critique = state["messages"][-1].content if state["messages"] else ""
    response = gpt4.invoke([
        SystemMessage(content="Improve this answer based on the critique."),
        HumanMessage(content=f"Original answer: {state['draft_answer']}\nCritique: {critique}")
    ])
    return {
        "draft_answer": response.content,
        "final_answer": response.content,
    }

def finalise_node(state: ResearchState) -> dict:
    """Package the final answer."""
    return {"final_answer": state["draft_answer"]}

# ─── Define Routing Logic ───
def should_refine(state: ResearchState) -> str:
    """Decide: refine more or we're done?"""
    # Loop at most 2 times
    if state.get("iteration_count", 0) >= 2:
        return "finalise"
    # Check if critique suggests improvement needed
    if state["messages"]:
        last_msg = state["messages"][-1].content.lower()
        if any(word in last_msg for word in ["incomplete", "incorrect", "missing", "improve"]):
            return "refine"
    return "finalise"

# ─── Build the Graph ───
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("retrieve",  retrieve_node)
workflow.add_node("draft",     draft_answer_node)
workflow.add_node("critique",  critique_node)
workflow.add_node("refine",    refine_node)
workflow.add_node("finalise",  finalise_node)

# Add edges (flow between nodes)
workflow.set_entry_point("retrieve")          # start here
workflow.add_edge("retrieve", "draft")        # retrieve → draft
workflow.add_edge("draft", "critique")        # draft → critique
workflow.add_conditional_edges(               # critique → (refine OR finalise)
    "critique",
    should_refine,
    {"refine": "refine", "finalise": "finalise"}
)
workflow.add_edge("refine", "critique")       # refine → critique again (loop!)
workflow.add_edge("finalise", END)            # done!

# Compile with checkpointing (enables persistence across sessions)
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# ─── Run the Workflow ───
result = app.invoke(
    {
        "question": "Explain the key differences between BERT and GPT architectures.",
        "messages": [],
        "retrieved_docs": [],
        "draft_answer": "",
        "final_answer": "",
        "iteration_count": 0,
    },
    config={"configurable": {"thread_id": "research_session_1"}}
)
print(result["final_answer"])

# Visualise the graph (Jupyter notebook)
# display(Image(app.get_graph().draw_mermaid_png()))
```

---

## 18. Handling Imbalanced Data

> 💡 **Class imbalance is one of the most common and most mishandled problems in real-world ML.** Credit card fraud: 0.1% fraud vs 99.9% legitimate. Medical diagnosis: 1% disease vs 99% healthy. Rare failure prediction: 0.01% failures. A model that predicts "no fraud / no disease / no failure" for EVERYTHING gets 99%+ accuracy — but is completely useless.

---

### 18.1 Diagnosing Imbalance

```
Imbalance Ratio (IR) = N_majority / N_minority

IR = 2:      mild (50/50 vs 67/33) — often manageable
IR = 10:     moderate (91% majority) — needs attention
IR = 100:    severe (99% majority) — significant measures needed
IR = 1000:   extreme (99.9% majority) — specialist techniques required

Visualisation:
  Balanced:      ●●●●● ○○○○○  (5:5)
  Mild:          ●●●●●●●● ○○  (8:2)
  Moderate:      ●●●●●●●●●● ○ (10:1)
  Severe:        ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●○ (99:1)

Real-world imbalance examples:
  Credit card fraud:       0.17% positive (IR = 588)
  Cancer detection (rare): 0.1% positive (IR = 1000)
  Manufacturing defects:   2% positive (IR = 49)
  Email spam:              ~30% positive (IR = 2.3 — mild!)
  Network intrusion:       0.01% attacks (IR = 10,000)
```

**Why Standard Metrics Fail:**
```
Scenario: cancer detection, 99% healthy, 1% cancer patients.

Model A: always predict "healthy" (zero effort model)
  Accuracy = 99% ← sounds amazing!
  True Positive Rate = 0% ← catches ZERO cancers
  This model kills patients while reporting great accuracy.

Model B: reasonable model (some cancer detection)
  Accuracy = 95%  ← lower than model A!
  True Positive Rate = 80%  ← catches most cancers
  Much more useful despite lower accuracy.

The right metrics for imbalanced data:
  Precision:  Of everything I flagged as cancer, how many actually had cancer?
  Recall:     Of all actual cancer patients, how many did I catch?
  F1:         Harmonic mean of precision and recall
  AUC-ROC:    How well does the model distinguish positives from negatives?
  AUC-PR:     Better than AUC-ROC for severe imbalance (precision-recall curve)
  G-Mean:     √(TPR × TNR) — balanced metric for all classes
```

---

### 18.2 Resampling Strategies

**Oversampling — Create More Minority Examples:**
```
Random Oversampling (simplest):
  Duplicate minority class examples randomly until balanced.
  
  Minority class: 100 cancer cases
  Majority class: 9,900 healthy cases
  → Duplicate each cancer case 99× → 9,900 cancer cases
  
  ✓ Simple, fast
  ✗ Exact duplicates → model memorises them (overfitting)
  ✗ No new information added

SMOTE (Synthetic Minority Over-sampling TEchnique):
  Create SYNTHETIC new minority examples by interpolation.
  
  Algorithm:
  1. For each minority class point x_i:
     a. Find its K nearest neighbours (among minority class examples)
     b. Randomly pick one neighbour x_nn
     c. Create a new synthetic point: x_syn = x_i + λ(x_nn − x_i)
        where λ ~ Uniform(0, 1)
  
  Visualisation:
  
  Minority points (●):          After SMOTE (+ = new synthetic):
  
  ●                             ●
     ●                         + + ●
        ●                   + + + ●
                           + + + +  ●
  
  The synthetic points lie on the LINE SEGMENTS between real minority points.
  They're interpolations, not exact copies → more varied data!
  
  ✓ Creates genuinely new examples
  ✓ Reduces exact duplication / overfitting
  ✗ Can create synthetic examples in overlapping region (ambiguous points)
  ✗ Doesn't consider majority class distribution
  
  Code example:
  from imblearn.over_sampling import SMOTE
  smote = SMOTE(k_neighbors=5, random_state=42)
  X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

ADASYN (Adaptive Synthetic Sampling):
  SMOTE treats all minority points equally.
  ADASYN creates MORE synthetic examples for HARDER minority points
  (those near the decision boundary, surrounded by majority class points).
  
  For each minority point x_i:
    Δᵢ = fraction of K nearest neighbours that are majority class
         (0 = pure minority neighbourhood, 1 = pure majority neighbourhood)
  
  nᵢ = ⌈G × Δᵢ/ΣΔ⌉   (number of synthetics for point i)
  
  Points surrounded by majority class → higher Δᵢ → more synthetics!
  
  Effect: focuses effort on the HARD cases near the decision boundary.
  ✓ Adapts difficulty of generation
  ✓ Can help when minority class has different density regions
  ✗ Can make noisy boundary regions even noisier

Borderline-SMOTE:
  Only generate synthetics for minority points in "DANGER" zone:
  Points that have 1 ≤ (majority neighbours among K) ≤ K-1
  (Some minority, some majority neighbours — near the boundary)
  
  Ignores minority points completely surrounded by minority class
  (They're already well-classified, no need for extra samples there)
  
  Better for well-separated classes than standard SMOTE.
```

**Undersampling — Reduce Majority Examples:**
```
Random Undersampling:
  Randomly remove majority class examples.
  
  ✓ Speeds up training (smaller dataset)
  ✗ Loses potentially useful information from majority class

Tomek Links:
  A Tomek Link is a pair (xᵢ, xⱼ) where:
    - xᵢ is majority class, xⱼ is minority class
    - Each is the nearest neighbour of the other
    - They are "too close" — ambiguous boundary region
  
  Remove the MAJORITY class member of each Tomek link.
  → Cleans the decision boundary (majority class retreats slightly)
  
  Visualisation:
  Before:                         After removing Tomek majority members:
  ● ● ● ● ● ○                    ● ● ● ● ●   ○
          ● ○                              ● ○    ← cleaner boundary
  ●  ●  ●  ○  ○                  ●  ●  ●  ○  ○
  ↑ ambiguous ↑

NearMiss:
  Select majority examples closest to minority class (keep "hard" majority):
  NearMiss-1: majority points with smallest mean distance to 3 nearest minority
  NearMiss-2: majority points with smallest mean distance to 3 FARTHEST minority
  
  Keeps boundary-relevant majority examples, removes "easy" far-away ones.

Combined (SMOTE + Undersampling):
  SMOTE-Tomek: SMOTE first (oversample minority), then remove Tomek links
  SMOTE-ENN:   SMOTE first, then ENN (Edited Nearest Neighbours) to remove
               noisy samples from both classes
```

**Cost-Sensitive Learning:**
```
Instead of resampling, change the LOSS FUNCTION to penalise minority mistakes more.

Class Weights:
  wₖ = n / (K × nₖ)   (K = number of classes, nₖ = count of class k)
  
  Example: 9,900 healthy, 100 cancer (IR = 99)
  w_healthy = 10000 / (2 × 9900) = 0.505
  w_cancer  = 10000 / (2 × 100)  = 50
  
  Misclassifying a cancer patient now has 50/0.505 ≈ 99× higher cost!
  The model is forced to "care" about the minority class.
  
  In sklearn: LogisticRegression(class_weight='balanced')
  In XGBoost: scale_pos_weight = negative_count / positive_count

Focal Loss:
  FL(p) = −α(1−p)^γ log(p)
  
  α = balancing factor (typically 0.25 for the minority class)
  γ = focusing parameter (typically 2)
  
  The (1−p)^γ term is the KEY INNOVATION:
  When the model is confident and correct (p close to 1):
    (1−0.99)^2 = 0.0001 → loss is DOWN-WEIGHTED (already learned this)
  When the model is uncertain/wrong (p close to 0.5):
    (1−0.5)^2 = 0.25 → loss is UP-WEIGHTED (hard example, focus here!)
  
  Effect: automatically down-weights easy examples (abundant majority, easy to classify)
          and up-weights hard examples (minority class near decision boundary)
  
  Why this beats resampling:
  ✓ Doesn't change the training distribution → model learns correct posteriors
  ✓ Adapts dynamically throughout training (changes as model improves)
  ✓ Used in RetinaNet (object detection) — made one-stage detectors competitive
  ✓ Very effective for extreme imbalance (object detection: objects << background)
```

---

## 19. Model Evaluation & Selection

> 💡 **Choosing the right evaluation metric is as important as choosing the right model.** The wrong metric can make a terrible model look good and send your project in the wrong direction. This section covers when to use each metric and why.

---

### 19.1 Classification Metrics — Deep Understanding

**The Confusion Matrix:**
```
For binary classification (positive = minority/important class):

                    Predicted
                  Positive  Negative
Actual  Positive  [  TP   |   FN  ]    TP = True Positive  (correctly caught)
        Negative  [  FP   |   TN  ]    FP = False Positive (false alarm)
                                        FN = False Negative (missed!)
                                        TN = True Negative  (correctly dismissed)

TP = caught cancer patients       (best outcome for patient)
TN = correctly dismissed healthy  (best outcome for healthy person)
FP = flagged healthy as cancer    (unnecessary biopsy, anxiety)
FN = missed cancer patient        (worst outcome — cancer progresses untreated)

The relative costs of FP vs FN define which metrics to optimise!
  Cancer screening: FN is CATASTROPHIC (missed cancer) → optimise recall
  Email spam filter: FP is bad (lost email) → optimise precision
  Fraud detection: both costly → optimise F1 or AUC
```

**Every Metric Explained Intuitively:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
  "What fraction of all predictions were correct?"
  ✗ Misleading for imbalanced classes (see cancer example above)
  ✓ Use when classes are balanced AND errors are equally costly

Precision = TP / (TP + FP)
  "Of everything I said was positive, what fraction actually was?"
  Alternative: "How trustworthy are my positive predictions?"
  
  Low precision: many false alarms → annoying (spam filter lets through good emails)
  High precision: when I say positive, I'm usually right → trustworthy
  
  ✓ When FP is costly: legal review (don't want to flag innocent documents)
  ✓ When acting on positives is expensive (each prediction triggers manual review)

Recall = TP / (TP + FN)  (also called Sensitivity, True Positive Rate)
  "Of all actual positives, what fraction did I catch?"
  Alternative: "How complete is my coverage of positives?"
  
  Low recall: missing many positives → dangerous (missing cancers)
  High recall: catching almost all positives → comprehensive coverage
  
  ✓ When FN is catastrophic: medical diagnosis, fraud detection, security
  ✓ When missing a positive is unacceptable

Specificity = TN / (TN + FP)  (True Negative Rate)
  "Of all actual negatives, what fraction did I correctly dismiss?"
  Complement of False Positive Rate (FPR = 1 − Specificity)

F1 Score = 2 × Precision × Recall / (Precision + Recall)
  Harmonic mean of precision and recall.
  
  Why harmonic mean (not arithmetic)?
  Arithmetic: (P=1.0 + R=0.0)/2 = 0.5 ← seems OK, but R=0 catches nothing!
  Harmonic:   2×1.0×0.0/(1.0+0.0) = 0 ← correctly says this model is useless
  
  The harmonic mean is LOW whenever EITHER component is low.
  It rewards BALANCE between precision and recall.
  
  ✓ When you want to balance precision and recall
  ✓ When class distribution is imbalanced (better than accuracy)
  ✗ When the costs of FP and FN are very different

Fβ = (1+β²) × Precision × Recall / (β²×Precision + Recall)
  β controls the relative importance of recall vs precision.
  β=0.5: precision twice as important (when FP is costly)
  β=1:   F1 (equal importance)
  β=2:   recall twice as important (when FN is costly)
  β=10:  recall much more important (cancer screening → almost never miss!)

MCC (Matthews Correlation Coefficient):
  = (TP×TN − FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]  ∈ [−1, +1]
  
  MCC = +1: perfect classifier
  MCC =  0: random (no better than chance)
  MCC = −1: perfectly wrong (systematically predicts the opposite)
  
  Why MCC is the MOST HONEST single metric:
  MCC = 0 ONLY IF the model has zero predictive power.
  Accuracy = 0.99 for the "predict always majority" model.
  F1 = 0 for a model that predicts no positives.
  MCC = 0 for BOTH of these — it correctly identifies them as useless.
  
  MCC considers all four quadrants (TP, TN, FP, FN) simultaneously.
  
  ✓ Best single metric for binary classification with imbalanced data
  ✓ Particularly recommended when class sizes are very different

AUC-ROC:
  The ROC curve plots:
    Y-axis: TPR (recall) — how many positives we catch
    X-axis: FPR (1 − specificity) — how many negatives we wrongly flag
  
  As you lower the classification threshold (predict more as positive):
    → TPR increases (catch more positives) ← good!
    → FPR increases (more false alarms) ← bad!
  
  AUC = Area Under the ROC Curve.
  
  Interpretation: AUC = P(score(positive) > score(negative))
  = probability that a randomly chosen positive outscores a randomly chosen negative.
  
  AUC = 0.5: random classifier (diagonal line)
  AUC = 0.7: reasonable classifier
  AUC = 0.9: excellent classifier
  AUC = 1.0: perfect (TPR=1 at FPR=0)
  
  Visualisation:
  TPR │              .───●
  1.0 │         .───●
  0.8 │    .───●
  0.6 │.───●                 ← AUC = area under this curve
  0.4 │●
  0.2 │.
  0.0 └─────────────────────
       0   0.2  0.4  0.6  FPR
  
  ✓ Threshold-invariant (evaluates the ranking, not a specific threshold)
  ✓ Good for balanced and mildly imbalanced problems
  ✗ Can be optimistic with severe imbalance (TN >> TP → FPR stays low even with many FP)

AUC-PR (Precision-Recall Curve):
  Y-axis: Precision
  X-axis: Recall
  
  Unlike ROC, PR curve is SENSITIVE to class imbalance.
  When positives are rare: a model must be PRECISE to achieve high precision.
  
  With IR=100 (1% positive):
  A model predicting everything as positive:
    ROC AUC: 0.5 (correctly penalised as random)
    PR AUC: 0.01 ← correctly shows this is terrible! 
             (precision = 1% positive rate = base rate)
  
  ✓ Better than AUC-ROC for severely imbalanced problems
  ✓ Directly shows the precision-recall trade-off
```

**Calibration — Are Probabilities Meaningful?**
```
A model outputs P(cancer) = 0.8 for a patient.
This means: if we see 100 patients with predicted probability 0.8,
            roughly 80 should actually have cancer.

Well-calibrated model:
  When it says 30%: ~30% of those cases actually positive
  When it says 70%: ~70% of those cases actually positive
  → Probabilities can be used for decision-making!

Poorly calibrated model (overconfident):
  When it says 90%: only 60% actually positive
  → Don't trust the probabilities, even if accuracy is high!

Reliability Diagram:
  Group predictions into bins by predicted probability.
  Plot: mean predicted probability vs actual positive rate.
  
  Perfect calibration: diagonal line (y = x)
  
  Positive │           /
  Fraction │         / ← actual model
           │       /
           │     / ← perfect calibration (diagonal)
           │   /
           │ /
           └──────────────────
             0.2  0.4  0.6  0.8  Predicted Probability
  
  If actual fraction < predicted → overconfident (compress probabilities toward 0.5)
  If actual fraction > predicted → underconfident (push probabilities toward extremes)

ECE (Expected Calibration Error):
  ECE = Σ_bins (n_b/n) × |acc(b) − conf(b)|
  
  Weighted average of |accuracy − confidence| per bin.
  Perfect: ECE = 0. Typical well-calibrated model: ECE < 0.05.

Calibration methods:
  Platt Scaling: fit logistic regression on validation set
    P_cal = sigmoid(a × logit + b)   (a, b learned on val set)
  
  Isotonic Regression: non-parametric monotone calibration
    More flexible than Platt, needs more validation data
  
  Temperature Scaling: scale logits by temperature T
    P_cal = softmax(logits / T)
    T > 1: more uncertain, T < 1: more confident
    Single scalar T, tuned by minimising NLL on val set
    Preserves accuracy (doesn't change argmax), only changes probabilities.
    Standard calibration method for deep learning models.
```

---

### 19.2 Cross-Validation — Unbiased Performance Estimation

```
The Problem:
  If you test on the SAME data you trained on, the model looks great.
  Training set performance is always optimistic (the model has memorised it).
  
  Need: performance estimate on DATA THE MODEL HAS NEVER SEEN.

k-Fold Cross-Validation (standard method):
  
  Divide dataset into k equal folds. For k=5:
  
  Fold 1: [Test |Train|Train|Train|Train] → score₁
  Fold 2: [Train| Test|Train|Train|Train] → score₂
  Fold 3: [Train|Train| Test|Train|Train] → score₃
  Fold 4: [Train|Train|Train| Test|Train] → score₄
  Fold 5: [Train|Train|Train|Train| Test] → score₅
  
  Final score = mean(score₁, ..., score₅)
  Uncertainty = std(scores) / √k
  
  Typical choice: k=5 or k=10.
  Larger k: less bias (more training data per fold), more variance (scores vary more)
  Smaller k: more bias (less training data), less variance

Stratified k-Fold (for classification):
  ESSENTIAL for imbalanced datasets!
  Ensure each fold has the SAME class distribution as the full dataset.
  
  Without stratification: one fold might have 0 positive examples → score meaningless!
  
  Always use StratifiedKFold for classification tasks.

Nested Cross-Validation (for hyperparameter tuning):
  
  The problem with naive hyperparameter tuning:
  "Tune on validation set, report test accuracy" → BIASED.
  The hyperparameters are chosen to maximise performance on that validation set.
  The test set has been used (indirectly) in hyperparameter selection.
  
  Correct approach: Nested CV separates model selection from evaluation.
  
  ┌────────────────────────────────────────────────────────────────┐
  │  OUTER FOLD 1 (evaluation):                                   │
  │  ┌───────────────────────────────────┐ ┌──────────────────┐  │
  │  │    Training + Validation Data     │ │   Test Data      │  │
  │  │                                   │ │                  │  │
  │  │  INNER FOLD (hyperparameter):     │ │ Evaluate final   │  │
  │  │  [Val|Tr|Tr|Tr] → pick best hp   │ │ model with       │  │
  │  │  [Tr|Val|Tr|Tr] → pick best hp   │ │ best hyperparams │  │
  │  │  [Tr|Tr|Val|Tr] → pick best hp   │ │ here             │  │
  │  │  → best hyperparams for this fold │ │                  │  │
  │  └───────────────────────────────────┘ └──────────────────┘  │
  │  OUTER FOLD 2, 3, 4, 5 similarly...                           │
  └────────────────────────────────────────────────────────────────┘
  
  Each outer fold: uses inner CV to find best hyperparameters, then evaluates on outer test.
  Final score: average of outer fold test scores = truly unbiased!
  
  Without nested CV: performance estimates can be 5-15% optimistic.

Time Series Cross-Validation:
  NEVER use random k-fold for time series data.
  Reason: future data would be in training set → data leakage!
  
  Correct approach: always train on PAST, test on FUTURE.
  
  Split 1: Train [t₁,...,t₁₀₀]   → Test [t₁₀₁,...,t₁₂₀]
  Split 2: Train [t₁,...,t₁₂₀]   → Test [t₁₂₁,...,t₁₄₀]  (expanding window)
  Split 3: Train [t₁,...,t₁₄₀]   → Test [t₁₄₁,...,t₁₆₀]
  
  Or with a gap between train and test (to prevent leakage from autocorrelation):
  Split 1: Train [t₁,...,t₁₀₀] → Gap [t₁₀₁,...,t₁₀₅] → Test [t₁₀₆,...,t₁₂₀]
```

---

## 20. Regularisation & Optimisation

> 💡 **Regularisation prevents overfitting by constraining the model.** Optimisation finds the model parameters. These two concepts together define the training process for virtually every ML model.

---

### 20.1 Regularisation — Constraining Models to Generalise

**Explicit Regularisation:**
```
L2 (Ridge / Weight Decay):
  Add ||θ||₂² penalty to the loss:
  L_reg(θ) = L(θ) + λ||θ||₂² = L(θ) + λΣⱼ θⱼ²
  
  Effect: shrinks all parameters toward zero.
  Gradient update: θ ← θ − α(∇L + 2λθ) = θ(1 − 2αλ) − α∇L
                                             ↑ weight decay!
  "Weight decay" and "L2 regularisation" are the same for SGD.
  BUT different for Adam! (AdamW fixes this — see earlier section)
  
  Geometric view: adds a sphere constraint ||θ||₂ ≤ t to the optimisation.
  The sphere has no corners → solution is NOT sparse (all weights kept, just shrunk).
  
  Bayesian interpretation: Gaussian prior N(0, 1/(2λ)) on parameters.

L1 (Lasso):
  Add ||θ||₁ penalty:
  L_reg(θ) = L(θ) + λ||θ||₁ = L(θ) + λΣⱼ |θⱼ|
  
  Effect: drives many parameters to EXACTLY ZERO (sparse solution).
  Not differentiable at 0 → use subgradient or proximal gradient methods.
  
  Geometric view: diamond constraint with CORNERS on coordinate axes.
  The loss function's contours touch a corner → sparse solution.
  
  Bayesian interpretation: Laplace prior Laplace(0, 1/λ) on parameters.

Elastic Net:
  α L1 + (1-α) L2 → best of both worlds (sparsity + grouping of correlated features)
  See Section 4.4 for full details.

Dropout (Neural Networks):
  During training: randomly set each activation to 0 with probability (1-p).
  During inference: use all neurons, scale by p (or use inverted dropout at train time).
  
  Why it works:
  1. Prevents co-adaptation: neurons can't rely on specific other neurons always being there.
     Each neuron must be useful on its own → more robust individual features.
  
  2. Ensemble effect: each forward pass uses a different "thinned" network.
     Training 2ⁿ different architectures (exponential in number of neurons).
     Inference averages all these models → ensemble benefit.
  
  3. Data augmentation: different dropout masks → effectively augments training data.
  
  Rate selection:
    p=0.5: original recommendation (50% kept)
    p=0.8: typical for input/embedding layers (less aggressive)
    p=0.1: typical for Transformers (less dropout than in fully connected layers)
  
  Spatial Dropout (for CNNs):
    Drop entire CHANNELS rather than individual neurons.
    Reason: adjacent pixels in a feature map are highly correlated.
    Dropping individual pixels doesn't help much — they can be reconstructed from neighbours.
    Dropping entire channels forces learning different feature detectors.
  
  DropPath / Stochastic Depth (for residual networks):
    Randomly drop entire RESIDUAL BLOCKS during training.
    Effective depth varies during training → implicit ensemble of different depths.
    Improves regularisation for very deep networks.

Early Stopping:
  Monitor validation loss during training.
  When validation loss stops improving (for patience=10 epochs): stop.
  
  Training loss │────────────────────────────────────────
  Validation    │       ╭───────────────────────────────
  loss          │     ╭─╯                          ← overfitting starts here
                │   ╭─╯ ← minimum validation loss
                │ ╭─╯
                │─╯
                └────────────────────────── epochs
                            ↑
                       Stop here! (early stopping point)
                       
  ✓ Free regularisation: no extra hyperparameters to tune
  ✓ Can be combined with any other regularisation
  ✓ Theoretically equivalent to L2 regularisation in some settings
  
  Patience: how many epochs to wait for improvement.
  patience=10: stop if no improvement for 10 consecutive epochs.
  patience too small: stops too early (underfitting).
  patience too large: doesn't prevent overfitting much.
```

**Implicit Regularisation:**
```
SGD's Noise Acts as Regularisation:
  Mini-batch SGD gradient = true gradient + NOISE (from mini-batch sampling).
  This noise has a regularising effect!
  
  SGD prefers FLAT MINIMA (wide basins in the loss landscape):
  If gradient noise pushes you away from a minimum, and that minimum is flat,
  you stay in the low-loss region. If it's sharp, you escape to higher loss.
  
  Flat minimum → better generalisation:
  If the test distribution differs slightly from train (inevitable),
  a flat minimum still has low loss. A sharp minimum might have high test loss
  even with nearly identical train/test distributions.
  
  Practical consequence:
    Larger batch size → less noise → sharper minima → worse generalisation
    Smaller batch size → more noise → flatter minima → better generalisation
  
  The "batch size dilemma":
    Large batch: faster compute (parallelism) BUT worse generalisation
    Small batch: slower compute BUT better generalisation
    
    Fix: linear scaling rule: if you double batch size, double the learning rate.
    This maintains the "effective noise level" while getting some parallelism benefits.
    But this only works up to a point (~10K batch size in practice).
```

---

### 20.2 Learning Rate Schedules

```
Fixed learning rate (no schedule):
  ✗ Too large: oscillates around minimum, never converges precisely
  ✗ Too small: converges very slowly
  ✓ Simple, sometimes sufficient for small problems

Step Decay:
  α(t) = α₀ × γ^{⌊t/k⌋}
  Reduce by factor γ every k steps.
  e.g., α₀=0.1, γ=0.1, k=30000: reduce by 10× every 30K steps.

Cosine Decay (most popular for neural networks):
  α(t) = α_min + (α_max − α_min) × (1 + cos(πt/T)) / 2
  
  Starts high (fast learning), smoothly decreases to α_min.
  
  α │╲
    │  ╲
    │    ╲
    │      ╲
    │        ─────────────
    └────────────────────── t
    T (total steps)
  
  ✓ No step discontinuities (smooth)
  ✓ Cosine shape matches the "easy → hard → easy" nature of learning
  ✓ Standard for image models, language models

Linear Warmup + Cosine Decay (standard for LLMs):
  
  α │            ╲
    │          ╲
    │        ╲
    │      /╲   ╲
    │    /    ╲   ─────────────
    │  /
    │/
    └────────────────────────── t
      ↑warmup↑  ↑ cosine decay ↑
      1000 steps
  
  Warmup: linear increase from 0 to α_max over warmup_steps.
  Decay: cosine decay from α_max to 0 (or small α_min).
  
  Why warmup is essential for large models:
  At the start of training, parameter estimates are very uncertain (random init).
  A large learning rate with noisy gradients → catastrophic updates.
  Warmup gives the optimizer time to gather statistics (Adam's m, v) before
  taking large steps. Then cosine decay reduces step size as we approach optimum.
  
  Warmup steps: typically 1,000-10,000 for LLM pre-training.
  Total steps: hundreds of thousands to millions.

1Cycle Policy (excellent for fast training):
  1. Increase α from α_min to α_max over first 30% of training
  2. Decrease from α_max to α_min over next 70%
  3. Brief period at very low α at the end
  
  Often achieves same accuracy as longer cosine decay in 2-5× fewer epochs!
  Combined with cyclical momentum (high momentum when lr is high, low when lr is low).
```

---

## 21. Probabilistic Machine Learning

> 💡 **Probabilistic methods treat uncertainty as a first-class citizen.** Instead of point predictions, they produce distributions over predictions. Instead of point estimates of model parameters, they produce distributions over parameters. This enables better calibration, uncertainty quantification, and principled decision-making.

---

### 21.1 Gaussian Processes (GP)

> 💡 **A Gaussian Process is a distribution over FUNCTIONS.** Instead of parameterising a specific function (like a neural network), a GP defines a probability distribution over all possible smooth functions. The prior encodes our beliefs about function smoothness. The posterior updates these beliefs after observing data.

**Intuition — From Multivariate Gaussian to Function:**
```
Multivariate Gaussian over finite points:
  For any set of N points x₁,...,xₙ, the function values
  [f(x₁),...,f(xₙ)] ~ N(m, K)
  m = [m(x₁),...,m(xₙ)] (mean vector)
  K = kernel matrix where Kᵢⱼ = k(xᵢ, xⱼ)

A Gaussian Process extends this to INFINITELY MANY points:
  f ~ GP(m(x), k(x, x'))
  
  For any finite set of points, the values are jointly Gaussian.
  The kernel function k(x,x') specifies the covariance between f(x) and f(x').

The kernel encodes STRUCTURE:
  k(x,x') large when x and x' are similar → f(x) and f(x') are correlated
  k(x,x') small when x and x' are distant → f(x) and f(x') are independent

Visualising GP samples (prior and posterior):

PRIOR (before seeing data):
  f(x)
  2│  ╭─╮  ╭─╮        ← sample 1 (random smooth function)
  0│─╯   ╰──  ╰──╮    ← sample 2
 -2│         ╭╮  ╰─   ← sample 3
   └──────────────── x
   (many equally plausible smooth functions)

POSTERIOR (after seeing data points ●):
  f(x)
  2│    ●              ← all samples PASS THROUGH the data!
  0│●     ●   ╭─╮    ← posterior concentrates near data
 -2│  ●       ╰──     
   └──────────────── x
   (uncertainty SHRINKS near data, GROWS away from data)
```

**The Math:**
```
GP Model:
  Prior:      f ~ GP(m(x), k(x,x'))   (often m=0 for simplicity)
  Likelihood: y | f(X) ~ N(f(X), σ²I)  (noisy observations)
  
  Common kernels:
  
  RBF (Radial Basis Function / Squared Exponential):
    k(x,x') = σ² exp(−||x−x'||² / (2ℓ²))
    
    σ² = signal variance (output scale)
    ℓ  = length scale (how fast function varies)
    
    Small ℓ: function changes rapidly (wiggly)
    Large ℓ: function changes slowly (smooth)
    
    → Infinitely differentiable: the smoothest possible functions
    → Use for: smooth physical processes, anything expected to be very smooth
  
  Matérn-5/2:
    k(x,x') = σ²(1 + √5r/ℓ + 5r²/(3ℓ²)) exp(−√5r/ℓ)
    r = ||x−x'||
    → Twice differentiable: less smooth than RBF
    → Use for: most practical applications (better default than RBF!)
    → RBF is "too smooth" for most real data
  
  Periodic:
    k(x,x') = σ² exp(−2sin²(π||x−x'||/p) / ℓ²)
    p = period
    → For seasonal data, cyclic patterns

Posterior (closed form!):
  Given training data (X, y), posterior at test points X*:
  
  f* | X, y, X* ~ N(μ*, Σ*)
  
  μ* = K(X*,X) [K(X,X) + σ²I]⁻¹ y
      ↑ posterior mean (our best guess at test points)
  
  Σ* = K(X*,X*) − K(X*,X) [K(X,X) + σ²I]⁻¹ K(X,X*)
      ↑ posterior variance (our uncertainty at test points)
  
  The SHRINKAGE in variance (K** − correction) is the Schur complement!
  Near training data: Σ* ≈ 0 (highly confident)
  Far from training data: Σ* ≈ K** (as uncertain as the prior)

Computational cost:
  Training: O(n³) — must invert [K(X,X) + σ²I]
  Prediction: O(n²) — matrix-vector products
  
  Bottleneck: cannot scale to n > ~10,000 with exact inference.
  
  Solutions for large n:
  Sparse GPs: use m << n "inducing points" to approximate the full GP
              O(nm² + m³) training, O(m²) prediction
  Random Fourier Features: approximate kernel with random features
                           O(nm + md) training — much cheaper!

Hyperparameter Learning:
  σ², ℓ, σ_noise are HYPERPARAMETERS of the GP.
  Learn them by maximising the log marginal likelihood:
  
  log P(y|X,θ) = −(1/2)yᵀ(K+σ²I)⁻¹y − (1/2)log|K+σ²I| − (n/2)log(2π)
                  ↑                      ↑
               data fit term        complexity penalty
  
  This is AUTOMATIC MODEL SELECTION: simpler functions (large ℓ) are penalised less.
  
  Use gradient descent on this objective over (σ², ℓ, σ_noise).
```

---

### 21.2 Bayesian Optimisation — Finding Optima Without Gradients

> 💡 **Bayesian optimisation is for optimising EXPENSIVE black-box functions** — functions that take a long time to evaluate and where you can't compute gradients. Hyperparameter tuning, drug discovery, materials science: finding the best setting of X when each evaluation costs hours or dollars.

```
Problem: find x* = argmin f(x) where:
  - f is expensive to evaluate (each evaluation = train a model for hours)
  - f is a black box (no gradient available)
  - We have a budget of only N evaluations (N=20-100 typically)

Key idea: Build a SURROGATE MODEL of f using all evaluations so far.
The surrogate is cheap to query → use it to plan the next evaluation wisely.

Algorithm:
  1. Evaluate f at a few initial points (random or Latin hypercube)
  2. Fit a GP to all evaluated points: GP models P(f(x) | observations)
  3. Use the GP to compute an ACQUISITION FUNCTION α(x)
     α(x) = "how worthwhile is it to evaluate f at x next?"
  4. x_next = argmax α(x)  (maximise acquisition function — cheap!)
  5. Evaluate f(x_next) (expensive — real evaluation)
  6. Add (x_next, f(x_next)) to observations
  7. Repeat from step 2

Acquisition Functions:

Expected Improvement (EI) — most popular:
  EI(x) = E[max(0, f_best − f(x))]
  = (f_best − μ(x)) × Φ((f_best − μ(x))/σ(x)) + σ(x) × φ((f_best − μ(x))/σ(x))
  
  Where μ(x) = GP posterior mean, σ(x) = GP posterior std
  Φ, φ = standard normal CDF and PDF
  f_best = best observed value so far
  
  High EI when:
  μ(x) is low (predicted to be good) → EXPLOITATION (go to promising region)
  σ(x) is high (uncertain) → EXPLORATION (learn about unknown region)
  EI naturally balances exploration and exploitation!

Upper Confidence Bound (UCB):
  UCB(x) = −μ(x) + β × σ(x)   (minus sign because minimising f)
  β = exploration weight (large β → more exploration)
  
  Simple, effective, theoretical convergence guarantees.
  Lower β → exploit more (trust current best region)
  Higher β → explore more (visit uncertain regions)

Visualisation of Bayesian Optimisation:
  
  f(x)                     α(x) (acquisition function)
   │     ●         ●        │          ↑ explore here (high uncertainty)
   │  ●     ●   ●           │    ↑           ↑
   │ GP mean ╭──╮           │  exploit   explore
   │ ─────── ╰──╯           │        ╱╲    ╱╲
   └──────────────── x      └──────────────────── x
   Data points ●             Next eval: argmax of α
   Shaded = uncertainty
   
   Each iteration: evaluate at the peak of α, update GP, repeat.

Practical performance:
  For tuning a neural network with 5 hyperparameters:
  Random search: needs ~100 evaluations for a good result.
  Bayesian optimisation: achieves same or better result in 20-30 evaluations.
  
  3-5× more sample efficient than random search!
  Critical when each evaluation = hours of GPU time.

Libraries:
  Optuna: most popular Python library for hyperparameter optimisation
          (TPE algorithm, similar to Bayesian optimisation)
  scikit-optimize (skopt): GP-based Bayesian optimisation
  Ax/BoTorch: Meta's production Bayesian optimisation framework
  SMAC: Hyperband + Bayesian optimisation (used in AutoML)
```

---

### 21.3 Variational Inference — Approximate Bayesian Inference

> 💡 **Exact Bayesian inference is intractable for most models** — the posterior P(θ|data) requires integrating over all parameter values, which is computationally infeasible. Variational inference approximates the posterior with a simpler distribution by turning inference into an optimisation problem.

```
Goal: compute P(θ|data) = P(data|θ) P(θ) / P(data)

Problem: P(data) = ∫ P(data|θ) P(θ) dθ is intractable (high-dimensional integral)

Variational Inference solution:
  Choose a family of tractable distributions q(θ; φ) (e.g., Gaussian).
  Find the parameters φ that make q(θ; φ) as close as possible to P(θ|data).
  
  Closeness measure: KL divergence.
  KL(q || P) = E_q[log q(θ) − log P(θ|data)]
             = E_q[log q(θ) − log P(data,θ)] + log P(data)
  
  Since log P(data) is a constant (we can't compute it, but it doesn't depend on φ):
  minimise KL(q || P) ⟺ maximise ELBO(φ)
  
  ELBO = E_q[log P(data,θ)] − E_q[log q(θ)]
       = E_q[log P(data|θ)] − KL(q(θ) || P(θ))
         ↑                     ↑
    expected likelihood    KL from prior (regularisation)
  
  "Maximise how likely the data is under our approximate posterior,
   while staying close to the prior."

Mean Field Approximation (simplest variational family):
  q(θ) = Πᵢ q(θᵢ)  (all parameters are INDEPENDENT in the approximation)
  
  Each q(θᵢ) optimised separately while fixing others → coordinate ascent.
  Very scalable, but IGNORES correlations between parameters.
  Can severely underestimate posterior uncertainty when parameters are correlated.

Amortised Variational Inference (VAE):
  Instead of optimising φ per data point, LEARN a function that maps data to φ.
  
  Encoder: φ(x) = (μ(x), σ(x)) (neural network outputs posterior parameters)
  The encoder "amortises" inference — shares computation across data points.
  
  This is the core of the VAE (Variational Autoencoder):
    Encoder (inference network): x → q(z|x) = N(μ(x), σ²(x))
    Decoder (generative network): z → P(x|z)
    
    ELBO = E_q[log P(x|z)] − KL(q(z|x) || P(z))
    
    Reparameterisation trick makes this differentiable:
    z = μ(x) + σ(x) × ε,   ε ~ N(0,I)
    → Can backpropagate through the sampling operation!
```

---

## Quick Reference — Architecture Decision Guide

```
CHOOSING THE RIGHT MODEL:

Task Type                   → Recommended Models
──────────────────────────────────────────────────────────────────────────
Tabular regression          → XGBoost/LightGBM (first try), Linear Regression
Tabular classification      → XGBoost/LightGBM/CatBoost, Random Forest
Image classification        → ViT (large data), ResNet/EfficientNet (medium data)
Image detection/segmentation→ YOLO, Mask-RCNN, Segment Anything Model
Text classification         → BERT fine-tuned (few thousand examples)
Text generation             → GPT-family decoder-only LLM
Text summarisation          → T5/BART fine-tuned, or LLM with RAG
Question answering          → RAG + LLM, or BERT for extractive QA
Named entity recognition    → BERT + CRF or token classification
Machine translation         → Encoder-Decoder Transformer (mBART, M2M)
Time series forecasting     → Transformer (Temporal Fusion Transformer), N-HiTS
Anomaly detection           → Isolation Forest, Autoencoder, DBSCAN
Clustering                  → K-Means (spherical), DBSCAN (arbitrary shape), GMM
Dimensionality reduction    → PCA (linear), UMAP (non-linear, visualisation)
Recommendation              → Matrix Factorisation, Two-Tower neural, LLM
Chat assistant              → Fine-tuned LLM + RAG for knowledge grounding
Code generation             → Decoder-only LLM (CodeLlama, StarCoder, GPT-4)
Multimodal (image+text)     → LLaVA, GPT-4V, Gemini (vision-language models)

CHOOSING THE RIGHT EVALUATION METRIC:
  Balanced classes, equal error costs:          → Accuracy
  Imbalanced classes:                           → F1, MCC, AUC-PR
  When false negatives are catastrophic:        → Recall (maximise sensitivity)
  When false positives are costly:              → Precision
  Want one honest metric for imbalanced data:   → MCC
  Threshold-independent, balanced:              → AUC-ROC
  Severe imbalance (IR > 10):                   → AUC-PR (AUC-ROC is optimistic)
  Regression:                                   → RMSE (penalise large errors)
                                                → MAE (robust to outliers)
  Sequence generation quality:                  → BLEU (translation), ROUGE (summarisation)
  LLM quality (human preference):               → RLHF reward, GPT-4-as-judge

CHOOSING THE RIGHT OPTIMIZER:
  Standard deep learning:                        → AdamW (β₁=0.9, β₂=0.999, ε=1e-8)
  Transformers/LLMs:                             → AdamW + linear warmup + cosine decay
  Fine-tuning with PEFT:                         → AdamW or Adam (for adapters/LoRA)
  Computer vision (Torchvision models):           → SGD + momentum (often beats Adam!)
  Convex optimisation / simple models:           → SGD or Adam
  Memory-constrained training:                   → Lion (uses only sign of gradients)

COMMON MISTAKES AND HOW TO AVOID THEM:
  1. Training-test data leakage → Always split BEFORE any preprocessing
  2. Using accuracy for imbalanced data → Use F1, MCC, or AUC-PR instead
  3. Not stratifying cross-validation → Use StratifiedKFold for classification
  4. Hyperparameter tuning without nested CV → Use nested CV for unbiased estimates
  5. Using full fine-tuning when LoRA suffices → Try LoRA first for large models
  6. No warmup for large model training → Always use warmup (1000+ steps)
  7. Trusting LLM outputs without verification → Always implement RAG + citation
  8. Not normalising before PCA → PCA requires standardised features!
  9. Using random k-fold for time series → Must use temporal split
  10. Forgetting to calibrate probabilities → Apply temperature scaling on val set
```

---

## References

### Landmark Papers Referenced in This Guide

- Vaswani et al. (2017) — *Attention Is All You Need* — Original Transformer
- Devlin et al. (2018) — *BERT: Pre-training of Deep Bidirectional Transformers*
- Radford et al. (2018, 2019, 2020) — *GPT-1, GPT-2, GPT-3*
- He et al. (2015) — *Deep Residual Learning for Image Recognition (ResNet)*
- Dosovitskiy et al. (2020) — *An Image is Worth 16×16 Words (ViT)*
- Brown et al. (2020) — *Language Models are Few-Shot Learners (GPT-3)*
- Hoffmann et al. (2022) — *Training Compute-Optimal Language Models (Chinchilla)*
- Hu et al. (2021) — *LoRA: Low-Rank Adaptation of Large Language Models*
- Dettmers et al. (2023) — *QLoRA: Efficient Finetuning of Quantized LLMs*
- Dao et al. (2022) — *FlashAttention: Fast and Memory-Efficient Attention*
- Ouyang et al. (2022) — *Training language models to follow instructions (InstructGPT/RLHF)*
- Rafailov et al. (2023) — *Direct Preference Optimisation (DPO)*
- Mikolov et al. (2013) — *Word2Vec: Efficient Estimation of Word Representations*
- Pennington et al. (2014) — *GloVe: Global Vectors for Word Representation*
- Bojanowski et al. (2017) — *FastText: Enriching Word Vectors with Subword Information*
- Chawla et al. (2002) — *SMOTE: Synthetic Minority Over-sampling Technique*
- Lin et al. (2017) — *Focal Loss for Dense Object Detection (RetinaNet)*
- Wei et al. (2022) — *Chain-of-Thought Prompting Elicits Reasoning in LLMs*
- Wei et al. (2022) — *Emergent Abilities of Large Language Models*
- Yao et al. (2022) — *ReAct: Synergizing Reasoning and Acting in LLMs*
- Guo et al. (2017) — *On Calibration of Modern Neural Networks*
- Radford et al. (2021) — *Learning Transferable Visual Models from Natural Language (CLIP)*
- Liu et al. (2023) — *LLaVA: Visual Instruction Tuning*
- Jiang et al. (2023) — *Mistral 7B*
- Jiang et al. (2024) — *Mixtral of Experts*
- Karpathy (2022) — *nanoGPT: building GPT from scratch*

### Resources for Going Deeper

- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — Model hub and library
- [LangChain Docs](https://docs.langchain.com) — Framework documentation
- [LangSmith](https://smith.langchain.com) — LLM observability platform
- [Papers With Code](https://paperswithcode.com) — SOTA benchmarks with implementations
- [Andrej Karpathy's YouTube](https://www.youtube.com/@AndrejKarpathy) — Best neural network tutorials
- [Jay Alammar's Blog](https://jalammar.github.io) — Illustrated explanations of Transformers
- [Lilian Weng's Blog](https://lilianweng.github.io) — Deep technical ML posts
- [Distill.pub](https://distill.pub) — Interactive visual ML explanations
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Must-read visual guide
- [fast.ai](https://www.fast.ai) — Top-down practical deep learning course
- [CS224N (Stanford NLP)](https://web.stanford.edu/class/cs224n/) — Best NLP course
- [CS231N (Stanford CV)](http://cs231n.stanford.edu/) — Best computer vision course

---

*Theory → Intuition → Math → Visualisation → Code → Production*
*Sections 15-21 — The Complete Reference for Transformers, NLP, and Production LLM Engineering*
*by @ravidaliparthy — Extended with NLP and Beginner-Friendly Explanations*
