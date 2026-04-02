# Top-N Recommender System via Matrix Completion

<p align="center">
   <img src="images/logo.jpg" alt="Amrita Vishwa Vidyapeetham" width="760"/>
</p>

**School of Artificial Intelligence — Amrita Vishwa Vidyapeetham**  
*A LogDet-Based Matrix Completion Approach for Collaborative Filtering*

**Group 8** |  
Sai Reddy (CB.SC.U4AIE24205) |  
Devana (CB.SC.U4AIE24213) |  
Manas (CB.SC.U4AIE24257) |  
Zahwa K (CB.SC.U4AIE24261)


---

## Table of Contents
1. [Project Title — Word-by-Word Explanation](#1-project-title--word-by-word-explanation)
2. [Problem Statement](#2-problem-statement)
3. [Why LogDet Instead of Nuclear Norm?](#3-why-logdet-instead-of-nuclear-norm)
4. [Optimization Problem](#4-optimization-problem)
5. [Augmented Lagrangian Formulation](#5-augmented-lagrangian-formulation)
6. [ADMM — Alternating Direction Method of Multipliers](#6-admm--alternating-direction-method-of-multipliers)
7. [X-Update: SVD-Based Proximal Step](#7-x-update-svd-based-proximal-step)
8. [Y-Update and Z-Update](#8-y-update-and-z-update)
9. [Methodology Overview](#9-methodology-overview)
10. [Performance Metrics](#10-performance-metrics)
11. [Datasets](#11-datasets)
12. [Results](#12-results)
13. [MATLAB Implementation Notes](#13-matlab-implementation-notes)
14. [Glossary](#14-glossary)
15. [References](#15-references)

---

## 1. Project Title — Word-by-Word Explanation
**Top-N Recommender System via Matrix Completion** — each part is precise:

| Word / Phrase          | Meaning |
|------------------------|---------|
| **Top-N**              | The system recommends the top N items most likely to interest a user, rather than predicting an exact rating score |
| **Recommender System** | An information filtering pipeline that suggests relevant items (movies, books, music) based on past user behaviour |
| **via**                | The recommendation is achieved through — i.e., powered by — the following mathematical technique |
| **Matrix Completion**  | The task of recovering missing entries in a user–item rating matrix using low-rank assumptions |

### The Core Intuition: Filling a Sparse Matrix
Imagine a giant grid where rows are users, columns are items (movies, songs, books), and cells contain ratings. Most cells are **empty** — users only rate a tiny fraction of all items:

```
         Item1 Item2 Item3 Item4 Item5
User1 [  5  ?   3   ?   ?  ]
User2 [  ?   4   ?   2   ?  ]
User3 [  ?   ?   5   ?   1  ]
User4 [  2   ?   ?   4   ?  ]
```

The `?` entries are what we want to predict. Once filled, we recommend the highest-predicted items the user has not yet seen.  
**Matrix Completion** recovers these missing values by assuming the true matrix has **low rank** — meaning user preferences can be described by a small number of latent factors (genre, tempo, style, etc.).

---

## 2. Problem Statement
### Limitations of Existing Approaches

| Problem                        | Description |
|--------------------------------|-------------|
| **Sparse Rating Matrix**       | Users rate only a tiny fraction of items; direct computation is unreliable on missing data |
| **NP-Hard Rank Minimisation**  | Finding the lowest-rank matrix that fits the observed ratings is computationally intractable |
| **Nuclear Norm Over-Penalises**| The nuclear norm (sum of singular values) is the standard convex relaxation of rank, but it uniformly shrinks all singular values — including large, important ones — leading to biased solutions |
| **Poor Recommendation Quality**| Over-regularised matrices miss nuanced user preferences and rank relevant items poorly |

> **This project addresses all of these** by replacing nuclear norm minimisation with a **LogDet surrogate**, which penalises small singular values more heavily than large ones — preserving important structure while still promoting low rank.

---

## 3. Why LogDet Instead of Nuclear Norm?
### Comparing Surrogates

| Property                        | Nuclear Norm $\|X\|_*$          | LogDet $\log\det((X^\top X)^{1/2} + I)$ |
|--------------------------------|----------------------------------|-------------------------------------------|
| Convex?                        | Yes                              | Non-convex (tighter)                     |
| Penalises large singular values? | Yes — uniformly                 | No — only small ones                     |
| Approximation quality of rank  | Loose                            | Tight                                     |
| Bias on dominant factors       | High                             | Low                                       |
| Practical recommendation quality | Moderate                      | Superior                                  |

**LogDet formula:**  
$$\log\det\left((X^\top X)^{1/2} + I\right) = \sum_i \log(1 + \sigma_i(X))$$

---

## 4. Optimization Problem
### Full Objective
$$\min_{X \geq 0} \; \log\det\left((X^\top X)^{1/2} + I\right) \;+\; \frac{\mu}{2} \left\| I_\Omega \odot (X - M) \right\|_F^2$$

### Symbol-by-Symbol Explanation

| Symbol       | Dimension | Meaning |
|--------------|-----------|---------|
| $X$          | $m \times n$ | Completed user–item rating matrix we are solving for |
| $M$          | $m \times n$ | Observed (sparse) ratings matrix from the dataset |
| $\Omega$     | —         | Index set of observed (known) rating entries |
| $I_\Omega$   | $m \times n$ | Indicator matrix: $1$ where a rating is observed, $0$ otherwise |
| $\odot$      | —         | Element-wise (Hadamard) product |
| $\|\cdot\|_F$| —         | Frobenius norm — total energy across all matrix entries |
| $\mu$        | scalar    | Penalty parameter controlling fidelity to observed ratings |
| $\log\det(\cdot)$ | scalar | LogDet surrogate approximating the rank of $X$ |
| $\sigma_i(X)$| scalar    | $i$-th singular value of $X$ |

### Two-Term Interpretation
**Term 1 — LogDet regulariser:**  
$$\log\det\left((X^\top X)^{1/2} + I\right) = \sum_i \log(1 + \sigma_i(X))$$  
Encourages $X$ to be low-rank by penalising the number and magnitude of non-zero singular values.

**Term 2 — Data fidelity:**  
$$\frac{\mu}{2} \left\| I_\Omega \odot (X - M) \right\|_F^2$$  
Penalises deviation from known ratings — only at the observed positions.

---

## 5. Augmented Lagrangian Formulation
### Why ADMM?
The objective has two complicating constraints:
- $X \geq 0$ (non-negativity of ratings)
- The LogDet term couples all singular values together

**ADMM** splits these into manageable sub-problems by introducing an auxiliary variable $Y$ (to handle the non-negativity constraint separately) and a dual variable $Z$ (to enforce $X = Y$).

### Augmented Lagrangian
$$\mathcal{L}_\mu(X, Y, Z) = \sum_i \log(1 + \sigma_i(X)) \;+\; \frac{\mu}{2} \left\| X - Y + \frac{Z}{\mu} \right\|_F^2 + \iota_{\mathbb{R}_+}(Y)$$

where:
- $\iota_{\mathbb{R}_+}(Y) = 0$ if $Y \geq 0$ entry-wise, else $+\infty$ — this enforces non-negativity through the $Y$-update
- $Z$ is the **dual variable** tracking the constraint violation $X - Y$
- $\mu$ and $\rho$ are penalty parameters controlling how tightly $X$ and $Y$ are coupled

---

## 6. ADMM — Alternating Direction Method of Multipliers
### Three-Variable Split
ADMM converts the hard constrained problem into three simple alternating updates:

```
Initialise: X⁰ = 0, Y⁰ = 0, Z⁰ = 0
Repeat until convergence:
    ┌─────────────────────────────────────────────────┐
    │ Step 1 — X-update: minimise LogDet + quadratic  │
    │ (closed-form via SVD)                           │
    ├─────────────────────────────────────────────────┤
    │ Step 2 — Y-update: project onto non-negative    │
    │ orthant (element-wise max with 0)               │
    ├─────────────────────────────────────────────────┤
    │ Step 3 — Z-update: dual ascent on residual      │
    │ Z ← Z + μ(X - Y)                                │
    └─────────────────────────────────────────────────┘
```

### Why This Works
Each sub-problem has a clean closed-form solution. Jointly, they converge to the solution of the original constrained optimisation problem. The dual variable $Z$ accumulates constraint violations, gradually forcing $X$ and $Y$ to agree.

---

## 7. X-Update: SVD-Based Proximal Step

The X-update solves the sub-problem

$$X^{t+1} = \arg\min_X \ \log\det\left((X^\top X)^{1/2} + I\right) + \frac{\mu_t}{2} \|X - D\|_F^2$$

where $D = Y^t - Z^t / \mu_t$.

### Step-by-Step Closed-Form Solution

**Step 1 — SVD of the centre**

$$D = U \Sigma V^\top, \qquad \sigma_D \in \mathrm{diag}(\Sigma)$$

**Step 2 — 1-D proximal operator per singular value**

For each singular value $\sigma_D$, solve independently:

$$s^* = \arg\min_{s \geq 0} \ \log(1 + s) + \rho(s - \sigma_D)^2, \qquad \rho = \frac{\mu_t}{2}$$

**Step 3 — First-order optimality condition**

Setting the derivative to zero:

$$\frac{1}{1+s} + 2\rho(s - \sigma_D) = 0$$

**Step 4 — Quadratic equation**

Multiply through by $(1+s)$ and rearrange:

$$s^2 + (1 - \sigma_D)s + \left(\frac{1}{2\rho} - \sigma_D\right) = 0$$

This is a standard quadratic $s^2 + bs + c = 0$ with:

- $b = 1 - \sigma_D$
- $c = \dfrac{1}{2\rho} - \sigma_D$

**Step 5 — Discriminant and roots**

$$\Delta = b^2 - 4c = (1 - \sigma_D)^2 - 4\left(\frac{1}{2\rho} - \sigma_D\right)$$

$$s_{1,2} = \frac{-b \pm \sqrt{\Delta}}{2} = \frac{-(1 - \sigma_D) \pm \sqrt{\Delta}}{2}$$

Keep the **non-negative root** that actually decreases the objective value; if no such root exists, set $s^* = 0$.

> **Note on the MATLAB code:** In `prox_logdet_1d.m`, the roots are computed as `(-b ± sqrt(disc))/2` which matches the formula above directly since `a = 1`. Both candidate roots are evaluated against the objective and the minimiser is selected. If neither candidate is non-negative, `s = 0` is returned. This is the correct implementation.

### Numerical Example

Suppose $D$ has singular values $\sigma_D \in \{3.2,\ 1.5,\ 0.4\}$ and $\rho = 0.03$:

- For $\sigma_D = 3.2$: the quadratic yields $s^* \approx 2.9$ — shrunken slightly, dominant factor preserved.
- For $\sigma_D = 0.4$: the positive root does not decrease the objective, so $s^* = 0$ — this singular value is zeroed out, promoting low rank.

Large singular values (dominant user–item factors) survive; small noisy ones are suppressed — exactly what low-rank completion requires.

## 8. Y-Update and Z-Update
### Y-Update — Non-Negativity Projection
$$Y^{t+1} = \arg\min_{Y} \; \iota_{\mathbb{R}_+}(Y) + \frac{\mu_t}{2} \left\| X^{t+1} - Y + \frac{Z^t}{\mu_t} \right\|_F^2$$

Closed form — element-wise clamp to zero:  
$$Y^{t+1} = \max\left(X^{t+1} + \frac{Z^t}{\mu_t},\ 0\right)$$

**Intuition:** Ratings cannot be negative. Any negative entries in the updated $X$ are projected to zero through $Y$.

### Z-Update — Dual Ascent
$$Z^{t+1} = Z^t + \mu_t \left(X^{t+1} - Y^{t+1}\right)$$

**Intuition:** $Z$ accumulates the mismatch between $X$ and $Y$ over iterations. As ADMM converges, $X \to Y$ and the mismatch $X - Y \to 0$.

### Parameter Update
$$\mu_{t+1} = \kappa \cdot \mu_t$$  
where $\kappa > 1$ is a growth factor. Increasing $\mu$ over iterations tightens the coupling between $X$ and $Y$, forcing faster convergence.

---

## 9. Methodology Overview
The full pipeline runs as follows:

```
┌──────────────────┐
│ Input Matrix M   │ (sparse user–item ratings)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Apply Mask IΩ    │ (identify observed vs missing entries)
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ ADMM Optimisation Loop               │
│                                      │
│ Repeat until convergence:            │
│ 1. X-update (SVD + 1-D prox)         │
│ 2. Restore X_Ω = M_Ω                 │
│ 3. Y-update (max with 0)             │
│ 4. Z-update (dual ascent)            │
│ 5. μ ← κ · μ                         │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────┐
│ Completed Matrix │ (all entries filled)
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ Top-N Recommendation                 │
│ For each user: rank unrated items    │
│ by predicted score, return top N     │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ Evaluation: HR@N and ARHR@N          │
└──────────────────────────────────────┘
```

---

## 10. Performance Metrics

### Hit Ratio (HR@N)

**What it measures:** Whether the system successfully retrieves the item a user actually liked within the Top-N list.

**Definition:** For each user, one item is held out as the ground-truth test item (leave-one-out protocol). HR@N equals 1 for a user if their test item appears anywhere in their personal Top-N recommended list, and 0 otherwise. HR@N is the average over all users:

$$\text{HR@}N = \frac{1}{|U|} \sum_{u \in U} \mathbf{1}\{p_u \in \text{Top-}N(u)\}$$

where $p_u$ is the held-out test item for user $u$, and $\text{Top-}N(u)$ is the set of top N items recommended to that user.

**Interpretation:** HR@10 = 0.9466 means that for 94.66% of users, the item they actually liked appears somewhere in their Top-10 list. HR ranges from 0 (no hits) to 1 (perfect retrieval for every user). It does **not** care about where in the list the item appears — rank 1 and rank 10 both count equally.

---

### Average Reciprocal Hit Rank (ARHR@N)

**What it measures:** Not just whether the test item was retrieved, but how highly it was ranked — rewarding the system for placing the correct item near the top.

**Definition:** For each user, if their test item $p_u$ appears at rank position $k$ within their Top-N list, it contributes $1/k$ to the score. If it does not appear in the Top-N list at all, it contributes 0. ARHR@N is the average over all users:

$$\text{ARHR@}N = \frac{1}{|U|} \sum_{u \in U} \frac{1}{\text{rank}_u(p_u)} \quad \text{(contributes 0 if } p_u \notin \text{Top-}N(u)\text{)}$$

**Interpretation:**  
- Rank 1 → contributes 1.0 (best possible)  
- Rank 5 → contributes 0.2  
- Rank 10 → contributes 0.1  
- Not in Top-N → contributes 0  

ARHR is always $\leq$ HR for the same N. A high HR with a low ARHR means the system finds the right item but often buries it near the bottom of the list. A high ARHR means the system consistently ranks the correct item near the top.

---

### Key Difference Between HR and ARHR

| Scenario | HR@10 | ARHR@10 |
|----------|-------|---------|
| Test item at rank 1 for every user | 1.0 | 1.0 |
| Test item at rank 10 for every user | 1.0 | 0.1 |
| Test item found for 50% of users at rank 1 | 0.5 | 0.5 |
| Test item not found for any user | 0.0 | 0.0 |

Both metrics are reported at cutoff N = 10 in this project.

---

## 11. Datasets
All datasets are used in their standard leave-one-out evaluation protocol: one rating per user is held out for testing, the rest are used for training.

| Dataset              | Domain      | Users  | Items   | Interactions | Density    |
|----------------------|-------------|--------|---------|--------------|------------|
| **MovieLens**        | Movies      | 943    | 1,682   | 100,000      | approx 6.30%    |
| **LastFM**           | Music       | 1,892  | 17,632  | 92,834       | approx 0.28%    |
| **Delicious**        | Bookmarks   | 2,078  | 12,096  | 437,000      | approx 0.20%    |
| **BX (BookCrossing)**| Books       | 2,078  | 9,300   | 96,000       | approx 0.50%    |
| **Netflix Subset**   | Movies      | 5,000  | 17,000  | 300,000      | approx 0.35%    |

### Data Sources
- HetRec 2011 (MovieLens + LastFM + Delicious): [GroupLens HetRec](https://grouplens.org/datasets/hetrec-2011/)
- Book-Crossing Dataset: [BX Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)
- Netflix Prize data: [Netflix Prize](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)

---

## 12. Results
### Performance Summary

| Dataset              | HR@10   | ARHR@10 | $\mu$ (initial) | $\rho$ | $\kappa$ |
|----------------------|---------|---------|-----------------|--------|----------|
| **MovieLens**        | 0.9466  | 0.4197  | 0.06            | 0.06   | 2        |
| **LastFM**           | 0.2573  | 0.1100  | 0.010           | 1      | 1        |
| **Delicious**        | 0.2366  | 0.0943  | 0.010           | 1      | 1        |
| **BX (BookCrossing)**| 0.7365  | 0.2976  | 0.06            | 0.06   | 2        |
| **Netflix Subset**   | 0.3597  | 0.0721  | 0.06            | 0.06   | 2        |

### Result Analysis
**MovieLens achieves the highest performance** — consistent with its higher density.  
**LastFM and Delicious are the hardest** (very sparse) but the model still extracts useful low-rank structure.  
**BX** performs strongly thanks to richer explicit ratings.  
**Netflix Subset** is the largest scalability test.

**Parameter Sensitivity**  
Dense datasets benefit from $\kappa=2$ (fast penalty growth). Sparse datasets use $\kappa=1$ (constant penalty) to avoid over-penalising.

---

## 13. MATLAB Implementation Notes
All code is implemented in **MATLAB**. Key files:

```
project/
├── main.m
├── admm_logdet.m
├── x_update.m
├── prox_logdet_1d.m
├── top_n_recommend.m
├── evaluate.m
├── load_*.m          (one per dataset)
└── datasets/
```

### Core ADMM Loop (excerpt)
```matlab
function X = admm_logdet(M, mask, mu, rho, kappa, max_iter)
    X = zeros(size(M)); Y = zeros(size(M)); Z = zeros(size(M));
    for t = 1:max_iter
        D = Y - Z / mu;
        X = x_update(D, mu);
        X(mask) = M(mask);           % restore observed entries
        Y = max(X + Z / mu, 0);
        Z = Z + mu * (X - Y);
        mu = kappa * mu;
    end
end
```

### X-Update via SVD and 1-D Proximal
```matlab
function X = x_update(D, mu)
    [U, S, V] = svd(D, 'econ');
    sigma_D = diag(S);
    rho = mu / 2;
    s_star = zeros(size(sigma_D));
    for i = 1:length(sigma_D)
        s_star(i) = prox_logdet_1d(sigma_D(i), rho);
    end
    X = U * diag(s_star) * V';
end
```

### Scalar Proximal Solve
The 1-D proximal operator solves $\min_{s \geq 0} \log(1+s) + \rho(s - \sigma_D)^2$ via the quadratic $s^2 + (1-\sigma_D)s + (1/(2\rho) - \sigma_D) = 0$:

```matlab
function s = prox_logdet_1d(sigma_D, rho)
    % Quadratic coefficients: s^2 + b*s + c = 0
    b = 1 - sigma_D;
    c = 1/(2*rho) - sigma_D;
    disc = b^2 - 4*c;      % discriminant (a=1, so -4*a*c = -4*c)
    if disc < 0
        s = 0; return;
    end
    r1 = (-b - sqrt(disc)) / 2;
    r2 = (-b + sqrt(disc)) / 2;
    cand = [r1, r2];
    cand = cand(cand >= 0);   % keep only non-negative roots
    if isempty(cand)
        s = 0;
    else
        obj = @(sv) log(1 + sv) + rho * (sv - sigma_D)^2;
        [~, idx] = min(arrayfun(obj, cand));
        s = cand(idx);
    end
end
```

### Running the Full Experiment (example for MovieLens)
```matlab
[M, mask] = load_movielens('datasets/ml-100k/u.data');
X = admm_logdet(M, mask, 0.06, 0.06, 2, 100);
[hr, arhr] = evaluate(X, M, mask, 10);
```

---

## 14. Glossary
| Term                          | Meaning |
|-------------------------------|---------|
| **Top-N Recommendation**      | Recommending N items most likely to be of interest to a user |
| **Matrix Completion**         | Recovering missing entries in a partially-observed matrix under low-rank assumptions |
| **LogDet Surrogate**          | $\log\det((X^\top X)^{1/2} + I) = \sum_i \log(1 + \sigma_i(X))$ — tighter non-convex approximation of rank |
| **ADMM**                      | Alternating Direction Method of Multipliers — an optimisation algorithm that splits a hard problem into alternating sub-problems, each with a closed-form solution |
| **HR@N**                      | Hit Ratio at N — fraction of users for whom the held-out test item appears anywhere in their Top-N list |
| **ARHR@N**                    | Average Reciprocal Hit Rank at N — like HR but weighted by rank position; higher rank = higher reward |
| **Leave-One-Out**             | Evaluation protocol: hold out one rating per user for testing, train on the rest |
| **Singular Value**            | A measure of the strength of each latent factor in a matrix; large values encode dominant user–item patterns |
| **Low-Rank**                  | A matrix property meaning its information can be captured by few latent factors; enables completion from sparse observations |

---

## 15. References
**Primary paper implemented:**
> Z. Kang, C. Peng, J. Cheng, and Q. Cheng, "LogDet rank minimization with application to subspace clustering," *Computational Intelligence and Neuroscience*, vol. 2015, Art. no. 824289, 2015. doi: 10.1155/2015/824289

**Additional references:**
> X. Ning and G. Karypis, "SLIM: Sparse linear methods for top-N recommender systems," in *2011 IEEE 11th International Conference on Data Mining*, 2011, pp. 497–506. doi: 10.1109/ICDM.2011.134

> X. Shi and P. S. Yu, "Limitations of matrix completion via trace norm minimization," *ACM SIGKDD Explorations Newsletter*, vol. 12, no. 2, pp. 16–24, 2010. doi: 10.1145/1964897.1964902

**Official code repository:**  
[sckangz/recom_mc](https://github.com/sckangz/recom_mc)
