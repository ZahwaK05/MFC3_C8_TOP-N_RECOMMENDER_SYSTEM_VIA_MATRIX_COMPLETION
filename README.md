#  📃 Top-N Recommender System via Matrix Completion

![MATLAB](https://img.shields.io/badge/MATLAB-Used-orange)
![Algorithm](https://img.shields.io/badge/ADMM-Optimization-blue)
![Model](https://img.shields.io/badge/Matrix%20Completion-LogDet-green)

## 📌 Overview

This project implements a **Top-N Recommender System** using **Matrix Completion with LogDet optimization**.

The goal is to predict missing entries in a sparse **user–item matrix** and generate accurate recommendations.

Recommender systems are widely used in platforms like Netflix, Amazon, and Spotify.

## 🧠 Key Highlights

- LogDet surrogate instead of Nuclear Norm  
- ADMM optimization for efficient solving  
- Handles highly sparse datasets  
- Evaluated using HR@N and ARHR@N

## 👥 Team C-8

| Name | Roll No |
|------|--------|
| Sai Reddy | CB.SC.U4AIE24205 |
| Devana | CB.SC.U4AIE24213 |
| Tharappel Manas | CB.SC.U4AIE24257 |
| Zahwa K | CB.SC.U4AIE24261 |

## 🎯 Problem Statement

- User–item matrices are sparse  
- Many ratings are missing  
- Need to predict missing values for Top-N recommendations  

### Challenges:
- Rank minimization is NP-hard  
- Nuclear norm over-penalizes  
- LogDet provides better approximation

## ⚙️ Methodology

```mermaid
graph LR
A[Input Matrix] --> B[Mask]
B --> C[ADMM Optimization]
C --> D[Completed Matrix]
D --> E[Top-N Recommendation]
E --> F[Evaluation HR/ARHR]

```
---

# 🧮 OPTIMIZATION MODEL
```md
🧮 Optimization Model

We recover matrix **X** from observed matrix **M** using:

- LogDet as a rank approximation  
- ADMM optimization  

### Steps:
- X Update → SVD + shrinkage  
- Y Update → constraint enforcement  
- Z Update → dual update
```
---

## 📊 Datasets

| Dataset | Domain | Users | Items | Density |
|--------|--------|------|------|--------|
| MovieLens | Movies | 943 | 1682 | 6.3% |
| LastFM | Music | 1892 | 17632 | 0.28% |
| Delicious | Bookmark | 2078 | 12096 | 0.20% |
| BX | Books | 2078 | 9300 | 0.50% |
| Netflix | Movies | 5000 | 17000 | 0.35% |

## 📈 Performance Metrics

### 🔹 Hit Ratio (HR@N)
Measures whether the correct item appears in Top-N recommendations.

### 🔹 ARHR@N
Rewards higher-ranked correct recommendations.

## 📊 Results

| Dataset | HR@10 | ARHR@10 |
|--------|------|--------|
| MovieLens | 0.9466 | 0.4197 |
| LastFM | 0.2573 | 0.1100 |
| Delicious | 0.2366 | 0.0943 |
| BX | 0.7365 | 0.2976 |
| Netflix | 0.3597 | 0.0721 |

## 📂 Project Structure

Project contents: :contentReference[oaicite:0]{index=0}

## 🛠️ How to Run

```bash
1. Open MATLAB
2. Load .mlx file
3. Run preprocessing
4. Execute ADMM model
5. Evaluate results

```
---

# 📚 REFERENCES
```md


- Ning & Karypis — SLIM Recommender Systems  
- Shi & Yu — Matrix Completion Limitations  
- Kang et al. — LogDet Rank Minimization

```
---
## 🌟 Conclusion

- LogDet improves rank approximation  
- ADMM ensures efficient optimization  
- Strong performance on sparse datasets

