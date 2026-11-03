<div align="center">

[![中文](https://img.shields.io/badge/🇨🇳_中文-ff6b4a?style=for-the-badge&labelColor=fff7ed)](./README.md)
[![English](https://img.shields.io/badge/🇬🇧_English-1f3b73?style=for-the-badge&labelColor=f8fafc)](./README_EN.md)

</div>
# Rednote Qilin Search & Recommendation Personal Project

![Qilin Cover](./Qilin_Cover_Image.png)

Qilin is a personal project for an integrated **search + recommendation** system inspired by real-world large-scale industry stacks for Rednote (Xiaohongshu)-style content platforms.

It is built around three layers:
- **Offline layer**: sample construction, feature generation, recall / preranking / ranking training, index building, artifact deployment
- **Online layer**: FastAPI serving, cold start handling, recall, multi-route fusion, preranking, ranking, real-time behavior write-back
- **Frontend layer**: Vue + Vite UI covering feed, detail page, behavior stream, and metrics dashboard

The project goal is not to collect isolated model scripts, but to connect the full loop of **offline training → online deployment → UI verification → metrics replay**.

**Demo videos:**  
- Bilibili: https://www.bilibili.com/video/BV1dDLm6qE9J/?share_source=copy_web&vd_source=e4fa524ab04ebed66f21b030503dfaaa  
- Xiaohongshu: https://www.xiaohongshu.com/explore/6a0d9df4000000003601c261?xsec_token=ABI5i6pPu-UkLbTQU14r4d7DGhMIrTiy-cbBKXE5coV1s=&xsec_source=pc_user

**Reference machine:** GPU: 5070 Ti 32GB VRAM, CPU: 9700X (8 cores), RAM: 32GB, SSD: ~500GB.

## 0. Overview and Results

### 0.1 Overview

Qilin is organized around the common serving path of **data preprocessing → feature engineering → recall → preranking → ranking → online serving**.

- Data scale
  - Notes: `1,983,938 (~2M)`
  - Users: `15,482`
- Search dataset
  - Training set: `44,024`
  - Test set: `6,192`
  - Features: rich query metadata, user interaction logs, clicked-label ground truth
- Recommendation dataset
  - Training set: `83,437`
  - Test set: `11,115`
  - Features: detailed user behavior sequences, candidate note pools, contextual features, clicked-label ground truth

### 0.2 Results

- Online service (fresh-inference latency)

  | Scene | P50 | P95 | P99 |
  | --- | ---: | ---: | ---: |
  | Search | 223.60ms | 496.60ms | 654.80ms |
  | Recommendation (recall) | 228.90ms | 369.40ms | 425.10ms |

- Search offline metrics
  - Full-catalog pipeline (all 6,192 test requests): **HitRate@10 = 55.23%**, **Recall@100 = 63.71%**, MRR@100 = 0.29.
  - Ranking over 128,191 test impressions: **NDCG@10 = 0.55**, AUC = 0.58, GAUC = 0.57.
- Recommendation offline metrics
  - DSSM recall (11,893 validation requests): **HitRate@50 = 6.00%**, **Recall@100 = 4.34%**.
  - Ranking over 182,567 test impressions: **NDCG@10 = 0.62**, AUC = 0.61, GAUC = 0.62.

Full-catalog evaluation covers 1,983,938 notes. Full precision and evaluation scopes are available in the [metrics snapshot](src/backend/online/full_test_metrics.json) and [DSSM validation results](src/frontend/src/data/recallValidation.json).

### 0.3 Pipeline

```text
Data preprocessing / feature engineering
  → recall training and index building
    → DSSM dual-tower
    → Swing
    → UserCF
    → Faiss IVFPQ
    → multi-route fusion
  → preranking training
    → LambdaMART (LightGBM + XGBoost)
  → ranking training
    → DIEN
  → deployment artifacts
  → FastAPI online serving
  → Vue frontend verification and metrics dashboard
```

## 1. Visual Showcase

### 1.1 Recommendation System Demo

<p align="center">
  <img src="./gif/推荐系统展示.gif" alt="Recommendation System Demo" width="100%">
</p>

### 1.2 Search System Demo

<p align="center">
  <img src="./gif/搜索系统展示.gif" alt="Search System Demo" width="100%">
</p>

### 1.3 Offline Metrics Dashboard

<p align="center">
  <img src="./gif/离线指标展示.gif" alt="Offline Metrics Dashboard" width="100%">
</p>

### 1.4 Example Comparison Demo

<p align="center">
  <img src="./gif/样例对比展示.gif" alt="Example Comparison Demo" width="100%">
</p>

## 2. Project Structure

### 2.1 Root Layout

```text
Qilin/
├─ datasets/
├─ features/
├─ embeddings/
├─ gif/
├─ image/
├─ outputs/
├─ src/
├─ start.sh
├─ stop.sh
└─ README.md
```

### 2.2 Backend / Training Layout

```text
src/
├─ backend/
├─ preprocess/
├─ recall/
├─ training/
└─ frontend/
```

## 3. System Architecture

### 3.1 Offline Training

```bash
uv run python src/backend/offline/pipeline.py --scene search
uv run python src/backend/offline/pipeline.py --scene rec
```

Main stages:
- `data`
- `feature`
- `training`
- `deploy`
- `feature-upload`

### 3.2 Online Serving

```bash
uv run python src/backend/online/api/main.py
```

The unified online orchestration lives in `src/backend/online/pipeline.py`, covering context parsing, cold start, recall, preranking, ranking, reranking, and final feed/detail response.

### 3.3 Frontend Interaction

The frontend calls backend APIs via `src/frontend/src/services/api.ts` and covers the feed, detail page, behavior stream, and metrics dashboard.

## 4. Models

### 4.1 Recall

- DSSM ANN
- Swing
- UserCF
- Search-specific lexical / semantic request-level fusion

### 4.2 Preranking

GBDT-based preranking compresses the recall pool and uses fast query / user / note / linkage features.

### 4.3 Ranking

DIEN performs finer user-interest modeling on top of preranking candidates.

## 5. Redis

### 5.1 Redis Cache

- user profiles
- recent requests
- history notes
- behavior streams
- exposed notes
- runtime request ids
- deduplication keys

### 5.2 Behavior Write Path

Frontend behavior calls flow through `api.ts` → `main.py` → `pipeline.py` → `realtime_cache.py` and are finally persisted into Redis.

## 6. Deployment Artifacts and Dashboard Snapshots

Online serving loads:

```text
outputs/deploy/{scene}/{tag}/models
outputs/deploy/{scene}/{tag}/index
```

Metrics dashboard snapshots are read from:

```text
outputs/serving_cache/{scene}/{tag}/
```

## 7. GitHub Star History

[![Star History Chart](https://api.star-history.com/svg?repos=WBoxian/Rednote-Qilin-Search-Rec-System&type=Date)](https://star-history.com/#WBoxian/Rednote-Qilin-Search-Rec-System&Date)


