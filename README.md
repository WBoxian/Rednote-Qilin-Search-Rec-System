# 小红书麒麟(Qilin)搜推系统个人项目

Qilin 是一个面向“小红书搜索推荐一体化”场景的个人项目，采用接近业界主流搜推系统的三层架构：

- 离线层：样本构建、特征生成、召回/粗排/精排训练、索引构建、部署产物发布
- 在线层：FastAPI 服务、冷启动、召回、多路融合、粗排、精排、实时行为写回
- 前端层：Vue + Vite 可视化界面，覆盖首页、详情页、最近行为、指标看板

项目目标不是做一个单纯的模型脚本集合，而是把“离线训练 -> 在线部署 -> 页面验证 -> 指标回放”串成一个完整闭环。

项目视频展示链接：

## 0. 项目简介与成果

### 0.1 项目简介

Qilin 面向“小红书搜索 + 推荐一体化”场景，围绕搜推系统常见的 **数据预处理 → 特征工程 → 召回 → 粗排 → 精排 → 在线服务** 主链路搭建：

- 数据规模
  - 笔记规模：`1,983,938 (~2M)`
  - 用户规模：`15,482`
- 搜索数据集（Search Dataset）
  - 训练集：`44,024` 条样本
  - 测试集：`6,192` 条样本
  - 特征：
    - 丰富的 Query 元数据
    - 用户交互行为日志
    - 点击标签真值
- 推荐数据集（Recommendation Dataset）
  - 训练集：`83,437` 条样本
  - 测试集：`11,115` 条样本
  - 特征：
    - 细粒度用户历史行为序列
    - 候选笔记池
    - 上下文特征
    - 点击标签真值

### 0.2 项目成果

- 在线服务
  - 搜推系统线上服务延迟稳定在 `~150ms`
- Search 离线指标
  - 召回：
    - `HitRate@500 = 0.88`
    - `Recall@500 = 0.65`
    - `MRR@100 = 0.11`
    - `MedianFirstHitRank = 47`
  - 排序：
    - `NDCG@10 = 0.69`
    - `AUC = 0.77`
    - `GAUC = 0.85`
- Recommendation 离线指标
  - 召回：
    - `HitRate@500 = 0.99`
    - `Recall@500 = 0.99`
    - `MRR@100 = 0.007`
    - `MedianFirstHitRank = 101`
  - 排序：
    - `NDCG@10 = 0.87`
    - `AUC = 0.84`
    - `GAUC = 0.90`

### 0.3 核心流程

```text
离线样本构建 / 特征生成
  → 召回训练与索引构建
    → DSSM 双塔
    → Swing
    → UserCF
    → Faiss IVFPQ
    → 多路召回融合
  → 粗排训练
    → LambdaMART (LightGBM + XGBoost)
  → 精排训练
    → DIEN
  → 部署产物发布
  → FastAPI 在线服务
  → Vue 前端验证与指标看板
```

## 1. 项目结构

### 1.1 根目录

```text
Qilin/
├─ datasets/
│  ├─ note_features/
│  ├─ recommendation_test/
│  ├─ recommendation_train/
│  ├─ search_test/
│  ├─ search_train/
│  └─ user_feat/
├─ features/
│  ├─ rec_test_features.parquet
│  ├─ rec_train_features.parquet
│  ├─ search_test_features.parquet
│  └─ search_train_features.parquet
├─ embeddings/
│  ├─ note_image_emb.parquet
│  ├─ note_text_emb.parquet
│  ├─ rec_query_emb.parquet
│  └─ search_query_emb.parquet
├─ image/
├─ outputs/
│  ├─ data/
│  ├─ deploy/
│  │  ├─ rec/{easy,hard}/
│  │  └─ search/{easy,hard}/
│  ├─ index/
│  ├─ models/
│  ├─ results/
│  └─ serving_cache/
├─ src/
│  ├─ backend/
│  │  ├─ offline/
│  │  └─ online/
│  ├─ frontend/
│  │  ├─ src/
│  │  └─ dist/
│  ├─ preprocess/
│  ├─ recall/
│  └─ training/
├─ start.sh
├─ stop.sh
└─ README.md
```

说明：

- `datasets/`
  - 原始训练/测试数据，包含 search / recommendation 两个场景的请求、用户画像、笔记内容等
- `features/`
  - 离线阶段生成的样本特征 parquet
- `embeddings/`
  - 文本、图片、序列等 embedding 存储目录
- `image/`
  - 笔记图片资源，在线接口 `/image/*` 直接读取
- `outputs/`
  - 全部中间产物与部署产物，包括模型、索引、离线快照与最终 deploy 目录

### 1.2 后端

```text
src/
├─ backend/
│  ├─ offline/
│  │  ├─ pipeline.py
│  │  ├─ storage/
│  │  └─ training/
│  └─ online/
│     ├─ api/
│     ├─ cold_start/
│     ├─ preranking/
│     ├─ ranking/
│     ├─ recall/
│     ├─ pipeline.py
│     └─ realtime_cache.py
├─ preprocess/
│  ├─ build_features.py
│  ├─ build_note_text_emb.py
│  ├─ build_query_text_emb.py
│  └─ build_samples.py
├─ recall/
│  ├─ build_multiroute_recall.py
│  ├─ build_seq_transition_index.py
│  ├─ dssm_trainer.py
│  └─ mine_hard_negatives.py
├─ training/
│  ├─ dien_ranker.py
│  ├─ ranker.py
│  └─ utils.py
└─ frontend/
   ├─ src/
   │  ├─ components/
   │  ├─ services/
   │  └─ views/
   └─ dist/
```

说明：

- `src/backend/offline/`
  - 离线总编排，负责样本、特征、训练、部署、特征写入 Redis
- `src/backend/online/`
  - 在线服务，负责 FastAPI、运行时状态管理、首页 feed、详情页、实时行为、指标看板
- `src/preprocess/`
  - 离线样本、特征、embedding 预处理脚本
- `src/recall/`
  - 召回训练与索引构建脚本，包括 DSSM、Faiss、Swing、UserCF、多路召回候选构建
- `src/training/`
  - 排序模型训练与离线打分脚本

### 1.3 前端

- `src/frontend/`
  - Vite + Vue 3 工程
  - 主要页面：
    - `LoginView.vue`
    - `HomeView.vue`
    - `DetailView.vue`
    - `UserView.vue`
    - `MetricsView.vue`

## 2. 系统架构

### 2.1 离线训练链路

离线主入口：

```bash
uv run python src/backend/offline/pipeline.py --scene search
uv run python src/backend/offline/pipeline.py --scene rec
```

离线流程分为五个主阶段：

- `data`
  - 构建 train / test 请求样本
- `feature`
  - 生成模型训练所需特征与 embedding
- `training`
  - 训练召回、multiroute、粗排、精排
- `deploy`
  - 将离线产物发布到 `outputs/deploy/{scene}/{tag}`
- `feature-upload`
  - 将用户画像与在线需要的实时特征写入 Redis

训练阶段支持子阶段控制：

- `recall`
- `multiroute`
- `preranking`
- `ranking`

常用示例：

```bash
# 全流程
uv run python src/backend/offline/pipeline.py --scene search

# 只重训排序相关
uv run python src/backend/offline/pipeline.py --scene rec --train preranking,ranking

# 只重训召回并部署
uv run python src/backend/offline/pipeline.py --scene search --train recall,multiroute

# 只部署已有产物
uv run python src/backend/offline/pipeline.py --scene rec --only deploy
```

### 2.2 在线服务链路

在线入口：

```bash
uv run python src/backend/online/api/main.py
```

实际推荐链路在 `src/backend/online/pipeline.py` 中统一编排，整体流程为：

1. 解析用户与请求上下文
2. 冷启动判断
3. 召回
4. 粗排
5. 精排
6. 重排（去重 / 多样性处理）
7. 首页结果返回或详情页补查

对应模块：

- `cold_start/`
  - 冷启动识别与热榜回填
- `recall/`
  - DSSM ANN、Swing、UserCF、多路召回融合
- `preranking/`
  - GBDT 粗排
- `ranking/`
  - DIEN 精排
- `realtime_cache.py`
  - 实时行为、曝光、用户请求与画像增量缓存

### 2.3 前端交互链路

前端通过 `src/frontend/src/services/api.ts` 调用后端接口，核心页面行为：

- 首页
  - 搜索/推荐 feed
  - “换一批”
  - 搜索建议
- 详情页
  - 帖子详情、交互行为
- 最近行为页
  - 用户画像与实时行为流
- 指标看板页
  - 离线快照 / 在线回放指标
  - 样例对比

## 3. 模型与召回/排序分层

### 3.1 召回层

召回层多路融合：

- DSSM ANN
  - 双塔向量召回
  - 在线使用部署目录中的 item / request 向量与 Faiss 索引
- Swing
  - item-item 协同
- UserCF
  - user-user 协同
- Search 场景还会融合 request 级 lexical / semantic 结果

召回相关代码：

- `src/recall/dssm_trainer.py`
- `src/recall/build_multiroute_recall.py`
- `src/backend/online/recall/dssm.py`
- `src/backend/online/recall/service.py`

### 3.2 粗排层

粗排采用 GBDT 路线，负责：

- 压缩召回候选规模
- 利用 query / user / note / linkage 特征快速排序

代码：

- `src/backend/offline/training/train_rankers.py`
- `src/backend/online/preranking/gbdt.py`
- `src/backend/online/preranking/service.py`

### 3.3 精排层

精排采用 DIEN 路线，负责：

- 对粗排候选做更细的用户兴趣建模
- 结合序列行为进一步重排

代码：

- `src/backend/online/ranking/dien.py`
- `src/backend/online/ranking/service.py`

## 4. Redis 的作用

Redis 不是可有可无的装饰组件，而是在线实时能力的一部分。

### 4.0 Redis 是什么

Redis 是一个基于内存的键值数据库，典型特点是：

- 读写延迟低，适合在线请求链路
- 支持字符串、哈希、列表等常见数据结构
- 适合存放“实时变化、频繁访问、可接受内存态管理”的数据

在这个项目里，Redis 不是用来存离线模型文件，而是用来承接：

- 用户实时行为
- 用户近期兴趣与历史
- 最近曝光结果
- 搜推联动所需的在线上下文

也就是说，Redis 解决的是“在线实时状态管理”，不是“离线训练产物持久化”。

### 4.1 Redis 存什么

主要承载：

- 用户画像特征
- 最近请求序列
- 最近点击 / 浏览 / 互动行为
- 最近曝光内容
- 跨 Search / Rec 场景联动所需的实时上下文

当前在线 Redis key 大致包括：

- `qilin:user:{user_idx}:profile`
  - 用户画像缓存
- `qilin:user:{user_idx}:{scene}:requests`
  - 用户最近请求序列
- `qilin:user:{user_idx}:{scene}:history_notes`
  - 用户最近历史 note 序列
- `qilin:user:{user_idx}:{scene}:behaviors`
  - 用户最近行为事件流
- `qilin:user:{user_idx}:{scene}:exposed_notes`
  - 最近曝光过的 note，用于去重和“换一批”
- `qilin:{scene}:runtime_request_id`
  - 在线 runtime request id 自增计数
- `qilin:dedup:*`
  - 行为去重窗口 key，避免极短时间重复 click/view/engage 被重复计入

### 4.2 Redis 为什么重要

如果没有 Redis，系统只能依赖离线快照：

- 最近行为页会退回离线数据
- 首页无法实时感知刚刚发生的点击/曝光
- 搜推联动上下文变弱
- 首页“换一批”与去重、多样性能力会变差
- 指标页和在线体验会更容易与离线训练分离

也就是说：

- 离线数据保证基础能力
- Redis 负责把在线实时行为接入当前服务链路

### 4.3 前端交互后的行为写入链路

前端行为入口在：

- `src/frontend/src/services/api.ts`

主要包括：

- `api.click(...)`
- `api.view(...)`
- `api.engage(...)`
- `api.deleteBehavior(...)`
- `api.deleteBehaviorsBatch(...)`

这些接口会进入：

- `src/backend/online/api/main.py`
  - `/api/behavior/click`
  - `/api/behavior/view`
  - `/api/behavior/engage`
  - `/api/behavior`
  - `/api/behavior/batch_delete`

然后由：

- `src/backend/online/pipeline.py`
  - `record_click`
  - `record_view`
  - `record_engage`
  - `delete_behavior`
  - `delete_behaviors`

最终落到：

- `src/backend/online/realtime_cache.py`

实际写入 Redis 的是行为 JSON 事件、历史 note 列表、曝光列表等实时状态。

这意味着：

- 首页点击、详情浏览、点赞收藏评论分享
- 最近行为页的单条删除和批量删除

都会同步修改 Redis 中的在线行为数据，而不是只改前端显示。

### 4.4 Redis 写入入口

离线写入入口：

```bash
uv run python src/backend/offline/pipeline.py --scene search --only feature-upload
uv run python src/backend/offline/pipeline.py --scene rec --only feature-upload
```

对应代码：

- `src/backend/offline/storage/redis_ingest.py`

这里的离线写入主要负责：

- 把用户画像等基础在线特征写入 Redis
- 让在线服务冷启动时不必完全依赖运行时现算

而前端交互触发的 click/view/engage/delete，则属于在线增量写入，不走这条离线脚本。

## 5. 部署产物

在线服务真正加载的是：

```text
outputs/deploy/{scene}/{tag}/models
outputs/deploy/{scene}/{tag}/index
```

不是 `outputs/models` 或 `outputs/index` 的训练中间目录直接上线上。

部署由：

- `src/backend/offline/storage/local_deploy.py`

负责完成，当前逻辑会在部署前清理旧目录，再拷贝新模型和索引，避免旧产物残留。

## 6. 指标看板

指标看板页面不是简单静态展示，它包含两类数据：

- 指标
  - `metrics`
  - 当前支持训练集 / 测试集、召回 / 粗排 / 精排
- 样例对比
  - `validation`
  - 展示真实 Top10 与 DSSM / GBDT / DIEN 的对比

为了降低在线重算开销，当前服务支持将结果写入：

```text
outputs/serving_cache/{scene}/{tag}/
```

这类快照会被在线接口优先读取，用来减少指标页首屏等待。

## 7. 启动方式

项目提供统一脚本：

```bash
./start.sh
./stop.sh
```

`start.sh` 会做这些事：

1. 检查 Redis
   - 如果 `redis://127.0.0.1:6379/0` 已经可用，则直接复用
   - 如果不可用且本机有 Docker，则尝试拉起/复用本地 Redis 容器
   - 如果没有 Redis 也没有 Docker，会给出 warning，但服务仍可继续启动
2. 构建前端
3. 启动后端服务
4. 根据环境变量执行首页/指标预热

需要注意：

- `start.sh` **不是每次都强制新建一个 Redis**
- 它的逻辑是“优先复用现有 Redis，不可用时再尝试启动 Docker Redis”

所以更准确地说：

- 如果本地 Redis 已经在运行，项目会直接用它
- 如果本地 Redis 没运行，但 Docker 可用，项目会尝试自动拉起一个本地 Redis 容器
- Redis 的作用主要是在线服务和行为记录加速，不是离线模型文件存储

默认线上 tag 由环境变量控制：

- `QILIN_TAG_SEARCH`
- `QILIN_TAG_REC`

## 8. 核心目录速查

### 离线

- `src/backend/offline/pipeline.py`
- `src/backend/offline/training/run_training.py`
- `src/backend/offline/storage/local_deploy.py`
- `src/backend/offline/storage/redis_ingest.py`

### 在线

- `src/backend/online/api/main.py`
- `src/backend/online/pipeline.py`
- `src/backend/online/recall/service.py`
- `src/backend/online/preranking/service.py`
- `src/backend/online/ranking/service.py`
- `src/backend/online/realtime_cache.py`

### 前端

- `src/frontend/src/services/api.ts`
- `src/frontend/src/views/HomeView.vue`
- `src/frontend/src/views/DetailView.vue`
- `src/frontend/src/views/UserView.vue`
- `src/frontend/src/views/MetricsView.vue`

## 9. 当前接口

- `GET /api/health`
- `GET /api/scenes`
- `GET /api/login`
- `GET /api/users`
- `GET /api/user`
- `GET /api/feed`
- `GET /api/note`
- `GET /api/suggest`
- `GET /api/metrics`
- `GET /api/validation`
- `POST /api/behavior/click`
- `POST /api/behavior/view`
- `POST /api/behavior/engage`
- `DELETE /api/behavior`
- `GET /image/{path}`

## 10. 说明

这个项目当前的重点不只是“训练出一个更高的离线分数”，而是让以下几个环节一致：

- 离线训练口径
- 在线召回/排序口径
- 页面展示口径
- 行为写回口径
- 指标看板口径

因此项目里既有训练代码，也有在线服务、缓存、部署和前端验证页面。
