# Qilin

Qilin 是一个搜索/推荐一体化系统，采用「离线训练 + 在线服务 + 前端展示」三层结构。

## 数据集描述与作用

当前项目使用的数据主要位于 `datasets/` 目录，按场景和用途拆分为训练集、测试集与画像/内容数据。

### 数据规模（当前版本）

- Notes：1,983,938
- Users：15,482
- Search：train 44,024 / test 6,192
- Rec：train 83,437 / test 11,115

### 在流程中的使用方式

- 离线阶段：`datasets/*` -> `features/*` -> 训练产物（模型与索引），再由模型上传阶段统一发布到 `outputs/deploy/{scene}/{tag}/{models,index}`。
- 在线阶段：只加载 `outputs/deploy/{scene}/{tag}` 下的 `models/` 与 `index/`，实时特征与行为序列通过 Redis 读取。

## 启动方式

### 离线 Pipeline 两个核心参数

- `--only`：控制主阶段（逗号分隔）
	- 可选：`data,feature,training,deploy,feature-upload`
- `--train`：控制训练子阶段（仅在 `training` 阶段生效）
	- 可选：`recall,multiroute,preranking,ranking`

### 参数生效规则（重要）

- 只写 `--train`：默认主阶段 = `training,deploy,feature-upload`（自动跳过 `data,feature`）。
- 只写 `--only`：只按主阶段执行。
- 同时写 `--only` + `--train`：只有 `--only` 包含 `training` 时，`--train` 才会生效。
- easy/hard 负样本模式：
	- 首次训练（不存在 `outputs/data/dssm_hard_neg_{scene}.parquet`）默认跑 `easy,hard`；
	- 后续增量训练（已存在该文件）默认只跑 `hard`。

### 最常用命令

#### 1) 全流程（默认）

```bash
uv run python src/backend/offline/pipeline.py --scene rec
uv run python src/backend/offline/pipeline.py --scene search
```

#### 2) 只跑训练（推荐重训方式）

```bash
uv run python src/backend/offline/pipeline.py --scene rec --train recall,multiroute,preranking,ranking
uv run python src/backend/offline/pipeline.py --scene search --train recall,multiroute,preranking,ranking
```

#### 3) 只跑某个训练子阶段

```bash
# 仅精排
uv run python src/backend/offline/pipeline.py --scene rec --train ranking

# 仅 multiroute+粗排+精排
uv run python src/backend/offline/pipeline.py --scene search --train multiroute,preranking,ranking
```

#### 4) 只部署模型

```bash
uv run python src/backend/offline/pipeline.py --scene rec --only deploy
uv run python src/backend/offline/pipeline.py --scene search --only deploy
```

#### 5) 在线服务

```bash
uv run python src/backend/online/api/main.py
```

#### 6) 前端

```bash
cd src/frontend
npm install
npm run dev
```

## 目录说明

### 根目录

- `datasets/`：离线原始数据（search/rec 训练与测试集、user_feat、notes 等）。
- `samples`：离线原始数据 join 成的 samples。
- `features/`：离线生成的样本特征 parquet。

- `image/`：笔记图片资源，在线接口 `/image/*` 直接读取。
- `outputs/`：统一产物目录。
	- `outputs/models/`：训练好的模型权重。
	- `outputs/data/`：候选集、训练样本与评估相关中间数据。
	- `outputs/logs/`：训练日志（如 TensorBoard）。
	- `outputs/results/`：评估结果与对比结果。
	- `outputs/index/`：离线索引中间目录（训练阶段会发布到 deploy）。
	- `outputs/deploy/{scene}/{tag}/models/`：线上模型目录。
	- `outputs/deploy/{scene}/{tag}/index/`：线上召回索引目录。

### `src/backend`

- `src/backend/offline/`：离线总流程。
	- `pipeline.py`：离线主入口，串联数据、特征、训练、上传。
	- `config.py`：离线统一配置（scene、modes、topk 等）。
	- `data/build_samples.py`：构建训练/测试样本。
	- `feature/build_features.py`：构建模型特征与向量。
	- `training/run_training.py`：训练阶段编排入口。
	- `training/train_recall.py`：召回训练（DSSM/Swing/UserCF/Faiss）。
	- `training/build_recall_candidates.py`：多路召回候选构建。
	- `training/train_rankers.py`：粗排（GBDT）与精排（DIEN）训练。
	- `storage/local_deploy.py`：离线模型部署工具（直达 deploy 目录结构）。
	- `storage/redis_ingest.py`：将用户特征写入 Redis。

- `src/backend/online/`：在线推理与 API。
	- `pipeline.py`：在线主入口，维护运行时状态并执行推荐全链路。
	- `config.py`：在线配置（host/port/tag/topn 等）。
	- `api/main.py`：FastAPI 接口入口。
	- `cold_start/`：冷启动识别与热榜兜底。
		- `detector.py`：冷启动判定。
		- `popular.py`：热榜候选构建。
		- `service.py`：冷启动候选服务。
	- `recall/`：召回阶段。
		- `dssm.py`：DSSM 用户塔加载与 ANN 召回。
		- `swing.py` / `usercf.py`：规则召回实现。
		- `service.py`：多路召回融合与回退策略。
	- `preranking/`：粗排阶段。
		- `gbdt.py`：GBDT 模型加载与打分。
		- `service.py`：粗排编排。
	- `ranking/`：精排阶段。
		- `dien.py`：DIEN 模型加载与打分。
		- `service.py`：精排编排与最终融合。

### `src/recall`

召回算法与索引构建脚本：

- `dssm_trainer.py`：DSSM 双塔训练，导出 item/query 向量。
- `build_faiss_ivfpq.py`：基于 DSSM item 向量构建 Faiss IVF-PQ 索引。
- `cf_shared_index.py`：构建并缓存 Swing/UserCF 共享 user-item 索引。
- `build_swing_index.py`：构建 Swing item-item 索引。
- `build_usercf_index.py`：构建 UserCF user-user 索引。
- `build_multiroute_recall.py`：ANN + Swing + UserCF 多路融合，产出候选 parquet。

### `src/frontend`

Vite + Vue 前端工程：

- `src/main.ts`：前端入口。
- `src/router/index.ts`：路由配置。
- `src/services/api.ts`：后端 API 封装与本地状态管理（scene/user/mode）。
- `src/views/LoginView.vue`：登录与场景/模式选择。
- `src/views/HomeView.vue`：主 feed 页面（搜索/推荐）。
- `src/views/DetailView.vue`：笔记详情与分数展示。
- `src/views/UserView.vue`：用户画像页面。
- `src/views/MetricsView.vue`：指标与验证结果页面。

## 在线接口

- `GET /api/health`：健康检查与模型就绪状态。
- `GET /api/login`：登录校验并返回用户信息。
- `GET /api/feed`：主推荐/搜索结果。
- `GET /api/note`：笔记详情与模型分数。
- `GET /api/user`：用户画像。
- `GET /api/metrics`：离线 test 指标。
- `GET /api/validation`：validation 对比。
- `POST /api/behavior/click`：记录用户点击行为并实时更新历史序列。
- `GET /image/{path}`：图片访问。