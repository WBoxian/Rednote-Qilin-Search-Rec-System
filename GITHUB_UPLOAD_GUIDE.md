# GitHub 私仓上传指引

本项目已准备好上传到 GitHub 私仓。以下是上传步骤。

## 1. 在 GitHub 上创建私仓

1. 访问 [GitHub New Repository](https://github.com/new)
2. **Repository name**: `Qilin`（或你喜欢的名称）
3. **Description**: Qilin Recommendation System - Search & Recommendation Pipeline
4. **Visibility**: 选择 **Private** ✓
5. 点击 **Create repository**

## 2. 获取 GitHub 个人访问令牌（Personal Access Token）

如果还未设置 SSH，建议使用 PAT：

1. 访问 [GitHub Settings → Developer settings → Personal access tokens](https://github.com/settings/tokens)
2. 点击 **Generate new token**
3. 选择 **Generate new token (classic)**
4. **Scopes** 勾选：`repo` 和 `gist`
5. 生成并 **保存令牌**（仅显示一次）

## 3. 配置上传

### 方式 A：使用 HTTPS（推荐简单）

替换 `<YOUR_USERNAME>` 和 `<YOUR_TOKEN>` 后执行：

```bash
cd /home/boxian/projects/Qilin
git remote add origin https://<YOUR_TOKEN>@github.com/<YOUR_USERNAME>/Qilin.git
git branch -M main
git push -u origin main
```

### 方式 B：使用 SSH（推荐安全）

如已配置 SSH 密钥：

```bash
cd /home/boxian/projects/Qilin
git remote add origin git@github.com:<YOUR_USERNAME>/Qilin.git
git branch -M main
git push -u origin main
```

## 4. 验证上传

执行后访问：
```
https://github.com/<YOUR_USERNAME>/Qilin
```

应该能看到所有代码文件（约 ~800 KB）。

## 5. 更新本地设置（可选）

如要后续推送更新：

```bash
cd /home/boxian/projects/Qilin
git status  # 检查未提交改动
git add .
git commit -m "Your message"
git push
```

## 项目结构

上传的內容包括：

```
Qilin/
├── src/
│   ├── backend/        # 在线服务（FastAPI）
│   ├── training/       # 离线训练脚本
│   ├── recall/         # 召回（DSSM / UserCF / Swing）
│   ├── preprocess/     # 数据预处理
│   └── frontend/       # Vue3 前端界面（不含 node_modules）
├── pyproject.toml      # Python 依赖
├── requirements.txt
├── start.sh            # 启动脚本
├── stop.sh             # 停止脚本
└── README.md
```

**已排除**（仅保留代码）：
- `datasets/`、`outputs/`、`embeddings/`、`features/` 等数据目录
- `src/recall/index/` 模型二进制文件
- `src/backend/online/deploy/` 部署产物
- `src/frontend/node_modules/` 前端依赖

## 工作量

- **已跟踪文件**：135 个
- **总代码量**：~50 MB（包含前端源码）
- **Git 仓库大小**：~20 MB
- **提交历史**：完整保留

---

有问题？检查：
- SSH/HTTPS 凭证是否正确
- 仓库是否设为 Private
- GitHub 登陆状态是否有效

