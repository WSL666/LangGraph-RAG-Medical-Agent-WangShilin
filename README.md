# 基于 LangGraph 与 RAG 的医疗智能体构建与实现

> **Construction and Implementation of a Medical Agent Based on LangGraph and RAG**
>
> 作者 / Author: 王石林 (Wang Shilin)

本项目基于 **LangGraph** 编排医疗领域 **RAG（Retrieval-Augmented Generation）**全流程，结合 **Elasticsearch（BM25 + 稠密向量）** 混合检索、**Redis** 问答缓存、**Gradio** 聊天 UI，构建了一个具备**意图路由、问句改写、混合检索、检索回退、流式生成、缓存命中**等能力的医疗智能体。

---

## 目录

- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
  - [1. 准备环境](#1-准备环境)
  - [2. 启动 Elasticsearch（Docker）](#2-启动-elasticsearchdocker)
  - [3. 启动 Redis（Docker）](#3-启动-redisdocker)
  - [4. 配置 `.env`](#4-配置-env)
  - [5. 导入医疗数据到 ES](#5-导入医疗数据到-es)
  - [6. 启动 Web 服务](#6-启动-web-服务)
- [配置说明](#配置说明)
- [模块说明](#模块说明)
- [可选：导出架构图](#可选导出架构图)
- [License](#license)

---

## 核心特性

- **LangGraph 状态图编排**：将"缓存查询 → 意图分流 → 问句改写 → 混合检索（带回退）→ 生成 → 写缓存"拆分为可观察、可分支的有向图节点；
- **意图路由（medical / chitchat / refuse）**：内置敏感词正则 + LLM 分类，敏感问题直接拒答，闲聊不走 RAG；
- **混合检索（BM25 + Embedding）**：支持 `hybrid`（RRF 融合）、`weighted`（min-max + 加权）、`bm25`、`embedding` 四种模式，检索为空时自动**模式回退**重试；
- **稠密向量字段（dense_vector）**：基于 `BAAI/bge-m3`（1024 维）做余弦相似度召回；
- **Redis 问答缓存 + 流式回放**：命中缓存时按字符切块"伪流式"输出，体验与首次生成一致；
- **Gradio 聊天界面**：支持流式 token、对话历史、示例问题、清空会话；
- **环境变量化配置**：所有外部依赖（LLM、ES、Redis、检索参数）均通过 `.env` 注入，便于切换模型 / 集群。

---

## 系统架构

```
                      ┌──────────────┐
                      │   用户提问    │
                      └──────┬───────┘
                             ▼
                    ┌────────────────┐  hit   ┌──────────────┐
                    │  cache_lookup   │──────►│ stream_cache  │──┐
                    └───────┬────────┘        └──────────────┘  │
                            │ miss                              │
                            ▼                                   │
                    ┌────────────────┐                          │
                    │  intent_route   │ refuse ─────► END        │
                    └───┬────────┬───┘                          │
                medical │        │ chitchat                     │
                        ▼        ▼                              │
              ┌──────────────┐ ┌──────────┐                     │
              │ query_rewrite│ │ chitchat │                     │
              └──────┬───────┘ └─────┬────┘                     │
                     ▼               │                          │
              ┌──────────────┐       │                          │
              │   retrieve   │◄──┐   │                          │
              └──────┬───────┘   │   │                          │
                     │ empty     │   │                          │
                     ├──retry────┘   │                          │
                     │ ok            │                          │
                     ▼               │                          │
              ┌──────────────┐       │                          │
              │   generate   │       │                          │
              └──────┬───────┘       │                          │
                     ▼               ▼                          ▼
                    ┌────────────────────────────┐
                    │         save_cache          │──► END
                    └────────────────────────────┘
```

完整 LangGraph 拓扑见 [`docs/architecture.png`](docs/architecture.png)。

---

## 项目结构

```
LangGraph-RAG-Medical-Agent-WangShilin/
├── main.py                       # 入口：启动 Gradio Web
├── gradio_show.py                # Gradio UI 定义
├── gradio_background.py          # MedicalChatSystem：包装 LangGraph 状态图
├── ES_search.py                  # 运行时检索模块（BM25 / Embedding / Hybrid / Weighted）
├── redis_search.py               # 运行时 Redis 缓存读写
├── export_architecture.py        # 导出 LangGraph 拓扑为 PNG / Mermaid
├── requirements.txt              # Python 依赖
├── pyproject.toml                # 项目元信息
├── .env.example                  # 环境变量模板
├── .gitignore
│
├── graph/                        # LangGraph 编排层
│   ├── __init__.py
│   ├── build.py                  # 装配节点 + 边 + 编译
│   ├── nodes.py                  # 节点：cache_lookup / intent_route / rewrite / retrieve / generate ...
│   ├── router.py                 # 条件边判定：意图分流 / 检索回退
│   └── state.py                  # GraphState：节点共享状态
│
├── Elasticsearch_database/       # 数据建库脚本 + 主知识库
│   ├── ES_search.py              # 建索引 + 字段映射 + 向量化 + 批量入库
│   └── medical.json              # 主医疗知识库（≈58 MB，药典）
│
├── redis_database/
│   └── redis_search.py           # Redis 连通性测试 + 演示数据预热
│
└── docs/
    ├── architecture.png          # 架构图
    └── architecture.mmd          # 架构图 Mermaid 源码
```

> 🔒 **隐私 / 大文件**：`.env`、模型权重以及其他扩展数据集（`medical111.json`、药典等）不会进入仓库；仓库仅保留主知识库 `Elasticsearch_database/medical.json`，详见 `.gitignore`。

---

## 快速开始

### 1. 准备环境（使用 [uv](https://docs.astral.sh/uv/)）

本项目使用 [`uv`](https://github.com/astral-sh/uv) 管理 Python 环境与依赖，`.python-version` 已锁定 Python `3.10`。

```bash
# 1) 安装 uv（任选其一）
# Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或通过 pip / pipx：pip install uv

# 2) 进入项目根目录
cd LangGraph-RAG-Medical-Agent-WangShilin

# 3) 创建虚拟环境（自动按 .python-version 选 3.10）
uv venv

# 4) 安装依赖
uv pip install -r requirements.txt
# —— 或者直接基于 pyproject.toml 同步 ——
# uv sync
```

激活虚拟环境（可选；`uv run` 会自动使用 `.venv`，无需手动激活）：

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

### 2. 启动 Elasticsearch（Docker）

```bash
docker network create elastic

docker run -d --name elasticsearch --network elastic \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=true" \
  -e "xpack.security.http.ssl.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  -v es-data:/usr/share/elasticsearch/data \
  docker.elastic.co/elasticsearch/elasticsearch:9.0.8

# 查看默认密码（首次启动后日志中会输出）
docker logs elasticsearch | grep "Password for the elastic user"
# 若无输出，进入容器重置：
# docker exec -it elasticsearch bin/elasticsearch-reset-password -u elastic
```

**安装 IK 中文分词器**（建议）：

```bash
# 1. 下载与 ES 版本匹配的 IK 压缩包：https://release.infinilabs.com/analysis-ik/stable/
# 2. 复制到容器并安装
docker cp elasticsearch-analysis-ik-9.0.8.zip elasticsearch:/tmp/
docker exec -it elasticsearch \
  ./bin/elasticsearch-plugin install file:///tmp/elasticsearch-analysis-ik-9.0.8.zip
docker restart elasticsearch
```

### 3. 启动 Redis（Docker）

```bash
docker run -d --name my-redis -p 6379:6379 redis \
  --requirepass your_redis_password
```

### 4. 配置 `.env`

```bash
cp .env.example .env
# 然后编辑 .env，至少填写：
#   OPENAI_API_KEY=...        # OpenAI 兼容 LLM 的 key
#   ES_PASSWORD=...           # ES elastic 用户密码
#   REDIS_PASSWORD=...        # Redis 密码（如未设置可留空）
```

### 5. 导入医疗数据到 ES

仓库已自带主知识库 `Elasticsearch_database/medical.json`（≈58 MB），字段已与脚本 mapping 对齐：`name / desc / symptom / cause / cure_way / common_drug / check ...`

```bash
cd Elasticsearch_database
uv run python ES_search.py
```

脚本会自动：
1. 创建索引并定义 `dense_vector` 向量字段；
2. 调用 `BAAI/bge-m3` 把医学文本转为 1024 维向量；
3. 批量写入并刷新索引。

> 💡 `medical.json` 单文件 58 MB，git push 时 GitHub 会给一条 "large file" 警告，但仍可推送（上限是 100 MB / 单文件）。如需更精简的仓库或托管更完整的数据集，可改用 [Git LFS](https://git-lfs.com/) 或 GitHub Releases。

### 6. 启动 Web 服务

```bash
# 回到项目根目录
uv run python main.py
# 或
uv run python gradio_show.py
```

浏览器访问：`http://127.0.0.1:7939`

---

## 配置说明

所有可调参数集中在 `.env`：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OPENAI_BASE_URL` | `https://api.siliconflow.cn/v1` | OpenAI 兼容 LLM 服务地址 |
| `OPENAI_API_KEY` | — | LLM API Key |
| `OPENAI_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | 生成 & 路由模型 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 向量化模型 |
| `EMBEDDING_DIM` | `1024` | 向量维度（需与索引 mapping 一致） |
| `ES_HOST` / `ES_PORT` | `127.0.0.1` / `9200` | ES 连接信息 |
| `ES_INDEX_NAME` | `medical_articles` | ES 索引名 |
| `RAG_RETRIEVAL_MODE` | `hybrid` | `hybrid` / `bm25` / `embedding` / `weighted` |
| `RAG_TOP_K` | `3` | 返回 top-k 文档 |
| `RAG_CANDIDATE_SIZE` | `9` | 各路召回候选数 |
| `RAG_HYBRID_ALPHA` | `0.5` | weighted 模式 BM25 权重 |
| `RAG_RRF_K` | `60` | RRF 平滑常数 |
| `RAG_RRF_W_BM25` / `RAG_RRF_W_EMB` | `1.0` / `1.0` | RRF 两路权重 |
| `REDIS_*` | — | Redis 连接信息 |
| `GRADIO_SERVER_NAME` / `GRADIO_SERVER_PORT` | `127.0.0.1` / `7939` | Gradio 监听 |

---

## 模块说明

### `graph/` —— LangGraph 编排核心

| 节点 | 作用 |
|---|---|
| `cache_lookup` | Redis 查 `qa:<query>`，命中则跳过整条 RAG 链 |
| `stream_cache` | 命中缓存时按字符切块"伪流式"回放 |
| `intent_route` | 敏感词正则 + LLM 二级分类 → `medical` / `chitchat` / `refuse` |
| `chitchat` | 闲聊分支，直接走 LLM 不检索 |
| `query_rewrite` | 把口语化提问改写成关键词式检索 query |
| `retrieve` | 调用 `main_ES(q, mode)`，按当前 `retrieval_mode` 检索 |
| `retrieve_retry` | 检索为空时按 `hybrid → bm25 → embedding` 切换重试（最多 2 次） |
| `generate` | 基于检索资料 + 历史会话流式生成最终答案 |
| `save_cache` | 非 `refuse` 答案写回 Redis，TTL 360s |

### `ES_search.py` —— 混合检索

- `search_bm25_documents`：标准 BM25，`multi_match` over 11 个医学字段；
- `search_embedding_documents`：基于 `cosineSimilarity` 的 `script_score` 稠密召回；
- `_merge_hits_by_rrf` / `_merge_hits_by_weighted_sum`：两种融合策略，前者更鲁棒，后者便于做消融。

### `Elasticsearch_database/ES_search.py` —— 建库

- 创建索引并显式声明字段类型与 `dense_vector` 维度；
- 批量调用 embedding API（带重试 / 退避 / 限流处理）；
- 失败的单条会自动 fallback 到截断重试，避免拖垮整批。

---

## 可选：导出架构图

```bash
uv run python export_architecture.py
```

会在 `docs/` 下生成 `architecture.png` 与 `architecture.mmd`。如机器无法访问 `mermaid.ink`，把 `architecture.mmd` 贴到 [mermaid.live](https://mermaid.live) 在线渲染即可。

---

## License

本项目用于学术 / 学习用途，仅作毕业设计与论文实验之用。
**医疗信息不构成诊断或处方建议，请以医生意见为准。**

The system is for academic and educational purposes only.
Medical information generated by this system does **NOT** constitute diagnosis or prescription advice — please consult a licensed physician.
