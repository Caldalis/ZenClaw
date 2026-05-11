# TaskWeave

[English](README.md) | 中文

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

TaskWeave 是一个基于 Python 3.12 的本地 AI Agent 框架。代码包内部名称为 `taskweave`，核心能力包括事件流式 Agent 循环、可插拔模型 Provider、持久化记忆、工具调用、按需加载的 `SKILL.md` 技能，以及可选的 Master/Subagent 多智能体 DAG 调度架构。

## 功能特性

- 单 Agent ReAct 循环：支持工具调用，并以 `Event` 对象形式向通道输出。
- Master/Subagent 模式：将复杂任务拆成 DAG，并按依赖关系调度多个子 Agent。
- Provider 故障转移：支持 OpenAI 兼容接口和 Anthropic Claude。
- SQLite 会话记忆：默认支持 FTS5 全文检索，可选向量检索与混合检索。
- CLI 本地交互通道和 WebSocket Gateway 集成通道。
- 内置工具覆盖文件读写、搜索、终端命令、计算、联网搜索、天气、技能加载、任务图创建、任务结果提交、产物清理和 Git 冲突处理等。
- Markdown 技能系统：在 `skills/<name>/SKILL.md` 中定义技能，由 `load_skill` 工具按需加载。
- 子 Agent 使用 Git worktree 隔离执行，并配套验证门禁、熔断器和失败提示注入。

## 项目结构

```text
.
├── taskweave/
│   ├── agents/                  # Agent 循环、Provider、Master/Subagent、Critic 逻辑
│   ├── channels/                # CLI 通道与通道抽象
│   ├── config/                  # Pydantic 配置模型和 YAML/环境变量加载器
│   ├── dispatcher/              # DAG 调度、事件总线、子 Agent 注册表
│   ├── gateway/                 # WebSocket 服务、路由、鉴权和 JSON 协议
│   ├── memory/                  # SQLite、向量和混合记忆存储
│   ├── sessions/                # 会话元数据与历史管理
│   ├── tools/                   # 工具基类、注册表、加载器和内置工具
│   ├── types/                   # 消息、事件、任务图和结果模型
│   ├── utils/                   # 日志和通用工具
│   ├── worktree/                # Git worktree 隔离与隔离工具
│   └── main.py                  # 直接运行的 CLI 入口
├── skills/                      # 内置 SKILL.md 技能
├── config.example.yaml          # 运行配置示例
├── requirements.txt             # 依赖版本锁定
└── pyproject.toml               # 包元数据和 `taskweave` 命令入口
```

运行时状态会写入 `data/`。Master/Subagent 模式还会使用 `.agents/worktrees/` 存放临时 Git worktree。

## 安装

需要 Python 3.12 或更高版本。

使用 `uv`：

```bash
uv sync
```

使用 `pip`：

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

也可以用可编辑安装启用命令入口：

```bash
pip install -e .
taskweave
```

## 配置

复制示例配置并填写凭据：

```bash
copy config.example.yaml config.yaml  # Windows
# cp config.example.yaml config.yaml  # macOS/Linux
```

至少需要一个 Provider 配置了 API Key。可以写在 `config.yaml` 中，也可以通过环境变量提供：

```bash
set OPENAI_API_KEY=sk-...
set ANTHROPIC_API_KEY=sk-ant-...
```

PowerShell 示例：

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

配置优先级为：

```text
默认值 < YAML 配置文件 < 环境变量 < CLI 参数
```

常用 Provider 环境变量：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `ANTHROPIC_API_KEY`

## 快速开始

运行默认单 Agent CLI：

```bash
python -m taskweave.main
```

指定配置文件：

```bash
python -m taskweave.main -c config.yaml
```

启用调试日志：

```bash
python -m taskweave.main --debug
```

启用 Master/Subagent 架构：

```bash
python -m taskweave.main --mode master_subagent
```

CLI 内置命令：

- `/quit`、`/exit`、`/q`：退出。
- `/new`：创建新会话。
- `/sessions`：列出已保存会话。

## 执行模式

`single_agent` 是默认模式。一个 `Agent` 负责处理会话、从历史和记忆检索中构建上下文、调用模型 Provider、执行工具，并把结果以 `Event` 流返回给通道。

`master_subagent` 模式使用 `MasterAgent` 通过 `create_task_graph` 创建任务图。调度器按 DAG 依赖执行任务节点，并通过专门的子 Agent 完成工作。子 Agent 通常在隔离 Git worktree 中执行，最后通过 `submit_task_result` 提交结构化结果；验证工具和门禁可以要求 lint 或测试成功后才允许提交。

可以通过命令行选择模式：

```bash
python -m taskweave.main --mode master_subagent
```

也可以写入 `config.yaml`：

```yaml
execution_mode: master_subagent
```

常用子 Agent 配置：

```yaml
subagent:
  max_spawn_depth: 5
  max_children_per_agent: 3
  max_total_agents: 50
  default_max_steps: 15
  default_timeout_ms: 120000
  max_concurrent_tasks: 3
  enable_worktree: true
  worktree_base_dir: ".agents/worktrees"
  auto_merge: true
  auto_cleanup: true
  require_validation: true
  validation_requirement: any
```

Master/Subagent 模式依赖 Git，建议在干净的 Git 仓库中运行。

## WebSocket Gateway

在 `config.yaml` 中启用 WebSocket 通道：

```yaml
channels:
  - cli
  - ws

gateway:
  host: "0.0.0.0"
  port: 8765
  auth_token: ""
```

Gateway 使用 JSON over WebSocket 协议。常见客户端消息：

```json
{"type": "auth", "token": "..."}
{"type": "message", "content": "Hello", "session_id": "optional-session-id"}
{"type": "new_session"}
{"type": "list_sessions"}
{"type": "ping"}
```

Agent 响应会以事件消息返回，包含 `event_type`、`data`、`session_id` 和 `timestamp`。

## 工具与技能

Python 工具继承 `taskweave.tools.base.Tool`，需要实现：

- `name`
- `description`
- `parameters`，格式为 JSON Schema
- `async execute(**kwargs) -> str`

内置工具从 `taskweave/tools/builtin/` 自动发现。额外工具目录可以通过配置添加：

```yaml
tool_dirs:
  - path/to/custom_tools
```

技能是带 YAML frontmatter 的 Markdown 文件：

```markdown
---
name: my-skill
description: Use this skill when ...
---

# My Skill

Instructions for the agent.
```

将技能放在 `skills/<name>/SKILL.md`，或者通过配置添加额外目录：

```yaml
skill_dirs:
  - path/to/skills
```

## 记忆系统

TaskWeave 使用 SQLite 存储会话和消息，默认数据库路径：

```text
data/taskweave.db
```

记忆配置选项：

- `enable_fts: true`：启用 SQLite FTS5/BM25 全文检索。
- `enable_vector: true`：启用基于 embedding 的向量检索。
- `enable_hybrid: true`：结合 FTS5 和向量检索，并加入权重、时间衰减与 MMR 去重。

向量和混合检索需要可用的 OpenAI 兼容 embedding API Key。

## 开发

运行测试：

```bash
pytest
```

运行单个测试：

```bash
pytest tests/test_file_tools.py::test_name -v
```

安装 Ruff 后运行 lint：

```bash
ruff check .
```

常用运行命令：

```bash
python -m taskweave.main
python -m taskweave.main -c config.yaml --debug
python -m taskweave.main --mode master_subagent
```

## 注意事项

- 不要提交生成的 SQLite 数据库、缓存或临时 worktree。
- Provider 凭据建议放在环境变量或本地未跟踪配置文件中。
- `web_search`、`weather` 等联网工具需要可用网络。
- `Agent.process_message()` 产出的是 `Event` 对象，不是原始字符串；自定义通道应使用 `async for` 消费事件流。

## 许可证

MIT License。详见 [LICENSE](LICENSE)。
