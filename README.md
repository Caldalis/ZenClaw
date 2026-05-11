# TaskWeave

English | [中文](README_zh.md)

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

TaskWeave is a Python 3.12 agent framework for local CLI and WebSocket-driven AI workflows. The codebase is internally named `taskweave` and is built around an event-streaming agent loop, pluggable model providers, persistent memory, tool calling, lazy-loaded `SKILL.md` skills, and an optional master/subagent architecture for decomposing larger work into isolated Git worktrees.

## Features

- Single-agent ReAct loop with tool calling and streamed `Event` objects.
- Master/subagent mode that builds DAG task graphs and schedules subagents with dependency-aware execution.
- Provider failover across OpenAI-compatible APIs and Anthropic Claude.
- SQLite-backed session memory with FTS5 search, plus optional vector and hybrid retrieval.
- CLI channel for local use and WebSocket gateway for client integrations.
- Built-in tools for file operations, search, terminal commands, calculation, web search, weather, skills, task graphs, task submission, artifact cleanup, and Git conflict handling.
- Markdown skill system: place `SKILL.md` files under `skills/<name>/` and load them on demand through the `load_skill` tool.
- Worktree isolation for subagents, with validation gates and circuit breakers to reduce repeated failing loops.

## Project Layout

```text
.
├── taskweave/
│   ├── agents/                  # Agent loop, providers, master/subagent orchestration, critic logic
│   ├── channels/                # CLI and channel abstractions
│   ├── config/                  # Pydantic settings and YAML/env loader
│   ├── dispatcher/              # DAG scheduling, event bus, subagent registry
│   ├── gateway/                 # WebSocket server, router, auth, JSON protocol
│   ├── memory/                  # SQLite, vector, and hybrid memory stores
│   ├── sessions/                # Session metadata and history management
│   ├── tools/                   # Tool base class, registry, loader, built-in tools
│   ├── types/                   # Message, event, task graph, result models
│   ├── utils/                   # Logging and shared utilities
│   ├── worktree/                # Git worktree isolation and isolated tools
│   └── main.py                  # Direct CLI entry point
├── skills/                      # Built-in SKILL.md definitions
├── config.example.yaml          # Example runtime configuration
├── requirements.txt             # Runtime/test dependency pins
└── pyproject.toml               # Package metadata and `taskweave` script entry point
```

Runtime state is created under `data/`. Master/subagent mode also uses `.agents/worktrees/` for temporary Git worktrees.

## Installation

Python 3.12 or newer is required.

Using `uv`:

```bash
uv sync
```

Using `pip`:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
taskweave
```

## Configuration

Copy the example configuration and add credentials:

```bash
copy config.example.yaml config.yaml  # Windows
# cp config.example.yaml config.yaml  # macOS/Linux
```

At least one configured provider must have an API key. You can set keys in `config.yaml` or through environment variables:

```bash
set OPENAI_API_KEY=sk-...
set ANTHROPIC_API_KEY=sk-ant-...
```

PowerShell example:

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Configuration precedence is:

```text
defaults < YAML file < environment variables < CLI flags
```

Common provider environment variables include:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `ANTHROPIC_API_KEY`

## Quick Start

Run the default single-agent CLI:

```bash
python -m taskweave.main
```

Run with an explicit configuration file:

```bash
python -m taskweave.main -c config.yaml
```

Enable debug logging:

```bash
python -m taskweave.main --debug
```

Use the master/subagent architecture:

```bash
python -m taskweave.main --mode master_subagent
```

CLI commands:

- `/quit`, `/exit`, `/q` exits the CLI.
- `/new` starts a new session.
- `/sessions` lists saved sessions.

## Execution Modes

`single_agent` is the default mode. One `Agent` handles the conversation, builds context from session history and memory search, calls the configured AI provider, executes tools, and yields `Event` objects back to the channel.

`master_subagent` mode uses a `MasterAgent` to create task graphs with `create_task_graph`. The scheduler executes DAG nodes through specialized subagents, usually in isolated Git worktrees. Subagents submit structured results through `submit_task_result`; validation tools and gatekeepers can require successful linting or tests before a task is accepted.

You can choose the mode with a CLI flag:

```bash
python -m taskweave.main --mode master_subagent
```

Or in `config.yaml`:

```yaml
execution_mode: master_subagent
```

Relevant subagent settings include:

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

Master/subagent mode expects Git to be available and works best inside a clean Git repository.

## WebSocket Gateway

Enable the WebSocket channel in `config.yaml`:

```yaml
channels:
  - cli
  - ws

gateway:
  host: "0.0.0.0"
  port: 8765
  auth_token: ""
```

The gateway speaks JSON over WebSocket. Common client messages:

```json
{"type": "auth", "token": "..."}
{"type": "message", "content": "Hello", "session_id": "optional-session-id"}
{"type": "new_session"}
{"type": "list_sessions"}
{"type": "ping"}
```

Agent responses are returned as event messages containing `event_type`, `data`, `session_id`, and `timestamp`.

## Tools and Skills

Python tools inherit from `taskweave.tools.base.Tool` and implement:

- `name`
- `description`
- `parameters` as JSON Schema
- `async execute(**kwargs) -> str`

Built-in tools are discovered from `taskweave/tools/builtin/`. Additional tool directories can be configured with:

```yaml
tool_dirs:
  - path/to/custom_tools
```

Skills are Markdown files with YAML frontmatter:

```markdown
---
name: my-skill
description: Use this skill when ...
---

# My Skill

Instructions for the agent.
```

Place skills under `skills/<name>/SKILL.md` or add extra directories with:

```yaml
skill_dirs:
  - path/to/skills
```

## Memory

TaskWeave stores sessions and messages in SQLite. The default database path is:

```text
data/taskweave.db
```

Memory options:

- `enable_fts: true` enables SQLite FTS5/BM25 full-text search.
- `enable_vector: true` enables embedding-based vector retrieval.
- `enable_hybrid: true` combines FTS5 and vector retrieval with weighting, time decay, and MMR deduplication.

Vector and hybrid modes require an OpenAI-compatible embedding API key.

## Development

Run tests:

```bash
pytest
```

Run a focused test:

```bash
pytest tests/test_file_tools.py::test_name -v
```

Run linting when Ruff is installed:

```bash
ruff check .
```

Useful runtime commands:

```bash
python -m taskweave.main
python -m taskweave.main -c config.yaml --debug
python -m taskweave.main --mode master_subagent
```

## Notes

- Do not commit generated SQLite databases, caches, or temporary worktrees.
- Keep provider credentials in environment variables or local untracked config files.
- Tools that access the network, such as `web_search` and `weather`, require network availability.
- `Agent.process_message()` yields `Event` objects, not raw strings; custom channels should consume it with `async for`.

## License

MIT License. See [LICENSE](LICENSE).
