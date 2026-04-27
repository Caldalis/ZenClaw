"""
任务图构建工具 — Master Agent 的核心规划工具

create_task_graph 是 Master Agent 用于将复杂任务分解为 DAG 的工具。
支持:
  - depends_on 字段构建依赖关系
  - 动态角色定义（custom_role_prompt）
  - 自动分析执行顺序

工具 Schema:
{
    "name": "create_task_graph",
    "parameters": {
        "tasks": [
            {"id": "t1", "role": "CoderAgent", "instruction": "..."},
            {"id": "t2", "role": "CoderAgent", "instruction": "...", "depends_on": ["t1"]},
            {"id": "t3", "role": "DBMigrationAgent", "custom_role_prompt": "...", "instruction": "...", "depends_on": ["t2"]}
        ]
    }
}
"""

from __future__ import annotations

import json
from typing import Any

from miniclaw.tools.base import Tool
from miniclaw.types.task_graph import (
    TaskGraphRequest,
    TaskNode,
    build_task_graph_result,
    TaskGraphResult
)
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


class PendingGraphStore:
    """待执行任务图的存储抽象

    历史上 create_task_graph 用一个模块级 dict 存 request，调度器通过
    get_pending_graph 取走。问题：
      - 进程级单例 → 多个 Orchestrator 会互相串数据
      - clear 从来没被调用 → dict 只增不减，长期运行内存泄漏

    这里抽成显式对象。每个 Orchestrator 创建自己的 store 并绑定到
    CreateTaskGraphTool 实例。DAG 执行完后调度器显式 pop。
    """

    def __init__(self) -> None:
        self._store: dict[str, TaskGraphRequest] = {}

    def put(self, graph_id: str, request: TaskGraphRequest) -> None:
        self._store[graph_id] = request

    def get(self, graph_id: str) -> TaskGraphRequest | None:
        return self._store.get(graph_id)

    def pop(self, graph_id: str) -> TaskGraphRequest | None:
        return self._store.pop(graph_id, None)

    def __contains__(self, graph_id: str) -> bool:
        return graph_id in self._store

    def __len__(self) -> int:
        return len(self._store)


# 兼容遗留调用：未显式绑定的 CreateTaskGraphTool 会落到这个默认 store。
# 新代码应该走 Orchestrator 注入的独立 store。
_default_store = PendingGraphStore()


def get_pending_graph(graph_id: str) -> TaskGraphRequest | None:
    """[兼容 API] 获取默认 store 的 request"""
    return _default_store.get(graph_id)


def clear_pending_graph(graph_id: str) -> None:
    """[兼容 API] 清除默认 store 的 request"""
    _default_store.pop(graph_id)


def _coerce_list_field(value: Any, field_name: str) -> list:
    """将 AI 可能序列化成字符串的 list 字段解析回 list。"""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("%s 无法解析为 JSON 列表: %r，已按空列表处理", field_name, value)
            return []
        return parsed if isinstance(parsed, list) else [parsed]
    if value is None:
        return []
    return [value]


def _coerce_dict_field(value: Any, field_name: str) -> dict:
    """将 AI 可能序列化成字符串的 dict 字段解析回 dict。"""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("%s 无法解析为 JSON 对象: %r，已按空对象处理", field_name, value)
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


class CreateTaskGraphTool(Tool):
    """任务图构建工具 — Master Agent 的核心规划工具

    当任务过大时，Master Agent 使用此工具割裂上下文，将复杂任务分解为可并行执行的 DAG。
    """

    def __init__(self, store: PendingGraphStore | None = None) -> None:
        self._store: PendingGraphStore = store if store is not None else _default_store

    def bind_store(self, store: PendingGraphStore) -> None:
        """把工具重新绑定到指定 store（Orchestrator 初始化时调用）"""
        self._store = store

    @property
    def name(self) -> str:
        return "create_task_graph"

    @property
    def description(self) -> str:
        return """将复杂任务分解为有依赖关系的任务图(DAG)。

使用场景:
  - 任务需要多个步骤，且有明确的先后顺序
  - 不同步骤需要不同专业技能的角色
  - 某些步骤可以并行执行

关键特性:
  - 通过 depends_on 字段定义任务间的依赖关系
  - 支持预定义角色(CoderAgent, SearcherAgent, ReviewerAgent, TesterAgent, PlannerAgent)
  - 支持动态角色定义：当没有合适角色时，使用 custom_role_prompt 自定义

示例:
{
    "tasks": [
        {"id": "research", "role": "SearcherAgent", "instruction": "调研相关技术方案"},
        {"id": "design", "role": "PlannerAgent", "instruction": "设计系统架构", "depends_on": ["research"]},
        {"id": "implement_api", "role": "CoderAgent", "instruction": "实现 API 层", "depends_on": ["design"]},
        {"id": "implement_db", "role": "DBAgent", "custom_role_prompt": "你是数据库专家，只负责编写SQL迁移脚本", "instruction": "创建数据库表", "depends_on": ["design"]},
        {"id": "review", "role": "ReviewerAgent", "instruction": "代码审查", "depends_on": ["implement_api", "implement_db"]}
    ]
}

注意: 调用此工具后，系统会自动调度执行。你需要等待子任务完成后继续决策。"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "任务节点列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "任务唯一ID，用于 depends_on 引用。建议使用简短有意义的名称如 'research', 'implement'。",
                            },
                            "role": {
                                "type": "string",
                                "description": "角色名称。预定义角色: CoderAgent, SearcherAgent, ReviewerAgent, TesterAgent, PlannerAgent。也可自定义新角色名。",
                            },
                            "instruction": {
                                "type": "string",
                                "description": "发送给该角色 Agent 的任务指令，应清晰具体。",
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "依赖的任务ID列表。这些任务必须完成后才能执行当前任务。",
                            },
                            "custom_role_prompt": {
                                "type": "string",
                                "description": "自定义角色的系统提示词。当使用非预定义角色时必填，定义该角色的专业能力和限制。",
                            },
                            "custom_role_config": {
                                "type": "object",
                                "description": "自定义角色的额外配置",
                                "properties": {
                                    "allowed_tools": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "允许使用的工具列表",
                                    },
                                    "forbidden_tools": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "禁止使用的工具列表",
                                    },
                                    "requires_worktree": {
                                        "type": "boolean",
                                        "description": "是否需要 Git Worktree 物理隔离",
                                    },
                                    "max_steps": {
                                        "type": "integer",
                                        "description": "最大 ReAct 步数",
                                    },
                                    "timeout_ms": {
                                        "type": "integer",
                                        "description": "超时时间(毫秒)",
                                    },
                                },
                            },
                            "priority": {
                                "type": "integer",
                                "description": "优先级(数值越大越优先)，默认0",
                            },
                        },
                        "required": ["id", "role", "instruction"],
                    },
                },
                "parallel_execution": {
                    "type": "boolean",
                    "description": "是否尽可能并行执行(依赖关系仍需遵守)，默认true",
                    "default": True,
                },
                "fail_fast": {
                    "type": "boolean",
                    "description": "是否快速失败(任一任务失败立即终止)，默认false",
                    "default": False,
                },
                "max_concurrent": {
                    "type": "integer",
                    "description": "最大并发执行数，默认3",
                    "default": 3,
                },
                "description": {
                    "type": "string",
                    "description": "任务图的描述",
                    "default": "",
                },
            },
            "required": ["tasks"],
        }

    async def execute(self, **kwargs: Any) -> str:
        """执行任务图构建

        1. 解析输入参数
        2. 验证任务定义
        3. 分析依赖关系和执行顺序
        4. 注册动态角色
        5. 返回执行计划
        """
        try:
            # 解析任务列表 — 容错: AI 模型有时将 tasks 参数序列化为字符串而非数组
            tasks_data = kwargs.get("tasks", [])
            if isinstance(tasks_data, str):
                try:
                    tasks_data = json.loads(tasks_data)
                except json.JSONDecodeError:
                    return "错误: tasks 参数应为 JSON 数组，但收到了无法解析的字符串"
            if not tasks_data:
                return "错误: tasks 参数为空"

            tasks = []
            for task_data in tasks_data:

                depends_on = _coerce_list_field(
                    task_data.get("depends_on", []), field_name="depends_on"
                )
                custom_role_config = _coerce_dict_field(
                    task_data.get("custom_role_config", {}), field_name="custom_role_config"
                )

                task = TaskNode(
                    id=task_data.get("id", f"task_{len(tasks)}"),
                    role=task_data.get("role", "GenericAgent"),
                    instruction=task_data.get("instruction", ""),
                    depends_on=depends_on,
                    custom_role_prompt=task_data.get("custom_role_prompt"),
                    custom_role_config=custom_role_config,
                    priority=task_data.get("priority", 0),
                )
                tasks.append(task)

            # 验证依赖关系
            task_ids = {t.id for t in tasks}
            for task in tasks:
                for dep_id in task.depends_on:
                    if dep_id not in task_ids:
                        return f"错误: 任务 '{task.id}' 的依赖 '{dep_id}' 不存在"

            # 检测循环依赖
            if self._has_cycle(tasks):
                return "错误: 任务图存在循环依赖，无法执行"

            # 构建请求
            request = TaskGraphRequest(
                tasks=tasks,
                parallel_execution=kwargs.get("parallel_execution", True),
                fail_fast=kwargs.get("fail_fast", False),
                max_concurrent=kwargs.get("max_concurrent", 3),
                description=kwargs.get("description", ""),
            )

            # 构建结果
            result = build_task_graph_result(request)

            # 存储待执行的任务图（供调度器使用）
            self._store.put(result.graph_id, request)

            logger.info(
                "任务图已创建: %s (%d 任务, %d 层级)",
                result.graph_id, result.total_tasks, result.max_depth + 1
            )

            # 返回结构化结果
            return self._format_result(result)

        except Exception as e:
            logger.error("任务图构建失败: %s", e)
            return f"错误: {str(e)}"



    def _has_cycle(self, tasks: list[TaskNode]) -> bool:
        """检测任务图是否存在循环依赖"""
        task_map = {t.id: t for t in tasks}
        visited = set()
        rec_stack = set()

        def dfs(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)

            task = task_map.get(task_id)
            if task:
                for dep_id in task.depends_on:
                    if dep_id not in visited:
                        if dfs(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        return True

            rec_stack.remove(task_id)
            return False

        for task in tasks:
            if task.id not in visited:
                if dfs(task.id):
                    return True

        return False

    def _format_result(self, result: TaskGraphResult) -> str:
        """格式化返回结果

        返回结构化 JSON，包含:
          - 任务图 ID
          - 执行顺序（按层级）
          - 关键路径
          - 动态角色定义
        """
        # 格式化执行顺序
        order_desc = []
        for level_idx, level_tasks in enumerate(result.execution_order):
            order_desc.append(f"层级{level_idx}: {', '.join(level_tasks)}")

        # 格式化动态角色
        dynamic_roles_desc = []
        for role in result.dynamic_roles:
            dynamic_roles_desc.append(f"- {role.name}: {role.custom_prompt[:50]}...")

        output = {
            "graph_id": result.graph_id,
            "status": "created",
            "total_tasks": result.total_tasks,
            "max_depth": result.max_depth + 1,
            "execution_order": order_desc,
            "can_parallel": "同层级任务可并行执行",
            "dynamic_roles": dynamic_roles_desc if dynamic_roles_desc else ["无动态角色"],
            "next_action": "等待系统调度执行，子任务完成后将返回结果",
        }

        return json.dumps(output, ensure_ascii=False, indent=2)


# 导出工具实例和辅助函数
create_task_graph_tool = CreateTaskGraphTool()

