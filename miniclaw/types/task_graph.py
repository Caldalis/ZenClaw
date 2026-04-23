"""
任务图数据模型

用于 Master Agent 构建 DAG 任务分解图。

核心概念:
  - TaskNode: 任务节点，包含 role、instruction、depends_on
  - TaskGraphRequest: create_task_graph 工具的输入参数
  - TaskGraphResult: 任务图构建结果
  - DynamicRole: 动态角色定义（Master 可自定义新角色）
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from miniclaw.types.enums import AgentRole


class DynamicRole(BaseModel):
    """动态角色定义

    Master Agent 可以在发现没有合适角色时，自定义新角色。
    """
    name: str
    """角色名称（如 'DBMigrationAgent'）"""

    custom_prompt: str
    """该角色的专属系统提示词"""

    allowed_tools: list[str] = Field(default_factory=list)
    """允许使用的工具列表（空=允许所有）"""

    forbidden_tools: list[str] = Field(default_factory=list)
    """禁止使用的工具列表"""

    requires_worktree: bool = False
    """是否需要 Git Worktree 物理隔离"""

    max_steps: int = 15
    """最大 ReAct 步数"""

    timeout_ms: int = 120000
    """超时时间（毫秒）"""


class TaskNode(BaseModel):
    """任务节点 — DAG 中的一个任务

    Master Agent 通过 create_task_graph 工具创建任务节点。
    """
    id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    """任务 ID（用于 depends_on 引用）"""

    role: str
    """角色名称（可以是预定义角色如 'CoderAgent'，或动态角色名）"""

    instruction: str
    """任务指令（发送给 Subagent 的输入消息）"""

    depends_on: list[str] = Field(default_factory=list)
    """依赖的任务 ID 列表（必须完成后才能执行此任务）"""

    # 动态角色定义（可选，当 role 不在预定义列表时使用）
    custom_role_prompt: str | None = None
    """自定义角色的系统提示词"""

    custom_role_config: dict[str, Any] = Field(default_factory=dict)
    """自定义角色的额外配置（如 allowed_tools, forbidden_tools）"""

    # 执行结果（运行时填充）
    result: str | None = None
    """执行结果"""
    error: str | None = None
    """错误信息"""
    is_error: bool = False
    """是否为错误结果"""

    # 优先级（可选）
    priority: int = 0
    """优先级（数值越大越优先）"""

    # 标签（可选）
    tags: list[str] = Field(default_factory=list)
    """任务标签（用于分类和筛选）"""

    def is_ready(self, completed_ids: set[str]) -> bool:
        """检查任务是否准备好执行（所有依赖已完成）"""
        return all(dep_id in completed_ids for dep_id in self.depends_on)

    def is_leaf(self) -> bool:
        """是否为叶子节点（没有其他任务依赖它）"""
        # 这个判断需要外部信息（谁依赖我）
        # 这里只检查是否没有自定义的 depends_on
        return len(self.depends_on) == 0


class TaskGraphRequest(BaseModel):
    """create_task_graph 工具的输入参数"""
    tasks: list[TaskNode]
    """任务节点列表"""

    parallel_execution: bool = True
    """是否尽可能并行执行（依赖关系仍需遵守）"""

    fail_fast: bool = False
    """是否快速失败（任一任务失败立即终止整个 DAG）"""

    max_concurrent: int = 3
    """最大并发执行数"""

    description: str = ""
    """任务图的描述（用于日志）"""


class TaskGraphResult(BaseModel):
    """任务图构建结果 — create_task_graph 工具的输出"""
    graph_id: str = Field(default_factory=lambda: f"graph_{uuid.uuid4().hex[:8]}")
    """任务图 ID"""

    total_tasks: int
    """总任务数"""

    max_depth: int
    """最大依赖深度（用于估算执行时间）"""

    execution_order: list[list[str]]
    """执行顺序（按层级分组，同层级可并行）"""

    dynamic_roles: list[DynamicRole]
    """动态角色定义列表"""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # 状态信息（运行时填充）
    status: str = "pending"
    """任务图状态: pending, running, completed, failed"""

    completed_tasks: list[str] = Field(default_factory=list)
    """已完成的任务 ID"""

    failed_tasks: list[str] = Field(default_factory=list)
    """失败的任务 ID"""

    task_errors: dict[str, str] = Field(default_factory=dict)
    """失败任务的错误详情 (task_id -> error_message)"""

    @property
    def is_complete(self) -> bool:
        """任务图是否全部完成"""
        return len(self.completed_tasks) == self.total_tasks

    @property
    def has_failures(self) -> bool:
        """是否有失败任务"""
        return len(self.failed_tasks) > 0


class ExecutionPlan(BaseModel):
    """执行计划 — 任务图的执行顺序分析"""
    graph_id: str
    """关联的任务图 ID"""

    levels: list[list[str]]
    """执行层级（同一层级内的任务可并行执行）"""

    level_count: int
    """层级数量"""

    total_tasks: int
    """总任务数"""

    estimated_steps: int
    """预估执行步数"""

    critical_path: list[str]
    """关键路径（最长依赖链）"""

    def get_ready_tasks(self, completed_ids: set[str], running_ids: set[str]) -> list[str]:
        """获取当前准备好执行的任务

        条件:
          1. 所有依赖已完成
          2. 当前未在执行中
        """
        ready = []
        for level in self.levels:
            for task_id in level:
                if task_id not in completed_ids and task_id not in running_ids:
                    # 找到该任务的依赖
                    # 这里需要外部信息，简化实现
                    ready.append(task_id)
        return ready


def analyze_execution_order(tasks: list[TaskNode]) -> list[list[str]]:
    """分析任务执行顺序（按依赖层级分组）

    返回: 执行层级列表，同一层级内的任务可并行执行

    算法:
      1. 构建依赖图
      2. 使用拓扑排序确定层级
      3. 同层级的任务可以并行
    """
    if not tasks:
        return []

    # 构建任务 ID 映射
    task_map = {t.id: t for t in tasks}

    # 计算每个任务的层级（最长依赖链长度）
    levels: dict[str, int] = {}

    def compute_level(task_id: str) -> int:
        if task_id in levels:
            return levels[task_id]

        task = task_map.get(task_id)
        if task is None:
            return 0

        if not task.depends_on:
            levels[task_id] = 0
            return 0

        # 层级 = 最大依赖层级 + 1
        max_dep_level = max(compute_level(dep_id) for dep_id in task.depends_on)
        levels[task_id] = max_dep_level + 1
        return levels[task_id]

    # 计算所有任务层级
    for task in tasks:
        compute_level(task.id)

    # 按层级分组
    max_level = max(levels.values()) if levels else 0
    execution_order = [[] for _ in range(max_level + 1)]

    for task_id, level in levels.items():
        execution_order[level].append(task_id)

    return execution_order


def find_critical_path(tasks: list[TaskNode]) -> list[str]:
    """找到关键路径（最长依赖链）

    关键路径决定了任务图的最短完成时间。
    """
    task_map = {t.id: t for t in tasks}

    def get_chain(task_id: str) -> list[str]:
        task = task_map.get(task_id)
        if task is None or not task.depends_on:
            return [task_id]

        # 选择最长依赖链
        longest_chain = []
        for dep_id in task.depends_on:
            chain = get_chain(dep_id)
            if len(chain) > len(longest_chain):
                longest_chain = chain

        return longest_chain + [task_id]

    # 找到没有后继的任务（叶子）
    all_depended = set()
    for task in tasks:
        for dep_id in task.depends_on:
            all_depended.add(dep_id)

    leaf_tasks = [t.id for t in tasks if t.id not in all_depended or not t.depends_on]

    # 选择最长链
    critical_path = []
    for leaf_id in leaf_tasks:
        chain = get_chain(leaf_id)
        if len(chain) > len(critical_path):
            critical_path = chain

    return critical_path


def extract_dynamic_roles(tasks: list[TaskNode]) -> list[DynamicRole]:
    """从任务中提取动态角色定义

    当任务的 role 不在预定义角色列表时，使用 custom_role_prompt 定义新角色。
    """
    dynamic_roles = []
    seen_roles = set()

    for task in tasks:
        # 检查是否需要动态角色
        if task.custom_role_prompt and task.role not in seen_roles:
            seen_roles.add(task.role)

            # 从 custom_role_config 提取配置
            config = task.custom_role_config

            dynamic_role = DynamicRole(
                name=task.role,
                custom_prompt=task.custom_role_prompt,
                allowed_tools=config.get("allowed_tools", []),
                forbidden_tools=config.get("forbidden_tools", []),
                requires_worktree=config.get("requires_worktree", False),
                max_steps=config.get("max_steps", 15),
                timeout_ms=config.get("timeout_ms", 120000),
            )
            dynamic_roles.append(dynamic_role)

    return dynamic_roles


def build_task_graph_result(request: TaskGraphRequest) -> TaskGraphResult:
    """构建任务图结果

    分析任务依赖关系，生成执行计划。
    """
    execution_order = analyze_execution_order(request.tasks)
    critical_path = find_critical_path(request.tasks)
    dynamic_roles = extract_dynamic_roles(request.tasks)

    max_depth = len(execution_order) - 1 if execution_order else 0

    return TaskGraphResult(
        total_tasks=len(request.tasks),
        max_depth=max_depth,
        execution_order=execution_order,
        dynamic_roles=dynamic_roles,
        status="pending",
    )