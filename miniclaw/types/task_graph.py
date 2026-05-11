"""
任务图数据模型

用于 Master Agent 构建 DAG 任务分解图。

核心概念:
  - TaskNode: 任务节点，包含 role、instruction、depends_on
  - TaskGraphRequest: create_task_graph 工具的输入参数
  - TaskGraphResult: 任务图构建结果
  - DynamicRole: 动态角色定义（Master 可自定义新角色）
  - FailureCategory: 失败分类枚举，使 Master 能基于根因决策而非自由文本
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from miniclaw.types.enums import AgentRole


class FailureCategory(str, Enum):
    """任务失败的结构化分类 — 由 task_scheduler 根据根因填写。

    Master Agent 根据分类决定下一步动作：
      - VALIDATION_UNMET    → 不要重派同样指令；要么放宽契约，要么换角色
      - AGENT_MAX_ITERATIONS → Agent 走入死循环；通常源于 prompt 含混，重写 instruction
      - TIMEOUT             → 任务太重；拆得更细或加 timeout
      - CIRCUIT_BREAKER     → 工具反复失败；不要重试同种动作
      - DEPENDENCY_FAILED   → 上游任务失败；本任务永远跑不起来，从 DAG 摘掉
      - WORKTREE_CREATION   → Git 隔离失败；可能是分支冲突或权限问题
      - PROVIDER_ERROR      → AI 服务故障；通常等几秒再来
      - UNKNOWN             → 兜底，需要人工分析日志
    """

    VALIDATION_UNMET = "validation_unmet"
    AGENT_MAX_ITERATIONS = "agent_max_iterations"
    TIMEOUT = "timeout"
    CIRCUIT_BREAKER = "circuit_breaker"
    DEPENDENCY_FAILED = "dependency_failed"
    WORKTREE_CREATION = "worktree_creation"
    WORKSPACE_CORRUPTED = "workspace_corrupted"
    """worktree 在执行中段被毁/状态不一致，健康检查失败"""

    SANDBOX_VIOLATION = "sandbox_violation"
    """文件工具在不存在的 worktree 根上尝试自动建目录被拒"""

    COMMIT_FAILED = "commit_failed"
    """git commit 失败 —— 通常 hook 拒绝 / 索引锁住"""

    PROVIDER_ERROR = "provider_error"
    UNKNOWN = "unknown"

    @classmethod
    def from_error(cls, error_msg: str) -> "FailureCategory":
        """根据错误文本启发式归类。
        task_scheduler 在抛 DispatcherError 时会带具体语义，但兜底也要稳。"""
        s = (error_msg or "").lower()
        if "处理轮数超过上限" in error_msg or "max iterations" in s:
            return cls.AGENT_MAX_ITERATIONS
        if "timeout" in s or "超时" in error_msg:
            return cls.TIMEOUT
        if "熔断" in error_msg or "circuit" in s:
            return cls.CIRCUIT_BREAKER
        if "上游依赖" in error_msg or "dependency" in s:
            return cls.DEPENDENCY_FAILED
        if "worktree" in s or "git" in s:
            return cls.WORKTREE_CREATION
        if "provider" in s or "ai" in s and ("调用失败" in error_msg or "调用错误" in error_msg):
            return cls.PROVIDER_ERROR
        if "验证" in error_msg or "validation" in s or "提交被阻止" in error_msg:
            return cls.VALIDATION_UNMET
        return cls.UNKNOWN


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

    requires_validation: bool | None = None
    """提交前是否必须通过 run_linter/run_tests 验证。

    None 表示 master 没显式声明，由 SubagentFactory 按 allowed_tools/forbidden_tools
    是否含写工具自动推断（写代码 → True；只读 → False）。
    """

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

    task_failure_categories: dict[str, FailureCategory] = Field(default_factory=dict)
    """失败任务的结构化分类 (task_id -> FailureCategory)。
    Master 据此决定下一步：见 FailureCategory 文档。"""

    # 双层结果：每个 task 的"系统观察 + agent 自报"。
    # task_outcomes[id].final_status 是 task 真正的最终状态。
    # Master 据此判定 completed_tasks vs failed_tasks，不再相信 agent 自报。
    # （这里用 dict[str, dict] 而不是 dict[str, TaskOutcome] 是为了 pydantic
    #  序列化兼容；具体类型在 task_outcome.py，scheduler 在填充时把它转成 dict。）
    task_outcomes: dict[str, Any] = Field(default_factory=dict)
    """系统观察 outcome (task_id -> TaskOutcome.to_dict 序列化形式)"""

    task_dependencies: dict[str, list[str]] = Field(default_factory=dict)
    """每个任务的依赖列表 (task_id -> [dep_id, ...])"""

    @property
    def is_complete(self) -> bool:
        """任务图是否全部完成"""
        return len(self.completed_tasks) == self.total_tasks

    @property
    def has_failures(self) -> bool:
        """是否有失败任务"""
        return len(self.failed_tasks) > 0


def compute_dag_signature(request: "TaskGraphRequest") -> str:
    """计算 DAG 的语义指纹（仅看 role + 归一化后的 instruction + 依赖拓扑）。

    Master Agent 用此指纹检测"用同一张 DAG 重派失败任务"的反模式：
      - 忽略 task.id（Master 把 'implement' 改名 'impl' 后仍命中）
      - 忽略空白差异（连续空格/换行被规整化）
      - **依赖关系按内容指纹表达**而不是 task.id —— 否则改名 task.id 同时
        改 depends_on 引用就能绕过查重；这里把 task 自身先指纹化，
        再用其他任务的指纹来表达 depends_on，达成结构相等性
    """

    def _content_fp(t: "TaskNode") -> str:
        """单 task 的内容指纹：仅考虑 role + 归一化 instruction + custom_role_prompt。
        不参考 task.id，因为它要在 depends_on 里被作为锚点。"""
        normalized_instr = " ".join((t.instruction or "").split()).strip().lower()
        custom = " ".join((t.custom_role_prompt or "").split()).strip().lower()
        raw = f"{(t.role or '').strip()}::{normalized_instr}::{custom}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]

    # 第一遍：构建 task.id -> 内容指纹 的映射
    id_to_fp: dict[str, str] = {t.id: _content_fp(t) for t in request.tasks}

    # 第二遍：每个 task 用 (内容指纹, 依赖指纹列表) 描述
    items: list[tuple[str, tuple[str, ...]]] = []
    for t in request.tasks:
        dep_fps = sorted(id_to_fp.get(dep_id, dep_id) for dep_id in t.depends_on)
        items.append((id_to_fp[t.id], tuple(dep_fps)))
    # 排序确保 task 顺序不影响指纹
    items.sort()
    serialized = "|".join(
        f"{fp}<-{','.join(deps)}" for fp, deps in items
    )
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:16]


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

    task_dependencies: dict[str, list[str]] = Field(default_factory=dict)
    """每个任务的依赖列表 (task_id -> [dep_id, ...])"""

    def get_ready_tasks(self, completed_ids: set[str], running_ids: set[str]) -> list[str]:
        """获取当前准备好执行的任务

        条件:
          1. 所有依赖已完成
          2. 当前未在执行中
          3. 当前未已完成
        """
        ready = []
        for level in self.levels:
            for task_id in level:
                if task_id in completed_ids or task_id in running_ids:
                    continue
                deps = self.task_dependencies.get(task_id, [])
                if all(dep_id in completed_ids for dep_id in deps):
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
                # None 信号：未显式声明 → 后续由 factory 智能推断
                requires_validation=config.get("requires_validation"),
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

    task_dependencies = {t.id: list(t.depends_on) for t in request.tasks}

    return TaskGraphResult(
        total_tasks=len(request.tasks),
        max_depth=max_depth,
        execution_order=execution_order,
        dynamic_roles=dynamic_roles,
        status="pending",
        task_dependencies=task_dependencies,
    )