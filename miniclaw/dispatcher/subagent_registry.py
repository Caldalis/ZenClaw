"""
Subagent Registry - 安全护栏配置

硬编码的安全边界，控制:
  - 最大衍生深度 (maxSpawnDepth)
  - 最大并发子节点数 (maxChildrenPerAgent)
  - Subagent 执行指标 (max_steps, timeout_ms)
  - 角色配置注册表

护栏检查:
  1. 深度检查: callerDepth + 1 > maxSpawnDepth → 抛出异常
  2. 广度检查: currentChildren >= maxChildrenPerAgent → 抛出异常
  3. 超时监控: Dispatcher 自动监控，超时后标记 TIMEOUT
"""



from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from miniclaw.types.enums import AgentRole


class GuardrailViolationError(Exception):
    """护栏违规异常"""

    def __init__(self, violation_type: str, message: str, details: dict[str, Any] = {}):
        self.violation_type = violation_type
        self.details = details
        super().__init__(f"[{violation_type}] {message}")


@dataclass
class GuardrailConfig:
    """护栏配置 — 系统安全边界"""

    # 深度限制
    max_spawn_depth: int = 5
    """最大衍生深度。当 callerDepth + 1 > maxSpawnDepth 时，拦截并抛出异常。
    Master Agent 深度为 0，其子 Agent 深度为 1，以此类推。
    """

    # 广度限制
    max_children_per_agent: int = 3
    """单个 Agent 一次性 Spawn 的最大子节点数量。
    防止任务爆炸，控制并发度。
    """

    # 执行限制（默认值，可被具体 SubagentSpec 覆盖）
    default_max_steps: int = 15
    """Subagent 默认最大 ReAct 步数。防止无限循环。"""

    default_timeout_ms: int = 120000
    """Subagent 默认超时时间（毫秒）。默认 2 分钟。"""

    # 全局熔断配置
    global_timeout_ms: int = 600000
    """全局任务执行超时（毫秒）。默认 10 分钟，超过此时间整个 DAG 终止。"""

    max_total_agents: int = 50
    """整个 DAG 中允许的最大 Agent 节点总数。防止资源耗尽。"""

    error_retry_limit: int = 2
    """失败节点的重试次数限制。"""

    def check_spawn_depth(self, caller_depth: int) -> None:
        """检查衍生深度是否超过限制"""
        new_depth = caller_depth + 1
        if new_depth > self.max_spawn_depth:
            raise GuardrailViolationError(
                violation_type="DEPTH_OVERFLOW",
                message=f"衍生深度超限: caller_depth={caller_depth}, new_depth={new_depth}, max={self.max_spawn_depth}",
                details={
                    "caller_depth": caller_depth,
                    "new_depth": new_depth,
                    "max_spawn_depth": self.max_spawn_depth,
                },
            )

    def check_children_count(self, current_children: int, parent_id: str) -> None:
        """检查并发子节点数是否超过限制"""
        if current_children >= self.max_children_per_agent:
            raise GuardrailViolationError(
                violation_type="CHILDREN_OVERFLOW",
                message=f"子节点数量超限: parent={parent_id}, current={current_children}, max={self.max_children_per_agent}",
                details={
                    "parent_id": parent_id,
                    "current_children": current_children,
                    "max_children": self.max_children_per_agent,
                },
            )

    def check_total_agents(self, total_count: int) -> None:
        """检查 Agent 总数是否超过限制"""
        if total_count >= self.max_total_agents:
            raise GuardrailViolationError(
                violation_type="AGENT_OVERFLOW",
                message=f"Agent 总数超限: current={total_count}, max={self.max_total_agents}",
                details={
                    "current_count": total_count,
                    "max_total_agents": self.max_total_agents,
                },
            )


@dataclass
class SubagentSpec:
    """Subagent 规格 — 定义具体角色的执行参数"""

    role: AgentRole
    """角色类型"""

    name: str
    """角色名称（用于日志和调试）"""

    description: str
    """角色描述（用于系统提示词）"""

    system_prompt: str
    """该角色的专属系统提示词"""

    max_steps: int = 15
    """最大 ReAct 步数（可覆盖默认值）"""

    timeout_ms: int = 120000
    """超时时间（毫秒，可覆盖默认值）"""

    allowed_tools: list[str] = field(default_factory=list)
    """允许使用的工具列表（空=允许所有）"""

    forbidden_tools: list[str] = field(default_factory=list)
    """禁止使用的工具列表"""

    # Worktree 配置
    requires_worktree: bool = False
    """是否需要 Git Worktree 物理隔离"""

    worktree_prefix: str = ""
    """Worktree 目录前缀（如 'feature/', 'fix/'）"""

    # 验证门禁
    # 审查/搜索/规划类角色不产出代码（forbidden 包含 write_file/edit_file），
    # 强制 run_linter/run_tests 验证毫无意义且会让它们陷入死循环（反复试 linter）。
    # 默认 True（产出代码的 coder/tester），False 用于只读角色。
    requires_validation: bool = True
    """提交结果前是否需要至少一次成功的 linter/tests 验证"""


class SubagentRegistry:
    """Subagent 注册表

    管理:
      1. 护栏配置
      2. 角色规格注册
      3. Spawn 检查
    """

    def __init__(self, guardrails: GuardrailConfig | None = None):
        self.guardrails = guardrails or GuardrailConfig()
        self._specs: dict[AgentRole, SubagentSpec] = {}
        self._register_builtin_specs()

    def _register_builtin_specs(self) -> None:
        """注册内置角色规格"""
        # 导入 Master Agent 提示词
        try:
            from miniclaw.agents.prompts import MASTER_AGENT_PROMPT
            master_prompt = MASTER_AGENT_PROMPT
        except ImportError:
            master_prompt = "你是 Master Agent，负责将复杂任务分解为 DAG 并调度子 Agent 执行。"

        # Master Agent
        self.register(SubagentSpec(
            role=AgentRole.MASTER,
            name="Master Dispatcher",
            description="主调度器，负责任务分解、DAG 构建、子 Agent 管理",
            system_prompt=master_prompt,
            max_steps=50,  # Master Agent 需要更多步数来管理整个流程
            timeout_ms=300000,  # 5 分钟
            requires_worktree=False,
        ))

        # Coder
        self.register(SubagentSpec(
            role=AgentRole.CODER,
            name="Coder Agent",
            description="编码执行者，负责实现具体代码",
            system_prompt="你是 Coder Agent，专注于代码实现。根据任务描述编写高质量、可维护的代码。完成后返回结构化结果。",
            max_steps=self.guardrails.default_max_steps,
            timeout_ms=self.guardrails.default_timeout_ms,
            requires_worktree=True,
            worktree_prefix="feature/",
            allowed_tools=["read_file", "write_file", "edit_file", "ls", "glob", "grep", "terminal"],
        ))

        # Searcher
        self.register(SubagentSpec(
            role=AgentRole.SEARCHER,
            name="Searcher Agent",
            description="搜索/研究执行者，负责信息检索和分析",
            system_prompt="你是 Searcher Agent，专注于信息检索和分析。使用搜索工具查找相关信息，整理并返回结构化报告。",
            max_steps=self.guardrails.default_max_steps,
            timeout_ms=90000,  # 搜索任务通常较快
            requires_worktree=False,
            allowed_tools=["web_search", "read_file", "grep", "glob"],
            forbidden_tools=["write_file", "edit_file", "terminal"],  # 搜索者不应修改文件
            requires_validation=False,
        ))

        # Reviewer
        self.register(SubagentSpec(
            role=AgentRole.REVIEWER,
            name="Reviewer Agent",
            description="代码审查执行者，负责质量检查",
            system_prompt="你是 Reviewer Agent，专注于代码审查和质量检查。检查代码的正确性、安全性、可维护性，返回审查报告。",
            max_steps=20,
            timeout_ms=120000,
            requires_worktree=True,
            worktree_prefix="review/",
            allowed_tools=["read_file", "grep", "glob", "ls"],
            forbidden_tools=["write_file", "edit_file", "terminal"],  # 审查者不应修改文件
            requires_validation=False,
        ))

        # Tester
        self.register(SubagentSpec(
            role=AgentRole.TESTER,
            name="Tester Agent",
            description="测试执行者，负责编写和运行测试",
            system_prompt="你是 Tester Agent，专注于测试编写和执行。根据需求编写测试用例，运行测试并返回结果报告。",
            max_steps=self.guardrails.default_max_steps,
            timeout_ms=180000,
            requires_worktree=True,
            worktree_prefix="test/",
            allowed_tools=["read_file", "write_file", "edit_file", "ls", "glob", "grep", "terminal"],
        ))

        # Planner
        self.register(SubagentSpec(
            role=AgentRole.PLANNER,
            name="Planner Agent",
            description="规划执行者，负责详细任务规划",
            system_prompt="你是 Planner Agent，专注于任务规划。将模糊的任务需求转化为清晰的执行计划，输出结构化的步骤列表。",
            max_steps=12,
            timeout_ms=90000,
            requires_worktree=False,
            requires_validation=False,
        ))

        # Generic
        self.register(SubagentSpec(
            role=AgentRole.GENERIC,
            name="Generic Agent",
            description="通用执行者，处理简单任务",
            system_prompt="你是通用 Agent，负责执行简单任务。根据指令完成任务并返回结果。",
            max_steps=self.guardrails.default_max_steps,
            timeout_ms=self.guardrails.default_timeout_ms,
        ))

    def register(self, spec: SubagentSpec) -> None:
        """注册角色规格"""
        self._specs[spec.role] = spec

    def get_spec(self, role: AgentRole) -> SubagentSpec | None:
        """获取角色规格"""
        return self._specs.get(role)

    def check_can_spawn(
        self,
        caller_depth: int,
        caller_children_count: int,
        caller_id: str,
        total_agent_count: int,
    ) -> None:
        """综合检查是否可以 Spawn 新 Agent

        检查项:
          1. 深度限制
          2. 广度限制
          3. Agent 总数限制

        任一检查失败都会抛出 GuardrailViolationError。
        """
        self.guardrails.check_spawn_depth(caller_depth)
        self.guardrails.check_children_count(caller_children_count, caller_id)
        self.guardrails.check_total_agents(total_agent_count)

    def validate_tool_usage(self, role: AgentRole, tool_name: str) -> bool:
        """验证工具使用权限

        Returns:
          True: 允许使用
          False: 禁止使用
        """
        spec = self.get_spec(role)
        if spec is None:
            return True  # 未注册的角色默认允许所有工具

        # 先检查禁止列表
        if tool_name in spec.forbidden_tools:
            return False

        # 再检查允许列表（空=允许所有）
        if spec.allowed_tools and tool_name not in spec.allowed_tools:
            return False

        return True

    @property
    def available_roles(self) -> list[AgentRole]:
        """列出所有已注册的角色"""
        return list(self._specs.keys())

    @property
    def guardrails_config(self) -> GuardrailConfig:
        """获取护栏配置"""
        return self.guardrails