"""
Subagent 执行器集成 — 将 Worktree 与 Subagent 执行流程整合

核心职责:
  1. 在 Subagent Spawn 时准备 worktree
  2. 绑定隔离工具到 Agent 执行器
  3. 收集执行结果
  4. 任务完成后提交、合并、清理

工作流程:
  1. prepare_for_execution() → 准备 worktree + 隔离工具
  2. Agent 执行任务（使用隔离工具）
  3. collect_result() → 收集结果
  4. finalize_execution() → 提交变更 + 合并 + 清理
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from miniclaw.agents.subagent_factory import SubagentConfig
from miniclaw.types.structured_result import StructuredResult, TaskStatus, validate_result
from miniclaw.types.task_graph import TaskNode
from miniclaw.types.turn_snapshot import AgentNode, TurnSnapshot
from miniclaw.worktree.isolator import IsolatedWorkspace, WorkspaceIsolator
from miniclaw.worktree.isolated_tools import IsolatedToolSet
from miniclaw.worktree.manager import WorktreeManager, WorktreeStatus
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SubagentExecutionContext:
    """Subagent 执行上下文"""
    task: TaskNode
    config: SubagentConfig
    workspace: IsolatedWorkspace | None = None
    tool_set: IsolatedToolSet | None = None
    result: StructuredResult | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @property
    def duration_ms(self) -> int | None:
        if self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None


class SubagentExecutor:
    """Subagent 执行器

    管理 Subagent 的完整执行生命周期，包括 worktree 隔离。
    """

    def __init__(
        self,
        workspace_isolator: WorkspaceIsolator,
    ):
        self._isolator = workspace_isolator
        self._contexts: dict[str, SubagentExecutionContext] = {}

    async def prepare_for_execution(
        self,
        task: TaskNode,
        config: SubagentConfig,
        agent_id: str,
        session_id: str,
    ) -> SubagentExecutionContext:
        """准备执行环境

        Args:
            task: 任务节点
            config: Subagent 配置
            agent_id: Agent ID
            session_id: 会话 ID

        Returns:
            SubagentExecutionContext: 执行上下文
        """
        ctx = SubagentExecutionContext(
            task=task,
            config=config,
        )

        # 如果需要 worktree
        if config.requires_worktree:
            workspace = await self._isolator.prepare_workspace(
                task=task,
                agent_id=agent_id,
                session_id=session_id,
                requires_worktree=True,
            )
            ctx.workspace = workspace

            # 创建隔离工具集
            if workspace.worktree_path:
                ctx.tool_set = IsolatedToolSet(workspace.worktree_path)
                logger.info(
                    "隔离工具集已创建: task=%s, cwd=%s",
                    task.id, workspace.worktree_path
                )

        self._contexts[task.id] = ctx
        return ctx

    async def get_tool_set(self, task_id: str) -> IsolatedToolSet | None:
        """获取任务的隔离工具集"""
        ctx = self._contexts.get(task_id)
        return ctx.tool_set if ctx else None

    async def get_workspace_path(self, task_id: str) -> Path | None:
        """获取任务的 worktree 路径"""
        ctx = self._contexts.get(task_id)
        if ctx and ctx.workspace:
            return ctx.workspace.worktree_path
        return None

    async def collect_result(
        self,
        task_id: str,
        result_data: dict[str, Any],
    ) -> StructuredResult:
        """收集执行结果

        Args:
            task_id: 任务 ID
            result_data: 结果数据（来自 submit_task_result 工具）

        Returns:
            StructuredResult: 验证后的结构化结果
        """
        ctx = self._contexts.get(task_id)
        if ctx is None:
            raise ValueError(f"任务上下文不存在: {task_id}")

        # 验证结果
        result = validate_result(result_data)
        ctx.result = result
        ctx.completed_at = datetime.now(timezone.utc)

        logger.info(
            "结果已收集: task=%s, status=%s, files=%d",
            task_id, result.status.value, len(result.files_changed)
        )

        return result

    async def finalize_execution(
        self,
        task_id: str,
    ) -> dict[str, Any]:
        """完成执行

        流程:
          1. 检查是否有变更
          2. 提交变更（如果有）
          3. 合并到主干（如果成功）
          4. 清理 worktree

        Args:
            task_id: 任务 ID

        Returns:
            完成状态
        """
        ctx = self._contexts.get(task_id)
        if ctx is None:
            return {"finalized": False, "error": "context not found"}

        if ctx.result is None:
            return {"finalized": False, "error": "no result collected"}

        if ctx.workspace is None:
            # 没有使用 worktree，直接返回
            return {
                "finalized": True,
                "task_id": task_id,
                "status": ctx.result.status.value,
                "used_worktree": False,
            }

        # 使用 isolator 完成工作空间
        finalize_result = await self._isolator.finalize_workspace(
            workspace_id=task_id,
            result=ctx.result,
        )

        finalize_result["finalized"] = True
        finalize_result["task_id"] = task_id
        finalize_result["used_worktree"] = True

        # 清理上下文
        del self._contexts[task_id]

        return finalize_result

    async def cancel_execution(
        self,
        task_id: str,
        reason: str = "cancelled",
    ) -> bool:
        """取消执行"""
        ctx = self._contexts.get(task_id)
        if ctx is None:
            return False

        # 创建取消结果
        ctx.result = StructuredResult(
            status=TaskStatus.FAILED,
            summary=f"任务已取消: {reason}",
        )
        ctx.completed_at = datetime.now(timezone.utc)

        # 清理 worktree
        if ctx.workspace:
            await self._isolator.cleanup_workspace(task_id, force=True)

        del self._contexts[task_id]

        logger.info("执行已取消: task=%s, reason=%s", task_id, reason)
        return True

    async def get_status(self, task_id: str) -> dict[str, Any]:
        """获取执行状态"""
        ctx = self._contexts.get(task_id)
        if ctx is None:
            return {"exists": False, "task_id": task_id}

        return {
            "exists": True,
            "task_id": task_id,
            "has_workspace": ctx.workspace is not None,
            "workspace_path": str(ctx.workspace.worktree_path) if ctx.workspace and ctx.workspace.worktree_path else None,
            "has_result": ctx.result is not None,
            "result_status": ctx.result.status.value if ctx.result else None,
            "duration_ms": ctx.duration_ms,
        }

    async def list_active_executions(self) -> list[str]:
        """列出所有活跃的执行"""
        return list(self._contexts.keys())

    async def cleanup_all(self) -> int:
        """清理所有执行"""
        count = 0
        for task_id in list(self._contexts.keys()):
            if await self.cancel_execution(task_id, "cleanup"):
                count += 1
        return count


async def create_subagent_executor(
    repo_root: str | Path,
    worktree_base: str | Path | None = None,
    auto_merge: bool = True,
    auto_cleanup: bool = True,
) -> SubagentExecutor:
    """创建 Subagent 执行器（工厂函数）

    Args:
        repo_root: Git 仓库根目录
        worktree_base: worktree 存放目录
        auto_merge: 是否自动合并
        auto_cleanup: 是否自动清理

    Returns:
        配置好的 SubagentExecutor
    """
    from miniclaw.worktree.isolator import create_workspace_isolator

    isolator = create_workspace_isolator(
        repo_root=repo_root,
        worktree_base=worktree_base,
        auto_merge=auto_merge,
        auto_cleanup=auto_cleanup,
    )

    # 初始化 worktree manager
    await isolator._worktree_mgr.initialize()

    return SubagentExecutor(isolator)


# 导出
__all__ = [
    "SubagentExecutionContext",
    "SubagentExecutor",
    "create_subagent_executor",
]