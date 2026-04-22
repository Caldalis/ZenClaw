"""
Workspace Isolator — 将 Worktree 与 Subagent 执行环境绑定

核心职责:
  1. 在 Spawn 时自动创建 worktree
  2. 将 worktree 路径绑定到 Agent 执行上下文
  3. 管理文件操作的目录限制
  4. 任务完成后自动提交、合并、清理

工作流程:
  1. Subagent 需要 worktree → 调用 prepare_workspace()
  2. 创建 worktree → 返回 worktree_path
  3. Agent 执行器使用 worktree_path 作为 cwd
  4. 任务完成 → 调用 finalize_workspace()
  5. 提交变更 → 合并到主干 → 清理 worktree
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from miniclaw.types.structured_result import StructuredResult, TaskStatus
from miniclaw.types.task_graph import TaskNode
from miniclaw.worktree.manager import (
    WorktreeInfo,
    WorktreeManager,
    WorktreeStatus,
)
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IsolatedWorkspace:
    """隔离的工作空间"""
    workspace_id: str
    """工作空间 ID（通常为 task_id）"""

    worktree_info: WorktreeInfo | None = None
    """Worktree 信息（如果使用 Git 隔离）"""

    worktree_path: Path | None = None
    """Worktree 目录路径"""

    agent_id: str | None = None
    """绑定的 Agent ID"""

    task_id: str | None = None
    """关联的任务 ID"""

    session_id: str | None = None
    """会话 ID"""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # 状态标记
    is_active: bool = False
    has_changes: bool = False
    is_committed: bool = False
    is_merged: bool = False

    # 结果
    result: StructuredResult | None = None

    @property
    def cwd(self) -> Path | None:
        """获取当前工作目录（用于 Agent 执行）"""
        return self.worktree_path


class WorkspaceIsolator:
    """工作空间隔离器

    负责在 Subagent Spawn 时自动创建隔离环境。
    """

    def __init__(
        self,
        worktree_manager: WorktreeManager,
        auto_merge: bool = True,
        auto_cleanup: bool = True,
    ):
        """
        Args:
            worktree_manager: Worktree 管理器
            auto_merge: 任务成功后是否自动合并到主干
            auto_cleanup: 是否自动清理 worktree
        """
        self._worktree_mgr = worktree_manager
        self._auto_merge = auto_merge
        self._auto_cleanup = auto_cleanup

        # 工作空间缓存
        self._workspaces: dict[str, IsolatedWorkspace] = {}

        # 结果回调
        self._result_callbacks: list[Callable[[IsolatedWorkspace], None]] = []

    async def prepare_workspace(
        self,
        task: TaskNode,
        agent_id: str,
        session_id: str,
        requires_worktree: bool = True,
        parent_branch: str | None = None,
    ) -> IsolatedWorkspace:
        """准备工作空间

        当 Subagent 需要 worktree 时自动创建隔离环境。

        Args:
            task: 任务节点
            agent_id: Agent ID
            session_id: 会话 ID
            requires_worktree: 是否需要 Git Worktree 隔离
            parent_branch: worktree 的基础分支。None 则自动检测默认分支。
                当任务有依赖时，应传入已完成依赖任务的分支名，
                使依赖产物可见。

        Returns:
            IsolatedWorkspace: 准备好的工作空间
        """
        workspace_id = task.id

        # 检查是否已存在
        if workspace_id in self._workspaces:
            return self._workspaces[workspace_id]

        workspace = IsolatedWorkspace(
            workspace_id=workspace_id,
            task_id=task.id,
            agent_id=agent_id,
            session_id=session_id,
        )

        # 如果需要 Git Worktree 隔离
        if requires_worktree:
            try:
                worktree_info = await self._worktree_mgr.create_worktree(
                    worktree_id=workspace_id,
                    parent_branch=parent_branch,
                    branch_prefix="subagent",
                )

                workspace.worktree_info = worktree_info
                workspace.worktree_path = worktree_info.path
                workspace.is_active = True

                logger.info(
                    "工作空间已准备: id=%s, path=%s",
                    workspace_id, workspace.worktree_path
                )

            except Exception as e:
                logger.warning("创建 worktree 失败，降级为非隔离模式: %s", e)
                workspace.worktree_path = self._worktree_mgr.repo_root
                workspace.is_active = True



        else:
            # 不需要 worktree，使用仓库根目录
            workspace.worktree_path = self._worktree_mgr.repo_root
            workspace.is_active = True

        self._workspaces[workspace_id] = workspace
        return workspace

    async def get_workspace(self, workspace_id: str) -> IsolatedWorkspace | None:
        """获取工作空间"""
        return self._workspaces.get(workspace_id)

    async def finalize_workspace(
        self,
        workspace_id: str,
        result: StructuredResult,
        merge_and_cleanup: bool = True,
    ) -> dict[str, Any]:
        """完成工作空间

        流程:
          1. 检查是否有变更
          2. 提交变更（如果有）
          3. 合并到主干（如果成功且配置了 auto_merge）— 仅当 merge_and_cleanup=True
          4. 清理 worktree（如果配置了 auto_cleanup）— 仅当 merge_and_cleanup=True

        当任务有下游依赖时，应设 merge_and_cleanup=False，
        仅提交变更保留分支，使依赖任务可基于此分支创建 worktree。
        DAG 全部完成后统一合并清理。

        Args:
            workspace_id: 工作空间 ID
            result: 任务结果
            merge_and_cleanup: 是否执行合并和清理（默认 True）

        Returns:
            完成状态
        """
        workspace = self._workspaces.get(workspace_id)
        if workspace is None:
            return {"finalized": False, "error": "workspace not found"}

        workspace.result = result
        finalize_result = {
            "workspace_id": workspace_id,
            "task_status": result.status.value,
        }

        # 如果没有使用 worktree，直接返回
        if workspace.worktree_info is None:
            workspace.is_active = False
            return finalize_result

        try:
            # 检查是否有变更
            status = await self._worktree_mgr.get_status(workspace_id)
            changed_files = status.get("changed_files", [])
            workspace.has_changes = len(changed_files) > 0

            finalize_result["changed_files"] = changed_files

            # 如果有变更，提交
            if workspace.has_changes:
                commit_result = await self._worktree_mgr.commit_changes(
                    worktree_id=workspace_id,
                    message=f"Subagent {workspace_id}: {result.summary[:50]}",
                )
                workspace.is_committed = commit_result.get("committed", False)
                finalize_result["commit"] = commit_result

            # 如果任务成功且配置了自动合并 — 仅在 merge_and_cleanup 模式下执行
            if merge_and_cleanup and result.is_success() and self._auto_merge and workspace.is_committed:
                merge_result = await self._worktree_mgr.merge_to_parent(
                    worktree_id=workspace_id,
                    delete_branch=self._auto_cleanup,
                )
                workspace.is_merged = merge_result.get("merged", False)
                finalize_result["merge"] = merge_result

                # 如果合并有冲突
                if merge_result.get("has_conflicts"):
                    finalize_result["conflicts"] = merge_result.get("conflict_files", [])

            # 清理 worktree — 仅在 merge_and_cleanup 模式下执行
            if merge_and_cleanup and self._auto_cleanup:
                await self._worktree_mgr.remove_worktree(
                    worktree_id=workspace_id,
                    force=True,
                )

            workspace.is_active = False

            # 触发回调
            for callback in self._result_callbacks:
                try:
                    callback(workspace)
                except Exception as e:
                    logger.warning("回调执行失败: %s", e)

            logger.info(
                "工作空间已完成: id=%s, status=%s, committed=%s, merged=%s",
                workspace_id, result.status.value, workspace.is_committed, workspace.is_merged
            )

            return finalize_result

        except Exception as e:
            logger.error("完成工作空间失败: %s", e)
            finalize_result["error"] = str(e)
            return finalize_result

    async def cleanup_workspace(
        self,
        workspace_id: str,
        force: bool = False,
    ) -> bool:
        """强制清理工作空间"""
        workspace = self._workspaces.get(workspace_id)
        if workspace is None:
            return False

        if workspace.worktree_info:
            await self._worktree_mgr.remove_worktree(
                worktree_id=workspace_id,
                force=force,
            )

        workspace.is_active = False
        del self._workspaces[workspace_id]

        logger.info("工作空间已清理: %s", workspace_id)
        return True

    async def cleanup_all(self) -> int:
        """清理所有工作空间"""
        count = 0
        for workspace_id in list(self._workspaces.keys()):
            if await self.cleanup_workspace(workspace_id, force=True):
                count += 1
        return count

    def register_result_callback(
        self,
        callback: Callable[[IsolatedWorkspace], None],
    ) -> None:
        """注册结果回调"""
        self._result_callbacks.append(callback)

    def get_active_workspaces(self) -> list[IsolatedWorkspace]:
        """获取所有活跃的工作空间"""
        return [ws for ws in self._workspaces.values() if ws.is_active]

    async def get_status(self) -> dict[str, Any]:
        """获取隔离器状态"""
        active = self.get_active_workspaces()

        return {
            "total_workspaces": len(self._workspaces),
            "active_workspaces": len(active),
            "auto_merge": self._auto_merge,
            "auto_cleanup": self._auto_cleanup,
            "workspaces": [
                {
                    "id": ws.workspace_id,
                    "task_id": ws.task_id,
                    "path": str(ws.worktree_path) if ws.worktree_path else None,
                    "has_changes": ws.has_changes,
                    "is_committed": ws.is_committed,
                }
                for ws in active
            ],
        }


def create_workspace_isolator(
    repo_root: str | Path,
    worktree_base: str | Path | None = None,
    auto_merge: bool = True,
    auto_cleanup: bool = True,
) -> WorkspaceIsolator:
    """创建工作空间隔离器（工厂函数）"""
    manager = WorktreeManager(repo_root, worktree_base)
    return WorkspaceIsolator(manager, auto_merge, auto_cleanup)


# 导出
__all__ = [
    "IsolatedWorkspace",
    "WorkspaceIsolator",
    "create_workspace_isolator",
]