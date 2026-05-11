
from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from taskweave.types.task_graph import FailureCategory
from taskweave.utils.logging import get_logger

logger = get_logger(__name__)
class WorkspaceState(str, Enum):
    """Worktree 生命周期的显式状态。

    每个状态都对应一种**外部可观察的事实**：
      - NOT_CREATED：从未 git worktree add 过
      - CREATING：git worktree add 进行中（极短窗口，几乎不会停留）
      - ACTIVE：worktree 在 disk 上、_worktrees 里有、git 认得；可写
      - COMMITTING：执行 git commit，进行中
      - COMMITTED：subagent 分支上有 commit，但没合到主干
      - MERGING：执行 git merge，进行中
      - MERGED：subagent 分支已合入用户工作分支（**唯一的成功**）
      - FAILED：某次操作失败，状态确定终止
      - CORRUPTED：健康检查发现内存 vs 磁盘 vs git 不一致
      - CLEANED：worktree 已被移除（无论成功/失败）
    """

    NOT_CREATED = "not_created"
    CREATING = "creating"
    ACTIVE = "active"
    COMMITTING = "committing"
    COMMITTED = "committed"
    MERGING = "merging"
    MERGED = "merged"
    FAILED = "failed"
    CORRUPTED = "corrupted"
    CLEANED = "cleaned"


# 合法的状态转换（origin → {target1, target2, ...}）
# 任何不在这张表里的转换都会被 transition_to 拒绝。
_LEGAL_TRANSITIONS: dict[WorkspaceState, frozenset[WorkspaceState]] = {
    WorkspaceState.NOT_CREATED: frozenset({
        WorkspaceState.CREATING, WorkspaceState.FAILED,
    }),
    WorkspaceState.CREATING: frozenset({
        WorkspaceState.ACTIVE, WorkspaceState.FAILED,
    }),
    WorkspaceState.ACTIVE: frozenset({
        WorkspaceState.COMMITTING, WorkspaceState.FAILED,
        WorkspaceState.CORRUPTED, WorkspaceState.CLEANED,
    }),
    WorkspaceState.COMMITTING: frozenset({
        WorkspaceState.COMMITTED, WorkspaceState.FAILED,
        WorkspaceState.CORRUPTED,
    }),
    WorkspaceState.COMMITTED: frozenset({
        WorkspaceState.MERGING, WorkspaceState.FAILED,
        WorkspaceState.CORRUPTED, WorkspaceState.CLEANED,
    }),
    WorkspaceState.MERGING: frozenset({
        WorkspaceState.MERGED, WorkspaceState.FAILED,
        WorkspaceState.CORRUPTED,
    }),
    WorkspaceState.MERGED: frozenset({
        WorkspaceState.CLEANED,
    }),
    # 终态
    WorkspaceState.FAILED: frozenset({WorkspaceState.CLEANED}),
    WorkspaceState.CORRUPTED: frozenset({WorkspaceState.CLEANED}),
    WorkspaceState.CLEANED: frozenset(),  # 不可再转换
}
def is_legal_transition(src: WorkspaceState, dst: WorkspaceState) -> bool:
    return dst in _LEGAL_TRANSITIONS.get(src, frozenset())
@dataclass(frozen=True)
class WorkspaceHealth:
    """Worktree 在某一时刻的客观健康快照。

    四个 bool 都为 True 才视为健康（is_coherent）。任一为 False 都
    意味着内存 / 磁盘 / git 三者间存在漂移，需要立刻处理。

    这是把 git 当作 SoT（Single Source of Truth）的物化形式。
    """
    in_memory: bool
    """WorktreeManager._worktrees 字典里有这条记录"""
    on_disk: bool
    """worktree_path 指向的目录在文件系统上存在"""
    in_git_worktree: bool
    """`git worktree list` 能列出这条 worktree"""
    branch_exists: bool
    """对应的 subagent 分支真实存在"""
    sampled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    @property
    def is_coherent(self) -> bool:
        return all((
            self.in_memory, self.on_disk, self.in_git_worktree, self.branch_exists,
        ))
    @property
    def diagnosis(self) -> str:
        """给出"哪几项不一致"的人读说明，用于错误消息和审计。"""
        if self.is_coherent:
            return "healthy"
        broken = []
        if not self.in_memory:
            broken.append("内存字典缺失")
        if not self.on_disk:
            broken.append("磁盘目录不存在")
        if not self.in_git_worktree:
            broken.append("git worktree list 不认")
        if not self.branch_exists:
            broken.append("分支已不存在")
        return f"状态不一致：{' / '.join(broken)}"
async def compute_health(
    repo_root: Path,
    worktree_path: Path,
    branch_name: str,
    in_memory: bool,
) -> WorkspaceHealth:
    """实时回查 worktree 的健康状态。
    成本：3 次 git 子进程 + 1 次 stat。在写操作前调用一次（cheap）。
    """
    on_disk = worktree_path.exists() and worktree_path.is_dir()

    # git worktree list 检查（用 `--porcelain` 解析）
    in_git_worktree = False
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "worktree", "list", "--porcelain",
            cwd=str(repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        wt_text = stdout.decode("utf-8", errors="replace")
        wt_resolved = str(worktree_path.resolve()).lower()
        for line in wt_text.splitlines():
            if line.startswith("worktree "):
                listed = line[len("worktree "):].strip()
                if Path(listed).resolve().__str__().lower() == wt_resolved:
                    in_git_worktree = True
                    break
    except Exception as e:
        logger.debug("compute_health: git worktree list 失败: %s", e)
    # 分支存在检查
    branch_exists = False
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "--verify", "--quiet", branch_name,
            cwd=str(repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        branch_exists = (proc.returncode == 0)
    except Exception as e:
        logger.debug("compute_health: rev-parse 失败: %s", e)
    return WorkspaceHealth(
        in_memory=in_memory,
        on_disk=on_disk,
        in_git_worktree=in_git_worktree,
        branch_exists=branch_exists,
    )
class WorktreeError(Exception):
    """Worktree 操作失败的基类。

    所有子类必须覆盖 category() 给出 FailureCategory，调度层据此决策。
    """

    @property
    def category(self) -> FailureCategory:
        return FailureCategory.UNKNOWN

    @property
    def is_retriable(self) -> bool:
        """该错误是否可以原样重试。默认否（保守）"""
        return False


class WorktreeCreateFailed(WorktreeError):
    """git worktree add 失败：分支冲突 / 父分支不存在 / 磁盘满 / 权限"""

    @property
    def category(self) -> FailureCategory:
        return FailureCategory.WORKTREE_CREATION
class WorkspaceCorrupted(WorktreeError):
    """健康检查发现状态不一致（内存 vs 磁盘 vs git 漂移）。

    这种错误意味着系统对该 worktree 的认知和真实情况已分歧，
    继续操作会产生未定义结果。**必须**立刻把 task 标为失败、
    清理痕迹，让 Master 知道这是基础设施级故障。
    """

    def __init__(self, msg: str, health: WorkspaceHealth | None = None):
        super().__init__(msg)
        self.health = health
    @property
    def category(self) -> FailureCategory:
        return FailureCategory.WORKSPACE_CORRUPTED
class WorktreeCommitFailed(WorktreeError):
    """git commit 失败：没变更 / hook 拒绝 / 索引锁住"""
    @property
    def category(self) -> FailureCategory:
        return FailureCategory.COMMIT_FAILED
class WorktreeMergeConflict(WorktreeError):
    """git merge 冲突。带冲突文件清单，给 master 调用 git_resolve_conflict 用。"""

    def __init__(self, msg: str, conflict_files: list[str] | None = None):
        super().__init__(msg)
        self.conflict_files = conflict_files or []

    @property
    def category(self) -> FailureCategory:
        return FailureCategory.WORKTREE_CREATION
class WorktreeNotFound(WorktreeError):
    """对一个不在 _worktrees 字典里的 id 调用 commit/merge —— 这种调用本身
    是程序逻辑错误（之前丢了对状态的追踪），通常源于状态机绕过。"""

    @property
    def category(self) -> FailureCategory:
        return FailureCategory.WORKTREE_CREATION
class SandboxViolation(WorktreeError):
    """文件工具试图在不存在的 worktree 根上创建新结构。

    sandbox 安全契约：worktree 根目录的存在性由资源管理层维护，
    不应该被 IsolatedWriteFileTool 自动 mkdir 补建。
    """

    @property
    def category(self) -> FailureCategory:
        return FailureCategory.SANDBOX_VIOLATION

@dataclass
class WorkspaceLifecycle:
    """单个 worktree 的状态机持有者。

    每个 WorktreeInfo 关联一个 lifecycle 实例，所有状态转换都通过
    transition_to 走，非法转换会抛 InvalidTransitionError 让调用方
    知道是逻辑错（而不是悄悄忽略）。

    历史记录用于审计 —— 把"什么时候从什么状态变到什么状态"留作
    EventBus 事件流的素材。
    """

    workspace_id: str
    state: WorkspaceState = WorkspaceState.NOT_CREATED
    history: list[tuple[WorkspaceState, WorkspaceState, datetime, str]] = field(
        default_factory=list,
    )
    def transition_to(
        self,
        new_state: WorkspaceState,
        reason: str = "",
    ) -> None:
        """驱动状态机前进。非法转换抛 InvalidTransitionError。"""
        if not is_legal_transition(self.state, new_state):
            raise InvalidTransitionError(
                f"workspace={self.workspace_id} 非法状态转换: "
                f"{self.state.value} → {new_state.value}（reason={reason}）"
            )
        prev = self.state
        self.state = new_state
        self.history.append((
            prev, new_state, datetime.now(timezone.utc), reason,
        ))
        logger.debug(
            "workspace %s 状态转换: %s → %s (%s)",
            self.workspace_id, prev.value, new_state.value, reason or "no-reason",
        )
    def is_terminal(self) -> bool:
        return self.state in (
            WorkspaceState.MERGED, WorkspaceState.FAILED,
            WorkspaceState.CORRUPTED, WorkspaceState.CLEANED,
        )
    def can_perform_writes(self) -> bool:
        """是否处于"agent 可以做文件写入操作"的状态。"""
        return self.state == WorkspaceState.ACTIVE
class InvalidTransitionError(WorktreeError):
    """状态机非法转换 —— 表明调用方逻辑错（绕过了正常生命周期）。"""

    @property
    def category(self) -> FailureCategory:
        return FailureCategory.UNKNOWN





__all__ = [
    "WorkspaceState",
    "WorkspaceHealth",
    "WorkspaceLifecycle",
    "is_legal_transition",
    "compute_health",
    # errors
    "WorktreeError",
    "WorktreeCreateFailed",
    "WorkspaceCorrupted",
    "WorktreeCommitFailed",
    "WorktreeMergeConflict",
    "WorktreeNotFound",
    "SandboxViolation",
    "InvalidTransitionError",
]
