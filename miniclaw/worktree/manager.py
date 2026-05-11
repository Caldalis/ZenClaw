"""
Git Worktree 管理器 — 实现物理隔离的并行沙盒

核心职责:
  1. 创建 worktree（`git worktree add`）
  2. 管理 worktree 生命周期
  3. 清理 worktree
  4. 合并分支到主干
  5. 处理冲突

设计原则:
  - 每个 Subagent 拥有独立的 worktree 目录
  - 文件操作限定在 worktree 内
  - 任务完成后自动提交并合并
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from miniclaw.utils.logging import get_logger
from miniclaw.worktree.lifecycle import (
    InvalidTransitionError,
    WorkspaceCorrupted,
    WorkspaceHealth,
    WorkspaceLifecycle,
    WorkspaceState,
    WorktreeCommitFailed,
    WorktreeCreateFailed,
    WorktreeError,
    WorktreeMergeConflict,
    WorktreeNotFound,
    compute_health,
)

logger = get_logger(__name__)


class WorktreeStatus(str, Enum):
    """Worktree 状态"""
    CREATING = "creating"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    MERGING = "merging"
    CLEANED = "cleaned"


@dataclass
class WorktreeInfo:
    """Worktree 信息

    新架构：每个 WorktreeInfo 关联一个 WorkspaceLifecycle 状态机实例。
    `status` 字段保留作为兼容旧调用的快照，新代码应当查 `lifecycle.state`。
    """
    worktree_id: str
    """Worktree ID（通常与 task_id 一致）"""

    branch_name: str
    """分支名称"""

    path: Path
    """Worktree 目录路径"""

    parent_branch: str = "main"
    """父分支（基于哪个分支创建）"""

    status: WorktreeStatus = WorktreeStatus.CREATING
    """当前状态（兼容字段；权威状态在 lifecycle.state）"""

    lifecycle: WorkspaceLifecycle | None = None
    """显式状态机持有者。create_worktree 创建时会注入。"""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """创建时间"""

    completed_at: datetime | None = None
    """完成时间"""

    agent_id: str | None = None
    """关联的 Agent ID"""

    task_id: str | None = None
    """关联的任务 ID"""

    commit_hash: str | None = None
    """提交哈希"""

    error_message: str | None = None
    """错误信息"""


class WorktreeManager:
    """Git Worktree 管理器

    管理 worktree 的完整生命周期：
      1. 创建：git worktree add -b <branch> <path> <parent>
      2. 列出：git worktree list
      3. 移除：git worktree remove <path>
      4. 清理：git worktree prune
    """

    def __init__(
        self,
        repo_root: str | Path,
        worktree_base: str | Path | None = None,
    ):
        """
        Args:
            repo_root: Git 仓库根目录
            worktree_base: worktree 存放目录（默认为 .agents/worktrees）
        """
        self.repo_root = Path(repo_root).resolve()

        if worktree_base:
            self.worktree_base = Path(worktree_base).resolve()
        else:
            self.worktree_base = self.repo_root / ".agents" / "worktrees"

        self._worktrees: dict[str, WorktreeInfo] = {}

    async def initialize(self) -> None:
        """初始化 Worktree 管理器"""
        # 确保 worktree 目录存在
        self.worktree_base.mkdir(parents=True, exist_ok=True)

        # 检查是否在 Git 仓库中
        if not await self._is_git_repo():
            raise RuntimeError(f"不是 Git 仓库: {self.repo_root}")

        # 清理孤立的 worktree 记录
        await self._prune_stale_worktrees()

        # 回收上次进程崩溃留下的 subagent worktree（.agents/worktrees/subagent-*）。
        # 这些目录在 `git worktree list` 里仍存在但 self._worktrees 内存里没有，
        # 新任务分配同名 id 会踩到 "分支已存在，复用分支" 的旧残骸。
        recovered = await self._recover_stale_subagent_worktrees()
        if recovered:
            logger.info("启动时清理 %d 个残留 subagent worktree", recovered)

        logger.info("Worktree 管理器已初始化: repo=%s, base=%s", self.repo_root, self.worktree_base)

    async def _recover_stale_subagent_worktrees(self) -> int:
        """启动时扫描 git worktree list，强制删除 subagent-* 分支留下的残骸。

        策略：只清理我们自己在 `worktree_base` 下创建的目录（以 `subagent-` 开头的分支），
        避免误删用户手工创建的 worktree。

        Returns:
            实际被清理的 worktree 数
        """
        result = await self._run_git_command(
            ["worktree", "list", "--porcelain"],
            cwd=self.repo_root,
        )
        if result["exit_code"] != 0:
            return 0

        # 按块解析：每个 worktree 由一个 "worktree <path>" 起头，紧跟 HEAD / branch 等行
        blocks: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for line in result["stdout"].splitlines():
            if not line.strip():
                if current:
                    blocks.append(current)
                    current = {}
                continue
            if " " in line:
                key, _, value = line.partition(" ")
                current[key] = value
        if current:
            blocks.append(current)

        cleaned = 0
        for block in blocks:
            wt_path = block.get("worktree")
            branch = block.get("branch", "")
            if not wt_path:
                continue
            wt = Path(wt_path)
            # 仅清理本 base 目录下且分支名以 refs/heads/subagent- 开头的记录
            try:
                rel = wt.resolve().relative_to(self.worktree_base.resolve())
            except ValueError:
                continue
            if not branch.startswith("refs/heads/subagent-") and not str(rel).startswith("subagent-"):
                continue

            # 强制移除 worktree
            remove = await self._run_git_command(
                ["worktree", "remove", "--force", str(wt)],
                cwd=self.repo_root,
            )
            if remove["exit_code"] != 0 and wt.exists():
                try:
                    await self._remove_directory(wt)
                except Exception as e:
                    logger.warning("清理残留 worktree 目录失败: %s, error=%s", wt, e)
                    continue
            logger.info("已清理残留 worktree: %s (branch=%s)", wt, branch)
            cleaned += 1

        # 最后执行一次 prune 让 git 把索引里的残留清理干净
        await self._run_git_command(["worktree", "prune"], cwd=self.repo_root)

        # 再清理subagent-* 分支已经没有对应 worktree（worktree 在上次进程异常退出后被清掉、但分支还留着）。如果不清理，下一次同 id
        # 任务 create_worktree 时会撞 "branch already exists" 警告然后重建。
        cleaned += await self._prune_orphan_subagent_branches()
        return cleaned

    async def safe_prune_subagent_branches(
        self,
        branch_names: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """安全清理 subagent-* 分支：仅删已合并到 HEAD 的，保留未合并的。

        架构原则：**绝不强删未合并的分支**。git branch -d (小写) 自带
        "已合并才删"的检查；分支上若还有未到达 HEAD 的 commit，git 会
        拒绝删除——这是用户成果（特别是 DAG 部分失败、叶子未 merge 时
        中间节点的 commit）的最后救生圈，强删等同于销毁数据。

        本方法是 DAG finalize 与启动恢复两条路径的共同实现：
          - 启动恢复（`_prune_orphan_subagent_branches`）：扫所有
            refs/heads/subagent-* 残留，传 branch_names=None。
          - DAG finalize：传入本次 DAG 产生的分支名列表，仅清理这些。
            包括叶子（已显式 merge）+ 中间节点（commit 通过 git 父子链
            随叶子 merge 进了用户分支，因此对 -d 而言同样"已合并"）。

        策略：
          1. 跳过仍被 worktree 引用的分支（git 也会拒绝，但提前过滤更干净）。
          2. 对每个候选分支跑 `git branch -d`；失败 → 未合并 → 保留。

        Args:
            branch_names: 限定清理范围。
                None → 扫所有 refs/heads/subagent-* （启动恢复）。
                空列表 → 不做任何事。
                非空列表 → 只对列表内的分支尝试安全删除。

        Returns:
            {"deleted": [...], "kept_unmerged": [...]}
            deleted: 实际删除的分支名
            kept_unmerged: 因未合并而保留的分支名（含用户成果救生圈）
        """
        if branch_names is None:
            branches_result = await self._run_git_command(
                ["for-each-ref", "--format=%(refname:short)", "refs/heads/subagent-"],
                cwd=self.repo_root,
            )
            if branches_result["exit_code"] != 0:
                return {"deleted": [], "kept_unmerged": []}
            candidates = [
                line.strip() for line in branches_result["stdout"].splitlines()
                if line.strip()
            ]
        else:
            candidates = [b for b in branch_names if b]

        if not candidates:
            return {"deleted": [], "kept_unmerged": []}

        # 收集仍在被 worktree 引用的分支，避免误删
        wt_list = await self._run_git_command(
            ["worktree", "list", "--porcelain"],
            cwd=self.repo_root,
        )
        live_branches: set[str] = set()
        if wt_list["exit_code"] == 0:
            for line in wt_list["stdout"].splitlines():
                if line.startswith("branch "):
                    ref = line[len("branch "):].strip()
                    if ref.startswith("refs/heads/"):
                        live_branches.add(ref[len("refs/heads/"):])

        deleted: list[str] = []
        kept_unmerged: list[str] = []
        for name in candidates:
            if name in live_branches:
                continue
            # -d (小写)：仅当分支已合并到 HEAD 时才删；未合并 → 退出码非 0，保留。
            res = await self._run_git_command(
                ["branch", "-d", name],
                cwd=self.repo_root,
            )
            if res["exit_code"] == 0:
                logger.info("已安全删除 subagent 分支: %s", name)
                deleted.append(name)
            else:
                # 未合并：保留，由用户决定 git checkout 找回还是手动 -D
                kept_unmerged.append(name)

        if kept_unmerged:
            logger.warning(
                "保留 %d 个未合并的 subagent-* 分支（可能含未恢复的产物）: %s",
                len(kept_unmerged),
                ", ".join(kept_unmerged[:5]) + ("..." if len(kept_unmerged) > 5 else ""),
            )

        return {"deleted": deleted, "kept_unmerged": kept_unmerged}

    async def _prune_orphan_subagent_branches(self) -> int:
        """启动期清理：扫所有 refs/heads/subagent-* 残留分支，安全删除已合并的。

        薄壳：委托给 `safe_prune_subagent_branches(branch_names=None)`，
        与 DAG finalize 路径共用同一套"-d 安全语义"。
        """
        result = await self.safe_prune_subagent_branches(branch_names=None)
        return len(result["deleted"])

    async def create_worktree(
        self,
        worktree_id: str,
        parent_branch: str | None = None,
        branch_prefix: str = "subagent",
    ) -> WorktreeInfo:
        """创建新的 Worktree

        Args:
            worktree_id: Worktree ID（通常为 task_id）
            parent_branch: 父分支名称
            branch_prefix: 分支名前缀

        Returns:
            WorktreeInfo: 创建的 worktree 信息
        """
        # 构建分支名和路径
        branch_name = f"{branch_prefix}-{worktree_id}"
        worktree_path = self.worktree_base / branch_name

        # 检查是否已存在
        if worktree_path.exists():
            logger.warning("Worktree 路径已存在，将清理: %s", worktree_path)
            await self._remove_directory(worktree_path)

        if parent_branch is None:
            parent_branch = await self._detect_default_branch()

        # 创建 worktree —— 同时初始化显式状态机
        lifecycle = WorkspaceLifecycle(workspace_id=worktree_id)
        lifecycle.transition_to(
            WorkspaceState.CREATING,
            reason=f"git worktree add {branch_name}",
        )

        info = WorktreeInfo(
            worktree_id=worktree_id,
            branch_name=branch_name,
            path=worktree_path,
            parent_branch=parent_branch,
            status=WorktreeStatus.CREATING,
            lifecycle=lifecycle,
        )

        try:
            # 执行 git worktree add
            result = await self._run_git_command(
                [
                    "worktree", "add",
                    "-b", branch_name,
                    str(worktree_path),
                    parent_branch,
                ],
                cwd=self.repo_root,
            )

            if result["exit_code"] != 0 and "already exists" in result["stderr"]:
                # 分支残留 — 来自上次崩溃或同名 task_id 的旧运行。
                # 安全策略（不丢数据）：
                #   1. 先 prune 释放幽灵 worktree 引用
                #   2. 尝试 git branch -d（仅当分支已合并到 HEAD 时成功）
                #   3. -d 失败 → 分支含未合并 commit，**不强删**，
                #      改用带时间戳的唯一分支名，把残留分支留给用户恢复
                logger.warning("分支已存在: %s — 尝试安全清理或绕开", branch_name)
                await self._run_git_command(["worktree", "prune"], cwd=self.repo_root)
                safe_delete = await self._run_git_command(
                    ["branch", "-d", branch_name],
                    cwd=self.repo_root,
                )
                if safe_delete["exit_code"] == 0:
                    logger.info("已删除已合并的同名残留分支: %s", branch_name)
                else:
                    # 未合并 → 改用唯一分支名，保留旧分支供恢复
                    import time
                    unique_suffix = str(int(time.time()))
                    new_branch = f"{branch_name}-{unique_suffix}"
                    new_path = self.worktree_base / new_branch
                    logger.warning(
                        "残留分支 %s 未合并（保留供恢复），本次任务改用唯一分支 %s",
                        branch_name, new_branch,
                    )
                    branch_name = new_branch
                    worktree_path = new_path
                    info.branch_name = branch_name
                    info.path = worktree_path

                # 再次尝试创建
                result = await self._run_git_command(
                    [
                        "worktree", "add",
                        "-b", branch_name,
                        str(worktree_path),
                        parent_branch,
                    ],
                    cwd=self.repo_root,
                )

            if result["exit_code"] != 0:
                # 类化错误：调度层据此判断 FailureCategory.WORKTREE_CREATION
                raise WorktreeCreateFailed(
                    f"创建 worktree 失败 (id={worktree_id}, branch={branch_name}): "
                    f"{result['stderr'].strip()[:200]}"
                )

            # 关键：注册到字典 + 状态机进入 ACTIVE 必须**同时**成功，
            # 不允许中间出错让两者漂移
            info.status = WorktreeStatus.ACTIVE
            lifecycle.transition_to(
                WorkspaceState.ACTIVE,
                reason="git worktree add succeeded",
            )
            self._worktrees[worktree_id] = info

            logger.info(
                "Worktree 已创建: id=%s, branch=%s, path=%s",
                worktree_id, branch_name, worktree_path
            )

            return info

        except WorktreeError:
            # 类化错误向上抛 —— 调度层会收到正确的 category
            info.status = WorktreeStatus.FAILED
            try:
                lifecycle.transition_to(WorkspaceState.FAILED, reason="create error")
            except InvalidTransitionError:
                pass  # 已经是终态
            raise

        except Exception as e:
            # 非预期异常：包装成 WorktreeCreateFailed 让上游统一处理
            info.status = WorktreeStatus.FAILED
            info.error_message = str(e)
            try:
                lifecycle.transition_to(WorkspaceState.FAILED, reason=f"unexpected: {e}")
            except InvalidTransitionError:
                pass
            logger.error("创建 worktree 异常: %s", e)
            raise WorktreeCreateFailed(
                f"创建 worktree 失败 (id={worktree_id}): 未预期异常 {type(e).__name__}: {e}"
            ) from e

    async def _detect_default_branch(self) -> str:
        # """检测仓库的默认分支名 (main, master, 等)"""
        #
        # result = await self._run_git_command(["symbolic-ref", "--short", "HEAD"], cwd=self.repo_root)
        # if result["exit_code"] == 0 and result["stdout"].strip():
        #     return result["stdout"].strip()
        # result = await self._run_git_command(["config", "init.defaultBranch"], cwd=self.repo_root)
        # if result["exit_code"] == 0 and result["stdout"].strip():
        #     return result["stdout"].strip()
        # return "main"

        """检测仓库的默认分支名，并验证分支确实存在"""
        candidates = []
        result = await self._run_git_command(["symbolic-ref", "--short", "HEAD"], cwd=self.repo_root)
        if result["exit_code"] == 0 and result["stdout"].strip():
            candidates.append(result["stdout"].strip())
        result = await self._run_git_command(["config", "init.defaultBranch"], cwd=self.repo_root)
        if result["exit_code"] == 0 and result["stdout"].strip():
            candidates.append(result["stdout"].strip())
        candidates.extend(["main", "master"])
        seen = set()
        candidates = [c for c in candidates if c not in seen and not seen.add(c)]
        for branch in candidates:
            result = await self._run_git_command(["rev-parse", "--verify", branch], cwd=self.repo_root)
            if result["exit_code"] == 0:
                return branch
            result = await self._run_git_command(["rev-parse", "--verify", f"origin/{branch}"], cwd=self.repo_root)
            if result["exit_code"] == 0:
                return f"origin/{branch}"
        raise RuntimeError(f"无法检测到有效的默认分支: {self.repo_root}（候选: {candidates}）")

    async def merge_branches_into_new_branch(
        self,
        source_branches: list[str],
        target_branch_name: str,
        base_branch: str | None = None,
    ) -> dict[str, Any]:
        """将多个源分支合并到一个新分支（用于多依赖任务创建 worktree）

        当任务 B 依赖 A 和 C 时，需要先将 A 和 C 的分支合并到
        一个中间分支，然后基于该中间分支创建 B 的 worktree。

        使用 git merge 的 OCTOPUS 策略一次性合并所有依赖分支，
        避免 checkout 操作（防止与并发任务冲突）。

        Args:
            source_branches: 要合并的源分支列表
            target_branch_name: 目标分支名称
            base_branch: 基础分支（第一个源分支），其他源分支合并到此基础上

        Returns:
            合并结果，包含 target_branch, merge_conflicts 等
        """
        if not source_branches:
            return {"merged": False, "error": "no source branches"}
        base_branch = base_branch or source_branches[0]
        other_branches = [b for b in source_branches if b != base_branch]
        try:
            create_result = await self._run_git_command(
                ["branch", target_branch_name, base_branch],
                cwd=self.repo_root,
            )
            if create_result["exit_code"] != 0:
                if "already exists" in create_result["stderr"]:
                    await self._run_git_command(
                        ["branch", "-D", target_branch_name],
                        cwd=self.repo_root,
                    )
                    create_result = await self._run_git_command(
                        ["branch", target_branch_name, base_branch],
                        cwd=self.repo_root,
                    )
                    if create_result["exit_code"] != 0:
                        raise RuntimeError(f"重建中间分支失败: {create_result['stderr']}")
            if not other_branches:
                logger.info(
                    "中间分支已创建（单依赖，无需合并）: %s <- %s",
                    target_branch_name, base_branch,
                )
                return {
                    "merged": True,
                    "target_branch": target_branch_name,
                    "conflict_files": [],
                    "base_branch": base_branch,
                    "merged_branches": source_branches,
                }

            # 使用 GIT_SEQUENCE_EDITOR 环境变量和临时工作目录执行合并
            # 不切换主仓库的 HEAD，而是使用 worktree 临时 checkout
            # 但更安全的方法是使用 git read-tree + git merge-tree（低级别）
            # 这里采用安全策略：创建临时 worktree 执行合并，然后删除临时 worktree
            # 创建临时目录用于合并操作（不使用主仓库的 worktree 机制）
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="zenclaw_merge_"))
            # 在临时目录创建 worktree checkout 中间分支
            worktree_add_result = await self._run_git_command(
                ["worktree", "add", str(temp_dir), target_branch_name],
                cwd=self.repo_root,
            )
            if worktree_add_result["exit_code"] != 0:
                # 清理临时目录
                await self._remove_directory(temp_dir)
                raise RuntimeError(f"创建临时 worktree 失败: {worktree_add_result['stderr']}")
            merge_conflicts = []
            merge_success = True
            # 逐个合并其他依赖分支（在临时 worktree 中操作）
            for source_branch in other_branches:
                merge_result = await self._run_git_command(
                    ["merge", source_branch, "-m", f"Merge dependency branch {source_branch} into {target_branch_name}"],
                    cwd=temp_dir,
                )
                if merge_result["exit_code"] != 0:
                    if "CONFLICT" in merge_result["stdout"] or "CONFLICT" in merge_result["stderr"]:
                        # 获取冲突文件列表
                        conflict_result = await self._run_git_command(
                            ["diff", "--name-only", "--diff-filter=U"],
                            cwd=temp_dir,
                        )
                        conflict_files = [
                            line.strip() for line in conflict_result["stdout"].strip().split("\n")
                            if line.strip()
                        ] if conflict_result["exit_code"] == 0 else []
                        merge_conflicts.extend(conflict_files)
                        merge_success = False
                        await self._run_git_command(
                            ["merge", "--abort"],
                            cwd=temp_dir,
                        )
                        logger.warning(
                            "依赖分支合并冲突: %s -> %s, 文件: %s, 已中止",
                            source_branch, target_branch_name, conflict_files,
                        )
                    else:
                        await self._run_git_command(
                            ["worktree", "remove", "--force", str(temp_dir)],
                            cwd=self.repo_root,
                        )
                        await self._remove_directory(temp_dir)
                        raise RuntimeError(f"合并依赖分支失败: {merge_result['stderr']}")

            # 清理临时 worktree（合并完成或中止后）
            await self._run_git_command(
                ["worktree", "remove", "--force", str(temp_dir)],
                cwd=self.repo_root,
            )
            # 确保临时目录被删除
            if temp_dir.exists():
                await self._remove_directory(temp_dir)

            logger.info(
                "中间分支已创建: %s (基于 %s, 合入 %s, 冲突=%d, 成功=%s)",
                target_branch_name, base_branch,
                other_branches, len(merge_conflicts), merge_success,
            )

            return {
                "merged": merge_success,
                "target_branch": target_branch_name,
                "conflict_files": merge_conflicts,
                "base_branch": base_branch,
                "merged_branches": source_branches,
            }

        except Exception as e:
            logger.error("创建中间合并分支失败: %s", e)

            # 清理可能的临时 worktree
            try:
                await self._run_git_command(
                    ["worktree", "prune"],
                    cwd=self.repo_root,
                )
            except Exception:
                pass

            return {"merged": False, "error": str(e)}

    async def delete_branch(self, branch_name: str) -> bool:
        """删除分支（用于清理中间合并分支）"""
        result = await self._run_git_command(
            ["branch", "-D", branch_name],
            cwd=self.repo_root,
        )
        if result["exit_code"] == 0:
            logger.info("分支已删除: %s", branch_name)
            return True
        logger.warning("删除分支失败: %s, error: %s", branch_name, result["stderr"])
        return False

    async def get_worktree(self, worktree_id: str) -> WorktreeInfo | None:
        """获取 worktree 信息"""
        return self._worktrees.get(worktree_id)

    async def health_check(self, worktree_id: str) -> WorkspaceHealth:
        """检查 worktree 的实时健康状态（git 是 SoT）。

        把 git 当作权威：每次都重新查 git worktree list 和 branch 是否存在。
        内存字典只是快照，不可信。

        所有 commit / merge / 文件写入操作前都应该走一次这个检查 ——
        发现不一致就立刻把任务标为 corrupted，不要继续在烂状态上叠操作。
        """
        info = self._worktrees.get(worktree_id)
        if info is None:
            # 内存字典里都没有，直接给一个全 False 的快照
            return WorkspaceHealth(
                in_memory=False,
                on_disk=False,
                in_git_worktree=False,
                branch_exists=False,
            )
        return await compute_health(
            repo_root=self.repo_root,
            worktree_path=info.path,
            branch_name=info.branch_name,
            in_memory=True,
        )

    async def assert_healthy(self, worktree_id: str) -> WorktreeInfo:
        """断言 worktree 健康，否则抛 WorkspaceCorrupted。

        返回 WorktreeInfo 让调用方继续操作；不健康时调用方拿到的是异常。
        这是把"健康"作为前置条件强制起来的工具。

        副作用：一旦发现不健康，会把 lifecycle 状态机推进到 CORRUPTED 终态。
        """
        info = self._worktrees.get(worktree_id)
        if info is None:
            raise WorktreeNotFound(
                f"WorktreeManager 中找不到 id={worktree_id}（可能从未创建或已被清理）"
            )
        health = await self.health_check(worktree_id)
        if not health.is_coherent:
            # 把状态机推到 CORRUPTED，避免后续操作再"健康检查通过"
            if info.lifecycle is not None:
                try:
                    info.lifecycle.transition_to(
                        WorkspaceState.CORRUPTED,
                        reason=health.diagnosis,
                    )
                except InvalidTransitionError:
                    pass  # 已经在终态，无需再变
            raise WorkspaceCorrupted(
                f"worktree {worktree_id} 健康检查失败：{health.diagnosis}",
                health=health,
            )
        return info

    async def list_worktrees(self) -> list[WorktreeInfo]:
        """列出所有 worktree"""
        result = await self._run_git_command(
            ["worktree", "list", "--porcelain"],
            cwd=self.repo_root,
        )

        if result["exit_code"] != 0:
            return list(self._worktrees.values())

        # 解析输出
        worktrees = []
        current_info = {}

        for line in result["stdout"].split("\n"):
            if line.startswith("worktree "):
                if current_info:
                    worktrees.append(current_info)
                current_info = {"path": line.split(" ", 1)[1]}
            elif line.startswith("HEAD "):
                current_info["commit"] = line.split(" ", 1)[1]
            elif line.startswith("branch "):
                current_info["branch"] = line.split(" ", 1)[1]

        if current_info:
            worktrees.append(current_info)

        return list(self._worktrees.values())

    async def commit_changes(
        self,
        worktree_id: str,
        message: str | None = None,
    ) -> dict[str, Any]:
        """在 worktree 中提交更改。

        新行为：
          1. 进入前必须健康（assert_healthy）。不健康抛 WorkspaceCorrupted。
          2. 状态机：ACTIVE → COMMITTING → COMMITTED（成功）或 FAILED（失败）。
          3. git commit 失败抛 WorktreeCommitFailed（类化错误）。
          4. 无变更不算失败 —— 返回 {"committed": False, "reason": "no_changes"}，
             状态机回到 ACTIVE（允许后续重试 / merge）。

        Returns:
            成功: {"committed": True, "commit_hash": "...", "message": "..."}
            无变更: {"committed": False, "reason": "no_changes"}

        Raises:
            WorkspaceCorrupted: 健康检查失败
            WorktreeCommitFailed: git commit 自身失败
            WorktreeNotFound: worktree_id 不在字典里
        """
        # 健康自检 —— 不健康直接抛
        info = await self.assert_healthy(worktree_id)

        if message is None:
            message = f"Subagent {worktree_id} task complete"

        worktree_path = info.path

        # 状态机：进入 COMMITTING
        if info.lifecycle is not None:
            try:
                info.lifecycle.transition_to(
                    WorkspaceState.COMMITTING,
                    reason="commit_changes invoked",
                )
            except InvalidTransitionError as e:
                # 状态机不允许提交 —— 通常是 ACTIVE 之外的状态
                raise WorktreeCommitFailed(
                    f"worktree {worktree_id} 不在可提交状态: {e}"
                ) from e

        try:
            # git add -A
            await self._run_git_command(["add", "-A"], cwd=worktree_path)

            # 检查是否有变更
            status_result = await self._run_git_command(
                ["status", "--porcelain"],
                cwd=worktree_path,
            )

            if not status_result["stdout"].strip():
                logger.info("Worktree 无变更，跳过提交: %s", worktree_id)
                if info.lifecycle is not None:
                    info.lifecycle.state = WorkspaceState.ACTIVE
                return {"committed": False, "reason": "no_changes"}

            # git commit
            commit_result = await self._run_git_command(
                ["commit", "-m", message],
                cwd=worktree_path,
            )

            if commit_result["exit_code"] != 0:
                if info.lifecycle is not None:
                    try:
                        info.lifecycle.transition_to(
                            WorkspaceState.FAILED,
                            reason=f"git commit failed: {commit_result['stderr'].strip()[:80]}",
                        )
                    except InvalidTransitionError:
                        pass
                raise WorktreeCommitFailed(
                    f"git commit 失败 (id={worktree_id}): "
                    f"{commit_result['stderr'].strip()[:200]}"
                )

            # 获取 commit hash
            hash_result = await self._run_git_command(
                ["rev-parse", "HEAD"],
                cwd=worktree_path,
            )

            info.commit_hash = (
                hash_result["stdout"].strip()
                if hash_result["exit_code"] == 0 else None
            )

            # 状态机推进到 COMMITTED
            if info.lifecycle is not None:
                try:
                    info.lifecycle.transition_to(
                        WorkspaceState.COMMITTED,
                        reason=f"commit {info.commit_hash[:8] if info.commit_hash else '?'}",
                    )
                except InvalidTransitionError:
                    pass

            logger.info(
                "Worktree 已提交: id=%s, commit=%s",
                worktree_id, info.commit_hash[:8] if info.commit_hash else "unknown"
            )

            return {
                "committed": True,
                "commit_hash": info.commit_hash,
                "message": message,
            }

        except WorktreeError:
            raise
        except Exception as e:
            if info.lifecycle is not None:
                try:
                    info.lifecycle.transition_to(
                        WorkspaceState.FAILED, reason=f"unexpected: {e}",
                    )
                except InvalidTransitionError:
                    pass
            logger.error("提交未预期异常: %s", e)
            raise WorktreeCommitFailed(
                f"提交未预期异常 (id={worktree_id}): {type(e).__name__}: {e}"
            ) from e

    async def merge_to_parent(
        self,
        worktree_id: str,
        parent_branch: str | None = None,
        delete_branch: bool = True,
    ) -> dict[str, Any]:
        """将 worktree 分支合并到父分支。

        新行为：
          - 不健康抛 WorkspaceCorrupted（git 是 SoT）
          - 状态机：COMMITTED → MERGING → MERGED（成功）或 FAILED（失败）
          - 冲突仍走"返回字典 has_conflicts=True"，不抛异常 ——
            因为冲突是用户决策点（git_resolve_conflict 工具介入），
            不是基础设施失败，分支需保留供用户处理

        Returns:
            成功: {"merged": True, "source_branch": ..., "target_branch": ...}
            冲突: {"merged": False, "has_conflicts": True, "conflict_files": [...]}

        Raises:
            WorkspaceCorrupted: 健康检查失败
            WorktreeMergeConflict: （不在这里抛 —— 冲突走返回值，便于上层 git_resolve_conflict）
            WorktreeError: git checkout/merge 自身异常（非冲突）
        """
        info = await self.assert_healthy(worktree_id)
        target_branch = parent_branch or info.parent_branch
        source_branch = info.branch_name

        # 状态机：COMMITTED → MERGING（如果还在 ACTIVE 也允许过渡）
        if info.lifecycle is not None:
            if info.lifecycle.state == WorkspaceState.ACTIVE:
                # 没 commit 就 merge —— 在历史上记一笔，但允许（git merge 会自己处理）
                logger.debug(
                    "merge_to_parent 在 ACTIVE 状态触发（未 commit）: id=%s",
                    worktree_id,
                )
            try:
                info.lifecycle.transition_to(
                    WorkspaceState.MERGING,
                    reason=f"merge {source_branch} -> {target_branch}",
                )
            except InvalidTransitionError:
                # 不在 COMMITTED/ACTIVE，不让走
                raise WorktreeError(
                    f"worktree {worktree_id} 当前状态不允许 merge: "
                    f"state={info.lifecycle.state.value}"
                )

        try:
            # 切换到目标分支
            checkout_result = await self._run_git_command(
                ["checkout", target_branch],
                cwd=self.repo_root,
            )

            if checkout_result["exit_code"] != 0:
                if info.lifecycle is not None:
                    try:
                        info.lifecycle.transition_to(
                            WorkspaceState.FAILED, reason="checkout failed",
                        )
                    except InvalidTransitionError:
                        pass
                raise WorktreeError(
                    f"切换到目标分支失败 ({target_branch}): "
                    f"{checkout_result['stderr'].strip()[:200]}"
                )

            info.status = WorktreeStatus.MERGING

            merge_result = await self._run_git_command(
                ["merge", source_branch, "-m", f"Merge {source_branch}"],
                cwd=self.repo_root,
            )

            if merge_result["exit_code"] != 0:
                # 冲突 vs 非冲突分两条路：
                #   冲突：返回 has_conflicts，让 master 用 git_resolve_conflict 介入；
                #   非冲突错（git 命令本身异常）：抛 WorktreeError，硬失败
                if "CONFLICT" in merge_result["stdout"] or "CONFLICT" in merge_result["stderr"]:
                    conflict_files = await self._get_conflict_files()
                    # 冲突状态保留 MERGING，供 git_resolve_conflict 接管（不转 FAILED）
                    return {
                        "merged": False,
                        "has_conflicts": True,
                        "conflict_files": conflict_files,
                        "message": "合并冲突，需要手动解决",
                        "source_branch": source_branch,
                        "target_branch": target_branch,
                    }
                else:
                    if info.lifecycle is not None:
                        try:
                            info.lifecycle.transition_to(
                                WorkspaceState.FAILED,
                                reason=f"merge non-conflict failure: {merge_result['stderr'].strip()[:80]}",
                            )
                        except InvalidTransitionError:
                            pass
                    raise WorktreeError(
                        f"合并失败 (id={worktree_id}): "
                        f"{merge_result['stderr'].strip()[:200]}"
                    )

            # 删除分支
            if delete_branch:
                await self._run_git_command(
                    ["branch", "-d", source_branch],
                    cwd=self.repo_root,
                )

            info.status = WorktreeStatus.COMPLETED
            info.completed_at = datetime.now(timezone.utc)
            if info.lifecycle is not None:
                try:
                    info.lifecycle.transition_to(
                        WorkspaceState.MERGED,
                        reason=f"merged into {target_branch}",
                    )
                except InvalidTransitionError:
                    pass

            logger.info(
                "Worktree 已合并: id=%s, branch=%s -> %s",
                worktree_id, source_branch, target_branch
            )

            return {
                "merged": True,
                "source_branch": source_branch,
                "target_branch": target_branch,
            }

        except WorktreeError:
            info.error_message = "see above"
            info.status = WorktreeStatus.FAILED
            raise
        except Exception as e:
            info.status = WorktreeStatus.FAILED
            info.error_message = str(e)
            if info.lifecycle is not None:
                try:
                    info.lifecycle.transition_to(
                        WorkspaceState.FAILED, reason=f"unexpected: {e}",
                    )
                except InvalidTransitionError:
                    pass
            logger.error("合并未预期异常: %s", e)
            raise WorktreeError(
                f"合并未预期异常 (id={worktree_id}): {type(e).__name__}: {e}"
            ) from e

    async def remove_worktree(
        self,
        worktree_id: str,
        force: bool = False,
    ) -> bool:
        """移除 worktree

        Args:
            worktree_id: Worktree ID
            force: 是否强制移除（即使有未提交的更改）

        Returns:
            是否成功移除
        """
        info = self._worktrees.get(worktree_id)
        if info is None:
            logger.warning("Worktree 不存在: %s", worktree_id)
            return False

        try:
            # 执行 git worktree remove
            args = ["worktree", "remove"]
            if force:
                args.append("--force")
            args.append(str(info.path))

            result = await self._run_git_command(
                args,
                cwd=self.repo_root,
            )

            if result["exit_code"] != 0 and not force:
                # 尝试强制移除
                result = await self._run_git_command(
                    ["worktree", "remove", "--force", str(info.path)],
                    cwd=self.repo_root,
                )

            # 清理目录（如果还存在）
            if info.path.exists():
                await self._remove_directory(info.path)

            info.status = WorktreeStatus.CLEANED
            del self._worktrees[worktree_id]

            logger.info("Worktree 已移除: id=%s", worktree_id)
            return True

        except Exception as e:
            logger.error("移除 worktree 失败: %s", e)
            return False

    async def cleanup_all(self) -> int:
        """清理所有 worktree"""
        count = 0
        for worktree_id in list(self._worktrees.keys()):
            if await self.remove_worktree(worktree_id, force=True):
                count += 1

        # 执行 git worktree prune
        await self._run_git_command(
            ["worktree", "prune"],
            cwd=self.repo_root,
        )

        logger.info("清理了 %d 个 worktree", count)
        return count

    async def get_status(self, worktree_id: str) -> dict[str, Any]:
        """获取 worktree 状态"""
        info = self._worktrees.get(worktree_id)
        if info is None:
            return {"exists": False}

        # 获取 git status
        status_result = await self._run_git_command(
            ["status", "--porcelain"],
            cwd=info.path,
        )

        changed_files = []
        if status_result["exit_code"] == 0:
            changed_files = [
                line[3:] for line in status_result["stdout"].strip().split("\n")
                if line.strip()
            ]

        return {
            "exists": True,
            "worktree_id": worktree_id,
            "branch": info.branch_name,
            "path": str(info.path),
            "status": info.status.value,
            "changed_files": changed_files,
            "commit_hash": info.commit_hash,
        }

    async def _is_git_repo(self) -> bool:
        """检查是否为 Git 仓库"""
        result = await self._run_git_command(
            ["rev-parse", "--is-inside-work-tree"],
            cwd=self.repo_root,
        )
        return result["exit_code"] == 0 and result["stdout"].strip() == "true"

    async def _prune_stale_worktrees(self) -> None:
        """清理过期的 worktree 记录"""
        await self._run_git_command(
            ["worktree", "prune"],
            cwd=self.repo_root,
        )

    async def _get_conflict_files(self) -> list[str]:
        """获取冲突文件列表"""
        result = await self._run_git_command(
            ["diff", "--name-only", "--diff-filter=U"],
            cwd=self.repo_root,
        )

        if result["exit_code"] != 0:
            return []

        return [
            line.strip() for line in result["stdout"].strip().split("\n")
            if line.strip()
        ]

    async def _remove_directory(self, path: Path) -> None:
        """异步删除目录"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, shutil.rmtree, path, True)

    async def _run_git_command(
        self,
        args: list[str],
        cwd: Path,
    ) -> dict[str, Any]:
        """执行 Git 命令"""
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            return {
                "exit_code": process.returncode or 0,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
            }

        except Exception as e:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
            }


# 导出
__all__ = [
    "WorktreeStatus",
    "WorktreeInfo",
    "WorktreeManager",
]