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
    """Worktree 信息"""
    worktree_id: str
    """Worktree ID（通常与 task_id 一致）"""

    branch_name: str
    """分支名称"""

    path: Path
    """Worktree 目录路径"""

    parent_branch: str = "main"
    """父分支（基于哪个分支创建）"""

    status: WorktreeStatus = WorktreeStatus.CREATING
    """当前状态"""

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

        logger.info("Worktree 管理器已初始化: repo=%s, base=%s", self.repo_root, self.worktree_base)

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

        # 创建 worktree
        info = WorktreeInfo(
            worktree_id=worktree_id,
            branch_name=branch_name,
            path=worktree_path,
            parent_branch=parent_branch,
            status=WorktreeStatus.CREATING,
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

            if result["exit_code"] != 0:
                # 可能是分支已存在，尝试不创建新分支
                if "already exists" in result["stderr"]:
                    logger.info("分支已存在，复用分支: %s", branch_name)
                    result = await self._run_git_command(
                        ["worktree", "add", str(worktree_path), branch_name],
                        cwd=self.repo_root,
                    )

                if result["exit_code"] != 0:
                    raise RuntimeError(f"创建 worktree 失败: {result['stderr']}")

            info.status = WorktreeStatus.ACTIVE
            self._worktrees[worktree_id] = info

            logger.info(
                "Worktree 已创建: id=%s, branch=%s, path=%s",
                worktree_id, branch_name, worktree_path
            )

            return info

        except Exception as e:
            info.status = WorktreeStatus.FAILED
            info.error_message = str(e)
            logger.error("创建 worktree 失败: %s", e)
            raise

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

    async def get_worktree(self, worktree_id: str) -> WorktreeInfo | None:
        """获取 worktree 信息"""
        return self._worktrees.get(worktree_id)

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
        """在 worktree 中提交更改

        Args:
            worktree_id: Worktree ID
            message: 提交消息（默认自动生成）

        Returns:
            提交结果
        """
        info = self._worktrees.get(worktree_id)
        if info is None:
            raise ValueError(f"Worktree 不存在: {worktree_id}")

        if message is None:
            message = f"Subagent {worktree_id} task complete"

        worktree_path = info.path

        try:
            # git add -A
            add_result = await self._run_git_command(
                ["add", "-A"],
                cwd=worktree_path,
            )

            # 检查是否有变更
            status_result = await self._run_git_command(
                ["status", "--porcelain"],
                cwd=worktree_path,
            )

            if not status_result["stdout"].strip():
                logger.info("Worktree 无变更，跳过提交: %s", worktree_id)
                return {"committed": False, "reason": "no_changes"}

            # git commit
            commit_result = await self._run_git_command(
                ["commit", "-m", message],
                cwd=worktree_path,
            )

            if commit_result["exit_code"] != 0:
                raise RuntimeError(f"提交失败: {commit_result['stderr']}")

            # 获取 commit hash
            hash_result = await self._run_git_command(
                ["rev-parse", "HEAD"],
                cwd=worktree_path,
            )

            info.commit_hash = hash_result["stdout"].strip() if hash_result["exit_code"] == 0 else None

            logger.info(
                "Worktree 已提交: id=%s, commit=%s",
                worktree_id, info.commit_hash[:8] if info.commit_hash else "unknown"
            )

            return {
                "committed": True,
                "commit_hash": info.commit_hash,
                "message": message,
            }

        except Exception as e:
            logger.error("提交失败: %s", e)
            return {"committed": False, "error": str(e)}

    async def merge_to_parent(
        self,
        worktree_id: str,
        parent_branch: str | None = None,
        delete_branch: bool = True,
    ) -> dict[str, Any]:
        """将 worktree 分支合并到父分支

        Args:
            worktree_id: Worktree ID
            parent_branch: 目标父分支（默认使用创建时的父分支）
            delete_branch: 合并后是否删除分支

        Returns:
            合并结果
        """
        info = self._worktrees.get(worktree_id)
        if info is None:
            raise ValueError(f"Worktree 不存在: {worktree_id}")

        target_branch = parent_branch or info.parent_branch
        source_branch = info.branch_name

        try:
            # 切换到目标分支
            checkout_result = await self._run_git_command(
                ["checkout", target_branch],
                cwd=self.repo_root,
            )

            if checkout_result["exit_code"] != 0:
                raise RuntimeError(f"切换分支失败: {checkout_result['stderr']}")

            # 执行合并
            info.status = WorktreeStatus.MERGING

            merge_result = await self._run_git_command(
                ["merge", source_branch, "-m", f"Merge {source_branch}"],
                cwd=self.repo_root,
            )

            if merge_result["exit_code"] != 0:
                # 检查是否有冲突
                if "CONFLICT" in merge_result["stdout"] or "CONFLICT" in merge_result["stderr"]:
                    return {
                        "merged": False,
                        "has_conflicts": True,
                        "conflict_files": await self._get_conflict_files(),
                        "message": "合并冲突，需要手动解决",
                    }
                else:
                    raise RuntimeError(f"合并失败: {merge_result['stderr']}")

            # 删除分支
            if delete_branch:
                await self._run_git_command(
                    ["branch", "-d", source_branch],
                    cwd=self.repo_root,
                )

            info.status = WorktreeStatus.COMPLETED
            info.completed_at = datetime.now(timezone.utc)

            logger.info(
                "Worktree 已合并: id=%s, branch=%s -> %s",
                worktree_id, source_branch, target_branch
            )

            return {
                "merged": True,
                "source_branch": source_branch,
                "target_branch": target_branch,
            }

        except Exception as e:
            info.status = WorktreeStatus.FAILED
            info.error_message = str(e)
            logger.error("合并失败: %s", e)
            return {"merged": False, "error": str(e)}

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