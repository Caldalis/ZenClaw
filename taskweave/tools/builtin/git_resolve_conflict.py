"""
Git 冲突解决工具 — Master Agent 在 merge 冲突时使用

工作流程契合 Phase 4 设计：
  Master 收到 merge 失败 → 调用 list_conflicts 拿到文件列表
                       → 决定每文件用 ours / theirs / 手工 patch
                       → 调用 resolve(action) 落盘
                       → 调用 finalize(commit / abort) 收尾

设计原则：
  - 工具只在主仓 repo_root 工作（不进 worktree）
  - 所有破坏性操作都需要显式指定 action，禁止默认值兜底
  - abort 永远是安全退路（git merge --abort 回到 merge 前状态）
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from taskweave.tools.base import Tool
from taskweave.utils.logging import get_logger

logger = get_logger(__name__)


async def _run_git(args: list[str], cwd: Path) -> dict[str, Any]:
    """运行 git 子进程并返回 (exit_code, stdout, stderr)。"""
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"  # 禁止 git 弹交互提示
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await proc.communicate()
    return {
        "exit_code": proc.returncode or 0,
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
    }


class GitResolveConflictTool(Tool):
    """合并冲突解决工具 — 给 Master Agent 用
    actions:
      - list           列出当前未解决的冲突文件
      - take_ours      对指定文件保留 HEAD 版本（git checkout --ours + add）
      - take_theirs    对指定文件采用合入分支版本（git checkout --theirs + add）
      - mark_resolved  对已经手工编辑过的文件直接 git add（声明已解决）
      - finalize       提交合并 commit（确认全部冲突已解决）
      - abort          放弃整次合并（git merge --abort），回到 merge 前状态

    每次调用必须显式提供 action；list/finalize/abort 不需要 files。
    """

    def __init__(self, repo_root: str | Path):
        self._repo_root = Path(repo_root).resolve()

    @property
    def name(self) -> str:
        return "git_resolve_conflict"

    @property
    def description(self) -> str:
        return """合并冲突解决工具（仅 Master 使用）。

调用时机：当 master 在收到子任务的 merge 失败信号、且 git status 存在
unmerged paths 时，按以下顺序使用：

1. action="list" 拿到冲突文件清单和每个文件的状态
2. 对每个冲突文件选一个策略：
   - "take_ours": 完全保留主分支版本（丢弃 subagent 改动）
   - "take_theirs": 完全采用 subagent 版本（覆盖主分支）
   - "mark_resolved": 你已经通过 read_file/write_file 手工编辑了文件，告诉 git 已解决
3. action="finalize" 完成合并 commit；或 action="abort" 放弃整次合并

不要默认选 ours 或 theirs —— 先 list 看清楚冲突再选。abort 永远是退路。"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "list", "take_ours", "take_theirs",
                        "mark_resolved", "finalize", "abort",
                    ],
                    "description": "操作类型",
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "目标文件列表（take_ours / take_theirs / mark_resolved 必填）",
                },
                "commit_message": {
                    "type": "string",
                    "description": "finalize 时的 merge commit 信息（可选，默认沿用 git 自动生成）",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        files = kwargs.get("files", []) or []
        commit_msg = kwargs.get("commit_message", "")

        if action == "list":
            return await self._list_conflicts()
        if action == "take_ours":
            return await self._take_side(files, side="ours")
        if action == "take_theirs":
            return await self._take_side(files, side="theirs")
        if action == "mark_resolved":
            return await self._mark_resolved(files)
        if action == "finalize":
            return await self._finalize_merge(commit_msg)
        if action == "abort":
            return await self._abort_merge()

        return json.dumps(
            {"status": "error", "message": f"未知 action: {action}"},
            ensure_ascii=False,
        )

    async def _list_conflicts(self) -> str:
        """列出所有 unmerged 路径及其阶段。"""
        # git ls-files -u 列出 stage 1/2/3 的条目（base/ours/theirs）
        ls = await _run_git(["ls-files", "-u"], cwd=self._repo_root)
        if ls["exit_code"] != 0:
            return json.dumps(
                {"status": "error", "message": ls["stderr"][:500]},
                ensure_ascii=False,
            )

        # 同时拿短状态便于人读
        status = await _run_git(["status", "--short"], cwd=self._repo_root)

        files: dict[str, dict[str, Any]] = {}
        for line in ls["stdout"].splitlines():
            # 格式: <mode> <hash> <stage>\t<path>
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            meta = parts[0].split()
            path = parts[1]
            if path not in files:
                files[path] = {"stages": [], "in_index": True}
            if len(meta) >= 3:
                files[path]["stages"].append(int(meta[2]))

        # 检测是否真的处于 merge 状态
        in_merge = (self._repo_root / ".git" / "MERGE_HEAD").exists()

        return json.dumps({
            "status": "ok",
            "in_merge_state": in_merge,
            "conflict_files": list(files.keys()),
            "details": files,
            "git_status_short": status["stdout"].strip().splitlines()[:50],
        }, ensure_ascii=False, indent=2)

    async def _take_side(self, files: list[str], side: str) -> str:
        """对每个文件 git checkout --ours/--theirs + git add。"""
        if not files:
            return json.dumps(
                {"status": "error", "message": "files 不能为空"},
                ensure_ascii=False,
            )
        flag = f"--{side}"
        applied: list[str] = []
        errors: list[dict[str, str]] = []
        for f in files:
            chk = await _run_git(["checkout", flag, "--", f], cwd=self._repo_root)
            if chk["exit_code"] != 0:
                errors.append({"file": f, "error": chk["stderr"][:200]})
                continue
            add = await _run_git(["add", "--", f], cwd=self._repo_root)
            if add["exit_code"] != 0:
                errors.append({"file": f, "error": add["stderr"][:200]})
                continue
            applied.append(f)
        return json.dumps({
            "status": "ok" if not errors else "partial",
            "side": side,
            "applied": applied,
            "errors": errors,
        }, ensure_ascii=False, indent=2)

    async def _mark_resolved(self, files: list[str]) -> str:
        """对手工编辑过的文件 git add。"""
        if not files:
            return json.dumps(
                {"status": "error", "message": "files 不能为空"},
                ensure_ascii=False,
            )
        applied: list[str] = []
        errors: list[dict[str, str]] = []
        for f in files:
            res = await _run_git(["add", "--", f], cwd=self._repo_root)
            if res["exit_code"] == 0:
                applied.append(f)
            else:
                errors.append({"file": f, "error": res["stderr"][:200]})
        return json.dumps({
            "status": "ok" if not errors else "partial",
            "applied": applied,
            "errors": errors,
        }, ensure_ascii=False, indent=2)

    async def _finalize_merge(self, commit_msg: str) -> str:
        """完成 merge commit。"""
        # 先确认无残留冲突
        ls = await _run_git(["ls-files", "-u"], cwd=self._repo_root)
        if ls["exit_code"] == 0 and ls["stdout"].strip():
            return json.dumps({
                "status": "error",
                "message": "仍存在未解决的冲突，无法 finalize。先用 list 查看，再 take_ours/take_theirs/mark_resolved 解决。",
                "remaining": ls["stdout"].strip().splitlines(),
            }, ensure_ascii=False, indent=2)

        args = ["commit", "--no-edit"]
        if commit_msg:
            args = ["commit", "-m", commit_msg]
        res = await _run_git(args, cwd=self._repo_root)
        if res["exit_code"] != 0:
            return json.dumps({
                "status": "error",
                "message": res["stderr"][:500],
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "status": "ok",
            "message": "merge commit 已生成",
            "git_output": res["stdout"][:500],
        }, ensure_ascii=False, indent=2)

    async def _abort_merge(self) -> str:
        """放弃合并，回到 merge 前状态。"""
        if not (self._repo_root / ".git" / "MERGE_HEAD").exists():
            return json.dumps({
                "status": "error",
                "message": "当前不在 merge 状态，无需 abort",
            }, ensure_ascii=False)
        res = await _run_git(["merge", "--abort"], cwd=self._repo_root)
        if res["exit_code"] != 0:
            return json.dumps({
                "status": "error",
                "message": res["stderr"][:500],
            }, ensure_ascii=False)
        return json.dumps({
            "status": "ok",
            "message": "merge 已 abort，工作树回到 merge 前状态",
        }, ensure_ascii=False)


__all__ = ["GitResolveConflictTool"]
