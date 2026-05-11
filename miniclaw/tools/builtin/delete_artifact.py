"""
Restricted cleanup deletion tool.

CleanupAgent runs in the main workspace without a git worktree, so it must not
use the generic terminal tool for deletion. This tool deletes only exact files
that the scheduler pre-approved from the cleanup instruction.

After unlinking a tracked file, the tool stages and commits the deletion on
the main repo. Without this, future subagent worktrees (created from main
HEAD) would resurrect the file from the previous commit. Untracked files are
deleted without a commit (nothing for git to record).
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from miniclaw.tools.base import Tool

_GLOB_CHARS = frozenset("*?[]")
_PATH_RE = re.compile(
    r"(?<![\w./\\-])"
    r"(?:[A-Za-z0-9_.-]+[\\/])*"
    r"[A-Za-z0-9_.-]+\.[A-Za-z0-9][A-Za-z0-9_.-]{0,15}"
    r"(?![\w./\\-])"
)
_QUOTED_RE = re.compile(r"[`\"']([^`\"'\r\n]+)[`\"']")
_SENTENCE_BOUNDARIES = "\n\r。.!?；;"
_NEGATIVE_MARKERS = (
    "不要动",
    "不要删",
    "不要删除",
    "别删",
    "勿删",
    "保留",
    "留着",
    "do not delete",
    "do not touch",
    "dont delete",
    "don't delete",
    "keep",
    "leave untouched",
    "exclude",
)


def extract_explicit_artifact_paths(instruction: str) -> list[str]:
    """Extract cleanup candidate paths from an instruction.

    This intentionally uses a conservative heuristic: exact file-looking tokens
    only, with candidates in "keep / do not delete" clauses excluded.
    """
    if not instruction:
        return []

    candidates: list[tuple[str, int, int]] = []
    for match in _PATH_RE.finditer(instruction):
        candidates.append((match.group(0), match.start(), match.end()))

    for match in _QUOTED_RE.finditer(instruction):
        raw = match.group(1).strip()
        if _looks_like_path(raw):
            candidates.append((raw, match.start(1), match.end(1)))

    # 去重前先做二次过滤，避免把通配符或 不要删除 语境里的路径放进候选列表。
    seen: set[str] = set()
    result: list[str] = []
    for raw, start, end in candidates:
        path = _clean_candidate(raw)
        if not path or not _looks_like_path(path):
            continue
        if _has_glob(path):
            continue
        if _is_negative_clause(instruction, start, end):
            continue
        key = path.replace("\\", "/").casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result
def _clean_candidate(value: str) -> str:
    return value.strip().strip(" \t\r\n,;:，。；：（）()[]{}<>")
def _looks_like_path(value: str) -> bool:
    value = _clean_candidate(value)
    if not value or len(value) > 240:
        return False
    lowered = value.lower()
    if lowered.startswith(("http://", "https://")):
        return False
    if any(ch in value for ch in "\r\n"):
        return False
    if " " in value and not ("/" in value or "\\" in value):
        return False
    return bool(Path(value).suffix) or "/" in value or "\\" in value
def _has_glob(value: str) -> bool:
    return any(ch in value for ch in _GLOB_CHARS)
def _is_negative_clause(text: str, start: int, end: int) -> bool:
    left = max(text.rfind(ch, 0, start) for ch in _SENTENCE_BOUNDARIES) + 1
    right_positions = [text.find(ch, end) for ch in _SENTENCE_BOUNDARIES]
    found_right = [pos for pos in right_positions if pos >= 0]
    right = min(found_right) if found_right else len(text)
    clause = text[left:right].casefold()
    return any(marker.casefold() in clause for marker in _NEGATIVE_MARKERS)


class DeleteArtifactTool(Tool):
    """Delete one pre-approved cleanup artifact file."""

    def __init__(
        self,
        repo_root: str | Path | None = None,
        allowed_paths: Iterable[str] | None = None,
    ) -> None:
        self._repo_root = Path(repo_root or Path.cwd()).resolve()
        self._allowed_paths: dict[str, str] = {}
        self.deleted_files: list[str] = []

        for raw_path in allowed_paths or []:
            try:
                # 白名单也走同一套仓库内路径解析，防止初始化时混入越界路径。
                _, rel = self._resolve_inside_root(str(raw_path))
            except ValueError:
                continue
            self._allowed_paths[self._key(rel)] = rel

    @property
    def name(self) -> str:
        return "delete_artifact"

    @property
    def description(self) -> str:
        if self._allowed_paths:
            allowed = ", ".join(sorted(self._allowed_paths.values()))
        else:
            allowed = "(none)"
        return (
            "Delete one cleanup artifact file from the main workspace. "
            "Only exact pre-approved paths are accepted; directories, glob "
            "patterns, and paths outside the repository are rejected. "
            f"Allowed paths: {allowed}"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Exact artifact file path to delete.",
                },
                "reason": {
                    "type": "string",
                    "description": "Short cleanup reason.",
                    "default": "",
                },
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        raw_path = str(kwargs.get("path", "")).strip()
        if not raw_path:
            return self._json_error("missing_path", "Provide an exact file path.")

        if not self._allowed_paths:
            return self._json_error(
                "no_allowed_paths",
                "No cleanup artifact paths were approved for this task.",
            )

        try:
            target, rel = self._resolve_inside_root(raw_path)
        except ValueError as exc:
            return self._json_error("invalid_path", str(exc))

        allowed_rel = self._allowed_paths.get(self._key(rel))
        # 真正删除前必须命中调度器批准的精确路径；大小写和分隔符差异由 key 统一处理。
        if allowed_rel is None:
            return self._json_error(
                "path_not_allowed",
                f"Path is not in this cleanup task's allowlist: {rel}",
                allowed_paths=sorted(self._allowed_paths.values()),
            )

        if not target.exists():
            return self._json(
                status="skipped",
                path=allowed_rel,
                deleted_files=[],
                skipped_files=[allowed_rel],
                message="File does not exist; skipped.",
            )

        if target.is_dir():
            return self._json_error(
                "directory_not_allowed",
                f"Refusing to delete a directory: {allowed_rel}",
            )

        try:
            target.unlink()
        except Exception as exc:
            return self._json_error(
                "delete_failed",
                f"Failed to delete {allowed_rel}: {exc}",
            )

        self.deleted_files.append(allowed_rel)
        git_result = await self._commit_deletion(allowed_rel)

        if git_result.get("committed"):
            message = "File deleted and commit recorded on main."
        else:
            reason = git_result.get("reason", "unknown")
            message = f"File deleted (git commit skipped: {reason})."

        return self._json(
            status="deleted",
            path=allowed_rel,
            deleted_files=[allowed_rel],
            skipped_files=[],
            message=message,
            git=git_result,
        )

    async def _commit_deletion(self, rel_path: str) -> dict[str, Any]:
        """Stage and commit the deletion of rel_path on self._repo_root.

        Path-scoped (`git add -A -- <path>` + `git diff --cached -- <path>`),
        so concurrent unrelated changes elsewhere in the worktree are not
        swept into this commit. Skips the commit if the path was untracked
        (nothing staged) or if the directory is not a git repo. Returns a
        dict suitable for embedding in the response JSON.
        """
        if not (self._repo_root / ".git").exists():
            return {"committed": False, "reason": "not_a_git_repo"}

        repo = str(self._repo_root)

        def _run(args: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                args,
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

        try:
            add_proc = await asyncio.to_thread(
                _run, ["git", "add", "-A", "--", rel_path]
            )
            if add_proc.returncode != 0:
                return {
                    "committed": False,
                    "reason": "git_add_failed",
                    "stderr": add_proc.stderr.strip()[:500],
                }

            # 只检查目标路径的暂存差异；未跟踪文件删除后不会产生可提交内容。
            diff_proc = await asyncio.to_thread(
                _run, ["git", "diff", "--cached", "--quiet", "--", rel_path], 5.0
            )
            if diff_proc.returncode == 0:
                return {"committed": False, "reason": "untracked_no_commit_needed"}

            commit_proc = await asyncio.to_thread(
                _run, ["git", "commit", "-m", f"cleanup: delete {rel_path}"]
            )
            if commit_proc.returncode != 0:
                return {
                    "committed": False,
                    "reason": "git_commit_failed",
                    "stderr": commit_proc.stderr.strip()[:500],
                }

            rev_proc = await asyncio.to_thread(
                _run, ["git", "rev-parse", "--short", "HEAD"], 5.0
            )
            commit_hash = rev_proc.stdout.strip() if rev_proc.returncode == 0 else ""
            return {"committed": True, "commit": commit_hash}
        except subprocess.TimeoutExpired:
            return {"committed": False, "reason": "git_timeout"}
        except FileNotFoundError:
            return {"committed": False, "reason": "git_not_installed"}
        except Exception as exc:
            return {
                "committed": False,
                "reason": f"git_error: {type(exc).__name__}: {exc}",
            }

    def get_deleted_files(self) -> list[str]:
        return list(self.deleted_files)

    def _resolve_inside_root(self, raw_path: str) -> tuple[Path, str]:
        path = _clean_candidate(raw_path)
        if not path:
            raise ValueError("Path is empty.")
        if _has_glob(path):
            raise ValueError("Glob patterns are not allowed; use an exact file path.")

        candidate = Path(path)
        target = candidate if candidate.is_absolute() else self._repo_root / candidate
        target = target.resolve(strict=False)
        try:
            rel_path = target.relative_to(self._repo_root)
        except ValueError as exc:
            raise ValueError("Path is outside the repository root.") from exc

        rel = rel_path.as_posix()
        # 禁止把仓库根目录本身当作 artifact，调用方必须给出具体文件。
        if rel in ("", "."):
            raise ValueError("Repository root is not a deletable artifact.")
        return target, rel

    @staticmethod
    def _key(path: str) -> str:
        return path.replace("\\", "/").casefold()

    @staticmethod
    def _json(**payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _json_error(self, code: str, message: str, **extra: Any) -> str:
        return self._json(
            status="error",
            error_code=code,
            message=message,
            deleted_files=[],
            **extra,
        )


__all__ = [
    "DeleteArtifactTool",
    "extract_explicit_artifact_paths",
]
