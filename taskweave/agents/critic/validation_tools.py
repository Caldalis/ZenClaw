"""
验证工具 — RunLinter 和 RunTests

核心职责:
  1. 提供代码质量验证工具
  2. 返回结构化的验证结果
  3. 与 ValidationGatekeeper 配合，强制闭环验证

设计原则:
  - 验证结果必须明确（通过/失败）
  - 提供具体的错误信息帮助 Agent 修复
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from taskweave.tools.base import Tool
from taskweave.utils.logging import get_logger

logger = get_logger(__name__)


class ValidationStatus(str, Enum):
    """验证状态"""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"  # 执行错误（如工具不可用）


@dataclass
class ValidationResult:
    """验证结果"""
    tool_name: str
    """工具名称"""

    status: ValidationStatus
    """验证状态"""

    message: str = ""
    """摘要消息"""

    errors: list[dict[str, Any]] = field(default_factory=list)
    """错误列表"""

    warnings: list[dict[str, Any]] = field(default_factory=list)
    """警告列表"""

    fixed_issues: int = 0
    """已修复的问题数"""

    execution_time_ms: int = 0
    """执行时间"""

    def is_passed(self) -> bool:
        """是否通过"""
        return self.status == ValidationStatus.PASSED

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "tool": self.tool_name,
            "status": self.status.value,
            "message": self.message,
            "errors": self.errors[:10],  # 限制输出
            "warnings": self.warnings[:10],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


class RunLinterTool(Tool):
    """Linter 工具 — 运行代码检查

    支持:
      - Python: ruff, flake8, pylint
      - JavaScript: eslint
      - 通用检查
    """

    def __init__(self, worktree_root: Path | None = None):
        self._worktree_root = worktree_root
        self._linters = {
            "python": ["ruff", "flake8", "pylint"],
            "javascript": ["eslint"],
        }

    def set_worktree_root(self, root: Path) -> None:
        """设置工作目录"""
        self._worktree_root = root

    @property
    def name(self) -> str:
        return "run_linter"

    @property
    def description(self) -> str:
        return """运行代码检查工具(Linter)检查代码质量。

**重要**: 这是验证工具。在调用 submit_task_result 声明成功前，必须至少通过一次验证。

支持的检查:
- Python: ruff, flake8, pylint （都未安装时自动降级到 stdlib `python -m py_compile` 做语法检查）
- JavaScript: eslint

返回:
- 通过: 无语法错误和关键问题
- 失败: 包含错误列表，需要修复后重新运行"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要检查的文件列表（空=检查所有）",
                },
                "linter": {
                    "type": "string",
                    "enum": ["auto", "ruff", "flake8", "eslint"],
                    "description": "指定 linter，默认 auto 自动检测",
                    "default": "auto",
                },
                "fix": {
                    "type": "boolean",
                    "description": "是否自动修复可修复的问题",
                    "default": False,
                },
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        if self._worktree_root is None:
            return "错误: 工作目录未设置"

        files = kwargs.get("files", [])
        linter = kwargs.get("linter", "auto")
        fix = kwargs.get("fix", False)

        result = await self._run_linter(files, linter, fix)

        # 返回 JSON 格式结果
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)

    async def _run_linter(
        self,
        files: list[str],
        linter: str,
        fix: bool,
    ) -> ValidationResult:
        """运行 linter"""
        import time

        start_time = time.time()

        # 入口闸：agent 显式传 files 时，至少要有一个代码文件是本 worktree 在
        # git 视角下的变更（即真的是本任务的产物）。不允许"显式传一个不相干的 .py 蹭 PASSED" 这种绕过门禁的路径。
        # 这条检查在 linter 检测前做 —— 不论 ruff/flake8 还是 fallback 都该受约束。
        if files:
            scope_block = await self._check_explicit_files_in_scope(files)
            if scope_block is not None:
                scope_block.execution_time_ms = int((time.time() - start_time) * 1000)
                return scope_block

        # 确定要使用的 linter
        if linter == "auto":
            linter = self._detect_linter()

        if linter is None:
            # 没有装 pylint 或者 eslint等 不再直接抛 ERROR 让验证
            # 门禁放行，而是降级到 stdlib 自带工具做最低
            # 限度的语法验证：
            #   - Python：`python -m py_compile`（stdlib，零依赖）
            #   - JS  ：`node --check`（Node 通常已装；没装则视为不能验证）
            # 这样配合 ValidationGatekeeper 不再认 ERROR 为 ACCEPTABLE，整套
            # "写代码 subagent 必须通过验证才能 submit" 的契约就闭合了。
            fallback = await self._run_stdlib_syntax_fallback(files)
            fallback.execution_time_ms = int((time.time() - start_time) * 1000)
            return fallback

        # 构建 linter 命令
        cmd = self._build_linter_command(linter, files, fix)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._worktree_root),
            )

            stdout, stderr = await process.communicate()
            execution_time = int((time.time() - start_time) * 1000)

            # 解析输出
            result = self._parse_linter_output(
                linter=linter,
                exit_code=process.returncode or 0,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
            )

            result.execution_time_ms = execution_time

            return result

        except FileNotFoundError:
            return ValidationResult(
                tool_name="run_linter",
                status=ValidationStatus.ERROR,
                message=f"Linter '{linter}' 未安装",
            )
        except Exception as e:
            return ValidationResult(
                tool_name="run_linter",
                status=ValidationStatus.ERROR,
                message=f"执行错误: {str(e)}",
            )

    def _detect_linter(self) -> str | None:
        """检测可用的 linter"""
        # 优先使用 ruff
        for linter in ["ruff", "flake8", "eslint"]:
            try:
                subprocess.run(
                    [linter, "--version"],
                    capture_output=True,
                    check=True,
                )
                return linter
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        return None

    # 收集源文件时跳过的目录（虚拟环境 / 缓存 / 第三方依赖 / git 内部）
    _FALLBACK_SKIP_DIRS = frozenset({
        "__pycache__", ".venv", "venv", "env", ".env",
        ".git", "node_modules", ".tox", ".pytest_cache",
        ".mypy_cache", ".ruff_cache", "build", "dist",
        ".eggs", ".next", ".turbo", ".cache",
    })

    # stdlib fallback 一次最多检查这么多文件，避免在大仓库里跑爆
    _FALLBACK_MAX_FILES = 100

    # JS fallback 认这些扩展名（.ts/.tsx 不能用 node --check 解析，需 tsc，
    # 不在 stdlib fallback 范围；用户用 TS 时应有 eslint）
    _JS_FALLBACK_SUFFIXES = frozenset({".js", ".mjs", ".cjs"})

    def _collect_files_for_fallback(
        self,
        files: list[str],
        suffixes: frozenset[str],
    ) -> list[Path]:
        """按扩展名收集 fallback 要检查的源文件。

        优先级：
          1. agent 显式传了 files 参数 → 取里面后缀匹配的文件（必须落在
             worktree 内，防越界）
          2. 没传 → 在 worktree_root 下递归扫，排除 __pycache__ / .venv /
             .git / node_modules 等噪声目录
          3. 命中数到 _FALLBACK_MAX_FILES 即停（保护大仓库）
        """
        if self._worktree_root is None:
            return []

        root = self._worktree_root.resolve()

        if files:
            collected: list[Path] = []
            for raw in files:
                p = (self._worktree_root / raw).resolve()
                # 沙箱：只接受 worktree 内的路径
                try:
                    p.relative_to(root)
                except ValueError:
                    continue
                if p.is_file() and p.suffix in suffixes:
                    collected.append(p)
            return collected[: self._FALLBACK_MAX_FILES]

        collected = []
        for cand in root.rglob("*"):
            if not cand.is_file() or cand.suffix not in suffixes:
                continue
            try:
                rel_parts = cand.relative_to(root).parts
            except ValueError:
                continue
            if any(part in self._FALLBACK_SKIP_DIRS for part in rel_parts):
                continue
            collected.append(cand)
            if len(collected) >= self._FALLBACK_MAX_FILES:
                break
        return collected

    # 给 agent 看的"非代码文件"扩展名集合 —— 用于诊断 message
    _COMMON_NON_CODE_SUFFIXES = frozenset({
        ".md", ".txt", ".rst", ".json", ".yaml", ".yml",
        ".toml", ".ini", ".cfg", ".csv", ".tsv",
        ".html", ".css", ".svg", ".png", ".jpg", ".jpeg", ".gif",
    })

    async def _run_stdlib_syntax_fallback(
        self,
        files: list[str],
    ) -> ValidationResult:
        """ruff/flake8/pylint/eslint 都未装时的兜底：用 stdlib 工具做语法检查。

        分语种处理：
          - .py 文件   →  `python -m py_compile`（stdlib 自带，零依赖）
          - .js/.mjs/.cjs → `node --check`（Node 通常已装；没装则该子检查 ERROR）

        scope 规则（关键 —— 防止"用历史残留 .py 蹭过验证"）：
          - agent 显式传了 files：严格按它给的清单过滤代码文件
          - agent 没传（隐式扫描）：**只扫本 worktree 在 git 视角下的变更文件**
            （`git status --porcelain`），不再 rglob 整个 worktree。
            理由：worktree 从父分支继承的历史 .py 不属于本任务的产物，
            把它们扫进来会让"agent 写了 .md，结果用别人的 .py 拿到 PASSED"
            这种假阳性发生。
          - git 命令本身不可用时（罕见，比如非 git 目录）回落到 rglob，保留可用性。

        组合规则（严格契约，杜绝"假装通过"）：
          - 任一子检查 FAILED → 整体 FAILED（合并 errors 列表）
          - 所有跑过的子检查 PASSED 且至少跑了一项 → 整体 PASSED
          - 一个子检查都没跑起来 → ERROR（消息按 explicit/implicit 分支给清晰诊断）
        """
        if files:
            return await self._fallback_explicit_files(files)
        return await self._fallback_implicit_scan()

    async def _fallback_explicit_files(
        self,
        files: list[str],
    ) -> ValidationResult:
        """agent 显式传了 files —— 按清单过滤后跑 fallback。"""
        py_files = self._collect_files_for_fallback(files, frozenset({".py"}))
        js_files = self._collect_files_for_fallback(files, self._JS_FALLBACK_SUFFIXES)

        if not py_files and not js_files:
            # 检查是不是传了非代码文件 —— 给清晰诊断而不是模糊的"linter 没装"
            preview = ", ".join(files[:3]) + (" ..." if len(files) > 3 else "")
            non_code_hint = ""
            non_code = [
                f for f in files
                if any(f.endswith(suf) for suf in self._COMMON_NON_CODE_SUFFIXES)
            ]
            if non_code:
                non_code_hint = (
                    f"\n\n你传的文件里包含非代码文件（如 {non_code[0]}）。"
                    "**run_linter 不验证文档/数据/配置文件**，它只对 .py/.js 等"
                    "源代码做语法检查。"
                )
            return ValidationResult(
                tool_name="run_linter",
                status=ValidationStatus.ERROR,
                message=(
                    f"传给 run_linter 的 files=[{preview}] 里没有可验证的代码文件"
                    "（既不是 .py 也不是 .js/.mjs/.cjs）。"
                    f"{non_code_hint}\n\n"
                    "处理方案：\n"
                    "1. 如果你**还有代码文件**没列上，把代码文件名一起传进来再调一次。\n"
                    "2. 如果本任务的产物**只有文档/数据**，没有代码可验证 —— "
                    "用 `skip_validation(reason='本任务只产出文档/数据，无代码')` "
                    "明确告知门禁本任务不需要代码验证。"
                ),
            )

        return await self._dispatch_subchecks(py_files, js_files)

    async def _fallback_implicit_scan(self) -> ValidationResult:
        """agent 没传 files —— 走 git status 只看本 worktree 的变更。

        分三种诊断分支：
          - git 不可用：rglob 全 worktree 兜底，仍无代码 → "未装 linter 也无 .py/.js"
          - git 可用 + worktree clean（无变更）：消息说"你还没产出"
          - git 可用 + 有变更但全是非代码：消息说"产物是文档/数据，请 skip_validation"
        """
        changed_files = await self._get_changed_files_from_git()
        git_unavailable = changed_files is None

        if git_unavailable:
            logger.info(
                "git status 不可用，run_linter fallback 回落到全 worktree 扫描"
            )
            py_files = self._collect_files_for_fallback([], frozenset({".py"}))
            js_files = self._collect_files_for_fallback([], self._JS_FALLBACK_SUFFIXES)
        else:
            py_files = [
                f for f in changed_files if f.suffix == ".py"
            ][: self._FALLBACK_MAX_FILES]
            js_files = [
                f for f in changed_files
                if f.suffix in self._JS_FALLBACK_SUFFIXES
            ][: self._FALLBACK_MAX_FILES]

        if py_files or js_files:
            return await self._dispatch_subchecks(py_files, js_files)

        # 以下三个分支都是"找不到代码可 lint" —— 给最贴切的 ERROR 消息
        if git_unavailable:
            # 兜底：环境真的不可验证，给老版本一致的消息
            return ValidationResult(
                tool_name="run_linter",
                status=ValidationStatus.ERROR,
                message=(
                    "未找到可用的 linter (ruff/flake8/pylint/eslint)，"
                    "且 worktree 内无 .py / .js / .mjs / .cjs 文件可降级到 "
                    "stdlib 语法验证。请安装 ruff (`pip install ruff`) 或 eslint，"
                    "或对非代码任务使用 skip_validation 显式说明原因。"
                ),
            )

        if not changed_files:
            # git 报告 clean —— agent 还没动手 / 写到了 worktree 外
            return ValidationResult(
                tool_name="run_linter",
                status=ValidationStatus.ERROR,
                message=(
                    "git 视角下本 worktree 没有任何文件变更（git status 为空）。\n\n"
                    "可能原因：\n"
                    "1. 你还没真正写文件 —— 先 write_file 创建产物，再调用 run_linter。\n"
                    "2. 你写到了 worktree 外的路径 —— 检查 write_file 的 path 参数。\n"
                    "3. 本任务本就不需要产出文件 —— 用 skip_validation 显式说明。"
                ),
            )

        # 有变更，但没一个是代码文件
        non_code_names = [
            f.name for f in changed_files
            if f.suffix in self._COMMON_NON_CODE_SUFFIXES
        ][:3]
        non_code_preview = (
            ", ".join(non_code_names) if non_code_names
            else ", ".join(f.name for f in changed_files[:3])
        )
        return ValidationResult(
            tool_name="run_linter",
            status=ValidationStatus.ERROR,
            message=(
                f"本 worktree 的变更里没有代码文件，只有非代码产出（如 "
                f"{non_code_preview}）。**run_linter 不验证文档/数据/配置**，"
                "它只检查 .py/.js 等源代码的语法。\n\n"
                "如果本任务确实只产出文档/数据（如 .md 报告 / .json 数据），"
                "用 `skip_validation(reason='本任务只产出文档/数据，无代码')` "
                "显式告知门禁；不要再尝试 run_linter。"
            ),
        )

    async def _dispatch_subchecks(
        self,
        py_files: list[Path],
        js_files: list[Path],
    ) -> ValidationResult:
        """跑各语种子检查并组合结果。"""
        sub_results: list[ValidationResult] = []
        if py_files:
            sub_results.append(await self._run_py_compile_check(py_files))
        if js_files:
            sub_results.append(await self._run_node_check(js_files))
        return self._combine_fallback_results(sub_results, py_files, js_files)

    async def _check_explicit_files_in_scope(
        self,
        files: list[str],
    ) -> ValidationResult | None:
        """检查 agent 显式传入的代码文件至少有一个属于本任务的 git 变更。

        防绕过设计：agent 之前可能学到"显式传一个跟本任务无关的 .py（如上一
        个任务遗留的代码）就能拿 PASSED"这种作弊路径，让验证沦为"刷脸时贴
        别人的脸"。本方法在 linter 真正运行前拦下来。

        返回值语义：
          - None  → 通过检查（至少一个代码文件是本任务变更，或 git 不可用兜底）
          - ValidationResult(ERROR) → 阻断，告诉 agent 它在蹭别的代码
        """
        changed = await self._get_changed_files_from_git()
        if changed is None:
            # git 不可用 → 无法判定，放行（保留兜底场景下的可用性）
            return None

        # 提取 agent 传入的代码文件（路径已经过 worktree 沙箱校验）
        requested_code_paths: list[Path] = list(
            self._collect_files_for_fallback(files, frozenset({".py"}))
        )
        requested_code_paths.extend(
            self._collect_files_for_fallback(files, self._JS_FALLBACK_SUFFIXES)
        )

        if not requested_code_paths:
            # agent 传的全是非代码文件（.md / .json 等），不是本规则要拦的场景，
            # 放行让下游 _fallback_explicit_files 给"非代码"诊断
            return None

        changed_set = {p.resolve() for p in changed}
        in_scope = [
            p for p in requested_code_paths if p.resolve() in changed_set
        ]
        if in_scope:
            return None  # 至少一个本任务的文件 → 透传

        out_of_scope_names = sorted({p.name for p in requested_code_paths})[:5]
        changed_code_names = sorted({
            p.name for p in changed
            if p.suffix == ".py" or p.suffix in self._JS_FALLBACK_SUFFIXES
        })[:5]

        suggestion = ""
        if changed_code_names:
            suggestion = (
                f"\n\n本任务实际修改的代码文件是：{changed_code_names}。"
                "请把这些文件传给 run_linter，而不是不相干的文件。"
            )
        else:
            suggestion = (
                "\n\n本 worktree 在 git 视角下没有任何代码文件被修改 —— "
                "如果本任务只产出文档/数据（无代码可验证），说明走 lint "
                "验证就是错的，请回到 master 改派或在 unresolved_issues 里"
                "如实写明任务性质。"
            )

        return ValidationResult(
            tool_name="run_linter",
            status=ValidationStatus.ERROR,
            message=(
                f"传给 run_linter 的代码文件 {out_of_scope_names} 都不在本 "
                "worktree 的 git 变更里 —— 它们不是本任务的产物，可能是上一"
                "任务遗留或上游分支带过来的。\n\n"
                "**run_linter 是用来验证你写的代码**，不是用来蹭别人的代码"
                "拿 PASSED 蒙混门禁。被验证的代码至少要有一个真的是本任务"
                "的产物。"
                f"{suggestion}"
            ),
        )

    async def _get_changed_files_from_git(self) -> list[Path] | None:
        """从 `git status --porcelain` 取本 worktree 的变更文件。

        返回值语义：
          - 正常成功 → 文件列表（可能为空，表示 worktree clean）
          - git 命令本身失败（非 git 目录 / git 不可用）→ None，
            外层据此回落到 rglob 全扫
        """
        if self._worktree_root is None:
            return None
        try:
            process = await asyncio.create_subprocess_exec(
                "git", "status", "--porcelain",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._worktree_root),
            )
            stdout, stderr = await process.communicate()
        except (FileNotFoundError, OSError) as e:
            logger.debug("git 命令不可用，无法取变更清单: %s", e)
            return None

        if (process.returncode or 0) != 0:
            err_text = stderr.decode("utf-8", errors="replace").strip()
            logger.debug("git status 失败 (rc=%s): %s", process.returncode, err_text[:120])
            return None

        out_text = stdout.decode("utf-8", errors="replace")
        root = self._worktree_root.resolve()
        changed: list[Path] = []
        for line in out_text.splitlines():
            if len(line) < 4:
                continue
            # porcelain 格式: "XY <path>"，X/Y 是状态码（M D A R C ? ! U 等）
            status = line[:2]
            path_part = line[3:]
            # 删除的文件磁盘上不存在，跳过
            if "D" in status:
                continue
            # 重命名: "R  old -> new"，取 new
            if "->" in path_part:
                path_part = path_part.split("->")[-1]
            path_part = path_part.strip().strip('"')
            if not path_part:
                continue
            full = (self._worktree_root / path_part).resolve()
            try:
                full.relative_to(root)
            except ValueError:
                continue
            if full.is_file():
                # 同时排除 SKIP 目录，防 agent 在 .venv 里搞事情
                rel_parts = full.relative_to(root).parts
                if any(p in self._FALLBACK_SKIP_DIRS for p in rel_parts):
                    continue
                changed.append(full)
        return changed

    async def _run_py_compile_check(
        self,
        target_files: list[Path],
    ) -> ValidationResult:
        """对一批 .py 文件跑 `python -m py_compile`，返回结构化结果。

        py_compile 行为：
          - 语法 OK → 退出码 0
          - 语法错 → 退出码 1，stderr 输出
              File "x.py", line 5
                def foo(:
                       ^
              SyntaxError: invalid syntax
          - 不会做名字解析、类型检查、风格检查 —— 严格弱于真 linter，
            但"至少能解析"是验证的最低底线。
        """
        cmd = [
            sys.executable, "-m", "py_compile",
            *[str(f) for f in target_files],
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._worktree_root),
            )
            stdout, stderr = await process.communicate()
        except Exception as e:
            return ValidationResult(
                tool_name="run_linter",
                status=ValidationStatus.ERROR,
                message=f"py_compile 执行失败: {e}",
            )

        rc = process.returncode or 0
        if rc == 0:
            return ValidationResult(
                tool_name="run_linter",
                status=ValidationStatus.PASSED,
                message=f"py_compile 检查 {len(target_files)} 个 .py 文件通过",
            )

        stderr_text = stderr.decode("utf-8", errors="replace")
        errors = self._parse_py_compile_stderr(stderr_text)
        if not errors:
            errors = [{
                "message": stderr_text.strip()[:300] or f"py_compile 退出码 {rc}",
            }]

        return ValidationResult(
            tool_name="run_linter",
            status=ValidationStatus.FAILED,
            message=f"py_compile 在 {len(target_files)} 个文件中发现语法错误",
            errors=errors,
        )

    async def _run_node_check(
        self,
        target_files: list[Path],
    ) -> ValidationResult:
        """对一批 .js/.mjs/.cjs 文件跑 `node --check`，返回结构化结果。

        node --check 行为：
          - 单文件：`node --check x.js`，语法 OK → 退出码 0
          - 语法错 → 退出码 1，stderr 输出
              x.js:5
                function foo({
                            ^
              SyntaxError: Unexpected token '{'
          - **不接受多文件参数**，所以本方法逐个文件调用、合并结果

        Node 没装时返回 ERROR：JS 项目没装 Node 视为环境不可验证，由门禁
        拒绝 submit；agent 应当装 Node 或 skip_validation。
        """
        # 先探测 node 可用性，避免逐文件都炸 FileNotFoundError
        try:
            probe = await asyncio.create_subprocess_exec(
                "node", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await probe.communicate()
            if (probe.returncode or 0) != 0:
                raise FileNotFoundError("node --version 非零退出")
        except (FileNotFoundError, OSError):
            return ValidationResult(
                tool_name="run_linter",
                status=ValidationStatus.ERROR,
                message=(
                    f"检测到 {len(target_files)} 个 JS 文件，但环境里既没装 "
                    "eslint 也没装 Node.js，无法做 syntax 验证。请安装 "
                    "Node.js 或 eslint，或用 skip_validation 显式说明。"
                ),
            )

        all_errors: list[dict[str, Any]] = []
        any_failed = False
        any_executed = False

        for f in target_files:
            try:
                process = await asyncio.create_subprocess_exec(
                    "node", "--check", str(f),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self._worktree_root),
                )
                stdout, stderr = await process.communicate()
            except Exception as e:
                all_errors.append({
                    "file": str(f),
                    "message": f"node --check 执行异常: {e}",
                })
                any_failed = True
                continue

            any_executed = True
            rc = process.returncode or 0
            if rc == 0:
                continue
            any_failed = True
            stderr_text = stderr.decode("utf-8", errors="replace")
            parsed = self._parse_node_check_stderr(stderr_text, default_file=str(f))
            if parsed:
                all_errors.extend(parsed)
            else:
                all_errors.append({
                    "file": str(f),
                    "message": stderr_text.strip()[:200] or f"node --check 退出码 {rc}",
                })

        if not any_executed:
            return ValidationResult(
                tool_name="run_linter",
                status=ValidationStatus.ERROR,
                message="node --check 全部执行失败，无法判定语法",
                errors=all_errors,
            )

        if any_failed:
            return ValidationResult(
                tool_name="run_linter",
                status=ValidationStatus.FAILED,
                message=f"node --check 在 {len(target_files)} 个 JS 文件中发现语法错误",
                errors=all_errors,
            )

        return ValidationResult(
            tool_name="run_linter",
            status=ValidationStatus.PASSED,
            message=f"node --check 检查 {len(target_files)} 个 JS 文件通过",
        )

    @staticmethod
    def _combine_fallback_results(
        sub_results: list[ValidationResult],
        py_files: list[Path],
        js_files: list[Path],
    ) -> ValidationResult:
        """合并各语种 fallback 子结果，遵守严格契约。

        优先级：FAILED > ERROR > PASSED。也就是说：
          - 只要有一项 FAILED，整体 FAILED（不放过任何语法错）
          - 没 FAILED 但有 ERROR（如 JS 文件没 node 可用），整体 ERROR
            （门禁会拒，agent 必须解决环境或 skip_validation）
          - 全 PASSED 才整体 PASSED
        """
        statuses = {r.status for r in sub_results}
        all_errors: list[dict[str, Any]] = []
        for r in sub_results:
            all_errors.extend(r.errors)

        summary_bits = []
        if py_files:
            summary_bits.append(f"{len(py_files)} 个 Python")
        if js_files:
            summary_bits.append(f"{len(js_files)} 个 JS")
        scope = " + ".join(summary_bits)

        if ValidationStatus.FAILED in statuses:
            return ValidationResult(
                tool_name="run_linter",
                status=ValidationStatus.FAILED,
                message=(
                    f"stdlib fallback 在 {scope} 文件中发现语法错误"
                    f"（共 {len(all_errors)} 项）。请修复后重新运行 run_linter。"
                ),
                errors=all_errors[:10],
            )

        if ValidationStatus.ERROR in statuses:
            # 把所有 ERROR 的 message 拼起来给 agent 看清楚哪个语种出了问题
            err_messages = "; ".join(
                r.message for r in sub_results
                if r.status == ValidationStatus.ERROR and r.message
            )
            return ValidationResult(
                tool_name="run_linter",
                status=ValidationStatus.ERROR,
                message=err_messages or f"stdlib fallback 在 {scope} 文件验证中发生环境错误",
                errors=all_errors[:10],
            )

        return ValidationResult(
            tool_name="run_linter",
            status=ValidationStatus.PASSED,
            message=(
                f"stdlib fallback 语法检查通过：{scope} 文件全部解析正常。"
                "建议安装 ruff / eslint 获得更全面的检查。"
            ),
        )

    @staticmethod
    def _parse_py_compile_stderr(stderr_text: str) -> list[dict[str, Any]]:
        """从 py_compile 的 stderr 抽取 (file, line, error_type, message)。

        典型块：
            File "x.py", line 5
              def foo(:
                     ^
            SyntaxError: invalid syntax
        """
        import re as _re

        errors: list[dict[str, Any]] = []
        # 匹配 File "...", line N + 后续若干行直到下一个 File 或 EOF
        block_pattern = _re.compile(
            r'File "([^"]+)", line (\d+)\s*\n([\s\S]*?)(?=\n\s*File ".+?", line \d+|\Z)',
            flags=_re.MULTILINE,
        )
        for m in block_pattern.finditer(stderr_text):
            file_path = m.group(1)
            line_no = int(m.group(2))
            body = m.group(3).strip()
            # 取 body 里的错误说明行（含 "Error" 或 "Exception" 关键字的最后一行）
            err_line = ""
            for ln in body.splitlines():
                ls = ln.strip()
                if "Error" in ls or "Exception" in ls:
                    err_line = ls
            if not err_line:
                err_line = body.splitlines()[-1].strip() if body else "py_compile error"

            errors.append({
                "file": file_path,
                "line": line_no,
                "code": err_line.split(":", 1)[0] if ":" in err_line else "SyntaxError",
                "message": err_line[:200],
            })
        return errors

    @staticmethod
    def _parse_node_check_stderr(
        stderr_text: str,
        default_file: str = "",
    ) -> list[dict[str, Any]]:
        """从 node --check 的 stderr 抽取 (file, line, message)。

        node 输出形如：
            /path/to/x.js:5
              function foo({
                          ^^^

            SyntaxError: Unexpected token '{'
                at wrapSafe (node:internal/modules/cjs/loader:1283:18)
                ...

        我们抓第一行 `<path>:<line>` 锚点 + 错误说明行（含 SyntaxError 等关键字）。
        """
        import re as _re

        errors: list[dict[str, Any]] = []
        text = stderr_text.strip()
        if not text:
            return errors

        # 第一行通常是 "<path>:<line>"
        loc_match = _re.match(r"(.+?):(\d+)\s*$", text.splitlines()[0]) if text else None
        file_path = loc_match.group(1) if loc_match else default_file
        line_no = int(loc_match.group(2)) if loc_match else 0

        err_line = ""
        for ln in text.splitlines():
            ls = ln.strip()
            if any(k in ls for k in ("SyntaxError", "ReferenceError", "TypeError", "Error:")):
                err_line = ls
                break
        if not err_line:
            err_line = text.splitlines()[-1].strip()[:200]

        errors.append({
            "file": file_path,
            "line": line_no,
            "code": err_line.split(":", 1)[0] if ":" in err_line else "SyntaxError",
            "message": err_line[:200],
        })
        return errors

    def _build_linter_command(
        self,
        linter: str,
        files: list[str],
        fix: bool,
    ) -> list[str]:
        """构建 linter 命令"""
        cmd = [linter]

        if linter == "ruff":
            cmd.extend(["check", "--output-format=json"])
            if fix:
                cmd.append("--fix")
        elif linter == "flake8":
            cmd.extend(["--format=default"])
        elif linter == "eslint":
            cmd.extend(["--format=json"])
            if fix:
                cmd.append("--fix")

        if files:
            cmd.extend(files)
        else:
            if linter in ["ruff", "flake8"]:
                cmd.append(".")
            elif linter == "eslint":
                cmd.append(".")

        return cmd

    def _parse_linter_output(
        self,
        linter: str,
        exit_code: int,
        stdout: str,
        stderr: str,
    ) -> ValidationResult:
        """解析 linter 输出"""
        errors = []
        warnings = []

        if linter == "ruff":
            try:
                data = json.loads(stdout) if stdout.strip() else []
                for item in data:
                    severity = "error" if item.get("severity") == "error" else "warning"
                    issue = {
                        "file": item.get("filename", ""),
                        "line": item.get("location", {}).get("row", 0),
                        "column": item.get("location", {}).get("column", 0),
                        "message": item.get("message", ""),
                        "code": item.get("code", ""),
                    }
                    if severity == "error":
                        errors.append(issue)
                    else:
                        warnings.append(issue)
            except json.JSONDecodeError:
                # 非 JSON 输出，按行解析
                for line in stdout.strip().split("\n"):
                    if line.strip():
                        errors.append({"message": line})

        elif linter == "flake8":
            for line in stdout.strip().split("\n"):
                if not line.strip():
                    continue
                # 格式: file:line:col: code message
                import re
                match = re.match(r"(.+):(\d+):(\d+): ([A-Z]\d+) (.+)", line)
                if match:
                    errors.append({
                        "file": match.group(1),
                        "line": int(match.group(2)),
                        "column": int(match.group(3)),
                        "code": match.group(4),
                        "message": match.group(5),
                    })

        elif linter == "eslint":
            try:
                data = json.loads(stdout) if stdout.strip() else []
                for file_result in data:
                    for msg in file_result.get("messages", []):
                        severity = "error" if msg.get("severity") == 2 else "warning"
                        issue = {
                            "file": file_result.get("filePath", ""),
                            "line": msg.get("line", 0),
                            "column": msg.get("column", 0),
                            "message": msg.get("message", ""),
                            "rule": msg.get("ruleId", ""),
                        }
                        if severity == "error":
                            errors.append(issue)
                        else:
                            warnings.append(issue)
            except json.JSONDecodeError:
                pass

        status = ValidationStatus.PASSED if len(errors) == 0 else ValidationStatus.FAILED
        message = f"发现 {len(errors)} 个错误, {len(warnings)} 个警告" if errors else "检查通过"

        return ValidationResult(
            tool_name="run_linter",
            status=status,
            message=message,
            errors=errors,
            warnings=warnings,
        )


class RunTestsTool(Tool):
    """测试工具 — 运行测试套件

    支持:
      - Python: pytest
      - JavaScript: jest, npm test
    """

    def __init__(self, worktree_root: Path | None = None):
        self._worktree_root = worktree_root

    def set_worktree_root(self, root: Path) -> None:
        """设置工作目录"""
        self._worktree_root = root

    @property
    def name(self) -> str:
        return "run_tests"

    @property
    def description(self) -> str:
        return """运行测试套件验证代码正确性。

**重要**: 这是验证工具。在调用 submit_task_result 声明成功前，必须至少通过一次验证。

支持:
- Python: pytest
- JavaScript: jest

返回:
- 通过: 所有测试通过
- 失败: 包含失败的测试用例"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "test_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要运行的测试文件（空=运行所有）",
                },
                "runner": {
                    "type": "string",
                    "enum": ["auto", "pytest", "jest", "npm"],
                    "description": "指定测试运行器，默认 auto 自动检测",
                    "default": "auto",
                },
                "coverage": {
                    "type": "boolean",
                    "description": "是否生成覆盖率报告",
                    "default": False,
                },
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        if self._worktree_root is None:
            return "错误: 工作目录未设置"

        test_files = kwargs.get("test_files", [])
        runner = kwargs.get("runner", "auto")
        coverage = kwargs.get("coverage", False)

        result = await self._run_tests(test_files, runner, coverage)

        return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)

    async def _run_tests(
        self,
        test_files: list[str],
        runner: str,
        coverage: bool,
    ) -> ValidationResult:
        """运行测试"""
        import time

        start_time = time.time()

        if runner == "auto":
            runner = self._detect_test_runner()

        if runner is None:
            return ValidationResult(
                tool_name="run_tests",
                status=ValidationStatus.ERROR,
                message="未找到测试运行器，请安装 pytest 或 jest",
            )

        cmd = self._build_test_command(runner, test_files, coverage)

        # 禁用 .pyc 生成：worktree 内 pytest 产生的 __pycache__ 会和主仓库的
        # __pycache__ 冲突，导致 git merge 拒绝覆盖未跟踪文件。从源头不生成。
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._worktree_root),
                env=env,
            )

            stdout, stderr = await process.communicate()
            execution_time = int((time.time() - start_time) * 1000)

            result = self._parse_test_output(
                runner=runner,
                exit_code=process.returncode or 0,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
            )

            result.execution_time_ms = execution_time

            return result

        except FileNotFoundError:
            return ValidationResult(
                tool_name="run_tests",
                status=ValidationStatus.ERROR,
                message=f"测试运行器 '{runner}' 未安装",
            )
        except Exception as e:
            return ValidationResult(
                tool_name="run_tests",
                status=ValidationStatus.ERROR,
                message=f"执行错误: {str(e)}",
            )

    def _detect_test_runner(self) -> str | None:
        """检测测试运行器"""
        # 检查 pytest
        try:
            subprocess.run(
                ["pytest", "--version"],
                capture_output=True,
                check=True,
            )
            return "pytest"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # 检查 jest
        try:
            subprocess.run(
                ["jest", "--version"],
                capture_output=True,
                check=True,
            )
            return "jest"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return None

    def _build_test_command(
        self,
        runner: str,
        test_files: list[str],
        coverage: bool,
    ) -> list[str]:
        """构建测试命令"""
        if runner == "pytest":
            cmd = ["pytest", "-v", "--tb=short"]
            if coverage:
                cmd.extend(["--cov=.", "--cov-report=term-missing"])
        elif runner == "jest":
            cmd = ["jest", "--verbose"]
            if coverage:
                cmd.append("--coverage")
        elif runner == "npm":
            cmd = ["npm", "test"]
        else:
            cmd = [runner]

        if test_files:
            cmd.extend(test_files)

        return cmd

    def _parse_test_output(
        self,
        runner: str,
        exit_code: int,
        stdout: str,
        stderr: str,
    ) -> ValidationResult:
        """解析测试输出

        语义优先级（决定最终 status）:
          1. pytest exit_code == 5 (no tests collected) → ERROR
             "没有测试可跑" 不是失败，而是"无法验证"。
          2. exit_code == 0 → PASSED
          3. exit_code != 0 → FAILED

        errors 字段填充原则：每条 = 一个失败用例，含 test 名 + reason 摘要。
        agent 拿到结构化 errors 就不需要回头跑 terminal pytest -v 自己看 verbose。
        """
        import re

        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        message: str
        status: ValidationStatus

        # 先剥 ANSI（pytest-rich / Windows 控制台经常注入颜色码，会污染锚点匹配）
        clean_stdout = _strip_ansi(stdout)

        if runner == "pytest":
            # exit_code 5 = "no tests ran"，区别于 1/2/3/4 的真失败
            if exit_code == 5:
                return ValidationResult(
                    tool_name="run_tests",
                    status=ValidationStatus.ERROR,
                    message="未发现可执行的测试用例（pytest exit_code=5）",
                    errors=[],
                    warnings=[],
                )

            errors = _extract_pytest_failures(clean_stdout)

            summary_match = re.search(r"(\d+) failed", clean_stdout)
            passed_match = re.search(r"(\d+) passed", clean_stdout)

            if exit_code == 0:
                status = ValidationStatus.PASSED
                message = (
                    f"{passed_match.group(1)} 个测试通过"
                    if passed_match else "测试通过"
                )
            else:
                status = ValidationStatus.FAILED
                if summary_match:
                    message = f"{summary_match.group(1)} 个测试失败"
                elif stderr.strip():
                    message = f"pytest 退出码 {exit_code}: {stderr.strip()[:120]}"
                else:
                    message = f"pytest 退出码 {exit_code}（未识别失败列表）"

                # 失败但 errors 为空时，把 message 自身作为占位，
                # 至少让下游 agent 不会在 errors 字段空 list 上做判断
                if not errors:
                    errors.append({
                        "test": "(未识别)",
                        "message": (
                            f"pytest 报告 {summary_match.group(1) if summary_match else '?'} 个失败但未能解析具体用例。"
                            "可在 message 字段查看摘要。"
                        ),
                    })

        elif runner == "jest":
            errors = _extract_jest_failures(clean_stdout)

            summary_match = re.search(r"Tests:\s+(\d+) failed", clean_stdout)
            passed_match = re.search(r"Tests:.*?(\d+) passed", clean_stdout)

            if exit_code == 0:
                status = ValidationStatus.PASSED
                message = (
                    f"{passed_match.group(1)} 个测试通过"
                    if passed_match else "测试通过"
                )
            else:
                status = ValidationStatus.FAILED
                message = (
                    f"{summary_match.group(1)} 个测试失败"
                    if summary_match else f"jest 退出码 {exit_code}"
                )

        else:
            if exit_code == 0:
                status = ValidationStatus.PASSED
                message = stdout[:200] if stdout else "测试完成"
            else:
                status = ValidationStatus.FAILED
                message = stderr[:200] if stderr else f"测试失败 (退出码 {exit_code})"
                errors.append({"message": message})

        return ValidationResult(
            tool_name="run_tests",
            status=status,
            message=message,
            errors=errors[:10],  # 不让超长失败列表撑爆下游上下文
            warnings=warnings,
        )


_ANSI_ESCAPE_RE = __import__("re").compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    """去除 ANSI 控制序列（颜色 / 光标移动）。

    pytest-rich、Windows Terminal、CI 上的 pytest 输出经常含 ESC[31m 等颜色码；
    它们会让基于 ^/$ 锚点的正则失效（颜色码可能出现在期望位置之前）。
    剥掉后续解析就能稳定命中。
    """
    if not text:
        return text
    return _ANSI_ESCAPE_RE.sub("", text)


def _extract_pytest_failures(stdout: str) -> list[dict[str, Any]]:
    """从 pytest 输出中提取失败用例的结构化列表。

    解析策略（优先级递降）：
      1. 优先用 "short test summary info" 段后的 "FAILED <id> - <reason>" 行；
         pytest 的标准摘要永远在尾部，最稳定的锚点
      2. 没有摘要段时（运行被中断 / 极简输出），退回扫全文 verbose 行
         "<id> FAILED" 拿用例名（reason 可能丢失，标 "(无 reason)"）
      3. 同一个 test_id 不重复加入

    返回值 schema 与 ValidationResult.errors 兼容：
      [{"test": "<test_id>", "message": "<reason 截到 200 字节>"}]
    """
    import re as _re

    failures: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # 定位摘要段
    summary_marker = _re.search(
        r"^=+\s*short test summary info\s*=+\s*$",
        stdout,
        flags=_re.MULTILINE,
    )
    summary_text = stdout[summary_marker.end():] if summary_marker else ""

    # 摘要段里 "FAILED <test_id> - <reason>"（reason 可空）
    summary_pattern = _re.compile(
        r"^FAILED\s+(\S+)(?:\s+-\s+(.+))?$",
        flags=_re.MULTILINE,
    )
    for m in summary_pattern.finditer(summary_text):
        test_id = m.group(1).strip()
        if test_id in seen_ids:
            continue
        seen_ids.add(test_id)
        reason = (m.group(2) or "").strip()
        failures.append({
            "test": test_id,
            "message": reason[:200] if reason else "(摘要中未给出 reason)",
        })

    # 摘要段为空 → 扫全文 verbose 行 "<test_id> FAILED"
    if not failures:
        verbose_pattern = _re.compile(
            r"^(\S+::\S+)\s+FAILED",
            flags=_re.MULTILINE,
        )
        for m in verbose_pattern.finditer(stdout):
            test_id = m.group(1).strip()
            if test_id in seen_ids:
                continue
            seen_ids.add(test_id)
            failures.append({
                "test": test_id,
                "message": "(verbose 模式未输出 reason，可在 stdout 全文中查看)",
            })

    # 找到失败用例后，尝试在 ___ test_name ___ 分隔块下补充 reason 片段
    # 仅当摘要给的 reason "看起来没信息量"才补（占位文本 / 极短 / 不含错误关键词），
    #  避免覆盖摘要已经提供的好 reason（如 "AssertionError: assert x == y"）。
    _good_reason_keywords = (
        "Error", "Exception", "Assertion", "Timeout", "TypeError", "ValueError",
    )

    def _reason_is_informative(msg: str) -> bool:
        if len(msg) >= 40:
            return True
        return any(kw in msg for kw in _good_reason_keywords)

    if failures:
        for failure in failures:
            if _reason_is_informative(failure["message"]):
                continue  # 摘要 reason 已够，不要覆盖
            test_id = failure["test"]
            # pytest 的"详细块"格式: ____ ClassName.method ____  或  ____ test_func ____
            # test_id 可能是 "test_x.py::TestClass::test_method" 形式
            # 取末段（method 名）做匹配
            short_name = test_id.split("::")[-1]
            block_re = _re.compile(
                rf"^_+\s*(?:\S+\.)?{_re.escape(short_name)}\s*_+\s*$([\s\S]*?)(?=^_{{3,}}|\Z)",
                flags=_re.MULTILINE,
            )
            block_match = block_re.search(stdout)
            if not block_match:
                continue
            block_body = block_match.group(1)
            # 取含 assert / Error / Exception 关键词的第一行
            for line in block_body.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if any(k in stripped for k in ("assert", "Error:", "Exception:", "AssertionError")):
                    if len(stripped) > 10:  # 避免抓到一个无信息空壳行
                        failure["message"] = stripped[:200]
                        break

    return failures


def _extract_jest_failures(stdout: str) -> list[dict[str, Any]]:
    """从 jest 输出提取失败用例。

    jest 输出常见两种锚点：
      1. `FAIL <path-to-test-file>` —— 文件级失败（顶部）
      2. `  ● <describe block> > <test name>` —— 用例级失败（在每个失败用例上方）
    我们优先抓 ● 行（更细粒度），fallback 到 FAIL 行。
    """
    import re as _re

    failures: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    bullet_pattern = _re.compile(r"^\s*●\s+(.+?)\s*$", flags=_re.MULTILINE)
    for m in bullet_pattern.finditer(stdout):
        test_id = m.group(1).strip()
        if test_id in seen_ids or not test_id:
            continue
        seen_ids.add(test_id)
        failures.append({
            "test": test_id,
            "message": "测试失败",
        })

    if not failures:
        file_pattern = _re.compile(r"^FAIL\s+(\S+)\s*$", flags=_re.MULTILINE)
        for m in file_pattern.finditer(stdout):
            test_id = m.group(1).strip()
            if test_id in seen_ids:
                continue
            seen_ids.add(test_id)
            failures.append({
                "test": test_id,
                "message": "测试失败",
            })

    return failures


# 导出
__all__ = [
    "ValidationStatus",
    "ValidationResult",
    "RunLinterTool",
    "RunTestsTool",
]