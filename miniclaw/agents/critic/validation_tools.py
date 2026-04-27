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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from miniclaw.tools.base import Tool
from miniclaw.utils.logging import get_logger

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
- Python: ruff, flake8, pylint
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

        # 确定要使用的 linter
        if linter == "auto":
            linter = self._detect_linter()

        if linter is None:
            return ValidationResult(
                tool_name="run_linter",
                status=ValidationStatus.ERROR,
                message="未找到可用的 linter，请安装 ruff 或 flake8",
            )

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
        """
        import re

        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        message: str
        status: ValidationStatus

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

            # 失败用例
            failed_pattern = r"FAILED (.*?) - (.*?)(?:\n|$)"
            for match in re.finditer(failed_pattern, stdout):
                errors.append({
                    "test": match.group(1).strip(),
                    "message": match.group(2).strip()[:200],
                })

            summary_match = re.search(r"(\d+) failed", stdout)
            passed_match = re.search(r"(\d+) passed", stdout)

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

        elif runner == "jest":
            failed_pattern = r"FAIL (.*?)(?:\n|$)"
            for match in re.finditer(failed_pattern, stdout):
                errors.append({
                    "test": match.group(1).strip(),
                    "message": "测试失败",
                })

            summary_match = re.search(r"Tests:\s+(\d+) failed", stdout)
            passed_match = re.search(r"Tests:.*?(\d+) passed", stdout)

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
            errors=errors,
            warnings=warnings,
        )


# 导出
__all__ = [
    "ValidationStatus",
    "ValidationResult",
    "RunLinterTool",
    "RunTestsTool",
]