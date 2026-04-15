"""
Worktree 模块 — Git Worktree 物理隔离

提供:
  - WorktreeManager: Git worktree 生命周期管理
  - WorkspaceIsolator: Subagent 工作空间隔离
  - IsolatedToolSet: 受限的工具集
"""

from miniclaw.worktree.manager import (
    WorktreeInfo,
    WorktreeManager,
    WorktreeStatus,
)
from miniclaw.worktree.isolator import (
    IsolatedWorkspace,
    WorkspaceIsolator,
    create_workspace_isolator,
)
from miniclaw.worktree.isolated_tools import (
    IsolatedFileReaderTool,
    IsolatedFileWriterTool,
    IsolatedTerminalTool,
    IsolatedToolSet,
    PathSecurityError,
    validate_path,
)

__all__ = [
    # Manager
    "WorktreeManager",
    "WorktreeInfo",
    "WorktreeStatus",
    # Isolator
    "WorkspaceIsolator",
    "IsolatedWorkspace",
    "create_workspace_isolator",
    # Tools
    "IsolatedToolSet",
    "IsolatedFileReaderTool",
    "IsolatedFileWriterTool",
    "IsolatedTerminalTool",
    "PathSecurityError",
    "validate_path",
]