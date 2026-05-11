"""
Worktree 模块 — Git Worktree 物理隔离

提供:
  - WorktreeManager: Git worktree 生命周期管理
  - WorkspaceIsolator: Subagent 工作空间隔离
  - IsolatedToolSet: 受限的工具集
"""

from taskweave.worktree.manager import (
    WorktreeInfo,
    WorktreeManager,
    WorktreeStatus,
)
from taskweave.worktree.isolator import (
    IsolatedWorkspace,
    WorkspaceIsolator,
    create_workspace_isolator,
)
from taskweave.worktree.isolated_tools import (
    IsolatedReadFileTool,
    IsolatedWriteFileTool,
    IsolatedEditFileTool,
    IsolatedLsTool,
    IsolatedGlobTool,
    IsolatedGrepTool,
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
    "IsolatedReadFileTool",
    "IsolatedWriteFileTool",
    "IsolatedEditFileTool",
    "IsolatedLsTool",
    "IsolatedGlobTool",
    "IsolatedGrepTool",
    "IsolatedTerminalTool",
    "PathSecurityError",
    "validate_path",
]