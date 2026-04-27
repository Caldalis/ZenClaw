"""
Turn 快照与 Agent 执行状态模型

用于多 Agent 架构的状态持久化和断点恢复。

核心概念:
  - Turn: Agent 的一次完整执行周期（从接收输入到产出结果）
  - TurnSnapshot: Turn 的状态快照，用于持久化和恢复
  - AgentNode: Agent 在任务树中的节点信息（包含层级关系）
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from miniclaw.types.enums import AgentRole, TurnStatus


class AgentNode(BaseModel):
    """Agent 在任务树中的节点信息

    记录 Agent 的层级关系和上下文隔离信息。
    """
    agent_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    role: AgentRole = AgentRole.GENERIC
    depth: int = 0                     # 在树中的深度（Master=0）
    parent_id: str | None = None       # 父节点 Agent ID
    session_id: str                    # 会话 ID（上下文隔离）
    worktree_path: str | None = None   # Git Worktree 路径（物理隔离）

    # 子节点信息
    children_ids: list[str] = Field(default_factory=list)
    max_children: int = 3              # 最大允许子节点数

    # 执行边界
    max_steps: int = 15                # 最大 ReAct 步数
    timeout_ms: int = 120000           # 超时时间（毫秒）

    # 完成状态：spawn_agent 用此字段判断子节点是否已结束。
    # 不依赖 DAG.completed_nodes/failed_nodes，因为 master_node 跨 DAG 复用时
    # 旧子节点不在新 DAG 的状态列表里，会被误判成"在跑"。
    is_finished: bool = False


class TurnSnapshot(BaseModel):
    """Turn 状态快照

    持久化记录 Agent 的每一次 Turn 状态，用于断点恢复。
    """
    turn_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    node: AgentNode                     # 执行该 Turn 的 Agent 信息

    # 输入输出
    input_message: str                  # 输入消息（用户或父 Agent）
    output_result: str | None = None    # 输出结果（成功时）
    output_error: str | None = None     # 错误信息（失败时）
    is_error: bool = False              # 是否为错误结果

    # 执行状态
    status: TurnStatus = TurnStatus.PENDING
    step_count: int = 0                 # 已执行的 ReAct 步数
    current_step: int = 0               # 当前正在执行的步骤

    # 时间信息
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # 中间状态（用于恢复）
    context_snapshot: dict[str, Any] = Field(default_factory=dict)  # 上下文快照
    tool_calls_pending: list[dict] = Field(default_factory=list)    # 待执行的工具调用

    @property
    def duration_ms(self) -> int | None:
        """执行耗时（毫秒）"""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None

    def mark_running(self) -> None:
        """标记为运行状态"""
        self.status = TurnStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def mark_completed(self, result: str) -> None:
        """标记为完成状态"""
        self.status = TurnStatus.COMPLETED
        self.output_result = result
        self.is_error = False
        self.completed_at = datetime.now(timezone.utc)

    def mark_failed(self, error: str, is_timeout: bool = False) -> None:
        """标记为失败状态"""
        self.status = TurnStatus.TIMEOUT if is_timeout else TurnStatus.FAILED
        self.output_error = error
        self.is_error = True
        self.completed_at = datetime.now(timezone.utc)

    def mark_interrupted(self) -> None:
        """标记为中断状态（进程崩溃时）"""
        self.status = TurnStatus.INTERRUPTED
        self.completed_at = datetime.now(timezone.utc)


class TaskDAG(BaseModel):
    """任务 DAG 结构

    Master Agent 构建的任务分解图。
    """
    dag_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    root_node_id: str                   # 根节点（Master Agent）
    nodes: dict[str, AgentNode] = Field(default_factory=dict)  # 所有节点
    edges: list[tuple[str, str]] = Field(default_factory=list)  # 边 (from_id, to_id)

    # 状态追踪
    pending_nodes: list[str] = Field(default_factory=list)      # 待执行节点
    running_nodes: list[str] = Field(default_factory=list)      # 正在执行节点
    completed_nodes: list[str] = Field(default_factory=list)    # 已完成节点
    failed_nodes: list[str] = Field(default_factory=list)       # 失败节点

    @property
    def is_complete(self) -> bool:
        """DAG 是否全部完成"""
        return len(self.pending_nodes) == 0 and len(self.running_nodes) == 0

    @property
    def last_incomplete_leaf(self) -> str | None:
        """最后一个未完成的叶子节点（用于断点恢复）"""
        # 叶子节点 = 没有子节点的节点
        leaf_nodes = [
            nid for nid, node in self.nodes.items()
            if len(node.children_ids) == 0
        ]
        # 找到第一个 pending 或 running 的叶子节点
        for leaf in leaf_nodes:
            if leaf in self.pending_nodes or leaf in self.running_nodes:
                return leaf
        return None