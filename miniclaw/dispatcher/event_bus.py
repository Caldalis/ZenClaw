"""
Event Bus - 状态持久化与断点恢复

核心职责:
  1. 持久化记录每一个 Agent 的每一次 Turn 状态
  2. 提供断点恢复机制：从最后一个未完成的叶子节点恢复执行
  3. 支持 SQLite 或 JSONL 双模式存储

EventBus 是多 Agent 架构的"黑匣子"，记录所有执行轨迹。

存储结构:
  - turns 表: 记录每个 Turn 的状态快照
  - dag_snapshots 表: 记录 DAG 整体状态
  - event_log 表: 记录所有事件（可选，用于调试）
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from miniclaw.types.enums import TurnStatus
from miniclaw.types.turn_snapshot import AgentNode, TaskDAG, TurnSnapshot
from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


class TurnLogStore:
    """Turn 日志持久化存储

    使用 SQLite 存储所有 Turn 状态快照。
    """

    def __init__(self, db_path: str = "data/turns.db"):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """初始化数据库表"""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")

        # Turn 状态表
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                turn_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                role TEXT NOT NULL,
                depth INTEGER NOT NULL,
                parent_id TEXT,
                session_id TEXT NOT NULL,
                worktree_path TEXT,
                input_message TEXT NOT NULL,
                output_result TEXT,
                output_error TEXT,
                is_error INTEGER DEFAULT 0,
                status TEXT NOT NULL,
                step_count INTEGER DEFAULT 0,
                current_step INTEGER DEFAULT 0,
                context_snapshot TEXT,
                tool_calls_pending TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                duration_ms INTEGER
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_turns_agent ON turns(agent_id)
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_turns_status ON turns(status)
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id)
        """)

        # DAG 快照表
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS dag_snapshots (
                dag_id TEXT PRIMARY KEY,
                root_node_id TEXT NOT NULL,
                nodes_json TEXT NOT NULL,
                edges_json TEXT NOT NULL,
                pending_nodes_json TEXT,
                running_nodes_json TEXT,
                completed_nodes_json TEXT,
                failed_nodes_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_dag_root ON dag_snapshots(root_node_id)
        """)

        # 事件日志表（可选，用于调试）
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS event_log (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_turn ON event_log(turn_id)
        """)

        await self._db.commit()
        logger.info("Turn 日志存储已初始化: %s", self._db_path)

    async def save_turn(self, snapshot: TurnSnapshot) -> None:
        """保存 Turn 状态快照"""
        assert self._db is not None

        context_json = json.dumps(snapshot.context_snapshot, ensure_ascii=False) if snapshot.context_snapshot else None
        tool_calls_json = json.dumps(snapshot.tool_calls_pending, ensure_ascii=False) if snapshot.tool_calls_pending else None

        await self._db.execute("""
            INSERT OR REPLACE INTO turns (
                turn_id, agent_id, role, depth, parent_id, session_id, worktree_path,
                input_message, output_result, output_error, is_error,
                status, step_count, current_step,
                context_snapshot, tool_calls_pending,
                created_at, started_at, completed_at, duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot.turn_id,
            snapshot.node.agent_id,
            snapshot.node.role.value,
            snapshot.node.depth,
            snapshot.node.parent_id,
            snapshot.node.session_id,
            snapshot.node.worktree_path,
            snapshot.input_message,
            snapshot.output_result,
            snapshot.output_error,
            1 if snapshot.is_error else 0,
            snapshot.status.value,
            snapshot.step_count,
            snapshot.current_step,
            context_json,
            tool_calls_json,
            snapshot.created_at.isoformat(),
            snapshot.started_at.isoformat() if snapshot.started_at else None,
            snapshot.completed_at.isoformat() if snapshot.completed_at else None,
            snapshot.duration_ms,
        ))
        await self._db.commit()
        logger.debug("Turn 快照已保存: %s (status=%s)", snapshot.turn_id, snapshot.status.value)

    async def get_turn(self, turn_id: str) -> TurnSnapshot | None:
        """获取单个 Turn 快照"""
        assert self._db is not None

        cursor = await self._db.execute("""
            SELECT turn_id, agent_id, role, depth, parent_id, session_id, worktree_path,
                   input_message, output_result, output_error, is_error,
                   status, step_count, current_step,
                   context_snapshot, tool_calls_pending,
                   created_at, started_at, completed_at, duration_ms
            FROM turns WHERE turn_id = ?
        """, (turn_id,))
        row = await cursor.fetchone()
        if row is None:
            return None

        return self._row_to_snapshot(row)

    async def get_turns_by_agent(self, agent_id: str) -> list[TurnSnapshot]:
        """获取指定 Agent 的所有 Turn"""
        assert self._db is not None

        cursor = await self._db.execute("""
            SELECT turn_id, agent_id, role, depth, parent_id, session_id, worktree_path,
                   input_message, output_result, output_error, is_error,
                   status, step_count, current_step,
                   context_snapshot, tool_calls_pending,
                   created_at, started_at, completed_at, duration_ms
            FROM turns WHERE agent_id = ? ORDER BY created_at
        """, (agent_id,))
        rows = await cursor.fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    async def get_interrupted_turns(self, session_id: str | None = None) -> list[TurnSnapshot]:
        """获取所有中断状态的 Turn（用于断点恢复）

        Args:
            session_id: 可选，限定在指定会话内
        """
        assert self._db is not None

        if session_id:
            cursor = await self._db.execute("""
                SELECT turn_id, agent_id, role, depth, parent_id, session_id, worktree_path,
                       input_message, output_result, output_error, is_error,
                       status, step_count, current_step,
                       context_snapshot, tool_calls_pending,
                       created_at, started_at, completed_at, duration_ms
                FROM turns WHERE session_id = ? AND status IN ('pending', 'running', 'interrupted')
                ORDER BY depth DESC, created_at DESC
            """, (session_id,))
        else:
            cursor = await self._db.execute("""
                SELECT turn_id, agent_id, role, depth, parent_id, session_id, worktree_path,
                       input_message, output_result, output_error, is_error,
                       status, step_count, current_step,
                       context_snapshot, tool_calls_pending,
                       created_at, started_at, completed_at, duration_ms
                FROM turns WHERE status IN ('pending', 'running', 'interrupted')
                ORDER BY depth DESC, created_at DESC
            """)
        rows = await cursor.fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    async def get_last_incomplete_leaf(self, dag_id: str) -> TurnSnapshot | None:
        """获取 DAG 中最后一个未完成的叶子节点

        用于断点恢复。叶子节点 = depth 最大且没有子节点的节点。
        """
        assert self._db is not None

        # 先获取 DAG 快照
        dag = await self.get_dag(dag_id)
        if dag is None:
            return None

        leaf_id = dag.last_incomplete_leaf
        if leaf_id is None:
            return None

        # 获取该叶子的 Turn
        cursor = await self._db.execute("""
            SELECT turn_id, agent_id, role, depth, parent_id, session_id, worktree_path,
                   input_message, output_result, output_error, is_error,
                   status, step_count, current_step,
                   context_snapshot, tool_calls_pending,
                   created_at, started_at, completed_at, duration_ms
            FROM turns WHERE agent_id = ? AND status IN ('pending', 'running', 'interrupted')
            ORDER BY created_at DESC LIMIT 1
        """, (leaf_id,))
        row = await cursor.fetchone()
        if row is None:
            return None

        return self._row_to_snapshot(row)

    async def save_dag(self, dag: TaskDAG) -> None:
        """保存 DAG 快照"""
        assert self._db is not None

        nodes_json = json.dumps(
            {k: v.model_dump() for k, v in dag.nodes.items()},
            ensure_ascii=False
        )
        edges_json = json.dumps(dag.edges, ensure_ascii=False)
        pending_json = json.dumps(dag.pending_nodes, ensure_ascii=False)
        running_json = json.dumps(dag.running_nodes, ensure_ascii=False)
        completed_json = json.dumps(dag.completed_nodes, ensure_ascii=False)
        failed_json = json.dumps(dag.failed_nodes, ensure_ascii=False)

        await self._db.execute("""
            INSERT OR REPLACE INTO dag_snapshots (
                dag_id, root_node_id, nodes_json, edges_json,
                pending_nodes_json, running_nodes_json, completed_nodes_json, failed_nodes_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dag.dag_id,
            dag.root_node_id,
            nodes_json,
            edges_json,
            pending_json,
            running_json,
            completed_json,
            failed_json,
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat(),
        ))
        await self._db.commit()
        logger.debug("DAG 快照已保存: %s", dag.dag_id)

    async def get_dag(self, dag_id: str) -> TaskDAG | None:
        """获取 DAG 快照"""
        assert self._db is not None

        cursor = await self._db.execute("""
            SELECT dag_id, root_node_id, nodes_json, edges_json,
                   pending_nodes_json, running_nodes_json, completed_nodes_json, failed_nodes_json
            FROM dag_snapshots WHERE dag_id = ?
        """, (dag_id,))
        row = await cursor.fetchone()
        if row is None:
            return None

        nodes_data = json.loads(row[2])
        nodes = {k: AgentNode(**v) for k, v in nodes_data.items()}
        edges = json.loads(row[3])
        pending = json.loads(row[4]) if row[4] else []
        running = json.loads(row[5]) if row[5] else []
        completed = json.loads(row[6]) if row[6] else []
        failed = json.loads(row[7]) if row[7] else []

        dag = TaskDAG(
            dag_id=row[0],
            root_node_id=row[1],
            nodes=nodes,
            edges=edges,
            pending_nodes=pending,
            running_nodes=running,
            completed_nodes=completed,
            failed_nodes=failed,
        )
        return dag

    async def log_event(self, turn_id: str, event_type: str, event_data: dict[str, Any] = {}) -> None:
        """记录事件日志（用于调试）"""
        assert self._db is not None

        await self._db.execute("""
            INSERT INTO event_log (turn_id, event_type, event_data, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            turn_id,
            event_type,
            json.dumps(event_data, ensure_ascii=False),
            datetime.now(timezone.utc).isoformat(),
        ))
        await self._db.commit()

    async def cleanup_completed_turns(self, older_than_days: int = 7) -> int:
        """清理已完成的旧 Turn 记录"""
        assert self._db is not None

        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        cursor = await self._db.execute("""
            DELETE FROM turns WHERE status IN ('completed', 'failed', 'timeout')
            AND completed_at < ?
        """, (cutoff.isoformat(),))
        deleted = cursor.rowcount
        await self._db.commit()
        logger.info("清理了 %d 条旧 Turn 记录", deleted)
        return deleted

    async def close(self) -> None:
        """关闭数据库连接"""
        if self._db:
            await self._db.close()
            self._db = None

    def _row_to_snapshot(self, row: tuple) -> TurnSnapshot:
        """将数据库行转换为 TurnSnapshot"""
        from miniclaw.types.enums import AgentRole

        node = AgentNode(
            agent_id=row[1],
            role=AgentRole(row[2]),
            depth=row[3],
            parent_id=row[4],
            session_id=row[5],
            worktree_path=row[6],
        )

        snapshot = TurnSnapshot(
            turn_id=row[0],
            node=node,
            input_message=row[7],
            output_result=row[8],
            output_error=row[9],
            is_error=bool(row[10]),
            status=TurnStatus(row[11]),
            step_count=row[12],
            current_step=row[13],
            context_snapshot=json.loads(row[14]) if row[14] else {},
            tool_calls_pending=json.loads(row[15]) if row[15] else [],
            created_at=datetime.fromisoformat(row[16]),
            started_at=datetime.fromisoformat(row[17]) if row[17] else None,
            completed_at=datetime.fromisoformat(row[18]) if row[18] else None,
        )
        return snapshot


class EventBus:
    """事件总线 — 管理 Turn 状态和事件分发

    核心功能:
      1. Turn 状态持久化（通过 TurnLogStore）
      2. 断点恢复支持
      3. 事件订阅/分发机制
    """

    def __init__(self, store: TurnLogStore):
        self._store = store
        self._subscribers: dict[str, list] = {}  # event_type -> [handlers]

    async def initialize(self) -> None:
        """初始化 EventBus"""
        await self._store.initialize()
        logger.info("EventBus 已初始化")

    async def record_turn_start(self, snapshot: TurnSnapshot) -> None:
        """记录 Turn 开始"""
        snapshot.mark_running()
        await self._store.save_turn(snapshot)
        await self._emit("turn_start", snapshot)

    async def record_turn_step(self, snapshot: TurnSnapshot, step: int) -> None:
        """记录 Turn 步骤进展"""
        snapshot.current_step = step
        snapshot.step_count = max(snapshot.step_count, step)
        await self._store.save_turn(snapshot)
        await self._emit("turn_step", {"snapshot": snapshot, "step": step})

    async def record_turn_complete(self, snapshot: TurnSnapshot, result: str) -> None:
        """记录 Turn 完成"""
        snapshot.mark_completed(result)
        await self._store.save_turn(snapshot)
        await self._emit("turn_complete", snapshot)

    async def record_turn_error(
        self, snapshot: TurnSnapshot, error: str, is_timeout: bool = False
    ) -> None:
        """记录 Turn 错误"""
        snapshot.mark_failed(error, is_timeout)
        await self._store.save_turn(snapshot)
        await self._emit("turn_error", {"snapshot": snapshot, "error": error, "is_timeout": is_timeout})

    async def record_turn_interrupted(self, snapshot: TurnSnapshot) -> None:
        """记录 Turn 中断（进程崩溃时调用）"""
        snapshot.mark_interrupted()
        await self._store.save_turn(snapshot)
        await self._emit("turn_interrupted", snapshot)

    async def get_recovery_point(self, dag_id: str) -> TurnSnapshot | None:
        """获取断点恢复位置

        返回最后一个未完成的叶子节点的 Turn 状态。
        """
        return await self._store.get_last_incomplete_leaf(dag_id)

    async def get_all_interrupted(self, session_id: str | None = None) -> list[TurnSnapshot]:
        """获取所有中断的 Turn（用于全局恢复）"""
        return await self._store.get_interrupted_turns(session_id)

    async def save_dag_state(self, dag: TaskDAG) -> None:
        """保存 DAG 状态"""
        await self._store.save_dag(dag)

    async def restore_dag(self, dag_id: str) -> TaskDAG | None:
        """恢复 DAG 状态"""
        return await self._store.get_dag(dag_id)

    def subscribe(self, event_type: str, handler) -> None:
        """订阅事件"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    async def _emit(self, event_type: str, data: Any) -> None:
        """分发事件"""
        handlers = self._subscribers.get(event_type, [])
        for handler in handlers:
            try:
                if hasattr(handler, "__call__"):
                    result = handler(data)
                    if hasattr(result, "__await__"):
                        await result
            except Exception as e:
                logger.warning("事件处理器错误: %s", e)

    async def close(self) -> None:
        """关闭 EventBus"""
        await self._store.close()