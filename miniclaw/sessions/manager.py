"""
会话管理器

职责:
  1. 创建、获取、列出会话
  2. 将消息持久化到 MemoryStore
  3. 从 MemoryStore 恢复历史会话

OpenClaw 中 SessionManager 是 Gateway 和 Agent 之间的中间层，
确保消息历史的一致性和持久化。
"""

from __future__ import annotations

import aiosqlite
from pathlib import Path

from miniclaw.memory.base import MemoryStore
from miniclaw.types.enums import SessionStatus
from miniclaw.types.messages import Message
from miniclaw.utils.logging import get_logger

from .session import Session

logger = get_logger(__name__)


class SessionManager:
    """会话管理器 — 管理所有活跃会话"""

    def __init__(self, memory_store: MemoryStore, db_path: str = "data/miniclaw.db"):
        self._memory = memory_store
        self._sessions: dict[str, Session] = {}  # 内存中的活跃会话缓存
        self._db_path = db_path

    async def initialize(self) -> None:
        """初始化会话元数据表"""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._meta_db = await aiosqlite.connect(self._db_path)
        await self._meta_db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT DEFAULT '',
                status TEXT DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                kind TEXT DEFAULT 'user',
                parent_session_id TEXT
            )
        """)
        # 兼容老库：如果旧版 schema 没有 kind / parent_session_id 列，补上
        cursor = await self._meta_db.execute("PRAGMA table_info(sessions)")
        cols = {row[1] for row in await cursor.fetchall()}
        if "kind" not in cols:
            await self._meta_db.execute(
                "ALTER TABLE sessions ADD COLUMN kind TEXT DEFAULT 'user'"
            )
        if "parent_session_id" not in cols:
            await self._meta_db.execute(
                "ALTER TABLE sessions ADD COLUMN parent_session_id TEXT"
            )
        await self._meta_db.commit()

    async def create_session(
        self,
        title: str = "",
        kind: str = "user",
        parent_session_id: str | None = None,
    ) -> Session:
        """创建新会话"""
        session = Session(title=title, kind=kind, parent_session_id=parent_session_id)
        self._sessions[session.id] = session

        await self._meta_db.execute(
            "INSERT INTO sessions (id, title, status, created_at, updated_at, kind, parent_session_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session.id, session.title, session.status.value,
             session.created_at.isoformat(), session.updated_at.isoformat(),
             session.kind, session.parent_session_id),
        )
        await self._meta_db.commit()
        logger.info("创建会话: %s (kind=%s)", session.id[:8], kind)
        return session

    async def get_session(self, session_id: str) -> Session | None:
        """获取会话（优先从缓存，否则从数据库恢复）"""
        # 缓存命中
        if session_id in self._sessions:
            return self._sessions[session_id]

        # 从数据库恢复
        cursor = await self._meta_db.execute(
            "SELECT id, title, status, created_at, updated_at, kind, parent_session_id "
            "FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        session = Session(
            id=row[0], title=row[1], status=SessionStatus(row[2]),
            kind=(row[5] or "user"),
            parent_session_id=row[6],
        )
        # 恢复消息历史
        messages = await self._memory.get_messages(session_id)
        session.messages = messages
        self._sessions[session_id] = session
        logger.info("恢复会话: %s (%d 条消息)", session_id[:8], len(messages))
        return session

    async def add_message(self, session_id: str, message: Message) -> None:
        """向会话添加消息并持久化"""
        session = await self.get_session(session_id)
        if session is None:
            raise ValueError(f"会话不存在: {session_id}")

        session.add_message(message)
        await self._memory.save_message(session_id, message)

        # 更新会话元数据
        await self._meta_db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (session.updated_at.isoformat(), session_id),
        )
        await self._meta_db.commit()

    async def list_sessions(self, status: SessionStatus = SessionStatus.ACTIVE) -> list[Session]:
        """列出指定状态的会话"""
        cursor = await self._meta_db.execute(
            "SELECT id, title, status, created_at, updated_at FROM sessions WHERE status = ? ORDER BY updated_at DESC",
            (status.value,),
        )
        rows = await cursor.fetchall()
        sessions = []
        for row in rows:
            sid = row[0]
            if sid in self._sessions:
                sessions.append(self._sessions[sid])
            else:
                sessions.append(Session(
                    id=row[0], title=row[1], status=SessionStatus(row[2]),
                ))
        return sessions

    async def get_or_create_session(
        self,
        session_id: str | None = None,
        title: str = "",
        kind: str = "user",
        parent_session_id: str | None = None,
    ) -> Session:
        """获取现有会话或创建新会话

        若传入 session_id 且该会话已存在 → 返回现有会话；
        若传入 session_id 但不存在 → **以该 ID 本身登记一条新会话**；
        若未传入 session_id → 创建一条全新的随机 ID 会话。
        """
        if session_id:
            existing = await self.get_session(session_id)
            if existing:
                return existing
            session = Session(
                id=session_id,
                title=title,
                kind=kind,
                parent_session_id=parent_session_id,
            )
            self._sessions[session_id] = session
            await self._meta_db.execute(
                "INSERT OR IGNORE INTO sessions "
                "(id, title, status, created_at, updated_at, kind, parent_session_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session.id, session.title, session.status.value,
                 session.created_at.isoformat(), session.updated_at.isoformat(),
                 session.kind, session.parent_session_id),
            )
            await self._meta_db.commit()
            logger.info("登记会话: %s (kind=%s)", session_id[:24], kind)
            return session
        return await self.create_session(title=title, kind=kind, parent_session_id=parent_session_id)

    async def get_or_create_subagent_session(
        self,
        session_id: str,
        title: str = "",
        parent_session_id: str | None = None,
    ) -> Session:
        """以指定 ID 登记或获取一个 subagent 会话。

        新创建时自动把 kind 标记为 "subagent"，便于 DAG 完成后按父会话归组清理。
        """
        return await self.get_or_create_session(
            session_id,
            title=title,
            kind="subagent",
            parent_session_id=parent_session_id,
        )

    async def archive_subagent_sessions(
        self,
        session_ids: list[str] | set[str],
    ) -> int:
        """把一批 subagent session 标记为 archived 并从缓存驱逐。

        用于 DAG 执行完毕后归档子任务会话，避免 list_sessions 里混入大量临时 session。
        消息行由 MemoryStore 持有，不在这里删除（便于日后审计/调试）。

        Returns:
            实际被归档的会话数
        """
        archived = 0
        for sid in list(session_ids):
            cursor = await self._meta_db.execute(
                "UPDATE sessions SET status = 'archived' WHERE id = ? AND kind = 'subagent'",
                (sid,),
            )
            if cursor.rowcount:
                archived += cursor.rowcount
            # 驱逐缓存，防止旧对象长期持有大量消息
            self._sessions.pop(sid, None)
        await self._meta_db.commit()
        if archived:
            logger.info("归档 subagent 会话: %d 条", archived)
        return archived

    async def close(self) -> None:
        if hasattr(self, "_meta_db") and self._meta_db:
            await self._meta_db.close()
