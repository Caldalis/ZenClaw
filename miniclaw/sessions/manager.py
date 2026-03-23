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
                updated_at TEXT NOT NULL
            )
        """)
        await self._meta_db.commit()

    async def create_session(self, title: str = "") -> Session:
        """创建新会话"""
        session = Session(title=title)
        self._sessions[session.id] = session

        await self._meta_db.execute(
            "INSERT INTO sessions (id, title, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session.id, session.title, session.status.value,
             session.created_at.isoformat(), session.updated_at.isoformat()),
        )
        await self._meta_db.commit()
        logger.info("创建会话: %s", session.id[:8])
        return session

    async def get_session(self, session_id: str) -> Session | None:
        """获取会话（优先从缓存，否则从数据库恢复）"""
        # 缓存命中
        if session_id in self._sessions:
            return self._sessions[session_id]

        # 从数据库恢复
        cursor = await self._meta_db.execute(
            "SELECT id, title, status, created_at, updated_at FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        session = Session(
            id=row[0], title=row[1], status=SessionStatus(row[2]),
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

    async def get_or_create_session(self, session_id: str | None = None) -> Session:
        """获取现有会话或创建新会话"""
        if session_id:
            session = await self.get_session(session_id)
            if session:
                return session
        return await self.create_session()

    async def close(self) -> None:
        if hasattr(self, "_meta_db") and self._meta_db:
            await self._meta_db.close()
