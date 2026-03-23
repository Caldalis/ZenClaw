"""
SQLite 记忆存储
使用 aiosqlite 实现异步 SQLite 存储。
test
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from miniclaw.types.enums import Role
from miniclaw.types.messages import Message, ToolCall, ToolResult
from miniclaw.utils.logging import get_logger

from .base import MemoryStore

logger = get_logger(__name__)


class SQLiteMemoryStore(MemoryStore):
    """SQLite 实现的记忆存储"""

    def __init__(self, db_path: str = "data/miniclaw.db"):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """创建数据库和表"""
        # 确保目录存在
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(self._db_path)
        # 启用 WAL 模式提高并发读性能
        await self._db.execute("PRAGMA journal_mode=WAL")

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT DEFAULT '',
                tool_calls_json TEXT,
                tool_result_json TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, timestamp)
        """)
        await self._db.commit()
        logger.info("SQLite 记忆存储已初始化: %s", self._db_path)

    async def save_message(self, session_id: str, message: Message) -> None:
        """持久化一条消息"""
        assert self._db is not None
        tool_calls_json = None
        if message.tool_calls:
            tool_calls_json = json.dumps([tc.model_dump() for tc in message.tool_calls], ensure_ascii=False)
        tool_result_json = None
        if message.tool_result:
            tool_result_json = json.dumps(message.tool_result.model_dump(), ensure_ascii=False)

        await self._db.execute(
            """INSERT OR REPLACE INTO messages
               (id, session_id, role, content, tool_calls_json, tool_result_json, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                message.id,
                session_id,
                message.role.value,
                message.content,
                tool_calls_json,
                tool_result_json,
                message.timestamp.isoformat(),
            ),
        )
        await self._db.commit()

    async def get_messages(self, session_id: str, limit: int = 100) -> list[Message]:
        """获取会话的最近 N 条消息"""
        assert self._db is not None
        cursor = await self._db.execute(
            """SELECT id, role, content, tool_calls_json, tool_result_json, timestamp
               FROM messages
               WHERE session_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (session_id, limit),
        )
        rows = await cursor.fetchall()

        messages = []
        for row in reversed(rows):  # 反转回时间正序
            msg_id, role, content, tc_json, tr_json, ts = row
            tool_calls = None
            if tc_json:
                tool_calls = [ToolCall(**tc) for tc in json.loads(tc_json)]
            tool_result = None
            if tr_json:
                tool_result = ToolResult(**json.loads(tr_json))
            messages.append(Message(
                id=msg_id,
                role=Role(role),
                content=content,
                tool_calls=tool_calls,
                tool_result=tool_result,
                session_id=session_id,
            ))
        return messages

    async def search(self, session_id: str, query: str, top_k: int = 5) -> list[Message]:
        """简单文本搜索（全文匹配）

        注意: 这是 SQLite 的简单 LIKE 搜索。
        如需语义搜索，使用 VectorStore 包装。
        """
        assert self._db is not None
        cursor = await self._db.execute(
            """SELECT id, role, content, tool_calls_json, tool_result_json, timestamp
               FROM messages
               WHERE session_id = ? AND content LIKE ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (session_id, f"%{query}%", top_k),
        )
        rows = await cursor.fetchall()
        messages = []
        for row in rows:
            msg_id, role, content, tc_json, tr_json, ts = row
            tool_calls = [ToolCall(**tc) for tc in json.loads(tc_json)] if tc_json else None
            tool_result = ToolResult(**json.loads(tr_json)) if tr_json else None
            messages.append(Message(
                id=msg_id, role=Role(role), content=content,
                tool_calls=tool_calls, tool_result=tool_result,
                session_id=session_id,
            ))
        return messages

    async def delete_session(self, session_id: str) -> None:
        assert self._db is not None
        await self._db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
