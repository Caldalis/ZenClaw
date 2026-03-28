"""
SQLite 记忆存储
使用 aiosqlite 实现异步 SQLite 存储，支持 FTS5 全文检索（BM25）。
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


def _sanitize_fts_query(query: str) -> str:
    """将用户输入安全转换为 FTS5 MATCH 查询字符串。

    FTS5 的特殊字符（"、*、(、)、-、OR、AND、NOT）可能导致语法错误。
    最安全的做法是将每个 token 包裹在双引号中，内部双引号转义为 ""。
    """
    tokens = query.split()
    safe = ['"' + t.replace('"', '""') + '"' for t in tokens if t]
    return " ".join(safe) if safe else '""'


class SQLiteMemoryStore(MemoryStore):
    """SQLite 实现的记忆存储，支持 FTS5 全文检索"""

    def __init__(self, db_path: str = "data/miniclaw.db", enable_fts: bool = True):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._enable_fts = enable_fts

    async def initialize(self) -> None:
        """创建数据库和表"""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(self._db_path)
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

        if self._enable_fts:
            await self._db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(
                    message_id UNINDEXED,
                    session_id UNINDEXED,
                    content,
                    tokenize = 'unicode61 remove_diacritics 1'
                )
            """)

        await self._db.commit()
        logger.info("SQLite 记忆存储已初始化: %s (FTS5=%s)", self._db_path, self._enable_fts)

    def _row_to_message(self, row: tuple, session_id: str) -> Message:
        """将数据库行解析为 Message 对象"""
        msg_id, role, content, tc_json, tr_json, ts = row
        tool_calls = [ToolCall(**tc) for tc in json.loads(tc_json)] if tc_json else None
        tool_result = ToolResult(**json.loads(tr_json)) if tr_json else None
        return Message(
            id=msg_id,
            role=Role(role),
            content=content,
            tool_calls=tool_calls,
            tool_result=tool_result,
            session_id=session_id,
        )

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

        if self._enable_fts and message.content and len(message.content.strip()) >= 5:
            await self._db.execute(
                "INSERT OR REPLACE INTO chunks_fts (message_id, session_id, content) VALUES (?, ?, ?)",
                (message.id, session_id, message.content),
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
        return [self._row_to_message(row, session_id) for row in reversed(rows)]

    async def search(self, session_id: str, query: str, top_k: int = 5) -> list[Message]:
        """搜索相关消息：FTS5 BM25 检索或降级到 LIKE 搜索"""
        if self._enable_fts:
            return await self._fts_search(session_id, query, top_k)
        return await self._like_search(session_id, query, top_k)

    async def _fts_search(self, session_id: str, query: str, top_k: int) -> list[Message]:
        """FTS5 BM25 全文检索"""
        assert self._db is not None
        safe_query = _sanitize_fts_query(query)
        try:
            cursor = await self._db.execute(
                """SELECT m.id, m.role, m.content, m.tool_calls_json,
                          m.tool_result_json, m.timestamp
                   FROM chunks_fts
                   JOIN messages m ON m.id = chunks_fts.message_id
                   WHERE chunks_fts.session_id = ? AND chunks_fts MATCH ?
                   ORDER BY -bm25(chunks_fts)
                   LIMIT ?""",
                (session_id, safe_query, top_k),
            )
            rows = await cursor.fetchall()
            return [self._row_to_message(row, session_id) for row in rows]
        except Exception as e:
            logger.warning("FTS5 搜索失败: %s，降级到 LIKE 搜索", e)
            return await self._like_search(session_id, query, top_k)

    async def _like_search(self, session_id: str, query: str, top_k: int) -> list[Message]:
        """LIKE 全文匹配（兜底方案）"""
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
        return [self._row_to_message(row, session_id) for row in rows]

    async def delete_message(self, message_id: str) -> None:
        """删除单条消息（用于上下文压缩）"""
        assert self._db is not None
        if self._enable_fts:
            await self._db.execute(
                "DELETE FROM chunks_fts WHERE message_id = ?", (message_id,)
            )
        await self._db.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        await self._db.commit()

    async def delete_session(self, session_id: str) -> None:
        assert self._db is not None
        if self._enable_fts:
            await self._db.execute("DELETE FROM chunks_fts WHERE session_id = ?", (session_id,))
        await self._db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
