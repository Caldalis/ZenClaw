"""
向量搜索存储

在 SQLiteMemoryStore 基础上增加语义搜索能力：
- 优先使用 sqlite-vec 扩展（真正的向量索引）
- 若 sqlite-vec 不可用，降级为 numpy BLOB 暴力搜索

注意：aiosqlite 使用线程池运行 SQLite，加载扩展必须通过
run_sync() 在底层 sqlite3.Connection 上操作。
"""

from __future__ import annotations

import sqlite3

import numpy as np

import aiosqlite

from miniclaw.types.messages import Message
from miniclaw.utils.logging import get_logger

from .embeddings import EmbeddingGenerator, cosine_similarity
from .sqlite_store import SQLiteMemoryStore

logger = get_logger(__name__)

try:
    import sqlite_vec
    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    _SQLITE_VEC_AVAILABLE = False


class VectorMemoryStore(SQLiteMemoryStore):
    """带向量搜索的 SQLite 存储

    优先使用 sqlite-vec 扩展实现真正的向量索引；
    sqlite-vec 不可用时降级为 numpy 内存暴力搜索。
    """

    def __init__(self, db_path: str, embedding_generator: EmbeddingGenerator, embedding_dim: int = 1536):
        super().__init__(db_path, enable_fts=True)
        self._embedder = embedding_generator
        self._dim = embedding_dim
        self._use_sqlite_vec = _SQLITE_VEC_AVAILABLE

    async def initialize(self) -> None:
        """初始化基础表 + 向量表"""
        await super().initialize()
        assert self._db is not None

        if self._use_sqlite_vec:
            def _load_extension(conn: sqlite3.Connection) -> None:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)

            try:
                await self._db.run_sync(_load_extension)
                await self._db.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec
                    USING vec0(
                        message_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        embedding float[{self._dim}]
                    )
                """)
                await self._db.commit()
                logger.info("sqlite-vec 向量表已初始化 (dim=%d)", self._dim)
            except Exception as e:
                logger.warning("sqlite-vec 加载失败: %s，降级到 numpy BLOB", e)
                self._use_sqlite_vec = False

        if not self._use_sqlite_vec:
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES messages(id)
                )
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_session
                ON embeddings(session_id)
            """)
            await self._db.commit()
            logger.info("numpy BLOB 向量表已初始化 (dim=%d) [sqlite-vec 不可用]", self._dim)

    async def save_message(self, session_id: str, message: Message) -> None:
        """保存消息 + 生成并存储嵌入向量"""
        await super().save_message(session_id, message)

        if not message.content or len(message.content.strip()) < 5:
            return

        try:
            vector = await self._embedder.embed(message.content)
            assert self._db is not None
            if self._use_sqlite_vec:
                await self._db.execute(
                    "INSERT OR REPLACE INTO chunks_vec (message_id, session_id, embedding) VALUES (?, ?, ?)",
                    (message.id, session_id, sqlite_vec.serialize_float32(vector.tolist())),
                )
            else:
                await self._db.execute(
                    "INSERT OR REPLACE INTO embeddings (message_id, session_id, vector) VALUES (?, ?, ?)",
                    (message.id, session_id, vector.tobytes()),
                )
            await self._db.commit()
        except Exception as e:
            logger.warning("向量生成失败 (message_id=%s): %s", message.id, e)

    async def search(self, session_id: str, query: str, top_k: int = 5) -> list[Message]:
        """语义搜索 — 用余弦相似度找最相关的历史消息"""
        assert self._db is not None

        try:
            query_vec = await self._embedder.embed(query)
        except Exception as e:
            logger.warning("查询向量生成失败: %s，降级到文本搜索", e)
            return await super().search(session_id, query, top_k)

        if self._use_sqlite_vec:
            return await self._sqlite_vec_search(session_id, query_vec, top_k)
        return await self._numpy_search(session_id, query_vec, top_k)

    async def _sqlite_vec_search(self, session_id: str, query_vec: np.ndarray, top_k: int) -> list[Message]:
        """使用 sqlite-vec 进行向量相似度搜索"""
        assert self._db is not None
        try:
            cursor = await self._db.execute(
                """SELECT cv.message_id,
                          vec_distance_cosine(cv.embedding, ?) AS distance
                   FROM chunks_vec cv
                   WHERE cv.session_id = ?
                   ORDER BY distance ASC
                   LIMIT ?""",
                (sqlite_vec.serialize_float32(query_vec.tolist()), session_id, top_k),
            )
            rows = await cursor.fetchall()
            top_ids = [row[0] for row in rows]
        except Exception as e:
            logger.warning("sqlite-vec 搜索失败: %s，降级到 numpy", e)
            return await self._numpy_search(session_id, query_vec, top_k)

        if not top_ids:
            return []
        return await self._fetch_messages_by_ids(session_id, top_ids)

    async def _numpy_search(self, session_id: str, query_vec: np.ndarray, top_k: int) -> list[Message]:
        """numpy 暴力余弦相似度搜索（兜底）"""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT message_id, vector FROM embeddings WHERE session_id = ?",
            (session_id,),
        )
        rows = await cursor.fetchall()
        if not rows:
            return []

        scored: list[tuple[str, float]] = []
        for msg_id, vec_bytes in rows:
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            sim = cosine_similarity(query_vec, vec)
            scored.append((msg_id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [msg_id for msg_id, _ in scored[:top_k]]
        return await self._fetch_messages_by_ids(session_id, top_ids)

    async def _fetch_messages_by_ids(self, session_id: str, msg_ids: list[str]) -> list[Message]:
        """按 ID 列表批量查询消息"""
        assert self._db is not None
        if not msg_ids:
            return []
        placeholders = ",".join("?" * len(msg_ids))
        cursor = await self._db.execute(
            f"SELECT id, role, content, tool_calls_json, tool_result_json, timestamp "
            f"FROM messages WHERE id IN ({placeholders})",
            msg_ids,
        )
        rows = await cursor.fetchall()
        return [self._row_to_message(row, session_id) for row in rows]

    async def delete_message(self, message_id: str) -> None:
        """删除单条消息及其向量"""
        assert self._db is not None
        if self._use_sqlite_vec:
            await self._db.execute("DELETE FROM chunks_vec WHERE message_id = ?", (message_id,))
        else:
            await self._db.execute("DELETE FROM embeddings WHERE message_id = ?", (message_id,))
        await super().delete_message(message_id)

    async def delete_session(self, session_id: str) -> None:
        assert self._db is not None
        if self._use_sqlite_vec:
            await self._db.execute("DELETE FROM chunks_vec WHERE session_id = ?", (session_id,))
        else:
            await self._db.execute("DELETE FROM embeddings WHERE session_id = ?", (session_id,))
        await super().delete_session(session_id)
