"""
向量搜索存储

在 SQLiteMemoryStore 基础上增加语义搜索能力:
  1. 保存消息时同时生成嵌入向量
  2. 搜索时用余弦相似度找到最相关的历史消息

存储方式: 向量数据也存在 SQLite 中（BLOB 类型），
避免引入额外的向量数据库依赖。

注意: 这是简化实现。生产环境应使用专门的向量数据库
（如 Pinecone, Qdrant, pgvector）。
"""

from __future__ import annotations

import numpy as np

import aiosqlite

from miniclaw.types.messages import Message
from miniclaw.utils.logging import get_logger

from .embeddings import EmbeddingGenerator, cosine_similarity
from .sqlite_store import SQLiteMemoryStore

logger = get_logger(__name__)


class VectorMemoryStore(SQLiteMemoryStore):
    """带向量搜索的 SQLite 存储

    继承 SQLiteMemoryStore，重写 search() 方法实现语义搜索。
    向量数据存在单独的 embeddings 表中。
    """

    def __init__(self, db_path: str, embedding_generator: EmbeddingGenerator, embedding_dim: int = 1536):
        super().__init__(db_path)
        self._embedder = embedding_generator
        self._dim = embedding_dim

    async def initialize(self) -> None:
        """初始化基础表 + 向量表"""
        await super().initialize()
        assert self._db is not None
        # 向量表: 每条消息对应一个嵌入向量
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
        logger.info("向量存储已初始化 (dim=%d)", self._dim)

    async def save_message(self, session_id: str, message: Message) -> None:
        """保存消息 + 生成并存储嵌入向量"""
        await super().save_message(session_id, message)

        # 只为有实质内容的消息生成向量
        if not message.content or len(message.content.strip()) < 5:
            return

        try:
            vector = await self._embedder.embed(message.content)
            assert self._db is not None
            await self._db.execute(
                "INSERT OR REPLACE INTO embeddings (message_id, session_id, vector) VALUES (?, ?, ?)",
                (message.id, session_id, vector.tobytes()),
            )
            await self._db.commit()
        except Exception as e:
            # 向量生成失败不应阻塞消息保存
            logger.warning("向量生成失败 (message_id=%s): %s", message.id, e)

    async def search(self, session_id: str, query: str, top_k: int = 5) -> list[Message]:
        """语义搜索 — 用余弦相似度找最相关的历史消息

        流程:
          1. 将查询文本转为向量
          2. 加载该会话所有向量
          3. 计算余弦相似度
          4. 返回最相似的 top_k 条消息

        注意: 这是内存中暴力搜索 (brute-force)。
        数据量大时应使用近似最近邻 (ANN) 算法或专门的向量数据库。
        """
        assert self._db is not None

        # Step 1: 查询向量
        try:
            query_vec = await self._embedder.embed(query)
        except Exception as e:
            logger.warning("查询向量生成失败: %s，降级到文本搜索", e)
            return await super().search(session_id, query, top_k)

        # Step 2: 加载会话所有向量
        cursor = await self._db.execute(
            "SELECT message_id, vector FROM embeddings WHERE session_id = ?",
            (session_id,),
        )
        rows = await cursor.fetchall()
        if not rows:
            return []

        # Step 3: 计算相似度
        scored: list[tuple[str, float]] = []
        for msg_id, vec_bytes in rows:
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            sim = cosine_similarity(query_vec, vec)
            scored.append((msg_id, sim))

        # Step 4: 排序取 top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [msg_id for msg_id, _ in scored[:top_k]]

        # 查询对应的完整消息
        if not top_ids:
            return []
        placeholders = ",".join("?" * len(top_ids))
        cursor = await self._db.execute(
            f"SELECT id, role, content, tool_calls_json, tool_result_json, timestamp "
            f"FROM messages WHERE id IN ({placeholders})",
            top_ids,
        )
        rows = await cursor.fetchall()

        # 复用父类的行解析逻辑 — 这里简单重建
        import json
        from miniclaw.types.enums import Role
        from miniclaw.types.messages import ToolCall, ToolResult

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
        await self._db.execute("DELETE FROM embeddings WHERE session_id = ?", (session_id,))
        await super().delete_session(session_id)
