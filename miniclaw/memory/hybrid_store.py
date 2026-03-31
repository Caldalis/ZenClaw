"""
混合记忆存储 (Hybrid Memory Store)

双引擎混合检索：
1. FTS5 关键词搜索（BM25 算分）
2. sqlite-vec / numpy 向量语义搜索（余弦相似度）

合并策略：
- Min-max 归一化 → 加权合并
- 时间衰减（半衰期惩罚）
- MMR（最大边际相关性）去重，保证多样性
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone

import numpy as np

from miniclaw.types.messages import Message
from miniclaw.utils.logging import get_logger

from .embeddings import EmbeddingGenerator, cosine_similarity
from .vector_store import VectorMemoryStore

logger = get_logger(__name__)


def _normalize_scores(scored: list[tuple[str, float]]) -> dict[str, float]:
    """Min-max 归一化，将分数映射到 [0, 1]"""
    if not scored:
        return {}
    scores = [s for _, s in scored]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return {mid: 1.0 for mid, _ in scored}
    return {mid: (s - min_s) / (max_s - min_s) for mid, s in scored}


def _mmr_select(
    candidates: dict[str, float],
    candidate_vecs: dict[str, np.ndarray],
    top_k: int,
    mmr_lambda: float,
) -> list[str]:
    """最大边际相关性 (MMR) 选择，保证结果多样性

    MMR 得分 = lambda * relevance(d) - (1 - lambda) * max_sim(d, selected)

    Args:
        candidates: {message_id: combined_score} — 已经过时间衰减的合并分数
        candidate_vecs: {message_id: embedding_vector} — 用于多样性计算
        top_k: 最终选取数量
        mmr_lambda: 平衡相关性与多样性（1.0=纯相关，0.0=纯多样）

    Returns:
        按 MMR 顺序排列的 message_id 列表
    """
    selected: list[str] = []
    remaining = set(candidates.keys())

    for _ in range(min(top_k, len(remaining))):
        best_id: str | None = None
        best_score = float("-inf")

        for mid in remaining:
            relevance = candidates[mid]

            if selected and mid in candidate_vecs:
                sims = [
                    cosine_similarity(candidate_vecs[mid], candidate_vecs[sid])
                    for sid in selected
                    if sid in candidate_vecs
                ]
                max_sim = max(sims) if sims else 0.0
            else:
                max_sim = 0.0

            mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_id = mid

        if best_id is None:
            break
        selected.append(best_id)
        remaining.discard(best_id)

    return selected


class HybridMemoryStore(VectorMemoryStore):
    """混合记忆存储 — FTS5 + 向量检索 + MMR + 时间衰减

    继承 VectorMemoryStore（已含 FTS5 + sqlite-vec/numpy 两路向量），
    重写 search() 实现双引擎合并搜索。
    """

    def __init__(
        self,
        db_path: str,
        embedding_generator: EmbeddingGenerator,
        embedding_dim: int = 1536,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        memory_half_life_days: float = 30.0,
        mmr_lambda: float = 0.5,
    ):
        super().__init__(db_path, embedding_generator, embedding_dim)
        self._vector_weight = vector_weight
        self._text_weight = text_weight
        self._half_life_days = memory_half_life_days
        self._mmr_lambda = mmr_lambda

    async def search(self, session_id: str, query: str, top_k: int = 5) -> list[Message]:
        """混合检索：FTS5 + 向量 → 合并 → 时间衰减 → MMR"""
        candidate_limit = top_k * 3

        # Step 1: 并行发起 FTS5 和向量搜索
        fts_task = asyncio.create_task(self._fts_candidates(session_id, query, candidate_limit))
        vec_task = asyncio.create_task(self._vec_candidates(session_id, query, candidate_limit))
        fts_results, vec_results = await asyncio.gather(fts_task, vec_task, return_exceptions=True)

        if isinstance(fts_results, Exception):
            logger.warning("FTS5 候选搜索失败: %s", fts_results)
            fts_results = []
        if isinstance(vec_results, Exception):
            logger.warning("向量候选搜索失败: %s", vec_results)
            vec_results = []

        # 若两路都为空，返回空结果
        if not fts_results and not vec_results:
            return []

        # Step 2: Min-max 归一化
        fts_norm = _normalize_scores(fts_results)   # type: ignore[arg-type]
        vec_norm = _normalize_scores(vec_results)   # type: ignore[arg-type]

        # Step 3: 加权合并（取所有候选的并集）
        all_ids = set(fts_norm) | set(vec_norm)
        combined: dict[str, float] = {
            mid: vec_norm.get(mid, 0.0) * self._vector_weight
                 + fts_norm.get(mid, 0.0) * self._text_weight
            for mid in all_ids
        }

        # Step 4: 加载完整消息
        messages_by_id = await self._load_messages_map(session_id, list(all_ids))
        if not messages_by_id:
            return []

        # Step 5: 时间衰减
        now = datetime.now(timezone.utc)
        decay_lambda = math.log(2) / self._half_life_days  # λ = ln2 / t_half
        for mid, msg in messages_by_id.items():
            days_elapsed = (now - msg.timestamp).total_seconds() / 86400.0
            decay = math.exp(-decay_lambda * days_elapsed)
            combined[mid] = combined.get(mid, 0.0) * decay

        # Step 6: 加载向量用于 MMR 多样性计算
        candidate_vecs = await self._load_embedding_map(list(messages_by_id.keys()))

        # Step 7: MMR 选择
        selected_ids = _mmr_select(
            candidates={mid: combined[mid] for mid in messages_by_id},
            candidate_vecs=candidate_vecs,
            top_k=top_k,
            mmr_lambda=self._mmr_lambda,
        )

        logger.debug(
            "混合检索: fts=%d, vec=%d, union=%d, mmr_selected=%d",
            len(fts_results), len(vec_results), len(all_ids), len(selected_ids),
        )

        return [messages_by_id[mid] for mid in selected_ids if mid in messages_by_id]

    async def _fts_candidates(
        self, session_id: str, query: str, limit: int
    ) -> list[tuple[str, float]]:
        """FTS5 BM25 候选集（message_id, score）"""
        assert self._db is not None
        from .sqlite_store import _sanitize_fts_query
        safe_query = _sanitize_fts_query(query)
        try:
            cursor = await self._db.execute(
                """SELECT message_id, -bm25(chunks_fts) AS score
                   FROM chunks_fts
                   WHERE session_id = ? AND chunks_fts MATCH ?
                   ORDER BY score DESC
                   LIMIT ?""",
                (session_id, safe_query, limit),
            )
            rows = await cursor.fetchall()
            return [(row[0], float(row[1])) for row in rows]
        except Exception as e:
            logger.warning("FTS5 候选搜索失败: %s", e)
            return []

    async def _vec_candidates(
        self, session_id: str, query: str, limit: int
    ) -> list[tuple[str, float]]:
        """向量相似度候选集（message_id, similarity）"""
        assert self._db is not None
        try:
            query_vec = await self._embedder.embed(query)
        except Exception as e:
            logger.warning("查询向量生成失败: %s", e)
            return []

        if self._use_sqlite_vec:
            try:
                import sqlite_vec
                cursor = await self._db.execute(
                    """SELECT cv.message_id,
                              1.0 - vec_distance_cosine(cv.embedding, ?) AS similarity
                       FROM chunks_vec cv
                       WHERE cv.session_id = ?
                       ORDER BY similarity DESC
                       LIMIT ?""",
                    (sqlite_vec.serialize_float32(query_vec.tolist()), session_id, limit),
                )
                rows = await cursor.fetchall()
                return [(row[0], float(row[1])) for row in rows]
            except Exception as e:
                logger.warning("sqlite-vec 候选搜索失败: %s，降级到 numpy", e)

        # numpy 暴力搜索
        cursor = await self._db.execute(
            "SELECT message_id, vector FROM embeddings WHERE session_id = ?",
            (session_id,),
        )
        rows = await cursor.fetchall()
        scored = []
        for msg_id, vec_bytes in rows:
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            sim = cosine_similarity(query_vec, vec)
            scored.append((msg_id, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    async def _load_messages_map(self, session_id: str, msg_ids: list[str]) -> dict[str, Message]:
        """按 ID 列表批量加载消息，返回 {id: Message} 字典"""
        messages = await self._fetch_messages_by_ids(session_id, msg_ids)
        return {m.id: m for m in messages}

    async def _load_embedding_map(self, msg_ids: list[str]) -> dict[str, np.ndarray]:
        """加载消息的向量，返回 {message_id: ndarray}"""
        assert self._db is not None
        if not msg_ids:
            return {}

        result: dict[str, np.ndarray] = {}
        placeholders = ",".join("?" * len(msg_ids))

        if self._use_sqlite_vec:
            try:
                import sqlite_vec
                cursor = await self._db.execute(
                    f"SELECT message_id, embedding FROM chunks_vec WHERE message_id IN ({placeholders})",
                    msg_ids,
                )
                rows = await cursor.fetchall()
                for msg_id, emb_bytes in rows:
                    result[msg_id] = np.frombuffer(emb_bytes, dtype=np.float32)
                return result
            except Exception as e:
                logger.warning("加载 sqlite-vec 向量失败: %s", e)

        # numpy BLOB fallback
        cursor = await self._db.execute(
            f"SELECT message_id, vector FROM embeddings WHERE message_id IN ({placeholders})",
            msg_ids,
        )
        rows = await cursor.fetchall()
        for msg_id, vec_bytes in rows:
            result[msg_id] = np.frombuffer(vec_bytes, dtype=np.float32)
        return result
