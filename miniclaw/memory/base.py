"""
记忆存储抽象基类
抽象基类定义统一接口，具体实现可以是 SQLite、PostgreSQL 或其他存储。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from miniclaw.types.messages import Message


class MemoryStore(ABC):
    """记忆存储抽象基类

    职责:
      - 存储和检索对话消息
      - 按 session_id 隔离不同会话的记忆
      - 支持消息的持久化和恢复
    """

    @abstractmethod
    async def initialize(self) -> None:
        """初始化存储（创建表、建立连接等）"""
        ...

    @abstractmethod
    async def save_message(self, session_id: str, message: Message) -> None:
        """保存一条消息到指定会话"""
        ...

    @abstractmethod
    async def get_messages(self, session_id: str, limit: int = 100) -> list[Message]:
        """获取指定会话的消息历史

        Args:
            session_id: 会话 ID
            limit: 最大返回条数（最新的 N 条）
        """
        ...

    @abstractmethod
    async def search(self, session_id: str, query: str, top_k: int = 5) -> list[Message]:
        """语义搜索相关消息（可选功能）

        默认实现返回空列表。启用向量搜索后可返回语义相关的历史消息。
        """
        ...

    @abstractmethod
    async def delete_message(self, message_id: str) -> None:
        """删除单条消息（用于上下文压缩）"""
        ...

    @abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """删除指定会话的所有消息"""
        ...

    @abstractmethod
    async def close(self) -> None:
        """关闭存储连接"""
        ...
