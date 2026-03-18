"""
嵌入向量生成
test
"""

from __future__ import annotations

import numpy as np

from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """嵌入向量生成器
    """
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", base_url: str | None = None):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    async def embed(self, text: str) -> np.ndarray:
        """将单段文本转为嵌入向量

        Returns:
            numpy 数组
        """
        client = self._get_client()
        response = await client.embeddings.create(input=[text], model=self._model)
        vector = response.data[0].embedding
        return np.array(vector, dtype=np.float32)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """批量生成嵌入向量
        """
        pass
