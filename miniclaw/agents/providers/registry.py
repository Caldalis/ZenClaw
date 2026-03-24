"""
提供商注册表 — 对标 OpenClaw 的 Provider Registry + 故障转移

核心功能:
  1. 根据配置创建 Provider 实例
  2. 按优先级排序（第一个可用的 Provider 被使用）
  3. 故障转移: 当前 Provider 失败时自动切换到下一个

这是 OpenClaw 高可用设计的关键：
  用户可以配置多个 Provider，主 Provider 不可用时自动降级。
"""

from __future__ import annotations

from miniclaw.config.settings import ProviderConfig
from miniclaw.utils.errors import ConfigError, ProviderError
from miniclaw.utils.logging import get_logger

from .anthropic_provider import AnthropicProvider
from .base import AIProvider
from .openai_provider import OpenAIProvider

logger = get_logger(__name__)


class ProviderRegistry:
    """AI 提供商注册表 — 管理提供商列表并支持故障转移

    使用方式:
      registry = ProviderRegistry(config.providers)
      provider = registry.get_provider()  # 获取当前活跃的提供商
    """

    def __init__(self, configs: list[ProviderConfig]):
        self._providers: list[AIProvider] = []
        self._current_index = 0

        for cfg in configs:
            provider = self._create_provider(cfg)
            if provider:
                self._providers.append(provider)

        if not self._providers:
            raise ConfigError("没有可用的 AI 提供商，请检查配置中的 providers 部分")

        logger.info(
            "提供商注册表已初始化: %s (共 %d 个)",
            [p.name for p in self._providers],
            len(self._providers),
        )

    def _create_provider(self, config: ProviderConfig) -> AIProvider | None:
        """根据配置创建 Provider 实例"""
        if not config.api_key:
            logger.warning("提供商 %s 缺少 api_key，跳过", config.type)
            return None

        if config.type == "openai":
            return OpenAIProvider(
                api_key=config.api_key,
                model=config.model,
                base_url=config.base_url,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
        elif config.type == "anthropic":
            return AnthropicProvider(
                api_key=config.api_key,
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
        else:
            logger.warning("未知的提供商类型: %s", config.type)
            return None

    def get_provider(self) -> AIProvider:
        """获取当前活跃的提供商"""
        if not self._providers:
            raise ConfigError("没有可用的 AI 提供商")
        return self._providers[self._current_index]

    def failover(self) -> AIProvider | None:
        """故障转移: 切换到下一个提供商

        Returns:
            新的 Provider，如果已经没有更多 Provider 则返回 None
        """
        next_index = self._current_index + 1
        if next_index >= len(self._providers):
            logger.error("所有提供商都已失败，无法故障转移")
            return None

        self._current_index = next_index
        provider = self._providers[self._current_index]
        logger.warning("故障转移: 切换到提供商 '%s'", provider.name)
        return provider

    def reset(self) -> None:
        """重置到主提供商（恢复正常后调用）"""
        self._current_index = 0

    @property
    def provider_count(self) -> int:
        return len(self._providers)
