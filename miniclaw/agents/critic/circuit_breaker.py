"""
熔断器 (Circuit Breaker) — 防止无限重试相同错误

核心职责:
  1. 监控错误模式（如连续 3 次 Linter 报同一个 Syntax Error）
  2. 触发熔断时立即中断 ReAct 循环
  3. 返回精简的错误信息给上层决策

设计原则:
  - 不等待 max_steps 耗尽
  - 识别"相同错误"模式（错误类型 + 错误位置）
  - 支持可配置的熔断阈值
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from miniclaw.utils.logging import get_logger

logger = get_logger(__name__)


class CircuitState(str, Enum):
    """熔断器状态"""
    CLOSED = "closed"        # 正常状态，允许执行
    OPEN = "open"            # 熔断状态，拒绝执行
    HALF_OPEN = "half_open"  # 半开状态，允许一次尝试


@dataclass
class ErrorPattern:
    """错误模式 — 用于识别相同类型的错误"""
    error_type: str
    """错误类型（如 SyntaxError, ImportError, TypeError）"""

    error_message: str
    """错误消息"""

    location: str | None = None
    """错误位置（文件:行号）"""

    fingerprint: str = ""
    """错误指纹（用于快速匹配）"""

    count: int = 1
    """出现次数"""

    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """首次出现时间"""

    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """最后出现时间"""

    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        """计算错误指纹"""
        content = f"{self.error_type}:{self.error_message}"
        if self.location:
            content += f"@{self.location}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def increment(self) -> None:
        """增加出现次数"""
        self.count += 1
        self.last_seen = datetime.now(timezone.utc)


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    # 熔断阈值
    failure_threshold: int = 3
    """连续失败多少次触发熔断"""

    # 时间窗口
    time_window_seconds: int = 300
    """时间窗口（秒），超过此时间重置计数"""

    # 相同错误判断
    same_error_threshold: int = 3
    """相同错误出现多少次触发熔断"""

    # 半开状态配置
    half_open_timeout_seconds: int = 60
    """半开状态超时时间"""

    # 错误类型过滤
    ignored_error_types: list[str] = field(default_factory=lambda: [])
    """忽略的错误类型（不计入熔断）"""


class CircuitBreakerError(Exception):
    """熔断器错误 — 表示任务被熔断"""

    def __init__(
        self,
        message: str,
        error_pattern: ErrorPattern | None = None,
        failure_count: int = 0,
        is_same_error: bool = False,
    ):
        self.error_pattern = error_pattern
        self.failure_count = failure_count
        self.is_same_error = is_same_error
        super().__init__(message)


class CircuitBreaker:
    """熔断器 — 监控错误并在达到阈值时中断执行

    使用方式:
        breaker = CircuitBreaker(config)

        for attempt in range(max_attempts):
            try:
                result = await execute()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure(e)
                if breaker.is_open:
                    raise breaker.get_breaker_error()
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._error_patterns: dict[str, ErrorPattern] = {}
        self._last_failure_time: datetime | None = None
        self._opened_at: datetime | None = None

    @property
    def state(self) -> CircuitState:
        """当前状态"""
        return self._state

    @property
    def is_open(self) -> bool:
        """是否处于熔断状态"""
        if self._state == CircuitState.OPEN:
            return True
        if self._state == CircuitState.HALF_OPEN:
            # 半开状态超时后转为开
            if self._opened_at:
                elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
                if elapsed > self.config.half_open_timeout_seconds:
                    self._state = CircuitState.OPEN
                    return True
        return False

    @property
    def is_closed(self) -> bool:
        """是否处于正常状态"""
        return self._state == CircuitState.CLOSED

    @property
    def failure_count(self) -> int:
        """失败计数"""
        return self._failure_count

    def record_success(self) -> None:
        """记录成功"""
        self._success_count += 1

        if self._state == CircuitState.HALF_OPEN:
            # 半开状态下成功，恢复正常
            self._state = CircuitState.CLOSED
            self._reset_counts()
            logger.info("熔断器恢复正常: success_count=%d", self._success_count)

        elif self._state == CircuitState.CLOSED:
            # 正常状态下成功，重置失败计数
            self._failure_count = 0

    def record_failure(self, error: Exception) -> ErrorPattern:
        """记录失败

        Args:
            error: 发生的错误

        Returns:
            ErrorPattern: 错误模式
        """
        # 检查是否忽略该错误类型
        error_type = type(error).__name__
        if error_type in self.config.ignored_error_types:
            return ErrorPattern(error_type=error_type, error_message=str(error))

        # 检查时间窗口，超时重置
        if self._last_failure_time:
            elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
            if elapsed > self.config.time_window_seconds:
                self._reset_counts()

        self._last_failure_time = datetime.now(timezone.utc)
        self._failure_count += 1

        # 提取错误模式
        pattern = self._extract_error_pattern(error)

        # 更新错误模式统计
        if pattern.fingerprint in self._error_patterns:
            self._error_patterns[pattern.fingerprint].increment()
        else:
            self._error_patterns[pattern.fingerprint] = pattern

        updated_pattern = self._error_patterns[pattern.fingerprint]

        # 检查是否应该熔断
        should_trip = self._should_trip(updated_pattern)

        if should_trip:
            self._trip(updated_pattern)

        logger.warning(
            "熔断器记录失败: failure_count=%d, pattern=%s, is_open=%s",
            self._failure_count, pattern.fingerprint, self.is_open
        )

        return updated_pattern

    def _extract_error_pattern(self, error: Exception) -> ErrorPattern:
        """从错误中提取模式"""
        error_type = type(error).__name__
        error_message = str(error)
        location = None

        # 尝试提取错误位置
        # 常见格式: File "xxx.py", line 123
        location_match = re.search(
            r'File ["\']([^"\']+)["\'], line (\d+)',
            error_message
        )
        if location_match:
            file_path = location_match.group(1).split("/")[-1]  # 只取文件名
            line_num = location_match.group(2)
            location = f"{file_path}:{line_num}"

        # 尝试从 traceback 提取
        if hasattr(error, "__traceback__") and error.__traceback__:
            import traceback
            tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
            tb_text = "".join(tb_lines)

            # 提取最后一个文件位置
            matches = re.findall(r'File ["\']([^"\']+)["\'], line (\d+)', tb_text)
            if matches:
                file_path = matches[-1][0].split("/")[-1]
                line_num = matches[-1][1]
                location = f"{file_path}:{line_num}"

        return ErrorPattern(
            error_type=error_type,
            error_message=error_message[:200],  # 截断长消息
            location=location,
        )

    def _should_trip(self, pattern: ErrorPattern) -> bool:
        """判断是否应该熔断"""
        # 检查总失败次数
        if self._failure_count >= self.config.failure_threshold:
            return True

        # 检查相同错误次数
        if pattern.count >= self.config.same_error_threshold:
            return True

        return False

    def _trip(self, pattern: ErrorPattern) -> None:
        """触发熔断"""
        self._state = CircuitState.OPEN
        self._opened_at = datetime.now(timezone.utc)

        logger.error(
            "熔断器已触发: failure_count=%d, pattern=%s, error_type=%s",
            self._failure_count, pattern.fingerprint, pattern.error_type
        )

    def get_breaker_error(self) -> CircuitBreakerError:
        """获取熔断错误"""
        # 找到最频繁的错误模式
        most_frequent = max(
            self._error_patterns.values(),
            key=lambda p: p.count,
            default=None
        )

        is_same_error = most_frequent and most_frequent.count >= self.config.same_error_threshold

        message = self._format_error_message(most_frequent, is_same_error)

        return CircuitBreakerError(
            message=message,
            error_pattern=most_frequent,
            failure_count=self._failure_count,
            is_same_error=is_same_error,
        )

    def _format_error_message(
        self,
        pattern: ErrorPattern | None,
        is_same_error: bool,
    ) -> str:
        """格式化错误消息"""
        if is_same_error and pattern:
            return (
                f"熔断: 相同错误重复出现 {pattern.count} 次。"
                f"错误类型: {pattern.error_type}, "
                f"消息: {pattern.error_message[:100]}"
            )
        else:
            return f"熔断: 连续失败 {self._failure_count} 次，超过阈值 {self.config.failure_threshold}"

    def _reset_counts(self) -> None:
        """重置计数"""
        self._failure_count = 0
        self._error_patterns.clear()

    def half_open(self) -> None:
        """进入半开状态"""
        if self._state == CircuitState.OPEN:
            self._state = CircuitState.HALF_OPEN
            self._opened_at = datetime.now(timezone.utc)
            logger.info("熔断器进入半开状态")

    def reset(self) -> None:
        """完全重置"""
        self._state = CircuitState.CLOSED
        self._reset_counts()
        self._last_failure_time = None
        self._opened_at = None
        logger.info("熔断器已重置")

    def get_status(self) -> dict[str, Any]:
        """获取状态"""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "error_patterns": {
                fp: {
                    "error_type": p.error_type,
                    "count": p.count,
                    "location": p.location,
                }
                for fp, p in self._error_patterns.items()
            },
            "is_open": self.is_open,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "same_error_threshold": self.config.same_error_threshold,
            },
        }


# 导出
__all__ = [
    "CircuitState",
    "ErrorPattern",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitBreaker",
]