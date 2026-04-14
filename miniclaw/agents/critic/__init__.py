"""
Critic 模块 — 验证闭环与熔断机制

提供:
  - CircuitBreaker: 熔断器，防止无限重试相同错误
  - CriticInjector: 错误警示注入器
  - RunLinter/RunTests: 验证工具
  - ValidationGatekeeper: 验证门禁器
"""

from miniclaw.agents.critic.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    ErrorPattern,
)
from miniclaw.agents.critic.critic_injector import (
    CriticConfig,
    CriticInjector,
    FailureContext,
)
from miniclaw.agents.critic.validation_tools import (
    RunLinterTool,
    RunTestsTool,
    ValidationResult,
    ValidationStatus,
)
from miniclaw.agents.critic.validation_gatekeeper import (
    GatekeeperConfig,
    ValidationAwareSubmitTool,
    ValidationGatekeeper,
    ValidationRecord,
    ValidationRequirement,
)

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitState",
    "ErrorPattern",
    # Critic Injector
    "CriticInjector",
    "CriticConfig",
    "FailureContext",
    # Validation Tools
    "RunLinterTool",
    "RunTestsTool",
    "ValidationResult",
    "ValidationStatus",
    # Validation Gatekeeper
    "ValidationGatekeeper",
    "GatekeeperConfig",
    "ValidationRequirement",
    "ValidationRecord",
    "ValidationAwareSubmitTool",
]