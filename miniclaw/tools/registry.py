"""
技能注册表
"""

from __future__ import annotations

from typing import Any

from miniclaw.utils.logging import get_logger

from .base import Tool

logger = get_logger(__name__)

# 全局注册表实例
_global_registry: ToolRegistry | None = None


class ToolRegistry:
    """技能注册表 — 管理所有已注册的技能

    使用方式:
      registry = ToolRegistry()
      registry.register(CalculatorSkill())
      registry.register(WeatherSkill())

      # Agent 使用
      schemas = registry.get_tool_schemas()  # 传给 AI API
      skill = registry.get("calculator")      # 执行时查找
    """

    def __init__(self):
        self._skills: dict[str, Tool] = {}

    def register(self, skill: Tool) -> None:
        if skill.name in self._skills:
            logger.warning("技能 '%s' 已注册，将被覆盖", skill.name)
        self._skills[skill.name] = skill
        logger.info("注册技能: %s — %s", skill.name, skill.description)

    def unregister(self, name: str) -> None:
        """取消注册一个技能"""
        self._skills.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """按名称获取技能"""
        return self._skills.get(name)

    def list_tools(self) -> list[Tool]:
        """列出所有已注册的技能"""
        return list(self._skills.values())

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """生成所有技能的 OpenAI tool schema 列表

        返回值直接传给 AI API 的 tools 参数:
          response = openai.chat.completions.create(
              ...,
              tools=registry.get_tool_schemas()
          )
        """
        return [skill.to_tool_schema() for skill in self._skills.values()]

    @property
    def tool_count(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills


def get_global_registry() -> ToolRegistry:
    """获取全局技能注册表（单例）"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


# 装饰器注册


def tool_decorator(
    name: str,
    description: str,
    parameters: dict[str, Any],
):
    """技能装饰器 — 将普通 async 函数包装为 Tool 并注册

    示例:
        @tool_decorator(
            name="calculator",
            description="执行数学计算",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        )
        async def calculator(expression: str) -> str:
            return str(eval(expression))
    """
    def decorator(func):

        skill_cls = type(
            f"{name.title()}Tool",
            (Tool,),
            {
                "name": property(lambda self: name),
                "description": property(lambda self: description),
                "parameters": property(lambda self: parameters),
                "execute": lambda self, **kw: func(**kw),
            },
        )
        # 注册到全局
        get_global_registry().register(skill_cls())
        return func
    return decorator
