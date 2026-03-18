"""
枚举定义
str + Enum 实现相同语义，同时兼容 JSON 序列化。
test
"""

from enum import Enum


class Role(str, Enum):
    """消息角色 — 对标 OpenClaw 的 MessageRole"""
    SYSTEM = "system"        # 系统提示词
    USER = "user"            # 用户输入
    ASSISTANT = "assistant"  # AI 回复
    TOOL = "tool"            # 工具调用结果

