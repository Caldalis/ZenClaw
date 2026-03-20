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
class ChannelType(str, Enum):
    """通道类型"""
    CLI = "cli"          # 命令行终端
    WEBSOCKET = "ws"     # WebSocket 远程客户端


class ProviderType(str, Enum):
    """AI 提供商类型 — 对标 OpenClaw 的 LLM Provider"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class EventType(str, Enum):
    """事件类型
    """
    # 流式输出事件
    TEXT_DELTA = "text_delta"          # 文本片段（流式）
    TEXT_DONE = "text_done"            # 文本完成
    # 工具调用事件
    TOOL_CALL_START = "tool_call_start"    # 开始调用工具
    TOOL_CALL_RESULT = "tool_call_result"  # 工具调用结果
    # 生命周期事件
    THINKING = "thinking"              # AI 正在思考
    ERROR = "error"                    # 错误
    DONE = "done"                      # 处理完成