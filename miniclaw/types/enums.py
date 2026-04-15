"""
枚举定义
str + Enum 实现相同语义，同时兼容 JSON 序列化。
"""

from enum import Enum


class Role(str, Enum):
    """消息角色 — 对标 OpenClaw 的 MessageRole"""
    SYSTEM = "system"        # 系统提示词
    USER = "user"            # 用户输入
    ASSISTANT = "assistant"  # AI 回复
    TOOL = "tool"            # 工具调用结果


class EventType(str, Enum):
    """事件类型 — 对标 OpenClaw 的 EventType

    事件驱动是 OpenClaw 的核心模式：Agent 处理消息后不直接返回字符串，
    而是产出一系列 Event，让上层（Gateway/Channel）决定如何呈现。
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


class ChannelType(str, Enum):
    """通道类型"""
    CLI = "cli"          # 命令行终端
    WEBSOCKET = "ws"     # WebSocket 远程客户端


class ProviderType(str, Enum):
    """AI 提供商类型 — 对标 OpenClaw 的 LLM Provider"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class SessionStatus(str, Enum):
    """会话状态"""
    ACTIVE = "active"
    ARCHIVED = "archived"


class TurnStatus(str, Enum):
    """Turn 执行状态 — 用于多 Agent 架构的状态追踪"""
    PENDING = "pending"        # 等待执行
    RUNNING = "running"        # 正在执行
    COMPLETED = "completed"    # 成功完成
    FAILED = "failed"          # 执行失败
    TIMEOUT = "timeout"        # 超时终止
    INTERRUPTED = "interrupted"  # 被中断（如进程崩溃）


class AgentRole(str, Enum):
    """Agent 角色类型 — 用于 Subagent 角色分化"""
    MASTER = "master"          # 主 Agent（调度器）
    CODER = "coder"            # 编码执行者
    SEARCHER = "searcher"      # 搜索/研究执行者
    REVIEWER = "reviewer"      # 代码审查执行者
    TESTER = "tester"          # 测试执行者
    PLANNER = "planner"        # 规划执行者
    GENERIC = "generic"        # 通用执行者
