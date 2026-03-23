"""

核心模块:
  - channels: 消息入口通道 (CLI, WebSocket)
  - gateway: 中央消息枢纽 (WebSocket 服务器 + 路由)
  - agents: AI 代理 (工具调用循环 + 流式响应)
  - tools: 技能/插件系统 (注册表 + 自动发现)
  - memory: 记忆存储 (SQLite + 向量搜索)
  - sessions: 会话管理 (上下文窗口 + 历史)
"""

__version__ = "0.1.0"
