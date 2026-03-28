"""
层级:  默认值 < YAML 文件 < 环境变量
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """单个 AI 提供商配置"""
    type: str = "openai"          # "openai" | "anthropic"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    base_url: str | None = None   # 自定义端点（如 Azure OpenAI）
    max_tokens: int = 4096
    temperature: float = 0.7


class GatewayConfig(BaseModel):
    """网关配置"""
    host: str = "0.0.0.0"
    port: int = 8765
    auth_token: str = ""          # 空字符串 = 不启用认证


class MemoryConfig(BaseModel):
    """记忆存储配置"""
    db_path: str = "data/miniclaw.db"       # SQLite 数据库路径
    enable_vector: bool = False              # 是否启用向量搜索
    enable_fts: bool = True                  # 是否启用 FTS5 全文检索（BM25）
    enable_hybrid: bool = False              # 是否启用混合检索（需要 enable_vector=True）
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536                # 嵌入向量维度
    vector_weight: float = 0.7              # 混合检索中向量分数的权重
    text_weight: float = 0.3               # 混合检索中 FTS5 分数的权重
    memory_half_life_days: float = 30.0    # 记忆时间衰减半衰期（天）
    tool_result_max_bytes: int = 102400    # 工具输出最大字节数（100KB）


class AgentConfig(BaseModel):
    """Agent 配置"""
    system_prompt: str = "你是 MiniClaw，一个智能助手。你可以使用工具来帮助用户解决问题。"
    max_iterations: int = 10   # 工具调用循环最大轮数（防死循环）
    max_context_tokens: int = 8000  # 上下文窗口最大 token 数
    compaction_threshold: float = 0.85  # 触发上下文压缩的 token 使用率阈值
    compaction_keep_recent: int = 6     # 压缩时始终保留的最近消息条数


class Settings(BaseModel):
    """MiniClaw 全局配置 — 所有模块的配置汇总

    对标 OpenClaw 的 AppConfig，是整个系统的配置中心。
    通过 config/loader.py 加载 YAML 文件并合并环境变量。
    """
    # AI 提供商列表 — 按优先级排序，支持故障转移
    providers: list[ProviderConfig] = Field(
        default_factory=lambda: [ProviderConfig()]
    )
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    # 启用的通道
    channels: list[str] = Field(default_factory=lambda: ["cli"])
    # 技能自动发现目录（额外的）
    skill_dirs: list[str] = Field(default_factory=list)
    # 技能自动发现目录（额外的）
    tool_dirs: list[str] = Field(default_factory=list)
    # 调试模式
    debug: bool = False
