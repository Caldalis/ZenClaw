"""
层级:  默认值 < YAML 文件 < 环境变量
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_path(path: str | Path, base: Path = PROJECT_ROOT) -> str:
    """将相对路径解析为绝对路径（基于项目根目录），绝对路径保持不变"""
    p = Path(path)
    return str(p) if p.is_absolute() else str(base / p)


class ExecutionMode(str, Enum):
    """执行模式"""
    SINGLE_AGENT = "single_agent"        # 单 Agent ReAct 循环
    MASTER_SUBAGENT = "master_subagent"  # Master-Subagent 多智能体架构


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


class SubagentConfig(BaseModel):
    """Subagent 架构配置 — 仅在 MASTER_SUBAGENT 模式下生效"""
    # 护栏配置
    max_spawn_depth: int = 5              # 最大衍生深度
    max_children_per_agent: int = 3       # 单节点最大子节点数
    max_total_agents: int = 50            # DAG 最大节点数

    # 执行配置
    default_max_steps: int = 15           # 默认最大 ReAct 步数
    default_timeout_ms: int = 120000      # 默认超时时间（毫秒）
    max_concurrent_tasks: int = 3         # 最大并发任务数

    # Worktree 配置
    enable_worktree: bool = True          # 是否启用 Git Worktree 隔离
    worktree_base_dir: str = ".agents/worktrees"  # worktree 存放目录
    auto_merge: bool = True               # 任务成功后自动合并
    auto_cleanup: bool = True             # 自动清理 worktree

    # Critic 配置
    enable_critic: bool = True            # 是否启用 Critic 警示
    enable_circuit_breaker: bool = True   # 是否启用熔断器
    circuit_breaker_threshold: int = 3    # 熔断阈值

    # 验证配置
    require_validation: bool = True       # 是否强制验证闭环
    validation_requirement: str = "any"   # any | linter | tests | both

    # Master Agent 循环配置
    max_task_graph_rounds: int = 3        # 单次消息最大 DAG 创建轮数


class Settings(BaseModel):
    """MiniClaw 全局配置 — 所有模块的配置汇总

    对标 OpenClaw 的 AppConfig，是整个系统的配置中心。
    通过 config/loader.py 加载 YAML 文件并合并环境变量。
    """
    # 执行模式
    execution_mode: ExecutionMode = ExecutionMode.SINGLE_AGENT

    # AI 提供商列表 — 按优先级排序，支持故障转移
    providers: list[ProviderConfig] = Field(
        default_factory=lambda: [ProviderConfig()]
    )
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    subagent: SubagentConfig = Field(default_factory=SubagentConfig)

    # 启用的通道
    channels: list[str] = Field(default_factory=lambda: ["cli"])
    # 技能自动发现目录（额外的）
    skill_dirs: list[str] = Field(default_factory=list)
    # 工具自动发现目录（额外的）
    tool_dirs: list[str] = Field(default_factory=list)
    # 调试模式
    debug: bool = False

    def model_post_init(self, __context) -> None:
        """将所有相对路径解析为项目根目录下的绝对路径"""
        self.memory.db_path = resolve_path(self.memory.db_path)
        self.subagent.worktree_base_dir = resolve_path(self.subagent.worktree_base_dir)
        self.skill_dirs = [resolve_path(d) for d in self.skill_dirs]
        self.tool_dirs = [resolve_path(d) for d in self.tool_dirs]

    def is_master_subagent_mode(self) -> bool:
        """是否为 Master-Subagent 模式"""
        return self.execution_mode == ExecutionMode.MASTER_SUBAGENT
