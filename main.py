"""
主入口 — 双模式启动器

启动模式:
  1. single_agent: 单 Agent ReAct 循环（原有模式）
  2. master_subagent: Master-Subagent 多智能体架构（新模式）

启动流程:
  1. 加载配置 (YAML + 环境变量)
  2. 根据执行模式初始化组件
  3. 启动 Gateway WebSocket 服务器（如果配置了 ws 通道）
  4. 启动 CLI 通道（如果配置了 cli 通道）
  5. 等待退出信号
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

from miniclaw.agents.agent import Agent
from miniclaw.agents.master_agent import MasterAgent, create_master_agent_config
from miniclaw.agents.providers.registry import ProviderRegistry
from miniclaw.agents.subagent_orchestrator import SubagentOrchestrator, create_orchestrator
from miniclaw.channels.cli_channel import CLIChannel
from miniclaw.config.loader import load_config
from miniclaw.config.settings import Settings, ExecutionMode
from miniclaw.dispatcher.subagent_registry import SubagentRegistry, GuardrailConfig
from miniclaw.dispatcher.event_bus import EventBus, TurnLogStore
from miniclaw.dispatcher.dispatcher import Dispatcher
from miniclaw.gateway.server import GatewayServer
from miniclaw.memory.embeddings import EmbeddingGenerator
from miniclaw.memory.hybrid_store import HybridMemoryStore
from miniclaw.memory.sqlite_store import SQLiteMemoryStore
from miniclaw.memory.vector_store import VectorMemoryStore
from miniclaw.sessions.manager import SessionManager
from miniclaw.tools.loader import load_builtin_tools, load_tool_dirs
from miniclaw.tools.registry import ToolRegistry
from miniclaw.tools.skill_md_loader import SkillManager
from miniclaw.tools.builtin.load_skill_tool import LoadSkillTool, set_skill_manager
from miniclaw.utils.logging import setup_logging, get_logger


async def bootstrap_single_agent(config: Settings) -> dict:
    """初始化单 Agent 模式

    这是原有的 ReAct 循环模式。
    """
    logger = get_logger("bootstrap.single_agent")
    logger.info("初始化单 Agent 模式...")

    # 1. 记忆存储
    memory = await _init_memory(config, logger)

    # 2. 会话管理
    session_mgr = SessionManager(memory, config.memory.db_path)
    await session_mgr.initialize()
    logger.info("会话管理器已初始化")

    # 3. 技能系统
    tool_registry = await _init_tools(config, logger)

    # 4. AI 提供商
    provider_registry = ProviderRegistry(config.providers)
    logger.info("AI 提供商已初始化: %d 个", provider_registry.provider_count)

    # 5. Agent
    agent = Agent(
        config=config.agent,
        memory_config=config.memory,
        provider_registry=provider_registry,
        tool_registry=tool_registry,
        session_manager=session_mgr,
        memory_store=memory,
    )
    logger.info("Agent 已初始化")

    return {
        "memory": memory,
        "session_mgr": session_mgr,
        "tool_registry": tool_registry,
        "provider_registry": provider_registry,
        "agent": agent,
        "mode": "single_agent",
    }


async def bootstrap_master_subagent(config: Settings) -> dict:
    """初始化 Master-Subagent 模式

    """
    logger = get_logger("bootstrap.master_subagent")
    logger.info("初始化 Master-Subagent 模式...")

    # 1. 记忆存储
    memory = await _init_memory(config, logger)

    # 2. 会话管理
    session_mgr = SessionManager(memory, config.memory.db_path)
    await session_mgr.initialize()
    logger.info("会话管理器已初始化")

    # 3. 技能系统
    tool_registry = await _init_tools(config, logger)

    # 4. AI 提供商
    provider_registry = ProviderRegistry(config.providers)
    logger.info("AI 提供商已初始化: %d 个", provider_registry.provider_count)

    # 5. 基础 Agent（Master Agent 底层使用）
    master_config = create_master_agent_config()
    base_agent = Agent(
        config=master_config,
        memory_config=config.memory,
        provider_registry=provider_registry,
        tool_registry=tool_registry,
        session_manager=session_mgr,
        memory_store=memory,
    )
    logger.info("基础 Agent 已初始化")

    # 6. Subagent Orchestrator
    repo_root = Path.cwd()
    orchestrator = await create_orchestrator(
        settings=config,
        agent=base_agent,
        session_manager=session_mgr,
        memory_store=memory,
        repo_root=repo_root,
    )
    logger.info("Subagent Orchestrator 已初始化")

    # 7. Master Agent
    master_agent = MasterAgent(
        base_agent=base_agent,
        orchestrator=orchestrator,
        settings=config,
    )
    logger.info("Master Agent 已初始化")

    return {
        "memory": memory,
        "session_mgr": session_mgr,
        "tool_registry": tool_registry,
        "provider_registry": provider_registry,
        "base_agent": base_agent,
        "orchestrator": orchestrator,
        "master_agent": master_agent,
        "mode": "master_subagent",
    }


async def _init_memory(config: Settings, logger) -> "MemoryStore":
    """初始化记忆存储"""
    mem_cfg = config.memory

    def _make_embedder() -> EmbeddingGenerator | None:
        for p in config.providers:
            if p.type == "openai" and p.api_key:
                return EmbeddingGenerator(
                    api_key=p.api_key,
                    model=mem_cfg.embedding_model,
                    base_url=p.base_url,
                )
        return None

    if mem_cfg.enable_hybrid and mem_cfg.enable_vector:
        embedder = _make_embedder()
        if embedder:
            memory = HybridMemoryStore(
                db_path=mem_cfg.db_path,
                embedding_generator=embedder,
                embedding_dim=mem_cfg.embedding_dim,
                vector_weight=mem_cfg.vector_weight,
                text_weight=mem_cfg.text_weight,
                memory_half_life_days=mem_cfg.memory_half_life_days,
            )
            logger.info("使用混合记忆存储 (FTS5 + 向量 + MMR + 时间衰减)")
        else:
            memory = SQLiteMemoryStore(mem_cfg.db_path, enable_fts=mem_cfg.enable_fts)
            logger.warning("混合存储需要 OpenAI API Key，降级到 SQLite+FTS")
    elif mem_cfg.enable_vector:
        embedder = _make_embedder()
        if embedder:
            memory = VectorMemoryStore(mem_cfg.db_path, embedder, mem_cfg.embedding_dim)
            logger.info("使用向量记忆存储")
        else:
            memory = SQLiteMemoryStore(mem_cfg.db_path, enable_fts=mem_cfg.enable_fts)
            logger.warning("向量存储已启用但缺少 OpenAI API Key，降级到 SQLite+FTS")
    else:
        memory = SQLiteMemoryStore(mem_cfg.db_path, enable_fts=mem_cfg.enable_fts)
        logger.info("使用 SQLite 记忆存储 (FTS5=%s)", mem_cfg.enable_fts)

    await memory.initialize()
    logger.info("记忆存储已初始化")
    return memory


async def _init_tools(config: Settings, logger) -> ToolRegistry:
    """初始化技能系统"""
    # SKILL.md 文件懒加载系统
    skill_md_dirs = config.skill_dirs.copy() if config.skill_dirs else []
    if not any("skills" in d for d in skill_md_dirs):
        skill_md_dirs.append("skills")
    skill_manager = SkillManager(skill_md_dirs)
    set_skill_manager(skill_manager)
    logger.info("SKILL.md 懒加载系统已初始化: %d 个技能", len(skill_manager.available_skills))

    # 预注入: 将技能清单追加到系统提示词
    preinject = skill_manager.get_preinject_content()
    if preinject:
        config.agent.system_prompt += preinject

    # 技能系统（Python class based）
    tool_registry = ToolRegistry()
    load_builtin_tools(tool_registry)
    if config.tool_dirs:
        load_tool_dirs(tool_registry, config.tool_dirs)
    logger.info("tools系统已初始化: %d 个技能 (含 %d 个 SKILL.md)", tool_registry.tool_count, len(skill_manager.available_skills))

    return tool_registry


async def bootstrap(config: Settings) -> dict:
    """初始化所有模块并返回核心组件

    根据配置的执行模式选择初始化路径。
    """
    logger = get_logger("bootstrap")
    logger.info("正在初始化 MiniClaw v0.2.0...")
    logger.info("执行模式: %s", config.execution_mode.value)

    if config.execution_mode == ExecutionMode.MASTER_SUBAGENT:
        return await bootstrap_master_subagent(config)
    else:
        return await bootstrap_single_agent(config)


async def run(config_path: str | None = None) -> None:
    """主运行函数"""
    # 加载配置
    config = load_config(config_path)
    setup_logging(config.debug)
    logger = get_logger("main")

    # 初始化
    components = await bootstrap(config)

    # 根据模式选择 Agent
    if components["mode"] == "master_subagent":
        agent = components["master_agent"]
        orchestrator = components["orchestrator"]
        logger.info("使用 Master-Subagent 模式")
    else:
        agent = components["agent"]
        orchestrator = None
        logger.info("使用单 Agent 模式")

    session_mgr = components["session_mgr"]
    memory = components["memory"]

    tasks = []

    try:
        # 启动 Gateway（如果有 ws 通道）
        gateway = None
        if "ws" in config.channels:
            gateway = GatewayServer(config.gateway, agent, session_mgr)
            await gateway.start()

        # 启动 CLI 通道
        if "cli" in config.channels:
            cli = CLIChannel(agent, session_mgr)
            await cli.start()  # 这会阻塞直到用户退出
        else:
            # 没有 CLI 通道时，保持运行直到收到信号
            logger.info("MiniClaw 正在运行... (Ctrl+C 退出)")
            stop_event = asyncio.Event()

            def _signal_handler():
                stop_event.set()

            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, _signal_handler)
                except NotImplementedError:
                    # Windows 不支持 add_signal_handler
                    pass

            await stop_event.wait()

    except KeyboardInterrupt:
        logger.info("收到中断信号")
    finally:
        # 清理
        logger.info("正在关闭...")

        # 关闭 orchestrator（如果有）
        if orchestrator:
            await orchestrator.shutdown()

        if gateway:
            await gateway.stop()
        await session_mgr.close()
        await memory.close()
        logger.info("MiniClaw 已关闭")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="MiniClaw — AI 助手平台")
    parser.add_argument("-c", "--config", help="配置文件路径", default=None)
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument(
        "--mode",
        choices=["single_agent", "master_subagent"],
        help="执行模式 (覆盖配置文件)",
    )
    args = parser.parse_args()

    # debug 模式通过环境变量传递
    if args.debug:
        import os
        os.environ["MINICLAW_DEBUG"] = "1"

    # 执行模式覆盖
    if args.mode:
        import os
        os.environ["MINICLAW_EXECUTION_MODE"] = args.mode

    asyncio.run(run(args.config))


if __name__ == "__main__":
    main()