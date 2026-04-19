"""
Agents 模块 — Agent 实现与多智能体架构

提供:
  - Agent: 核心 ReAct 循环 Agent
  - MasterAgent: Master-Subagent 主调度器
  - SubagentFactory: Subagent 角色实例化
  - SubagentExecutor: Subagent 执行器
  - SubagentOrchestrator: Subagent 编排器
  - ContextIsolator: 上下文隔离器
"""

from miniclaw.agents.agent import Agent
from miniclaw.agents.master_agent import MasterAgent, create_master_agent_config
from miniclaw.agents.subagent_factory import SubagentFactory, SubagentConfig
from miniclaw.agents.subagent_executor import SubagentExecutor, SubagentExecutionContext
from miniclaw.agents.subagent_orchestrator import SubagentOrchestrator, OrchestratorState
from miniclaw.agents.context_isolator import ContextIsolator, IsolatedContext, IsolationConfig

__all__ = [
    "Agent",
    "MasterAgent",
    "create_master_agent_config",
    "SubagentFactory",
    "SubagentConfig",
    "SubagentExecutor",
    "SubagentExecutionContext",
    "SubagentOrchestrator",
    "OrchestratorState",
    "ContextIsolator",
    "IsolatedContext",
    "IsolationConfig",
]