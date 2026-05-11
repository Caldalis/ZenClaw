"""
配置加载器 — YAML 文件 + 环境变量
加载优先级: 默认值 < YAML 文件 < 环境变量
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_config(config_path: str | Path | None = None) -> Settings:
    """加载配置

    Args:
        config_path: YAML 配置文件路径。None 则按顺序搜索默认位置。

    Returns:
        合并后的 Settings 对象
    """
    data: dict = {}

    # Step 1: 加载 YAML 文件
    if config_path is None:
        # 按优先级搜索默认位置
        for candidate in ["config.yaml", "config.yml", "config.example.yaml"]:
            if Path(candidate).exists():
                config_path = candidate
                break
            project_path = PROJECT_ROOT / candidate
            if project_path.exists():
                config_path = str(project_path)
                break

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    # Step 2: 环境变量覆盖（MINICLAW_ 前缀）
    # 支持的环境变量:
    #   MINICLAW_OPENAI_API_KEY    → providers[0].api_key (openai)
    #   MINICLAW_ANTHROPIC_API_KEY → providers[1].api_key (anthropic)
    #   MINICLAW_AUTH_TOKEN        → gateway.auth_token
    #   MINICLAW_DEBUG             → debug
    _apply_env_overrides(data)

    return Settings(**data)


def _apply_env_overrides(data: dict) -> None:
    """将环境变量覆盖到配置字典中"""

    # OpenAI API Key
    openai_key = os.environ.get("MINICLAW_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if openai_key:
        providers = data.setdefault("providers", [{}])
        # 找到 openai 类型的 provider，或修改第一个
        for p in providers:
            if p.get("type", "openai") == "openai":
                p["api_key"] = openai_key
                break

    # Anthropic API Key
    anthropic_key = os.environ.get("MINICLAW_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        providers = data.setdefault("providers", [{}])
        for p in providers:
            if p.get("type") == "anthropic":
                p["api_key"] = anthropic_key
                break

    # OpenAI Base URL (支持自定义端点)
    base_url = os.environ.get("MINICLAW_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if base_url:
        providers = data.setdefault("providers", [{}])
        for p in providers:
            if p.get("type", "openai") == "openai":
                p["base_url"] = base_url
                break

    # Gateway auth token
    auth_token = os.environ.get("MINICLAW_AUTH_TOKEN")
    if auth_token:
        data.setdefault("gateway", {})["auth_token"] = auth_token

    # Debug mode
    debug = os.environ.get("MINICLAW_DEBUG")
    if debug and debug.lower() in ("1", "true", "yes"):
        data["debug"] = True
    execution_mode = os.environ.get("MINICLAW_EXECUTION_MODE")
    if execution_mode:
        data["execution_mode"] = execution_mode
