"""
网页搜索技能 — 内置技能示例
使用 DuckDuckGo 搜索 API（通过 duckduckgo-search 库），无需 API Key。
"""

from __future__ import annotations

from typing import Any

from miniclaw.skills.base import Skill


class WebSearchSkill(Skill):
    """网页搜索 — 使用 DuckDuckGo 搜索互联网"""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "搜索互联网获取最新信息。当需要查找实时信息、新闻、或你不确定的事实时使用。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                },
                "max_results": {
                    "type": "integer",
                    "description": "最大返回结果数（默认 5）",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)

        if not query:
            return "请提供搜索关键词"

        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "搜索功能不可用：需要安装 duckduckgo-search 库 (pip install duckduckgo-search)"

        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(r)

            if not results:
                return f"未找到关于 '{query}' 的结果"

            output_lines = [f"搜索 '{query}' 的结果:\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "无标题")
                body = r.get("body", "")
                href = r.get("href", "")
                output_lines.append(f"{i}. **{title}**")
                if body:
                    output_lines.append(f"   {body[:200]}")
                if href:
                    output_lines.append(f"   链接: {href}")
                output_lines.append("")

            return "\n".join(output_lines)
        except Exception as e:
            return f"搜索出错: {e}"
