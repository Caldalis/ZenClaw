"""
网页搜索技能 — 内置技能示例
使用 DuckDuckGo 搜索 API（通过 ddgs 库），无需 API Key。
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from miniclaw.tools.base import Tool


# CJK 字符范围（含中日韩）—— 用于自动选择 region
_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-힯]")


def _default_region(query: str) -> str:
    """根据 query 内容自动选 region。

    - 含 CJK 字符 → cn-zh（中文搜索默认 us-en 命中率极差）
    - 其他 → wt-wt（全球，无地区偏好；比 us-en 更通用）
    """
    return "cn-zh" if _CJK_RE.search(query or "") else "wt-wt"


class WebSearchSkill(Tool):
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
                    "description": "最大返回结果数（默认 8）",
                    "default": 8,
                },
                "region": {
                    "type": "string",
                    "description": (
                        "搜索地区。不传则按 query 自动选："
                        "含中文 → cn-zh；其他 → wt-wt（全球）。"
                        "可选：cn-zh, us-en, wt-wt, jp-jp, kr-kr 等。"
                    ),
                },
                "safesearch": {
                    "type": "string",
                    "description": "安全搜索过滤：on / moderate / off（默认 moderate）",
                    "default": "moderate",
                },
                "backend": {
                    "type": "string",
                    "description": (
                        "搜索后端，默认 auto（多引擎自动轮询）。"
                        "可选：auto / bing / duckduckgo / brave / google / yandex。"
                        "限速时本工具会自动切换 backend 重试一次。"
                    ),
                    "default": "auto",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 8)
        region = kwargs.get("region") or _default_region(query)
        safesearch = kwargs.get("safesearch", "moderate")
        backend = kwargs.get("backend", "auto")

        if not query:
            return "请提供搜索关键词"

        try:
            from ddgs import DDGS
            from ddgs.exceptions import (
                DDGSException,
                RatelimitException,
                TimeoutException as DDGSTimeoutException,
            )
        except ImportError:
            return "搜索功能不可用：需要安装 ddgs 库 (pip install ddgs)"

        def _do_search(use_backend: str) -> list[dict[str, Any]]:
            with DDGS() as ddgs:
                return ddgs.text(
                    query,
                    region=region,
                    safesearch=safesearch,
                    backend=use_backend,
                    max_results=max_results,
                )

        async def _run(use_backend: str) -> list[dict[str, Any]] | None:
            """单次搜索，15s 硬上限。

            返回:
                list - 找到的结果
                None - 明确"无结果"（DDGSException 的 "No results found"）

            异常:
                RatelimitException / DDGSTimeoutException / asyncio.TimeoutError
                    交给外层处理（重试或转译）
                其他 DDGSException
                    向上抛出
            """
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(_do_search, use_backend),
                    timeout=15.0,
                )
            except RatelimitException:
                raise
            except DDGSTimeoutException:
                raise
            except DDGSException as e:
                if "no results found" in str(e).lower():
                    return None
                raise

        try:
            try:
                results = await _run(backend)
            except RatelimitException:
                # 限速 → 退避 1.5s 换 backend 再试一次
                await asyncio.sleep(1.5)
                fallback = "bing" if backend == "auto" else "auto"
                try:
                    results = await _run(fallback)
                except RatelimitException:
                    return (
                        f"搜索被限速：'{query}' 在所有可用搜索引擎上均触发限速。"
                        f"这通常是临时的，但当前无法获取数据。"
                        f"建议把这个事实写进 summary（不要再换关键词重试）。"
                    )

            if results is None:
                return (
                    f"未找到关于 '{query}' 的结果。"
                    f"这是搜索引擎的明确返回——继续换关键词搜大概率仍然无果，"
                    f"建议把'未找到'作为事实纳入结论。"
                )

            output_lines = [f"搜索 '{query}' 的结果:\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "无标题")
                body = r.get("body", "")
                href = r.get("href", "")

                output_lines.append(f"{i}. **{title}**")
                if body:
                    # 截断结果以节省大模型的 Token 上下文
                    output_lines.append(f"   {body[:200]}...")
                if href:
                    output_lines.append(f"   链接: {href}")
                output_lines.append("")

            return "\n".join(output_lines)
        except DDGSTimeoutException:
            return (
                f"搜索超时（搜索引擎响应慢）：'{query}'。"
                f"建议简化关键词或承认当前数据源不可用。"
            )
        except asyncio.TimeoutError:
            return (
                f"搜索超时（>15s）：'{query}'。"
                f"建议简化关键词或承认当前数据源不可用。"
            )
        except Exception as e:
            return f"搜索出错: {type(e).__name__}: {e}"
