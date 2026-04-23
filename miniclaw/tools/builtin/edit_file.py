"""
文件编辑工具 — 精确的查找替换编辑

核心设计：old_string 必须在文件中唯一（除非使用 replace_all），
避免 LLM 误修改多处。比重写整个文件更节省 token。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miniclaw.tools.base import Tool


class EditFileTool(Tool):
    """文件编辑 — 通过查找替换精确修改文件内容"""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "通过查找替换来精确编辑文件。"
            "old_string 必须与文件中的内容精确匹配（包括缩进和空格）。"
            "默认 old_string 在文件中必须唯一，避免误修改；"
            "使用 replace_all=true 可替换所有匹配（适合变量重命名等场景）。"
            "如果 old_string 未找到，将返回错误。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径（相对或绝对路径）",
                },
                "old_string": {
                    "type": "string",
                    "description": "要替换的原始文本（必须精确匹配，包括缩进）",
                },
                "new_string": {
                    "type": "string",
                    "description": "替换后的新文本",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "是否替换所有匹配（默认 false，只替换第一个）",
                    "default": False,
                },
            },
            "required": ["path", "old_string", "new_string"],
        }

    async def execute(self, **kwargs: Any) -> str:
        file_path = kwargs.get("path", "")
        old_string = kwargs.get("old_string", "")
        new_string = kwargs.get("new_string", "")
        replace_all = kwargs.get("replace_all", False)

        if not file_path:
            return "错误：请提供文件路径"

        if not old_string:
            return "错误：old_string 不能为空"

        path = Path(file_path)

        # 安全检查必须在文件存在检查之前
        base_dir = Path.cwd().resolve()
        target_path = path.resolve()

        try:
            target_path.relative_to(base_dir)
        except ValueError:
            return f"错误：安全限制，不允许操作当前目录之外的文件。"

        if not target_path.exists():
            return f"文件不存在: {file_path}"

        if not target_path.is_file():
            return f"不是文件: {file_path}"

        # 读取文件内容
        content = _read_with_encoding(target_path)
        if content is None:
            return "错误：文件编码无法识别"

        # 查找 old_string
        count = content.count(old_string)
        if count == 0:
            # 提供上下文提示：显示最接近的行
            return _not_found_error(path, old_string, content)

        if count > 1 and not replace_all:
            return (
                f"错误：old_string 在文件中出现 {count} 次，不是唯一的。"
                f"请提供更多上下文使 old_string 唯一，或使用 replace_all=true 替换所有匹配。"
            )

        # 执行替换
        if replace_all:
            new_content = content.replace(old_string, new_string)
            replaced_count = count
        else:
            new_content = content.replace(old_string, new_string, 1)
            replaced_count = 1

        # 写回文件
        try:
            target_path.write_text(new_content, encoding="utf-8")
        except Exception as e:
            return f"错误：写入文件失败 - {e}"

        rel_path = target_path.relative_to(base_dir)
        return f"已编辑: {rel_path} (替换了 {replaced_count} 处)"


def _read_with_encoding(path: Path) -> str | None:
    encodings = ["utf-8", "gbk", "latin-1"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return None


def _not_found_error(path: Path, old_string: str, content: str) -> str:
    """构建 old_string 未找到时的错误信息，提供上下文提示"""
    lines = content.splitlines()
    # 取 old_string 的第一行作为搜索线索
    first_line = old_string.strip().splitlines()[0].strip()

    # 搜索近似行
    candidates = []
    for i, line in enumerate(lines, 1):
        if first_line in line:
            candidates.append((i, line))

    msg = f"错误：未找到 old_string，文件 {path} 中没有精确匹配的内容。"
    if candidates:
        msg += f"\n可能的近似位置（行号 {candidates[0][0]}）:\n"
        for i, line in candidates[:3]:
            msg += f"  {i}: {line}\n"
        msg += "请确保 old_string 与文件内容精确匹配（包括缩进、空格）。"
    return msg