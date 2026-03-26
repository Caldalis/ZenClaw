"""
技能动态加载器
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from miniclaw.utils.logging import get_logger

from .base import Tool
from .registry import ToolRegistry

logger = get_logger(__name__)


def load_builtin_tools(registry: ToolRegistry) -> None:
    """加载所有内置技能

    扫描 miniclaw.tools.builtin 包下的所有模块，
    查找 Skill 子类并注册。
    """
    import miniclaw.tools.builtin as builtin_pkg

    for importer, module_name, is_pkg in pkgutil.iter_modules(builtin_pkg.__path__):
        full_name = f"miniclaw.tools.builtin.{module_name}"
        try:
            module = importlib.import_module(full_name)
            _register_skills_from_module(module, registry)
        except Exception as e:
            logger.warning("加载内置技能 %s 失败: %s", module_name, e)


def load_tool_dirs(registry: ToolRegistry, dirs: list[str]) -> None:
    """从用户指定的目录加载自定义技能

    Args:
        dirs: 目录路径列表，每个目录下的 .py 文件会被导入
    """
    import sys

    for dir_path in dirs:
        path = Path(dir_path)
        if not path.is_dir():
            logger.warning("技能目录不存在: %s", dir_path)
            continue

        # 将目录加入 Python 路径
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

        for py_file in path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            module_name = py_file.stem
            try:
                module = importlib.import_module(module_name)
                _register_skills_from_module(module, registry)
            except Exception as e:
                logger.warning("加载自定义技能 %s 失败: %s", py_file, e)


def _register_skills_from_module(module, registry: ToolRegistry) -> None:
    """从模块中查找所有 Skill 子类并注册"""
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, Tool)
            and attr is not Tool
            and not getattr(attr, "__abstractmethods__", None)
        ):
            try:
                instance = attr()
                registry.register(instance)
            except Exception as e:
                logger.warning("实例化技能 %s 失败: %s", attr_name, e)
