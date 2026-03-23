"""
计算器技能 — 内置技能示例

安全地执行数学表达式计算。
使用 Python 的 ast.literal_eval + 有限的数学运算，避免任意代码执行。
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from miniclaw.tools.base import Skill


# 允许的运算符和函数
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "pi": math.pi, "e": math.e,
}


class CalculatorSkill(Skill):
    """安全计算器 — 执行数学表达式"""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "执行数学计算。支持四则运算、幂运算、以及 sqrt/sin/cos/log 等数学函数。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 '2 + 3 * 4' 或 'sqrt(16) + log(100)'",
                },
            },
            "required": ["expression"],
        }

    async def execute(self, **kwargs: Any) -> str:
        expression = kwargs.get("expression", "")
        try:
            result = _safe_eval(expression)
            return f"{expression} = {result}"
        except Exception as e:
            return f"计算错误: {e}"


def _safe_eval(expr: str) -> float | int:
    """安全地求值数学表达式

    通过 AST 解析确保只执行数学运算，不允许任意代码执行。
    """
    tree = ast.parse(expr.strip(), mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float | int:
    """递归求值 AST 节点"""
    if isinstance(node, ast.Constant):  # 数字字面量
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的常量: {node.value}")

    if isinstance(node, ast.BinOp):  # 二元运算: a + b
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"不支持的运算符: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _SAFE_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):  # 一元运算: -a
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"不支持的运算符: {op_type.__name__}")
        operand = _eval_node(node.operand)
        return _SAFE_OPERATORS[op_type](operand)

    if isinstance(node, ast.Call):  # 函数调用: sqrt(16)
        if not isinstance(node.func, ast.Name):
            raise ValueError("不支持复杂函数调用")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(f"不支持的函数: {func_name}")
        func = _SAFE_FUNCTIONS[func_name]
        if callable(func):
            args = [_eval_node(arg) for arg in node.args]
            return func(*args)
        return func  # 常量如 pi, e

    if isinstance(node, ast.Name):  # 变量名: pi, e
        if node.id in _SAFE_FUNCTIONS:
            val = _SAFE_FUNCTIONS[node.id]
            if not callable(val):
                return val
        raise ValueError(f"未知变量: {node.id}")

    raise ValueError(f"不支持的表达式类型: {type(node).__name__}")
