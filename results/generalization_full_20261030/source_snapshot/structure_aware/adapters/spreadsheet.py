"""Named ranges and spreadsheet formulas, lowered without a custom executor."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..sarr import Graph, Node, Operation


@dataclass(frozen=True)
class Workbook:
    ranges: Mapping[str, np.ndarray | float]
    formulas: Mapping[str, str]
    outputs: tuple[str, ...]


class SpreadsheetAdapter:
    operations = {
        "constant": Operation(lambda value: value, pure=True),
        "identity": Operation(lambda x: x, pure=True),
        "add": Operation(lambda x, y: x + y, pure=True),
        "subtract": Operation(lambda x, y: x - y, pure=True),
        "multiply": Operation(lambda x, y: x * y, pure=True),
        "divide": Operation(lambda x, y: x / y, pure=True),
        "negative": Operation(lambda x: -x, pure=True),
        "sum": Operation(lambda x: np.sum(x), pure=True),
        "mean": Operation(lambda x: np.mean(x), pure=True),
        "minimum": Operation(lambda x: np.min(x), pure=True),
        "maximum": Operation(lambda x: np.max(x), pure=True),
    }

    def build(self, request: Workbook) -> Graph:
        nodes, sizes, visiting = [], {}, set()
        if set(request.ranges) & set(request.formulas):
            raise ValueError("a range and formula cannot have the same name")
        for name, value in request.ranges.items():
            sizes[name] = int(np.size(value))
        binary = {ast.Add: "add", ast.Sub: "subtract", ast.Mult: "multiply",
                  ast.Div: "divide"}
        functions = {"sum": "sum", "mean": "mean", "min": "minimum", "max": "maximum"}

        def cell(name):
            if name in sizes:
                return name, sizes[name]
            if name not in request.formulas:
                raise ValueError(f"unknown spreadsheet reference: {name}")
            if name in visiting:
                raise ValueError("cyclic spreadsheet formula")
            visiting.add(name)
            text = request.formulas[name].lstrip("=")
            counter = 0

            def expression(tree):
                nonlocal counter
                if isinstance(tree, ast.Name):
                    return cell(tree.id)
                counter += 1
                uid = f"{name}::expression:{counter}"
                params = {}
                if isinstance(tree, ast.Constant) and type(tree.value) in (int, float):
                    op, dependencies, size = "constant", (), 1
                    params = {"value": tree.value}
                elif isinstance(tree, ast.BinOp) and type(tree.op) in binary:
                    left, nl = expression(tree.left)
                    right, nr = expression(tree.right)
                    op, dependencies, size = binary[type(tree.op)], (left, right), max(nl, nr)
                elif isinstance(tree, ast.UnaryOp) and isinstance(tree.op, (ast.USub, ast.UAdd)):
                    value, size = expression(tree.operand)
                    op = "negative" if isinstance(tree.op, ast.USub) else "identity"
                    dependencies = (value,)
                elif (isinstance(tree, ast.Call) and isinstance(tree.func, ast.Name)
                      and tree.func.id.lower() in functions and len(tree.args) == 1
                      and not tree.keywords):
                    value, length = expression(tree.args[0])
                    op, dependencies, size = functions[tree.func.id.lower()], (value,), 1
                    nodes.append(Node(uid, op, dependencies, work=max(1, length)))
                    return uid, size
                else:
                    raise ValueError("formula uses unsupported syntax; no Python eval is used")
                nodes.append(Node(uid, op, dependencies, params, max(1, size)))
                return uid, size

            result, length = expression(ast.parse(text, mode="eval").body)
            nodes.append(Node(name, "identity", (result,), work=max(1, length)))
            sizes[name] = length
            visiting.remove(name)
            return name, length

        # Include all declared formulas, so output-ancestor pruning is observable.
        for name in request.formulas:
            cell(name)
        return Graph("spreadsheet-v1", request.ranges, tuple(nodes), request.outputs)


def construct(seed=7, *, regions=12, rows=65536, views=5,
              duplicate=True) -> Workbook:
    rng = np.random.default_rng(seed)
    ranges, formulas = {}, {}
    for i in range(regions):
        ranges[f"Revenue_{i}"] = rng.uniform(80, 180, rows)
        ranges[f"Expense_{i}"] = rng.uniform(20, 70, rows)
        for j in range(views if duplicate else 1):
            # Different report cells reuse the same net-profit expression.
            formulas[f"Report_{i}_{j}"] = (
                f"sum((Revenue_{i} - Expense_{i}) * 0.9) * {j + 1} / 100.0")
    return Workbook(ranges, formulas, tuple(formulas))


def changed(book: Workbook, fraction: float) -> Workbook:
    ranges = dict(book.ranges)
    count = int(np.ceil(len(ranges) / 2 * fraction))
    for i in range(count):
        ranges[f"Revenue_{i}"] = ranges[f"Revenue_{i}"] + 0.125
    return Workbook(ranges, book.formulas, book.outputs)


def expected(book: Workbook) -> dict[str, float]:
    # Independent, direct business-report formula (does not use graph execution).
    result = {}
    for name in book.outputs:
        _, i, j = name.split("_")
        net = (book.ranges[f"Revenue_{i}"] - book.ranges[f"Expense_{i}"]) * 0.9
        result[name] = np.sum(net) * (int(j) + 1) / 100.0
    return result
