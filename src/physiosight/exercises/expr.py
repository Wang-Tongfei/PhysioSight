"""A tiny, safe expression language for exercise specs (no ``eval``).

Grammar (precedence low -> high)::

    expr    := or
    or      := and ('or' and)*
    and     := not ('and' not)*
    not     := 'not' not | cmp
    cmp     := sum (('<'|'<='|'>'|'>='|'=='|'!=') sum)?
    sum     := prod (('+'|'-') prod)*
    prod    := unary (('*'|'/') unary)*
    unary   := '-' unary | atom
    atom    := NUMBER | 'true' | 'false' | NAME | NAME '(' args ')' | '(' expr ')'
    NAME    := identifier, optionally dotted: ``p.max_asymmetry``

Functions: ``abs, min, max, mean``. Booleans are numbers (true=1.0) so that
derived conditions can be reused inside other expressions. Any NaN operand
in a comparison makes it false, so a missing feature never satisfies a
condition.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Union

_TOKEN = re.compile(
    r"\s*(?:(?P<num>\d+(?:\.\d*)?|\.\d+)|(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
    r"|(?P<op><=|>=|==|!=|[-+*/<>(),]))"
)

_FUNCS: dict[str, Callable[..., float]] = {
    "abs": lambda x: abs(x),
    "min": lambda *xs: min(xs),
    "max": lambda *xs: max(xs),
    "mean": lambda *xs: sum(xs) / len(xs),
}
_ARITY = {"abs": (1, 1), "min": (2, 16), "max": (2, 16), "mean": (1, 16)}
_KEYWORDS = {"and", "or", "not", "true", "false"}


class ExprError(ValueError):
    pass


@dataclass(frozen=True)
class Num:
    value: float


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Call:
    func: str
    args: tuple[Node, ...]


@dataclass(frozen=True)
class Unary:
    op: str
    operand: Node


@dataclass(frozen=True)
class Binary:
    op: str
    left: Node
    right: Node


Node = Union[Num, Var, Call, Unary, Binary]


def _tokenize(src: str) -> list[tuple[str, str]]:
    pos, out = 0, []
    src = src.strip()
    while pos < len(src):
        m = _TOKEN.match(src, pos)
        if not m or m.end() == pos:
            raise ExprError(f"unexpected character at {pos}: {src[pos:pos + 10]!r}")
        kind = m.lastgroup
        assert kind is not None
        out.append((kind, m.group(kind)))
        pos = m.end()
    out.append(("end", ""))
    return out


class _Parser:
    def __init__(self, src: str) -> None:
        self.src = src
        self.toks = _tokenize(src)
        self.i = 0

    def peek(self) -> tuple[str, str]:
        return self.toks[self.i]

    def take(self, value: str | None = None) -> tuple[str, str]:
        tok = self.toks[self.i]
        if value is not None and tok[1] != value:
            raise ExprError(f"expected {value!r}, got {tok[1] or 'end'!r} in {self.src!r}")
        self.i += 1
        return tok

    def parse(self) -> Node:
        node = self.or_()
        if self.peek()[0] != "end":
            raise ExprError(f"unexpected {self.peek()[1]!r} in {self.src!r}")
        return node

    def or_(self) -> Node:
        node = self.and_()
        while self.peek() == ("name", "or"):
            self.take()
            node = Binary("or", node, self.and_())
        return node

    def and_(self) -> Node:
        node = self.not_()
        while self.peek() == ("name", "and"):
            self.take()
            node = Binary("and", node, self.not_())
        return node

    def not_(self) -> Node:
        if self.peek() == ("name", "not"):
            self.take()
            return Unary("not", self.not_())
        return self.cmp()

    def cmp(self) -> Node:
        node = self.sum_()
        if self.peek()[1] in ("<", "<=", ">", ">=", "==", "!="):
            op = self.take()[1]
            node = Binary(op, node, self.sum_())
            if self.peek()[1] in ("<", "<=", ">", ">=", "==", "!="):
                raise ExprError(f"chained comparisons are not supported: {self.src!r}")
        return node

    def sum_(self) -> Node:
        node = self.prod()
        while self.peek()[1] in ("+", "-") and self.peek()[0] == "op":
            op = self.take()[1]
            node = Binary(op, node, self.prod())
        return node

    def prod(self) -> Node:
        node = self.unary()
        while self.peek()[1] in ("*", "/") and self.peek()[0] == "op":
            op = self.take()[1]
            node = Binary(op, node, self.unary())
        return node

    def unary(self) -> Node:
        if self.peek() == ("op", "-"):
            self.take()
            return Unary("-", self.unary())
        return self.atom()

    def atom(self) -> Node:
        kind, val = self.take()
        if kind == "num":
            return Num(float(val))
        if kind == "name":
            if val == "true":
                return Num(1.0)
            if val == "false":
                return Num(0.0)
            if val in _KEYWORDS:
                raise ExprError(f"unexpected keyword {val!r} in {self.src!r}")
            if "__" in val:
                raise ExprError(f"names may not contain '__': {val!r}")
            if self.peek() == ("op", "("):
                if val not in _FUNCS:
                    raise ExprError(f"unknown function {val!r} in {self.src!r}")
                self.take("(")
                args = [self.or_()]
                while self.peek() == ("op", ","):
                    self.take()
                    args.append(self.or_())
                self.take(")")
                lo, hi = _ARITY[val]
                if not lo <= len(args) <= hi:
                    raise ExprError(f"{val}() takes {lo}..{hi} arguments in {self.src!r}")
                return Call(val, tuple(args))
            return Var(val)
        if val == "(":
            node = self.or_()
            self.take(")")
            return node
        raise ExprError(f"unexpected {val or 'end'!r} in {self.src!r}")


@dataclass(frozen=True)
class Expr:
    source: str
    ast: Node

    @classmethod
    def parse(cls, source: str | float | int | bool) -> Expr:
        if isinstance(source, bool):
            source = "true" if source else "false"
        src = str(source)
        return cls(src, _Parser(src).parse())

    def names(self) -> set[str]:
        out: set[str] = set()

        def walk(n: Node) -> None:
            if isinstance(n, Var):
                out.add(n.name)
            elif isinstance(n, Call):
                for a in n.args:
                    walk(a)
            elif isinstance(n, Unary):
                walk(n.operand)
            elif isinstance(n, Binary):
                walk(n.left)
                walk(n.right)

        walk(self.ast)
        return out

    def evaluate(self, env: Mapping[str, float]) -> float:
        return _eval(self.ast, env)

    def truth(self, env: Mapping[str, float]) -> bool:
        v = self.evaluate(env)
        return not math.isnan(v) and v != 0.0


def _eval(n: Node, env: Mapping[str, float]) -> float:
    if isinstance(n, Num):
        return n.value
    if isinstance(n, Var):
        return float(env[n.name])
    if isinstance(n, Call):
        args = [_eval(a, env) for a in n.args]
        if any(math.isnan(a) for a in args):
            return math.nan
        return float(_FUNCS[n.func](*args))
    if isinstance(n, Unary):
        v = _eval(n.operand, env)
        if n.op == "-":
            return -v
        return 1.0 if (math.isnan(v) or v == 0.0) else 0.0  # not
    op = n.op
    if op in ("and", "or"):
        lv = _eval(n.left, env)
        lt = not math.isnan(lv) and lv != 0.0
        if op == "and" and not lt:
            return 0.0
        if op == "or" and lt:
            return 1.0
        rv = _eval(n.right, env)
        return 1.0 if (not math.isnan(rv) and rv != 0.0) else 0.0
    a, b = _eval(n.left, env), _eval(n.right, env)
    if op in ("+", "-", "*", "/"):
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        return a / b if b != 0 else math.nan
    if math.isnan(a) or math.isnan(b):
        return 0.0
    return float(
        {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "==": a == b, "!=": a != b}[op]
    )
