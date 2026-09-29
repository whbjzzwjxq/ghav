"""GitHub Actions expression subset: parsing, substitution, simplification, evaluation.

The public AST is `ghav.wir.Expr`. Parsing goes through an *internal* AST that is a superset
of the public one: it also has ordering comparisons (`lt`, `le`, `gt`, `ge`), non-path
indexing (`index`), calls to unknown functions or known functions with a wrong arity
(`fcall`), and malformed context paths (`badref`). Before a tree leaves this module every
internal-only node is turned into `Expr("opaque", (text, *refs))`, where `text` is a
canonical rendering of that sub-expression (re-parseable) and `refs` are the sorted
reference paths it mentions (`github.ref`, `steps.<id>.<field>`, `needs.<job>.<field>`, ...).
Text that does not parse at all becomes one opaque node whose refs are found by a regex.

Every function here that inspects or rewrites an expression first re-parses opaque text into
the internal AST, so substitution and constant folding also work under opaque nodes.

Semantics follow the GitHub documentation: `&&`/`||` return operand values, `==` and the
ordering operators coerce mismatched types to numbers, string comparison ignores case,
and the falsy values are `false`, `0`, `-0`, `""`, `null` (and NaN).
"""

from __future__ import annotations

import json
import math
import re
from typing import Callable, Iterator, Mapping

from ghav.wir import TRUE, Expr

__all__ = [
    "CONTEXTS",
    "FALSE",
    "FUNCTIONS",
    "STATUS_FUNCTIONS",
    "SUCCESS",
    "Unresolved",
    "conj",
    "constant_value",
    "ctx_refs",
    "evaluate",
    "from_value",
    "has_opaque",
    "interpolate",
    "loose_eq",
    "needs_field_refs",
    "needs_refs",
    "parse",
    "parse_template",
    "ref_paths",
    "rename_needs",
    "simplify",
    "step_refs",
    "substitute",
    "template_needs_refs",
    "template_segments",
    "to_number",
    "to_str",
    "truthy",
    "unparse",
    "uses_status_function",
]

FALSE = Expr("lit", (False,))
SUCCESS = Expr("status", ("success",))

STATUS_FUNCTIONS = ("always", "cancelled", "failure", "success")
_CALL_FUNCTIONS = ("contains", "endsWith", "format", "fromJSON", "hashFiles", "join", "startsWith", "toJSON")
FUNCTIONS = {name.lower(): name for name in (*_CALL_FUNCTIONS, *STATUS_FUNCTIONS)}
CONTEXTS = frozenset(
    {"env", "github", "inputs", "job", "jobs", "matrix", "needs", "runner", "secrets", "steps", "strategy", "vars"}
)

# (min, max) argument counts; None = unbounded.
_ARITY: dict[str, tuple[int, int | None]] = {
    "contains": (2, 2),
    "startsWith": (2, 2),
    "endsWith": (2, 2),
    "format": (1, None),
    "join": (1, 2),
    "toJSON": (1, 1),
    "fromJSON": (1, 1),
    "hashFiles": (1, None),
    **{fn: (0, 0) for fn in STATUS_FUNCTIONS},
}

_INTERNAL_OPS = frozenset({"lt", "le", "gt", "ge", "index", "fcall", "badref"})
_BINARY_OPS = frozenset({"and", "or", "eq", "ne", "lt", "le", "gt", "ge", "index"})
_LEAF_OPS = frozenset({"lit", "ctx", "status", "step", "needs", "badref", "opaque"})
_FOLDABLE = frozenset({"contains", "endsWith", "format", "fromJSON", "join", "startsWith"})
_CMP_TOKENS = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
_CMP_TEXT = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}
_PREC = {"or": 1, "and": 2, "eq": 3, "ne": 3, "lt": 4, "le": 4, "gt": 4, "ge": 4, "not": 5}

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*\Z")
_STATUS_TEXT_RE = re.compile(r"\b(success|failure|always|cancelled)\s*\(", re.IGNORECASE)
_REF_TEXT_RE = re.compile(
    r"(?<![A-Za-z0-9_\-.])(github|env|vars|job|jobs|steps|runner|secrets|strategy|matrix|needs|inputs)"
    r"((?:\.(?:[A-Za-z0-9_\-]+|\*))*)",
    re.IGNORECASE,
)
_JSON_NUMBER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


class Unresolved(Exception):
    """Raised by `evaluate` when a reference cannot be resolved to a concrete value."""


class _ParseError(Exception):
    pass


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<num>-?(?:0[xX][0-9a-fA-F]+|(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?))
  | (?P<ident>[A-Za-z_][A-Za-z0-9_\-]*)
  | (?P<str>'(?:[^']|'')*')
  | (?P<op>==|!=|<=|>=|&&|\|\||[<>!()\[\].,*])
    """,
    re.VERBOSE,
)
_POSTFIXABLE = frozenset({"ident", ")", "]", "*"})


def _tokenize(text: str) -> list[tuple[str, object]]:
    tokens: list[tuple[str, object]] = []
    pos = 0
    while pos < len(text):
        prev = tokens[-1][0] if tokens else None
        if text[pos] == "." and prev in _POSTFIXABLE:
            tokens.append((".", "."))
            pos += 1
            continue
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            raise _ParseError(f"unexpected character {text[pos]!r} at {pos}")
        kind = m.lastgroup
        raw = m.group(kind)
        pos = m.end()
        if kind == "ws":
            continue
        if kind == "num":
            tokens.append(("lit", _parse_number(raw)))
        elif kind == "str":
            tokens.append(("lit", raw[1:-1].replace("''", "'")))
        elif kind == "ident":
            if raw in ("true", "false"):
                tokens.append(("lit", raw == "true"))
            elif raw == "null":
                tokens.append(("lit", None))
            elif raw == "NaN":
                tokens.append(("lit", math.nan))
            elif raw == "Infinity":
                tokens.append(("lit", math.inf))
            else:
                tokens.append(("ident", raw))
        else:
            tokens.append((raw, raw))
    tokens.append(("eof", None))
    return tokens


def _parse_number(raw: str) -> int | float:
    neg = raw.startswith("-")
    body = raw[1:] if neg else raw
    if body[:2].lower() == "0x":
        value: int | float = int(body, 16)
    elif any(c in body for c in ".eE"):
        value = float(body)
    else:
        value = int(body)
    return -value if neg else value


# ---------------------------------------------------------------------------
# Parser (produces the internal AST)
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, text: str) -> None:
        self.tokens = _tokenize(text)
        self.pos = 0

    def peek(self) -> str:
        return self.tokens[self.pos][0]

    def take(self) -> tuple[str, object]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def accept(self, kind: str) -> bool:
        if self.peek() == kind:
            self.pos += 1
            return True
        return False

    def expect(self, kind: str) -> tuple[str, object]:
        if self.peek() != kind:
            raise _ParseError(f"expected {kind!r}, got {self.peek()!r}")
        return self.take()

    def parse(self) -> Expr:
        e = self.or_()
        self.expect("eof")
        return e

    def or_(self) -> Expr:
        left = self.and_()
        while self.accept("||"):
            left = Expr("or", (left, self.and_()))
        return left

    def and_(self) -> Expr:
        left = self.eq()
        while self.accept("&&"):
            left = Expr("and", (left, self.eq()))
        return left

    def eq(self) -> Expr:
        left = self.cmp()
        while self.peek() in ("==", "!="):
            op = "eq" if self.take()[0] == "==" else "ne"
            left = Expr(op, (left, self.cmp()))
        return left

    def cmp(self) -> Expr:
        left = self.unary()
        while self.peek() in _CMP_TOKENS:
            op = _CMP_TOKENS[self.take()[0]]
            left = Expr(op, (left, self.unary()))
        return left

    def unary(self) -> Expr:
        if self.accept("!"):
            return Expr("not", (self.unary(),))
        return self.postfix()

    def postfix(self) -> Expr:
        kind, value = self.take()
        path: list[str] | None = None
        base: Expr | None = None
        if kind == "(":
            base = self.or_()
            self.expect(")")
        elif kind == "lit":
            base = Expr("lit", (value,))
        elif kind == "ident":
            assert isinstance(value, str)
            if self.accept("("):
                base = self.call(value)
            else:
                path = [value]
        else:
            raise _ParseError(f"unexpected token {kind!r}")
        while True:
            if self.accept("."):
                if self.accept("*"):
                    seg = "*"
                else:
                    tok_kind, tok_val = self.take()
                    if tok_kind == "ident":
                        seg = str(tok_val)
                    elif tok_kind == "lit" and isinstance(tok_val, bool | type(None)):
                        seg = {True: "true", False: "false", None: "null"}[tok_val]  # type: ignore[index]
                    else:
                        raise _ParseError("expected property name")
                if path is not None:
                    path.append(seg)
                else:
                    base = Expr("index", (base, Expr("lit", (seg,))))
            elif self.accept("["):
                idx = self.or_()
                self.expect("]")
                literal_key = (
                    idx.op == "lit"
                    and isinstance(idx.args[0], str | int)
                    and not isinstance(idx.args[0], bool)
                )
                if path is not None and literal_key:
                    path.append(str(idx.args[0]))
                else:
                    if path is not None:
                        base = _path_node(path)
                        path = None
                    base = Expr("index", (base, idx))
            else:
                break
        if path is not None:
            return _path_node(path)
        assert base is not None
        return base

    def call(self, name: str) -> Expr:
        args: list[Expr] = []
        if not self.accept(")"):
            while True:
                args.append(self.or_())
                if self.accept(")"):
                    break
                self.expect(",")
        canon = FUNCTIONS.get(name.lower())
        if canon is None:
            return Expr("fcall", (name, *args))
        lo, hi = _ARITY[canon]
        if len(args) < lo or (hi is not None and len(args) > hi):
            return Expr("fcall", (canon, *args))
        if canon in STATUS_FUNCTIONS:
            return Expr("status", (canon,))
        return Expr("call", (canon, *args))


def _path_node(path: list[str]) -> Expr:
    root = path[0].lower()
    rest = path[1:]
    if root not in CONTEXTS:
        return Expr("badref", (".".join(path),))
    if root in ("steps", "needs"):
        if len(rest) < 2 or rest[0] == "*":
            return Expr("badref", (".".join([root, *rest]),))
        return Expr("step" if root == "steps" else "needs", (rest[0], ".".join(rest[1:])))
    return Expr("ctx", (".".join([root, *rest]),))


def _parse_internal(text: str) -> Expr:
    return _Parser(text).parse()


# ---------------------------------------------------------------------------
# Internal <-> public conversion
# ---------------------------------------------------------------------------

def _map_children(e: Expr, fn: Callable[[Expr], Expr]) -> Expr:
    if e.op in _LEAF_OPS:
        return e
    if e.op in ("call", "fcall"):
        return Expr(e.op, (e.args[0], *(fn(a) for a in e.args[1:])))
    return Expr(e.op, tuple(fn(a) for a in e.args))


def _children(e: Expr) -> tuple[Expr, ...]:
    if e.op in _LEAF_OPS:
        return ()
    if e.op in ("call", "fcall"):
        return tuple(e.args[1:])
    return tuple(e.args)


def _internal(e: Expr) -> Expr:
    """Re-parse opaque nodes (where possible) into the internal AST."""
    if e.op == "opaque":
        try:
            return _parse_internal(e.args[0])
        except _ParseError:
            return e
    return _map_children(e, _internal)


def _publicize(e: Expr) -> Expr:
    if e.op in _INTERNAL_OPS:
        return Expr("opaque", (unparse(e), *sorted(ref_paths(e))))
    return _map_children(e, _publicize)


def _walk(e: Expr) -> Iterator[Expr]:
    yield e
    for c in _children(e):
        yield from _walk(c)


# ---------------------------------------------------------------------------
# Public parsing API
# ---------------------------------------------------------------------------

def _single_wrapper(text: str) -> str | None:
    segs = _segments(text)
    if len(segs) == 1 and segs[0][0]:
        return segs[0][1]
    return None


def parse(text: object) -> Expr:
    """Parse the body of `${{ }}` or a bare `if:` string. Never raises.

    A string that is exactly one `${{ ... }}` is unwrapped. A string that mixes text and
    `${{ }}` is a template (see `parse_template`). Non-string YAML scalars are converted
    with `to_str` first (so `if: false` parses as the literal false).
    """
    if not isinstance(text, str):
        text = to_str(text)
    stripped = text.strip()
    inner = _single_wrapper(stripped)
    if inner is not None:
        stripped = inner.strip()
    elif "${{" in stripped:
        return parse_template(stripped)
    try:
        return _publicize(_parse_internal(stripped))
    except _ParseError:
        return _opaque_text(stripped)


def _opaque_text(text: str) -> Expr:
    refs = set()
    for m in _REF_TEXT_RE.finditer(text):
        refs.add(m.group(1).lower() + (m.group(2) or ""))
    return Expr("opaque", (text, *sorted(refs)))


def _segments(s: str) -> list[tuple[bool, str, str]]:
    """Split a string into (is_expr, inner_text, raw_text) segments around `${{ }}`."""
    out: list[tuple[bool, str, str]] = []
    i = lit_start = 0
    n = len(s)
    while True:
        j = s.find("${{", i)
        if j < 0:
            break
        k = j + 3
        in_str = False
        closed = False
        while k < n:
            c = s[k]
            if in_str:
                if c == "'":
                    if k + 1 < n and s[k + 1] == "'":
                        k += 2
                        continue
                    in_str = False
            elif c == "'":
                in_str = True
            elif s.startswith("}}", k):
                closed = True
                break
            k += 1
        if not closed:
            break
        if j > lit_start:
            out.append((False, s[lit_start:j], s[lit_start:j]))
        out.append((True, s[j + 3 : k], s[j : k + 2]))
        i = lit_start = k + 2
    if lit_start < n:
        out.append((False, s[lit_start:], s[lit_start:]))
    return out


def template_segments(s: str) -> list[tuple[bool, str, str]]:
    """Split `s` into (is_expression, inner_text, raw_text) segments around `${{ }}`.
    `}}` inside single-quoted string literals does not close a segment."""
    return _segments(s)


def parse_template(s: object) -> Expr:
    """Parse a string that may contain `${{ }}` segments.

    No segments: a string literal. Exactly one segment and nothing else: that expression
    (keeping its type). Otherwise: `format('<text with {0}...>', e0, ...)`.
    """
    if not isinstance(s, str):
        return from_value(s)
    segs = _segments(s)
    exprs = [seg for seg in segs if seg[0]]
    if not exprs:
        return Expr("lit", (s,))
    if len(segs) == 1:
        return parse(segs[0][1])
    fmt: list[str] = []
    args: list[Expr] = []
    for is_expr, inner, raw in segs:
        if is_expr:
            fmt.append("{" + str(len(args)) + "}")
            args.append(parse(inner))
        else:
            fmt.append(raw.replace("{", "{{").replace("}", "}}"))
    return Expr("call", ("format", Expr("lit", ("".join(fmt),)), *args))


# ---------------------------------------------------------------------------
# Unparsing
# ---------------------------------------------------------------------------

def _lit_text(v: object) -> str:
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return _num_text(v)
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return "'" + to_str(v).replace("'", "''") + "'"


def _path_text(segs: list[str]) -> str:
    out = segs[0]
    for seg in segs[1:]:
        if seg == "*":
            out += ".*"
        elif _IDENT_RE.match(seg) and seg not in ("true", "false", "null"):
            out += "." + seg
        else:
            out += "['" + seg.replace("'", "''") + "']"
    return out


def unparse(e: Expr) -> str:
    """Render an expression (public or internal) as GitHub expression text."""
    return _up(_internal(e), 0)


def _up(e: Expr, min_prec: int) -> str:
    op = e.op
    prec = _PREC.get(op, 6)
    if op == "lit":
        s = _lit_text(e.args[0])
    elif op == "ctx":
        s = _path_text(e.args[0].split("."))
    elif op == "status":
        s = f"{e.args[0]}()"
    elif op in ("step", "needs"):
        root = "steps" if op == "step" else "needs"
        s = _path_text([root, e.args[0], *e.args[1].split(".")])
    elif op == "badref":
        s = _path_text(e.args[0].split("."))
    elif op == "opaque":  # unparseable text
        s = e.args[0]
        prec = 1
    elif op == "not":
        s = "!" + _up(e.args[0], 5)
    elif op == "and":
        s = _up(e.args[0], 2) + " && " + _up(e.args[1], 3)
    elif op == "or":
        s = _up(e.args[0], 1) + " || " + _up(e.args[1], 2)
    elif op in _CMP_TEXT:
        s = _up(e.args[0], prec) + f" {_CMP_TEXT[op]} " + _up(e.args[1], prec + 1)
    elif op in ("call", "fcall"):
        s = f"{e.args[0]}(" + ", ".join(_up(a, 0) for a in e.args[1:]) + ")"
    elif op == "index":
        s = _up(e.args[0], 6) + "[" + _up(e.args[1], 0) + "]"
    else:
        raise ValueError(f"unknown expression op {op!r}")
    return f"({s})" if prec < min_prec else s


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def ref_paths(e: Expr) -> frozenset[str]:
    """All reference paths: ctx keys, `steps.<id>.<field>`, `needs.<job>.<field>`, bad refs."""
    out: set[str] = set()
    for n in _walk(_internal(e)):
        if n.op == "ctx":
            out.add(n.args[0])
        elif n.op == "step":
            out.add(f"steps.{n.args[0]}.{n.args[1]}")
        elif n.op == "needs":
            out.add(f"needs.{n.args[0]}.{n.args[1]}")
        elif n.op == "badref":
            out.add(n.args[0])
        elif n.op == "opaque":
            out.update(n.args[1:])
    return frozenset(out)


def ctx_refs(e: Expr) -> frozenset[str]:
    """Context keys read by `e` (everything except `steps.*` and `needs.*`)."""
    return frozenset(
        p for p in ref_paths(e) if p.split(".", 1)[0].lower() not in ("steps", "needs")
    )


def step_refs(e: Expr) -> frozenset[str]:
    """Step ids referenced through `steps.<id>...`."""
    out: set[str] = set()
    for n in _walk(_internal(e)):
        if n.op == "step":
            out.add(n.args[0])
        elif n.op in ("badref", "opaque"):
            for p in (n.args[0],) if n.op == "badref" else n.args[1:]:
                parts = p.split(".")
                if parts[0].lower() == "steps" and len(parts) > 1 and parts[1] != "*":
                    out.add(parts[1])
    return frozenset(out)


def needs_refs(e: Expr) -> frozenset[str]:
    """Job ids referenced through `needs.<job>...` (`*` for `needs.*`)."""
    out: set[str] = set()
    for n in _walk(_internal(e)):
        if n.op == "needs":
            out.add(n.args[0])
        elif n.op in ("badref", "opaque"):
            for p in (n.args[0],) if n.op == "badref" else n.args[1:]:
                parts = p.split(".")
                if parts[0].lower() == "needs" and len(parts) > 1:
                    out.add(parts[1])
    return frozenset(out)


def needs_field_refs(e: Expr) -> frozenset[tuple[str, str]]:
    """`(job, field)` for every `needs.<job>.<field>` reference in `e`.

    `field` is the path after the job id (`result`, `outputs.<name>`, ...), or `*` when the
    reference names the whole `needs.<job>` object (job `*` for `needs` itself or
    `needs.*`). Opaque text is re-parsed where possible; otherwise its reference paths are used.
    """
    out: set[tuple[str, str]] = set()
    for n in _walk(_internal(e)):
        if n.op == "needs":
            out.add((str(n.args[0]), str(n.args[1])))
        elif n.op in ("badref", "opaque"):
            for p in (n.args[0],) if n.op == "badref" else n.args[1:]:
                parts = str(p).split(".")
                if parts[0].lower() != "needs":
                    continue
                job = parts[1] if len(parts) > 1 else "*"
                out.add((job, ".".join(parts[2:]) or "*"))
    return frozenset(out)


def template_needs_refs(s: object) -> frozenset[tuple[str, str]]:
    """`needs_field_refs` of every `${{ }}` segment of a string (empty for non-strings)."""
    if not isinstance(s, str) or "${{" not in s:
        return frozenset()
    out: set[tuple[str, str]] = set()
    for is_expr, inner, _raw in _segments(s):
        if is_expr:
            out |= needs_field_refs(parse(inner))
    return frozenset(out)


def uses_status_function(e: Expr) -> bool:
    """True if success()/failure()/always()/cancelled() occurs anywhere in `e`."""
    for n in _walk(_internal(e)):
        if n.op == "status":
            return True
        if n.op == "fcall" and str(n.args[0]).lower() in STATUS_FUNCTIONS:
            return True
        if n.op == "opaque" and _STATUS_TEXT_RE.search(n.args[0]):
            return True
    return False


def has_opaque(e: Expr) -> bool:
    return any(n.op == "opaque" for n in _walk(e))


# ---------------------------------------------------------------------------
# Values and coercions (GitHub semantics)
# ---------------------------------------------------------------------------

def _num_text(v: float) -> str:
    if v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def to_str(v: object) -> str:
    """GitHub's string conversion of a value (used for interpolation and matrix ids)."""
    if v is None:
        return ""
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return _num_text(v)
    if isinstance(v, str):
        return v
    if isinstance(v, Expr):
        ok, value = constant_value(v)
        return to_str(value) if ok else "${{ " + unparse(v) + " }}"
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


def to_number(v: object) -> float:
    if v is None:
        return 0.0
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return 0.0
        if _JSON_NUMBER_RE.match(s):
            return float(s)
        if s.lower().startswith("0x"):
            try:
                return float(int(s, 16))
            except ValueError:
                return math.nan
        return math.nan
    return math.nan


def truthy(v: object) -> bool:
    if v is None or v is False:
        return False
    if v is True:
        return True
    if isinstance(v, int | float):
        return not (v == 0 or (isinstance(v, float) and math.isnan(v)))
    if isinstance(v, str):
        return v != ""
    return True


def _kind(v: object) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int | float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    return "object"


def loose_eq(a: object, b: object) -> bool:
    ka, kb = _kind(a), _kind(b)
    if ka == kb:
        if ka == "string":
            return a.casefold() == b.casefold()  # type: ignore[union-attr]
        if ka == "number":
            return float(a) == float(b)  # type: ignore[arg-type]
        if ka in ("bool", "null"):
            return a == b
        return a is b
    na, nb = to_number(a), to_number(b)
    return na == nb


def _compare(a: object, b: object, op: str) -> bool:
    ka, kb = _kind(a), _kind(b)
    if ka == kb == "string":
        x: object = a.casefold()  # type: ignore[union-attr]
        y: object = b.casefold()  # type: ignore[union-attr]
    else:
        x, y = to_number(a), to_number(b)
        if math.isnan(x) or math.isnan(y):  # type: ignore[arg-type]
            return False
    if op == "lt":
        return x < y  # type: ignore[operator]
    if op == "le":
        return x <= y  # type: ignore[operator]
    if op == "gt":
        return x > y  # type: ignore[operator]
    return x >= y  # type: ignore[operator]


def _format(fmt: str, args: list[object]) -> str:
    out: list[str] = []
    i = 0
    n = len(fmt)
    while i < n:
        c = fmt[i]
        if c == "{":
            if fmt.startswith("{{", i):
                out.append("{")
                i += 2
                continue
            j = fmt.find("}", i)
            if j < 0:
                raise ValueError("unterminated format placeholder")
            idx = int(fmt[i + 1 : j])
            if idx >= len(args):
                raise ValueError("format index out of range")
            out.append(to_str(args[idx]))
            i = j + 1
        elif c == "}":
            if fmt.startswith("}}", i):
                out.append("}")
                i += 2
                continue
            raise ValueError("unbalanced '}' in format string")
        else:
            out.append(c)
            i += 1
    return "".join(out)


class _Filtered(list):
    """Result of an object filter (`.*`); later property accesses map over its elements."""


def _index(base: object, key: object) -> object:
    if isinstance(base, _Filtered):
        if key == "*":
            flat: list[object] = []
            for x in base:
                if isinstance(x, dict):
                    flat.extend(x[k] for k in sorted(x))
                elif isinstance(x, list):
                    flat.extend(x)
            return _Filtered(flat)
        return _Filtered(v for v in (_index(x, key) for x in base) if v is not None)
    if isinstance(base, dict):
        if isinstance(key, str):
            if key == "*":
                return _Filtered(base[k] for k in sorted(base))
            for k in sorted(base):
                if k.casefold() == key.casefold():
                    return base[k]
        return None
    if isinstance(base, list):
        if key == "*":
            return _Filtered(base)
        n = to_number(key)
        if math.isnan(n) or not n.is_integer():
            return None
        i = int(n)
        return base[i] if 0 <= i < len(base) else None
    return None


def _call(fname: str, args: list[object]) -> object:
    if fname == "contains":
        search, item = args
        if isinstance(search, list):
            return any(loose_eq(x, item) for x in search)
        return to_str(item).casefold() in to_str(search).casefold()
    if fname == "startsWith":
        return to_str(args[0]).casefold().startswith(to_str(args[1]).casefold())
    if fname == "endsWith":
        return to_str(args[0]).casefold().endswith(to_str(args[1]).casefold())
    if fname == "format":
        return _format(to_str(args[0]), args[1:])
    if fname == "join":
        sep = to_str(args[1]) if len(args) > 1 else ","
        if isinstance(args[0], list):
            return sep.join(to_str(x) for x in args[0])
        return to_str(args[0])
    if fname == "fromJSON":
        return json.loads(to_str(args[0]))
    if fname == "toJSON":
        return json.dumps(args[0], indent=2, sort_keys=True)
    raise Unresolved(fname)


def constant_value(e: Expr) -> tuple[bool, object]:
    """(True, value) if `e` is a literal or `fromJSON('<literal json>')`, else (False, None)."""
    if e.op == "lit":
        return True, e.args[0]
    if e.op == "call" and e.args[0] == "fromJSON" and len(e.args) == 2 and e.args[1].op == "lit":
        try:
            return True, json.loads(to_str(e.args[1].args[0]))
        except (ValueError, TypeError):
            return False, None
    return False, None


def from_value(v: object) -> Expr:
    """Expression denoting a concrete value (`Expr` values are returned unchanged)."""
    if isinstance(v, Expr):
        return v
    if v is None or isinstance(v, bool | int | float | str):
        return Expr("lit", (v,))
    return Expr("call", ("fromJSON", Expr("lit", (json.dumps(v, sort_keys=True, separators=(",", ":")),))))


def _is_scalar(v: object) -> bool:
    return v is None or isinstance(v, bool | int | float | str)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(e: Expr, lookup: Callable[[Expr], object] | None = None) -> object:
    """Evaluate concretely. `lookup` resolves leaf references (ctx/status/step/needs nodes,
    opaque text that does not parse, and `hashFiles(...)` calls); it may raise KeyError.
    Raises `Unresolved` when a reference cannot be resolved."""
    return _eval(_internal(e), lookup)


def _resolve(n: Expr, lookup: Callable[[Expr], object] | None) -> object:
    if lookup is None:
        raise Unresolved(unparse(n))
    try:
        return lookup(_publicize(n))
    except KeyError as exc:
        raise Unresolved(unparse(n)) from exc


def _eval(n: Expr, lookup: Callable[[Expr], object] | None) -> object:
    op = n.op
    if op == "lit":
        return n.args[0]
    if op in ("ctx", "status", "step", "needs", "badref", "opaque"):
        return _resolve(n, lookup)
    if op == "not":
        return not truthy(_eval(n.args[0], lookup))
    if op == "and":
        left = _eval(n.args[0], lookup)
        return _eval(n.args[1], lookup) if truthy(left) else left
    if op == "or":
        left = _eval(n.args[0], lookup)
        return left if truthy(left) else _eval(n.args[1], lookup)
    if op == "eq":
        return loose_eq(_eval(n.args[0], lookup), _eval(n.args[1], lookup))
    if op == "ne":
        return not loose_eq(_eval(n.args[0], lookup), _eval(n.args[1], lookup))
    if op in ("lt", "le", "gt", "ge"):
        return _compare(_eval(n.args[0], lookup), _eval(n.args[1], lookup), op)
    if op == "index":
        return _index(_eval(n.args[0], lookup), _eval(n.args[1], lookup))
    if op == "call":
        if n.args[0] == "hashFiles":
            return _resolve(n, lookup)
        args = [_eval(a, lookup) for a in n.args[1:]]
        try:
            return _call(n.args[0], args)
        except (ValueError, TypeError, IndexError) as exc:
            raise Unresolved(unparse(n)) from exc
    raise Unresolved(unparse(n))


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------

def _lower_env(env: Mapping[str, object]) -> dict[str, object]:
    return {k.lower(): env[k] for k in sorted(env)}


def _descend(value: object, segs: list[str]) -> tuple[bool, object]:
    for i, seg in enumerate(segs):
        if isinstance(value, Expr):
            ok, const = constant_value(value)
            if ok:
                value = const
            elif value.op == "ctx":
                return True, Expr("ctx", (".".join([value.args[0], *segs[i:]]),))
            else:
                return False, None
        value = _index(value, seg)
    return True, value


def _lookup_env(key: str, envl: Mapping[str, object]) -> tuple[bool, object]:
    segs = key.split(".")
    for i in range(len(segs), 0, -1):
        prefix = ".".join(segs[:i]).lower()
        if prefix in envl:
            return _descend(envl[prefix], segs[i:])
    return False, None


def _subst(n: Expr, envl: Mapping[str, object], rename: Mapping[str, str]) -> Expr:
    if n.op == "ctx":
        found, value = _lookup_env(n.args[0], envl)
        if found:
            return _internal(from_value(value))
        return n
    if n.op == "needs" and n.args[0] in rename:
        return Expr("needs", (rename[n.args[0]], n.args[1]))
    return _map_children(n, lambda c: _subst(c, envl, rename))


def substitute(
    e: Expr, env: Mapping[str, object], needs_rename: Mapping[str, str] | None = None
) -> Expr:
    """Replace context references by values.

    `env` maps context keys or key prefixes (case-insensitive), e.g. `"matrix"` -> dict of
    the combination or `"inputs.os"` -> `"ubuntu-latest"`, to Python values or `Expr`s.
    Sub-paths are looked up inside dict/list values (missing keys give null, as on GitHub).
    `needs_rename` maps job ids in `needs.<job>` references to new ids.
    """
    return _publicize(_subst(_internal(e), _lower_env(env), dict(needs_rename or {})))


def rename_needs(e: Expr, mapping: Mapping[str, str]) -> Expr:
    return substitute(e, {}, mapping)


def _touches(e: Expr, envl: Mapping[str, object], rename: Mapping[str, str]) -> bool:
    for n in _walk(_internal(e)):
        if n.op == "ctx" and _lookup_env(n.args[0], envl)[0]:
            return True
        if n.op == "needs" and n.args[0] in rename:
            return True
    return False


def interpolate(
    s: object, env: Mapping[str, object], needs_rename: Mapping[str, str] | None = None
) -> str:
    """Substitute known context values inside the `${{ }}` segments of a string.

    Segments that mention no substituted key are kept verbatim. A segment that becomes a
    scalar constant is replaced by its string form; otherwise it is re-rendered as
    `${{ <simplified expression> }}`. Non-strings are converted with `to_str`.
    """
    if not isinstance(s, str):
        return to_str(s)
    if "${{" not in s:
        return s
    envl = _lower_env(env)
    rename = dict(needs_rename or {})
    out: list[str] = []
    for is_expr, inner, raw in _segments(s):
        if not is_expr:
            out.append(raw)
            continue
        e = parse(inner)
        if not _touches(e, envl, rename):
            out.append(raw)
            continue
        e2 = simplify(substitute(e, env, rename))
        ok, value = constant_value(e2)
        if ok and _is_scalar(value):
            out.append(to_str(value))
        else:
            out.append("${{ " + unparse(e2) + " }}")
    return "".join(out)


# ---------------------------------------------------------------------------
# Simplification (constant folding)
# ---------------------------------------------------------------------------

def simplify(e: Expr, boolean: bool = False) -> Expr:
    """Fold constants. With `boolean=True` only the truthiness of the result is preserved
    (used for `if:` conditions); otherwise the value is preserved exactly."""
    return _publicize(_simp(_internal(e), boolean))


def _is_const(n: Expr) -> bool:
    return constant_value(n)[0]


def _try_fold(n: Expr) -> Expr:
    try:
        value = _eval(n, None)
    except (Unresolved, ValueError, TypeError, IndexError):
        return n
    return Expr("lit", (value,)) if _is_scalar(value) else n


def _simp(n: Expr, boolean: bool) -> Expr:
    out = _simp0(n, boolean)
    if boolean and out.op == "lit" and not isinstance(out.args[0], bool):
        return Expr("lit", (truthy(out.args[0]),))
    return out


def _simp0(n: Expr, boolean: bool) -> Expr:
    op = n.op
    if op in _LEAF_OPS:
        return n
    if op == "not":
        c = _simp(n.args[0], True)
        if c.op == "lit":
            return Expr("lit", (not truthy(c.args[0]),))
        if boolean and c.op == "not":
            return c.args[0]
        return Expr("not", (c,))
    if op == "and":
        left, right = _simp(n.args[0], boolean), _simp(n.args[1], boolean)
        if left.op == "lit":
            return right if truthy(left.args[0]) else left
        if boolean and right.op == "lit":
            return left if truthy(right.args[0]) else FALSE
        return Expr("and", (left, right))
    if op == "or":
        left, right = _simp(n.args[0], boolean), _simp(n.args[1], boolean)
        if left.op == "lit":
            return left if truthy(left.args[0]) else right
        if boolean and right.op == "lit":
            return TRUE if truthy(right.args[0]) else left
        return Expr("or", (left, right))
    if op in ("eq", "ne", "lt", "le", "gt", "ge", "index"):
        args = tuple(_simp(a, False) for a in n.args)
        out = Expr(op, args)
        return _try_fold(out) if all(_is_const(a) for a in args) else out
    if op == "call":
        args = tuple(_simp(a, False) for a in n.args[1:])
        out = Expr("call", (n.args[0], *args))
        if n.args[0] in _FOLDABLE and all(_is_const(a) for a in args):
            return _try_fold(out)
        return out
    if op == "fcall":
        return Expr("fcall", (n.args[0], *(_simp(a, False) for a in n.args[1:])))
    raise ValueError(f"unknown expression op {op!r}")


def conj(parts: list[Expr]) -> Expr:
    """Left-nested conjunction; TRUE for an empty list."""
    out: Expr | None = None
    for p in parts:
        out = p if out is None else Expr("and", (out, p))
    return TRUE if out is None else out
