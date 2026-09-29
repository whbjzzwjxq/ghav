"""Small WIR program builders shared by tests/test_semantics.py and tests/test_checker.py.

Contracts follow docs/CONTRACT_GUIDE.md templates; every step also reads ctx:platform.
Loaded by path (tests/fixtures is not a package).
"""

from __future__ import annotations

from ghav.contracts import transfer_bindings
from ghav.wir import (
    CTX,
    DEP,
    EXT,
    OUT,
    OUTPUT,
    RUN,
    SRC,
    TOOL,
    TRUE,
    Binding,
    Contract,
    Expr,
    JobInstance,
    Program,
    Region,
    Step,
)

SUCCESS = Expr("status", ("success",))
ALWAYS = Expr("status", ("always",))


def R(spec: str) -> Region:
    kind, _, rest = spec.partition(":")
    if kind in (SRC, DEP, OUT, TOOL):
        return Region(kind)
    if kind in (CTX, RUN, EXT):
        return Region(kind, None, rest)
    if kind == OUTPUT:
        job, _, key = rest.partition(":")
        return Region(OUTPUT, job, key)
    raise ValueError(spec)


def contract(template: str, reads=(), writes=(), out_paths=(), top: bool = False) -> Contract:
    if top:
        return Contract(top=True, template="top")
    return Contract(
        frozenset(R(r) for r in tuple(reads) + ("ctx:platform",)),
        frozenset(R(w) for w in writes),
        False,
        tuple(out_paths),
        template,
    )


CHECKOUT = contract("checkout", ("ctx:github.sha", "ctx:github.ref", "src", "dep", "out", "tool"),
                    ("src", "dep", "out", "tool"))
SETUP = contract("setup-runtime", ("src",), ("dep",))
INSTALL = contract("install-pinned", ("src", "dep"), ("dep", "tool"))
BUILD = contract("build", ("src", "dep", "out"), ("out", "tool"), ("dist",))
BUILD_LIB = contract("build", ("src", "dep", "out"), ("out", "tool"), ("lib",))
LINT = contract("lint", ("src", "dep", "out"))
TYPECHECK = contract("typecheck", ("src", "dep", "out"), ("tool",))
TEST = contract("test", ("src", "dep", "out"), ("tool",))
TEST_SHARD = contract("test", ("src", "dep", "out", "ctx:matrix.shard"), ("tool",))
TEST_NODE = contract("test", ("src", "dep", "out", "ctx:matrix.node"), ("tool",))
CODECOV = contract("report-upload", ("src", "tool", "out", "ctx:github.sha"), ("ext:codecov",))
PUBLISH = contract("publish", ("src", "dep", "out", "ctx:github.ref"), ("ext:npm",))
CACHE = contract("cache")
TOP = contract("top", top=True)


def run(text: str, c: Contract, origin, *, cond: Expr = SUCCESS, step_id: str | None = None) -> Step:
    return Step(Binding("run", text, step_id=step_id), cond, c, (origin[0], origin[1]))


def uses(action: str, c: Contract, origin, *, cond: Expr = SUCCESS, with_=(), step_id: str | None = None) -> Step:
    b = Binding("uses", action, with_=tuple(sorted(with_)), step_id=step_id)
    return Step(b, cond, c, (origin[0], origin[1]))


def upload(name: str, origin, path: str = "dist", *, cond: Expr = SUCCESS) -> Step:
    c = contract("upload-artifact", ("out",), (f"run:{name}",))
    return uses("actions/upload-artifact@v4", c, origin, cond=cond, with_=(("name", name), ("path", path)))


def download(name: str, origin, path: str = "dist", *, cond: Expr = SUCCESS) -> Step:
    c = contract("download-artifact", (f"run:{name}",), ("out",), (path,))
    return uses("actions/download-artifact@v4", c, origin, cond=cond, with_=(("name", name), ("path", path)))


def pack_upload(name: str, paths=("dist",), origin=("p", 9), *, cond: Expr = SUCCESS) -> list[Step]:
    """The producer half of the transfer idiom (contracts.transfer_bindings): pack + upload."""
    pack, up, _, _ = transfer_bindings(name, paths)
    return [
        Step(pack, cond, contract("file-util", ("out", "ctx:github.workspace"), ("tool",)), (origin[0], origin[1])),
        Step(up, cond, contract("upload-artifact", ("tool",), (f"run:{name}",)), (origin[0], origin[1])),
    ]


def download_unpack(name: str, paths=("dist",), origin=("c", 9), *, cond: Expr = SUCCESS) -> list[Step]:
    """The consumer half of the transfer idiom: download + unpack."""
    _, _, down, unpack = transfer_bindings(name, paths)
    return [
        Step(down, cond, contract("download-artifact", (f"run:{name}",), ("tool",)), (origin[0], origin[1])),
        Step(unpack, cond, contract("file-util", ("tool", "ctx:github.workspace"), ("out",), tuple(sorted(paths))),
             (origin[0], origin[1])),
    ]


def checkout(origin) -> Step:
    return uses("actions/checkout@v4", CHECKOUT, origin)


def setup(origin) -> Step:
    return uses("actions/setup-node@v4", SETUP, origin)


def install(origin) -> Step:
    return run("npm ci", INSTALL, origin)


def build(origin) -> Step:
    return run("npm run build", BUILD, origin)


def prefix(job: str) -> list[Step]:
    """checkout, setup-node, npm ci, build with origins (job, 0..3)."""
    return [checkout((job, 0)), setup((job, 1)), install((job, 2)), build((job, 3))]


def job(
    jid: str,
    steps,
    *,
    needs=(),
    check: str | None = None,
    matrix=(),
    runs_on: str = "ubuntu-latest",
    cond: Expr | None = None,
    orig: str | None = None,
    ff: str | None = None,
    outputs=(),
    frozen: bool = False,
    env=(),
    reasons=(),
) -> JobInstance:
    if cond is None:
        cond = SUCCESS if needs else TRUE
    return JobInstance(
        id=jid,
        orig_job=orig or jid,
        check_name=check or jid,
        matrix=tuple(sorted(matrix)),
        runs_on=runs_on,
        cond=cond,
        needs=frozenset(needs),
        steps=tuple(steps),
        env=tuple(sorted(env)),
        outputs=tuple(sorted(outputs, key=lambda kv: kv[0])),
        fail_fast_group=ff,
        frozen=frozen or bool(reasons),
        frozen_reasons=tuple(reasons) if reasons else (("services",) if frozen else ()),
    )


def program(*jobs: JobInstance, env=(), name: str = "ci") -> Program:
    return Program(name=name, on={"push": None, "pull_request": None}, jobs=tuple(sorted(jobs, key=lambda j: j.id)), env=tuple(env))


def eq(a: Expr, b: Expr) -> Expr:
    return Expr("eq", (a, b))


def ne(a: Expr, b: Expr) -> Expr:
    return Expr("ne", (a, b))


def and_(a: Expr, b: Expr) -> Expr:
    return Expr("and", (a, b))


def ctx(key: str) -> Expr:
    return Expr("ctx", (key,))


def lit(v) -> Expr:
    return Expr("lit", (v,))
