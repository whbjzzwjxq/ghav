"""Machine-time cost model. See docs/COST_MODEL.md.

Metric: expected machine time per run, i.e. the sum over executed job instances of the job
overhead plus the seconds of their executed steps, weighted by context frequency. Every
report gives two numbers side by side: raw machine seconds and billed minutes (each
executed job rounded up to a whole minute, as GitHub bills it).

Two backends share one interface (pass a different :class:`CostModel`):

* ``MOCK`` (default): the fixed mock table from docs/COST_MODEL.md, keyed by the contract
  template of each step, with equal weights for ``push`` and ``pull_request``.
* ``historical``: :func:`historical_model` matches the medians file written by
  ``scripts/resources/fetch_timings.py`` (``resources/timings/<benchmark-id>/medians.json``)
  to a program: GitHub job names to ``JobInstance.check_name`` and GitHub step names (or
  positions) to WIR steps. Matched steps cost their median seconds, the job overhead is the
  median of (job duration - step durations) per runner label, event weights come from the
  fetched run counts, and every unmatched step falls back to its mock value. The matched
  fraction is reported.

Conditions are evaluated concretely per event, assuming every task succeeds. The
evaluator implements the ``wir.Expr`` AST with GitHub's loose-equality and truthiness
rules over a three-valued domain: a value is concrete, a known string prefix, or
``UNKNOWN``. Values the cost model cannot know (repository-specific contexts, secrets,
step outputs, ``hashFiles``, opaque expressions) are ``UNKNOWN``; a condition whose value
is unknown counts as true, so the job or step is charged. Caches use cold-cache
semantics here as well: ``cache-hit`` outputs are unknown, so guarded steps are charged.

Concrete facts assumed per event (all others are unknown):

* ``github.event_name`` is the event.
* ``push``: ``github.head_ref`` and ``github.base_ref`` are ``''``; every
  ``github.event.pull_request*`` key is ``null``.
* ``pull_request``: ``github.ref`` starts with ``refs/pull/``; every
  ``github.event.head_commit*`` key is ``null``.
* ``matrix.<k>`` is the instance's matrix value; ``env.<K>`` is the workflow- or
  job-level env value when it contains no ``${{ }}``; ``runner.os`` follows the
  standard GitHub-hosted label families (``ubuntu-*``, ``windows-*``, ``macos-*``).
* Status functions: ``success()`` is true at step level; at job level it is true iff
  every needed instance ran. ``failure()`` and ``cancelled()`` are false, ``always()`` true.
* ``steps.<id>.outcome|conclusion`` is ``success`` or ``skipped``; ``needs.<job>.result``
  likewise, where ``<job>`` may be an instance id, an original job key, or a group key
  (instance id without its matrix suffix); several instances aggregate (any ran).

A workflow whose ``on:`` block does not list an event costs 0 for that event; an empty
``on:`` (hand-built programs) means every event triggers it. Path and branch filters are
ignored (the workflow is assumed to trigger).

Frozen jobs cost the job overhead plus ``FROZEN_STEP_SECONDS`` per step (steps from the
WIR, else from the raw YAML, at least one) when their job condition holds; under the
historical backend a frozen instance matched to a GitHub job costs its median duration.

Billed minutes: an executed job bills ``max(1, ceil(seconds / 60))`` minutes (seconds are
rounded to 6 decimals first so float noise never adds a minute); a job that does not run
bills 0.

Everything here is deterministic: jobs are visited in sorted id order and all sums are
taken in a fixed order.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from ghav.wir import EVENTS, Expr, JobInstance, Program, Step

# ---------------------------------------------------------------------------
# Mock duration table (docs/COST_MODEL.md). Fixed before any experiment; not tuned.
# ---------------------------------------------------------------------------

JOB_OVERHEAD_SECONDS: float = 15.0

TEMPLATE_SECONDS: Mapping[str, float] = MappingProxyType({
    "checkout": 5.0,
    "setup-runtime": 10.0,
    "install-pinned": 45.0,
    "install-unpinned": 45.0,
    "build": 60.0,
    "build-orchestrated": 90.0,
    "typecheck": 40.0,
    "lint": 30.0,
    "format-write": 20.0,
    "test": 120.0,
    "upload-artifact": 10.0,
    "download-artifact": 8.0,
    "cache": 5.0,
    "report-upload": 10.0,
    "publish": 30.0,
    "noop": 1.0,
    # Rows for the templates added by contract guide v1.1 (2026-09-28). Each copies the row of
    # what it refines, so no duration was chosen for them: a file utility is a shell-level
    # command like `noop`; an opaque cache restore is still a cache step; an observed contract
    # describes the effects of a step the guide leaves at top, not its duration.
    "file-util": 1.0,
    "cache-opaque": 5.0,
    "observed": 30.0,
    "top": 30.0,
})

TOP_SECONDS: float = TEMPLATE_SECONDS["top"]
FROZEN_STEP_SECONDS: float = TOP_SECONDS

DEFAULT_CONTEXTS: tuple[tuple[str, float], ...] = (("push", 0.5), ("pull_request", 0.5))

# Official actions whose template is implied by the action reference; used only when a
# step's contract template is not in the table (e.g. synthesized transfer steps).
_ACTION_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("actions/upload-artifact@", "upload-artifact"),
    ("actions/download-artifact@", "download-artifact"),
    ("actions/cache@", "cache"),
    ("actions/cache/restore@", "cache"),
    ("actions/cache/save@", "cache"),
)


_EMPTY: Mapping = MappingProxyType({})


@dataclass(frozen=True)
class CostModel:
    """Duration parameters. ``MOCK`` is the mock table; :func:`historical_model` builds a
    model whose tables come from historical GitHub runs.

    The historical tables are keyed by the ORIGINAL program (the one the timings were
    matched against), so candidate programs produced by synthesis are priced through
    ``Step.origin``/``Step.origin_instance``:

    * ``step_table``: (original instance id, original job key, original step index) -> median
      seconds (the job key guards against a synthesized job reusing an original instance id).
    * ``origin_table``: (original job key, original step index) -> median seconds pooled over
      every matched instance of that job (used when the instance itself has no sample).
    * ``overhead_table``: ``runs_on`` label -> median job overhead seconds.
    * ``job_table``: original instance id -> median job duration (prices frozen jobs, which
      are kept verbatim).

    Anything not found in these tables uses the mock value (per step), so the mock model is
    simply the model with empty tables.
    """

    job_overhead_seconds: float = JOB_OVERHEAD_SECONDS
    template_seconds: Mapping[str, float] = field(default_factory=lambda: TEMPLATE_SECONDS)
    unknown_seconds: float = TOP_SECONDS
    frozen_step_seconds: float = FROZEN_STEP_SECONDS
    label: str = "mock"
    step_table: Mapping[tuple[str, str, int], float] = field(default_factory=lambda: _EMPTY)
    origin_table: Mapping[tuple[str, int], float] = field(default_factory=lambda: _EMPTY)
    overhead_table: Mapping[str, float] = field(default_factory=lambda: _EMPTY)
    job_table: Mapping[str, float] = field(default_factory=lambda: _EMPTY)
    contexts: tuple[tuple[str, float], ...] | None = None   # event weights; None = default
    coverage: Mapping[str, Any] | None = None               # static match report (historical)

    def overhead(self, job: JobInstance) -> float:
        """Per-job fixed cost: provisioning, ``Set up job``, ``Complete job`` and gaps."""
        return float(self.overhead_table.get(job.runs_on, self.job_overhead_seconds))

    def price_step(self, job: JobInstance, step: Step) -> tuple[float, bool]:
        """(seconds, priced from history?) for one executed step occurrence of ``job``."""
        orig_job, idx = step.origin
        key = (step.origin_instance or job.id, orig_job, idx)
        if key in self.step_table:
            return float(self.step_table[key]), True
        if (orig_job, idx) in self.origin_table:
            return float(self.origin_table[(orig_job, idx)]), True
        return step_seconds(step, self), False


MOCK = CostModel()


def template_seconds(template: str, model: CostModel = MOCK) -> float:
    """Seconds for a contract template. Composite templates are joined by ``+`` and summed;
    unknown template names cost ``model.unknown_seconds`` (the ⊤ row)."""
    parts = [t.strip() for t in template.split("+") if t.strip()]
    if not parts:
        return model.unknown_seconds
    return float(sum(model.template_seconds.get(t, model.unknown_seconds) for t in parts))


def step_template(step: Step, model: CostModel = MOCK) -> str:
    """The template used to price ``step``."""
    template = step.contract.template
    parts = [t.strip() for t in template.split("+") if t.strip()]
    if parts and all(t in model.template_seconds for t in parts):
        return template
    if step.binding.kind == "uses":
        for prefix, name in _ACTION_TEMPLATES:
            if step.binding.text.startswith(prefix):
                return name
    return template


def step_seconds(step: Step, model: CostModel = MOCK) -> float:
    """Mock duration of one step occurrence."""
    return template_seconds(step_template(step, model), model)


# ---------------------------------------------------------------------------
# Three-valued concrete values
# ---------------------------------------------------------------------------

class _Unknown:
    _instance: "_Unknown | None" = None

    def __new__(cls) -> "_Unknown":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNKNOWN"


UNKNOWN = _Unknown()


@dataclass(frozen=True)
class Prefix:
    """A string value of which only a (case-insensitive) prefix is known."""

    prefix: str


def truthiness(v: Any) -> bool | None:
    """GitHub truthiness; ``None`` means unknown."""
    if v is UNKNOWN:
        return None
    if isinstance(v, Prefix):
        return True if v.prefix else None
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return not (v == 0 or (isinstance(v, float) and math.isnan(v)))
    if isinstance(v, str):
        return v != ""
    return True


def _kind(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    return "object"


_JSON_NUMBER = re.compile(r"^\s*[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\s*$")
_HEX_NUMBER = re.compile(r"^\s*0x[0-9a-fA-F]+\s*$")


def _to_number(v: Any) -> float:
    k = _kind(v)
    if k == "null":
        return 0.0
    if k == "bool":
        return 1.0 if v else 0.0
    if k == "number":
        return float(v)
    if k == "string":
        s = v.strip()
        if s == "":
            return 0.0
        if _JSON_NUMBER.match(s):
            return float(s)
        if _HEX_NUMBER.match(s):
            return float(int(s, 16))
        return math.nan
    return math.nan


def _to_str(v: Any) -> str:
    k = _kind(v)
    if k == "null":
        return ""
    if k == "bool":
        return "true" if v else "false"
    if k == "number":
        f = float(v)
        if math.isfinite(f) and f == int(f):
            return str(int(f))
        return repr(f)
    if k == "string":
        return v
    return "Array" if k == "array" else "Object"


def loose_equal(a: Any, b: Any) -> bool | None:
    """GitHub ``==``: case-insensitive strings, number coercion across types."""
    if a is UNKNOWN or b is UNKNOWN:
        return None
    if isinstance(a, Prefix) or isinstance(b, Prefix):
        if isinstance(a, Prefix) and isinstance(b, Prefix):
            pa, pb = a.prefix.lower(), b.prefix.lower()
            return None if (pa.startswith(pb) or pb.startswith(pa)) else False
        p, other = (a, b) if isinstance(a, Prefix) else (b, a)
        if isinstance(other, str):
            return None if other.lower().startswith(p.prefix.lower()) else False
        return None
    ka, kb = _kind(a), _kind(b)
    if ka == kb:
        if ka == "string":
            return a.lower() == b.lower()
        if ka == "number":
            return float(a) == float(b)
        if ka == "null":
            return True
        if ka == "bool":
            return a == b
        return a is b
    x, y = _to_number(a), _to_number(b)
    return x == y


def _not(t: bool | None) -> bool | None:
    return None if t is None else not t


def _as_value(t: bool | None) -> Any:
    return UNKNOWN if t is None else t


# ---------------------------------------------------------------------------
# Expression evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalEnv:
    """Concrete environment for evaluating one condition."""

    ctx: Mapping[str, Any]                         # exact context keys -> values
    null_prefixes: tuple[str, ...] = ()            # ctx keys under these prefixes are null
    status_success: bool = True                    # value of success() in this scope
    steps: Mapping[str, bool] = field(default_factory=dict)   # step id -> executed
    needs: Mapping[str, bool] = field(default_factory=dict)   # job ref -> ran


def _lookup_ctx(key: str, env: EvalEnv) -> Any:
    if key in env.ctx:
        return env.ctx[key]
    low = key.lower()
    for k in sorted(env.ctx):
        if k.lower() == low:
            return env.ctx[k]
    for pre in env.null_prefixes:
        if low == pre or low.startswith(pre + "."):
            return None
    return UNKNOWN


def _format(fmt: str, args: list[Any]) -> Any:
    out: list[str] = []
    i = 0
    while i < len(fmt):
        c = fmt[i]
        if c == "{" and fmt.startswith("{{", i):
            out.append("{")
            i += 2
        elif c == "}" and fmt.startswith("}}", i):
            out.append("}")
            i += 2
        elif c == "{":
            j = fmt.find("}", i)
            if j < 0 or not fmt[i + 1:j].isdigit():
                return UNKNOWN
            n = int(fmt[i + 1:j])
            if n >= len(args):
                return UNKNOWN
            a = args[n]
            if a is UNKNOWN or isinstance(a, Prefix):
                return UNKNOWN
            out.append(_to_str(a))
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _call(fname: str, args: list[Any]) -> Any:
    f = fname.lower()
    if f in ("success", "always"):
        return True
    if f in ("failure", "cancelled"):
        return False
    if f == "hashfiles":
        return UNKNOWN
    if f == "fromjson":
        if len(args) != 1 or not isinstance(args[0], str):
            return UNKNOWN
        try:
            return json.loads(args[0])
        except ValueError:
            return UNKNOWN
    if any(a is UNKNOWN for a in args):
        return UNKNOWN
    if f == "contains" and len(args) == 2:
        search, item = args
        if isinstance(search, list):
            results = [loose_equal(x, item) for x in search]
            if any(r is True for r in results):
                return True
            return UNKNOWN if any(r is None for r in results) else False
        if isinstance(item, Prefix):
            return UNKNOWN
        if isinstance(search, Prefix):
            return True if _to_str(item).lower() in search.prefix.lower() else UNKNOWN
        return _to_str(item).lower() in _to_str(search).lower()
    if f in ("startswith", "endswith") and len(args) == 2:
        s, v = args
        if isinstance(v, Prefix):
            return UNKNOWN
        needle = _to_str(v).lower()
        if isinstance(s, Prefix):
            if f == "endswith":
                return UNKNOWN
            p = s.prefix.lower()
            if p.startswith(needle):
                return True
            return UNKNOWN if needle.startswith(p) else False
        hay = _to_str(s).lower()
        return hay.startswith(needle) if f == "startswith" else hay.endswith(needle)
    if any(isinstance(a, Prefix) for a in args):
        return UNKNOWN
    if f == "format" and args:
        return _format(_to_str(args[0]), args[1:])
    if f == "join" and 1 <= len(args) <= 2:
        sep = _to_str(args[1]) if len(args) == 2 else ","
        if isinstance(args[0], list):
            return sep.join(_to_str(x) for x in args[0])
        return _to_str(args[0])
    if f == "tojson" and len(args) == 1:
        return json.dumps(args[0], indent=2)
    return UNKNOWN


def evaluate(e: Expr, env: EvalEnv) -> Any:
    """Concrete value of ``e`` (possibly ``UNKNOWN`` or a :class:`Prefix`)."""
    op, a = e.op, e.args
    if op == "lit":
        return a[0]
    if op == "ctx":
        return _lookup_ctx(a[0], env)
    if op == "status":
        fn = str(a[0]).lower()
        if fn == "success":
            return env.status_success
        return fn == "always"
    if op == "step":
        sid, fld = a[0], str(a[1])
        if sid not in env.steps:
            return None
        ran = env.steps[sid]
        if fld in ("outcome", "conclusion"):
            return "success" if ran else "skipped"
        if fld.startswith("outputs"):
            return UNKNOWN if ran else None
        return UNKNOWN
    if op == "needs":
        jid, fld = a[0], str(a[1])
        if jid not in env.needs:
            return None
        ran = env.needs[jid]
        if fld == "result":
            return "success" if ran else "skipped"
        if fld.startswith("outputs"):
            return UNKNOWN if ran else None
        return UNKNOWN
    if op == "not":
        return _as_value(_not(truthiness(evaluate(a[0], env))))
    if op == "and":
        left = evaluate(a[0], env)
        tl = truthiness(left)
        if tl is False:
            return left
        right = evaluate(a[1], env)
        if tl is True:
            return right
        return False if truthiness(right) is False else UNKNOWN
    if op == "or":
        left = evaluate(a[0], env)
        tl = truthiness(left)
        if tl is True:
            return left
        right = evaluate(a[1], env)
        if tl is False:
            return right
        return True if truthiness(right) is True else UNKNOWN
    if op in ("eq", "ne"):
        r = loose_equal(evaluate(a[0], env), evaluate(a[1], env))
        return _as_value(r if op == "eq" else _not(r))
    if op == "call":
        return _call(str(a[0]), [evaluate(x, env) for x in a[1:]])
    if op == "opaque":
        return UNKNOWN
    return UNKNOWN


def executes(cond: Expr, env: EvalEnv) -> bool:
    """A condition holds unless it is definitely falsy (unknown counts as true)."""
    return truthiness(evaluate(cond, env)) is not False


# ---------------------------------------------------------------------------
# Program-level evaluation
# ---------------------------------------------------------------------------

_EVENT_FACTS: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "push": MappingProxyType({"github.head_ref": "", "github.base_ref": ""}),
    "pull_request": MappingProxyType({"github.ref": Prefix("refs/pull/")}),
})
_EVENT_NULL_PREFIXES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "push": ("github.event.pull_request",),
    "pull_request": ("github.event.head_commit",),
})
_RUNNER_OS: tuple[tuple[str, str], ...] = (
    ("ubuntu-", "Linux"),
    ("windows-", "Windows"),
    ("macos-", "macOS"),
)


def triggers(on: Any) -> frozenset[str] | None:
    """Event names listed in an ``on:`` block; ``None`` when the block is empty."""
    if not on:
        return None
    if isinstance(on, str):
        return frozenset({on})
    if isinstance(on, Mapping):
        return frozenset(str(k) for k in on)
    if isinstance(on, (list, tuple)):
        return frozenset(str(k) for k in on)
    return None


def triggered(p: Program, event: str) -> bool:
    t = triggers(p.on)
    return t is None or event in t


def _runner_os(runs_on: str) -> Any:
    label = runs_on.strip().lower()
    for prefix, name in _RUNNER_OS:
        if label.startswith(prefix):
            return name
    return UNKNOWN


def _literal_env(pairs: Iterable[tuple[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in sorted(pairs):
        out[f"env.{k}"] = UNKNOWN if "${{" in str(v) else v
    return out


def _base_ctx(p: Program, job: JobInstance, event: str) -> dict[str, Any]:
    ctx: dict[str, Any] = {"github.event_name": event}
    ctx.update(_EVENT_FACTS.get(event, {}))
    ctx["runner.os"] = _runner_os(job.runs_on)
    ctx.update(_literal_env(p.env))
    ctx.update(_literal_env(job.env))
    for k, v in job.matrix:
        ctx[f"matrix.{k}"] = v
    return ctx


def _frozen_step_count(job: JobInstance) -> int:
    if job.steps:
        return len(job.steps)
    raw_steps = job.raw.get("steps") if isinstance(job.raw, Mapping) else None
    if isinstance(raw_steps, (list, tuple)):
        return max(1, len(raw_steps))
    return 1


def billed_minutes_for(seconds: float, ran: bool = True) -> int:
    """GitHub billing of one job: executed jobs round up to a whole minute (at least 1)."""
    if not ran:
        return 0
    return max(1, math.ceil(round(float(seconds), 6) / 60.0))


@dataclass(frozen=True)
class JobRun:
    """Concrete outcome of one instance under one event (all tasks succeed).

    ``step_seconds`` holds the price of each step (0 when it does not execute) and
    ``step_historical`` whether that price came from historical timings.
    """

    ran: bool
    steps: tuple[bool, ...]
    seconds: float
    step_seconds: tuple[float, ...] = ()
    step_historical: tuple[bool, ...] = ()

    @property
    def billed_minutes(self) -> int:
        return billed_minutes_for(self.seconds, self.ran)


def simulate(p: Program, event: str, model: CostModel = MOCK) -> dict[str, JobRun]:
    """Which instances and steps execute for ``event``, and their machine seconds."""
    jobs = {j.id: j for j in p.jobs}
    ids = sorted(jobs)
    on = triggered(p, event)
    ran: dict[str, bool] = {}
    visiting: set[str] = set()

    def job_ran(jid: str) -> bool:
        if jid in ran:
            return ran[jid]
        if jid not in jobs:
            raise ValueError(f"unknown job instance in needs: {jid}")
        if jid in visiting:
            raise ValueError(f"needs cycle through {jid}")
        visiting.add(jid)
        job = jobs[jid]
        deps = {n: job_ran(n) for n in sorted(job.needs)}
        visiting.discard(jid)
        if not on:
            ran[jid] = False
            return False
        env = EvalEnv(
            ctx=_base_ctx(p, job, event),
            null_prefixes=_EVENT_NULL_PREFIXES.get(event, ()),
            status_success=all(deps.values()),
            needs=_needs_view(deps, jobs),
        )
        ran[jid] = executes(job.cond, env)
        return ran[jid]

    result: dict[str, JobRun] = {}
    for jid in ids:
        job = jobs[jid]
        if not job_ran(jid):
            result[jid] = JobRun(False, tuple(False for _ in job.steps), 0.0,
                                 tuple(0.0 for _ in job.steps), tuple(False for _ in job.steps))
            continue
        if job.frozen:
            n = _frozen_step_count(job)
            hist = jid in model.job_table
            if hist:  # priced as a whole; the per-step split is only for reporting
                secs = float(model.job_table[jid])
                per = secs / len(job.steps) if job.steps else 0.0
            else:
                secs = model.overhead(job) + n * model.frozen_step_seconds
                per = model.frozen_step_seconds
            result[jid] = JobRun(True, tuple(True for _ in job.steps), secs,
                                 tuple(per for _ in job.steps), tuple(hist for _ in job.steps))
            continue
        deps = {n: ran[n] for n in sorted(job.needs)}
        ctx = _base_ctx(p, job, event)
        nulls = _EVENT_NULL_PREFIXES.get(event, ())
        needs_view = _needs_view(deps, jobs)
        executed: dict[str, bool] = {}
        flags: list[bool] = []
        prices: list[float] = []
        sources: list[bool] = []
        secs = model.overhead(job)
        for step in job.steps:
            env = EvalEnv(ctx=ctx, null_prefixes=nulls, status_success=True,
                          steps=dict(executed), needs=needs_view)
            ok = executes(step.cond, env)
            flags.append(ok)
            s, hist = model.price_step(job, step) if ok else (0.0, False)
            prices.append(s)
            sources.append(hist)
            secs += s
            if step.binding.step_id:
                executed[step.binding.step_id] = ok
        result[jid] = JobRun(True, tuple(flags), secs, tuple(prices), tuple(sources))
    return result


def _needs_view(deps: Mapping[str, bool], jobs: Mapping[str, JobInstance]) -> dict[str, bool]:
    """Resolve ``needs.<ref>`` by instance id, by original job key, and by group key (an
    instance id without its matrix suffix, e.g. ``caller[os=x]/build``). A ref that denotes
    several instances (a matrix job) ran iff any of them ran (all share one condition)."""
    view: dict[str, bool] = dict(deps)
    groups: dict[str, list[str]] = {}
    for d in sorted(deps):
        keys = {d[: d.rfind("[")] if d.endswith("]") else d}
        if d in jobs:
            keys.add(jobs[d].orig_job)
        for k in sorted(keys):
            groups.setdefault(k, []).append(d)
    for key in sorted(groups):
        if key not in view:
            view[key] = any(deps[d] for d in groups[key])
    return view


def _contexts(contexts: Sequence[tuple[str, float]] | None,
              model: CostModel) -> tuple[tuple[str, float], ...]:
    """Explicit contexts win; else the model's event weights; else ``DEFAULT_CONTEXTS``."""
    if contexts is not None:
        chosen = tuple((str(e), float(w)) for e, w in contexts)
    else:
        chosen = model.contexts if model.contexts is not None else DEFAULT_CONTEXTS
    for event, weight in chosen:
        if weight < 0:
            raise ValueError(f"negative context weight for {event}")
    return tuple(chosen)


def machine_time(p: Program, event: str, model: CostModel = MOCK) -> float:
    """Machine seconds of one run under ``event`` (all tasks succeed)."""
    runs = simulate(p, event, model)
    return float(sum(runs[jid].seconds for jid in sorted(runs)))


def billed_minutes(p: Program, event: str, model: CostModel = MOCK) -> int:
    """Billed minutes of one run under ``event``: Σ over executed jobs of ceil(seconds/60)."""
    runs = simulate(p, event, model)
    return int(sum(runs[jid].billed_minutes for jid in sorted(runs)))


def expected_machine_time(p: Program,
                          contexts: Sequence[tuple[str, float]] | None = None,
                          model: CostModel = MOCK) -> float:
    """Σ_context weight × machine_time. Weights are used as given (not normalized).
    ``contexts=None`` uses the model's event weights (historical) or ``DEFAULT_CONTEXTS``."""
    total = 0.0
    for event, weight in _contexts(contexts, model):
        total += weight * machine_time(p, event, model)
    return total


def expected_billed_minutes(p: Program,
                            contexts: Sequence[tuple[str, float]] | None = None,
                            model: CostModel = MOCK) -> float:
    """Σ_context weight × billed_minutes (same context rules as expected_machine_time)."""
    total = 0.0
    for event, weight in _contexts(contexts, model):
        total += weight * billed_minutes(p, event, model)
    return total


def cost_report(p: Program, contexts: Sequence[tuple[str, float]] | None = None,
                model: CostModel = MOCK) -> dict:
    """JSON-ready breakdown: machine seconds and billed minutes, per event and per instance,
    with the executed steps, their prices, and where each price came from."""
    chosen = _contexts(contexts, model)
    events: dict[str, Any] = {}
    exp_seconds = 0.0
    exp_minutes = 0.0
    for event, weight in chosen:
        runs = simulate(p, event, model)
        secs = float(sum(runs[j].seconds for j in sorted(runs)))
        mins = int(sum(runs[j].billed_minutes for j in sorted(runs)))
        exp_seconds += weight * secs
        exp_minutes += weight * mins
        executed = [h for j in sorted(runs) for ok, h in zip(runs[j].steps, runs[j].step_historical) if ok]
        events[event] = {
            "weight": weight,
            "seconds": secs,
            "billed_minutes": mins,
            "executed_steps": len(executed),
            "executed_steps_historical": sum(1 for h in executed if h),
            "jobs": {j: {"ran": runs[j].ran, "seconds": runs[j].seconds,
                         "billed_minutes": runs[j].billed_minutes,
                         "steps": list(runs[j].steps),
                         "step_seconds": list(runs[j].step_seconds),
                         "step_historical": list(runs[j].step_historical)}
                     for j in sorted(runs)},
        }
    return {
        "model": model.label,
        "contexts": [[e, w] for e, w in chosen],
        "expected_seconds": exp_seconds,
        "expected_billed_minutes": exp_minutes,
        "coverage": dict(model.coverage) if model.coverage is not None else None,
        "events": events,
    }


# ---------------------------------------------------------------------------
# Historical backend (resources/timings/<benchmark-id>/medians.json)
# ---------------------------------------------------------------------------
#
# The medians file is written by scripts/resources/fetch_timings.py (network) and read here
# (offline). Its format, per GitHub job name, is
#   {"samples": n, "durations": [s...], "overheads": [s...], "main_step_counts": [n...],
#    "steps": [{"name": ..., "occurrence": k, "positions": [i...], "durations": [s...],
#               "seconds": median | null, "skipped": n}, ...]}   # in position order
# plus "events": {event: {"runs": [...], "window_count": n}}. See docs/COST_MODEL.md.

TIMINGS_DIR = Path(__file__).resolve().parents[2] / "resources" / "timings"
MEDIANS_FILE = "medians.json"
_EXPR = re.compile(r"\$\{\{.*?\}\}", re.S)


def timings_path(benchmark_id: str, timings_dir: Path | None = None) -> Path:
    return (timings_dir or TIMINGS_DIR) / benchmark_id / MEDIANS_FILE


def load_timings(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _median(xs: Iterable[float]) -> float:
    return float(statistics.median(sorted(float(x) for x in xs)))


def _norm_name(s: Any) -> str:
    return " ".join(str(s).split())


def _name_pattern(name: str) -> re.Pattern[str]:
    parts = _EXPR.split(name)
    return re.compile(".*?".join(re.escape(x) for x in parts), re.S)


_TRUNCATED_MIN = 40   # GitHub cuts long job names to a prefix followed by "..."


def names_match(expected: str, github: str) -> bool:
    """GitHub display names vs names derived from the workflow. Whitespace is normalized, an
    unevaluated ``${{ ... }}`` on either side matches any text, and a long GitHub name ending
    in ``...`` (GitHub's truncation) matches every name it is a prefix of."""
    a, b = _norm_name(expected), _norm_name(github)
    if a == b:
        return True
    if "${{" in a and _name_pattern(a).fullmatch(b):
        return True
    if "${{" in b and _name_pattern(b).fullmatch(a):
        return True
    if b.endswith("...") and len(b) >= _TRUNCATED_MIN and "${{" not in a and a.startswith(b[:-3].rstrip()):
        return True
    return False


def github_step_name(step: Step) -> str:
    """The display name GitHub gives a step: its ``name:``, else ``Run <first line of run>``,
    else ``Run <uses ref>``."""
    b = step.binding
    if b.name:
        return b.name
    if b.kind == "run":
        for line in b.text.splitlines():
            if line.strip():
                return f"Run {line.strip()}"
        return "Run"
    return f"Run {b.text}"


def _flatten_value(v: Any) -> list[str]:
    if isinstance(v, list):
        return [x for item in v for x in _flatten_value(item)]
    if isinstance(v, dict):
        return [x for item in v.values() for x in _flatten_value(item)]
    if isinstance(v, bool):
        return ["true" if v else "false"]
    if v is None:
        return [""]
    return [str(v)]


def _split_suffix(inner: str) -> list[str] | None:
    """Matrix values of a default job-name suffix, JSON arrays/objects flattened."""
    dec = json.JSONDecoder()
    parts: list[str] = []
    i = 0
    while i <= len(inner):
        if inner.startswith(("[", "{"), i):
            try:
                obj, end = dec.raw_decode(inner, i)
            except ValueError:
                obj, end = None, -1
            if end >= 0 and (end == len(inner) or inner.startswith(", ", end)):
                parts.extend(_flatten_value(obj))
                i = end + 2
                continue
        j = inner.find(", ", i)
        if j < 0:
            parts.append(inner[i:])
            break
        parts.append(inner[i:j])
        i = j + 2
    return parts


def github_job_names(job: JobInstance) -> list[str]:
    """Names GitHub may display for ``job``: its check name and, for a matrix suffix
    ``job (v1, v2, ...)``, the rendering GitHub uses for structured values (JSON arrays and
    objects flattened to their values, empty values omitted)."""
    names = [job.check_name]
    name = job.check_name
    if job.matrix and name.endswith(")") and " (" in name:
        depth = 0
        for k in range(len(name) - 1, -1, -1):
            depth += {")": 1, "(": -1}.get(name[k], 0)
            if depth == 0:
                break
        if depth == 0 and k > 0 and name[k - 1] == " ":
            parts = [x for x in (_split_suffix(name[k + 1:-1]) or []) if x != ""]
            alt = f"{name[:k - 1]} ({', '.join(parts)})" if parts else name[:k - 1]
            if alt != name:
                names.append(alt)
    return names


def match_jobs(p: Program, gh_jobs: Mapping[str, Any]) -> dict[str, str]:
    """instance id -> GitHub job name. An exact (normalized) match of one of
    :func:`github_job_names` wins; otherwise the unique GitHub job whose name matches one of
    them with ``${{ }}`` wildcards or truncation,
    provided no other instance matches that GitHub job the same way; else no match."""
    names = sorted(gh_jobs)
    out: dict[str, str] = {}
    fuzzy: dict[str, list[str]] = {}
    for job in sorted(p.jobs, key=lambda j: j.id):
        wants = github_job_names(job)
        norm = {_norm_name(w) for w in wants}
        exact = [n for n in names if _norm_name(n) in norm]
        if len(exact) == 1:
            out[job.id] = exact[0]
            continue
        cands = [] if exact else [n for n in names if any(names_match(w, n) for w in wants)]
        if len(cands) == 1:
            fuzzy.setdefault(cands[0], []).append(job.id)
    for name in sorted(fuzzy):   # a wildcard/truncated match must also be unique per GitHub job
        if len(fuzzy[name]) == 1:
            out[fuzzy[name][0]] = name
    return out


def align_steps(job: JobInstance, gh_job: Mapping[str, Any]) -> list[tuple[int, str] | None]:
    """For each WIR step, (index into gh_job["steps"], "name"|"index") or None.

    Names are aligned greedily and monotonically (the k-th step with a given name matches the
    k-th GitHub step with that name, in order). A step left unmatched then falls back to its
    position when every fetched run had exactly as many workflow steps as the WIR job and the
    GitHub step at that position was always there."""
    entries = list(gh_job.get("steps") or [])
    result: list[tuple[int, str] | None] = [None] * len(job.steps)
    used: set[int] = set()
    ptr = 0
    for i, step in enumerate(job.steps):
        want = github_step_name(step)
        for k in range(ptr, len(entries)):
            if names_match(want, entries[k].get("name", "")):
                result[i] = (k, "name")
                used.add(k)
                ptr = k + 1
                break
    if list(gh_job.get("main_step_counts") or []) == [len(job.steps)]:
        for i in range(len(job.steps)):
            if result[i] is not None:
                continue
            cands = [k for k, e in enumerate(entries)
                     if k not in used and list(e.get("positions") or []) == [i]]
            if len(cands) == 1:
                result[i] = (cands[0], "index")
                used.add(cands[0])
    return result


def event_weights(timings: Mapping[str, Any]) -> tuple[tuple[tuple[str, float], ...], str]:
    """(contexts, source). Weights are the fetched run counts per event (``window_count``)
    normalized over push and pull_request; 0.5/0.5 when no count is available."""
    ev = timings.get("events") or {}
    counts: dict[str, int] = {}
    for e in EVENTS:
        c = (ev.get(e) or {}).get("window_count")
        counts[e] = int(c) if isinstance(c, (int, float)) and c >= 0 else 0
    total = sum(counts.values())
    if total <= 0:
        return DEFAULT_CONTEXTS, "default"
    return tuple((e, counts[e] / total) for e in EVENTS), "run-counts"


def historical_model(p: Program, timings: Mapping[str, Any], base: CostModel = MOCK) -> CostModel:
    """A :class:`CostModel` whose tables come from ``timings`` (a parsed medians file)
    matched against ``p``, the ORIGINAL program. Deterministic."""
    gh_jobs: Mapping[str, Any] = timings.get("jobs") or {}
    matched = match_jobs(p, gh_jobs)
    step_table: dict[tuple[str, str, int], float] = {}
    pooled: dict[tuple[str, int], list[float]] = {}
    overhead_names: dict[str, set[str]] = {}
    job_table: dict[str, float] = {}
    by_name = by_index = 0
    for job in sorted(p.jobs, key=lambda j: j.id):
        name = matched.get(job.id)
        if name is None:
            continue
        g = gh_jobs[name]
        overhead_names.setdefault(job.runs_on, set()).add(name)
        if job.frozen and g.get("durations"):
            job_table[job.id] = _median(g["durations"])
        for step, a in zip(job.steps, align_steps(job, g)):
            if a is None:
                continue
            by_name += a[1] == "name"
            by_index += a[1] == "index"
            durations = list(g["steps"][a[0]].get("durations") or [])
            if durations:
                step_table[(job.id, step.origin[0], step.origin[1])] = _median(durations)
                pooled.setdefault((step.origin[0], step.origin[1]), []).extend(durations)
    origin_table = {k: _median(v) for k, v in sorted(pooled.items()) if v}
    overhead_table: dict[str, float] = {}
    all_overheads: list[float] = []
    for label in sorted(overhead_names):
        xs = [float(x) for n in sorted(overhead_names[label]) for x in gh_jobs[n].get("overheads") or []]
        all_overheads.extend(xs)
        if xs:
            overhead_table[label] = _median(xs)
    default_overhead = _median(all_overheads) if all_overheads else base.job_overhead_seconds
    contexts, weights_source = event_weights(timings)
    model = CostModel(
        job_overhead_seconds=default_overhead,
        template_seconds=base.template_seconds,
        unknown_seconds=base.unknown_seconds,
        frozen_step_seconds=base.frozen_step_seconds,
        label="historical",
        step_table=MappingProxyType(step_table),
        origin_table=MappingProxyType(origin_table),
        overhead_table=MappingProxyType(overhead_table),
        job_table=MappingProxyType(job_table),
        contexts=contexts,
    )
    total = sum(len(j.steps) for j in p.jobs)
    priced = 0
    for job in sorted(p.jobs, key=lambda j: j.id):
        if job.frozen and job.id in job_table:
            priced += len(job.steps)
            continue
        priced += sum(1 for s in job.steps if model.price_step(job, s)[1])
    used_names = set(matched.values())
    coverage = {
        "benchmark": timings.get("benchmark"),
        "jobs_total": len(p.jobs),
        "jobs_matched": len(matched),
        "github_jobs_unmatched": sorted(n for n in gh_jobs if n not in used_names),
        "steps_total": total,
        "steps_matched_by_name": by_name,
        "steps_matched_by_index": by_index,
        "steps_priced_historical": priced,
        "matched_fraction": (priced / total) if total else 0.0,
        "event_weights": {e: w for e, w in contexts},
        "event_weights_source": weights_source,
        "overhead_seconds": dict(sorted(overhead_table.items())),
        "default_overhead_seconds": default_overhead,
    }
    return replace(model, coverage=MappingProxyType(coverage))


def load_historical_model(p: Program, benchmark_id: str, timings_dir: Path | None = None,
                          base: CostModel = MOCK) -> CostModel:
    """Historical model for ``benchmark_id``; the mock model (labelled ``historical`` with an
    empty match) when no medians file exists."""
    path = timings_path(benchmark_id, timings_dir)
    timings = load_timings(path) if path.exists() else {}
    return historical_model(p, timings, base)
