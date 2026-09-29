"""WIR data model shared by every GHAV module.

All containers are immutable or treated as immutable after construction, and every
collection that is iterated must be iterated in sorted order so that all outputs are
deterministic. Do not add fields here without updating docs/ARCHITECTURE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------

# Region kinds. Job-local kinds carry the owning job-instance id in `job`.
CTX = "ctx"          # read-only context value; key = context key, e.g. "github.event_name",
                     # "matrix.node", "env.CI", "platform"; job = instance id (context is per instance)
SRC = "src"          # job-local: checked-out tracked files (including .git); not transferable
DEP = "dep"          # job-local: installed dependencies, setup-* runtimes, PATH; not transferable
OUT = "out"          # job-local: untracked build outputs inside the workspace; transferable
TOOL = "tool"        # job-local: tool caches, temp files, coverage and test reports, HOME config
ST_STEP = "st_step"  # status cell of one step; job = instance id, key = step index as string
ST_JOB = "st_job"    # status cell of one job instance; job = instance id
OUTPUT = "output"    # job output value; job = producing instance id, key = output name
RUN = "run"          # run-shared named artifact; key = artifact name
EXT = "ext"          # external state; key = external object name (e.g. "codecov", "check:<name>")

JOB_LOCAL_KINDS = frozenset({SRC, DEP, OUT, TOOL, ST_STEP})
TRANSFERABLE_KINDS = frozenset({OUT})


@dataclass(frozen=True, order=True)
class Region:
    kind: str
    job: str | None = None
    key: str | None = None

    def __str__(self) -> str:  # stable, human-readable, used in reports
        parts = [self.kind]
        if self.job is not None:
            parts.append(self.job)
        if self.key is not None:
            parts.append(self.key)
        return ":".join(parts)


@dataclass(frozen=True)
class Effect:
    """Read and write sets over regions. `top` means "may read and write anything"."""

    reads: frozenset[Region] = frozenset()
    writes: frozenset[Region] = frozenset()
    top: bool = False

    def union(self, other: "Effect") -> "Effect":
        return Effect(self.reads | other.reads, self.writes | other.writes, self.top or other.top)

    def independent(self, other: "Effect") -> bool:
        if self.top or other.top:
            return False
        if self.writes & (other.reads | other.writes):
            return False
        if other.writes & self.reads:
            return False
        return True

    def masked(self, job: str) -> "Effect":
        """Drop atoms on the job-local regions of `job` (they die with the job)."""
        keep = lambda r: not (r.kind in JOB_LOCAL_KINDS and r.job == job)
        return Effect(frozenset(filter(keep, self.reads)), frozenset(filter(keep, self.writes)), self.top)


# ---------------------------------------------------------------------------
# Contracts (job-independent task effects; instantiated per job instance)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Contract:
    """A task's latent effect, written against a placeholder job.

    Region atoms use kinds from above with job=None; `instantiate` fills in the
    job-instance id for job-local kinds and for CTX. `out_paths` lists workspace-relative
    paths that make up the task's OUT writes (needed to emit artifact transfers);
    empty means OUT writes are not transferable. `template` names the contract-guide
    template that produced this contract (for reporting and auditing).
    """

    reads: frozenset[Region] = frozenset()
    writes: frozenset[Region] = frozenset()
    top: bool = False
    out_paths: tuple[str, ...] = ()
    template: str = "unknown"

    def instantiate(self, job: str) -> Effect:
        def inst(r: Region) -> Region:
            if r.kind in JOB_LOCAL_KINDS or r.kind in (CTX,):
                return Region(r.kind, job, r.key)
            return r
        return Effect(frozenset(map(inst, self.reads)), frozenset(map(inst, self.writes)), self.top)


TOP_CONTRACT = Contract(top=True, template="top")


# ---------------------------------------------------------------------------
# Expressions (GitHub Actions `if:` and `${{ }}` subset); see ghav/expr.py
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Expr:
    """Expression AST node.

    op is one of:
      "lit"      args=(value,)                     string/bool/number literal
      "ctx"      args=(key,)                       context value, e.g. "github.event_name", "matrix.os"
      "status"   args=(fn,)                        fn in success|failure|always|cancelled (job-scoped)
      "step"     args=(step_id, field)             steps.<id>.outcome|conclusion|outputs.<name>
      "needs"    args=(job_id, field)              needs.<job>.result|outputs.<name>
      "not"      args=(e,)
      "and"/"or" args=(e1, e2)
      "eq"/"ne"  args=(e1, e2)
      "call"     args=(fname, e1, ..., en)         contains|startsWith|endsWith|format|fromJSON|toJSON|hashFiles
      "opaque"   args=(text, refs...)              unparsed text; refs are ctx keys it mentions (sorted)
    """

    op: str
    args: tuple = ()


TRUE = Expr("lit", (True,))


# ---------------------------------------------------------------------------
# Program structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Binding:
    """The original executable content of a step, emitted verbatim."""

    kind: str                                  # "run" or "uses"
    text: str                                  # run script, or action reference (e.g. "actions/checkout@v4")
    with_: tuple[tuple[str, str], ...] = ()    # sorted (key, value) pairs; values keep ${{ }} text
    env: tuple[tuple[str, str], ...] = ()      # sorted step-level env
    working_directory: str | None = None
    shell: str | None = None
    name: str | None = None
    step_id: str | None = None                 # the YAML `id:` if any

    def key(self) -> tuple:
        """Identity of the computation, used to name its uninterpreted functions."""
        return (self.kind, self.text, self.with_, self.env, self.working_directory, self.shell)


@dataclass(frozen=True)
class Step:
    binding: Binding
    cond: Expr                                 # EXPLICIT condition, default conditions already folded in
    contract: Contract
    origin: tuple[str, int]                    # (original job id in source YAML, original step index)
    raw: Mapping | None = None                 # original YAML mapping of the step (for emission)
    origin_instance: str | None = None         # original job-INSTANCE id this occurrence came from (finer check groups)


@dataclass(frozen=True)
class JobInstance:
    id: str                                    # unique, stable, e.g. "test[node=18,os=ubuntu-latest]"
    orig_job: str                              # job key in the source YAML (after reusable expansion: "caller/callee")
    check_name: str                            # name GitHub shows for this instance
    matrix: tuple[tuple[str, str], ...]        # sorted concrete matrix assignment
    runs_on: str                               # resolved platform label (matrix substituted)
    cond: Expr                                 # explicit job condition, default needs-success folded in
    needs: frozenset[str]                      # instance ids this instance waits for
    steps: tuple[Step, ...]
    env: tuple[tuple[str, str], ...] = ()      # job-level env (sorted)
    permissions: tuple[tuple[str, str], ...] | str | None = None
    outputs: tuple[tuple[str, Expr], ...] = () # job outputs (sorted by name)
    fail_fast_group: str | None = None         # instances sharing a matrix with fail-fast enabled
    frozen: bool = False                       # uses constructs outside the core; kept verbatim
    frozen_reasons: tuple[str, ...] = ()
    raw: Mapping | None = None                 # original YAML job mapping (for frozen jobs and emission)


@dataclass(frozen=True)
class Program:
    name: str
    on: Mapping                                # original `on:` block, emitted verbatim
    jobs: tuple[JobInstance, ...]              # sorted by id
    env: tuple[tuple[str, str], ...] = ()      # workflow-level env
    permissions: object = None
    concurrency: object = None                 # workflow-level concurrency, emitted verbatim
    defaults: object = None
    source: str = ""                           # path of the source workflow, for reporting

    def job(self, jid: str) -> JobInstance:
        for j in self.jobs:
            if j.id == jid:
                return j
        raise KeyError(jid)

    def with_jobs(self, jobs) -> "Program":
        return replace(self, jobs=tuple(sorted(jobs, key=lambda j: j.id)))


# Execution contexts used by the correctness quantifier and the cost model.
EVENTS = ("push", "pull_request")
